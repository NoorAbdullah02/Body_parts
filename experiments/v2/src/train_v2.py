"""
train_v2 -- one training run (one backbone, one fold) designed for CLEAN out-of-fold (OOF)
predictions.

Differences from src/train.py / engine.py (and why)
---------------------------------------------------
* NO checkpoint selection.  The old engine kept the epoch with the best validation
  macro_f1 of the SAME fold that was later reported / used to tune thresholds, which makes
  every reported number optimistic.  Here the schedule is fixed (cosine, N epochs) and the
  final EMA weights are used.  Per-epoch validation numbers are printed for monitoring only.
* EMA of the weights (smoother, usually better than the last raw epoch, needs no selection).
* Two heads on one backbone: a multilabel sigmoid head (22 logits) and a single-label
  softmax head (22 logits).  ~92 % of UNIFESP images carry exactly one label, and the
  2nd-place Kaggle solution framed it as single-label multiclass.  Both heads are saved;
  evalkit/run_ensemble choose / blend them on OOF only.
* Predictions for OOF fold, held-out set (optional) and the Kaggle test set are written to
  ``preds.npz`` so the ensemble stage is pure numpy (no GPU).
* fp16 autocast with logits-space losses only (the V3 crash was BCE on probabilities).

Usage (Kaggle, GPU):
    python -m src.train_v2 --backbone swin_tiny --img_size 224 --fold 0 \
        --folds_csv /kaggle/working/splits/cv_folds.csv --fold_col fold \
        --holdout_csv /kaggle/working/splits/holdout_uids.csv --run_name swin_tiny_s224_f0

Smoke test (CPU, a few images, no pretrained weights, no real training):
    python -m src.train_v2 --backbone resnet18 --img_size 64 --fold 0 --epochs 1 --smoke_test ...
"""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import config as C
from . import evalkit as E
from .dataset import UnifespDataset, build_uid_to_path

N = C.N_CLASSES
LABEL_COLS = [f"label_{k}_{C.LABEL_NAMES[k].replace(' ', '_')}" for k in range(N)]

# short name -> timm name (anything else is passed to timm unchanged)
TIMM_NAMES = {
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnetv2_s": "tf_efficientnetv2_s",
    "densenet121": "densenet121",
    "convnext_tiny": "convnext_tiny",
    "convnext_small": "convnext_small",
    "resnet50": "resnet50",
    "resnet152": "resnet152",
    "efficientnet_b3": "efficientnet_b3",
}


# ----------------------------------------------------------------------------- model
class TwoHeadNet(nn.Module):
    """timm backbone (pooled features) + one linear layer producing 2*N logits:
    [:N] -> multilabel (sigmoid), [N:] -> single-label (softmax)."""

    def __init__(self, backbone, pretrained=True, drop=0.2):
        super().__init__()
        import timm
        name = TIMM_NAMES.get(backbone, backbone)
        self.body = timm.create_model(name, pretrained=pretrained, num_classes=0)
        self.drop = nn.Dropout(drop)
        with torch.no_grad():          # measure the real feature size (timm's num_features is wrong for e.g. VGG: 512 vs 4096)
            was_training = self.body.training
            feat = self.body.eval()(torch.zeros(1, 3, 224, 224)).shape[1]
            self.body.train(was_training)
        self.head = nn.Linear(feat, 2 * N)

    def forward(self, x):
        z = self.head(self.drop(self.body(x)))
        return z[:, :N], z[:, N:]


def build_model(name, pretrained=True):
    """timm backbone (TwoHeadNet) or one of the DenseNet+Swin fusion variants (fusion.py, input 224)."""
    if name.startswith("hybrid_"):
        from .fusion import FusionNet
        return FusionNet(name, pretrained=pretrained)
    return TwoHeadNet(name, pretrained=pretrained)


def compute_pos_weight(y, cap=20.0):
    pos = np.clip(y.sum(0), 1, None)
    return torch.tensor(np.clip((len(y) - pos) / pos, 1.0, cap), dtype=torch.float32)


class TwoHeadLoss(nn.Module):
    """BCE-with-logits (pos_weight, capped) on the sigmoid head
       + lam * soft-target cross-entropy on the softmax head (target = y / y.sum())."""

    def __init__(self, pos_weight=None, lam=0.5, smoothing=0.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.lam, self.smoothing = lam, smoothing

    def forward(self, outs, y):
        sig, soft = outs
        loss = self.bce(sig.float(), y)
        if self.lam > 0:
            t = y / y.sum(1, keepdim=True).clamp(min=1.0)
            if self.smoothing > 0:
                t = t * (1 - self.smoothing) + self.smoothing / N
            loss = loss + self.lam * (-(t * F.log_softmax(soft.float(), dim=1)).sum(1)).mean()
        return loss


class EMA:
    """Exponential moving average of parameters AND buffers (BN statistics)."""

    def __init__(self, model, decay=0.998):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.steps = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


# ----------------------------------------------------------------------------- data
def get_transforms(img_size, train, aug="light"):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if not train:
        return A.Compose([A.Resize(img_size, img_size), A.Normalize(mean=mean, std=std), ToTensorV2()])
    if aug == "heavy":      # in the spirit of the 2nd-place Kaggle writeup; defaults only (API-stable)
        ops = [A.Resize(img_size, img_size), A.HorizontalFlip(p=0.5),
               A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=15, p=0.6),
               A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
               A.GaussNoise(p=0.2), A.GaussianBlur(p=0.1)]
    else:                   # "light": the project's earlier default
        ops = [A.Resize(img_size, img_size), A.HorizontalFlip(p=0.5),
               A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
               A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)]
    return A.Compose(ops + [A.Normalize(mean=mean, std=std), ToTensorV2()])


def make_frames(args):
    index_df = pd.read_csv(C.INDEX_CSV)
    folds = pd.read_csv(args.folds_csv)[["uid", args.fold_col]]
    tr_all = index_df[index_df["split"] == "train"].merge(folds, on="uid", how="inner")
    train_df = tr_all[tr_all[args.fold_col] != args.fold].reset_index(drop=True)
    oof_df = tr_all[tr_all[args.fold_col] == args.fold].reset_index(drop=True)
    hold_df = None
    if args.holdout_csv:
        hold_uids = set(pd.read_csv(args.holdout_csv)["uid"])
        hold_df = index_df[index_df["uid"].isin(hold_uids)].reset_index(drop=True)
    test_df = index_df[index_df["split"] == "test"].reset_index(drop=True)
    if args.smoke_test:
        train_df, oof_df, test_df = train_df.iloc[:8], oof_df.iloc[:4], test_df.iloc[:4]
        hold_df = None if hold_df is None else hold_df.iloc[:4]
    if getattr(args, "pseudo_csv", None):
        ps = pd.read_csv(args.pseudo_csv)
        miss = [c for c in LABEL_COLS if c not in ps.columns]
        assert not miss, f"pseudo csv lacks label columns {miss[:3]}..."
        if args.smoke_test:
            ps = ps.iloc[:4]
        train_df = pd.concat([train_df, ps.reindex(columns=train_df.columns)], ignore_index=True)
        print(f"[pseudo] +{len(ps)} pseudo-labelled images added to the training part")
    return train_df, oof_df, hold_df, test_df


# ----------------------------------------------------------------------------- predict
@torch.no_grad()
def predict(model, loader, device, tta=False):
    model.eval()
    sig_all, soft_all, uids = [], [], []
    for images, second in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            s, f = model(images)
            ps, pf = torch.sigmoid(s.float()), torch.softmax(f.float(), dim=1)
            if tta:
                s2, f2 = model(torch.flip(images, dims=[3]))
                ps = (ps + torch.sigmoid(s2.float())) / 2
                pf = (pf + torch.softmax(f2.float(), dim=1)) / 2
        sig_all.append(ps.cpu().numpy()); soft_all.append(pf.cpu().numpy())
        uids.extend(list(second) if not torch.is_tensor(second) else [None] * len(images))
    return np.concatenate(sig_all).astype(np.float32), np.concatenate(soft_all).astype(np.float32), uids


def run_predictions(model, df, has_labels, args, device, uid_to_path, img_size):
    ds = UnifespDataset(df, C.DATA_ROOT, transform=get_transforms(img_size, False), has_labels=False,
                        uid_to_path=uid_to_path)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    out = {}
    for tag, tta in (("", False), ("_tta", True)):
        s, f, uids = predict(model, dl, device, tta=tta)
        out["sig" + tag], out["soft" + tag] = s, f
    out["uid"] = np.array(uids if uids and uids[0] is not None else df["uid"].tolist())
    if has_labels:
        out["y"] = df[LABEL_COLS].to_numpy(dtype=np.int64)
    return out


# ----------------------------------------------------------------------------- main
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--folds_csv", required=True)
    p.add_argument("--fold_col", default="fold")
    p.add_argument("--holdout_csv", default=None)
    p.add_argument("--pseudo_csv", default=None,
                    help="optional CSV (uid + label_* columns) of pseudo-labelled UNLABELLED images (e.g. Kaggle test images); "
                         "added to the TRAINING part of every fold only, never to OOF / held-out")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.998)
    p.add_argument("--aug", default="light", choices=["light", "heavy"])
    p.add_argument("--lam_ce", type=float, default=0.5, help="weight of the softmax single-label head loss (0 = off)")
    p.add_argument("--smoothing", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--run_name", default=None)
    p.add_argument("--out_dir", default=C.OUTPUT_DIR)
    p.add_argument("--smoke_test", action="store_true", help="tiny subset, no pretrained weights (CPU code-path check)")
    p.add_argument("--no_pretrained", action="store_true", help="random init (local sanity runs only)")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or f"{args.backbone}_s{args.img_size}_f{args.fold}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    train_df, oof_df, hold_df, test_df = make_frames(args)
    print(f"[{run_name}] train={len(train_df)} oof={len(oof_df)} "
          f"holdout={0 if hold_df is None else len(hold_df)} test={len(test_df)} device={device}")
    uid_to_path = build_uid_to_path(C.DATA_ROOT)

    tr_ds = UnifespDataset(train_df, C.DATA_ROOT, transform=get_transforms(args.img_size, True, args.aug),
                           uid_to_path=uid_to_path)
    va_ds = UnifespDataset(oof_df, C.DATA_ROOT, transform=get_transforms(args.img_size, False),
                           uid_to_path=uid_to_path)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                       pin_memory=device.type == "cuda", drop_last=len(train_df) > args.batch_size)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.backbone, pretrained=not (args.smoke_test or args.no_pretrained)).to(device)
    ema = EMA(model, args.ema)
    loss_fn = TwoHeadLoss(compute_pos_weight(train_df[LABEL_COLS].to_numpy(np.float32)).to(device),
                          lam=args.lam_ce, smoothing=args.smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, len(tr_dl))
    total, warm = args.epochs * steps_per_epoch, int(args.warmup_epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    t0, history = time.time(), []
    for epoch in range(args.epochs):
        model.train(); run_loss = 0.0
        for images, y in tr_dl:
            images, y = images.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                outs = model(images)
            loss = loss_fn(outs, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step(); ema.update(model)
            run_loss += float(loss.detach())
        # monitoring only -- NEVER used to pick a checkpoint
        s, _, _ = predict(ema.ema, va_dl, device)
        yv = oof_df[LABEL_COLS].to_numpy(np.int64)
        mon = E.sample_mean_f1(yv, E.predict_labels(s, 0.5, min_one=True))
        history.append({"epoch": epoch, "train_loss": run_loss / steps_per_epoch, "oof_sample_mean_f1_monitor": mon})
        print(f"epoch {epoch}: loss={run_loss / steps_per_epoch:.4f} monitor_oof_f1={mon:.4f} ({time.time() - t0:.0f}s)")

    # final EMA weights -> predictions (no selection)
    preds = {}
    for tag, df, has_y in (("oof", oof_df, True), ("hold", hold_df, True), ("test", test_df, False)):
        if df is None or len(df) == 0:
            continue
        r = run_predictions(ema.ema, df, has_y, args, device, uid_to_path, args.img_size)
        for k, v in r.items():
            preds[f"{tag}_{k}"] = v
    np.savez_compressed(os.path.join(run_dir, "preds.npz"), **preds)
    torch.save(ema.ema.state_dict(), os.path.join(run_dir, "final_ema.pt"))   # state_dict only (weights_only-safe)
    meta = {"args": vars(args), "history": history, "seconds": time.time() - t0, "torch": torch.__version__}
    if hasattr(ema.ema, "gate_weights"):      # fusion variants: how much weight does the CNN branch get?
        ws = []
        with torch.no_grad():
            for images, _ in va_dl:
                ws.append(ema.ema.gate_weights(images.to(device))[:, 0].float().cpu())
        w = torch.cat(ws)
        meta["gate_cnn_weight_oof_mean"], meta["gate_cnn_weight_oof_std"] = float(w.mean()), float(w.std())
        print(f"[{run_name}] mean CNN-branch gate weight on OOF fold: {float(w.mean()):.4f}")
    json.dump(meta, open(os.path.join(run_dir, "meta.json"), "w"), indent=1, default=str)
    print(f"[{run_name}] done in {time.time() - t0:.0f}s -> {run_dir}")


if __name__ == "__main__":
    main()
