"""
fm_probe -- frozen foundation-model features + linear probe for UNIFESP body-part classification (gap G7).

    python -m src.fm_probe --models dinov2_s,dinov2_b,biomedclip,swin_in1k --out_dir fm_results

Question answered: how far does a FROZEN pretrained encoder + a linear layer get, compared with the fine-tuned networks of the main study?
Protocol = the same as everywhere else: patient-level 5-fold CV on the CV pool, hyper-parameter (weight decay) chosen on OOF only,
held-out set (same uids as the main study) used once, patient-cluster bootstrap CI. Decoding: 0.5 threshold + min-one rule
(the "reference" decoding of the ablation report, so numbers are comparable with its hold_sample_f1 column).

Also reports a label-efficiency curve (OOF F1 when only 10/25/50/100 % of the TRAINING PATIENTS of each fold are used).
Writes: fm_summary.csv, fm_label_efficiency.csv, fm_<model>_probs.npz (OOF + held-out probabilities), fm_report.md.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as C
from . import evalkit as E
from .dataset import UnifespDataset, build_uid_to_path

LABEL_COLS = [f"label_{k}_{C.LABEL_NAMES[k].replace(' ', '_')}" for k in range(C.N_CLASSES)]
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

# name -> (loader kind, identifier, input size, (mean, std))
MODELS = {
    "dinov2_s": ("timm", "vit_small_patch14_dinov2.lvd142m", 224, IMAGENET),
    "dinov2_b": ("timm", "vit_base_patch14_dinov2.lvd142m", 224, IMAGENET),
    "swin_in1k": ("timm", "swin_tiny_patch4_window7_224.ms_in1k", 224, IMAGENET),
    "convnext_in1k": ("timm", "convnext_tiny.fb_in1k", 224, IMAGENET),
    "biomedclip": ("open_clip", "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base-patch16-224", 224, CLIP),
    "resnet18_random": ("timm_random", "resnet18", 224, IMAGENET),         # sanity floor: random weights, same probe
}


def load_encoder(name):
    kind, ident, size, norm = MODELS[name]
    if kind == "timm":
        import timm
        kw = {"img_size": size} if "dinov2" in ident else {}      # timm DINOv2 defaults to 518 px; 224 px matches the other encoders
        m = timm.create_model(ident, pretrained=True, num_classes=0, **kw)
        fn = lambda x: m(x)
    elif kind == "timm_random":
        import timm
        m = timm.create_model(ident, pretrained=False, num_classes=0)
        fn = lambda x: m(x)
    else:
        import open_clip
        m, _ = open_clip.create_model_from_pretrained(ident)
        fn = lambda x: m.encode_image(x)
    m.eval()
    return m, fn, size, norm


@torch.no_grad()
def extract(name, df, uid_to_path, device, batch_size=64, num_workers=2):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    m, fn, size, (mean, std) = load_encoder(name)
    m.to(device)
    tf = A.Compose([A.Resize(size, size), A.Normalize(mean=mean, std=std), ToTensorV2()])
    ds = UnifespDataset(df, C.DATA_ROOT, transform=tf, has_labels=False, uid_to_path=uid_to_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    out = []
    for images, _ in dl:
        images = images.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            f = fn(images)
        out.append(f.float().cpu().numpy())
    del m
    return np.concatenate(out)


def fit_linear(X, Y, wd, epochs=300, lr=1e-2, seed=0):
    """Full-batch logistic regression (22 sigmoid outputs, BCE, L2 = wd); X is standardized beforehand."""
    g = torch.Generator().manual_seed(seed)
    X, Y = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32)
    lin = nn.Linear(X.shape[1], Y.shape[1])
    nn.init.normal_(lin.weight, std=0.01, generator=g); nn.init.zeros_(lin.bias)
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(lin(X), Y) + wd * (lin.weight ** 2).sum()
        loss.backward(); opt.step()
    return lin


def predict_prob(lin, X):
    with torch.no_grad():
        return torch.sigmoid(lin(torch.as_tensor(X, dtype=torch.float32))).numpy()


def f1_at_half(y, p):
    return E.sample_mean_f1(y, E.predict_labels(p, 0.5, True))


def run_model(name, feats_pool, y_pool, folds, groups_pool, feats_hold, y_hold, groups_hold, wds, fractions, n_boot, seed=0):
    rng = np.random.RandomState(seed)
    mu, sd = feats_pool.mean(0), feats_pool.std(0) + 1e-6          # standardisation from the CV pool only
    Xp, Xh = (feats_pool - mu) / sd, (feats_hold - mu) / sd
    best = None
    table = []
    for wd in wds:
        oof = np.zeros_like(y_pool, dtype=np.float32)
        for k in sorted(set(folds)):
            tr, va = folds != k, folds == k
            oof[va] = predict_prob(fit_linear(Xp[tr], y_pool[tr], wd), Xp[va])
        f = f1_at_half(y_pool, oof)
        table.append({"model": name, "wd": wd, "oof_sample_f1": f})
        if best is None or f > best[0]:
            best = (f, wd, oof)
    f_oof, wd, oof = best                                           # wd chosen on OOF only
    hold = np.mean([predict_prob(fit_linear(Xp[folds != k], y_pool[folds != k], wd), Xh) for k in sorted(set(folds))], axis=0)
    pred = E.predict_labels(hold, 0.5, True)
    bs = E.cluster_bootstrap(E.sample_mean_f1, y_hold, pred, groups_hold, n_boot=n_boot) if n_boot else {}
    lo, hi = bs.get('ci_low', np.nan), bs.get('ci_high', np.nan)
    res = {"model": name, "dim": int(feats_pool.shape[1]), "chosen_wd": wd, "oof_sample_f1": f_oof,
           "hold_sample_f1": E.sample_mean_f1(y_hold, pred), "hold_ci_low": lo, "hold_ci_high": hi, "hold_macro_f1": E.macro_f1(y_hold, pred)}
    # label efficiency: use a random subset of the TRAINING PATIENTS of each fold
    eff = []
    pats = np.asarray(groups_pool)
    for frac in fractions:
        o = np.zeros_like(y_pool, dtype=np.float32)
        for k in sorted(set(folds)):
            tr_idx = np.where(folds != k)[0]
            up = np.unique(pats[tr_idx]); rng.shuffle(up)
            keep = set(up[:max(1, int(round(frac * len(up))))])
            sub = np.array([i for i in tr_idx if pats[i] in keep])
            o[folds == k] = predict_prob(fit_linear(Xp[sub], y_pool[sub], wd), Xp[folds == k])
        eff.append({"model": name, "train_patient_fraction": frac, "oof_sample_f1": f1_at_half(y_pool, o)})
    return res, table, eff, oof, hold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="dinov2_s,dinov2_b,swin_in1k,biomedclip")
    p.add_argument("--index_csv", default=C.INDEX_CSV)
    p.add_argument("--cv_folds_csv", required=True)
    p.add_argument("--holdout_csv", required=True)
    p.add_argument("--out_dir", default="fm_results")
    p.add_argument("--fractions", default="0.1,0.25,0.5,1.0")
    p.add_argument("--wds", default="1e-5,1e-4,1e-3,1e-2")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--smoke_test", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx = pd.read_csv(a.index_csv)
    cv = pd.read_csv(a.cv_folds_csv)[["uid", "fold"]]
    pool = idx[idx["split"] == "train"].merge(cv, on="uid", how="inner").reset_index(drop=True)
    hold = idx[idx["uid"].isin(set(pd.read_csv(a.holdout_csv)["uid"]))].reset_index(drop=True)
    if a.smoke_test:
        pool, hold = pool.iloc[:60], hold.iloc[:20]
    u2p = build_uid_to_path(C.DATA_ROOT)
    y_pool, y_hold = pool[LABEL_COLS].to_numpy(np.int64), hold[LABEL_COLS].to_numpy(np.int64)
    folds, g_pool, g_hold = pool["fold"].to_numpy(), pool["patient_id"].astype(str).to_numpy(), hold["patient_id"].astype(str).to_numpy()
    print(f"pool {len(pool)} images / {len(set(g_pool))} patients | held-out {len(hold)} / {len(set(g_hold))} | device {device}")
    wds = [float(x) for x in a.wds.split(",")]
    fractions = [float(x) for x in a.fractions.split(",")]

    summ, effs, tabs = [], [], []
    for name in a.models.split(","):
        try:
            fp = extract(name, pool, u2p, device, a.batch_size, a.num_workers)
            fh = extract(name, hold, u2p, device, a.batch_size, a.num_workers)
        except Exception as e:            # one unavailable model (no internet / missing package) must not kill the run
            print(f"[SKIP] {name}: {type(e).__name__}: {e}")
            summ.append({"model": name, "error": f"{type(e).__name__}: {e}"[:200]})
            continue
        res, tab, eff, oof, hp = run_model(name, fp, y_pool, folds, g_pool, fh, y_hold, g_hold, wds, fractions,
                                          n_boot=0 if a.smoke_test else a.n_boot)
        np.savez_compressed(os.path.join(a.out_dir, f"fm_{name}_probs.npz"), oof=oof, hold=hp, pool_uid=pool["uid"].to_numpy(), hold_uid=hold["uid"].to_numpy())
        summ.append(res); effs += eff; tabs += tab
        print(f"[ok] {name}: OOF {res['oof_sample_f1']:.4f} | held-out {res['hold_sample_f1']:.4f} [{res['hold_ci_low']:.4f}, {res['hold_ci_high']:.4f}] (wd {res['chosen_wd']})")
    pd.DataFrame(summ).to_csv(os.path.join(a.out_dir, "fm_summary.csv"), index=False)
    pd.DataFrame(effs).to_csv(os.path.join(a.out_dir, "fm_label_efficiency.csv"), index=False)
    pd.DataFrame(tabs).to_csv(os.path.join(a.out_dir, "fm_wd_search.csv"), index=False)
    md = ["# Frozen foundation-model linear probe (UNIFESP, patient-level protocol)", "",
          "Reference (fine-tuned Swin-T, same held-out set, 0.5 + min-one): hold_sample_f1 0.9089 [0.8576, 0.9509] (ablation report).", "",
          pd.DataFrame(summ).round(4).to_string(index=False), "",
          "## Label efficiency (OOF sample-F1 vs fraction of training patients)", "",
          pd.DataFrame(effs).round(4).to_string(index=False)]
    open(os.path.join(a.out_dir, "fm_report.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
