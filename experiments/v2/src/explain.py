"""
explain -- explainability for the body-part models: Grad-CAM + a quantitative check that the maps mean something + error analysis.

What it produces for ONE trained model (a fold model's final_ema.pt) on the held-out images:
  1. gallery_page*.png   per class: a correctly predicted image and a missed image, with Grad-CAM for the true class and for the wrongly predicted class
  2. explain_metrics.json
       * faithfulness (deletion test): delete the most-salient pixels first and watch the class probability fall; compared with deleting
         in random order.  ``deletion_gain`` = (random score - CAM score) > 0 means the map points at pixels the model really uses.
       * sanity check (Adebayo et al. 2018 style): CAM of the trained model vs CAM of a randomly initialised model of the same architecture
         (Spearman rank correlation); a map that does not change when the weights are randomised is an edge detector, not an explanation.
  3. top_confusions.csv  most frequent (true class missed -> class predicted instead) pairs  -> error analysis table
Works for CNNs (NCHW feature maps) and Swin (NHWC feature maps) without a per-architecture table: the target layer is found by a probe forward
pass (the last module that outputs a spatial 4-D tensor).

LIMITATIONS to state in the paper: Grad-CAM maps are coarse (7x7 to 8x8 feature grid), there are NO ground-truth localisation labels in UNIFESP (so no
pointing-game / IoU), and no radiologist has rated the maps unless you add that review.

    python -m src.explain --weights runs/convnext_tiny_s256_f0/final_ema.pt --backbone convnext_tiny --img_size 256 \
        --index_csv index.csv --holdout_csv splits/holdout_uids.csv --out_dir explain_convnext --title convnext_tiny
"""
import argparse
import json
import os

import numpy as np

from . import evalkit as E

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ------------------------------------------------------------------------------------------------ Grad-CAM
def is_nhwc(s):
    """Shape (B, a, b, c): channels-last if the two equal dims are 1,2 (square spatial first), channels-first if they are 2,3; else the
    larger of dim 1 / dim 3 is the channel dim."""
    if s[2] == s[3] and s[1] != s[2]:
        return False
    if s[1] == s[2] and s[3] != s[2]:
        return True
    return s[-1] > s[1]


def auto_target(model, img_size, device="cpu"):
    """(module_name, channels_last) of the last module whose output is a spatial 4-D feature map (>=2x2)."""
    import torch
    found, hooks = [], []
    for name, m in model.named_modules():
        if name:
            def mk(n):
                def f(_m, _i, o):
                    if torch.is_tensor(o) and o.ndim == 4:
                        found.append((n, tuple(o.shape)))
                return f
            hooks.append(m.register_forward_hook(mk(name)))
    with torch.no_grad():
        model.eval()(torch.zeros(1, 3, img_size, img_size, device=device))
    for h in hooks:
        h.remove()
    best = None
    for n, s in found:
        cl = is_nhwc(s)
        hw = (s[1], s[2]) if cl else (s[2], s[3])
        if min(hw) >= 2:
            best = (n, cl)
    assert best is not None, "no spatial feature map found"
    return best


class GradCAM:
    def __init__(self, model, img_size=224, device="cpu"):
        self.model = model.eval()
        self.name, self.channels_last = auto_target(model, img_size, device)
        self._act = None
        self._h = dict(model.named_modules())[self.name].register_forward_hook(lambda m, i, o: setattr(self, "_act", o))

    def close(self):
        self._h.remove()

    def __call__(self, x, class_idx, head="sig"):
        """x: (B,3,S,S) normalised tensor; class_idx: (B,) ints. Returns cam (B,h,w) in [0,1], sigmoid probs, softmax probs."""
        import torch
        with torch.enable_grad():
            sig, soft = self.model(x)
            out = sig if head == "sig" else soft
            idx = torch.as_tensor(class_idx, device=out.device).long()
            score = out[torch.arange(len(idx)), idx].sum()
            act = self._act
            grad = torch.autograd.grad(score, act)[0]
        if self.channels_last:
            act, grad = act.permute(0, 3, 1, 2), grad.permute(0, 3, 1, 2)
        cam = torch.relu((grad.mean(dim=(2, 3), keepdim=True) * act).sum(1))
        flat = cam.flatten(1)
        mn, mx = flat.min(1)[0][:, None, None], flat.max(1)[0][:, None, None]
        cam = (cam - mn) / (mx - mn + 1e-8)
        return cam.detach().cpu().numpy(), torch.sigmoid(sig).detach().cpu().numpy(), torch.softmax(soft, 1).detach().cpu().numpy()


def upsample(cam, size):
    import torch
    t = torch.as_tensor(cam, dtype=torch.float32)[None, None]
    return torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


# ------------------------------------------------------------------------------------------------ faithfulness / sanity
def trapezoid(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    return float(((y[:-1] + y[1:]) / 2 * np.diff(x)).sum())


def deletion_curve(model, x, order, class_idx, steps):
    """Replace the first k pixels of ``order`` (flattened pixel ranking) with the per-channel image mean; return P(class) at each step."""
    import torch
    S = x.shape[-1]
    fill = x.mean(dim=(2, 3), keepdim=True)
    probs = []
    for f in steps:
        k = int(f * S * S)
        m = torch.zeros(S * S, device=x.device)
        m[torch.as_tensor(order[:k], device=x.device)] = 1.0
        m = m.view(1, 1, S, S)
        with torch.no_grad():
            probs.append(float(torch.sigmoid(model(x * (1 - m) + fill * m)[0])[0, class_idx]))
    return np.array(probs)


def deletion_scores(model, x, cam_up, class_idx, steps=(0, .05, .1, .15, .2, .3, .4, .5), n_random=3, seed=0, grid=None):
    """Mean P(class) over the deletion range for CAM order vs random order (lower = deleting hurt more).
    ``grid`` = (h, w) of the CAM's native feature grid: the random baseline is a random map on that SAME coarse grid, bilinearly upsampled like the CAM,
    so it deletes blocks of the same granularity (a pixel-scatter baseline is far more destructive than a block and would make any CAM look bad)."""
    rng = np.random.default_rng(seed)
    S = cam_up.shape[0]
    cam_p = deletion_curve(model, x, np.argsort(-cam_up.ravel()), class_idx, steps)
    if grid is None:
        rnd_orders = [rng.permutation(cam_up.size) for _ in range(n_random)]
    else:
        rnd_orders = [np.argsort(-upsample(rng.random(grid), S).ravel()) for _ in range(n_random)]
    rnd = [deletion_curve(model, x, o, class_idx, steps) for o in rnd_orders]
    s_cam = trapezoid(cam_p, steps) / steps[-1]
    s_rnd = float(np.mean([trapezoid(r, steps) / steps[-1] for r in rnd]))
    return s_cam, s_rnd, cam_p[0]


def spearman(a, b):
    """Spearman rank correlation with AVERAGE ranks for ties (CAMs contain many exact zeros after ReLU / upsampling;
    a plain argsort-of-argsort ranks ties by position and would invent a correlation)."""
    from scipy.stats import rankdata
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ra, rb = rankdata(a, method="average"), rankdata(b, method="average")
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def boot_ci(v, n=2000, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    m = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return {"mean": float(v.mean()), "ci_low": float(np.quantile(m, .025)), "ci_high": float(np.quantile(m, .975)), "n": int(len(v))}


# ------------------------------------------------------------------------------------------------ driver
def run(a):
    import torch
    import pandas as pd
    from . import config as C
    from .dataset import UnifespDataset, build_uid_to_path
    from .train_v2 import build_model, get_transforms
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    names = [C.LABEL_NAMES[k] for k in sorted(C.LABEL_NAMES)]
    N = len(names)
    label_cols = [f"label_{k}_{C.LABEL_NAMES[k].replace(' ', '_')}" for k in range(N)]
    os.makedirs(a.out_dir, exist_ok=True)

    model = build_model(a.backbone, pretrained=False)
    model.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=True))
    model.to(device).eval()
    index = pd.read_csv(a.index_csv)
    hold = index[index["uid"].isin(set(pd.read_csv(a.holdout_csv)["uid"]))].reset_index(drop=True)
    if a.max_images:
        hold = hold.iloc[:a.max_images]
    ds = UnifespDataset(hold, C.DATA_ROOT, transform=get_transforms(a.img_size, False), uid_to_path=build_uid_to_path(C.DATA_ROOT))
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    Y = hold[label_cols].to_numpy(np.int64)

    sig, soft = [], []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            s, f = model(X[i:i + 32].to(device))
            sig.append(torch.sigmoid(s.float()).cpu().numpy()); soft.append(torch.softmax(f.float(), 1).cpu().numpy())
    blend = E.combine_heads(np.concatenate(sig), np.concatenate(soft), 0.5)
    pred = E.predict_labels(blend, 0.5, min_one=True)
    metrics = {"title": a.title or a.backbone, "weights": a.weights, "n_images": int(len(X)), "holdout_sample_f1_blend": E.sample_mean_f1(Y, pred)}

    cam_engine = GradCAM(model, a.img_size, device)
    metrics["target_layer"] = cam_engine.name
    metrics["channels_last"] = bool(cam_engine.channels_last)
    mean_t, std_t = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)

    def cam_for(i, cls):
        x = X[i:i + 1].to(device)
        cam, _, _ = cam_engine(x, [cls])
        return upsample(cam[0], a.img_size), cam[0].shape

    def disp(i):
        return (X[i:i + 1] * std_t + mean_t).clamp(0, 1)[0].permute(1, 2, 0).numpy()

    # ---- gallery: per class one correct and one missed example
    rows = []
    for c in range(N):
        pos = np.where(Y[:, c] == 1)[0]
        ok = [i for i in pos if pred[i, c] == 1]
        bad = [i for i in pos if pred[i, c] == 0]
        if ok:
            i = max(ok, key=lambda j: blend[j, c]); rows.append((c, "correct", i, c, c))
        if bad:
            i = min(bad, key=lambda j: blend[j, c]); wrong = int(np.argmax(np.where(Y[i] == 1, -1, blend[i]))); rows.append((c, "missed", i, c, wrong))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    per_page = 6
    for pg in range(0, len(rows), per_page):
        chunk = rows[pg:pg + per_page]
        fig, ax = plt.subplots(len(chunk), 3, figsize=(7.5, 2.5 * len(chunk)), squeeze=False)
        for r, (c, kind, i, tc, wc) in enumerate(chunk):
            img = disp(i)
            ax[r, 0].imshow(img); ax[r, 0].set_title(f"{kind}: true {names[c]}", fontsize=8)
            for col, cls, lab in ((1, tc, "true"), (2, wc, "predicted")):
                ax[r, col].imshow(img); ax[r, col].imshow(cam_for(i, cls)[0], cmap="jet", alpha=0.45)
                ax[r, col].set_title(f"CAM {lab}: {names[cls]} (p={blend[i, cls]:.2f})", fontsize=8)
            for k in range(3):
                ax[r, k].axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, f"gallery_page{pg // per_page + 1}.png"), dpi=130); plt.close(fig)
    metrics["gallery_rows"] = len(rows)

    # ---- faithfulness (deletion) and randomisation sanity check on predicted-correct (image, class) pairs
    pairs = [(i, int(c)) for i in range(len(X)) for c in np.where((Y[i] == 1) & (pred[i] == 1))[0]]
    rng = np.random.default_rng(a.seed)
    if len(pairs) > a.max_pairs:
        pairs = [pairs[j] for j in rng.choice(len(pairs), a.max_pairs, replace=False)]
    torch.manual_seed(a.seed)
    rnd_model = build_model(a.backbone, pretrained=False).to(device).eval()
    rnd_engine = GradCAM(rnd_model, a.img_size, device)
    d_cam, d_rnd, rho = [], [], []
    for i, c in pairs:
        up, grid = cam_for(i, c)
        sc, sr, _ = deletion_scores(model, X[i:i + 1].to(device), up, c, seed=a.seed + i, grid=grid)
        d_cam.append(sc); d_rnd.append(sr)
        cam_r, _, _ = rnd_engine(X[i:i + 1].to(device), [c])
        rho.append(spearman(up, upsample(cam_r[0], a.img_size)))
    d_cam, d_rnd, rho = np.array(d_cam), np.array(d_rnd), np.array(rho)
    metrics["faithfulness_deletion"] = {"pairs": int(len(pairs)), "score_cam": boot_ci(d_cam), "score_random_order": boot_ci(d_rnd),
                                        "deletion_gain_random_minus_cam": boot_ci(d_rnd - d_cam),
                                        "note": "mean P(class) over deleting 0-50 % of pixels; lower = more faithful; gain > 0 means CAM beats a random map of the same grid granularity"}
    metrics["randomization_sanity"] = {"spearman_trained_vs_random_weights": boot_ci(rho), "fraction_abs_rho_below_0.3": float((np.abs(rho) < 0.3).mean()),
                                       "note": "low correlation = the map depends on the learned weights (passes the check)"}

    # ---- error analysis
    conf = {}
    for i in range(len(Y)):
        for t in np.where((Y[i] == 1) & (pred[i] == 0))[0]:
            for p in np.where((Y[i] == 0) & (pred[i] == 1))[0]:
                conf[(names[t], names[p])] = conf.get((names[t], names[p]), 0) + 1
    tc = pd.DataFrame([{"true_class_missed": t, "predicted_instead": p, "count": n} for (t, p), n in conf.items()]).sort_values("count", ascending=False) if conf else \
        pd.DataFrame(columns=["true_class_missed", "predicted_instead", "count"])
    tc.to_csv(os.path.join(a.out_dir, "top_confusions.csv"), index=False)
    per_class = pd.DataFrame({"class": names, "n_true": Y.sum(0), "recall": [float(((pred[:, c] == 1) & (Y[:, c] == 1)).sum() / max(1, Y[:, c].sum())) for c in range(N)]})
    per_class.to_csv(os.path.join(a.out_dir, "per_class_recall.csv"), index=False)
    json.dump(metrics, open(os.path.join(a.out_dir, "explain_metrics.json"), "w"), indent=1)
    print(json.dumps({k: metrics[k] for k in ("title", "n_images", "target_layer", "faithfulness_deletion", "randomization_sanity")}, indent=1)[:2500])
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--index_csv", required=True)
    p.add_argument("--holdout_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--max_pairs", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
