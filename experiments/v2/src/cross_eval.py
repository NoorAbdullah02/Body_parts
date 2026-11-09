"""
cross_eval -- zero-shot transfer of body-part models between public datasets through an explicit class map.

    UNIFESP model  --> MURA images        (6 shared classes; MURA "Humerus" has no UNIFESP counterpart and is dropped)
    MURA model     --> UNIFESP images     (single-label images of the 6 shared classes)
    either model   --> FracAtlas images   (hand / shoulder / hip / "leg" group; images with several regions are dropped)

Predictions are restricted to the groups present in BOTH taxonomies and renormalised, i.e. a closed-set decision among
the shared groups.  The class map is an ASSUMPTION to be checked by a radiologist (especially FracAtlas "leg", which can
cover lower leg / knee / ankle); edit ``GROUPS_*`` or pass --map_json.  FracAtlas column names are NOT verified: the
loader prints the columns it finds and fails loudly if the region columns are missing.

Inputs per source run: <weights_dir>/<run_prefix>_f<k>/final_ema.pt (written by train_v2), averaged over folds (bagging).
"""
import argparse
import json
import os
import re

import numpy as np

from . import evalkit as E

# ------------------------------------------------------------------ taxonomy -> shared group (assumption!)
GROUPS_UNIFESP = {"Elbow": "elbow", "Finger": "finger", "Forearm": "forearm", "Hand": "hand", "Shoulder": "shoulder",
                  "Wrist": "wrist", "Hip": "hip", "Lower Leg": "leg", "Knee": "leg", "Thigh": "leg"}
GROUPS_MURA = {"Elbow": "elbow", "Finger": "finger", "Forearm": "forearm", "Hand": "hand", "Shoulder": "shoulder",
               "Wrist": "wrist"}                                    # Humerus: no counterpart
GROUPS_FRACATLAS = {"hand": "hand", "leg": "leg", "hip": "hip", "shoulder": "shoulder"}
GROUPS = {"unifesp": GROUPS_UNIFESP, "mura": GROUPS_MURA, "fracatlas": GROUPS_FRACATLAS}


def class_names_from_columns(cols):
    """label_3_Chest -> 'Chest' ; label_12_Lower_Leg -> 'Lower Leg' (index order preserved)."""
    out = {}
    for c in cols:
        m = re.match(r"label_(\d+)_(.*)$", c)
        if m:
            out[int(m[1])] = m[2].replace("_", " ")
    return [out[k] for k in sorted(out)]


def shared_groups(src_map, tgt_map):
    return sorted(set(src_map.values()) & set(tgt_map.values()))


def group_probabilities(prob, class_names, class_to_group, groups, how="sum"):
    """(N, n_classes) -> (N, n_groups): sum (softmax head) or max (sigmoid head) over the classes of each group, renormalised."""
    out = np.zeros((prob.shape[0], len(groups)))
    for gi, g in enumerate(groups):
        idx = [i for i, n in enumerate(class_names) if class_to_group.get(n) == g]
        if idx:
            out[:, gi] = prob[:, idx].sum(1) if how == "sum" else prob[:, idx].max(1)
    s = out.sum(1, keepdims=True)
    return out / np.where(s > 0, s, 1.0)


def transfer_metrics(prob_groups, y_group, groups):
    """Closed-set accuracy / macro-F1 / per-group recall + confusion matrix over the shared groups."""
    pred = prob_groups.argmax(1)
    acc = float((pred == y_group).mean())
    f1s, rec = [], {}
    for gi, g in enumerate(groups):
        tp = int(((pred == gi) & (y_group == gi)).sum()); fp = int(((pred == gi) & (y_group != gi)).sum())
        fn = int(((pred != gi) & (y_group == gi)).sum())
        den = 2 * tp + fp + fn
        f1s.append(0.0 if den == 0 else 2 * tp / den)
        rec[g] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    cm = np.zeros((len(groups), len(groups)), dtype=int)
    for t, p in zip(y_group, pred):
        cm[t, p] += 1
    return {"n_images": int(len(y_group)), "accuracy": acc, "macro_f1": float(np.mean(f1s)),
            "per_group_recall": rec, "groups": list(groups), "confusion": cm.tolist()}


# ------------------------------------------------------------------ target datasets
def _stem_uid(path):                    # FracAtlas: IMG0000019.jpg -> IMG0000019
    return os.path.splitext(os.path.basename(path))[0]


def build_path_map(root, kind):
    """uid -> file path for a target dataset (independent of the global config)."""
    from .dataset import _basename_uid
    m = {}
    for cur, _d, files in os.walk(root):
        for f in files:
            if f.startswith("._"):
                continue
            low = f.lower()
            p = os.path.join(cur, f)
            if kind == "unifesp" and low.endswith(".dcm"):
                m[_basename_uid(f)] = p
            elif kind == "mura" and low.endswith(".png"):
                m[os.path.relpath(p, root).replace(os.sep, "__")[:-4]] = p
            elif kind == "fracatlas" and low.endswith((".jpg", ".jpeg", ".png")):
                m[_stem_uid(p)] = p
    return m


MURA_RE = re.compile(r"XR_([A-Z]+)")


def target_items(kind, root, index_csv=None, fracatlas_csv=None, uids_csv=None):
    """Returns (uids, paths, class_names_of_target_taxonomy, labels) with ONE label (class name) per image, dropping
    multi-label / unmapped images. ``uids_csv`` (optional, column uid) restricts to a given subset (e.g. the held-out set)."""
    import pandas as pd
    paths = build_path_map(root, kind)
    if kind == "mura":
        items = []
        for uid, p in paths.items():
            m = MURA_RE.search(p.upper())
            if m:
                items.append((uid, p, m[1].title()))
    elif kind == "unifesp":
        df = pd.read_csv(index_csv)
        df = df[df["split"] == "train"]
        cols = [c for c in df.columns if c.startswith("label_")]
        names = class_names_from_columns(cols)
        single = df[df[cols].sum(1) == 1]
        items = [(r["uid"], paths[r["uid"]], names[int(np.argmax(r[cols].to_numpy(float)))])
                 for _, r in single.iterrows() if r["uid"] in paths]
    elif kind == "fracatlas":
        df = pd.read_csv(fracatlas_csv)
        low = {c.lower(): c for c in df.columns}
        need = ["hand", "leg", "hip", "shoulder"]
        missing = [c for c in need if c not in low]
        if missing or "image_id" not in low:
            raise SystemExit(f"FracAtlas csv columns not as expected (need image_id + {need}); found {list(df.columns)}")
        reg = df[[low[c] for c in need]].to_numpy(int)
        items = []
        for (_, r), flags in zip(df.iterrows(), reg):
            if flags.sum() != 1:
                continue
            uid = _stem_uid(str(r[low["image_id"]]))
            if uid in paths:
                items.append((uid, paths[uid], need[int(flags.argmax())]))
    else:
        raise ValueError(kind)
    if uids_csv:
        keep = set(pd.read_csv(uids_csv)["uid"])
        items = [it for it in items if it[0] in keep]
    uids, ps, labs = zip(*items) if items else ([], [], [])
    return list(uids), list(ps), list(labs)


# ------------------------------------------------------------------ inference
def predict_images(paths, weights, backbone, img_size, device, batch=32, minmax=True, tta=False):
    """Average softmax / sigmoid probabilities of several state_dicts (fold models) on a list of image paths.
    Every image is decoded ONCE and passed through all models."""
    import torch
    from .dataset import load_dicom_as_image, load_png_as_image
    from .train_v2 import build_model, get_transforms
    tf = get_transforms(img_size, train=False)

    from PIL import ImageFile
    n_trunc = [0]

    def load(p):
        try:
            img = load_dicom_as_image(p) if p.lower().endswith(".dcm") else load_png_as_image(p)
        except OSError:                      # a truncated file (seen in FracAtlas): decode what is there, count it, report it
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            n_trunc[0] += 1
            print(f"[warn] truncated image decoded partially: {p}")
            img = load_png_as_image(p)
            ImageFile.LOAD_TRUNCATED_IMAGES = False
        if minmax and not p.lower().endswith(".dcm"):                      # same per-image min-max as the DICOM loader
            a = img.astype(np.float32); lo, hi = a.min(), a.max()
            img = (((a - lo) / (hi - lo + 1e-6)) * 255).astype(np.uint8)
        return tf(image=img)["image"]

    models = []
    for w in weights:
        m = build_model(backbone, pretrained=False)
        m.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
        models.append(m.to(device).eval())
    soft, sig = [], []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            x = torch.stack([load(p) for p in paths[i:i + batch]]).to(device)
            ps_all, pf_all = 0.0, 0.0
            for m in models:
                s_, f_ = m(x)
                ps, pf = torch.sigmoid(s_.float()), torch.softmax(f_.float(), 1)
                if tta:
                    s2, f2 = m(torch.flip(x, dims=[3]))
                    ps, pf = (ps + torch.sigmoid(s2.float())) / 2, (pf + torch.softmax(f2.float(), 1)) / 2
                ps_all, pf_all = ps_all + ps, pf_all + pf
            sig.append((ps_all / len(models)).cpu().numpy()); soft.append((pf_all / len(models)).cpu().numpy())
    return np.concatenate(soft), np.concatenate(sig)


def run(a):
    import torch
    os.environ.setdefault("UNIFESP_DATASET", a.src_dataset)
    from . import config as C
    src_names = [C.LABEL_NAMES[k] for k in sorted(C.LABEL_NAMES)]
    src_map = json.load(open(a.map_json))[a.src_dataset] if a.map_json else GROUPS[a.src_dataset]
    tgt_map = json.load(open(a.map_json))[a.target] if a.map_json else GROUPS[a.target]
    groups = shared_groups(src_map, tgt_map)
    uids, paths, labs = target_items(a.target, a.target_root, a.target_index, a.fracatlas_csv, a.target_uids_csv)
    keep = [i for i, l in enumerate(labs) if tgt_map.get(l) in groups]
    if a.max_images and len(keep) > a.max_images:                       # reproducible random subsample (speed)
        keep = sorted(np.random.default_rng(0).choice(keep, size=a.max_images, replace=False).tolist())
    paths = [paths[i] for i in keep]; uids = [uids[i] for i in keep]
    y = np.array([groups.index(tgt_map[labs[i]]) for i in keep])
    print(f"[{a.src_dataset} -> {a.target}] shared groups {groups}; {len(y)} target images "
          f"(of {len(labs)} single-label); per-group n = {dict(zip(groups, np.bincount(y, minlength=len(groups)).tolist()))}")
    folds = [int(f) for f in a.folds.split(",")]
    weights = [os.path.join(a.weights_dir, f"{a.run_prefix}_f{f}", "final_ema.pt") for f in folds]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    soft, sig = predict_images(paths, weights, a.backbone, a.img_size, device, a.batch, tta=a.tta)
    out = {"source": a.src_dataset, "target": a.target, "run": a.run_prefix, "groups": groups}
    for head, p, how in (("softmax", soft, "sum"), ("sigmoid", sig, "max")):
        pg = group_probabilities(p, src_names, src_map, groups, how)
        out[head] = transfer_metrics(pg, y, groups)
        # patient/image bootstrap is image-level here (no patient ids for the cross-dataset targets) -- say so in the paper
        pred = pg.argmax(1)
        ones = np.ones((len(y), 1), int)
        out[head]["accuracy_ci_image_bootstrap"] = E.cluster_bootstrap(
            lambda yt, yp: float((yt[:, 0] == yp[:, 0]).mean()), y[:, None], pred[:, None], np.arange(len(y)),
            n_boot=a.n_boot)
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"transfer_{a.src_dataset}_to_{a.target}_{a.run_prefix}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps({k: {m: out[k][m] for m in ("n_images", "accuracy", "macro_f1")} for k in ("softmax", "sigmoid")}, indent=1))
    print("saved", path)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dataset", required=True, choices=["unifesp", "mura"])
    p.add_argument("--target", required=True, choices=["unifesp", "mura", "fracatlas"])
    p.add_argument("--target_root", required=True)
    p.add_argument("--target_index", default=None, help="index.csv of the UNIFESP target")
    p.add_argument("--fracatlas_csv", default=None)
    p.add_argument("--target_uids_csv", default=None, help="optional: restrict to these uids (e.g. holdout_uids.csv)")
    p.add_argument("--weights_dir", required=True)
    p.add_argument("--run_prefix", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--map_json", default=None)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--max_images", type=int, default=0, help="0 = all target images; else random subsample of this size")
    p.add_argument("--out_dir", default="transfer_results")
    run(p.parse_args())


if __name__ == "__main__":
    main()
