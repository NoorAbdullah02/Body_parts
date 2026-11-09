"""Test of src/make_mura_official_splits.split on a synthetic MURA-like index. Run: python test_mura_split.py"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ["UNIFESP_DATASET"] = "mura"
from src import make_mura_official_splits as M   # noqa: E402

rng = np.random.default_rng(0)
names = ["Elbow", "Finger", "Forearm", "Hand", "Humerus", "Shoulder", "Wrist"]
rows = []
for split, pats in (("train", range(0, 400)), ("valid", range(400, 460))):
    for p in pats:
        part = rng.integers(0, 7)
        for im in range(rng.integers(1, 4)):
            r = {"uid": f"{split}__XR_{names[part].upper()}__patient{p:05d}__study1_positive__image{im+1}", "split": "train", "patient_id": f"patient{p:05d}"}
            r.update({f"label_{k}_{n}": int(k == part) for k, n in enumerate(names)})
            rows.append(r)
idx = pd.DataFrame(rows)
hold, pool = M.split(idx)
assert hold["uid"].str.startswith("valid__").all() and pool["uid"].str.startswith("train__").all()
assert len(hold) + len(pool) == len(idx)
assert (pool.groupby("patient_id")["fold"].nunique() == 1).all()                  # patient-level folds
assert sorted(pool["fold"].unique()) == [0, 1, 2, 3, 4]
fr = pool.groupby("fold").size(); assert fr.min() > 0.5 * fr.max(), fr.to_dict()    # roughly balanced
print("per-fold images", fr.to_dict(), "| holdout", len(hold), "| mura official split test passed")
