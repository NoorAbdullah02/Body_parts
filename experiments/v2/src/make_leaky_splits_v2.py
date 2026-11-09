"""
make_leaky_splits_v2 -- DELIBERATELY leaky image-level 5-fold split of the same CV pool (same held-out set), used only for
the split-type ablation ("how much does image-level splitting inflate the validation score?").

    python -m src.make_leaky_splits_v2            # reads splits/cv_folds.csv, writes splits/cv_folds_leaky.csv
"""
import os

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from . import config as C


def main():
    src = os.path.join(C.INDEX_ROOT, "splits", "cv_folds.csv")
    df = pd.read_csv(src)
    label_cols = [c for c in df.columns if c.startswith("label_")]
    Y = df[label_cols].to_numpy(int)
    fold = np.full(len(df), -1)
    for k, (_, val) in enumerate(MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=C.SEED).split(np.zeros(len(df)), Y)):
        fold[val] = k
    out = df.copy()
    out["fold"] = fold
    n_pat_multi = int((out.groupby("patient_id")["fold"].nunique() > 1).sum())
    dst = os.path.join(C.INDEX_ROOT, "splits", "cv_folds_leaky.csv")
    out.to_csv(dst, index=False)
    print(f"image-level split: {n_pat_multi} / {out['patient_id'].nunique()} patients have images in more than one fold (patient-level split: 0)")
    print("saved", dst)


if __name__ == "__main__":
    main()
