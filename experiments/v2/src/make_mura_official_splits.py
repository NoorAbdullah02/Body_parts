"""
make_mura_official_splits -- MURA with the OFFICIAL protocol so results are comparable with published work (e.g. Shah et al. 2026,
who test on MURA's public validation set): held-out set = every image under MURA's  valid/  folder; CV pool = every image under
train/, split into 5 PATIENT-level folds.  Replaces make_holdout_cv_splits for MURA.

    python -m src.make_mura_official_splits        # needs index.csv from build_index_mura (uids start with "train__" / "valid__")
Writes splits/holdout_uids.csv and splits/cv_folds.csv (same columns as the generic script).
"""
import os

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from . import config as C


def split(index_df, seed=C.SEED, n_folds=5):
    label_cols = [c for c in index_df.columns if c.startswith("label_")]
    is_valid = index_df["uid"].str.startswith("valid__")
    hold, pool = index_df[is_valid].reset_index(drop=True), index_df[~is_valid].reset_index(drop=True)
    assert len(hold) and len(pool), "need both train__ and valid__ uids (is DATA_ROOT the MURA-v1.1 root?)"
    overlap = set(hold["patient_id"]) & set(pool["patient_id"])
    assert not overlap, f"{len(overlap)} patient ids appear in both train and valid"
    pl = pool.groupby("patient_id")[label_cols].max().reset_index()
    fold = np.full(len(pl), -1)
    for k, (_, val) in enumerate(MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                                 .split(np.zeros(len(pl)), pl[label_cols].to_numpy(int))):
        fold[val] = k
    pool["fold"] = pool["patient_id"].map(dict(zip(pl["patient_id"], fold)))
    assert (pool["fold"] >= 0).all() and (pool.groupby("patient_id")["fold"].nunique() <= 1).all()
    return hold[["uid", "patient_id"] + label_cols], pool[["uid", "patient_id", "fold"] + label_cols]


def main():
    index = pd.read_csv(C.INDEX_CSV)
    hold, pool = split(index)
    out = os.path.join(C.INDEX_ROOT, "splits")
    os.makedirs(out, exist_ok=True)
    hold.to_csv(os.path.join(out, "holdout_uids.csv"), index=False)
    pool.to_csv(os.path.join(out, "cv_folds.csv"), index=False)
    print(f"held-out (MURA valid/): {len(hold)} images, {hold['patient_id'].nunique()} patients | "
          f"CV pool (train/): {len(pool)} images, {pool['patient_id'].nunique()} patients")
    print("per-fold images:", pool["fold"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
