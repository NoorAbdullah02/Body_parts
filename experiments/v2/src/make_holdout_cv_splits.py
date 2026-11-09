"""
Notebook 2's split scheme: a permanent, never-trained-on held-out test set,
plus a fresh 5-fold CV split restricted to the remaining pool.

This is different from make_splits.py's plain 5-fold split (where every
fold is, in turn, somebody's validation set -- fine for model *selection*,
but every image still ends up influencing model checkpointing at some
point). Here, one of the original 5 folds (--holdout_fold, default 4) is
carved out as a fixed test set that no training run -- across any of the
5 new CV folds -- ever sees, train or val. Each CV fold's best checkpoint
is later evaluated against this same fixed set (see evaluate_on_holdout.py),
giving a genuine mean +/- std generalization estimate instead of 5 folds
that are each other's test set.

Patient-level throughout (same reasoning as make_splits.py: many UNIFESP
patients have >1 image, so splitting by image would leak).

Output:
    <INDEX_ROOT>/splits/holdout_uids.csv   -- uid column, the fixed test set
    <INDEX_ROOT>/splits/cv_folds.csv       -- uid, cv_fold (0..4), pool only

Usage (after make_splits.py has already produced splits/folds.csv):
    python -m src.make_holdout_cv_splits --holdout_fold 4
"""
import argparse
import os

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from . import config as C

N_CLASSES = C.N_CLASSES
N_CV_FOLDS = 5
SEED = C.SEED


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--holdout_fold", type=int, default=4,
                    help="which of the original 5 folds (from make_splits.py) becomes the fixed holdout test set")
    args = p.parse_args()

    folds_df = pd.read_csv(C.FOLDS_CSV)
    label_cols = [c for c in folds_df.columns if c.startswith("label_")]
    assert len(label_cols) == N_CLASSES, label_cols

    holdout_df = folds_df[folds_df["fold"] == args.holdout_fold].reset_index(drop=True)
    pool_df = folds_df[folds_df["fold"] != args.holdout_fold].reset_index(drop=True)
    print(f"Holdout test (original fold {args.holdout_fold}): {len(holdout_df)} images, "
          f"{holdout_df['patient_id'].nunique()} patients")
    print(f"CV pool (remaining images): {len(pool_df)} images, {pool_df['patient_id'].nunique()} patients "
          f"({len(pool_df) / len(folds_df):.1%} of all train images)")

    # fresh patient-level multilabel-stratified 5-fold split, restricted to the pool only
    patient_labels = pool_df.groupby("patient_id")[label_cols].max().reset_index()
    patients = patient_labels["patient_id"].values
    Y = patient_labels[label_cols].values.astype(int)

    mskf = MultilabelStratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
    patient_cv_fold = np.full(len(patients), -1, dtype=int)
    for cv_fold, (_, val_idx) in enumerate(mskf.split(patients, Y)):
        patient_cv_fold[val_idx] = cv_fold
    patient_to_cv_fold = dict(zip(patients, patient_cv_fold))
    pool_df = pool_df.copy()
    pool_df["cv_fold"] = pool_df["patient_id"].map(patient_to_cv_fold)
    assert (pool_df["cv_fold"] >= 0).all()

    # sanity: no patient split across folds or across the holdout boundary
    leak = pool_df.groupby("patient_id")["cv_fold"].nunique()
    assert (leak <= 1).all(), "a pool patient landed in >1 cv_fold"
    assert set(holdout_df["patient_id"]) & set(pool_df["patient_id"]) == set(), \
        "a patient appears in both the holdout set and the CV pool"

    print("\nPer-cv_fold image count:\n", pool_df["cv_fold"].value_counts().sort_index())
    print("\nPer-cv_fold positive image count per class:")
    print(pool_df.groupby("cv_fold")[label_cols].sum())
    print("\nHoldout set positive image count per class:")
    print(holdout_df[label_cols].sum())

    splits_dir = os.path.join(C.INDEX_ROOT, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    holdout_out = os.path.join(splits_dir, "holdout_uids.csv")
    holdout_df[["uid", "patient_id"] + label_cols].to_csv(holdout_out, index=False)
    print("\nSaved:", holdout_out)

    cv_out = os.path.join(splits_dir, "cv_folds.csv")
    pool_df[["uid", "patient_id", "cv_fold"] + label_cols].rename(columns={"cv_fold": "fold"}).to_csv(
        cv_out, index=False
    )
    print("Saved:", cv_out)


if __name__ == "__main__":
    main()
