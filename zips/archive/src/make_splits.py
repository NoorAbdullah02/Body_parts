"""
Step 1b: Patient-level, multilabel-stratified K-fold split.

Why patient-level: build_index.py found 483/768 train patients have >1 image,
so a plain random (or even plain multilabel-stratified) image-level split can
put the same patient's images in both train and validation -> leaky, overly
optimistic validation scores. We group by patient_id first, then stratify.

Why multilabel-stratified: UNIFESP's 22 classes are heavily imbalanced
(Chest 738 vs Clavicles 9), so a plain GroupKFold can starve a fold of a rare
class entirely. We stratify on each patient's aggregated (union) label
vector using iterative stratification.

Output: splits/fold_{i}.csv with columns [uid, fold] for every train image.
"""
import os
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_CLASSES = 22
N_FOLDS = 5
SEED = 42


def main():
    index_df = pd.read_csv(os.path.join(BASE, "index.csv"))
    train_df = index_df[index_df["split"] == "train"].copy()

    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    assert len(label_cols) == N_CLASSES, label_cols

    # aggregate to patient level: union of all labels this patient's images carry
    patient_labels = (
        train_df.groupby("patient_id")[label_cols]
        .max()
        .reset_index()
    )
    patients = patient_labels["patient_id"].values
    Y = patient_labels[label_cols].values.astype(int)

    print(f"Patients: {len(patients)} | images: {len(train_df)}")
    print("Per-class positive patient count:\n", pd.Series(Y.sum(axis=0), index=label_cols))

    mskf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    patient_fold = np.full(len(patients), -1, dtype=int)
    for fold, (_, val_idx) in enumerate(mskf.split(patients, Y)):
        patient_fold[val_idx] = fold

    patient_to_fold = dict(zip(patients, patient_fold))
    train_df["fold"] = train_df["patient_id"].map(patient_to_fold)

    assert (train_df["fold"] >= 0).all(), "every image must get a fold"

    out_cols = ["uid", "patient_id", "fold"] + label_cols
    out_path = os.path.join(BASE, "splits", "folds.csv")
    train_df[out_cols].to_csv(out_path, index=False)
    print("Saved", out_path)

    # sanity: no patient split across folds, per-fold class coverage
    leak = train_df.groupby("patient_id")["fold"].nunique()
    print("Patients appearing in >1 fold (should be 0):", (leak > 1).sum())

    print("\nPer-fold image count:\n", train_df["fold"].value_counts().sort_index())
    print("\nPer-fold positive image count per class:")
    print(train_df.groupby("fold")[label_cols].sum())


if __name__ == "__main__":
    main()
