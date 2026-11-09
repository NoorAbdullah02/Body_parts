"""
Step 1a (ChestX-ray14 variant of build_index.py): builds index.csv for the
NIH ChestX-ray14 dataset in the exact schema build_index.py produces for
UNIFESP, so make_splits.py / make_holdout_cv_splits.py / train.py /
train_hybrid.py / evaluate_on_holdout.py all run completely unchanged --
"same code, just dataset path + last-layer neuron count change" (the neuron
count changes automatically: see config.py's DATASET switch).

Reads Data_entry_2017.csv's "Finding Labels" column (pipe-separated, e.g.
"Cardiomegaly|Effusion", or "No Finding") into one-hot label_k_<Name>
columns. "No Finding" is deliberately not its own class -- it's the
all-zero label vector, the same multilabel convention the CheXNet-style
literature on this dataset uses, and engine.py's sample_mean_f1/
compute_metrics already handle an all-zero row correctly.

Every row is marked split="train" -- there's no Kaggle-leaderboard test set
to preserve here. The pipeline's own 80/20 holdout is carved out downstream
by make_holdout_cv_splits.py, same as for UNIFESP.

Output: <INDEX_ROOT>/index.csv with columns:
    uid, split, patient_id, label_0_Atelectasis, ...

Kaggle usage (attach nih-chest-xrays/data as the images dataset, then):
    import os
    os.environ["UNIFESP_DATASET"] = "chestxray14"
    os.environ["UNIFESP_DATA_ROOT"] = "/kaggle/input/data"
    os.environ["UNIFESP_INDEX_ROOT"] = "/kaggle/working"
    os.environ["UNIFESP_OUTPUT_DIR"] = "/kaggle/working/runs"
    !python -m src.build_index_chestxray14
"""
import os

import pandas as pd

from . import config as C
from .dataset import build_uid_to_path

LABEL_NAMES = C.LABEL_NAMES
N_CLASSES = C.N_CLASSES


ENTRY_CSV_NAMES = {"data_entry_2017.csv", "sample_labels.csv"}  # full dataset vs. the 5% nih-chest-xrays/sample


def find_entry_csv(data_root):
    for root, _dirs, files in os.walk(data_root):
        for f in files:
            if f.lower() in ENTRY_CSV_NAMES:
                return os.path.join(root, f)
    raise FileNotFoundError(
        f"Neither Data_entry_2017.csv nor sample_labels.csv found anywhere under {data_root}. "
        f"Check UNIFESP_DATA_ROOT points at the attached nih-chest-xrays/data (or nih-chest-xrays/sample) dataset."
    )


def main():
    assert C.DATASET == "chestxray14", (
        f"UNIFESP_DATASET={C.DATASET!r}, expected 'chestxray14' -- set the env var "
        f"before running this (config.py picks the 14-class LABEL_NAMES from it)."
    )

    print("Indexing PNG files under", C.DATA_ROOT, "...")
    uid_to_path = build_uid_to_path(C.DATA_ROOT)
    print(f"found {len(uid_to_path)} .png files")

    entry_csv = find_entry_csv(C.DATA_ROOT)
    entry_df = pd.read_csv(entry_csv)
    entry_df["uid"] = entry_df["Image Index"].str.replace(".png", "", regex=False)
    meta_by_uid = entry_df.set_index("uid")[["Patient ID", "Finding Labels"]].to_dict("index")

    rows = []
    missing_meta = 0
    for uid in uid_to_path:
        meta = meta_by_uid.get(uid)
        if meta is None:
            missing_meta += 1
            continue
        rows.append({
            "uid": uid,
            "split": "train",
            "patient_id": str(meta["Patient ID"]),
            "finding_labels": meta["Finding Labels"],
        })
    if missing_meta:
        print(f"WARN: {missing_meta} PNG file(s) had no matching row in Data_entry_2017.csv, skipped")

    index_df = pd.DataFrame(rows)

    for k in range(N_CLASSES):
        name = LABEL_NAMES[k]
        col = f"label_{k}_{name.replace(' ', '_')}"
        index_df[col] = index_df["finding_labels"].apply(
            lambda s, name=name: int(name in str(s).split("|"))
        )

    index_df = index_df.drop(columns=["finding_labels"])

    os.makedirs(C.INDEX_ROOT, exist_ok=True)
    out_path = os.path.join(C.INDEX_ROOT, "index.csv")
    index_df.to_csv(out_path, index=False)
    print("Saved index to", out_path, "| rows:", len(index_df))

    # ---- sanity checks (mirrors build_index.py's patient-level checks) ----
    n_patients = index_df["patient_id"].nunique()
    dup = index_df.groupby("patient_id").size()
    multi_image_patients = (dup > 1).sum()
    print(f"\nUnique patients: {n_patients}")
    print(f"Patients with >1 image: {multi_image_patients} / {n_patients}")

    label_cols = [c for c in index_df.columns if c.startswith("label_")]
    print("\nPer-class positive image count:\n", index_df[label_cols].sum())
    no_finding = (index_df[label_cols].sum(axis=1) == 0).sum()
    print(f"\n'No Finding' images (all-zero label vector): {no_finding} / {len(index_df)}")


if __name__ == "__main__":
    main()
