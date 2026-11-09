"""
Step 1a (MURA variant of build_index.py): builds index.csv for the Stanford
MURA musculoskeletal radiograph dataset, re-tasked here as a BODY-PART
classification problem -- the same task UNIFESP solves -- instead of MURA's
own normal/abnormal task. The body part each image shows is already
encoded in its folder name (.../XR_ELBOW/patientNNNNN/study1_.../image1.png,
official MURA v1.1 layout), so this is a relabeling, not new annotation.

Unlike UNIFESP/ChestX-ray14, MURA is single-label here (each image shows
exactly one of 7 body parts), not multilabel -- but a one-hot row is just a
degenerate multilabel row, so make_splits.py / make_holdout_cv_splits.py /
train.py / train_hybrid.py / evaluate_on_holdout.py all run completely
unchanged. "Same code, dataset path + last-layer neuron count (7) change."

Both MURA's train/ and valid/ folders are pooled here (every row marked
split="train") since our own 80/20 holdout + 5-fold CV
(make_holdout_cv_splits.py) replaces MURA's own split, same as for
chestxray14 -- there's no Kaggle-leaderboard test set to preserve.

Patient id is parsed from each image's path ("patientNNNNN"), for the same
patient-level splitting make_splits.py already does (many patients have
multiple studies/images here too).

Output: <INDEX_ROOT>/index.csv with columns:
    uid, split, patient_id, label_0_Elbow, ...

Kaggle usage (attach cjinny/mura-v11 as the images dataset, then):
    import os
    os.environ["UNIFESP_DATASET"] = "mura"
    os.environ["UNIFESP_DATA_ROOT"] = "/kaggle/input/mura-v11"
    os.environ["UNIFESP_INDEX_ROOT"] = "/kaggle/working"
    os.environ["UNIFESP_OUTPUT_DIR"] = "/kaggle/working/runs"
    !python -m src.build_index_mura
"""
import os
import re

import pandas as pd

from . import config as C
from .dataset import build_uid_to_path

LABEL_NAMES = C.LABEL_NAMES
N_CLASSES = C.N_CLASSES

BODY_PART_RE = re.compile(r"XR_([A-Z]+)")
PATIENT_RE = re.compile(r"(patient\d+)", re.IGNORECASE)


def parse_body_part(path):
    m = BODY_PART_RE.search(path.upper())
    return m.group(1).title() if m else None


def parse_patient_id(path):
    m = PATIENT_RE.search(path)
    return m.group(1) if m else None


def main():
    assert C.DATASET == "mura", (
        f"UNIFESP_DATASET={C.DATASET!r}, expected 'mura' -- set the env var "
        f"before running this (config.py picks the 7-class LABEL_NAMES from it)."
    )

    print("Indexing PNG files under", C.DATA_ROOT, "...")
    uid_to_path = build_uid_to_path(C.DATA_ROOT)
    print(f"found {len(uid_to_path)} .png files")

    name_to_label_idx = {name: k for k, name in LABEL_NAMES.items()}

    rows = []
    skipped = 0
    for uid, path in uid_to_path.items():
        body_part = parse_body_part(path)
        patient_id = parse_patient_id(path)
        if body_part is None or patient_id is None or body_part not in name_to_label_idx:
            skipped += 1
            continue
        rows.append({"uid": uid, "split": "train", "patient_id": patient_id, "body_part": body_part})
    if skipped:
        print(f"WARN: {skipped} file(s) didn't match the expected "
              f".../XR_<PART>/patientNNNNN/... path pattern, skipped")

    index_df = pd.DataFrame(rows)

    for k in range(N_CLASSES):
        name = LABEL_NAMES[k]
        col = f"label_{k}_{name.replace(' ', '_')}"
        index_df[col] = (index_df["body_part"] == name).astype(int)

    index_df = index_df.drop(columns=["body_part"])

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
    print("\nPer-class image count:\n", index_df[label_cols].sum())


if __name__ == "__main__":
    main()
