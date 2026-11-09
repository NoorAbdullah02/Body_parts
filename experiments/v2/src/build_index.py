"""
Step 1a: Build a flat index of the UNIFESP dataset.

Reads every DICOM's header (no pixel decode, so it's fast) and writes a
single flat CSV that the rest of the pipeline uses. Uses config.DATA_ROOT /
config.INDEX_ROOT (not a path relative to this file) so it works no matter
where the raw images or the output location actually are -- e.g. run fresh
inside a Kaggle notebook against the attached images dataset, writing to
/kaggle/working, with no separately-uploaded "index" dataset needed at all.

Output: <INDEX_ROOT>/index.csv with columns:
    uid, split, patient_id, study_uid, series_uid, labels, label_0_Abdomen, ...
(no filepath column -- dataset.py locates files by uid via build_uid_to_path,
which is robust to whatever nesting the raw folders happen to have.)
"""
import os
import pydicom
import pandas as pd

from . import config as C
from .dataset import build_uid_to_path

LABEL_NAMES = C.LABEL_NAMES
N_CLASSES = C.N_CLASSES


def main():
    print("Indexing DICOM files under", C.DATA_ROOT, "...")
    train_uid_to_path = build_uid_to_path(os.path.join(C.DATA_ROOT, "train"))
    test_uid_to_path = build_uid_to_path(os.path.join(C.DATA_ROOT, "test"))
    print(f"train dcm files: {len(train_uid_to_path)} | test dcm files: {len(test_uid_to_path)}")

    rows = []
    for split, uid_to_path in (("train", train_uid_to_path), ("test", test_uid_to_path)):
        for uid, path in uid_to_path.items():
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
                patient_id = str(getattr(ds, "PatientID", ""))
                study_uid = str(getattr(ds, "StudyInstanceUID", ""))
                series_uid = str(getattr(ds, "SeriesInstanceUID", ""))
            except Exception as e:
                patient_id = study_uid = series_uid = ""
                print("WARN: failed to read header for", path, e)
            rows.append({
                "uid": uid, "split": split,
                "patient_id": patient_id, "study_uid": study_uid, "series_uid": series_uid,
            })

    index_df = pd.DataFrame(rows)

    # attach multilabel targets for the train split
    train_csv = pd.read_csv(os.path.join(C.DATA_ROOT, "train.csv"))
    train_csv["SOPInstanceUID"] = train_csv["SOPInstanceUID"].str.strip()
    label_map = dict(zip(train_csv["SOPInstanceUID"], train_csv["Target"]))
    index_df["labels"] = index_df["uid"].map(label_map)

    # one-hot columns for the 22 classes (NaN/absent for test rows)
    for k in range(N_CLASSES):
        col = f"label_{k}_{LABEL_NAMES[k].replace(' ', '_')}"
        index_df[col] = index_df["labels"].apply(
            lambda s, k=k: int(k in [int(x) for x in str(s).split()]) if pd.notna(s) else pd.NA
        )

    os.makedirs(C.INDEX_ROOT, exist_ok=True)
    out_path = os.path.join(C.INDEX_ROOT, "index.csv")
    index_df.to_csv(out_path, index=False)
    print("Saved index to", out_path, "| rows:", len(index_df))

    # ---- patient-level leakage sanity check ----
    train_idx = index_df[index_df["split"] == "train"]
    test_idx = index_df[index_df["split"] == "test"]
    train_patients = set(train_idx["patient_id"]) - {""}
    test_patients = set(test_idx["patient_id"]) - {""}
    overlap = train_patients & test_patients
    print(f"\nUnique patients -> train: {len(train_patients)}, test: {len(test_patients)}")
    print(f"Patient overlap between train/test (Kaggle's own split): {len(overlap)}")

    dup = train_idx.groupby("patient_id").size()
    multi_image_patients = (dup > 1).sum()
    print(f"Train patients with >1 image: {multi_image_patients} / {len(dup)}")


if __name__ == "__main__":
    main()
