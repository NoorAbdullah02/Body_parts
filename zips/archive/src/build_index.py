"""
Step 1a: Build a flat index of the UNIFESP dataset.

Walks the (messily nested) train/ and test/ folders, reads each DICOM's
header (no pixel decode, so it's fast), and writes a single flat CSV that
the rest of the pipeline uses instead of touching the raw folder structure
again. This avoids physically moving/renaming the original files.

Output: index.csv with columns:
    uid, split, filepath, patient_id, study_uid, series_uid, labels (train only)
"""
import os
import csv
import pydicom
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LABEL_NAMES = {
    0: "Abdomen", 1: "Ankle", 2: "Cervical Spine", 3: "Chest", 4: "Clavicles",
    5: "Elbow", 6: "Feet", 7: "Finger", 8: "Forearm", 9: "Hand", 10: "Hip",
    11: "Knee", 12: "Lower Leg", 13: "Lumbar Spine", 14: "Others", 15: "Pelvis",
    16: "Shoulder", 17: "Sinus", 18: "Skull", 19: "Thigh", 20: "Thoracic Spine",
    21: "Wrist",
}
N_CLASSES = len(LABEL_NAMES)


def list_dcm(folder):
    out = []
    for root, _dirs, files in os.walk(os.path.join(BASE, folder)):
        for f in files:
            if f.lower().endswith(".dcm"):
                out.append(os.path.join(root, f))
    return out


def basename_uid(path):
    b = os.path.basename(path)[:-4]
    if b.endswith("-c"):
        b = b[:-2]
    return b


def main():
    train_files = list_dcm("train")
    test_files = list_dcm("test")
    print(f"train dcm files: {len(train_files)} | test dcm files: {len(test_files)}")

    rows = []
    for split, files in (("train", train_files), ("test", test_files)):
        for path in files:
            uid = basename_uid(path)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
                patient_id = str(getattr(ds, "PatientID", ""))
                study_uid = str(getattr(ds, "StudyInstanceUID", ""))
                series_uid = str(getattr(ds, "SeriesInstanceUID", ""))
            except Exception as e:
                patient_id = study_uid = series_uid = ""
                print("WARN: failed to read header for", path, e)
            rows.append({
                "uid": uid,
                "split": split,
                "filepath": os.path.relpath(path, BASE).replace("\\", "/"),
                "patient_id": patient_id,
                "study_uid": study_uid,
                "series_uid": series_uid,
            })

    index_df = pd.DataFrame(rows)

    # attach multilabel targets for the train split
    train_csv = pd.read_csv(os.path.join(BASE, "train.csv"))
    train_csv["SOPInstanceUID"] = train_csv["SOPInstanceUID"].str.strip()
    label_map = dict(zip(train_csv["SOPInstanceUID"], train_csv["Target"]))
    index_df["labels"] = index_df["uid"].map(label_map)

    # one-hot columns for the 22 classes (NaN/absent for test rows)
    for k in range(N_CLASSES):
        col = f"label_{k}_{LABEL_NAMES[k].replace(' ', '_')}"
        index_df[col] = index_df["labels"].apply(
            lambda s, k=k: int(k in [int(x) for x in str(s).split()]) if pd.notna(s) else pd.NA
        )

    out_path = os.path.join(BASE, "index.csv")
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
