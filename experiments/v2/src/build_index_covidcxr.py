"""
Step 1a (COVID-19 Radiography Database variant of build_index.py): builds
index.csv for the Qatar University / University of Dhaka COVID-19
Radiography Database, re-tasked as 4-class chest X-ray diagnosis
classification (COVID / Lung_Opacity / Normal / Viral_Pneumonia) -- the
class is already encoded in each image's top-level folder name
(official Kaggle layout: COVID-19_Radiography_Dataset/<Class>/images/*.png,
plus a parallel <Class>/masks/*.png tree of lung-segmentation masks that
build_uid_to_path (dataset.py) already skips).

Unlike UNIFESP/ChestX-ray14 this is single-label (one diagnosis per image),
same as MURA -- a one-hot row is just a degenerate multilabel row, so
every downstream script runs unchanged. "Same code, dataset path +
last-layer neuron count (4) change."

No patient-level grouping here: this dataset is aggregated from many
different source datasets/publications (SIRM, Cohen's GitHub repo, the
Kaggle pneumonia dataset, 43 publications, ...) with no single consistent
patient-id field across all of them, so each image's own uid is used as
its own "patient_id" below -- i.e. an ordinary image-level split, not
patient-level. Worth naming as a limitation in the paper (the same reason
patient-level splitting matters for the other datasets applies here too,
we just don't have the id to do it).

Output: <INDEX_ROOT>/index.csv with columns:
    uid, split, patient_id, label_0_COVID, ...

Kaggle usage (attach tawsifurrahman/covid19-radiography-database, then):
    import os
    os.environ["UNIFESP_DATASET"] = "covidcxr"
    os.environ["UNIFESP_DATA_ROOT"] = "/kaggle/input/covid19-radiography-database"
    os.environ["UNIFESP_INDEX_ROOT"] = "/kaggle/working"
    os.environ["UNIFESP_OUTPUT_DIR"] = "/kaggle/working/runs"
    !python -m src.build_index_covidcxr
"""
import os

import pandas as pd

from . import config as C
from .dataset import build_uid_to_path

LABEL_NAMES = C.LABEL_NAMES
N_CLASSES = C.N_CLASSES


def parse_class_name(path, name_to_label_idx):
    # path components are like .../COVID-19_Radiography_Dataset/COVID/images/COVID-1.png
    # (or "Viral Pneumonia" with a space, or "Lung_Opacity") -- match against
    # the known class names directly rather than assuming exact folder casing.
    parts = [p.replace(" ", "_") for p in os.path.normpath(path).split(os.sep)]
    for p in parts:
        for name in name_to_label_idx:
            if p.lower() == name.lower():
                return name
    return None


def main():
    assert C.DATASET == "covidcxr", (
        f"UNIFESP_DATASET={C.DATASET!r}, expected 'covidcxr' -- set the env var "
        f"before running this (config.py picks the 4-class LABEL_NAMES from it)."
    )

    print("Indexing PNG files under", C.DATA_ROOT, "...")
    uid_to_path = build_uid_to_path(C.DATA_ROOT)
    print(f"found {len(uid_to_path)} .png files (mask files already excluded)")

    name_to_label_idx = {name: k for k, name in LABEL_NAMES.items()}

    rows = []
    skipped = 0
    for uid, path in uid_to_path.items():
        class_name = parse_class_name(path, name_to_label_idx)
        if class_name is None:
            skipped += 1
            continue
        # no shared patient id across this dataset's many source collections
        # -- fall back to image-level "patients" (see module docstring)
        rows.append({"uid": uid, "split": "train", "patient_id": uid, "class_name": class_name})
    if skipped:
        print(f"WARN: {skipped} file(s) didn't match any of {list(name_to_label_idx)}, skipped")

    index_df = pd.DataFrame(rows)

    for k in range(N_CLASSES):
        name = LABEL_NAMES[k]
        col = f"label_{k}_{name.replace(' ', '_')}"
        index_df[col] = (index_df["class_name"] == name).astype(int)

    index_df = index_df.drop(columns=["class_name"])

    os.makedirs(C.INDEX_ROOT, exist_ok=True)
    out_path = os.path.join(C.INDEX_ROOT, "index.csv")
    index_df.to_csv(out_path, index=False)
    print("Saved index to", out_path, "| rows:", len(index_df))

    label_cols = [c for c in index_df.columns if c.startswith("label_")]
    print("\nPer-class image count:\n", index_df[label_cols].sum())
    print(f"\nNote: patient_id == uid for every row (no shared patient metadata "
          f"across this dataset's source collections) -- splits are image-level, not patient-level.")


if __name__ == "__main__":
    main()
