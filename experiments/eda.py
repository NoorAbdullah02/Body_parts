"""
eda -- exploratory statistics + figures of the UNIFESP dataset for the paper's Data section (CPU only, ~1-3 min).

    python eda.py --data_root <folder with train/ test/ train.csv> --splits_dir <folder with cv_folds.csv, holdout_uids.csv> --out results/eda

Writes (all numbers are computed from the files, nothing is typed in by hand):
    eda_summary.json, dataset_table.md, class_table.md, fig_class_counts.png, fig_labels_per_image.png, fig_images_per_patient.png,
    fig_cooccurrence.png, fig_montage.png, dicom_tags.csv
"""
import argparse
import collections
import glob
import json
import os

import numpy as np
import pandas as pd

NAMES = ["Abdomen", "Ankle", "Cervical Spine", "Chest", "Clavicles", "Elbow", "Feet", "Finger", "Forearm", "Hand", "Hip", "Knee", "Lower Leg",
         "Lumbar Spine", "Others", "Pelvis", "Shoulder", "Sinus", "Skull", "Thigh", "Thoracic Spine", "Wrist"]


def uid_of(path):
    b = os.path.basename(path)[:-4]
    return b[:-2] if b.endswith("-c") else b


def main():
    import pydicom
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--splits_dir", default=None)
    ap.add_argument("--out", default="results/eda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    train_csv = pd.read_csv(os.path.join(a.data_root, "train.csv"))
    train_csv["SOPInstanceUID"] = train_csv["SOPInstanceUID"].str.strip()
    labels = {r.SOPInstanceUID: sorted(int(x) for x in str(r.Target).split()) for r in train_csv.itertuples()}
    files = {split: {uid_of(p): p for p in glob.glob(os.path.join(a.data_root, split, "**", "*.dcm"), recursive=True)} for split in ("train", "test")}

    rows = []
    for split, m in files.items():
        for uid, p in m.items():
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            rows.append({"uid": uid, "split": split, "patient_id": str(getattr(ds, "PatientID", "")), "photometric": str(getattr(ds, "PhotometricInterpretation", "")),
                         "bits_stored": int(getattr(ds, "BitsStored", 0) or 0), "rows": int(getattr(ds, "Rows", 0) or 0), "cols": int(getattr(ds, "Columns", 0) or 0),
                         "modality": str(getattr(ds, "Modality", "")), "body_part_examined": str(getattr(ds, "BodyPartExamined", "")),
                         "view_position": str(getattr(ds, "ViewPosition", "")), "study_desc": str(getattr(ds, "StudyDescription", ""))})
    meta = pd.DataFrame(rows)
    meta.to_csv(os.path.join(a.out, "dicom_tags.csv"), index=False)
    tr = meta[meta.split == "train"].copy()
    tr["labels"] = tr.uid.map(labels)
    tr["n_labels"] = tr.labels.map(len)

    S = {"n_train": int((meta.split == "train").sum()), "n_test": int((meta.split == "test").sum()),
         "n_train_patients": int(tr.patient_id.nunique()), "n_test_patients": int(meta[meta.split == "test"].patient_id.nunique()),
         "patient_overlap_train_test": len(set(tr.patient_id) & set(meta[meta.split == "test"].patient_id)),
         "train_patients_with_gt1_image": int((tr.groupby("patient_id").size() > 1).sum()),
         "max_images_per_patient": int(tr.groupby("patient_id").size().max()),
         "labels_per_image": {int(k): int(v) for k, v in tr.n_labels.value_counts().sort_index().items()},
         "single_label_fraction": float((tr.n_labels == 1).mean()),
         "image_size_counts": {f"{r}x{c}": int(n) for (r, c), n in meta.groupby(["rows", "cols"]).size().items()},
         "photometric_counts": {k: int(v) for k, v in meta.photometric.value_counts().items()},
         "bits_stored_counts": {int(k): int(v) for k, v in meta.bits_stored.value_counts().items()},
         "modality_counts": {k: int(v) for k, v in meta.modality.value_counts().items()},
         "body_part_examined_nonempty": int((meta.body_part_examined.str.strip() != "").sum()),
         "body_part_examined_values": {k: int(v) for k, v in meta.body_part_examined.value_counts().head(10).items()},
         "view_position_values": {k: int(v) for k, v in meta.view_position.value_counts().head(8).items()},
         "study_description_nonempty": int((meta.study_desc.str.strip() != "").sum())}

    # class table + co-occurrence
    cnt = collections.Counter(c for l in tr.labels for c in l)
    single = collections.Counter(l[0] for l in tr.labels if len(l) == 1)
    pat_per_class = {c: len({p for p, l in zip(tr.patient_id, tr.labels) if c in l}) for c in range(22)}
    ct = pd.DataFrame({"class": NAMES, "images": [cnt[c] for c in range(22)], "share_%": [round(100 * cnt[c] / len(tr), 1) for c in range(22)],
                       "single_label_images": [single[c] for c in range(22)], "patients": [pat_per_class[c] for c in range(22)]})
    M = np.zeros((22, 22), int)
    for l in tr.labels:
        for i in l:
            for j in l:
                M[i, j] += 1
    pairs = sorted(((M[i, j], NAMES[i], NAMES[j]) for i in range(22) for j in range(i + 1, 22) if M[i, j]), reverse=True)[:8]
    S["top_cooccurring_pairs"] = [{"pair": f"{b} + {c}", "images": int(n)} for n, b, c in pairs]
    S["imbalance_ratio_max_over_min"] = float(max(cnt.values()) / min(cnt.values()))

    # splits (if given)
    if a.splits_dir and os.path.exists(os.path.join(a.splits_dir, "holdout_uids.csv")):
        ho = set(pd.read_csv(os.path.join(a.splits_dir, "holdout_uids.csv"))["uid"])
        cv = pd.read_csv(os.path.join(a.splits_dir, "cv_folds.csv"))
        hmeta = tr[tr.uid.isin(ho)]
        S["holdout"] = {"images": int(len(hmeta)), "patients": int(hmeta.patient_id.nunique())}
        S["cv_pool"] = {"images": int(len(cv)), "patients": int(cv.patient_id.nunique()), "images_per_fold": {int(k): int(v) for k, v in cv.groupby("fold").size().items()}}
        S["holdout_pool_patient_overlap"] = len(set(hmeta.patient_id) & set(cv.patient_id))
        ct["holdout_images"] = [sum(1 for l in hmeta.labels if c in l) for c in range(22)]
        ct["cv_pool_images"] = [cnt[c] - ct["holdout_images"][c] for c in range(22)]
    ct.to_csv(os.path.join(a.out, "class_table.csv"), index=False)
    json.dump(S, open(os.path.join(a.out, "eda_summary.json"), "w"), indent=1)

    # pixel statistics + montage (decoded exactly like the pipeline: rescale, MONOCHROME1 inversion, per-image min-max)
    def load(p):
        ds = pydicom.dcmread(p, force=True)
        arr = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
        if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
            arr = arr.max() - arr
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    fig, ax = plt.subplots(4, 6, figsize=(12, 8.4))
    for k, axx in enumerate(ax.ravel()):
        axx.axis("off")
        if k < 22:
            uid = next((u for u, l in labels.items() if l == [k] and u in files["train"]), None)
            if uid:
                axx.imshow(load(files["train"][uid]), cmap="gray"); axx.set_title(f"{NAMES[k]} (n={cnt[k]})", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_montage.png"), dpi=130); plt.close(fig)

    # figures
    order = ct.sort_values("images", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4)); ax.bar(order["class"], order["images"]); ax.set_yscale("log"); ax.set_ylabel("images (log scale)")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right"); ax.set_title(f"UNIFESP training images per class (n={S['n_train']}; max/min = {S['imbalance_ratio_max_over_min']:.0f}x)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_class_counts.png"), dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 3)); v = tr.n_labels.value_counts().sort_index(); ax.bar(v.index.astype(str), v.values); ax.set_xlabel("labels per image"); ax.set_ylabel("images")
    for i, n in enumerate(v.values): ax.text(i, n, str(n), ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_labels_per_image.png"), dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.5, 3)); v = tr.groupby("patient_id").size().value_counts().sort_index(); ax.bar(v.index.astype(str), v.values)
    ax.set_xlabel("images per patient"); ax.set_ylabel("patients"); fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_images_per_patient.png"), dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6)); Mo = M.astype(float).copy(); np.fill_diagonal(Mo, np.nan)
    im = ax.imshow(np.log1p(Mo), cmap="viridis"); ax.set_xticks(range(22)); ax.set_yticks(range(22)); ax.set_xticklabels(NAMES, rotation=90, fontsize=7); ax.set_yticklabels(NAMES, fontsize=7)
    ax.set_title("Label co-occurrence (log1p of image count)"); fig.colorbar(im, ax=ax, fraction=0.04); fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_cooccurrence.png"), dpi=140); plt.close(fig)

    tbl = ["| item | value |", "|---|---|", f"| training images / patients | {S['n_train']} / {S['n_train_patients']} |", f"| test images / patients (labels hidden) | {S['n_test']} / {S['n_test_patients']} |",
           f"| train-test patient overlap | {S['patient_overlap_train_test']} |", f"| training patients with >1 image | {S['train_patients_with_gt1_image']} ({100 * S['train_patients_with_gt1_image'] / S['n_train_patients']:.0f} %) |",
           f"| labels per image | {S['labels_per_image']} (single-label {100 * S['single_label_fraction']:.1f} %) |", f"| image size | {S['image_size_counts']} |",
           f"| photometric interpretation | {S['photometric_counts']} |", f"| BitsStored | {S['bits_stored_counts']} |",
           f"| DICOM BodyPartExamined non-empty | {S['body_part_examined_nonempty']} of {len(meta)} |", f"| class imbalance (max/min) | {S['imbalance_ratio_max_over_min']:.0f}x |"]
    if "holdout" in S:
        tbl += [f"| held-out set | {S['holdout']['images']} images / {S['holdout']['patients']} patients |", f"| CV pool | {S['cv_pool']['images']} images / {S['cv_pool']['patients']} patients; per fold {S['cv_pool']['images_per_fold']} |",
                f"| patient overlap held-out vs pool | {S['holdout_pool_patient_overlap']} |"]
    open(os.path.join(a.out, "dataset_table.md"), "w").write("\n".join(tbl) + "\n")
    print("\n".join(tbl)); print(ct.to_string(index=False)); print("top co-occurring pairs:", S["top_cooccurring_pairs"][:5])


if __name__ == "__main__":
    main()
