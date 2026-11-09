"""
Central config. Two roots, because on Kaggle they'll usually be two
different attached datasets:
  - DATA_ROOT: where train/, test/ (the DICOM images) live
  - INDEX_ROOT: where index.csv and splits/folds.csv live

Locally both default to the repo root. On Kaggle, set them explicitly as
environment variables in the notebook's first cell, e.g.:
    import os
    os.environ["UNIFESP_DATA_ROOT"] = "/kaggle/input/unifesp-xray-bodypart-classification"
    os.environ["UNIFESP_INDEX_ROOT"] = "/kaggle/input/unifesp-index-files"  # your uploaded index.csv+splits/
    os.environ["UNIFESP_OUTPUT_DIR"] = "/kaggle/working"
(see KAGGLE_SETUP.md for the full first-cell snippet.)
"""
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)

ON_KAGGLE = os.path.exists("/kaggle/input")

if ON_KAGGLE:
    DATA_ROOT = os.environ.get("UNIFESP_DATA_ROOT", "/kaggle/input")
    INDEX_ROOT = os.environ.get("UNIFESP_INDEX_ROOT", DATA_ROOT)
    OUTPUT_DIR = os.environ.get("UNIFESP_OUTPUT_DIR", "/kaggle/working")
else:
    DATA_ROOT = os.environ.get("UNIFESP_DATA_ROOT", _REPO_ROOT)
    INDEX_ROOT = os.environ.get("UNIFESP_INDEX_ROOT", DATA_ROOT)
    OUTPUT_DIR = os.environ.get("UNIFESP_OUTPUT_DIR", os.path.join(_REPO_ROOT, "runs"))

INDEX_CSV = os.path.join(INDEX_ROOT, "index.csv")
FOLDS_CSV = os.path.join(INDEX_ROOT, "splits", "folds.csv")

# Optional: where cache_images.py pre-decodes DICOM -> PNG once, so every
# later training run reads a small PNG instead of re-decoding DICOM every
# epoch (top Kaggle finishers reported this cuts training from hours to
# minutes). None (default) means "no cache, decode DICOM every time" --
# fully backward compatible, nothing breaks if you skip this step.
CACHE_DIR = os.environ.get("UNIFESP_CACHE_DIR", os.path.join(OUTPUT_DIR, "image_cache") if ON_KAGGLE else None)

if os.environ.get("UNIFESP_CACHE_DIR", "").lower() in ("none", "off", "0"):
    CACHE_DIR = None        # no PNG cache (e.g. MURA / COVID: PNG decoding is cheap, a 40k-image cache would fill the disk)

# Which dataset this run targets -- "unifesp" (default), "chestxray14",
# "mura", or "covidcxr". Switching this is the ONLY config change needed to
# retarget the whole pipeline (train.py/train_hybrid.py/evaluate_on_holdout.py/
# etc. are all dataset-agnostic): it picks the label set below (-> last-layer
# neuron count, via build_model's/HybridCNNTransformer's default n_classes=
# C.N_CLASSES) and, in dataset.py, the image file format to load (.dcm vs
# .png). Still set UNIFESP_DATA_ROOT to that dataset's own attached path.
DATASET = os.environ.get("UNIFESP_DATASET", "unifesp")

if DATASET == "chestxray14":
    # NIH ChestX-ray14: 14 disease classes. "No Finding" is deliberately not
    # its own class -- it's the all-zero label vector (same multilabel
    # convention the CheXNet-style literature on this dataset uses), which
    # engine.py's sample_mean_f1/compute_metrics already handle correctly.
    LABEL_NAMES = {
        0: "Atelectasis", 1: "Cardiomegaly", 2: "Consolidation", 3: "Edema",
        4: "Effusion", 5: "Emphysema", 6: "Fibrosis", 7: "Hernia",
        8: "Infiltration", 9: "Mass", 10: "Nodule", 11: "Pleural_Thickening",
        12: "Pneumonia", 13: "Pneumothorax",
    }
elif DATASET == "mura":
    # Stanford MURA, re-tasked as BODY-PART classification (the body part is
    # already encoded in each image's folder name, XR_ELBOW/XR_HAND/...) --
    # the same task UNIFESP solves, on an independent patient population and
    # imaging source. Single-label (one body part per image) rather than
    # multilabel, but a one-hot row is just a degenerate multilabel row, so
    # nothing downstream needs to know the difference.
    LABEL_NAMES = {
        0: "Elbow", 1: "Finger", 2: "Forearm", 3: "Hand",
        4: "Humerus", 5: "Shoulder", 6: "Wrist",
    }
elif DATASET == "covidcxr":
    # COVID-19 Radiography Database (Qatar University + University of Dhaka),
    # re-tasked as 4-class chest X-ray diagnosis classification. Single-label
    # (one diagnosis per image), aggregated from many source datasets/
    # publications with no consistent patient-id field across them, so
    # build_index_covidcxr.py uses each image's own uid as its "patient_id"
    # -- an image-level split, not patient-level (a limitation worth naming
    # in the paper, same as why the other datasets' patient-level splitting
    # matters in the first place).
    LABEL_NAMES = {
        0: "COVID", 1: "Lung_Opacity", 2: "Normal", 3: "Viral_Pneumonia",
    }
else:
    LABEL_NAMES = {
        0: "Abdomen", 1: "Ankle", 2: "Cervical Spine", 3: "Chest", 4: "Clavicles",
        5: "Elbow", 6: "Feet", 7: "Finger", 8: "Forearm", 9: "Hand", 10: "Hip",
        11: "Knee", 12: "Lower Leg", 13: "Lumbar Spine", 14: "Others", 15: "Pelvis",
        16: "Shoulder", 17: "Sinus", 18: "Skull", 19: "Thigh", 20: "Thoracic Spine",
        21: "Wrist",
    }
N_CLASSES = len(LABEL_NAMES)
IMG_SIZE = 256
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
