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
