"""
UNIFESP DICOM dataset for multilabel body-part classification.

Handles the two DICOM quirks we found in EDA:
  - MONOCHROME1 vs MONOCHROME2 photometric interpretation (2367 vs 114 files)
    -> MONOCHROME1 is inverted, so we flip it to a consistent polarity.
  - Variable BitsStored (8/10/12/15-bit) -> min-max normalize per image instead
    of assuming a fixed bit depth.
"""
import os
import numpy as np
import pydicom
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import config as C


def load_cached_or_decode(uid, src_path, cache_dir):
    """Read uid's pre-cached PNG if cache_dir has one; otherwise decode the
    source image (DICOM for unifesp, PNG passthrough for chestxray14 -- see
    load_image_for_dataset) and, if cache_dir is set, write the cache PNG
    for next time. Write is atomic (temp file + os.replace) so two training
    processes caching the same image concurrently (e.g. two backbones on two
    GPUs both hitting fold 0's validation set) never leaves a half-written
    file for the other to read -- worst case both just redundantly decode it
    once.
    """
    if cache_dir is not None:
        png_path = os.path.join(cache_dir, f"{uid}.png")
        if os.path.exists(png_path):
            return np.array(Image.open(png_path).convert("RGB"))

    image = load_image_for_dataset(src_path)

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        tmp_path = os.path.join(cache_dir, f".{uid}.{os.getpid()}.tmp.png")
        Image.fromarray(image).save(tmp_path)
        os.replace(tmp_path, os.path.join(cache_dir, f"{uid}.png"))

    return image


def load_image_for_dataset(path):
    """Dispatch on C.DATASET: DICOM for unifesp, plain PNG for chestxray14/mura/covidcxr."""
    if C.DATASET in ("chestxray14", "mura", "covidcxr"):
        return load_png_as_image(path)
    return load_dicom_as_image(path)


def load_png_as_image(path):
    arr = np.array(Image.open(path).convert("L"))  # ChestX-ray14 PNGs are grayscale (some stored as RGBA)
    arr = np.stack([arr, arr, arr], axis=-1)  # 1-channel X-ray -> pseudo-RGB for ImageNet backbones
    return arr.astype(np.uint8)


def load_dicom_as_image(path):
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        arr = arr.max() - arr

    lo, hi = arr.min(), arr.max()
    arr = (arr - lo) / (hi - lo + 1e-6)
    arr = np.stack([arr, arr, arr], axis=-1)  # 1-channel X-ray -> pseudo-RGB for ImageNet backbones
    return (arr * 255.0).astype(np.uint8)


def _basename_uid(path):
    b = os.path.basename(path)[:-4]  # strip .dcm
    if b.endswith("-c"):
        b = b[:-2]
    return b


def build_uid_to_path(data_root):
    """Walk data_root once and map uid -> absolute file path. We do this at
    runtime instead of trusting a stored relative path, because the raw
    image folders are nested inconsistently (a local zip-extract quirk gave
    us train/train/train/<id>/..., while the Kaggle-mounted dataset or a
    re-extraction elsewhere may only have train/<id>/...). Matching by the
    filename's UID instead of a fixed path makes this robust to wherever the
    raw files actually end up on disk."""
    if C.DATASET == "chestxray14":
        lookup = {}
        for root, _dirs, files in os.walk(data_root):
            for f in files:
                # skip macOS AppleDouble metadata files (e.g. "._image1.png") --
                # they match the .png extension but aren't real images, and
                # crash PIL with UnidentifiedImageError (seen on the Kaggle
                # mirror of this dataset, which was zipped from a Mac)
                if f.lower().endswith(".png") and not f.startswith("._"):
                    lookup[f[:-4]] = os.path.join(root, f)
        return lookup

    if C.DATASET in ("mura", "covidcxr"):
        # MURA's own filenames (image1.png, image2.png, ...) repeat in every
        # study folder, so the bare filename can't be a uid -- use the full
        # path relative to data_root instead (unique, and keeps the
        # body-part/class-folder info parseable by build_index_mura.py /
        # build_index_covidcxr.py). covidcxr also has a parallel "masks/"
        # tree of lung-segmentation masks with the same filenames as
        # "images/" -- skip it here so masks never get treated as data.
        lookup = {}
        for root, _dirs, files in os.walk(data_root):
            if C.DATASET == "covidcxr" and os.path.basename(root).lower() == "masks":
                continue
            for f in files:
                # skip macOS AppleDouble metadata files (e.g. "._image1.png") --
                # they match the .png extension but aren't real images, and
                # crash PIL with UnidentifiedImageError (seen on the Kaggle
                # mirror of MURA, which was zipped from a Mac)
                if f.lower().endswith(".png") and not f.startswith("._"):
                    full_path = os.path.join(root, f)
                    rel = os.path.relpath(full_path, data_root)
                    uid = rel.replace(os.sep, "__")[:-4]
                    lookup[uid] = full_path
        return lookup

    lookup = {}
    for root, _dirs, files in os.walk(data_root):
        for f in files:
            if f.lower().endswith(".dcm") and not f.startswith("._"):
                lookup[_basename_uid(f)] = os.path.join(root, f)
    return lookup


class UnifespDataset(Dataset):
    def __init__(self, df, data_root, transform=None, has_labels=True, uid_to_path=None, cache_dir=C.CACHE_DIR):
        """
        df: a slice of index.csv (optionally merged with folds.csv), must have a 'uid' column
        data_root: base dir containing the (possibly nested) train/test DICOM folders
        uid_to_path: pass a precomputed dict to avoid re-walking data_root for every split
        cache_dir: optional PNG cache (see load_cached_or_decode); None disables caching
        """
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform
        self.has_labels = has_labels
        self.label_cols = [f"label_{k}_{C.LABEL_NAMES[k].replace(' ', '_')}" for k in range(C.N_CLASSES)]
        self.uid_to_path = uid_to_path if uid_to_path is not None else build_uid_to_path(data_root)
        self.cache_dir = cache_dir

        missing = set(self.df["uid"]) - set(self.uid_to_path)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} uid(s) in this split have no matching .dcm file under {data_root} "
                f"(e.g. {next(iter(missing))}). Check UNIFESP_DATA_ROOT points at the right dataset."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = row["uid"]
        path = self.uid_to_path[uid]
        image = load_cached_or_decode(uid, path, self.cache_dir)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        if self.has_labels:
            target = row[self.label_cols].to_numpy(dtype=np.float32)
            target = torch.from_numpy(target)
            return image, target
        return image, row["uid"]


def get_transforms(img_size=C.IMG_SIZE, train=True):
    """Train-time augmentation.

    We tried matching the UNIFESP Kaggle competition's top finishers
    (2nd place writeup used GaussNoise + GaussianBlur + ShiftScaleRotate +
    brightness/contrast) and measured it head-to-head on our own fold 0:

        backbone          light-aug  heavy-aug(+noise/blur)
        efficientnetv2_s     0.876      0.851   (worse)
        efficientnet_b3      0.866      0.847   (worse)
        inception_v3         0.872      0.841   (worse)
        convnext_tiny        0.738      0.558   (much worse)
        densenet121          0.869      0.873   (~same)
        vgg16                0.827      0.836   (~same)

    GaussNoise/GaussianBlur made 6/8 backbones worse (macro_f1), ConvNeXt-
    Tiny catastrophically so -- likely because blurring removes the fine
    bone-edge texture that separates visually similar classes (Lumbar vs
    Thoracic Spine, Hand vs Wrist), and our fixed 30-epoch budget (no
    warmup, no early stopping) doesn't give heavier augmentation enough
    iterations to pay off the way it did for the competition's longer,
    early-stopped fastai runs. So: geometric augmentation (flip, mild
    shift/scale/rotate) + brightness/contrast, no synthetic noise/blur.
    This negative result is itself worth keeping for the paper's ablation
    section.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
