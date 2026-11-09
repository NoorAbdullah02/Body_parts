"""
Optional one-time speed-up: pre-decode every DICOM into a PNG cache before
launching the 8 parallel training runs. UnifespDataset already caches
lazily on first read (see dataset.py's load_cached_or_decode), so this
script isn't required for correctness -- it just avoids every one of the
first 2 (parallel) backbones redundantly decoding the same ~1700 training
images during their first epoch. Top Kaggle finishers on this exact
competition reported DICOM decoding as their biggest bottleneck.

Kaggle usage, right after environment setup, before Step 2 training:
    python -m src.cache_images
"""
import time

from . import config as C
from .dataset import build_uid_to_path, load_cached_or_decode


def main():
    if C.CACHE_DIR is None:
        print("UNIFESP_CACHE_DIR / CACHE_DIR is not set -- nothing to do. "
              "Caching is off by default outside Kaggle; set the env var if you want it locally.")
        return

    print("Indexing DICOM files under", C.DATA_ROOT, "...")
    uid_to_path = build_uid_to_path(C.DATA_ROOT)
    print(f"Found {len(uid_to_path)} files. Caching to {C.CACHE_DIR} ...")

    t0 = time.time()
    for i, (uid, path) in enumerate(uid_to_path.items()):
        load_cached_or_decode(uid, path, C.CACHE_DIR)
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(uid_to_path)} ({time.time() - t0:.1f}s elapsed)")

    print(f"Done. Cached {len(uid_to_path)} images in {time.time() - t0:.1f}s -> {C.CACHE_DIR}")


if __name__ == "__main__":
    main()
