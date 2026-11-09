"""
Step 2 entry point: train one backbone on one fold.

Kaggle usage (12h session limit, dual GPU):
    python -m src.train --backbone efficientnet_b3 --fold 0 --epochs 30 --max_hours 11
If the session gets cut off or you stop it, just re-run the exact same
command later (new Kaggle session) -- it auto-resumes from checkpoint_last.pt
in OUTPUT_DIR/<run_name>/.

Local smoke test (CPU, tiny subset, no real training):
    python -m src.train --backbone resnet50 --fold 0 --epochs 1 --max_hours 0.01 --smoke_test
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config as C
from .dataset import UnifespDataset, get_transforms, build_uid_to_path
from .models import build_model, input_size_for
from .losses import build_loss
from .engine import run_training


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataframes(fold, smoke_test=False):
    index_df = pd.read_csv(C.INDEX_CSV)
    folds_df = pd.read_csv(C.FOLDS_CSV)

    train_index = index_df[index_df["split"] == "train"].merge(
        folds_df[["uid", "fold"]], on="uid", how="inner"
    )
    train_df = train_index[train_index["fold"] != fold].reset_index(drop=True)
    val_df = train_index[train_index["fold"] == fold].reset_index(drop=True)

    if smoke_test:
        train_df = train_df.iloc[:8].reset_index(drop=True)
        val_df = val_df.iloc[:4].reset_index(drop=True)
    return train_df, val_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="efficientnet_b3")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_hours", type=float, default=11.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--smoke_test", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"[info] {torch.cuda.device_count()} GPUs visible — this script uses GPU 0 only "
              f"by default; run separate backbones on separate GPUs in parallel for Step 2's "
              f"model comparison instead of DataParallel (simpler, and safer for the 12h budget).")

    train_df, val_df = build_dataframes(args.fold, smoke_test=args.smoke_test)
    print(f"train: {len(train_df)} images | val: {len(val_df)} images")

    print("indexing DICOM files under", C.DATA_ROOT, "...")
    uid_to_path = build_uid_to_path(C.DATA_ROOT)
    print(f"found {len(uid_to_path)} .dcm files")

    img_size = input_size_for(args.backbone)
    train_ds = UnifespDataset(train_df, C.DATA_ROOT, transform=get_transforms(img_size, train=True),
                               uid_to_path=uid_to_path)
    val_ds = UnifespDataset(val_df, C.DATA_ROOT, transform=get_transforms(img_size, train=False),
                             uid_to_path=uid_to_path)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.backbone, pretrained=not args.smoke_test).to(device)

    label_cols = [f"label_{k}_{C.LABEL_NAMES[k].replace(' ', '_')}" for k in range(C.N_CLASSES)]
    loss_fn = build_loss(train_df[label_cols].to_numpy(dtype="float32"), device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = f"{args.backbone}_fold{args.fold}"
    run_dir = os.path.join(C.OUTPUT_DIR, run_name)

    history, best_metric = run_training(
        model, train_loader, val_loader, optimizer, scheduler, loss_fn,
        device, run_dir, n_epochs=args.epochs, max_hours=args.max_hours,
    )
    print(f"done. best macro_f1={best_metric:.4f}. checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
