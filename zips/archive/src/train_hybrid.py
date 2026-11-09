"""
Step 3 entry point: train the Hybrid CNN-Transformer, with a --loss flag to
run the plain-BCE vs hierarchical-loss ablation (same engine/checkpointing
as Step 2's train.py, so 12h-session auto-resume works the same way).

Kaggle usage (run both variants, one per GPU, for the ablation):
    CUDA_VISIBLE_DEVICES=0 python -m src.train_hybrid --loss plain --epochs 30 --max_hours 5
    CUDA_VISIBLE_DEVICES=1 python -m src.train_hybrid --loss hierarchical --epochs 30 --max_hours 5

Local smoke test (CPU, tiny subset):
    python -m src.train_hybrid --loss hierarchical --epochs 1 --max_hours 0.01 --smoke_test
"""
import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config as C
from .dataset import UnifespDataset, get_transforms, build_uid_to_path
from .models_hybrid import HybridCNNTransformer, HYBRID_INPUT_SIZE
from .losses import build_loss
from .losses_hierarchical import build_hierarchical_loss
from .engine import run_training
from .train import set_seed, build_dataframes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loss", choices=["plain", "hierarchical"], default="hierarchical")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=24)  # hybrid model is heavier than a single backbone
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--group_weight", type=float, default=0.3, help="hierarchical loss's group-level term weight")
    p.add_argument("--max_hours", type=float, default=11.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--smoke_test", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df = build_dataframes(args.fold, smoke_test=args.smoke_test)
    print(f"train: {len(train_df)} images | val: {len(val_df)} images | loss: {args.loss}")

    print("indexing DICOM files under", C.DATA_ROOT, "...")
    uid_to_path = build_uid_to_path(C.DATA_ROOT)
    print(f"found {len(uid_to_path)} .dcm files")

    train_ds = UnifespDataset(train_df, C.DATA_ROOT, transform=get_transforms(HYBRID_INPUT_SIZE, train=True),
                               uid_to_path=uid_to_path)
    val_ds = UnifespDataset(val_df, C.DATA_ROOT, transform=get_transforms(HYBRID_INPUT_SIZE, train=False),
                             uid_to_path=uid_to_path)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = HybridCNNTransformer(pretrained=not args.smoke_test).to(device)

    label_cols = [f"label_{k}_{C.LABEL_NAMES[k].replace(' ', '_')}" for k in range(C.N_CLASSES)]
    label_matrix = train_df[label_cols].to_numpy(dtype="float32")
    if args.loss == "hierarchical":
        loss_fn = build_hierarchical_loss(label_matrix, group_weight=args.group_weight, device=device)
    else:
        loss_fn = build_loss(label_matrix, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = f"hybrid_{args.loss}_fold{args.fold}"
    run_dir = os.path.join(C.OUTPUT_DIR, run_name)

    history, best_metric = run_training(
        model, train_loader, val_loader, optimizer, scheduler, loss_fn,
        device, run_dir, n_epochs=args.epochs, max_hours=args.max_hours,
    )
    print(f"done. best macro_f1={best_metric:.4f}. checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
