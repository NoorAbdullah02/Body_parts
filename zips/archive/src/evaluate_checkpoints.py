"""
Recompute metrics (including sample_mean_f1, the actual UNIFESP Kaggle
competition leaderboard metric) on already-trained checkpoints, WITHOUT
retraining. Use this after a train.py run so you don't have to redo 30
epochs just because we added a new metric.

Kaggle usage (after your 8 backbones are already trained):
    python -m src.evaluate_checkpoints --fold 0
"""
import argparse
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config as C
from .dataset import UnifespDataset, get_transforms, build_uid_to_path
from .models import BASELINE_BACKBONES, build_model, input_size_for
from .losses import build_loss
from .engine import validate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index_df = pd.read_csv(C.INDEX_CSV)
    folds_df = pd.read_csv(C.FOLDS_CSV)
    train_index = index_df[index_df["split"] == "train"].merge(folds_df[["uid", "fold"]], on="uid", how="inner")
    val_df = train_index[train_index["fold"] == args.fold].reset_index(drop=True)

    print("indexing DICOM files under", C.DATA_ROOT, "...")
    uid_to_path = build_uid_to_path(C.DATA_ROOT)

    label_cols = [f"label_{k}_{C.LABEL_NAMES[k].replace(' ', '_')}" for k in range(C.N_CLASSES)]
    loss_fn = build_loss(None, device=device)  # pos_weight doesn't matter for eval-only loss value

    rows = []
    for name in BASELINE_BACKBONES:
        run_dir = os.path.join(C.OUTPUT_DIR, f"{name}_fold{args.fold}")
        ckpt_path = os.path.join(run_dir, "checkpoint_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"[skip] {name}: no checkpoint_best.pt in {run_dir}")
            continue

        img_size = input_size_for(name)
        val_ds = UnifespDataset(val_df, C.DATA_ROOT, transform=get_transforms(img_size, train=False),
                                 uid_to_path=uid_to_path)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = build_model(name, pretrained=False).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)  # our own checkpoint, trusted
        model.load_state_dict(ckpt["model_state"])

        metrics, _, _ = validate(model, val_loader, loss_fn, device)
        metrics["backbone"] = name
        metrics["checkpoint_epoch"] = ckpt["epoch"]
        rows.append(metrics)
        print(f"[{name}] sample_mean_f1={metrics['sample_mean_f1']:.4f}  "
              f"(macro_f1={metrics['macro_f1']:.4f}, epoch {ckpt['epoch']})")
        del model

    df = pd.DataFrame(rows).sort_values("sample_mean_f1", ascending=False)
    out_path = os.path.join(C.OUTPUT_DIR, f"leaderboard_comparable_fold{args.fold}.csv")
    df.to_csv(out_path, index=False)
    print("\nSaved:", out_path)
    print(df[["backbone", "sample_mean_f1", "macro_f1", "micro_f1", "subset_accuracy"]].to_string(index=False))
    print("\nFor reference, the real UNIFESP Kaggle competition's private leaderboard "
          "(same Mean-F1-Score metric, 115 teams) topped out at 0.93378 (1st place), "
          "with the top ~49 teams all above 0.898.")


if __name__ == "__main__":
    main()
