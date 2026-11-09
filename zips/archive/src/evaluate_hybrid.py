"""
Step 3 evaluation: recompute metrics for the two hybrid-model checkpoints
(plain vs hierarchical loss) and print them next to the Step 2 baseline
leaderboard-comparable numbers, so the ablation and the "did Step 3 beat
Step 2" question are both answered in one place.

Kaggle usage (after both hybrid variants are trained):
    python -m src.evaluate_hybrid --fold 0
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
from .engine import validate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=24)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index_df = pd.read_csv(C.INDEX_CSV)
    folds_df = pd.read_csv(C.FOLDS_CSV)
    train_index = index_df[index_df["split"] == "train"].merge(folds_df[["uid", "fold"]], on="uid", how="inner")
    val_df = train_index[train_index["fold"] == args.fold].reset_index(drop=True)

    print("indexing DICOM files under", C.DATA_ROOT, "...")
    uid_to_path = build_uid_to_path(C.DATA_ROOT)
    val_ds = UnifespDataset(val_df, C.DATA_ROOT, transform=get_transforms(HYBRID_INPUT_SIZE, train=False),
                             uid_to_path=uid_to_path)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    loss_fn = build_loss(None, device=device)

    rows = []
    for variant in ["plain", "hierarchical"]:
        run_dir = os.path.join(C.OUTPUT_DIR, f"hybrid_{variant}_fold{args.fold}")
        ckpt_path = os.path.join(run_dir, "checkpoint_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"[skip] hybrid_{variant}: no checkpoint_best.pt in {run_dir}")
            continue

        model = HybridCNNTransformer(pretrained=False).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

        metrics, _, _ = validate(model, val_loader, loss_fn, device)
        metrics["model"] = f"hybrid_{variant}"
        metrics["checkpoint_epoch"] = ckpt["epoch"]
        rows.append(metrics)
        print(f"[hybrid_{variant}] sample_mean_f1={metrics['sample_mean_f1']:.4f}  "
              f"macro_f1={metrics['macro_f1']:.4f}  (epoch {ckpt['epoch']})")
        del model

    df = pd.DataFrame(rows)
    out_path = os.path.join(C.OUTPUT_DIR, f"hybrid_results_fold{args.fold}.csv")
    df.to_csv(out_path, index=False)
    print("\nSaved:", out_path)

    baseline_path = os.path.join(C.OUTPUT_DIR, f"leaderboard_comparable_fold{args.fold}.csv")
    if os.path.exists(baseline_path) and len(df):
        base_df = pd.read_csv(baseline_path).rename(columns={"backbone": "model"})
        combined = pd.concat([base_df[["model", "sample_mean_f1", "macro_f1"]],
                               df[["model", "sample_mean_f1", "macro_f1"]]], ignore_index=True)
        combined = combined.sort_values("sample_mean_f1", ascending=False)
        print("\n=== Step 2 baselines + Step 3 hybrid, side by side ===")
        print(combined.to_string(index=False))
        combined.to_csv(os.path.join(C.OUTPUT_DIR, f"step2_vs_step3_fold{args.fold}.csv"), index=False)
    else:
        print(f"\n(Run src.evaluate_checkpoints first to get {baseline_path} for a side-by-side comparison.)")


if __name__ == "__main__":
    main()
