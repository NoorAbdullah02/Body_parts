"""
Step 2 final comparison table: merges architecture stats
(backbone_architecture_stats.csv, from compare_backbones.py) with each
backbone's trained performance (history.json, from train.py) into one table:
backbone | params_M | flops_G | inference_ms | avg_epoch_train_s |
best_macro_f1 | best_macro_auc | best_kappa | epochs_to_best

Run after training every backbone on at least one fold:
    python -m src.aggregate_results --fold 0
"""
import argparse
import json
import os

import pandas as pd

from . import config as C
from .models import BASELINE_BACKBONES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0)
    args = p.parse_args()

    arch_path = os.path.join(C.OUTPUT_DIR, "backbone_architecture_stats.csv")
    arch_df = pd.read_csv(arch_path) if os.path.exists(arch_path) else None
    if arch_df is None:
        print(f"[warn] {arch_path} not found -- run compare_backbones.py first for params/FLOPs columns")

    rows = []
    for name in BASELINE_BACKBONES:
        run_dir = os.path.join(C.OUTPUT_DIR, f"{name}_fold{args.fold}")
        history_path = os.path.join(run_dir, "history.json")
        if not os.path.exists(history_path):
            print(f"[skip] {name}: no history.json in {run_dir} (not trained yet)")
            continue
        with open(history_path) as f:
            history = json.load(f)
        if not history:
            continue
        best_row = max(history, key=lambda r: r["macro_f1"])
        avg_epoch_s = sum(r["epoch_seconds"] for r in history) / len(history)
        rows.append({
            "backbone": name,
            "epochs_trained": len(history),
            "epochs_to_best": best_row["epoch"] + 1,
            "avg_epoch_train_s": avg_epoch_s,
            "best_macro_f1": best_row["macro_f1"],
            "best_micro_f1": best_row["micro_f1"],
            "best_macro_auc": best_row["macro_auc"],
            "best_kappa_flat": best_row["kappa_flat"],
            "best_subset_accuracy": best_row["subset_accuracy"],
        })

    perf_df = pd.DataFrame(rows)
    if arch_df is not None and len(perf_df):
        final_df = arch_df.merge(perf_df, on="backbone", how="right")
    else:
        final_df = perf_df

    out_path = os.path.join(C.OUTPUT_DIR, f"comparison_table_fold{args.fold}.csv")
    final_df.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print(final_df.to_string(index=False))


if __name__ == "__main__":
    main()
