"""
Step 2 "architecture stats" half of the comparison table (params, FLOPs,
inference time). Run this first, it's cheap (no training). The "performance
+ training time" half comes from each backbone's history.json after
src/train.py has actually trained it -- see aggregate_results.py.

Kaggle usage:
    python -m src.compare_backbones
"""
import time
import torch

from . import config as C
from .models import BASELINE_BACKBONES, build_model, input_size_for

try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False


def measure_inference_time(model, img_size, device, n_runs=30):
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        for _ in range(5):  # warmup
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_runs):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / n_runs
    return dt * 1000  # ms/image


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name in BASELINE_BACKBONES:
        print(f"[{name}] building...")
        model = build_model(name, pretrained=False).to(device)
        img_size = input_size_for(name)
        n_params = sum(p.numel() for p in model.parameters())

        flops = None
        if HAS_THOP:
            try:
                dummy = torch.randn(1, 3, img_size, img_size, device=device)
                flops, _ = profile(model, inputs=(dummy,), verbose=False)
            except Exception as e:
                print(f"  thop failed for {name}: {e}")

        inf_ms = measure_inference_time(model, img_size, device)
        rows.append({
            "backbone": name,
            "img_size": img_size,
            "params_M": n_params / 1e6,
            "flops_G": (flops / 1e9) if flops else None,
            "inference_ms_per_image": inf_ms,
        })
        print(f"  params={n_params/1e6:.2f}M  flops={(flops/1e9) if flops else 'n/a'}G  "
              f"inference={inf_ms:.2f}ms")
        del model

    import pandas as pd
    df = pd.DataFrame(rows)
    out_path = f"{C.OUTPUT_DIR}/backbone_architecture_stats.csv"
    df.to_csv(out_path, index=False)
    print("\nSaved:", out_path)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
