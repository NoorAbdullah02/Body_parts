"""Tests for the stage-2 pieces: ensemble WITHOUT a held-out set (all-data folds) and confident pseudo-label selection.
Run from experiments/v2:  python test_stage2.py"""
import os, shutil, sys, tempfile
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
shutil.copy(os.path.join(HERE, "..", "evalkit.py"), os.path.join(HERE, "src", "evalkit.py"))
import test_run_ensemble as T                     # noqa: E402
from src import run_ensemble as RE, make_pseudo_labels as PL, evalkit as E   # noqa: E402


def main():
    tmp = tempfile.mkdtemp()
    try:
        out = T.build(tmp)
        class A: pass
        a = A()
        a.out_dir, a.runs, a.folds = out, "strong_s224,mid_s224", "0,1,2,3,4"
        a.index_csv, a.cv_folds_csv = os.path.join(tmp, "index.csv"), os.path.join(tmp, "cv_folds.csv")
        a.holdout_csv = None                                            # stage-2: no held-out set
        a.sample_submission = os.path.join(tmp, "sample.csv")
        a.result_dir, a.n_boot, a.caruana_iters, a.min_pos = os.path.join(tmp, "res"), 50, 10, 20.0
        res = RE.run(a)
        assert "holdout" not in res and res["submission"]["check"]["rows"] == 120
        z = np.load(os.path.join(a.result_dir, "ensemble_arrays.npz"))
        assert z["test"].shape == (120, 22) and len(z["test_uid"]) == 120 and z["thresholds"].shape == (22,)
        # confident pseudo-labels
        idx, pred = PL.select_confident(z["test"], z["thresholds"], min_conf=0.9)
        loose, _ = PL.select_confident(z["test"], z["thresholds"], min_conf=0.6)
        assert len(loose) >= len(idx), "a lower confidence bar must keep at least as many images"
        capped, _ = PL.select_confident(z["test"], z["thresholds"], min_conf=0.6, max_images=10)
        assert len(capped) <= 10 and set(capped) <= set(loose)
        p = z["test"][idx]
        assert (pred[idx].sum(1) >= 1).all()
        assert np.where(pred[idx] == 1, p >= 0.9, p <= 0.1).all(), "kept images must be confident for every class"
        # CLI round trip + columns compatible with train_v2 --pseudo_csv
        ps = os.path.join(tmp, "pseudo.csv")
        sys.argv = ["x", "--ensemble_arrays", os.path.join(a.result_dir, "ensemble_arrays.npz"), "--index_csv", a.index_csv, "--out", ps, "--min_conf", "0.6"]
        PL.main()
        df = pd.read_csv(ps)
        labs = [c for c in df.columns if c.startswith("label_")]
        assert df.columns[0] == "uid" and len(labs) == 22 and len(df) == len(loose)
        print(f"no-holdout ensemble OK; pseudo-labels kept {len(idx)}/120 @0.9, {len(loose)}/120 @0.6")
        print("stage-2 tests passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()
