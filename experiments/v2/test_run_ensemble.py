"""End-to-end test of src/run_ensemble.py on synthetic fold outputs (numpy/pandas/sklearn only).
Run from experiments/v2:   python test_run_ensemble.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))          # evalkit + test_evalkit.make_data
shutil.copy(os.path.join(HERE, "..", "evalkit.py"), os.path.join(HERE, "src", "evalkit.py"))
sys.path.insert(0, HERE)
from test_evalkit import make_data           # noqa: E402
from src import run_ensemble as RE           # noqa: E402

NAMES = [f"label_{k}_c{k}" for k in range(22)]


SHARED = None   # per-image noise shared by ALL models/folds (real fold models make correlated mistakes)


def synth_probs(y, rng, sigma, rows):
    mu = np.where(y == 1, 1.4, -3.0)
    return (1 / (1 + np.exp(-(mu + 1.3 * SHARED[rows] + 0.5 * sigma * rng.normal(0, 1, y.shape))))).astype(np.float32)


def softmax_like(y, rng, sigma, rows):
    z = np.where(y == 1, 1.8, 0.0) + 1.3 * SHARED[rows] + 0.6 * sigma * rng.normal(0, 1, y.shape)
    e = np.exp(z - z.max(1, keepdims=True))
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def build(tmp):
    global SHARED
    rng = np.random.default_rng(0)
    n_pool, n_hold, n_test = 600, 200, 120
    y_all, _, groups = make_data(n_pool + n_hold + n_test, seed=5)
    SHARED = np.random.default_rng(99).normal(size=y_all.shape)
    uids = [f"uid{i:05d}" for i in range(len(y_all))]
    split = ["train"] * (n_pool + n_hold) + ["test"] * n_test
    df = pd.DataFrame({"uid": uids, "split": split, "patient_id": [f"p{g}" for g in groups]})
    Y = y_all.astype(float)
    Y[n_pool + n_hold:] = np.nan                      # test labels are hidden, as on Kaggle
    for k, nm in enumerate(NAMES):
        df[nm] = Y[:, k]
    df.to_csv(os.path.join(tmp, "index.csv"), index=False)
    fold = rng.integers(0, 5, size=n_pool)
    pd.DataFrame({"uid": uids[:n_pool], "cv_fold": fold}).to_csv(os.path.join(tmp, "cv_folds.csv"), index=False)
    pd.DataFrame({"uid": uids[n_pool:n_pool + n_hold]}).to_csv(os.path.join(tmp, "holdout.csv"), index=False)
    pd.DataFrame({"SOPInstanceUID": uids[n_pool + n_hold:], "Target": "0 1"}).to_csv(os.path.join(tmp, "sample.csv"), index=False)
    out = os.path.join(tmp, "runs")
    for run, sg in (("strong_s224", 1.5), ("weak_s256", 2.8), ("mid_s224", 2.0)):
        for f in range(5):
            m = fold == f
            idx = np.where(m)[0]
            d = {}
            d["oof_uid"] = np.array(uids[:n_pool])[idx]
            h = slice(n_pool, n_pool + n_hold); t = slice(n_pool + n_hold, None)
            for tag, rows, uu in (("oof", idx, d["oof_uid"]), ("hold", np.arange(n_pool, n_pool + n_hold), np.array(uids[h])),
                                  ("test", np.arange(n_pool + n_hold, len(y_all)), np.array(uids[t]))):
                yy = y_all[rows]
                d[f"{tag}_uid"] = uu
                s = synth_probs(yy, rng, sg, rows); f_ = softmax_like(yy, rng, sg, rows)
                d[f"{tag}_sig"], d[f"{tag}_soft"] = s, f_
                d[f"{tag}_sig_tta"] = (s + synth_probs(yy, rng, sg, rows)) / 2
                d[f"{tag}_soft_tta"] = (f_ + softmax_like(yy, rng, sg, rows)) / 2
            os.makedirs(os.path.join(out, f"{run}_f{f}"))
            np.savez_compressed(os.path.join(out, f"{run}_f{f}", "preds.npz"), **d)
    return out


def main():
    tmp = tempfile.mkdtemp()
    try:
        out = build(tmp)
        class A: pass
        a = A()
        a.out_dir, a.runs, a.folds = out, "strong_s224,weak_s256,mid_s224", "0,1,2,3,4"
        a.index_csv, a.cv_folds_csv = os.path.join(tmp, "index.csv"), os.path.join(tmp, "cv_folds.csv")
        a.holdout_csv, a.sample_submission = os.path.join(tmp, "holdout.csv"), os.path.join(tmp, "sample.csv")
        a.result_dir, a.n_boot, a.caruana_iters, a.min_pos = os.path.join(tmp, "res"), 200, 15, 20.0
        res = RE.run(a)
        for f in ("results.json", "per_run.csv", "ensemble_arrays.npz", "submission.csv"):
            assert os.path.exists(os.path.join(a.result_dir, f)), f
        h = res["holdout"]
        print(json.dumps({"weights": res["weights"], "chosen": res["chosen_variant"]}, indent=1))
        print("holdout ens 0.5            :", round(h["ens_0.5"]["sample_mean_f1"], 4))
        print("holdout ens 0.5 + min_one  :", round(h["ens_0.5_min_one"]["sample_mean_f1"], 4))
        print("holdout ens tuned + min_one:", round(h["ens_tuned_min_one"]["sample_mean_f1"], 4),
              " CI", [round(h["ci_sample_mean_f1_tuned"][k], 4) for k in ("ci_low", "ci_high")])
        print("best single by OOF         :", h["best_single_by_oof"], round(h["best_single_metrics"]["sample_mean_f1"], 4))
        print("paired ens-single          :", {k: round(v, 4) for k, v in h["paired_ens_vs_single_sample_f1"].items() if k != 'n_boot'})
        assert res["weights"]["weak_s256"] <= max(res["weights"]["strong_s224"], res["weights"]["mid_s224"])
        assert h["ci_sample_mean_f1_tuned"]["ci_low"] <= h["ens_tuned_min_one"]["sample_mean_f1"] <= h["ci_sample_mean_f1_tuned"]["ci_high"]
        assert h["best_single_by_oof"] in ("strong_s224", "mid_s224")
        assert res["submission"]["check"]["rows"] == 120 and res["submission"]["check"]["mean_labels_per_image"] >= 1
        # decisions came from OOF: thresholds fixed, holdout never used to choose
        assert len(res["thresholds"]) == 22
        print("\nrun_ensemble test passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
