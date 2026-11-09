"""
run_ensemble -- CPU-only stage that turns the per-fold ``preds.npz`` files written by
train_v2 into (a) honest numbers and (b) a Kaggle submission.

Honesty rules enforced here
---------------------------
* Every DECISION (which head / TTA variant per run, ensemble weights, per-class thresholds)
  is made on the out-of-fold (OOF) predictions of the CV pool ONLY.
* The held-out set is touched exactly once, at the end, to REPORT.
* Held-out and Kaggle-test probabilities of a run are the average of its 5 fold models
  (bagging), which is how the Kaggle submission is produced too.

Inputs:  <out_dir>/<run>_f<k>/preds.npz  for every run prefix and fold k.
Outputs: <result_dir>/results.json, per_run.csv, ensemble_arrays.npz, submission.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from . import evalkit as E

N = 22


def _load_run(out_dir, run, folds, pool_uids, with_hold, with_test, n_classes=N):
    pos = {u: i for i, u in enumerate(pool_uids)}
    keys = ["sig", "soft", "sig_tta", "soft_tta"]
    oof = {k: np.full((len(pool_uids), n_classes), np.nan, np.float32) for k in keys}
    hold = {k: [] for k in keys}
    test = {k: [] for k in keys}
    hold_uid = test_uid = None
    for f in folds:
        d = np.load(os.path.join(out_dir, f"{run}_f{f}", "preds.npz"), allow_pickle=False)
        idx = np.array([pos[u] for u in d["oof_uid"]])
        for k in keys:
            oof[k][idx] = d[f"oof_{k}"]
        if with_hold and "hold_uid" in d.files:
            hu = d["hold_uid"]
            hold_uid = hu if hold_uid is None else hold_uid
            assert (hu == hold_uid).all(), "holdout uid order differs between folds"
            for k in keys:
                hold[k].append(d[f"hold_{k}"])
        if with_test and "test_uid" in d.files:
            tu = d["test_uid"]
            test_uid = tu if test_uid is None else test_uid
            assert (tu == test_uid).all(), "test uid order differs between folds"
            for k in keys:
                test[k].append(d[f"test_{k}"])
    for k in keys:
        assert not np.isnan(oof[k]).any(), f"{run}: OOF not complete for {k} (folds={folds})"
    avg = lambda L: {k: np.mean(L[k], axis=0) for k in keys} if L[keys[0]] else None
    return {"oof": oof, "hold": avg(hold), "test": avg(test), "hold_uid": hold_uid, "test_uid": test_uid}


def _variants(p):
    """Candidate probability matrices for one run (all selected on OOF later)."""
    v = {"sig": p["sig"], "soft": p["soft"], "blend": E.combine_heads(p["sig"], p["soft"], 0.5),
         "sig_tta": p["sig_tta"], "soft_tta": p["soft_tta"],
         "blend_tta": E.combine_heads(p["sig_tta"], p["soft_tta"], 0.5)}
    return v


def run(args):
    os.makedirs(args.result_dir, exist_ok=True)
    index = pd.read_csv(args.index_csv)
    label_cols = [c for c in index.columns if c.startswith("label_")]
    n_classes = len(label_cols)             # 22 for UNIFESP, 7 MURA, 4 COVID, ...
    pool = pd.read_csv(args.cv_folds_csv)
    pool_uids = pool["uid"].tolist()
    lab = index.set_index("uid")
    y_pool = lab.loc[pool_uids, label_cols].to_numpy(np.int64)

    have_hold = bool(args.holdout_csv)
    hold_uids = pd.read_csv(args.holdout_csv)["uid"].tolist() if have_hold else []
    folds = [int(x) for x in args.folds.split(",")]
    runs = [r for r in args.runs.split(",") if r]

    data, chosen, rows = {}, {}, []
    for r in runs:
        d = _load_run(args.out_dir, r, folds, pool_uids, have_hold, True, n_classes)
        vs_oof = _variants(d["oof"])
        scores = {k: E.sample_mean_f1(y_pool, E.predict_labels(v, 0.5, min_one=True)) for k, v in vs_oof.items()}
        best = max(scores, key=scores.get)                     # decision on OOF only
        chosen[r] = best
        data[r] = {"oof": vs_oof[best],
                   "hold": _variants(d["hold"])[best] if d["hold"] is not None else None,
                   "test": _variants(d["test"])[best] if d["test"] is not None else None,
                   "hold_uid": d["hold_uid"], "test_uid": d["test_uid"]}
        rows.append({"run": r, "chosen_variant": best, **{f"oof_{k}": s for k, s in scores.items()}})

    # ---- ensemble weights on OOF only
    oof_list = [data[r]["oof"] for r in runs]
    w = E.caruana_ensemble(oof_list, y_pool, n_iter=args.caruana_iters)
    ens_oof = E.average_probs(oof_list, w)
    thr = E.tune_thresholds_oof(ens_oof, y_pool, min_pos=args.min_pos)

    result = {"runs": runs, "chosen_variant": chosen, "weights": dict(zip(runs, map(float, w))),
              "thresholds": [float(t) for t in thr],
              "oof": {"ens_0.5_min_one": E.all_metrics(y_pool, E.predict_labels(ens_oof, 0.5, True)),
                      "note": "OOF numbers are for model selection only; report the held-out numbers"}}

    # ---- held-out set: used once, to report
    if have_hold:
        hold_uid_order = data[runs[0]]["hold_uid"].tolist()
        y_hold = lab.loc[hold_uid_order, label_cols].to_numpy(np.int64)
        groups = lab.loc[hold_uid_order, "patient_id"].astype(str).to_numpy()
        hold_list = [data[r]["hold"] for r in runs]
        ens_hold = E.average_probs(hold_list, w)
        pred_05 = E.predict_labels(ens_hold, 0.5, min_one=False)
        pred_05m = E.predict_labels(ens_hold, 0.5, min_one=True)
        pred_tuned = E.predict_labels(ens_hold, thr, min_one=True)
        result["holdout"] = {"n_images": len(y_hold), "n_patients": int(len(set(groups))),
                             "ens_0.5": E.all_metrics(y_hold, pred_05),
                             "ens_0.5_min_one": E.all_metrics(y_hold, pred_05m),
                             "ens_tuned_min_one": E.all_metrics(y_hold, pred_tuned)}
        result["holdout"]["ci_sample_mean_f1_tuned"] = E.cluster_bootstrap(
            E.sample_mean_f1, y_hold, pred_tuned, groups, n_boot=args.n_boot)
        result["holdout"]["ci_macro_f1_tuned"] = E.cluster_bootstrap(
            E.macro_f1, y_hold, pred_tuned, groups, n_boot=args.n_boot)
        # best single run (chosen on OOF!) vs ensemble, paired on the same patient resamples
        single_oof = {r: E.sample_mean_f1(y_pool, E.predict_labels(data[r]["oof"], 0.5, True)) for r in runs}
        best_single = max(single_oof, key=single_oof.get)
        pred_single = E.predict_labels(data[best_single]["hold"], 0.5, min_one=True)
        result["holdout"]["best_single_by_oof"] = best_single
        result["holdout"]["best_single_metrics"] = E.all_metrics(y_hold, pred_single)
        result["holdout"]["paired_ens_vs_single_sample_f1"] = E.paired_bootstrap(
            E.sample_mean_f1, y_hold, pred_tuned, pred_single, groups, n_boot=args.n_boot)
        for row in rows:
            r = row["run"]
            row["holdout_sample_mean_f1_0.5_min_one"] = E.sample_mean_f1(
                y_hold, E.predict_labels(data[r]["hold"], 0.5, min_one=True))

    # ---- Kaggle test submission (UNIFESP only: other datasets have no hidden test set)
    ens_test = None
    if data[runs[0]]["test"] is not None:
        test_uid_order = data[runs[0]]["test_uid"].tolist()
        ens_test = E.average_probs([data[r]["test"] for r in runs], w)
        sub_path = os.path.join(args.result_dir, "submission.csv")
        pred = E.write_submission(test_uid_order, ens_test, thr, sub_path, min_one=True)
        result["submission"] = {"path": sub_path, "mean_labels": float(pred.sum(1).mean())}
        if args.sample_submission:
            result["submission"]["check"] = E.check_submission(sub_path, args.sample_submission, n_classes)

    pd.DataFrame(rows).to_csv(os.path.join(args.result_dir, "per_run.csv"), index=False)
    json.dump(result, open(os.path.join(args.result_dir, "results.json"), "w"), indent=1, default=float)
    np.savez_compressed(os.path.join(args.result_dir, "ensemble_arrays.npz"), oof=ens_oof,
                        hold=ens_hold if have_hold else np.zeros(0),
                        test=ens_test if ens_test is not None else np.zeros(0), thresholds=thr, weights=w,
                        hold_uid=np.array(data[runs[0]]["hold_uid"]) if have_hold else np.zeros(0),
                        test_uid=np.array(data[runs[0]]["test_uid"]) if ens_test is not None else np.zeros(0),
                        pool_uid=np.array(pool_uids))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True, help="folder that contains <run>_f<k>/preds.npz")
    p.add_argument("--runs", required=True, help="comma-separated run prefixes, e.g. swin_tiny_s224,densenet121_s256")
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--index_csv", required=True)
    p.add_argument("--cv_folds_csv", required=True)
    p.add_argument("--holdout_csv", default=None)
    p.add_argument("--sample_submission", default=None)
    p.add_argument("--result_dir", required=True)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--caruana_iters", type=int, default=40)
    p.add_argument("--min_pos", type=float, default=20.0)
    res = run(p.parse_args())
    print(json.dumps({k: res[k] for k in ("chosen_variant", "weights", "holdout") if k in res}, indent=1, default=float))


if __name__ == "__main__":
    main()
