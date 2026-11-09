"""
reliability -- CPU-only analysis of the saved per-fold predictions (gaps G8, G9, G12):

    python -m src.reliability --out_dir runs --runs swin_tiny_s224,densenet121_s256,convnext_tiny_s256 \
        --index_csv index.csv --cv_folds_csv splits/cv_folds.csv --holdout_csv splits/holdout_uids.csv --result_dir results_reliability

1. Calibration of the top-1 confidence (max probability): reliability bins, ECE before / after temperature scaling
   (temperature fitted on OOF only, evaluated on the held-out set).
2. Selective prediction ("abstain and send to a human"): risk-coverage on the held-out set; the confidence threshold for a target accuracy
   is chosen on OOF only. Top-1 correct = the arg-max class is one of the image's true labels.
3. Per-class precision / recall / F1 on the held-out set with patient-level bootstrap CIs and the number of positives.
4. Label-error audit of the CV pool from OUT-OF-FOLD predictions (confident-learning style): images where the ensemble confidently predicts a class that is
   NOT in the given label set. These are SUSPECTS for expert review, not proven errors.

Decisions (run variant, ensemble weights, temperature, thresholds) use OOF only; the held-out set is used once.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from . import evalkit as E
from .run_ensemble import _load_run, _variants

NAMES = ["Abdomen", "Ankle", "Cervical Spine", "Chest", "Clavicles", "Elbow", "Feet", "Finger", "Forearm", "Hand", "Hip", "Knee", "Lower Leg",
         "Lumbar Spine", "Others", "Pelvis", "Shoulder", "Sinus", "Skull", "Thigh", "Thoracic Spine", "Wrist"]


def top1(prob, y):
    """confidence = max probability; correct = arg-max class is a true label of the image."""
    a = prob.argmax(1)
    return prob.max(1), y[np.arange(len(y)), a] == 1


def ece(conf, correct, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    rows, tot = [], 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum():
            rows.append({"bin": f"({lo:.1f},{hi:.1f}]", "n": int(m.sum()), "mean_confidence": float(conf[m].mean()), "accuracy": float(correct[m].mean())})
            tot += m.sum() / len(conf) * abs(conf[m].mean() - correct[m].mean())
    return float(tot), rows


def fit_temperature(prob, y):
    """One scalar T on the logit of the probabilities (multi-label: sigmoid scaling), minimising the binary NLL on OOF."""
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    best, bt = 1e18, 1.0
    for t in np.linspace(0.5, 3.0, 51):
        q = np.clip(1 / (1 + np.exp(-z / t)), 1e-6, 1 - 1e-6)
        nll = -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()
        if nll < best:
            best, bt = nll, float(t)
    return bt


def apply_temperature(prob, t):
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-np.log(p / (1 - p)) / t))


def class_prf(y, pred, c):
    tp = int(((y[:, c] == 1) & (pred[:, c] == 1)).sum()); fp = int(((y[:, c] == 0) & (pred[:, c] == 1)).sum()); fn = int(((y[:, c] == 1) & (pred[:, c] == 0)).sum())
    p = tp / (tp + fp) if tp + fp else float("nan"); r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
    return p, r, f


def per_class_table(y, pred, groups, n_boot=1000, seed=0):
    _, members = E._cluster_indices(groups)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        boots.append([class_prf(y[idx], pred[idx], c)[1:] for c in range(y.shape[1])])     # recall, f1
    boots = np.array(boots)                                                                  # (B, C, 2)
    rows = []
    for c in range(y.shape[1]):
        p, r, f = class_prf(y, pred, c)
        lo, hi = np.nanquantile(boots[:, c, 0], [0.025, 0.975]); flo, fhi = np.nanquantile(boots[:, c, 1], [0.025, 0.975])
        rows.append({"class": NAMES[c] if y.shape[1] == 22 else str(c), "n_pos": int(y[:, c].sum()), "precision": p, "recall": r, "recall_ci_low": lo, "recall_ci_high": hi,
                     "f1": f, "f1_ci_low": flo, "f1_ci_high": fhi})
    return pd.DataFrame(rows)


def selective(conf, correct, thr_conf):
    keep = conf >= thr_conf
    return float(keep.mean()), float(correct[keep].mean()) if keep.any() else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--index_csv", required=True)
    p.add_argument("--cv_folds_csv", required=True)
    p.add_argument("--holdout_csv", required=True)
    p.add_argument("--result_dir", default="results_reliability")
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--audit_conf", type=float, default=0.9, help="flag a pool image if a class NOT in its labels has OOF probability >= this")
    p.add_argument("--audit_top", type=int, default=60)
    a = p.parse_args()
    os.makedirs(a.result_dir, exist_ok=True)

    index = pd.read_csv(a.index_csv); label_cols = [c for c in index.columns if c.startswith("label_")]
    pool = pd.read_csv(a.cv_folds_csv); pool_uids = pool["uid"].tolist(); lab = index.set_index("uid")
    y_pool = lab.loc[pool_uids, label_cols].to_numpy(np.int64)
    folds = [int(x) for x in a.folds.split(",")]; runs = [r for r in a.runs.split(",") if r]

    data = {}
    for r in runs:
        d = _load_run(a.out_dir, r, folds, pool_uids, True, False, len(label_cols))
        vs = _variants(d["oof"]); sc = {k: E.sample_mean_f1(y_pool, E.predict_labels(v, 0.5, True)) for k, v in vs.items()}
        best = max(sc, key=sc.get)                                                     # OOF-only decision
        data[r] = {"oof": vs[best], "hold": _variants(d["hold"])[best], "hold_uid": d["hold_uid"], "variant": best}
    w = E.caruana_ensemble([data[r]["oof"] for r in runs], y_pool, n_iter=40)
    oof = E.average_probs([data[r]["oof"] for r in runs], w); hold = E.average_probs([data[r]["hold"] for r in runs], w)
    hold_uid = data[runs[0]]["hold_uid"].tolist()
    y_hold = lab.loc[hold_uid, label_cols].to_numpy(np.int64); groups = lab.loc[hold_uid, "patient_id"].astype(str).to_numpy()
    out = {"runs": runs, "variants": {r: data[r]["variant"] for r in runs}, "weights": dict(zip(runs, map(float, w)))}

    # 1. calibration of the top-1 confidence ------------------------------------------------------------
    c_oof, k_oof = top1(oof, y_pool); c_h, k_h = top1(hold, y_hold)
    e_before, rows_before = ece(c_h, k_h)
    t = fit_temperature(oof, y_pool); oof_t, hold_t = apply_temperature(oof, t), apply_temperature(hold, t)
    c_ht, k_ht = top1(hold_t, y_hold); e_after, rows_after = ece(c_ht, k_ht)
    out["calibration"] = {"top1_accuracy_holdout": float(k_h.mean()), "ece_holdout_before": e_before, "temperature_fitted_on_oof": t, "ece_holdout_after": e_after,
                          "bins_before": rows_before, "bins_after": rows_after}

    # 2. selective prediction ----------------------------------------------------------------------------
    order = np.sort(c_h)[::-1]; rc = []
    _, members = E._cluster_indices(groups); rng = np.random.default_rng(0)
    for cov in (1.0, 0.99, 0.95, 0.9, 0.8, 0.7):
        thr = order[int(np.ceil(cov * len(order))) - 1]
        keep = c_h >= thr; accs = []
        for _ in range(a.n_boot):
            idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))]); kk = keep[idx]
            accs.append(k_h[idx][kk].mean() if kk.any() else np.nan)
        rc.append({"target_coverage": cov, "actual_coverage": float(keep.mean()), "top1_accuracy_kept": float(k_h[keep].mean()),
                   "acc_ci_low": float(np.nanquantile(accs, 0.025)), "acc_ci_high": float(np.nanquantile(accs, 0.975))})
    out["risk_coverage_holdout"] = rc
    honest = []                                                              # threshold chosen on OOF for a target accuracy, applied to the held-out set
    for target in (0.95, 0.98, 0.99):
        cand = [th for th in np.unique(c_oof) if k_oof[c_oof >= th].mean() >= target]
        th = min(cand) if cand else 1.1
        cov, acc = selective(c_h, k_h, th)
        honest.append({"target_accuracy_on_oof": target, "threshold": float(th), "holdout_coverage": cov, "holdout_accuracy_kept": acc,
                       "holdout_sent_to_human": float(1 - cov)})
    out["selective_threshold_from_oof"] = honest

    # 3. per-class table ------------------------------------------------------------------------------------
    pred_h = E.predict_labels(hold, 0.5, min_one=True)
    pc = per_class_table(y_hold, pred_h, groups, n_boot=a.n_boot)
    pc.round(4).to_csv(os.path.join(a.result_dir, "per_class_holdout.csv"), index=False)
    out["holdout_sample_f1_0.5_min_one"] = E.sample_mean_f1(y_hold, pred_h)

    # 4. label-error audit (OOF, whole CV pool) --------------------------------------------------------------
    n = len(pool_uids); rows = []
    for i in range(n):
        given = set(np.where(y_pool[i] == 1)[0]); alt = [c for c in np.argsort(-oof[i]) if c not in given]
        if not alt: continue
        c = alt[0]
        if given and oof[i, c] >= a.audit_conf and max(oof[i, g] for g in given) < 0.5:
            rows.append({"uid": pool_uids[i], "patient_id": str(lab.loc[pool_uids[i], "patient_id"]), "given": "+".join(NAMES[g] if len(label_cols) == 22 else str(g) for g in sorted(given)),
                         "model_says": NAMES[c] if len(label_cols) == 22 else str(c), "p_model_class": float(oof[i, c]), "p_given_max": float(max(oof[i, g] for g in given))})
    audit = pd.DataFrame(rows).sort_values("p_model_class", ascending=False)
    audit.head(a.audit_top).round(4).to_csv(os.path.join(a.result_dir, "label_audit_suspects.csv"), index=False)
    pair = audit.groupby(["given", "model_says"]).size().sort_values(ascending=False).head(15) if len(audit) else pd.Series(dtype=int)
    out["label_audit"] = {"n_pool": n, "n_suspects": int(len(audit)), "fraction": float(len(audit) / n), "threshold": a.audit_conf,
                          "top_pairs": {f"{k[0]} -> {k[1]}": int(v) for k, v in pair.items()},
                          "note": "SUSPECTS only. The model may be wrong; a clinician must review a sample before any claim about label errors."}
    json.dump(out, open(os.path.join(a.result_dir, "reliability.json"), "w"), indent=1, default=float)

    print(f"runs={runs} weights={[round(float(x), 3) for x in w]} | held-out sample-F1 (0.5, min-one) = {out['holdout_sample_f1_0.5_min_one']:.4f}")
    print(f"calibration: top-1 acc {k_h.mean():.4f} | ECE before {e_before:.4f} -> after T={t:.2f}: {e_after:.4f}")
    print(pd.DataFrame(rc).round(4).to_string(index=False)); print(pd.DataFrame(honest).round(4).to_string(index=False))
    print(pc.round(3).to_string(index=False))
    print(f"label-audit: {len(audit)} suspects of {n} ({100 * len(audit) / n:.1f} %) at p>={a.audit_conf}"); print(pair.to_string() if len(pair) else "none")


if __name__ == "__main__":
    main()
