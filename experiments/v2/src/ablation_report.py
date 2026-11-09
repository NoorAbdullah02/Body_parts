"""
ablation_report -- turns several finished configurations (each = one or more run prefixes under --out_dir) into

  1. a CONFIG table: OOF F1, held-out F1 (+ patient-bootstrap CI), spread over the 5 fold models, and the paired
     difference to a reference configuration (CI + p-value)             -> augmentation / loss / backbone / fusion ablations
  2. a DECODING table for one configuration: no threshold, + min-one rule, global threshold, per-class thresholds tuned
     on OOF (honest) vs tuned on the held-out set itself (leaky) and the resulting "leak gap"
  3. a HEAD/TTA table: sigmoid / softmax / blend x with / without TTA (selected on OOF, reported on the held-out set)

All choices are made on OOF; the held-out set is only used to REPORT (except the deliberately leaky row, which is labelled).

configs.json example:
    {"light_aug": ["swin_tiny_s224"], "heavy_aug": ["swin_tiny_s224_heavy"], "fusion_gate": ["hybrid_gate_s224"]}
A config with several prefixes is an equal-weight probability average of those runs.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from . import evalkit as E
from .run_ensemble import _load_run, _variants


def _config_probs(out_dir, prefixes, folds, pool_uids, n_classes, y_pool):
    """OOF / holdout probabilities of a config: best head/TTA variant per run (chosen on OOF), equal-weight average."""
    oofs, holds, per_fold_hold = [], [], []
    chosen = {}
    hold_uid = None
    for p in prefixes:
        d = _load_run(out_dir, p, folds, pool_uids, True, False, n_classes)
        vs = _variants(d["oof"])
        sc = {k: E.sample_mean_f1(y_pool, E.predict_labels(v, 0.5, True)) for k, v in vs.items()}
        best = max(sc, key=sc.get)
        chosen[p] = best
        oofs.append(vs[best]); holds.append(_variants(d["hold"])[best]); hold_uid = d["hold_uid"]
        # individual fold models on the holdout (spread across folds)
        pf = []
        for f in folds:
            z = np.load(os.path.join(out_dir, f"{p}_f{f}", "preds.npz"))
            key = {"sig": "sig", "soft": "soft", "sig_tta": "sig_tta", "soft_tta": "soft_tta"}.get(best)
            h = z[f"hold_{key}"] if key else E.combine_heads(z["hold_sig" + ("_tta" if best.endswith("tta") else "")],
                                                             z["hold_soft" + ("_tta" if best.endswith("tta") else "")], 0.5)
            pf.append(h)
        per_fold_hold.append(pf)
    oof = E.average_probs(oofs)
    hold = E.average_probs(holds)
    fold_probs = [E.average_probs([per_fold_hold[r][i] for r in range(len(prefixes))]) for i in range(len(folds))]
    return {"oof": oof, "hold": hold, "fold_hold": fold_probs, "chosen": chosen, "hold_uid": hold_uid}


def _md(df):
    """Tiny markdown table writer (avoids the optional 'tabulate' dependency of DataFrame.to_markdown)."""
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r.tolist()) + " |")
    return "\n".join(out)


def _best_global_threshold(oof, y):
    grid = E.DEFAULT_GRID
    scores = [E.sample_mean_f1(y, E.predict_labels(oof, t, True)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def run(a):
    os.makedirs(a.result_dir, exist_ok=True)
    index = pd.read_csv(a.index_csv)
    label_cols = [c for c in index.columns if c.startswith("label_")]
    n = len(label_cols)
    pool_uids = pd.read_csv(a.cv_folds_csv)["uid"].tolist()
    lab = index.set_index("uid")
    y_pool = lab.loc[pool_uids, label_cols].to_numpy(np.int64)
    folds = [int(x) for x in a.folds.split(",")]
    configs = json.load(open(a.config_json))
    assert a.reference in configs, f"reference {a.reference} not in {list(configs)}"

    cache, y_hold, groups = {}, None, None
    for name, prefixes in configs.items():
        cache[name] = _config_probs(a.out_dir, prefixes, folds, pool_uids, n, y_pool)
        if y_hold is None:
            order = cache[name]["hold_uid"].tolist()
            y_hold = lab.loc[order, label_cols].to_numpy(np.int64)
            groups = lab.loc[order, "patient_id"].astype(str).to_numpy()

    ref_pred = E.predict_labels(cache[a.reference]["hold"], 0.5, True)
    rows = []
    for name, c in cache.items():
        pred = E.predict_labels(c["hold"], 0.5, True)
        ci = E.cluster_bootstrap(E.sample_mean_f1, y_hold, pred, groups, n_boot=a.n_boot)
        fold_f1 = [E.sample_mean_f1(y_hold, E.predict_labels(p, 0.5, True)) for p in c["fold_hold"]]
        row = {"config": name, "runs": "+".join(configs[name]),
               "oof_sample_f1": E.sample_mean_f1(y_pool, E.predict_labels(c["oof"], 0.5, True)),
               "hold_sample_f1": ci["estimate"], "hold_ci_low": ci["ci_low"], "hold_ci_high": ci["ci_high"],
               "hold_macro_f1": E.macro_f1(y_hold, pred),
               "oof_minus_hold": None,
               "fold_models_mean": float(np.mean(fold_f1)), "fold_models_std": float(np.std(fold_f1, ddof=1)) if len(fold_f1) > 1 else 0.0}
        row["oof_minus_hold"] = row["oof_sample_f1"] - row["hold_sample_f1"]      # large positive = validation over-optimistic
        if name != a.reference:
            d = E.paired_bootstrap(E.sample_mean_f1, y_hold, pred, ref_pred, groups, n_boot=a.n_boot)
            row.update({"diff_vs_ref": d["diff"], "diff_ci_low": d["ci_low"], "diff_ci_high": d["ci_high"], "p_value": d["p_value"]})
        rows.append(row)
    cfg_df = pd.DataFrame(rows)
    cfg_df.to_csv(os.path.join(a.result_dir, "ablation_configs.csv"), index=False)

    # ---- decoding ablation on the reference config
    ref = cache[a.reference]
    oof, hold = ref["oof"], ref["hold"]
    thr_oof = E.tune_thresholds_oof(oof, y_pool)
    thr_leaky = E.tune_thresholds_oof(hold, y_hold)                 # DELIBERATELY leaky: tuned on what it is scored on
    t_glob = _best_global_threshold(oof, y_pool)
    rules = {"0.5, empty allowed": E.predict_labels(hold, 0.5, False),
             "0.5 + min-one": E.predict_labels(hold, 0.5, True),
             f"global threshold {t_glob:.2f} (tuned on OOF) + min-one": E.predict_labels(hold, t_glob, True),
             "per-class thresholds tuned on OOF (honest) + min-one": E.predict_labels(hold, thr_oof, True),
             "per-class thresholds tuned on HELD-OUT (leaky) + min-one": E.predict_labels(hold, thr_leaky, True)}
    drows = []
    for k, pred in rules.items():
        ci = E.cluster_bootstrap(E.sample_mean_f1, y_hold, pred, groups, n_boot=a.n_boot)
        drows.append({"decoding": k, "hold_sample_f1": ci["estimate"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                      "macro_f1": E.macro_f1(y_hold, pred)})
    dec_df = pd.DataFrame(drows)
    leak_gap = float(drows[4]["hold_sample_f1"] - drows[3]["hold_sample_f1"])
    dec_df.to_csv(os.path.join(a.result_dir, "ablation_decoding.csv"), index=False)

    # ---- head / TTA variants of the reference config's first run
    p0 = configs[a.reference][0]
    d0 = _load_run(a.out_dir, p0, folds, pool_uids, True, False, n)
    vo, vh = _variants(d0["oof"]), _variants(d0["hold"])
    hrows = [{"variant": k, "oof_sample_f1": E.sample_mean_f1(y_pool, E.predict_labels(vo[k], 0.5, True)),
              "hold_sample_f1": E.sample_mean_f1(y_hold, E.predict_labels(vh[k], 0.5, True))} for k in vo]
    head_df = pd.DataFrame(hrows)
    head_df.to_csv(os.path.join(a.result_dir, "ablation_heads_tta.csv"), index=False)

    md = ["# Ablation report", "", f"Reference: **{a.reference}**. Held-out set: {len(y_hold)} images, {len(set(groups))} patients; "
          f"CI = patient bootstrap ({a.n_boot}).", "", "## Configurations", _md(cfg_df.round(4)), "",
          "## Decoding (reference config)", _md(dec_df.round(4)), "",
          f"Leak gap (leaky - honest per-class thresholds) = **{leak_gap:+.4f}**", "",
          f"## Head / TTA variants of `{p0}`", _md(head_df.round(4)), ""]
    open(os.path.join(a.result_dir, "ablation.md"), "w").write("\n".join(md))
    return {"configs": cfg_df, "decoding": dec_df, "heads": head_df, "leak_gap": leak_gap}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--config_json", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--index_csv", required=True)
    p.add_argument("--cv_folds_csv", required=True)
    p.add_argument("--result_dir", required=True)
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--n_boot", type=int, default=1000)
    r = run(p.parse_args())
    print(r["configs"].round(4).to_string(index=False)); print(r["decoding"].round(4).to_string(index=False))
    print("leak gap:", round(r["leak_gap"], 4))


if __name__ == "__main__":
    main()
