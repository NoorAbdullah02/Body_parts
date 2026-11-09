"""
make_paper_tables -- turns the result files of the notebooks into the manuscript tables (markdown, optional LaTeX).

Put the downloaded results next to each other in ONE folder (default experiments/results/):

    main_results.json        results/results.json of UNIFESP_v2_pipeline (stage 1)
    main_per_run.csv         results/per_run.csv of the same notebook
    results_ablation/        ablation.md, ablation_configs.csv, ablation_decoding.csv, ablation_heads_tta.csv, gate_weights.csv
    results_mura/            results.json, shah_comparison.json
    results_covid/           results.json
    transfer/                summary.csv                      (UNIFESP_v2_cross)
    scores.json              {"stage1": {"private": 0.92563, "public": 0.92704}, "s2a": {...}, "s2b": {...}, "s2u": {...}}

Every input is optional; a missing one only skips its table (and says so).  Numbers are copied, never recomputed or rounded away.

    python make_paper_tables.py --root results --out paper_tables.md [--latex]
"""
import argparse
import json
import os

import pandas as pd

NICE = {"efficientnetv2_s_s256": "EfficientNetV2-S", "convnext_tiny_s256": "ConvNeXt-T", "resnet50_s256_heavy": "ResNet50 (heavy augmentation)", "efficientnet_b3_s256_hv": "EfficientNet-B3"}
LB = [("1st place", 0.93378), ("2nd place (ResNet50/152, FastAI)", 0.92787), ("3rd place", 0.92488),
      ("4th place", 0.92414), ("5th place", 0.92182)]      # private leaderboard read from kaggle.com, Nov 2026
SHAH = [("Xception-Swin (Shah 2026)", 0.9643, 0.9574), ("DenseNet-201", 0.9187, 0.9093), ("EfficientNet-B0", 0.9043, 0.8952),
        ("ResNet-101", 0.8934, 0.8831), ("VGG-19", 0.8612, 0.8498)]            # Shah et al. 2026, J Imaging 12(7):298 (their test = MURA valid)


def md_table(df):
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join("" if (isinstance(v, float) and pd.isna(v)) else str(v) for v in r.tolist()) + " |")
    return "\n".join(rows)


def tex_table(df, caption):
    body = " \\\\\n".join(" & ".join(str(v).replace("_", "\\_").replace("%", "\\%") for v in r) for r in df.values.tolist())
    return ("\\begin{table}[t]\\centering\\caption{" + caption + "}\n\\begin{tabular}{" + "l" * len(df.columns) + "}\\hline\n"
            + " & ".join(c.replace("_", "\\_") for c in df.columns) + " \\\\\\hline\n" + body + " \\\\\\hline\n\\end{tabular}\\end{table}")


def ci(est, lo, hi, nd=4):
    return f"{est:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def load_json(root, *names):
    for n in names:
        p = os.path.join(root, n)
        if os.path.exists(p):
            return json.load(open(p))
    return None


def load_csv(root, *names):
    for n in names:
        p = os.path.join(root, n)
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


def build(root):
    sections, skipped = [], []

    def add(title, df, note="", tex=None):
        sections.append((title, df, note, tex or title))

    # 1. main UNIFESP held-out result ------------------------------------------------------------
    r = load_json(root, "main_results.json", os.path.join("results", "results.json"))
    if r and "holdout" in r:
        h = r["holdout"]
        rows = []
        for k, lab in (("ens_0.5", "ensemble, threshold 0.5"), ("ens_0.5_min_one", "ensemble, 0.5 + min-one"),
                       ("ens_tuned_min_one", "ensemble, per-class thresholds tuned on OOF + min-one")):
            m = h[k]; rows.append({"decoding": lab, "sample_mean_F1": f"{m['sample_mean_f1']:.4f}", "macro_F1": f"{m['macro_f1']:.4f}",
                                   "micro_F1": f"{m['micro_f1']:.4f}", "subset_acc": f"{m['subset_accuracy']:.4f}"})
        c1, c2 = h["ci_sample_mean_f1_tuned"], h["ci_macro_f1_tuned"]
        rows[-1]["sample_mean_F1"] = ci(c1["estimate"], c1["ci_low"], c1["ci_high"])
        rows[-1]["macro_F1"] = ci(c2["estimate"], c2["ci_low"], c2["ci_high"])
        b = h["best_single_metrics"]
        rows.append({"decoding": f"best single run by OOF ({h['best_single_by_oof']}), 0.5 + min-one", "sample_mean_F1": f"{b['sample_mean_f1']:.4f}",
                     "macro_F1": f"{b['macro_f1']:.4f}", "micro_F1": f"{b['micro_f1']:.4f}", "subset_acc": f"{b['subset_accuracy']:.4f}"})
        p = h["paired_ens_vs_single_sample_f1"]
        note = (f"Held-out set: {h['n_images']} images / {h['n_patients']} patients; 95 % CIs by patient-level bootstrap. Ensemble minus best single: "
                f"{p['diff']:+.4f} [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}], p = {p['p_value']:.3f}. Ensemble weights fitted on OOF predictions (Caruana selection; the other two runs, Swin-T and DenseNet121, received weight 0): "
                + ", ".join(f"{NICE.get(k, k)} {v:.2f}" for k, v in r['weights'].items() if v > 0) + ".")
        add("Table 1. UNIFESP held-out performance (decisions made on out-of-fold predictions only)", pd.DataFrame(rows), note)
    else:
        skipped.append("main_results.json")

    pr = load_csv(root, "main_per_run.csv", os.path.join("results", "per_run.csv"))
    if pr is not None:
        keep = [c for c in ("run", "chosen_variant", "oof_chosen_variant", "oof_blend", "holdout_sample_mean_f1_0.5_min_one") if c in pr.columns]
        add("Table 2. Single runs (head/TTA variant chosen on OOF)", pr[keep].round(4), "OOF = pooled out-of-fold sample-mean F1 of the chosen variant is in the full CSV.")
    else:
        skipped.append("main_per_run.csv")

    # 3. Kaggle comparison -------------------------------------------------------------------------
    sc = load_json(root, "scores.json")
    rows = [{"entry": n, "private_score": f"{v:.5f}", "public_score": "n/a"} for n, v in LB]
    if sc:
        for k, lab in (("stage1", "ours: stage-1 ensemble"), ("s2a", "ours: stage-2a all-data"), ("s2b", "ours: stage-2b +pseudo-labels"), ("s2u", "ours: stage-2 union")):
            if k in sc:
                rows.append({"entry": lab + " (late submission)", "private_score": f"{sc[k]['private']:.5f}", "public_score": f"{sc[k].get('public', ''):}"})
    else:
        skipped.append("scores.json")
    d = pd.DataFrame(rows)
    d["_s"] = d["private_score"].astype(float); d = d.sort_values("_s", ascending=False).drop(columns="_s")
    add("Table 3. Kaggle UNIFESP private score vs. the competition leaderboard",
        d, "Mean F1 (sample-wise); private score uses ~60 % of the test set with the original (partly noisy) labels, so differences of ~0.003 are within noise. "
           "Late submissions are not listed on the leaderboard; we report every submission we made.")

    # 4-6 ablation --------------------------------------------------------------------------------
    ab = root if os.path.exists(os.path.join(root, "ablation_configs.csv")) else os.path.join(root, "results_ablation")
    cfg = load_csv(ab, "ablation_configs.csv")
    if cfg is not None:
        t = pd.DataFrame({"config": cfg["config"], "held-out sample-F1 [95% CI]": [ci(a, b, c) for a, b, c in zip(cfg.hold_sample_f1, cfg.hold_ci_low, cfg.hold_ci_high)],
                          "macro-F1": cfg.hold_macro_f1.round(4), "OOF - held-out": cfg["oof_minus_hold"].round(4)})
        if "diff_vs_ref" in cfg.columns:
            t["diff vs reference [95% CI]"] = ["" if pd.isna(a) else ci(a, b, c, 4) for a, b, c in zip(cfg.diff_vs_ref, cfg.diff_ci_low, cfg.diff_ci_high)]
            t["p"] = cfg.p_value.round(3)
        add("Table 4. Ablations (5 CV folds each, same held-out set; paired patient bootstrap vs. the reference)", t,
            "`OOF - held-out` large and positive = the validation score over-estimates (compare `_leaky` with the patient-level reference for the split ablation).")
    else:
        skipped.append("ablation_configs.csv")
    dec = load_csv(ab, "ablation_decoding.csv")
    if dec is not None:
        t = pd.DataFrame({"decoding": dec["decoding"], "held-out sample-F1 [95% CI]": [ci(a, b, c) for a, b, c in zip(dec.hold_sample_f1, dec.ci_low, dec.ci_high)], "macro-F1": dec.macro_f1.round(4)})
        gap = dec.hold_sample_f1.iloc[4] - dec.hold_sample_f1.iloc[3] if len(dec) >= 5 else float("nan")
        add("Table 5. Decoding ablation and evaluation leakage", t, f"Leak gap (thresholds tuned on the held-out set minus thresholds tuned on OOF) = {gap:+.4f}.")
    else:
        skipped.append("ablation_decoding.csv")
    g = load_csv(ab, "gate_weights.csv")
    if g is not None:
        if "run" in g.columns:
            g = g.assign(config=g["run"].str.replace(r"_f\d$", "", regex=True))
            t = g.groupby("config")["cnn_gate_weight_mean"].agg(["mean", "std", "min", "max"]).round(4).reset_index()
        else:  # aggregate as printed by the notebook
            t = g.round(4)
        add("Table 6. Mean weight given to the CNN branch by the fusion gate (5 folds)", t, "hybrid_avg is fixed at 0.5; hybrid_cnn / hybrid_swin are 1 / 0 by construction.")
    else:
        skipped.append("gate_weights.csv")

    # 7. MURA vs Shah ------------------------------------------------------------------------------
    sh = load_json(root, os.path.join("results_mura", "shah_comparison.json"), "shah_comparison.json")
    if sh:
        rows = [{"model": "ours: 5-fold bagged ensemble (official MURA split)", "accuracy [95% CI]": ci(sh["accuracy"]["estimate"], sh["accuracy"]["ci_low"], sh["accuracy"]["ci_high"]),
                 "macro-F1 [95% CI]": ci(sh["macro_f1"]["estimate"], sh["macro_f1"]["ci_low"], sh["macro_f1"]["ci_high"])}]
        rows += [{"model": n, "accuracy [95% CI]": f"{a:.4f} (CI n/a)", "macro-F1 [95% CI]": f"{f:.4f} (CI n/a)"} for n, a, f in SHAH]
        add("Table 7. MURA body-part classification: held-out = MURA validation set", pd.DataFrame(rows),
            "Shah et al. 2026 (results table in their paper; verified in the full text): same MURA validation set (3,197 images), fixed seed 42, image-level evaluation; no CI is shown for these body-part numbers.")
    else:
        skipped.append("results_mura/shah_comparison.json")

    # 8. COVID --------------------------------------------------------------------------------------
    cv = load_json(root, os.path.join("results_covid", "results.json"))
    if cv and "holdout" in cv:
        h = cv["holdout"]; m = h["ens_tuned_min_one"]; c1 = h["ci_sample_mean_f1_tuned"]; c2 = h["ci_macro_f1_tuned"]
        add("Table 8. COVID-19 Radiography (4 classes, image-level split)", pd.DataFrame([{"held-out images": h["n_images"], "sample-F1 [95% CI]": ci(c1["estimate"], c1["ci_low"], c1["ci_high"]),
            "macro-F1 [95% CI]": ci(c2["estimate"], c2["ci_low"], c2["ci_high"]), "subset acc": f"{m['subset_accuracy']:.4f}"}]),
            "No patient identifiers exist in this dataset: the split is image-level and the numbers may be optimistic (state this limitation).")
    else:
        skipped.append("results_covid/results.json")

    # 9. transfer -----------------------------------------------------------------------------------
    tr = load_csv(os.path.join(root, "transfer"), "summary.csv") if os.path.isdir(os.path.join(root, "transfer")) else load_csv(root, "summary.csv")
    if tr is not None:
        add("Table 9. Zero-shot cross-dataset transfer through the class map", tr, "Closed-set decision among the shared groups; CIs are image-level (the targets have no patient ids). The class map is an assumption to be checked by a clinician.")
    else:
        skipped.append("transfer/summary.csv")
    # 10. explainability ----------------------------------------------------------------------------
    ex = load_csv(root, "explain_summary.csv")
    if ex is not None:
        add("Table 10. Explainability: faithfulness (deletion test) and randomisation sanity check of Grad-CAM on the held-out set", ex,
            "gain = random-map deletion score minus CAM deletion score (> 0 and CI above 0 = the map points at pixels the model uses); rho = Spearman correlation of the CAM with the CAM of a "
            "randomly initialised model (low = passes). Grad-CAM grids are coarse and no localisation ground truth exists: say so.")
    else:
        skipped.append("explain_summary.csv")
    return sections, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--out", default="paper_tables.md")
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    sections, skipped = build(a.root)
    out = ["# Paper tables (generated; do not edit by hand)", ""]
    for title, df, note, tex in sections:
        out += [f"## {title}", "", md_table(df), "", f"_{note}_" if note else "", ""]
        if a.latex:
            out += ["```latex", tex_table(df, title.split('. ', 1)[-1]), "```", ""]
    if skipped:
        out += ["## Not generated (input missing)", ""] + [f"* {s}" for s in skipped]
    open(a.out, "w").write("\n".join(out))
    print(f"wrote {a.out}: {len(sections)} tables; skipped: {skipped}")


if __name__ == "__main__":
    main()
