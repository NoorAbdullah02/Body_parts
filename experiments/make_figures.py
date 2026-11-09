"""Publication figures from the result files.  uv run --with matplotlib --with pandas --with numpy python make_figures.py"""
import re, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
R, O = "results", "figures"; os.makedirs(O, exist_ok=True)
NAVY, TEAL, AMBER, CORAL, GREY, LIGHT, INK = "#1B3A5C", "#2A9D8F", "#E9A23B", "#D1495B", "#8D99AE", "#EEF2F7", "#1F2937"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#9CA3AF", "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 11, "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.dpi": 150})
def save(fig, name, axes=None):
    fig.savefig(f"{O}/{name}.png", dpi=300, bbox_inches="tight", facecolor="white"); fig.savefig(f"{O}/{name}.pdf", bbox_inches="tight", facecolor="white")
    if axes is not None:     # Elsevier asks for one graph per file: also write each panel separately
        r = fig.canvas.get_renderer()
        for letter, a in zip("AB", axes):
            bb = a.get_tightbbox(r).transformed(fig.dpi_scale_trans.inverted()).expanded(1.02, 1.04)
            fig.savefig(f"{O}/{name}_{letter}.png", dpi=300, bbox_inches=bb, facecolor="white"); fig.savefig(f"{O}/{name}_{letter}.pdf", bbox_inches=bb, facecolor="white")
    plt.close(fig); print("wrote", name)
def panel(ax, s): ax.text(-0.12, 1.06, s, transform=ax.transAxes, fontsize=13, fontweight="bold")

# ---------- Fig 1: protocol flow chart
fig, ax = plt.subplots(figsize=(10, 4.3)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 44)
def box(x, y, w, h, t, fc, ec=None, fs=8.6, tc="white", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.4", fc=fc, ec=ec or fc, lw=1.2))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal", linespacing=1.35)
def arr(x1, y1, x2, y2, c=GREY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.5, color=c))
box(1, 17, 17, 10, "UNIFESP\n1,738 images\n768 patients", NAVY, bold=True)
arr(18.8, 22, 22.2, 22)
box(23, 17, 16, 10, "Patient-level\nmultilabel-stratified\nsplit (seed 42)", TEAL)
arr(39.8, 24.5, 45.2, 32.5); arr(39.8, 19.5, 45.2, 11.5)
box(46, 29, 19, 10, "CV pool\n1,415 images, 612 pts\n5 folds", LIGHT, GREY, tc=INK)
box(46, 5, 19, 10, "Held-out set\n323 images, 156 pts\nnever trained on", AMBER, tc=INK, bold=True)
arr(65.8, 34, 67.2, 34)
box(68, 29, 31, 10, "Train 6 backbones x 5 folds\n-> out-of-fold (OOF) predictions", LIGHT, GREY, tc=INK, fs=8.2)
arr(83.5, 28.6, 83.5, 24.4)
box(68, 16.5, 31, 7.4, "ALL choices on OOF only:\nhead/TTA, ensemble weights, thresholds", CORAL, bold=True, fs=7.6)
arr(83.5, 16.0, 83.5, 13.0); arr(65.8, 9.5, 67.2, 9.5)
box(68, 4.5, 31, 8, "ONE-TIME report on held-out set\n+ patient-bootstrap 95% CI", NAVY, bold=True, fs=7.8)
ax.text(50, 41.5, "Leakage-controlled evaluation protocol", ha="center", fontsize=12.5, fontweight="bold", color=NAVY)
ax.text(50, 1.2, "Extensions: ablations - frozen foundation-model probe - MURA / FracAtlas transfer - reliability and label audit - Grad-CAM audit", ha="center", fontsize=8.5, color=GREY)
save(fig, "fig1_protocol")

# ---------- Fig 2: ablation forest
d = pd.read_csv(f"{R}/results_ablation/ablation_configs.csv").dropna(subset=["diff_vs_ref"]).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 5.6)); y = np.arange(len(d))
ax.axvspan(-0.03, 0.03, color="none"); ax.axvline(0, color=INK, lw=1)
ax.errorbar(d.diff_vs_ref, y, xerr=[d.diff_vs_ref - d.diff_ci_low, d.diff_ci_high - d.diff_vs_ref], fmt="none", ecolor=GREY, elinewidth=2, capsize=3, zorder=2)
ax.scatter(d.diff_vs_ref, y, s=55, c=[NAVY if v >= 0 else CORAL for v in d.diff_vs_ref], zorder=3, edgecolor="white", linewidth=1)
for yi, p in zip(y, d.p_value): ax.text(0.0345, yi, f"p = {p:.2f}", va="center", fontsize=8, color=GREY)
ax.set_xlim(-0.035, 0.048); ax.set_yticks(y); ax.set_yticklabels(d.config, fontsize=8.5); ax.grid(axis="y", visible=False)
ax.set_xlabel("Change in held-out sample-F1 vs reference Swin-T  (95% paired patient-bootstrap CI)")
ax.set_title("No configuration is significantly better or worse than the reference", loc="left")
save(fig, "fig2_ablation_forest")

# ---------- Fig 3: per-class recall (lollipop)
txt = open(f"{R}/reliability/reliability_printed_output.md", encoding="utf-8").read()
rows = [(m[1].strip(), int(m[2]), float(m[4]), float(m[5]), float(m[6])) for m in re.finditer(r"^\| ([A-Za-z ]+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+)-([\d.]+) \|", txt, re.M)]
pc = pd.DataFrame(rows, columns=["cls", "n", "recall", "lo", "hi"]).sort_values(["recall", "n"]); y = np.arange(len(pc))
fig, ax = plt.subplots(figsize=(6.6, 6.6))
col = [CORAL if r < 0.7 else AMBER if r < 0.95 else TEAL for r in pc.recall]
ax.hlines(y, pc.lo, pc.hi, color="#CBD5E1", lw=5, zorder=1, capstyle="round"); ax.scatter(pc.recall, y, s=70, c=col, zorder=3, edgecolor="white", linewidth=1)
ax.set_yticks(y); ax.set_yticklabels([f"{c}  (n={n})" for c, n in zip(pc.cls, pc.n)], fontsize=9); ax.set_xlim(-0.03, 1.04); ax.grid(axis="y", visible=False)
ax.set_xlabel("Held-out recall (bar = 95% patient-bootstrap CI)"); ax.set_title("Per-class recall: weak classes are the rare ones", loc="left")
for lab, c in (("recall >= 0.95", TEAL), ("0.70-0.95", AMBER), ("< 0.70", CORAL)): ax.scatter([], [], c=c, s=40, label=lab)
ax.legend(loc="lower left", frameon=False, fontsize=8)
save(fig, "fig3_per_class_recall")

# ---------- Fig 4: reliability (risk-coverage + threshold transfer)
cov = np.array([1.0, 0.9905, 0.9505, 0.9009, 0.8019, 0.7028]); acc = np.array([0.9257, 0.9312, 0.9414, 0.9450, 0.9730, 0.9824])
lo = np.array([0.8718, 0.8788, 0.8942, 0.8997, 0.9305, 0.9478]); hi = np.array([0.9696, 0.9747, 0.9801, 0.9849, 1.0, 1.0])
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.7), gridspec_kw={"width_ratios": [1.25, 1]})
a = ax[0]; a.fill_between(cov * 100, lo, hi, color=TEAL, alpha=0.18, lw=0); a.plot(cov * 100, acc, "-o", color=TEAL, lw=2.2, ms=6, mec="white")
for c_, a_ in list(zip(cov, acc))[:1] + list(zip(cov, acc))[2:]: a.annotate(f"{a_:.3f}", (c_ * 100, a_), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8, color=INK)
a.invert_xaxis(); a.set_ylim(0.86, 1.01); a.set_xlabel("Coverage: share of images answered by the model (%)"); a.set_ylabel("Top-1 accuracy of answered images")
a.set_title("Selective prediction (held-out)", loc="left"); panel(a, "A")
b = ax[1]; tg = np.array([0.95, 0.98, 0.99]); got = np.array([0.9371, 0.9833, 0.9858]); cv = np.array([98.45, 73.99, 65.63]); x = np.arange(3); w = 0.36
b.bar(x - w / 2, tg, w, color="#CBD5E1", label="target (on OOF)"); b.bar(x + w / 2, got, w, color=NAVY, label="achieved (held-out)")
for xi, c_ in zip(x, cv): b.text(xi + w / 2, 0.865, f"{c_:.0f}%\ncovered", ha="center", fontsize=7.5, color="white")
b.set_ylim(0.85, 1.0); b.set_xticks(x); b.set_xticklabels(["0.95", "0.98", "0.99"]); b.set_xlabel("Target accuracy"); b.legend(frameon=False, fontsize=8, loc="upper left")
b.set_title("OOF threshold only partly transfers", loc="left"); panel(b, "B"); fig.tight_layout()
save(fig, "fig4_reliability", axes=ax)

# ---------- Fig 5: frozen encoders + label efficiency
e = pd.read_csv(f"{R}/fm/fm_label_efficiency.csv"); e = e[e.model != "resnet18_random"]; fs = pd.read_csv(f"{R}/fm/fm_summary.csv").dropna(subset=["hold_sample_f1"]); fs = fs[fs.model != "resnet18_random"]
fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
names = {"dinov2_s": "DINOv2-S (frozen)", "dinov2_b": "DINOv2-B (frozen)", "swin_in1k": "Swin-T ImageNet (frozen)"}; cols = {"dinov2_s": TEAL, "dinov2_b": AMBER, "swin_in1k": GREY}
a = ax[0]
for m, g in e.groupby("model"): a.plot(g.train_patient_fraction * 100, g.oof_sample_f1, "-o", color=cols[m], lw=2.2, ms=6, mec="white", label=names[m])
a.set_xscale("log"); a.set_xticks([10, 25, 50, 100]); a.set_xticklabels(["10", "25", "50", "100"]); a.set_xlabel("Training patients used (%)"); a.set_ylabel("OOF sample-F1"); a.legend(frameon=False, fontsize=8)
a.set_title("Label efficiency still rising", loc="left"); panel(a, "A")
b = ax[1]; lab = ["DINOv2-S", "DINOv2-B", "Swin-T\n(ImageNet)", "Swin-T\nfine-tuned", "Ensemble"]
v = list(fs.set_index("model").loc[["dinov2_s", "dinov2_b", "swin_in1k"], "hold_sample_f1"]) + [0.9089, 0.9189]
l = list(fs.set_index("model").loc[["dinov2_s", "dinov2_b", "swin_in1k"], "hold_ci_low"]) + [0.8576, 0.8669]; h = list(fs.set_index("model").loc[["dinov2_s", "dinov2_b", "swin_in1k"], "hold_ci_high"]) + [0.9509, 0.9602]
c = [GREY, GREY, GREY, NAVY, TEAL]; v, l, h = map(np.array, (v, l, h))
b.bar(range(5), v, color=c, yerr=[v - l, h - v], capsize=3, ecolor=INK, error_kw={"lw": 1}); b.set_ylim(0.78, 0.98); b.set_xticks(range(5)); b.set_xticklabels(lab, fontsize=8)
for i, vv in enumerate(v): b.text(i, 0.795, f"{vv:.3f}", ha="center", color="white", fontsize=8, fontweight="bold")
b.set_ylabel("Held-out sample-F1 (95% CI)"); b.set_title("Frozen vs fine-tuned", loc="left"); panel(b, "B"); fig.tight_layout()
save(fig, "fig5_foundation_models", axes=ax)

# ---------- Fig 6: zero-shot transfer (both heads)
t = pd.read_csv(f"{R}/transfer/summary.csv"); ci = t.ci.str.strip("[]").str.split(",", expand=True).astype(float); t["lo"], t["hi"] = ci[0], ci[1]
pairs = list(dict.fromkeys(zip(t.source, t.target))); fig, ax = plt.subplots(figsize=(7.6, 3.9)); w = 0.36
for k, (hd, c) in enumerate((("softmax", GREY), ("sigmoid", NAVY))):
    s = t[t["head"] == hd].set_index(["source", "target"]).loc[pairs]
    ax.bar(np.arange(len(pairs)) + (k - 0.5) * w, s.accuracy, w, color=c, label=hd + " head", yerr=[s.accuracy - s.lo, s.hi - s.accuracy], capsize=3, ecolor=INK, error_kw={"lw": 1})
    for i, vv in enumerate(s.accuracy): ax.text(i + (k - 0.5) * w, 0.52, f"{vv:.2f}", ha="center", color="white", fontsize=8, fontweight="bold")
nn = t[t["head"] == "sigmoid"].set_index(["source", "target"]).loc[pairs].n
ax.set_xticks(range(len(pairs))); ax.set_xticklabels([f"{s.upper()} -> {g.capitalize() if g=='fracatlas' else g.upper()}\nn = {n}" for (s, g), n in zip(pairs, nn)], fontsize=8)
ax.set_ylim(0.5, 1.0); ax.set_ylabel("Closed-set accuracy (shared groups)"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Zero-shot transfer is moderate and head-dependent", loc="left"); ax.text(0.99, 0.02, "image-level CIs: optimistic; class maps unreviewed", transform=ax.transAxes, ha="right", fontsize=7.5, color=GREY)
save(fig, "fig6_transfer")

# ---------- Fig 7: leakage + gate collapse
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6)); a = ax[0]
vals = [0.0004, 0.0144, 0.0]; labs = ["patient-level\nvalidation", "random image-level\nvalidation", ]
a.bar([0, 1], [0.0004, 0.0144], color=[TEAL, CORAL], width=0.55); a.bar([2.0], [0.0083], color=AMBER, width=0.55)
for x_, v_ in zip([0, 1, 2], [0.0004, 0.0144, 0.0083]): a.text(x_, v_ + 0.0004, f"+{v_:.4f}", ha="center", fontsize=9, fontweight="bold")
a.set_xticks([0, 1, 2]); a.set_xticklabels(["patient-level\nsplit", "random image-\nlevel split", "thresholds tuned\non held-out set"], fontsize=8)
a.set_ylabel("Over-estimate of F1 (validation or test)"); a.set_ylim(0, 0.0185); a.set_title("Evaluation shortcuts inflate estimates", loc="left"); panel(a, "A")
g = pd.read_csv(f"{R}/results_ablation/gate_weights.csv").set_index("config"); b = ax[1]
nm = {"hybrid_gate_s224": "learned gate", "hybrid_gate_ln_s224": "gate + LayerNorm", "hybrid_avg_s224": "fixed average"}
vv = [g.loc[k, "mean"] for k in nm]; b.bar(range(3), vv, color=[CORAL, CORAL, GREY], width=0.55, yerr=[g.loc[k, "std"] for k in nm], capsize=3)
for i, v_ in enumerate(vv): b.text(i, v_ + 0.025, f"{v_:.4f}" if v_ < 0.4 else f"{v_:.2f}", ha="center", fontsize=9, fontweight="bold")
b.set_xticks(range(3)); b.set_xticklabels(list(nm.values()), fontsize=8.5); b.set_ylim(0, 0.7); b.set_ylabel("Mean weight of the CNN branch"); b.set_title("The learned gate ignores the CNN", loc="left"); panel(b, "B"); fig.tight_layout()
save(fig, "fig7_leakage_and_gate", axes=ax)

# ---------- Fig 8: Kaggle
names = ["1st place", "2nd place", "ours: Stage 2 (s2u)", "ours: Stage 1", "3rd place", "4th place", "5th place"]; sc = np.array([0.93378, 0.92787, 0.92638, 0.92563, 0.92488, 0.92414, 0.92182])
fig, ax = plt.subplots(figsize=(7.2, 3.8)); y = np.arange(len(sc))[::-1]; col = [GREY, GREY, NAVY, TEAL, GREY, GREY, GREY]
ax.hlines(y, 0.921, sc, color="#CBD5E1", lw=2.4); ax.scatter(sc, y, s=[80 if "ours" in n else 55 for n in names], c=col, zorder=3, edgecolor="white", linewidth=1.2)
for yi, s_, n in zip(y, sc, names): ax.text(0.9337 if False else s_ + 0.0004, yi + 0.0, f" {s_:.5f}", va="center", fontsize=8.5, color=INK, fontweight="bold" if "ours" in n else "normal")
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9); ax.set_xlim(0.9205, 0.9385); ax.grid(axis="y", visible=False); ax.set_xlabel("Private leaderboard score (sample-mean F1)")
ax.set_title("Late submissions would rank between 2nd and 3rd", loc="left"); ax.text(0.0, -0.22, "Standard error of one private score is about 0.012 (our estimate, ~446 images): gaps between these entries are not statistically resolvable.", transform=ax.transAxes, ha="left", fontsize=7.5, color=GREY)
save(fig, "fig8_kaggle")

# ---------- Fig 9: MURA vs Shah
labs = ["Ours\n(5-fold bag)", "Xception-Swin\n(Shah 2026)", "DenseNet-201", "EfficientNet-B0", "ResNet-101", "VGG-19"]; acc = np.array([0.9747, 0.9643, 0.9187, 0.9043, 0.8934, 0.8612]); lo_, hi_ = 0.9663, 0.9818
fig, ax = plt.subplots(figsize=(6.8, 3.6)); ax.bar(range(6), acc, color=[TEAL] + [GREY] * 5, width=0.62)
ax.errorbar([0], [acc[0]], yerr=[[acc[0] - lo_], [hi_ - acc[0]]], fmt="none", ecolor=INK, capsize=4)
for i, v in enumerate(acc): ax.text(i, 0.828, f"{v:.3f}", ha="center", color="white", fontsize=8.5, fontweight="bold")
ax.set_ylim(0.82, 1.0); ax.set_xticks(range(6)); ax.set_xticklabels(labs, fontsize=8); ax.set_ylabel("Accuracy, MURA validation (body part)")
ax.set_title("MURA body-part classification (7 classes)", loc="left"); ax.text(0.99, 0.03, "other models: numbers from Table 8 of Shah et al. (2026), no CIs shown", transform=ax.transAxes, ha="right", fontsize=7, color=GREY)
save(fig, "fig9_mura")

# ---------- Fig 10: graphical abstract style overview of held-out F1
fig, ax = plt.subplots(figsize=(7.4, 3.6)); items = [("Frozen DINOv2-S\n+ linear probe", 0.8859, 0.8327, 0.9318, GREY), ("Fine-tuned Swin-T", 0.9089, 0.8576, 0.9509, NAVY), ("Best single\n(ConvNeXt-T)", 0.9123, np.nan, np.nan, NAVY), ("Ensemble\n(6 runs, OOF thr.)", 0.9189, 0.8669, 0.9602, TEAL)]
for i, (n, v, l_, h_, c) in enumerate(items):
    ax.bar(i, v, color=c, width=0.58)
    if not np.isnan(l_): ax.errorbar(i, v, yerr=[[v - l_], [h_ - v]], fmt="none", ecolor=INK, capsize=4)
    ax.text(i, 0.805, f"{v:.3f}", ha="center", color="white", fontsize=10, fontweight="bold")
ax.set_ylim(0.8, 0.98); ax.set_xticks(range(4)); ax.set_xticklabels([x[0] for x in items], fontsize=8.5); ax.set_ylabel("Held-out sample-F1 (95% CI)")
ax.set_title("All CIs overlap: differences are within noise", loc="left")
save(fig, "fig10_heldout_overview")
