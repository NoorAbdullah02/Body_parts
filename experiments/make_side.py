"""Side-by-side composite figures (A | B) for Figs 4, 11, 12. uv run --with matplotlib --with pandas --with numpy python make_side.py"""
import re, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R, O = "results", "figures"
NAVY, TEAL, AMBER, CORAL, GREY, INK = "#1B3A5C", "#2A9D8F", "#E9A23B", "#D1495B", "#8D99AE", "#1F2937"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": "#9CA3AF", "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 10, "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.7, "axes.axisbelow": True})
def save(fig, name):
    fig.savefig(f"{O}/{name}.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig); print("wrote", name)
def letter(ax, s): ax.text(-0.02, 1.07, s, transform=ax.transAxes, fontsize=14, fontweight="bold", ha="right")
short = lambda s: s.replace(" Spine", "")
ct = pd.read_csv(f"{R}/eda/class_table.csv").sort_values("images")
idx = pd.read_csv(f"{R}/kaggle_raw/stage2/folds.csv"); lab = [c for c in idx.columns if c.startswith("label_")]
names = [re.sub(r"^label_\d+_", "", c).replace("_", " ") for c in lab]; Y = idx[lab].to_numpy(int)

# ---- c1: label pairs | split composition
M = Y.T @ Y; pairs = sorted(((int(M[i, j]), short(names[i]) + " + " + short(names[j])) for i in range(22) for j in range(i + 1, 22)), reverse=True)[:10][::-1]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 5.0), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.95})
a.barh([p[1] for p in pairs], [p[0] for p in pairs], color=NAVY); a.grid(axis="y", visible=False)
for i, p in enumerate(pairs): a.text(p[0] + 1, i, str(p[0]), va="center", fontsize=8.5)
a.set_xlabel("Images with both labels"); a.set_title("Most frequent label pairs", loc="left"); a.tick_params(axis="y", labelsize=8.5); letter(a, "A")
y = np.arange(len(ct)); b.barh(y, ct.cv_pool_images, color=NAVY, label="CV pool (1,415)"); b.barh(y, ct.holdout_images, left=ct.cv_pool_images, color=AMBER, label="held-out (323)")
b.set_yticks(y); b.set_yticklabels(ct["class"], fontsize=8); b.set_xscale("log"); b.grid(axis="y", visible=False); b.set_xlabel("Images per class (log scale)")
b.legend(frameon=False, fontsize=8, loc="lower right"); b.set_title("Both splits cover every class", loc="left"); letter(b, "B")
save(fig, "c1_pairs_split")

# ---- c2: label audit | confusions
au = pd.read_csv(f"{R}/reliability/label_audit_suspects_kaggle.csv"); ac = au.groupby(["given", "model_says"]).size().sort_values().tail(10)
cf = pd.concat([pd.read_csv(f"{R}/kaggle_raw/explain/{m}_top_confusions.csv").assign(model=m) for m in ("convnext_tiny_s256", "resnet50_s256_heavy", "swin_tiny_s224")])
cf["pair"] = cf.true_class_missed + " → " + cf.predicted_instead; pv = cf.pivot_table(index="pair", columns="model", values="count", fill_value=0)
pv["t"] = pv.sum(1); pv = pv.sort_values("t").tail(7).drop(columns="t")
fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 4.2), gridspec_kw={"wspace": 0.8})
a.barh([f"{x} → {z}" for x, z in ac.index], ac.values, color=AMBER); a.grid(axis="y", visible=False)
for i, v in enumerate(ac.values): a.text(v + 0.08, i, str(v), va="center", fontsize=8.5)
a.set_xlabel("Training images flagged (of 39)"); a.set_title("Suspected label errors\n(given → model's label)", loc="left"); a.tick_params(axis="y", labelsize=8.5); letter(a, "A")
yy = np.arange(len(pv)); h = 0.26
for k, (m, c, n) in enumerate((("convnext_tiny_s256", NAVY, "ConvNeXt-T"), ("resnet50_s256_heavy", AMBER, "ResNet50"), ("swin_tiny_s224", TEAL, "Swin-T"))): b.barh(yy + (k - 1) * h, pv[m], h, color=c, label=n)
b.set_yticks(yy); b.set_yticklabels(pv.index, fontsize=8.5); b.grid(axis="y", visible=False); b.set_xlabel("Missed held-out images"); b.legend(frameon=False, fontsize=8, loc="lower right")
b.set_title("Held-out confusions\n(true → predicted)", loc="left"); letter(b, "B")
save(fig, "c2_audit_confusions")

# ---- c3: MURA | zero-shot transfer
labs = ["Ours (5-fold bag)", "Xception-Swin (Shah)", "DenseNet-201", "EfficientNet-B0", "ResNet-101", "VGG-19"]; acc = np.array([0.9747, 0.9643, 0.9187, 0.9043, 0.8934, 0.8612])
fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 4.3), gridspec_kw={"wspace": 0.28})
a.bar(range(6), acc, color=[TEAL] + [GREY] * 5, width=0.66); a.errorbar([0], [acc[0]], yerr=[[acc[0] - 0.9663], [0.9818 - acc[0]]], fmt="none", ecolor=INK, capsize=4)
for i, v in enumerate(acc): a.text(i, 0.828, f"{v:.3f}", ha="center", color="white", fontsize=7.5, fontweight="bold", rotation=90, va="bottom")
a.set_ylim(0.82, 1.0); a.set_xticks(range(6)); a.set_xticklabels(labs, fontsize=8, rotation=35, ha="right", rotation_mode="anchor"); a.set_ylabel("Accuracy, MURA validation"); a.set_title("MURA body part (7 classes)", loc="left"); letter(a, "A")
t = pd.read_csv(f"{R}/transfer/summary.csv"); ci = t.ci.str.strip("[]").str.split(",", expand=True).astype(float); t["lo"], t["hi"] = ci[0], ci[1]
prs = list(dict.fromkeys(zip(t.source, t.target))); w = 0.38
for k, (hd, c) in enumerate((("softmax", GREY), ("sigmoid", NAVY))):
    s = t[t["head"] == hd].set_index(["source", "target"]).loc[prs]
    b.bar(np.arange(len(prs)) + (k - 0.5) * w, s.accuracy, w, color=c, label=hd, yerr=[s.accuracy - s.lo, s.hi - s.accuracy], capsize=2.5, ecolor=INK, error_kw={"lw": 0.9})
    for i, vv in enumerate(s.accuracy): b.text(i + (k - 0.5) * w, 0.52, f"{vv:.2f}", ha="center", color="white", fontsize=7, fontweight="bold", rotation=90, va="bottom")
nn = t[t["head"] == "sigmoid"].set_index(["source", "target"]).loc[prs].n
nm = lambda g: "Frac" if g == "fracatlas" else g.upper()
b.set_xticks(range(len(prs))); b.set_xticklabels([f"{s.upper()}→\n{nm(g)}\nn={n}" for (s, g), n in zip(prs, nn)], fontsize=7.5)
b.set_ylim(0.5, 1.0); b.set_ylabel("Closed-set accuracy"); b.legend(frameon=False, fontsize=8, loc="upper right"); b.set_title("Zero-shot transfer", loc="left"); letter(b, "B")
save(fig, "c3_mura_transfer")
