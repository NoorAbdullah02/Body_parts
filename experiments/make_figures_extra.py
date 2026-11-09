"""Extra figures (dataset, split, calibration, per-class, label audit, Stage 2, confusions). uv run --with matplotlib --with pandas --with numpy python make_figures_extra.py"""
import os, json, re, shutil
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R, O = "results", "figures"
NAVY, TEAL, AMBER, CORAL, GREY, INK = "#1B3A5C", "#2A9D8F", "#E9A23B", "#D1495B", "#8D99AE", "#1F2937"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#9CA3AF", "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11, "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.dpi": 150})
def save(fig, name):
    fig.savefig(f"{O}/{name}.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig); print("wrote", name)
def panel(ax, s): ax.text(-0.12, 1.06, s, transform=ax.transAxes, fontsize=13, fontweight="bold")
ct = pd.read_csv(f"{R}/eda/class_table.csv"); ct = ct.sort_values("images")
idx = pd.read_csv(f"{R}/kaggle_raw/stage2/folds.csv"); lab = [c for c in idx.columns if c.startswith("label_")]
names = [re.sub(r"^label_\d+_", "", c).replace("_", " ") for c in lab]; Y = idx[lab].to_numpy(int)

# D1: examples montage (UNIFESP images, CC BY-NC-SA 4.0)
shutil.copy(f"{R}/eda/fig_montage.png", f"{O}/d1_examples.png")

# D2: class frequency + labels per image
fig, ax = plt.subplots(1, 2, figsize=(10.2, 5.6), gridspec_kw={"width_ratios": [1.5, 1]})
a = ax[0]; col = [CORAL if n < 20 else AMBER if n < 60 else TEAL for n in ct.images]
a.barh(ct["class"], ct.images, color=col); a.set_xscale("log"); a.grid(axis="y", visible=False)
for y, n in enumerate(ct.images): a.text(n * 1.08, y, str(n), va="center", fontsize=8)
a.set_xlabel("Training images per class (log scale)"); a.set_title("Class frequency: an 82-fold range", loc="left"); panel(a, "A")
b = ax[1]; k = Y.sum(1); cnt = [int((k == i).sum()) for i in (1, 2, 3)]
w, _ = b.pie(cnt, colors=[TEAL, AMBER, CORAL], startangle=90, counterclock=False, wedgeprops=dict(width=0.38, edgecolor="white"))
b.text(0, 0, f"{len(k):,}\nimages", ha="center", va="center", fontsize=11, fontweight="bold"); b.set_title("Labels per image", loc="left"); panel(b, "B")
b.legend(w, [f"1 label: {cnt[0]:,} ({cnt[0]/len(k)*100:.1f} %)", f"2 labels: {cnt[1]}", f"3 labels: {cnt[2]}"], loc="lower center", bbox_to_anchor=(0.5, -0.17), frameon=False, fontsize=9)
b.axis("equal"); b.grid(False); fig.tight_layout(); save(fig, "d2_classes_labels")

# D3: most frequent label pairs
M = Y.T @ Y; pairs = sorted(((int(M[i, j]), names[i] + " + " + names[j]) for i in range(22) for j in range(i + 1, 22)), reverse=True)[:10][::-1]
fig, ax = plt.subplots(figsize=(6.4, 4.0)); ax.barh([p[1] for p in pairs], [p[0] for p in pairs], color=NAVY); ax.grid(axis="y", visible=False)
for i, p in enumerate(pairs): ax.text(p[0] + 0.6, i, str(p[0]), va="center", fontsize=9)
ax.set_xlabel("Images carrying both labels"); ax.set_title("Multilabel images are dominated by two pairs", loc="left", fontsize=10); save(fig, "d3_label_pairs")

# D4: split composition per class (pool vs held-out)
c2 = ct.copy(); fig, ax = plt.subplots(figsize=(7.2, 5.6)); y = np.arange(len(c2))
ax.barh(y, c2.cv_pool_images, color=NAVY, label="cross-validation pool (1,415 images)"); ax.barh(y, c2.holdout_images, left=c2.cv_pool_images, color=AMBER, label="held-out set (323 images)")
ax.set_yticks(y); ax.set_yticklabels(c2["class"], fontsize=8.5); ax.set_xscale("log"); ax.grid(axis="y", visible=False); ax.set_xlabel("Images per class (log scale)")
ax.legend(frameon=False, fontsize=8.5, loc="lower right"); ax.set_title("Both splits cover the same classes", loc="left"); save(fig, "d4_split_composition")

# R2: per-class F1 vs number of positives
pc = pd.read_csv(f"{R}/reliability/per_class_holdout.csv")
fig, ax = plt.subplots(figsize=(6.6, 4.4)); ax.set_xscale("log")
ax.vlines(pc.n_pos, pc.f1_ci_low, pc.f1_ci_high, color="#CBD5E1", lw=3, zorder=1)
ax.scatter(pc.n_pos, pc.f1, s=45, c=[CORAL if f < 0.7 else AMBER if f < 0.9 else TEAL for f in pc.f1], edgecolor="white", zorder=3)
for _, r in pc.iterrows():
    if r.f1 < 0.8 or r.n_pos > 100: ax.annotate(r["class"], (r.n_pos, r.f1), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xlabel("Positive images in the held-out set (log scale)"); ax.set_ylabel("Held-out F1 (bar = 95 % CI)"); ax.set_ylim(-0.03, 1.05)
ax.set_title("Weak classes are the ones with few positives", loc="left"); save(fig, "r2_f1_vs_npos")

# R3: label audit pairs
au = pd.read_csv(f"{R}/reliability/label_audit_suspects_kaggle.csv"); ac = au.groupby(["given", "model_says"]).size().sort_values().tail(10)
fig, ax = plt.subplots(figsize=(6.4, 4.0)); ax.barh([f"{a} → {b}" for a, b in ac.index], ac.values, color=AMBER); ax.grid(axis="y", visible=False)
for i, v in enumerate(ac.values): ax.text(v + 0.08, i, str(v), va="center", fontsize=9)
ax.set_xlabel("Training images flagged (of 39)"); ax.set_title("Suspected label errors: given label → model's label", loc="left", fontsize=10); save(fig, "r3_label_audit")

# R4: Stage 2 OOF with CI
ci = json.load(open(f"{R}/stage2/stage2_ci.json")); labs = ["8 models\n(plain)", "8 models\n(pseudo-labels)", "16 models\n(union)"]; keys = ["s2a", "s2b", "s2u"]
v = [ci[k]["sample_f1"] for k in keys]; lo = [ci[k]["ci"][0] for k in keys]; hi = [ci[k]["ci"][1] for k in keys]
fig, ax = plt.subplots(figsize=(5.8, 3.8)); ax.bar(range(3), v, color=[NAVY, GREY, TEAL], yerr=[np.array(v) - lo, np.array(hi) - v], capsize=4, ecolor=INK, width=0.55)
for i, x in enumerate(v): ax.text(i, 0.895, f"{x:.4f}", ha="center", color="white", fontsize=9, fontweight="bold")
ax.set_ylim(0.89, 0.95); ax.set_xticks(range(3)); ax.set_xticklabels(labs); ax.set_ylabel("OOF sample-F1 (95 % CI)"); ax.set_title("Stage 2: the three ensembles are indistinguishable", loc="left", fontsize=10); save(fig, "r4_stage2")

# R5: top confusions across three models
cf = pd.concat([pd.read_csv(f"{R}/kaggle_raw/explain/{m}_top_confusions.csv").assign(model=m) for m in ("convnext_tiny_s256", "resnet50_s256_heavy", "swin_tiny_s224")])
cf["pair"] = cf.true_class_missed + " → " + cf.predicted_instead; pv = cf.pivot_table(index="pair", columns="model", values="count", fill_value=0)
pv["t"] = pv.sum(1); pv = pv.sort_values("t").tail(7).drop(columns="t")
fig, ax = plt.subplots(figsize=(6.6, 4.0)); h = 0.26; y = np.arange(len(pv))
for k, (m, c, n) in enumerate((("convnext_tiny_s256", NAVY, "ConvNeXt-T"), ("resnet50_s256_heavy", AMBER, "ResNet50"), ("swin_tiny_s224", TEAL, "Swin-T"))): ax.barh(y + (k - 1) * h, pv[m], h, color=c, label=n)
ax.set_yticks(y); ax.set_yticklabels(pv.index, fontsize=9); ax.grid(axis="y", visible=False); ax.set_xlabel("Missed held-out images"); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
ax.set_title("The same confusions appear in all three models", loc="left", fontsize=10); save(fig, "r5_confusions")

# M1: model and decision pipeline
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, ax = plt.subplots(figsize=(11, 4.6)); ax.axis("off"); ax.set_xlim(0, 110); ax.set_ylim(0, 52); ax.grid(False)
def box(x, y, w, h, t, fc, tc="white", fs=8.4, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2", fc=fc, ec=fc)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal", linespacing=1.3)
def arr(x1, y1, x2, y2): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1.4, color=GREY))
box(1, 19, 15, 12, "Radiograph\n256 px DICOM\npolarity fixed,\nscaled to 8 bit", NAVY, bold=True, fs=8)
arr(16.8, 25, 20.2, 25)
box(21, 19, 15, 12, "Light augmentation\nflip, shift-scale-rotate,\nbrightness/contrast", "#E5E7EB", tc=INK)
arr(36.8, 25, 41.2, 25)
names_b = ["Swin-T", "DenseNet121", "EfficientNetV2-S", "ConvNeXt-T", "ResNet50", "EfficientNet-B3"]
for i, n in enumerate(names_b): box(42, 41 - i * 6.6, 17, 5, n, TEAL, fs=8)
ax.text(50.5, 49.2, "6 ImageNet-pretrained backbones", ha="center", fontsize=8.6, fontweight="bold", color=INK)
arr(59.8, 25, 64.2, 25)
box(65, 26.5, 15, 6.5, "Sigmoid head\n22 labels (BCE)", AMBER, tc=INK); box(65, 17, 15, 6.5, "Softmax head\n22 classes (soft CE)", AMBER, tc=INK)
arr(80.8, 25, 85.2, 25)
box(86, 16, 22, 18, "Decisions on OOF only\nblend of heads, TTA flip\nCaruana ensemble weights\nper-class thresholds\nat least one label", CORAL, fs=8, bold=True)
ax.text(55, 0.8, "Five-fold patient-level training; the held-out set is scored once. EMA weights, no checkpoint selection.", ha="center", fontsize=8.8, color=GREY)
save(fig, "m1_architecture")

# Combined figures (one figure per page in the manuscript)
from PIL import Image
def combine(files, out, widths=None):
    ims = [Image.open(f"{O}/{f}").convert("RGB") for f in files]
    W = max(i.width for i in ims); ims = [i.resize((W, int(i.height * W / i.width)), Image.LANCZOS) for i in ims]
    H = sum(i.height for i in ims) + 60 * (len(ims) - 1); S = Image.new("RGB", (W, H), "white"); y = 0
    for i in ims: S.paste(i, (0, y)); y += i.height + 60
    S.save(f"{O}/{out}", dpi=(300, 300)); print("wrote", out)
combine(["d3_label_pairs.png", "d4_split_composition.png"], "c1_pairs_split.png")
combine(["r3_label_audit.png", "r5_confusions.png"], "c2_audit_confusions.png")
combine(["fig9_mura.png", "fig6_transfer.png"], "c3_mura_transfer.png")
combine(["fig8_kaggle.png", "r4_stage2.png"], "c4_kaggle_stage2.png")
