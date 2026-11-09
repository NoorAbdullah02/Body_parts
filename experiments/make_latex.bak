"""Build latex/ (Overleaf-ready): tables/*.tex, figures/, sections/*.tex (from MANUSCRIPT_v2.md), refs.bib, main.tex.
uv run --with pandas --with numpy python make_latex.py"""
import re, os, shutil, json, glob
import pandas as pd, numpy as np
L = "latex"; [os.makedirs(f"{L}/{d}", exist_ok=True) for d in ("tables", "figures", "sections")]
ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
def esc(s): return "".join(ESC.get(c, c) for c in s)
MAINMAP = {1: 1, 3: 2, 11: 3, 4: 4, 7: 5, 9: 6}
SUPPMAP = {2: 1, 5: 2, 6: 3, 8: 4, 10: 5, 12: 6, 13: 7, 14: 8, 15: 9}
MAIN = set(MAINMAP)
def fixref(s):
    lab = lambda n: ("Table %d" % MAINMAP[n]) if n in MAINMAP else ("Supplementary Table S%d" % SUPPMAP[n])
    def sub(mm):
        a, sep, b = int(mm[2]), mm[3], mm[4]
        return lab(a) if not b else "%s %s %s" % (lab(a), sep, lab(int(b)))
    return re.sub(r"(Supplementary )?Tables? (\d+)(?: (and|to) (\d+))?", sub, s)
URL = re.compile(r"https?://[^\s)\]]+[^\s).,;\]]")
def tex(s):
    s = fixref(s)
    urls = []
    def keep(mm): urls.append(mm[0]); return "URLPLACEHOLDER%dX" % (len(urls) - 1)
    s = URL.sub(keep, s)
    r = _tex(s)
    for i, u in enumerate(urls): r = r.replace("URLPLACEHOLDER%dX" % i, "\\url{%s}" % u)
    return r
def _tex(s):
    s = esc(s)
    s = s.replace("->", r"$\rightarrow$").replace(">=", r"$\geq$").replace("<=", r"$\leq$").replace("+-", r"$\pm$").replace(" x ", r" $\times$ ")
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s); s = re.sub(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])", r"\\emph{\1}", s)
    s = re.sub(r"`([^`]+)`", r"\\texttt{\1}", s)
    s = re.sub(r"\[((?:VERIFY|FILL)[^\]]*)\]", r"\\textcolor{red}{[\1]}", s)
    return s
# ------------------------------------------------------------------ tables
def md_tables(path):
    out = []
    for block in re.split(r"\n(?=## Table )", open(path, encoding="utf-8").read()):
        if not block.startswith("## Table"): continue
        lines = block.strip().split("\n"); title = lines[0][3:]
        rows = [[c.strip() for c in l.strip("|").split("|")] for l in lines if l.startswith("|") and not re.match(r"^\|[-| ]+\|$", l)]
        note = " ".join(l.strip("_") for l in lines if l.startswith("_") and l.endswith("_"))
        out.append((title, rows, note))
    return out
def to_tex(title, rows, note, label, fit=True):
    n = len(rows[0]); spec = "l" + "".join("c" for _ in range(n - 1))
    HM = {"sample_mean_F1": "sample-mean F1", "macro_F1": "macro-F1", "micro_F1": "micro-F1", "macro_f1": "macro-F1", "subset_acc": "subset acc.", "ci": "95 % CI", "private_score": "private score", "public_score": "public score"}
    hd = [HM.get(h, h.replace("_", " ")) for h in rows[0]]
    rows = [rows[0]] + [[("0.0000" if c in ("-0.0", "-0.0000") else c) for c in r] for r in rows[1:]]
    body = "\\toprule\n" + " & ".join(r"\textbf{%s}" % esc(h) for h in hd) + " \\\\\n\\midrule\n"
    for r in rows[1:]: body += " & ".join(esc(c).replace("->", r"$\rightarrow$") for c in (r + [""] * n)[:n]) + " \\\\\n"
    body += "\\bottomrule\n"
    tab = "\\begin{tabular}{%s}\n%s\\end{tabular}" % (spec, body)
    if fit: tab = "\\resizebox{\\textwidth}{!}{%s}" % tab
    cap = re.sub(r"^Table \d+\.\s*", "", title); num = int(re.match(r"Table (\d+)", title)[1])
    new = MAINMAP.get(num) or SUPPMAP[num]
    return "\\setcounter{table}{%d}\n\\Needspace*{9\\baselineskip}\n\\begin{center}\\begin{minipage}{\\textwidth}\\small\n\\captionof{table}{%s}\\label{%s}\n%s\n\\par\\smallskip{\\footnotesize %s}\n\\end{minipage}\\end{center}\n" % (new - 1, tex(cap), label, tab, tex(note))
names = ["main", "runs", "kaggle", "ablation", "decoding", "gate", "mura", "covid", "transfer", "explain"]
for i, (t, rows, note) in enumerate(md_tables("results/paper_tables.md")):
    open(f"{L}/tables/t{i+1:02d}_{names[i]}.tex", "w", encoding="utf-8").write(to_tex(t, rows, note, f"tab:{names[i]}"))
# extra tables ------------------------------------------------------
ci = json.load(open("results/stage2/stage2_ci.json"))
rows = [["ensemble", "OOF sample-F1 [95% CI]", "macro-F1", "subset acc.", "Kaggle private / public"],
        ["s2a: 8 models, no pseudo-labels", "0.9262 [0.9092, 0.9414]", "0.8749", "0.8867", "not submitted"],
        ["s2b: 8 models, pseudo-labels", "0.9233 [0.9058, 0.9392]", "0.8587", "0.8855", "not submitted"],
        ["s2u: union of s2a and s2b (16 models)", "0.9279 [0.9116, 0.9428]", "0.8754", "0.8884", "0.92638 / 0.93378"],
        ["difference s2u - s2a", "+0.0017 [-0.0010, +0.0046], p = 0.25", "", "", ""],
        ["difference s2b - s2a", "-0.0029 [-0.0092, +0.0034], p = 0.36", "", "", ""]]
open(f"{L}/tables/t11_stage2.tex", "w", encoding="utf-8").write(to_tex("Table 11. Stage 2: out-of-fold results on all 1,738 training images and Kaggle late-submission scores", rows,
    "OOF numbers are not comparable with the held-out numbers of Table 1 (different protocol: no fixed held-out set). Ensemble weights are fitted on the same OOF predictions, so the union estimate is optimistic. CIs: patient-level bootstrap (2,000 resamples); pseudo-labels use unlabelled test images (transductive).", "tab:stage2"))
cf = pd.concat([pd.read_csv(f"results/kaggle_raw/explain/{m}_top_confusions.csv").assign(model=m) for m in ("convnext_tiny_s256", "resnet50_s256_heavy", "swin_tiny_s224")])
pv = cf.pivot_table(index=["true_class_missed", "predicted_instead"], columns="model", values="count", fill_value=0).reset_index()
pv["total"] = pv[["convnext_tiny_s256", "resnet50_s256_heavy", "swin_tiny_s224"]].sum(1); pv = pv.sort_values("total", ascending=False).head(10)
rows = [["true class (missed)", "predicted instead", "ConvNeXt-T", "ResNet50", "Swin-T"]] + [[r.true_class_missed, r.predicted_instead, str(int(r.convnext_tiny_s256)), str(int(r.resnet50_s256_heavy)), str(int(r.swin_tiny_s224))] for r in pv.itertuples()]
open(f"{L}/tables/t12_confusions.tex", "w", encoding="utf-8").write(to_tex("Table 12. Most frequent held-out confusions (number of missed images, per model; fold-0 models)", rows, "'Others' is missed mostly as Abdomen or Pelvis by all three models; Hip is missed as Pelvis. Source: top_confusions.csv of the explainability notebook.", "tab:confusions"))
pc = pd.read_csv("results/reliability/per_class_holdout.csv").sort_values("recall")
rows = [["class", "n positives", "precision", "recall [95% CI]", "F1 [95% CI]"]] + [[r["class"], str(r.n_pos), f"{r.precision:.3f}", f"{r.recall:.3f} [{r.recall_ci_low:.3f}, {r.recall_ci_high:.3f}]", f"{r.f1:.3f} [{r.f1_ci_low:.3f}, {r.f1_ci_high:.3f}]"] for r in pc.itertuples() if False] if False else \
       [["class", "n positives", "precision", "recall [95% CI]", "F1 [95% CI]"]] + [[r["class"], str(int(r["n_pos"])), f"{r['precision']:.3f}", f"{r['recall']:.3f} [{r['recall_ci_low']:.3f}, {r['recall_ci_high']:.3f}]", f"{r['f1']:.3f} [{r['f1_ci_low']:.3f}, {r['f1_ci_high']:.3f}]"] for _, r in pc.iterrows()]
open(f"{L}/tables/t13_per_class.tex", "w", encoding="utf-8").write(to_tex("Table 13. Per-class held-out performance of the five-model ensemble (threshold 0.5, min-one rule)", rows, "Patient-level bootstrap CIs (1,000 resamples); classes with 1-6 positives have uninformative intervals.", "tab:perclass"))
au = pd.read_csv("results/reliability/label_audit_suspects_kaggle.csv"); ac = au.groupby(["given", "model_says"]).size().sort_values(ascending=False).reset_index(name="n")
rows = [["given label", "model says (p >= 0.9)", "n"]] + [[r.given, r.model_says, str(r.n)] for r in ac.itertuples()]
open(f"{L}/tables/t14_label_audit.tex", "w", encoding="utf-8").write(to_tex("Table 14. Suspected label errors in the training pool (out-of-fold, 39 of 1,415 images, 2.8 %)", rows, "Suspects only: the model may be wrong; none has been reviewed by an expert. The 39 images come from 33 patients.", "tab:audit"))
rows = [["property", "value"], ["images / patients (train)", "1,738 / 768"], ["test images / patients (labels hidden)", "743 / 330"], ["classes", "22 (20 body parts + Others), multilabel"], ["single-label images", "92.4 %"], ["image size", "256 x 256, DICOM; 95 % MONOCHROME1"],
        ["BodyPartExamined tag", "empty in all 2,481 files"], ["annotators", "radiology residents (four named in the dataset page), MD.ai platform; number of readers per image not stated"], ["licence", "CC BY-NC-SA 4.0"], ["held-out set", "323 images / 156 patients"], ["CV pool", "1,415 images / 612 patients, 5 folds"]]
open(f"{L}/tables/t15_dataset.tex", "w", encoding="utf-8").write(to_tex("Table 15. UNIFESP dataset and split summary", rows, "From the dataset page and our own analysis (results/eda).", "tab:dataset", fit=False))
# ------------------------------------------------------------------ figures
for f in glob.glob("figures/*.png"): shutil.copy(f, f"{L}/figures/")
for f in glob.glob("figures/gradcam_gallery_*.png"): shutil.copy(f, f"{L}/figures/")
# ------------------------------------------------------------------ sections
import figlib
md = open("MANUSCRIPT_v2.md", encoding="utf-8").read(); md = figlib.process(md)[0]; md = md.split("\n---\n", 1)[1]; md = md.split("## Figures (files in")[0]
parts = re.split(r"\n(?=## )", md); fnames = {}
for p in parts:
    m = re.match(r"## (.+)", p)
    if not m: continue
    head = m[1]; key = re.sub(r"[^a-z]+", "_", re.sub(r"^\d+\s*", "", head.lower())).strip("_")[:24]
    body = p.split("\n", 1)[1] if "\n" in p else ""
    out = []
    for line in body.split("\n"):
        s = line.rstrip()
        if not s.strip() or s.strip() == "---": out.append(""); continue
        if s.startswith("[[FIG|"):
            _, num, fn, cap = s.strip("[]").split("|", 3)
            W = {"fig1_protocol": 1.0, "fig10_heldout_overview": 0.8, "fig8_kaggle": 0.85, "r4_stage2": 0.7, "fig7_leakage_and_gate": 1.0, "fig2_ablation_forest": 0.9,
                 "fig5_foundation_models": 1.0, "fig4_reliability": 0.8, "r2_f1_vs_npos": 0.8, "fig3_per_class_recall": 0.85, "r3_label_audit": 0.8, "r5_confusions": 0.85,
                 "fig9_mura": 0.85, "fig6_transfer": 0.9, "d1_examples": 1.0, "d2_classes_labels": 1.0, "d3_label_pairs": 0.8, "d4_split_composition": 0.85, "m1_architecture": 1.0, "c1_pairs_split": 1.0, "c2_audit_confusions": 1.0, "c3_mura_transfer": 1.0, "c4_kaggle_stage2": 0.8}
            files = fn.split(";")
            if len(files) > 1:
                body = "\\hfill".join("\\includegraphics[width=0.325\\textwidth]{%s}" % f for f in files)
            else:
                body = "\\includegraphics[width=%s\\textwidth]{%s}" % (W.get(fn.rsplit(".", 1)[0], 0.7), fn)
            pl = "t"
            out.append("\\begin{figure}[%s]\\centering%s\\caption{%s}\\label{fig:%s}\\end{figure}\\suppressfloats[t]" % (pl, body, tex(cap), num)); continue
        if s.startswith("[[TAB|"):
            out.append("\\input{tables/%s}" % s.strip("[]").split("|")[1]); continue
        if s.startswith("### "): out.append("\\subsection{%s}" % tex(re.sub(r"^\d+\.\d+\s*", "", s[4:]))); continue
        if s.startswith("* ") or re.match(r"^\d+\. ", s):
            out.append("\\begin{itemize}\\item " + tex(re.sub(r"^(\* |\d+\. )", "", s)) + "\\end{itemize}"); continue
        out.append(tex(s))
    txt = "\n".join(out).replace("\\end{itemize}\n\\begin{itemize}", "")
    if key == "abstract": txt = "\n".join(l for l in txt.split("\n") if not l.startswith("\\textbf{Keywords"))
    open(f"{L}/sections/{key}.tex", "w", encoding="utf-8").write(("\\section{%s}\n" % tex(re.sub(r"^\d+\s*", "", head)) if key not in ("abstract",) else "") + txt + "\n"); fnames[key] = head
json.dump(list(fnames), open(f"{L}/sections/_index.json", "w"))
print("sections:", list(fnames))
