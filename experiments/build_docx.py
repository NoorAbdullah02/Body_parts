"""Build submission/manuscript.docx from MANUSCRIPT_v2.md + results/paper_tables.md + figures/.  uv run --with python-docx python build_docx.py"""
import re, os
from docx import Document
from docx.shared import Pt, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import figlib
from tabmap import fixref
md = fixref(figlib.process(open("MANUSCRIPT_v2.md", encoding="utf-8").read())[0])
tables_md = open("results/paper_tables.md", encoding="utf-8").read()
doc = Document()
for s in doc.sections: s.left_margin = s.right_margin = Cm(2.2); s.top_margin = s.bottom_margin = Cm(2.2)
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11)
def runs(p, text):
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text):
        if not part: continue
        if part.startswith("**"): r = p.add_run(part[2:-2]); r.bold = True
        elif part.startswith("*") and len(part) > 2: r = p.add_run(part[1:-1]); r.italic = True
        elif part.startswith("`"): r = p.add_run(part[1:-1]); r.font.name = "Consolas"
        else: p.add_run(part)
# ---- split the manuscript: body (up to Figures) and figure list
body, _, figs = md.partition("## Figures (files in")
body = body.split("\n---\n", 1)[1] if "\n---\n" in body else body       # drop the working-status header
title = "Leakage-controlled benchmarking of radiograph body-part classification: reliability, label audit, explainability and cross-dataset transfer on public data"
h = doc.add_heading(title, 0); h.alignment = WD_ALIGN_PARAGRAPH.LEFT
p = doc.add_paragraph(); r = p.add_run("Sheikh Noor Abdullah"); r.bold = True; p.add_run("1,*, ")
r = p.add_run("A.B.M. Kabir Hossain"); r.bold = True; p.add_run("1")
a = doc.add_paragraph(); a.add_run("1 Department of Information and Communication Engineering, Bangladesh Army University of Engineering & Technology (BAUET), Qadirabad Cantonment, Natore, Bangladesh").italic = True
c = doc.add_paragraph(); c.add_run("* Corresponding author: sheikhnoorabdullah02@gmail.com. ORCID: 0009-0002-7193-4284").italic = True
body = body.split("\n## References")[0]
for line in body.split("\n"):
    line = line.rstrip()
    if not line.strip() or line.strip() == "---": continue
    if line.startswith("[[FIG|"):
        _, num, fn, cap = line.strip("[]").split("|", 3)
        files = fn.split(";"); p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for f in files:
            png = "figures/" + f
            if os.path.exists(png): p.add_run().add_picture(png, width=Inches(2.0 if len(files) > 1 else 5.2))
        q = doc.add_paragraph(); rr = q.add_run("Fig. %s. " % num); rr.bold = True; q.add_run(cap)
        continue
    if line.startswith("[[TAB|"):
        key = line.strip("[]").split("|")[1]
        mp = {"t16_literature": (16, 1), "t01_main": (1, 2), "t03_kaggle": (3, 3), "t11_stage2": (11, 4), "t04_ablation": (4, 5), "t07_mura": (7, 6), "t09_transfer": (9, 7)}
        old, new = mp[key]
        if old == 16:
            from lit_table import LIT_TITLE, LIT_HEAD, LIT_ROWS, LIT_NOTE
            title = LIT_TITLE.split(". ", 1)[1]; rows = [LIT_HEAD] + LIT_ROWS; note = LIT_NOTE
        elif old == 11:
            title = "Stage 2: out-of-fold results on all 1,738 training images and Kaggle late-submission scores"
            rows = [["ensemble", "OOF sample-F1 [95% CI]", "macro-F1", "subset acc.", "Kaggle private / public"],
                    ["s2a: 8 models, no pseudo-labels", "0.9262 [0.9092, 0.9414]", "0.8749", "0.8867", "not submitted"],
                    ["s2b: 8 models, pseudo-labels", "0.9233 [0.9058, 0.9392]", "0.8587", "0.8855", "not submitted"],
                    ["s2u: union (16 models)", "0.9279 [0.9116, 0.9428]", "0.8754", "0.8884", "0.92638 / 0.93378"],
                    ["difference s2u - s2a", "+0.0017 [-0.0010, +0.0046], p = 0.25", "", "", ""],
                    ["difference s2b - s2a", "-0.0029 [-0.0092, +0.0034], p = 0.36", "", "", ""]]
            note = "OOF numbers are not comparable with the held-out numbers; ensemble weights are fitted on the same OOF predictions, so the union estimate is optimistic. Pseudo-labels use unlabelled test images."
        else:
            blk = [b for b in re.split(r"\n(?=## Table )", tables_md) if b.startswith("## Table %d." % old)][0]
            L = blk.strip().split("\n"); title = re.sub(r"^Table \d+\.\s*", "", L[0][3:])
            rows = [[c.strip() for c in l.strip("|").split("|")] for l in L if l.startswith("|") and not re.match(r"^\|[-| ]+\|$", l)]
            note = " ".join(l.strip("_") for l in L if l.startswith("_") and l.endswith("_"))
        cp = doc.add_paragraph(); r0 = cp.add_run("Table %d. " % new); r0.bold = True; cp.add_run(title)
        t = doc.add_table(rows=len(rows), cols=len(rows[0])); t.style = "Light Grid Accent 1"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i2, row in enumerate(rows):
            for j2, c in enumerate(row[:len(rows[0])]):
                cell = t.cell(i2, j2); cell.text = ""; rr = cell.paragraphs[0].add_run(c.replace("_", " ") if i2 == 0 else c); rr.font.size = Pt(8); rr.bold = (i2 == 0)
        nn = doc.add_paragraph(); rn = nn.add_run(note); rn.italic = True; rn.font.size = Pt(8)
        continue
    if line.startswith("## "): doc.add_heading(re.sub(r"^## \d* ?", "", line) if False else line[3:], 1)
    elif line.startswith("### "): doc.add_heading(line[4:], 2)
    elif re.match(r"^\* ", line): runs(doc.add_paragraph(style="List Bullet"), line[2:])
    elif re.match(r"^\d+\. ", line): runs(doc.add_paragraph(style="List Number"), re.sub(r"^\d+\. ", "", line))
    else: runs(doc.add_paragraph(), line)
doc.add_heading("References", 1)
def deltex(t):
    t = re.sub(r"\\textit\{([^}]*)\}", r"\1", t); t = re.sub(r"\\url\{([^}]*)\}", r"\1", t)
    t = t.replace("{\\\"u}", "ü").replace("{\\~a}", "ã").replace("\\,", "").replace("--", "–").replace("\\&", "&").replace("\\%", "%")
    return re.sub(r"[{}]", "", t)
for blk in open("latex/references.tex", encoding="utf-8").read().split("\\bibitem")[1:]:
    txt = blk.split("\n", 1)[1].split("\\end{thebibliography}")[0].strip()
    pp = doc.add_paragraph(deltex(txt)); pp.paragraph_format.left_indent = Cm(0.8); pp.paragraph_format.first_line_indent = Cm(-0.8)
doc.save("submission/manuscript.docx"); print("wrote submission/manuscript.docx")
