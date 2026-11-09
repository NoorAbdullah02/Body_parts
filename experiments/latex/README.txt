How to use this LaTeX package (Elsevier elsarticle, no BibTeX needed)

1. Overleaf: New Project > Upload Project > choose this zip. Main document: main.tex. Compiler: pdfLaTeX.
2. Where the text is (edit these to change wording):
   sections/abstract.tex, introduction.tex, related_work.tex, data.tex, methods.tex, results.tex,
   discussion.tex, limitations.tex, conclusion.tex, declarations.tex, declaration_of_generativ.tex
3. Tables: tables/*.tex (t16_literature = Table 1, t01_main = Table 2, t03_kaggle = Table 3, t11_stage2 = Table 4,
   t04_ablation = Table 5, t07_mura = Table 6, t09_transfer = Table 7). Other t*.tex are in supplementary.tex.
4. References: references.tex (hand-written reference list, 43 entries). refs.bib is only a source copy.
5. Figures: figures/*.png (13 numbered figures are used by sections/*.tex).
6. supplementary.tex compiles on its own (Supplementary Tables S1-S9, Figures S1-S2).
7. Note: if you re-run make_latex.py the sections are regenerated from MANUSCRIPT_v2.md and your manual edits will be overwritten.
