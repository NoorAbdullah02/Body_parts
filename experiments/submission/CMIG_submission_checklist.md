# Computerized Medical Imaging and Graphics -- submission checklist
Source: the journal's official Guide for Authors (sciencedirect.com/journal/computerized-medical-imaging-and-graphics/publish/guide-for-authors), read in full on 6 Nov 2026; and its Open access information page.

## Facts read on the official pages
* Metrics shown on the journal page: **Impact Factor 5.5, CiteScore 8.8** (quartile not shown).
* Peer review: **single anonymized** (authors are not blinded; reviewers are). A minimum of one reviewer; editors decide.
* **Two publishing choices:** open access (APC **USD 2,920**, personalised at submission by country/institution, "best price promise") or **subscription: "No open access publication fee."** Choose subscription. Self-archiving of the accepted manuscript after a 12-month embargo.
* Only the **corresponding author's affiliation** decides eligibility for a publishing agreement / discounts (corresponding author: S.N. Abdullah, BAUET, Bangladesh).
* Authorship cannot be changed after acceptance: fix author list, order and corresponding author BEFORE submitting.

## Requirements and our status
| Requirement | Status |
|---|---|
| Abstract <= 250 words | done (246 incl. labels) |
| Keywords 1-7 | done (6) |
| Highlights: 3-5 bullets, <= 85 characters, separate editable file whose name includes "highlights" | content done (`highlights.md`, max 82); export as `highlights.docx` |
| Files: Word (.docx, single column) or LaTeX (.tex; two columns allowed); PDF not accepted | `submission/manuscript.docx` / `latex_overleaf.zip` |
| Tables as editable text, no vertical rules or shading, used sparingly, no duplication of text | tables are editable; **15 tables is too many**: keep Tables 1, 3, 4, 7, 9, 11 in the paper and move the rest to supplementary material |
| Figures: separate files, each cited in the text, one graph per file, 300 dpi, EPS/PDF vectors | PDF + PNG ready; two-panel figures also exported as `_A` / `_B` files |
| Captions: brief title + description | to finalise |
| CRediT statement | done (needs both authors' confirmation) |
| Declaration of competing interests (declarations tool + Word document upload) | tool step at submission |
| Funding statement (standard sentence) | done |
| **Declaration of generative AI use** (own section before the references; tool named) | done; wording uses the Elsevier template |
| Research data (Option C): deposit data/code in a repository and cite it, or explain why not | repository + Zenodo DOI still to create; statement written |
| Reference style: Elsevier Harvard (author-year), DOIs, [dataset] marker for dataset references, preprints marked, software cited | `refs.bib` uses plainnat: switch to `elsarticle-harv` when using the Elsevier class; add [dataset] tags to the dataset references |
| Acknowledgements section before the references | to write if needed |
| All references real and checked (AI-generated references can be wrong) | most checked; see `docs/CITATIONS_VERIFIED.md`; entries marked [M] in refs.bib still need a look at the publisher page |
| Sections numbered 1, 1.1, 1.2 | done |
| Preprint allowed (not prior publication) | yes |

## Done on 7 Nov 2026
* LaTeX switched to the Elsevier class (`elsarticle`, review layout, `elsarticle-harv` references) in `latex/main.tex`; 6 tables stay in the paper (1, 3, 4, 7, 9, 11) and 9 moved to `latex/supplementary.tex` (text now cites them as Supplementary Table S2, S5, ...).
* Quartile: Scopus Q1 in all five categories (SCImago); JIF 5.5 on the journal page; JCR quartile still needs the library.
* Zenodo: needs the first author to log in (see the answer in chat); not done by the assistant.
