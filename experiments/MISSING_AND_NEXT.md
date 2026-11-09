# What is still missing (Nov 2026) -- honest list
## Cannot be produced by me / needs you
1. Author names, affiliations, ORCID, funding, competing interests, CRediT contributions, ethics wording, AI-use disclosure (title_page.md, cover_letter.md).
2. Official journal pages: confirm the subscription route has no author fee and the current JCR quartile (JOURNAL_SHORTLIST.md; only aggregator sites were reachable).
3. Full-text checks: Shah et al. (split, seeds, CIs), Krishnan & Padmavathi (metric/split), Mitsuyama et al. (do not cite the MURA claim), Langer (preprint), author lists marked [M] in refs.bib, Kaggle dataset DOI author list.
4. Pinned library versions (pip freeze) for the release; GitHub/Zenodo upload.
5. Licence wording for MURA and COVID-19 Radiography (no licence text was found).
## Technical gaps
6. fm_probe rerun with the corrected ConvNeXt tag (source fixed, not executed); BiomedCLIP download failed; random-weight control inconsistent (excluded).
7. FracAtlas transfer JSONs (confusion matrices) not downloaded (notebook private); only the summary numbers exist.
8. Ablation-suite predictions (preds.npz) not on disk, so no new per-class CIs for the ablation runs.
9. Clinician review (class maps, Grad-CAM ratings, label-audit suspects): skipped by decision; stated as limitation.

## Update 6 Nov 2026
Resolved: CMIG fee/peer review/limits (official page), Shah split/seed/CI and baseline numbers, Pham and Langer numbers, MURA access approved, corresponding author = S.N. Abdullah, ORCID of the first author.
Still open: JCR quartile (library); Kabir Hossain's ORCID; both authors' confirmation of CRediT/competing interests; Krishnan PDF (split/metric); Northcutt reference check; MURA mirror hash comparison; Zenodo DOI and public repo; Overleaf compile test; cut tables from 15 to about 6 for CMIG.
