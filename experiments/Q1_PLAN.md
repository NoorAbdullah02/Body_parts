# Q1 plan: what the paper must contain, what is done, what is still open

Hard truth first: nobody can guarantee acceptance in a Q1 journal. This plan removes the avoidable reasons for rejection and makes every claim evidence-backed.
Target tier (free subscription route, see `JOURNAL_SHORTLIST.md`): first choice JIIM / CMIG / CBM; escalate to MedIA / Radiology: AI only if the fusion fix gives a significant gain.

## 1. Claim -> evidence matrix (do not claim anything in the "open" rows)

| # | Claim in the paper | Evidence | Status |
|---|---|---|---|
| C1 | Leakage-controlled benchmark on the public UNIFESP data (patient-level folds, held-out set, nested OOF decisions, patient-bootstrap CIs) | stage-1 notebook: held-out sample-F1 0.9189 [0.867, 0.960], macro-F1 0.863 [0.689, 0.892] | **done** |
| C2 | Competitive with the competition leaderboard | late submission private 0.92563 / public 0.92704 (between #2 0.92787 and #3 0.92488) | **done**; stage-2 may improve |
| C3 | Ensemble adds only a small, non-significant gain over the best single model | +0.0066 [-0.0002, +0.0142], p = 0.063 | **done** |
| C4 | Evaluation shortcuts inflate scores (threshold tuned on the scored fold, checkpoint selection, image-level split) | old leaky CV 0.9356 / 0.909 vs nested 0.9189 / 0.863; ablation notebook measures it on one set | **partly** (ablation running) |
| C5 | Gated CNN+Swin fusion collapses to the Transformer branch; a LayerNorm fix does / does not help | old: CNN weight ~1e-4 in 5/5 folds; ablation `hybrid_gate` vs `hybrid_gate_ln` vs `hybrid_avg` | **open** (ablation running) |
| C6 | Recipe ablations (augmentation, softmax head, seeds, backbones) | ablation notebook | **open** |
| C7 | Generalization: MURA body-part comparable with Shah 2026 but with CIs | MURA notebook (official split) | **open** (running) |
| C8 | Task generalization (COVID, image-level split caveat) | COVID notebook | **open** (running) |
| C9 | Public-data cross-dataset transfer UNIFESP <-> MURA <-> FracAtlas | cross notebook | **open** (needs the weights of the ablation and MURA runs) |
| C11 | Explainability: Grad-CAM maps are faithful (deletion test beats a random map of the same grid) and not an edge detector (randomisation sanity check); error analysis of confusions | `UNIFESP_v2_explain` (3 models, held-out set) | **open** (notebook ready) |
| C10 | Label noise limits the ceiling ("Others", rare classes) | host post (test labels re-annotated), per-class recall | **partly** (needs an error-analysis table) |

## 2. Manuscript outline
1. **Title** (suggestion): *How reliable are radiograph body-part classifiers? A leakage-controlled, cross-dataset benchmark on public data.*
2. **Abstract** (structured): problem (wrong/missing DICOM body-part labels, ~15 % tag errors per prior literature), data, protocol, headline numbers with CIs, leaderboard position, key negative findings.
3. **Introduction**: clinical need (PACS routing, cohort building), gap G1-G4 (`NOVELTY_AND_GAP.md`), contributions as C1-C9 only.
4. **Related work**: body-part classification (Pham 2021, Langer 2024, Eur Radiol 2025, Shah 2026, Krishnan 2025), hybrid CNN-Transformer fusion, hierarchical losses, leakage and evaluation (Kapoor & Narayanan 2023, Roberts 2021).
5. **Data**: UNIFESP (2,481 DICOM, 22 classes, 768 train patients; cite Farina, Abdala, Kitamura 2022, doi:10.34740/KAGGLE/DSV/3399135; licence CC BY-NC-SA 4.0), MURA, COVID-19 Radiography, FracAtlas; split tables and the label-taxonomy map.
6. **Methods**: preprocessing (MONOCHROME1 inversion, per-image min-max), backbones, two-head objective (sigmoid + soft-target softmax), EMA and fixed schedule (no checkpoint selection), augmentation, OOF-only decisions (head/TTA variant, Caruana weights, shrunk per-class thresholds, min-one rule), patient bootstrap, paired bootstrap, fusion variants.
7. **Results**: Table 1 (held-out), Table 2 (runs), Table 3 (Kaggle), Tables 4-6 (ablations, decoding/leak gap, gate), Table 7 (MURA vs Shah), Table 8 (COVID), Table 9 (transfer). `python make_paper_tables.py` builds them from the result files.
8. **Discussion**: why the ensemble does not help much, where errors concentrate ("Others", Hip, Forearm, Lower Leg), label noise, gate collapse mechanism, clinical meaning.
9. **Limitations**: single-centre data, 323-image held-out set and rare classes (1-6 positives), COVID image-level split, FracAtlas class map assumed, noisy private test labels, transductive stage-2.
10. **Conclusion**; **Declarations**: ethics (public de-identified data, licences), data/code availability (GitHub + Zenodo DOI: code, split files, seeds, library versions), competing interests, funding, AI-assistance disclosure per the journal's policy.

Explainability section (Results/Discussion): gallery (one correct + one missed image per class, CAM for the true and the wrongly predicted class), Table 10 (faithfulness + sanity check), top-confusion table; state the limits (coarse grid, no localisation labels, no radiologist review unless added).
Figures (suggested): protocol flow chart; class distribution; per-class recall bar chart with CIs; ablation forest plot (diff vs reference with CI); gate-weight bars; transfer confusion matrices; Grad-CAM panel (not yet generated in v2 -- an older notebook has it).

## 2b. Facts from the data itself (`results/eda/`)
* The DICOM *BodyPartExamined* tag is EMPTY in all 2,481 files: do not claim to measure metadata error on UNIFESP; cite the literature (15.3 %, [VERIFY]) for the motivation only.
* 92.4 % single-label, 82x class imbalance, 1-6 held-out positives for Clavicles/Skull/Forearm/Elbow/Thigh -> per-class and macro numbers need their CIs and a caveat.

## 3. Submission package checklist
- [ ] Every number in the text is copied from `paper_tables.md` / result files; no number from the old leaky notebooks except in the "leakage" comparison, labelled as such.
- [ ] CLAIM 2024 checklist filled (Radiology: AI 2024;6(4):e240300).
- [ ] All citations opened and checked (the notebook-cited "Cheng et al. BMC Med Imaging" was not found; do not cite it).
- [ ] Reproducibility: code, `splits/*.csv`, seeds, `requirements`, hardware (Kaggle T4 x2), run times.
- [ ] Journal fee check (no APC on the subscription route), scope check, 5-10 recent similar papers read, optional presubmission enquiry.
- [ ] Preprint policy checked before posting to arXiv/TechRxiv.
- [ ] Cover letter: state the three contributions and the limitations honestly; list the Kaggle late submissions made (all of them).
- [ ] Co-author contributions and the multi-account issue: all Kaggle runs must be reproducible from the released code; list which scores came from which run.

## 4. Can the Kaggle #1 score (0.93378) be beaten? -- what the numbers say
Statistics of the private score (assumptions stated; my calculation, not Kaggle's):
* Private score uses ~60 % of 743 test images = ~446 images. With our error rate 7.4 %, the standard error of one private score is about **0.012** (95 % half-width ~0.024).
* The gap to #1 is 0.0082, i.e. ~0.65 SE; for two models that disagree on 4-8 % of the images the standard error of their DIFFERENCE is ~0.010-0.013, so the gap is ~0.6-0.9 SE.
  If true skill were equal, a gap this large or larger would appear with probability ~0.2-0.27. **So #1 and our stage-1 submission are statistically indistinguishable.**
* Consequence: a submission whose true skill is +0.003 to +0.006 higher has a real (not certain) chance of landing above 0.93378 on this noisy private score. Nothing can guarantee it.
Levers, in order of expected value (all unmeasured until run): (1) all-data folds + 2 seeds (stage 2a), (2) pseudo-labels from the unlabelled test images (stage 2b; allowed by the rules),
(3) the best fusion variant / extra backbones if the ablation shows a significant gain, (4) more seeds of the top 3 backbones.
Integrity rule: report every late submission; never choose a configuration by repeatedly probing the private score; the paper's primary numbers stay those of the held-out protocol.
