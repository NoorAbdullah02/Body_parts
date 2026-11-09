# Novelty and gaps, round 2 (Nov 2026) -- adds to NOVELTY_AND_GAP.md

Method: 7 web searches + 2 page fetches (abstract level only; no full texts read). "I did not find it" is NOT proof that it does not exist: check Google Scholar / PubMed before writing "first".

## A. New facts from this round (each with its source)

| Finding | Source | Use in paper |
|---|---|---|
| Patient-level leakage inflation is documented in other tasks: leukaemia (random split +~0.04 AUROC), kidney-stone CT (acc 0.991 vs 0.905), colposcopy (AUC 0.91-0.93 vs ~0.5) | arXiv 2606.24944; Research Square rs-10604877; NeurIPS 2026 colposcopy paper | Cite as motivation. Our contribution = the first measurement on body-part classification (OOF 0.9307 image-level vs 0.9093 patient-level, gap 0.0144; threshold-tuning gap 0.0083). |
| No threshold-tuning-specific quantification found (mammography feature-selection leakage is the closest) | search result | Supports G3 (medium-low evidence). |
| Radiograph foundation models exist: SKELEX (1.2 M musculoskeletal radiographs, 12 diagnostic tasks), XR-0 (1.15 M, 20 tasks, private data); neither abstract lists body-part identification as a task | arXiv 2602.03076, 2509.12146 | Gap G7 below. |
| DICOM Imaging Router (Pham 2021): 5 coarse groups, 16,093 images, recall 0.982; authors say routing needs near-perfect accuracy; no OOD rejection or calibration evaluated in the summary I saw | arXiv 2108.06490 | Gap G8 below. |
| Only academic UNIFESP paper found is still Krishnan & Padmavathi 2025 (acc 86.94 %, EfficientNetB3) | Procedia CS 10.1016/j.procs.2025.03.182 | G1 stands. |
| Search for hand/wrist/forearm multi-label body-part work returned only chest-finding multilabel papers (AnaXNet) | search | G6 below. |

## B. New gaps, with evidence level and whether I can fill them

| # | Gap | Evidence | Can be filled with data already in hand? |
|---|---|---|---|
| G6 | **22-class multilabel formulation** (7.6 % of UNIFESP images carry >1 label, e.g. hand+wrist). Prior body-part work uses 5-16 single-label groups. | medium-low | YES: report subset accuracy, per-combination F1, and the min-one decoding effect (Table 5: +0.0031 F1 [0.9058 -> 0.9089]) |
| G7 | **Frozen foundation-model baselines for body-part identification** (DINOv2 / BiomedCLIP / SKELEX / XR-0 linear probe vs fine-tuned CNN/Swin). I found no paper that reports this on UNIFESP or MURA body-part labels. | low (abstract-level search) | NOT yet. Needs a run: frozen DINOv2 + linear probe on UNIFESP (2.5 k images, T4 ~10 min or CPU). Weights need internet. |
| G8 | **Reliability for routing: calibration, selective prediction (risk-coverage), OOD rejection.** Router papers demand ~100 % accuracy but I found none evaluating abstention. | low-medium | YES, on CPU, once `results/stage1/ensemble_arrays.npz` is placed: ECE, reliability diagram, accuracy at coverage 90/95/99 %, "Others" as the OOD class. |
| G9 | **Label-error audit of the training set** using out-of-fold predictions (confident-learning style), producing a list of suspected mislabelled images for expert review. Hosts re-annotated test labels offline, training noise unstudied in the sources I saw. | low | YES, on CPU, from OOF probabilities (same file). Needs a clinician to confirm a sample (e.g. 50 flagged images). |
| G10 | **Metadata-shortcut audit.** DICOM BodyPartExamined is empty in all 2,481 files; do the other tags (photometric, bits stored, size) leak the label? | measured | DONE, see C. |
| G11 | **Data-scale curve**: how F1 changes with 25/50/75/100 % of training patients (shows the ceiling is data/label-limited, not model-limited, which would also explain why no architecture wins). | not searched | Needs a run (cheap: one backbone, 4 sizes x 5 folds, ~1 h). |
| G12 | **Ceiling analysis**: per-class recall with CIs and which classes are limited by label noise vs rarity (Others, Hip, Forearm, Lower Leg; 1-6 positives in several classes). | partly done in `COMPARISON_TABLE.md` | YES, from `ensemble_arrays.npz` + class table. |

## C. Result of the one check I could run now (CPU, no new GPU run)

Metadata-only shortcut test: random forest on (photometric interpretation, BitsStored, rows, cols), patient-grouped 5-fold CV, single-label UNIFESP train images (1,606 of 1,738).
* Rows/cols are 256 x 256 for every image (no information), BitsStored has 4 values, MONOCHROME1 in 1,527 of 1,606.
* Accuracy 0.486 vs majority-class baseline 0.451; macro-F1 0.059.
* **Conclusion: acquisition metadata carry almost no label information; the classifier is not exploiting a DICOM-tag shortcut.** (A small, honest "negative" robustness result worth one sentence in the Data section; it does not exclude other shortcuts such as image borders or markers.)
* Caveat: only tags I extracted in `dicom_tags.csv`; I did not test pixel-level shortcuts (e.g. laterality markers).

## D. How this changes the novelty statement (ranked, still no "first/SOTA" claims)

1. **Protocol + measured inflation** on a 22-class multilabel body-part task (G1, G3, G6): strongest, already evidenced.
2. **Reliability for routing** (G8): calibration + selective prediction + OOD "Others". New angle, cheap, and it answers the clinical use case directly. Recommended next analysis.
3. **Cross-dataset transfer with explicit taxonomy map** (G2): needs MURA run.
4. **Foundation-model baseline** (G7) and **label-error audit** (G9): would raise novelty most but need an extra run / clinician; decide after Stage 2.
5. **Negative findings**: gate collapse (hybrid_gate CNN weight 0.0000, hybrid_gate_ln 0.0029; no significant gain from any fusion), no metadata shortcut.

## E. What I could NOT verify
* Whether SKELEX/XR-0/other recent models report body-part identification inside the full texts.
* The three leakage papers were read as search summaries only.
* Everything labelled "low" must be re-searched on Google Scholar before the words "to our knowledge" are used.

## Sources
https://www.nature.com/articles/s41598-026-37516-9 ; https://arxiv.org/pdf/2602.03076 ; https://arxiv.org/pdf/2509.12146 ; https://arxiv.org/pdf/2108.06490 ; https://arxiv.org/pdf/2606.24944 ; https://www.researchsquare.com/article/rs-10604877 ; https://neurips.cc/virtual/2026/180732 ; https://arxiv.org/pdf/2105.09937 ; https://arxiv.org/pdf/2312.02366
