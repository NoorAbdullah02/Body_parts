# Novelty and research gap (evidence-based, with what is NOT novel)

Method: targeted web searches + opening sources (Nov 2026). I read abstracts / search summaries / one PMC page, **not** full texts, except where noted.
"Not found in my search" is NOT proof that nothing exists -- verify with Google Scholar / PubMed before claiming "first".

## 1. What already exists (so it cannot be your novelty)

| Topic | Prior work found | Consequence for you |
|---|---|---|
| Radiograph body-part classification, large scale | Eur Radiol 2025 multi-institutional (>860k radiographs, EfficientNet, 7 classes, external accuracy 98.5%; **MURA used as an external test, micro-acc 98.6%**; per its summary the outside hospital's labels contained errors) | "Body-part classifier + MURA as external test" is NOT new. Their training data are not public, so they cannot be reproduced.  **[VERIFIED in full text (CITATIONS_VERIFIED.md, third pass): MURA was an external test with every image labelled 'Extremities'; micro-accuracy 98.6 %; not comparable with 7-class MURA body-part classification]** |
| Label noise in body-part data | Langer et al. 2024, arXiv 2412.15967: 48,434 hospital radiographs, 14 regions, SSL ResNet18, 96.6 % (single) / 97.7 % (ensemble); an expert found **35 % of test-error labels wrong, 11 % out-of-domain** | Label noise in this task is documented. You may cite it to explain "Others"/rare-class errors; you may not claim to discover it. |
| DICOM `BodyPartExamined` unreliability | Güld et al. via Dratsch 2019: tag wrong in 15.3 %; BMC Med Imaging 2025 (Kiel) 20k series, recall 98.7 % | Motivation only. |
| MURA 7-class body-part with a hybrid | Shah et al. 2026 (Xception-Swin, acc 0.9643, macro-F1 0.9574; **one seed, no CI, no k-fold**; test = MURA public validation set) | Your plain Swin-T on MURA is on par. Their weaknesses are your opening. |
| Gated CNN + Swin/ViT fusion | ConvTransGFusion (ConvNeXt+Swin, dual-attention gate), HPD-Net (EffNetV2-S+Swin-T gate), CNN-ViT adaptive attention gate (brain MRI), CheX-DS (DenseNet+Swin ensemble) | A gated CNN-Transformer fusion is NOT novel. |
| Hierarchical loss | Asadi 2025 (CheXpert disease hierarchy), Wu/Tygert/LeCun 2019 (hierarchical loss can hurt on non-hierarchical problems), PadChest / RadGenome anatomy hierarchies (labels, not losses) | An anatomical-group loss for body parts was not found, but your result is negative, so it is a footnote, not a headline. |
| Threshold tuned on the evaluation set | CARE-X 2026 preprint states the same flaw and re-ran with 3-fold CV: differences <= 0.038 ("uniformly small"); nested CV is textbook | The flaw is known; the *size* depends on the setting. Do not claim a discovery, claim a measurement on this task. |
| Cross-anatomy zero-shot | Sci Rep 2026 "anatomical similarity in zero-shot learning" (abnormality detection across MURA body parts) | Different task (abnormality), not body-part identification. |

## 2. Research gaps (what I could not find) and how they map to your project

| # | Gap | Evidence level | Fills it |
|---|---|---|---|
| G1 | **No rigorous, reproducible academic benchmark on the public UNIFESP dataset**: the only academic paper I found (Krishnan & Padmavathi 2025, Procedia CS) reports accuracy only, five CNNs, no split description on the abstract page, best 86.94 %. Kaggle write-ups exist (2nd place: ResNet50/152 + heavy aug, single-label framing) but no patient-level, CI-reporting study. | medium (abstract-level) | v2 pipeline: patient-level nested protocol, CIs, official Late-Submission score |
| G2 | **Public-data-only cross-dataset body-part transfer with an explicit label-taxonomy mapping** (UNIFESP <-> MURA <-> FracAtlas). Eur Radiol 2025 used MURA externally but trained on private data; the zero-shot MURA/FracAtlas papers study fracture/abnormality, not body part. I found no study using MURA + FracAtlas for body-part identification. | medium-low (absence of evidence) | `cross_eval.py` + taxonomy map; needs FracAtlas region labels verified |
| G3 | **Quantified effect of evaluation pitfalls on multilabel radiograph body-part classification** (threshold tuned on scored fold, checkpoint selection on scored fold, empty-prediction decoding, image- vs patient-level split), with CIs. Theory is known; I found no measurement for this task. | medium-low | `ablation_report.py` + v2 ablation grid |
| G4 | **Diagnosis of gate collapse in CNN-Transformer fusion** (your 5/5-fold finding: CNN weight ~1e-4). I found no hybrid-fusion paper reporting gate-saturation analysis; they report accuracy only. | low (could exist in full texts) | `hybrid_*` variants in v2 (gate, gate+LayerNorm, fixed average, single-branch) |
| G5 | Reliability of the **public test labels**: the competition hosts re-annotated the test set offline with 4 radiologists (confirmed by their post); the corrected labels are not public. | confirmed (host post) | Error analysis of "Others"/rare classes; Late-Submission caveat |

## 3. Defensible novelty statements (ranked)

1. **Protocol + benchmark (G1+G3), the main claim.** "A leakage-controlled, statistically rigorous benchmark for multilabel radiograph body-part classification on public data, quantifying how common evaluation shortcuts inflate scores." Needs: v2 run, ablation grid, official LB score. Strongest because every number is reproducible.
2. **Cross-dataset study with taxonomy mapping (G2).** "Does a body-part model trained on one public dataset transfer to others?" Needs: MURA runs, FracAtlas zero-shot, a clinician-approved class map.
3. **Negative-result analysis (G4 + hierarchical loss).** "Gated CNN+Swin fusion collapses to the Transformer branch in 5/5 folds; a LayerNorm/entropy fix (if it works) restores balance but does not beat a single Swin." Publishable as a secondary finding; becomes primary only if the fix yields a real gain with CI.
4. Do **not** claim: novel hybrid architecture, first body-part classifier, first cross-dataset evaluation, state of the art.

Suggested title direction: *"How reliable are radiograph body-part classifiers? A leakage-controlled, cross-dataset benchmark on public data."*

## 4. Generalization & ablation matrix (all must be run for B)

Datasets
* D1 UNIFESP (22 classes, multilabel, DICOM) -- main.
* D2 MURA body-part (7 classes, single-label PNG) -- in-distribution + Shah-2026 comparison.
* D3 FracAtlas (4,083 images, hand/leg/hip/shoulder regions, CC BY 4.0, three Bangladeshi hospitals) -- **external zero-shot only**. Region column names and per-region counts NOT verified; check `dataset.csv` after download.
* D4 COVID-19 Radiography (4 classes, image-level split) -- task-generalization, caveat in the paper.
* D5 ChestX-ray14 -- only if the full dataset is used; the 5 % Kaggle sample gave 0.17-0.20 and should be dropped or moved to the supplement.

| Block | What varies | Datasets | Report |
|---|---|---|---|
| A. Recipe ablations | augmentation light/heavy; softmax-head weight `lam_ce` 0/0.5/1; sigmoid/softmax/blend head; TTA; EMA vs not | D1 (5 folds), then best setting on D2 | OOF + holdout F1, paired bootstrap CI vs reference |
| B. Decoding ablations | threshold none / global / per-class (nested) / per-class (leaky, for contrast); min-one rule | D1, D2, D4 | holdout F1, "leak gap" = leaky - nested |
| C. Model ablations | backbone family (CNN vs Swin), ensemble size 1..6, Caruana vs equal weights, seeds 1..3 | D1, D2 | rank correlation of backbones across datasets (does the winner transfer?) |
| D. Fusion ablations | gate (raw), gate + LayerNorm, fixed average, DenseNet-only, Swin-only | D1 (5 folds) | holdout F1, mean CNN-gate weight per fold |
| E. Split ablations | patient-level vs image-level folds | D1, D2 | OOF-vs-holdout gap, 5 folds |
| F. Transfer | UNIFESP->MURA, MURA->UNIFESP on shared classes; both->FracAtlas on mapped regions | D1<->D2, D3 | accuracy, macro-F1, confusion matrix, comparison with in-distribution |
| G. Label noise | error review of "Others", Hip, Forearm; effect of dropping suspected-noisy images from the *evaluation* | D1 | per-class F1 with/without, clearly labelled as post-hoc |

Compute (unmeasured guesses): D1 grid ~2-3 h per block of 6 backbones x 5 folds on T4 x2; D2 is 40k images (their earlier run: ~3.5 min per epoch per fold) so use 1-2 backbones, 5 folds, ~6-8 h; D4 ~2 h per backbone; F is minutes. All on ONE Kaggle account (terms of use).

## 5. Addendum (re-search, Nov 2026)
* Fink et al., Eur Radiol Exp 2024 (doi:10.1186/s41747-023-00417-x): projection and body-side classification in musculoskeletal radiographs with a patient-level split (accuracy about 0.975). Closest MSK-radiograph work found; it does not measure what the split changes. Cited in Related work.
* Shamim and Hussain, medRxiv 2026 (doi:10.64898/2026.10.14.26365516): calibration, uncertainty and data efficiency of chest-radiograph disease classifiers under cross-dataset shift. Preprint, different task. Cited as such.
* Searches for a UNIFESP benchmark with patient-level evaluation, calibration, label audit and transfer again found only Krishnan and Padmavathi 2025. "Not found" is not proof; wording in the paper stays hedged ("we did not find").
