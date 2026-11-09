# Comparison with published / benchmark results (only entries that can be compared; caveats stated)

"Verified" = I read the number on the source page or abstract (Nov 2026); full texts NOT read. Re-check each against the paper's results table before putting it in your manuscript.
"Pending" = a run is in progress; fill in from `results_*.json`.

## A. UNIFESP (22-class, multilabel)

| Source | Type | Metric as reported | Their number | Ours (same metric) | Comparable? |
|---|---|---|---|---|---|
| Kaggle competition, #1 Harshit Sheoran | Leaderboard (not a paper) | Mean F1 (sample-wise), private (~60 % of test, original noisy labels) | **0.93378** | **0.92563** private / 0.92704 public (late submission, v2 stage-1 ensemble) | Yes, same metric and test set; gap -0.0081 |
| Kaggle #2 (JGK; ResNet50/152, FastAI, heavy aug) | Leaderboard | same | 0.92787 | 0.92563 | Yes; gap -0.0022 (inside noise ~0.003) |
| Kaggle #3 / #4 / #5 | Leaderboard | same | 0.92488 / 0.92414 / 0.92182 | 0.92563 | Yes; ours would sit between #2 and #3 |
| Krishnan & Padmavathi 2025, *Procedia CS* (EfficientNetB3, 22 classes, Gabor enhancement) | Journal/proceedings | "accuracy" (definition, split, image count not visible on the abstract page) | **86.94 %** | subset accuracy (exact label-set match) **88.85 %** on our 323-image patient-level holdout (CI not computed for this metric) | **Partly**: closest published figure, but metric definition and split unknown -> state this caveat |

Our own protocol numbers (not comparable to the above): held-out 323 images / 156 patients, ensemble per-class thresholds tuned on OOF: sample-F1 0.9189 [0.867, 0.960], macro-F1 0.863 [0.689, 0.892].

## B. MURA body-part (7 classes: elbow, finger, forearm, hand, humerus, shoulder, wrist)

| Source | Their protocol | Their number | Ours | Comparable? |
|---|---|---|---|---|
| Shah et al. 2026, *J Imaging* 12(7):298 (Xception-Swin, one seed 42, no CI) | test = MURA public validation set (3,197 images) | acc **0.9643**, macro-F1 **0.9574** | **Pending** (`UNIFESP_v2_mura`: same test set, 5-fold bagged, with CIs). Earlier notebook (different split): acc ~0.964, macro-F1 0.958 | Yes once the official-split run finishes |
| Shah et al. baselines (same paper) | same | DenseNet-201 0.9187/0.9093; EfficientNet-B0 0.9043/0.8952; ResNet-101 0.8934/0.8831; VGG-19 0.8612/0.8498 (acc / macro-F1) | ours DenseNet-121 + Swin-T pending; set `RUN_BASELINES=True` for their exact baselines | Yes (only with the baselines run) |
| Eur Radiol 2025 multi-institutional (EfficientNet, 7 categories, private training data > 860k) | MURA used as an EXTERNAL test | micro-accuracy 98.6 % | UNIFESP->MURA zero-shot (`UNIFESP_v2_cross`) pending | No for in-distribution; partly for the cross-dataset block (different training data, different class set)  **[VERIFIED in full text (CITATIONS_VERIFIED.md, third pass): MURA was an external test with every image labelled 'Extremities'; micro-accuracy 98.6 %; not comparable with 7-class MURA body-part classification]** |

## C. COVID-19 Radiography (task generalization)

| Source | Protocol | Their number | Ours | Comparable? |
|---|---|---|---|---|
| Chowdhury et al. 2020, *IEEE Access* 8:132665 | their dataset version (423 COVID / 1,485 viral / 1,579 normal), 3-class and 2-class | 3-class accuracy 97.9 %, 2-class 99.7 % | Pending (4-class, ~21k images, image-level split) ; earlier run sample-F1 0.957 | **No** (different class set and dataset version); use only as context |

## D. Other body-part classifiers (context only, NOT comparable)

| Source | Data | Number |
|---|---|---|
| Langer et al. 2024 (arXiv 2412.15967) | 48,434 private hospital radiographs, 14 regions | 96.6 % (single) / 97.7 % (ensemble); 98.0 / 98.8 % after expert relabeling of noisy test labels |
| Pham et al. 2021 (DICOM Imaging Router, arXiv 2108.06490) | 16,093 images, 5 groups | recall 0.982 (MobileNet-V1) |
| Eur Radiol 2025 | > 860k radiographs, external sites | 98.5 % accuracy |

## How to present it
Put A and B in the main results (metric-matched rows only). Put C and D in the discussion or supplement with the "different data / classes" caveat.
Never write "better than" for a row marked Partly/No; write "higher reported accuracy under a different/unknown protocol" instead.
