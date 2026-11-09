# Paper tables (generated; do not edit by hand)

## Table 1. UNIFESP held-out performance (decisions made on out-of-fold predictions only)

| decoding | sample_mean_F1 | macro_F1 | micro_F1 | subset_acc |
|---|---|---|---|---|
| ensemble, threshold 0.5 | 0.9164 | 0.8537 | 0.9130 | 0.8885 |
| ensemble, 0.5 + min-one | 0.9164 | 0.8537 | 0.9130 | 0.8885 |
| ensemble, per-class thresholds tuned on OOF + min-one | 0.9189 [0.8669, 0.9602] | 0.8630 [0.6891, 0.8916] | 0.9148 | 0.8885 |
| best single run by OOF (convnext_tiny_s256), 0.5 + min-one | 0.9123 | 0.8482 | 0.9070 | 0.8762 |

_Held-out set: 323 images / 156 patients; 95 % CIs by patient-level bootstrap. Ensemble minus best single: +0.0066 [-0.0002, +0.0142], p = 0.063. Ensemble weights fitted on OOF predictions (Caruana selection; the other two runs, Swin-T and DenseNet121, received weight 0): EfficientNetV2-S 0.10, ConvNeXt-T 0.27, ResNet50 (heavy augmentation) 0.29, EfficientNet-B3 0.34._

## Table 2. Single runs (head/TTA variant chosen on OOF)

| run | chosen_variant | oof_chosen_variant | holdout_sample_mean_f1_0.5_min_one |
|---|---|---|---|
| swin_tiny_s224 | blend | 0.9147 | 0.9133 |
| densenet121_s256 | blend | 0.918 | 0.9221 |
| efficientnetv2_s_s256 | blend_tta | 0.9186 | 0.9173 |
| convnext_tiny_s256 | blend_tta | 0.9234 | 0.9123 |
| resnet50_s256_heavy | blend | 0.9064 | 0.9104 |
| efficientnet_b3_s256_hv | blend_tta | 0.9133 | 0.9172 |

_OOF = pooled out-of-fold sample-mean F1 of the chosen variant is in the full CSV._

## Table 3. Kaggle UNIFESP private score vs. the competition leaderboard

| entry | private_score | public_score |
|---|---|---|
| 1st place | 0.93378 | n/a |
| 2nd place (ResNet50/152, FastAI) | 0.92787 | n/a |
| ours: stage-2 union (late submission) | 0.92638 | 0.93378 |
| ours: stage-1 ensemble (late submission) | 0.92563 | 0.92704 |
| 3rd place | 0.92488 | n/a |
| 4th place | 0.92414 | n/a |
| 5th place | 0.92182 | n/a |

_Mean F1 (sample-wise); private score uses ~60 % of the test set with the original (partly noisy) labels, so differences of ~0.003 are within noise. Late submissions are not listed on the leaderboard; we report every submission we made. Public scores of other teams are not shown (n/a)._

## Table 4. Ablations (5 CV folds each, same held-out set; paired patient bootstrap vs. the reference)

| config | held-out sample-F1 [95% CI] | macro-F1 | OOF - held-out | diff vs reference [95% CI] | p |
|---|---|---|---|---|---|
| swin_tiny_s224 | 0.9089 [0.8576, 0.9509] | 0.837 | 0.0004 |  |  |
| swin_tiny_s224_heavy | 0.9013 [0.8494, 0.9440] | 0.829 | 0.0074 | -0.0075 [-0.0164, 0.0009] | 0.088 |
| swin_tiny_s224_ce0 | 0.9038 [0.8529, 0.9470] | 0.8261 | 0.0041 | -0.0051 [-0.0124, 0.0016] | 0.152 |
| swin_tiny_s224_ce1 | 0.9132 [0.8616, 0.9568] | 0.8495 | 0.0012 | 0.0043 [-0.0063, 0.0149] | 0.447 |
| swin_tiny_s224_leaky | 0.9163 [0.8644, 0.9587] | 0.856 | 0.0144 | 0.0074 [-0.0020, 0.0178] | 0.141 |
| swin_tiny_s224_seed1 | 0.9143 [0.8620, 0.9576] | 0.8533 | 0.0028 | 0.0055 [-0.0060, 0.0173] | 0.366 |
| swin_tiny_s224_seed2 | 0.9154 [0.8635, 0.9592] | 0.8585 | -0.0014 | 0.0065 [-0.0049, 0.0178] | 0.268 |
| hybrid_gate_s224 | 0.9136 [0.8609, 0.9568] | 0.8497 | -0.0 | 0.0047 [-0.0063, 0.0161] | 0.401 |
| hybrid_gate_ln_s224 | 0.9158 [0.8641, 0.9577] | 0.8536 | -0.0019 | 0.0069 [-0.0032, 0.0182] | 0.226 |
| hybrid_avg_s224 | 0.9164 [0.8651, 0.9585] | 0.8537 | -0.0004 | 0.0075 [-0.0048, 0.0201] | 0.235 |
| hybrid_cnn_s224 | 0.9205 [0.8684, 0.9613] | 0.8705 | -0.0012 | 0.0117 [-0.0016, 0.0257] | 0.097 |
| hybrid_swin_s224 | 0.9127 [0.8611, 0.9565] | 0.8467 | 0.0013 | 0.0038 [-0.0070, 0.0145] | 0.539 |
| densenet121_s256 | 0.9200 [0.8694, 0.9618] | 0.8542 | -0.0042 | 0.0111 [-0.0022, 0.0275] | 0.119 |
| convnext_tiny_s256 | 0.9117 [0.8605, 0.9546] | 0.8472 | 0.0081 | 0.0028 [-0.0084, 0.0142] | 0.674 |
| swin_3seeds_average | 0.9137 [0.8623, 0.9574] | 0.848 | 0.0012 | 0.0049 [-0.0052, 0.0148] | 0.38 |

_`OOF - held-out` large and positive = the validation score over-estimates (compare `_leaky` with the patient-level reference for the split ablation)._

## Table 5. Decoding ablation and evaluation leakage

| decoding | held-out sample-F1 [95% CI] | macro-F1 |
|---|---|---|
| 0.5, empty allowed | 0.9058 [0.8544, 0.9480] | 0.8361 |
| 0.5 + min-one | 0.9089 [0.8576, 0.9509] | 0.837 |
| global threshold 0.50 (tuned on OOF) + min-one | 0.9089 [0.8576, 0.9509] | 0.837 |
| per-class thresholds tuned on OOF (honest) + min-one | 0.9099 [0.8599, 0.9523] | 0.8413 |
| per-class thresholds tuned on HELD-OUT (leaky) + min-one | 0.9182 [0.8668, 0.9592] | 0.8565 |

_Leak gap (thresholds tuned on the held-out set minus thresholds tuned on OOF) = +0.0083._

## Table 6. Mean weight given to the CNN branch by the fusion gate (5 folds)

| config | mean | std | min | max |
|---|---|---|---|---|
| hybrid_avg_s224 | 0.5 | 0.0 | 0.5 | 0.5 |
| hybrid_cnn_s224 | 1.0 | 0.0 | 1.0 | 1.0 |
| hybrid_gate_ln_s224 | 0.0029 | 0.003 | 0.0006 | 0.008 |
| hybrid_gate_s224 | 0.0 | 0.0 | 0.0 | 0.0001 |
| hybrid_swin_s224 | 0.0 | 0.0 | 0.0 | 0.0 |

_hybrid_avg is fixed at 0.5; hybrid_cnn / hybrid_swin are 1 / 0 by construction._

## Table 7. MURA body-part classification: held-out = MURA validation set

| model | accuracy [95% CI] | macro-F1 [95% CI] |
|---|---|---|
| ours: 5-fold bagged ensemble (official MURA split) | 0.9747 [0.9663, 0.9818] | 0.9707 [0.9616, 0.9788] |
| Xception-Swin (Shah 2026) | 0.9643 (CI n/a) | 0.9574 (CI n/a) |
| DenseNet-201 | 0.9187 (CI n/a) | 0.9093 (CI n/a) |
| EfficientNet-B0 | 0.9043 (CI n/a) | 0.8952 (CI n/a) |
| ResNet-101 | 0.8934 (CI n/a) | 0.8831 (CI n/a) |
| VGG-19 | 0.8612 (CI n/a) | 0.8498 (CI n/a) |

_Shah et al. 2026 (results table in their paper; verified in the full text): same MURA validation set (3,197 images), fixed seed 42, image-level evaluation; no CI is shown for these body-part numbers._

## Table 8. COVID-19 Radiography (4 classes, image-level split)

| held-out images | sample-F1 [95% CI] | macro-F1 [95% CI] | subset acc |
|---|---|---|---|
| 4233 | 0.9647 [0.9595, 0.9697] | 0.9681 [0.9631, 0.9728] | 0.9490 |

_No patient identifiers exist in this dataset: the split is image-level and the numbers may be optimistic (state this limitation)._

## Table 9. Zero-shot cross-dataset transfer through the class map

| source | target | head | groups | n | accuracy | ci | macro_f1 |
|---|---|---|---|---|---|---|---|
| mura | unifesp | softmax | elbow,finger,forearm,hand,shoulder,wrist | 184 | 0.875 | [0.826, 0.924] | 0.8478 |
| mura | unifesp | sigmoid | elbow,finger,forearm,hand,shoulder,wrist | 184 | 0.8804 | [0.837, 0.924] | 0.856 |
| unifesp | mura | softmax | elbow,finger,forearm,hand,shoulder,wrist | 12000 | 0.8133 | [0.806, 0.820] | 0.7444 |
| unifesp | mura | sigmoid | elbow,finger,forearm,hand,shoulder,wrist | 12000 | 0.8208 | [0.814, 0.828] | 0.7481 |
| mura | fracatlas | softmax | hand,shoulder | 1382 | 0.691 | [0.667, 0.715] | 0.5606 |
| mura | fracatlas | sigmoid | hand,shoulder | 1382 | 0.8575 | [0.840, 0.875] | 0.7115 |
| unifesp | fracatlas | softmax | hand,hip,leg,shoulder | 3685 | 0.708 | [0.693, 0.722] | 0.7409 |
| unifesp | fracatlas | sigmoid | hand,hip,leg,shoulder | 3685 | 0.6982 | [0.684, 0.713] | 0.699 |

_Closed-set decision among the shared groups; CIs are image-level (the targets have no patient ids). The class map is an assumption to be checked by a clinician._

## Table 10. Explainability: faithfulness (deletion test) and randomisation sanity check of Grad-CAM on the held-out set

| model | target_layer | holdout_sample_F1(blend) | pairs | deletion_CAM | deletion_random_grid | gain [95% CI] | rho trained-vs-random | frac |rho|<0.3 |
|---|---|---|---|---|---|---|---|---|
| convnext_tiny_s256 | body.norm_pre | 0.9085 | 150 | 0.455 | 0.739 | 0.284 [0.252, 0.316] | 0.11 | 0.56 |
| resnet50_s256_heavy | body.layer4 | 0.8972 | 150 | 0.426 | 0.63 | 0.204 [0.177, 0.231] | -0.314 | 0.38 |
| swin_tiny_s224 | body.norm | 0.9028 | 150 | 0.755 | 0.92 | 0.164 [0.133, 0.197] | 0.02 | 0.67 |

_gain = random-map deletion score minus CAM deletion score (> 0 and CI above 0 = the map points at pixels the model uses); rho = Spearman correlation of the CAM with the CAM of a randomly initialised model (low = passes). Grad-CAM grids are coarse and no localisation ground truth exists: say so._
