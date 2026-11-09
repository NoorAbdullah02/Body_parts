# Stage 1 -- main v2 pipeline (UNIFESP), read from `unifesp-v2-pipeline.ipynb` (executed on Kaggle, T4 x2)

Run facts (from the notebook's printed output): 30/30 jobs `[ok]`, `failed: []`. Training jobs finished between 20:11 and 21:31 (~1.4 h wall) after ~5 min setup + 3.7 min image cache.
Splits: patient-level; held-out set 323 images / 156 patients; CV pool 5 folds (274-307 images each). Late-Submission score: **not yet obtained**.

## Held-out set (323 images, 156 patients), decisions made on OOF only

| Decoding of the 6-run Caruana ensemble | sample_mean_f1 | macro_f1 | micro_f1 | subset acc |
|---|---|---|---|---|
| 0.5 threshold | 0.9164 | 0.8537 | 0.9130 | 0.8885 |
| 0.5 + min-one | 0.9164 | 0.8537 | 0.9130 | 0.8885 |
| per-class thresholds tuned on OOF + min-one (honest) | **0.9189** [95 % CI 0.8669, 0.9602] | **0.8630** [0.6891, 0.8916] | 0.9148 | 0.8885 |

* Best single run chosen on OOF: `convnext_tiny_s256` -> holdout 0.9123 (macro-F1 0.8482).
* Ensemble - best single = +0.0066, 95 % CI [-0.0002, +0.0142], p = 0.063 (paired patient bootstrap): **not significant at 5 %**.
* Caruana weights: effnetv2_s 0.098, convnext_tiny 0.268, resnet50_heavy 0.293, efficientnet_b3_heavy 0.341, swin_tiny 0, densenet121 0.

## Per run (chosen head/TTA variant on OOF -> OOF F1 -> holdout F1 at 0.5 + min-one)

| run | chosen | OOF | holdout |
|---|---|---|---|
| swin_tiny_s224 | blend | 0.9147 | 0.9133 |
| densenet121_s256 | blend | 0.9180 | 0.9221 |
| efficientnetv2_s_s256 | blend_tta | 0.9186 | 0.9173 |
| convnext_tiny_s256 | blend_tta | 0.9234 | 0.9123 |
| resnet50_s256_heavy | blend | 0.9064 | 0.9104 |
| efficientnet_b3_s256_hv | blend_tta | 0.9133 | 0.9172 |

(OOF sigmoid-only head of resnet50_heavy was 0.876 vs softmax 0.895 vs blend 0.906: the blend of the two heads helps most with heavy augmentation.)

## What this does and does not show
* Untuned numbers agree between the old CV (ensemble @0.5: sample 0.9135, macro 0.852) and this held-out set (0.9164, 0.854) -> the two evaluations are comparable.
* The OLD "tuned" numbers (sample 0.9356, macro 0.909, tuned on the fold being scored) vs the NEW nested ones (0.9189, 0.863): the tuning gain was
  +0.022 sample / +0.057 macro in the leaky evaluation but only +0.0025 / +0.009 when thresholds are tuned on OOF and frozen. => most of the earlier
  gain was evaluation leakage (indicative, because the two sets differ; the ablation notebook measures it on one set).
* The CI is wide (323 images, rare classes with 1-6 positives): a Late-Submission score is needed before comparing with Kaggle's 0.93378.

## Kaggle Late Submission (account noorabdullah02, file `results/submission_stage1.csv`, 743 rows, validated)

| Private score (~60 % of test) | Public score |
|---|---|
| **0.92563** | 0.92704 |

Private leaderboard of the finished competition (read from the Kaggle page): #1 0.93378, #2 0.92787, #3 0.92488, #4 0.92414, #5 0.92182. A late submission is not listed on the
board, but 0.92563 sits between #2 (0.92787) and #3 (0.92488), i.e. it would rank about 3rd; the gap to #1 is 0.0081. The held-out estimate (0.9189, CI [0.867, 0.960])
contains this score. Caveats: the private score uses ~60 % of the test set with the original, partly noisy labels (hosts re-annotated offline), so differences of ~0.003
are within noise; the submission was produced by the v2 notebook run in another of the user's accounts and only SUBMITTED from noorabdullah02.
