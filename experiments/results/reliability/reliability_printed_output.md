# Reliability run (noorabdullah02/unifesp-v2-reliability, v1, CPU, 384.6 s, successful) -- printed output, copied verbatim

Ensemble of saved ablation runs: swin_tiny_s224 (3 seeds), densenet121_s256, convnext_tiny_s256; weights [0.146, 0.122, 0.293, 0.171, 0.268] (fitted on OOF).
Held-out sample-F1 (0.5, min-one) = 0.9127.

## Calibration (single-label top-1 view)
top-1 acc 0.9257 | ECE before 0.0545 -> after temperature scaling (T=1.15, fitted on OOF): 0.0517
=> already fairly calibrated; temperature scaling barely helps (do not oversell).

## Risk-coverage (held-out, patient-bootstrap CI on accuracy of kept images)
| target coverage | actual | top-1 acc kept | CI low | CI high |
|---|---|---|---|---|
| 1.00 | 1.0000 | 0.9257 | 0.8718 | 0.9696 |
| 0.99 | 0.9907 | 0.9312 | 0.8788 | 0.9747 |
| 0.95 | 0.9505 | 0.9414 | 0.8942 | 0.9801 |
| 0.90 | 0.9009 | 0.9450 | 0.8997 | 0.9849 |
| 0.80 | 0.8019 | 0.9730 | 0.9305 | 1.0000 |
| 0.70 | 0.7028 | 0.9824 | 0.9478 | 1.0000 |

Threshold chosen on OOF for a target accuracy, applied once to held-out:
| target acc (OOF) | threshold | held-out coverage | held-out acc kept | sent to human |
|---|---|---|---|---|
| 0.95 | 0.7158 | 0.9845 | 0.9371 | 0.0155 |
| 0.98 | 0.9904 | 0.7399 | 0.9833 | 0.2601 |
| 0.99 | 0.9980 | 0.6563 | 0.9858 | 0.3437 |
Note: the 0.95 target is missed on held-out (0.937) -> the OOF-chosen threshold does not transfer exactly; report as is.

## Per-class held-out (n_pos small for many classes; CIs wide)
| class | n_pos | precision | recall | recall CI | F1 | F1 CI |
|---|---|---|---|---|---|---|
| Abdomen | 15 | 0.636 | 0.933 | 0.800-1.000 | 0.757 | 0.533-0.970 |
| Ankle | 12 | 0.750 | 0.750 | 0.286-1.000 | 0.750 | 0.182-0.929 |
| Cervical Spine | 15 | 1.000 | 1.000 | 1.000-1.000 | 1.000 | 1.000-1.000 |
| Chest | 138 | 0.993 | 0.993 | 0.976-1.000 | 0.993 | 0.981-1.000 |
| Clavicles | 1 | 1.000 | 1.000 | 1.000-1.000 | 1.000 | 1.000-1.000 |
| Elbow | 3 | 1.000 | 0.667 | 0.500-1.000 | 0.800 | 0.667-1.000 |
| Feet | 14 | 1.000 | 1.000 | 1.000-1.000 | 1.000 | 1.000-1.000 |
| Finger | 4 | 0.800 | 1.000 | 1.000-1.000 | 0.889 | 0.500-1.000 |
| Forearm | 3 | 1.000 | 0.333 | 0.000-1.000 | 0.500 | 0.000-1.000 |
| Hand | 16 | 1.000 | 1.000 | 1.000-1.000 | 1.000 | 1.000-1.000 |
| Hip | 6 | 1.000 | 0.333 | 0.000-1.000 | 0.500 | 0.000-1.000 |
| Knee | 22 | 1.000 | 0.909 | 0.667-1.000 | 0.952 | 0.800-1.000 |
| Lower Leg | 4 | 0.571 | 1.000 | 1.000-1.000 | 0.727 | 0.000-1.000 |
| Lumbar Spine | 14 | 0.778 | 1.000 | 1.000-1.000 | 0.875 | 0.685-1.000 |
| Others | 22 | 1.000 | 0.409 | 0.133-0.909 | 0.581 | 0.235-0.952 |
| Pelvis | 13 | 0.650 | 1.000 | 1.000-1.000 | 0.788 | 0.571-1.000 |
| Shoulder | 5 | 1.000 | 0.600 | 0.000-1.000 | 0.750 | 0.000-1.000 |
| Sinus | 5 | 1.000 | 1.000 | 1.000-1.000 | 1.000 | 1.000-1.000 |
| Skull | 2 | 1.000 | 1.000 | 1.000-1.000 | 1.000 | 1.000-1.000 |
| Thigh | 3 | 1.000 | 1.000 | 1.000-1.000 | 1.000 | 1.000-1.000 |
| Thoracic Spine | 11 | 0.846 | 1.000 | 1.000-1.000 | 0.917 | 0.737-1.000 |
| Wrist | 26 | 0.913 | 0.808 | 0.608-1.000 | 0.857 | 0.690-0.978 |
(Hip and Pelvis rows: 5 of the label-audit suspects are "given Hip, model says Pelvis" -- consistent with the Hip recall of 0.333.)

## Label-error audit (training pool, OOF, p >= 0.9) -- SUSPECTS, NOT PROVEN ERRORS
39 suspects of 1415 images (2.8 %).
| given label | model says | n |
|---|---|---|
| Hip | Pelvis | 5 |
| Wrist | Ankle | 4 |
| Others | Chest | 3 |
| Others | Feet | 2 |
| Forearm | Elbow | 2 |
| Others | Sinus | 2 |
| Chest | Cervical Spine | 1 |
| Ankle | Wrist | 1 |
| Cervical Spine | Thoracic Spine | 1 |
| Abdomen | Pelvis | 1 |
| Abdomen | Chest | 1 |
| Hand | Feet | 1 |
| Finger | Hand | 1 |
| Elbow | Lower Leg | 1 |
| Elbow | Forearm | 1 |

Files on Kaggle output: results_reliability/ (reliability.json, per_class_holdout.csv, label_audit_suspects.csv) -- the CSVs still have to be downloaded from the Output tab (this file is the printed copy).

## Added later
* label_audit_suspects.csv (39 rows, uid/patient_id/given/model_says/p_model_class/p_given_max) transcribed from the Kaggle Output preview into this folder. Note: several patients appear more than once (983 x3; 781, 757, 837, 163 x2) -> review per patient.
* reliability.json summary (not downloaded as a file; all values are in the printed output above). Ensemble variants per run: swin_tiny_s224 sig, seed1 blend_tta, seed2 blend, densenet121_s256 blend, convnext_tiny_s256 sig.
* per_class_holdout.csv not downloaded; its content is the per-class table above.
