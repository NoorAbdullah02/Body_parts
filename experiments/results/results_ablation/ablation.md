# Ablation report

Reference: **swin_tiny_s224**. Held-out set: 323 images, 156 patients; CI = patient bootstrap (2000).
(source: printed output of Kaggle notebook noorabdullah02/unifesp-v2-ablation-suite, Version 1, 3h43m51s)

## Configurations
| config | runs | oof_sample_f1 | hold_sample_f1 | hold_ci_low | hold_ci_high | hold_macro_f1 | oof_minus_hold | fold_models_mean | fold_models_std | diff_vs_ref | diff_ci_low | diff_ci_high | p_value |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| swin_tiny_s224 | swin_tiny_s224 | 0.9093 | 0.9089 | 0.8576 | 0.9509 | 0.837 | 0.0004 | 0.9034 | 0.0062 |  |  |  |  |
| swin_tiny_s224_heavy | swin_tiny_s224_heavy | 0.9088 | 0.9013 | 0.8494 | 0.944 | 0.829 | 0.0074 | 0.8999 | 0.0028 | -0.0075 | -0.0164 | 0.0009 | 0.088 |
| swin_tiny_s224_ce0 | swin_tiny_s224_ce0 | 0.9079 | 0.9038 | 0.8529 | 0.947 | 0.8261 | 0.0041 | 0.9039 | 0.0031 | -0.0051 | -0.0124 | 0.0016 | 0.152 |
| swin_tiny_s224_ce1 | swin_tiny_s224_ce1 | 0.9144 | 0.9132 | 0.8616 | 0.9568 | 0.8495 | 0.0012 | 0.9096 | 0.0038 | 0.0043 | -0.0063 | 0.0149 | 0.447 |
| swin_tiny_s224_leaky | swin_tiny_s224_leaky | 0.9307 | 0.9163 | 0.8644 | 0.9587 | 0.856 | 0.0144 | 0.9107 | 0.0054 | 0.0074 | -0.002 | 0.0178 | 0.141 |
| swin_tiny_s224_seed1 | swin_tiny_s224_seed1 | 0.9171 | 0.9143 | 0.862 | 0.9576 | 0.8533 | 0.0028 | 0.909 | 0.0059 | 0.0055 | -0.006 | 0.0173 | 0.366 |
| swin_tiny_s224_seed2 | swin_tiny_s224_seed2 | 0.914 | 0.9154 | 0.8635 | 0.9592 | 0.8585 | -0.0014 | 0.912 | 0.0019 | 0.0065 | -0.0049 | 0.0178 | 0.268 |
| hybrid_gate_s224 | hybrid_gate_s224 | 0.9136 | 0.9136 | 0.8609 | 0.9568 | 0.8497 | -0.0 | 0.9091 | 0.0035 | 0.0047 | -0.0063 | 0.0161 | 0.401 |
| hybrid_gate_ln_s224 | hybrid_gate_ln_s224 | 0.9139 | 0.9158 | 0.8641 | 0.9577 | 0.8536 | -0.0019 | 0.9139 | 0.006 | 0.0069 | -0.0032 | 0.0182 | 0.226 |
| hybrid_avg_s224 | hybrid_avg_s224 | 0.916 | 0.9164 | 0.8651 | 0.9585 | 0.8537 | -0.0004 | 0.9119 | 0.006 | 0.0075 | -0.0048 | 0.0201 | 0.235 |
| hybrid_cnn_s224 | hybrid_cnn_s224 | 0.9193 | 0.9205 | 0.8684 | 0.9613 | 0.8705 | -0.0012 | 0.9135 | 0.0023 | 0.0117 | -0.0016 | 0.0257 | 0.097 |
| hybrid_swin_s224 | hybrid_swin_s224 | 0.914 | 0.9127 | 0.8611 | 0.9565 | 0.8467 | 0.0013 | 0.9094 | 0.0049 | 0.0038 | -0.007 | 0.0145 | 0.539 |
| densenet121_s256 | densenet121_s256 | 0.9159 | 0.92 | 0.8694 | 0.9618 | 0.8542 | -0.0042 | 0.9156 | 0.0043 | 0.0111 | -0.0022 | 0.0275 | 0.119 |
| convnext_tiny_s256 | convnext_tiny_s256 | 0.9198 | 0.9117 | 0.8605 | 0.9546 | 0.8472 | 0.0081 | 0.9058 | 0.0026 | 0.0028 | -0.0084 | 0.0142 | 0.674 |
| swin_3seeds_average | swin_tiny_s224+swin_tiny_s224_seed1+swin_tiny_s224_seed2 | 0.9149 | 0.9137 | 0.8623 | 0.9574 | 0.848 | 0.0012 | 0.9124 | 0.0052 | 0.0049 | -0.0052 | 0.0148 | 0.38 |

## Decoding (reference config)
| decoding | hold_sample_f1 | ci_low | ci_high | macro_f1 |
|---|---|---|---|---|
| 0.5, empty allowed | 0.9058 | 0.8544 | 0.948 | 0.8361 |
| 0.5 + min-one | 0.9089 | 0.8576 | 0.9509 | 0.837 |
| global threshold 0.50 (tuned on OOF) + min-one | 0.9089 | 0.8576 | 0.9509 | 0.837 |
| per-class thresholds tuned on OOF (honest) + min-one | 0.9099 | 0.8599 | 0.9523 | 0.8413 |
| per-class thresholds tuned on HELD-OUT (leaky) + min-one | 0.9182 | 0.8668 | 0.9592 | 0.8565 |

Leak gap (leaky - honest per-class thresholds) = **+0.0083**

## Head / TTA variants of `swin_tiny_s224`
| variant | oof_sample_f1 | hold_sample_f1 |
|---|---|---|
| sig | 0.9093 | 0.9089 |
| soft | 0.8963 | 0.8947 |
| blend | 0.9092 | 0.9142 |
| sig_tta | 0.9069 | 0.9094 |
| soft_tta | 0.8961 | 0.8947 |
| blend_tta | 0.9089 | 0.9142 |
