# Stage 2 (shahriarkabir02/stage-2) -- run successful, 18,811 s, 16 of 16 runs complete. Copied from the printed output.

Protocol: plain patient-level 5-fold OOF on all 1,738 train images (folds.csv, NO fixed holdout) -> NOT the same protocol as the 1,415-image CV pool + holdout of the ablation/reliability runs.

| ensemble | models | OOF sample-F1 (0.5 + min-one) | macro-F1 | micro-F1 | subset acc |
|---|---|---|---|---|---|
| s2a (no pseudo-labels) | 8 | 0.9262 | 0.8749 | 0.9181 | 0.8867 |
| s2b (with pseudo-labels, "s2p_*") | 8 | 0.9233 | 0.8587 | 0.9154 | 0.8855 |
| s2u (s2a + s2b, 16 models) | 16 | 0.9279 | 0.8754 | 0.9199 | 0.8884 |

Pseudo-labels: 539 of 743 Kaggle test images kept (min_conf 0.9), all single-label; classes absent from them: Clavicles, Forearm, Hand, Hip, Lumbar Spine, Skull, Thigh, Thoracic Spine; Chest 322.
s2u weights (OOF-fitted): s2 convnext seed1 .122, resnet50 seed0 .049, resnet50 seed1 .146, effb3 seed1 .024, effv2s .220, s2p convnext seed0 .073, seed1 .049, resnet50 seed1 .049, effb3 seed0 .049, effv2s .098, densenet121 .122.
Stage-1 reference (from the notebook): private 0.92563 / public 0.92704; competition #1 = 0.93378.

Honest reading: s2b < s2a on OOF (pseudo-labelling alone did not help); s2u is +0.0017 over s2a, no CI computed -> within noise. Weights were fitted on the same OOF, so 0.9279 is optimistic. Kaggle scores come only after the USER late-submits submission_s2a/s2b/s2u.csv (files not downloaded here).

## Kaggle late-submission scores (read from the Submissions page of account noorabdullah02, 2026-11-06)
| file | private | public |
|---|---|---|
| submission_stage1.csv | 0.92563 | 0.92704 |
| submission_s2u.csv (user clicked Submit) | 0.92638 | 0.93378 |
s2u vs stage1: private +0.00075, public +0.00674. Private difference is tiny (no CI on 743 test images) -> do NOT claim improvement or ranking. s2a/s2b were not submitted.
