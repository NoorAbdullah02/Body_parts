# Explainability results (notebook `shahriarkabir02/explainable`, Version 1, successful, 19m40s; numbers copied from its printed summary table)
Setup: fold-0 models (trained on ~1,141 images) explained on the 323 held-out images; 150 (image, true-and-predicted class) pairs per model; CIs are bootstrap over pairs (a patient can contribute several pairs: say so).

| model | held-out F1 (blend) | deletion CAM | random map, same grid | gain [95% CI] | rho trained vs random weights | share of pairs with abs(rho) < 0.3 |
|---|---|---|---|---|---|---|
| ConvNeXt-T | 0.9085 | 0.455 | 0.739 | 0.284 [0.252, 0.316] | 0.110 | 0.56 |
| ResNet50 (heavy aug) | 0.8972 | 0.426 | 0.630 | 0.204 [0.177, 0.231] | -0.314 | 0.38 |
| Swin-T | 0.9028 | 0.755 | 0.920 | 0.164 [0.133, 0.197] | 0.020 | 0.67 |

Reading (stated plainly):
* Faithfulness: all three gains are > 0 with CIs far above 0 -> deleting the most salient pixels first removes the class evidence much faster than deleting a random block of the same granularity (PASS).
* Randomisation sanity: ConvNeXt (rho 0.11) and Swin (0.02) maps are essentially uncorrelated with the maps of a randomly initialised network (PASS). ResNet50 is NEGATIVELY correlated (-0.31, CI [-0.355, -0.271]); maps still differ from the random-weights maps
  (PASS in the sense of not being similar), but a systematic negative correlation suggests the random network's CAM favours regions (e.g. borders/centre) that the trained network avoids -> do not describe it as "unrelated"; say "differs".
* Swin has the highest absolute deletion scores (0.755 / 0.920): its class probability is more robust to deleting half of the pixels, so its CAM-vs-random gap is the smallest of the three (0.164).
* Limits: Grad-CAM grid 7x7 to 8x8; no localisation ground truth; no radiologist review; single fold per model; galleries and confusion tables are in the notebook Output (not downloaded yet).
