# Review pack for a clinician (radiologist / orthopaedic doctor). Estimated time: 1.5-2 h. Nothing here is a diagnosis.

## Part 1. Class map used for cross-dataset transfer (please correct it)
| Source labels | Treated as the same group as | Question for the clinician |
|---|---|---|
| UNIFESP Elbow = MURA XR_ELBOW | elbow | OK? |
| UNIFESP Finger = MURA XR_FINGER | finger | OK? |
| UNIFESP Forearm = MURA XR_FOREARM | forearm | OK? (forearm views often include wrist/elbow) |
| UNIFESP Hand = MURA XR_HAND | hand | OK? |
| UNIFESP Shoulder (+ Clavicles?) = MURA XR_SHOULDER | shoulder | Should clavicle images count as shoulder? (currently NOT) |
| UNIFESP Wrist = MURA XR_WRIST | wrist | OK? |
| MURA XR_HUMERUS | no UNIFESP counterpart (dropped) | Agree? |
| FracAtlas hand / shoulder / hip | hand / shoulder / UNIFESP Hip | OK? |
| FracAtlas leg | UNIFESP Lower Leg + Knee + Thigh | OK? (FracAtlas "leg" may also include ankle/foot) |
Please mark each row: agree / disagree / unsure, and write the correction.

## Part 2. Grad-CAM review (50 images, from the Explain notebook output: galleries of shoulder/hand/knee etc.)
For each image give ONE score for the highlighted region: 2 = on the relevant anatomy, 1 = partly, 0 = on irrelevant area (border, marker, text, background); note if a laterality marker or text is highlighted.
Return a CSV: image_id, model, class, score, comment. The paper reports the share of 2s with a 95 % CI and the reviewer's role (no diagnosis, anonymised images). Ethics: public de-identified data only.

## Part 3. Suspected label errors (from `results_reliability/label_audit_suspects.csv`, top 50)
For each flagged training image: is the given label correct? yes / no (and the correct body part) / unreadable. The paper may then report "x of 50 flagged images were confirmed label errors" (a precision estimate for the audit), never a total error rate.

## Reporting
Name the clinician as a co-author or in the acknowledgements as their institution requires; state "single reviewer" (no inter-rater agreement) unless two reviewers score a subset (report Cohen's kappa on 20 shared images).
