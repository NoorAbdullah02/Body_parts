# PAPER_FACTS -- every number for the paper, with its source file (copy from here, not from memory)
Protocol: patient-level multilabel-stratified split (seed 42); held-out 323 img / 156 pts; CV pool 1,415 img / 612 pts (5 folds); OOF-only decisions; patient bootstrap 2,000 (reliability 1,000); sample-mean F1 = competition metric.

## A. Data (source: results/eda, dataset page, split_hashes.json)
UNIFESP: 2,481 DICOM (1,738 train / 768 patients; 743 test / 330 patients); 22 classes; 92.4 % single-label (122 two labels, 10 three); Chest 738 ... Clavicles 9 (82x); 95 % MONOCHROME1; BodyPartExamined empty in all files; metadata-only RF acc 0.486 vs majority 0.451, macro-F1 0.059. Annotated by radiology residents (MD.ai); readers/adjudication/demographics not stated. SHA-256: index.csv 04cc621c..., folds.csv ff44019b... (results/kaggle_raw/stage2/split_hashes.json).

## B. Main result (results/paper_tables.md Table 1)
Ensemble (6 runs, OOF thresholds): sample-F1 0.9189 [0.8669, 0.9602]; macro-F1 0.8630 [0.6891, 0.8916]; micro 0.9148; subset 0.8885. 0.5 threshold: 0.9164 / 0.8537. Best single (ConvNeXt-T) 0.9123. Ensemble minus best single +0.0066 [-0.0002, +0.0142], p = 0.063.
Single runs (held-out, 0.5+min-one): Swin 0.9133, DenseNet121 0.9221, EffNetV2-S 0.9173, ConvNeXt-T 0.9123, ResNet50-heavy 0.9104, EffNet-B3 0.9172.

## C. Ablations (Table 4; results/results_ablation/*.csv)
Reference Swin-T 0.9089 [0.8576, 0.9509]. All diffs vs reference have CIs containing 0 (smallest p: heavy aug -0.0075, p = 0.088; hybrid_cnn +0.0117, p = 0.097). Leaky split: OOF-minus-held-out 0.0144 (patient-level 0.0004). Thresholds tuned on held-out: 0.9182 vs 0.9099 honest (+0.0083). Gate CNN weight: raw 0.0000, LayerNorm 0.0029. Min-one rule +0.0031 (0.9058 -> 0.9089).

## D. Kaggle (leaderboard page, Nov 2026)
Private leaderboard: #1 0.93378 (Harshit Sheoran), #2 0.92787 (JGK), #3 0.92488 (Colin Greeley), #4 0.92414, #5 0.92182; 115 teams. Ours (late, not ranked): Stage 1 0.92563 / public 0.92704; Stage 2 union (s2u) 0.92638 / public 0.93378. Both fall between #2 and #3. Gap to #1: 0.0074 (s2u). SE of one private score ~0.012 (own estimate, ~446 images). s2a and s2b were not submitted.

## E. Stage 2 (results/stage2, results/kaggle_raw/stage2; CIs by stage2_ci.py)
OOF all 1,738: s2a 0.9262 [0.9092, 0.9414] (macro 0.8749); s2b 0.9233 [0.9058, 0.9392] (0.8587); s2u 0.9279 [0.9116, 0.9428] (0.8754). s2u-s2a +0.0017 [-0.0010, +0.0046] p 0.25; s2b-s2a -0.0029 [-0.0092, +0.0034] p 0.36; s2u-s2b +0.0046 [-0.0008, +0.0102] p 0.10. Pseudo-labels: 539/743 test images (conf >= 0.9), 100 % single-label; absent: Clavicles, Forearm, Hand, Hip, Lumbar, Skull, Thigh, Thoracic; Chest 322. GPU time 18,811 s on 2xT4.

## F. Frozen foundation models (results/fm)
DINOv2-S OOF 0.8699, held-out 0.8859 [0.8327, 0.9318] (macro 0.7656); DINOv2-B 0.8676 / 0.8839 [0.8300, 0.9276]; Swin-T IN1k 0.8555 / 0.8762 [0.8248, 0.9204]. Label efficiency (DINOv2-S, OOF): 10 % 0.6435, 25 % 0.7457, 50 % 0.8235, 100 % 0.8699. Random-weight ResNet18 control: OOF 0.563 but held-out 0.095 (inconsistent, excluded). BiomedCLIP failed (HF 401); ConvNeXt tag fixed in source, not rerun.

## G. Reliability / label audit (results/reliability)
Held-out F1 of 5-model ensemble 0.9127. Top-1 acc 0.9257; ECE 0.0545 -> 0.0517 (T = 1.15, fitted on OOF). Accuracy of kept images vs coverage: 100 % 0.9257, 99 % 0.9312, 95 % 0.9414, 90 % 0.9450 [0.8997, 0.9849], 80 % 0.9730, 70 % 0.9824. OOF-chosen thresholds: target 0.95 -> held-out 0.9371 at 98.45 % coverage (missed); 0.98 -> 0.9833 at 73.99 %; 0.99 -> 0.9858 at 65.63 %. Per-class: results/reliability/per_class_holdout.csv (Table 13). Label audit: 39/1,415 (2.8 %) suspects at p >= 0.9, 33 patients; top pairs Hip->Pelvis 5, Wrist->Ankle 4, Others->Chest 3.

## H. Generalisation (Tables 7-9)
MURA body part, official split: acc 0.9747 [0.9663, 0.9818], macro-F1 0.9707 [0.9616, 0.9788]. Shah et al. 2026: acc 0.9643, macro-F1 0.9574, AUC 0.9963, kappa 0.9579 (verified at abstract level; split/seeds/CIs not). COVID-19 Radiography (image-level split): sample-F1 0.9647 [0.9595, 0.9697], macro 0.9681, n = 4,233. Zero-shot: MURA->UNIFESP 0.875 (softmax) / 0.880 (sigmoid), n = 184; UNIFESP->MURA 0.813 / 0.821, n = 12,000; MURA->FracAtlas (hand 1,281 + shoulder 101) 0.691 / 0.858, n = 1,382; UNIFESP->FracAtlas (hand/hip/leg/shoulder) 0.708 / 0.698, n = 3,685.

## I. Explainability (Table 10, results/kaggle_raw/explain)
Faithfulness gain (random minus CAM deletion): ConvNeXt-T 0.284 [0.252, 0.316]; ResNet50 0.204 [0.177, 0.231]; Swin-T 0.164 [0.133, 0.197]; rho vs random weights 0.11 / -0.314 / 0.02. 150 pairs per model, fold-0 models. Top confusions (Table 12): Others->Abdomen 6-8, Others->Pelvis 4-5, Hip->Pelvis 2, Forearm->Lower Leg 2 (all three models).

## J. Verified citation facts (CITATIONS_VERIFIED.md)
Gueld et al. SPIE 2002: 15.3 % false Body Part Examined entries (562 images). Langer 2024 (preprint): 48,434 radiographs, 14 regions, 96.6/97.7 %; 35 % of test errors wrong labels, 11 % out-of-domain. Pham 2021: 16,093 images, 5 groups, recall 0.982. Krishnan & Padmavathi 2025: 22 UNIFESP classes, EfficientNetB3 86.94 %. Wu/Tygert/LeCun PLOS ONE 2019;14(12):e0226222. FracAtlas: 4,083 images, three Bangladeshi hospitals, CC BY 4.0. COVID DB: Chowdhury 2020 (99.7 % / 97.9 %).
