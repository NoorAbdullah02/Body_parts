# Runbook: reach the best UNIFESP score (A) AND a defensible Q1 paper (B)

Goal A -- **number**: beat the Kaggle private leaderboard #1 (0.93378, Mean F1). *Not guaranteed*: the private LB uses ~60 % of the test set with
the original, partly noisy labels (the hosts say annotators found label errors), so differences of ~0.003 are inside the noise.
Goal B -- **paper**: leakage-free protocol, patient-level CIs, strong tuned baselines, external check, honest negative results.

Both use the same pipeline; B only adds more *evaluation* and *ablation* runs.

## 0. Files

| File | What | Tested how |
|---|---|---|
| `experiments/evalkit.py` | metrics, nested OOF thresholds, min-one decoding, patient bootstrap, Caruana ensemble, submission writer/validator | `test_evalkit.py` (8 tests, pass) |
| `experiments/v2/src/train_v2.py` | one run (backbone x fold): EMA, fixed schedule, **no checkpoint selection**, sigmoid + softmax heads, saves OOF/holdout/test probabilities | CPU smoke test (see status below) |
| `experiments/v2/src/run_ensemble.py` | all decisions on OOF only, holdout reported once, CIs, `submission.csv` | `test_run_ensemble.py` on synthetic fold outputs (pass) |
| `experiments/v2/UNIFESP_v2_pipeline.ipynb` | generated self-contained Kaggle notebook (`python build_notebook.py` rebuilds it; embedded files verified identical to `src/`) | JSON + every code cell parses |

## 1. Corrections to earlier advice

* **Resolution:** UNIFESP DICOMs are natively **256x256** (checked: Rows=Columns=256, 16-bit, ~132 kB each). The earlier "try 384-512" advice was wrong.
  Use 256 (CNNs) / 224 (Swin-T). Upscaling may be tried as a cheap ablation but expect ~0.
* **0.9356 / 0.9445 ensemble numbers are not reportable**: thresholds and checkpoints were chosen on the fold being scored.
  Leakage size must be re-measured with real OOF predictions (synthetic data gave only +0.003; the old pipeline had no shrinkage, so it may be larger).

## 2. Run order on Kaggle (T4 x2) -- UNIFESP v2 only

1. Kaggle -> New Notebook -> File -> Import Notebook -> `experiments/v2/UNIFESP_v2_pipeline.ipynb`.
2. Right panel: **Add Input** -> `felipekitamura/unifesp-xray-bodypart-classification`; **Accelerator** GPU T4 x2; **Internet** On.
3. First check (2 min): Run the first 3 code cells only (dataset detect, write sources, env). Expect `Images dataset: /kaggle/input/...`, `wrote src/...`, `CUDA: True | GPUs: 2`.
4. **Save Version -> Save & Run All (Commit)**. It runs in the background (tab can be closed), keeps all outputs and the 12 h limit is far above the ~2-3 h needed.
   (Estimate, not measured on GPU: 30 jobs x ~6-8 min / 2 GPUs + ~5 min setup. Heavy-aug runs may be CPU/data-loader bound on Kaggle's 4 cores.)
5. Open the finished version -> **Output** tab: `results/results.json`, `results/per_run.csv`, `submission.csv`, `logs/*.log`.
6. If a job failed, the notebook still ensembles all runs that finished all 5 folds and prints which were skipped; read `logs/<run>.log` for the traceback.
7. **Late Submission** with `submission.csv`, ONE account. Record the score in section 3.

Note on resuming: finished runs are skipped only inside the SAME live session. A new session starts with an empty `/kaggle/working`, so a failed
commit is simply re-run; for less to lose, use the 2-part route in section 7 (each part ~1 h).

What to send back for the next step: `results.json` (`holdout`, `weights`, `chosen_variant`), `per_run.csv`, the Late-Submission score, and any failing `logs/<run>.log`.

## 3. Goal A -- improvement backlog (run in this order, keep a row per experiment)

Each change is accepted only if it improves **OOF sample_mean_f1 AND the Late-Submission score is not worse** (OOF alone can overfit).

| # | Change | Why / expectation (unverified guesses) | OOF F1 | Holdout F1 | LB |
|---|---|---|---|---|---|
| 0 | baseline v2 notebook as shipped | establishes the honest number | n/a (not read) | 0.9189 [0.867, 0.960] | private 0.92563 / public 0.92704 |
| 1 | `--aug heavy` vs `light` per backbone (5 folds) | 2nd place won with heavy aug; our 1-fold ablation said light | | | |
| 2 | `--lam_ce 0` vs `0.5` vs `1.0` (softmax head weight) | ~92 % of images are single-label | | | |
| 3 | lr / epochs per backbone (ConvNeXt lr 1e-4 is the first fix) | earlier ConvNeXt 0.73 was a tuning failure | | | |
| 4 | bigger backbones: `convnext_small`, `resnet152`, `swin_base_patch4_window7_224` | more capacity; ResNet152 was in the 2nd-place solution | | | |
| 5 | seeds x2-3 for the best 3 backbones | cheapest ensemble gain | | | |
| 6 | label-noise pass on "Others" and rare classes (inspect OOF errors manually) | hosts confirmed label errors | | | |
| 7 | Stage 2: retrain on all 1,738 images (`splits/folds.csv`) for the LB submission only | +25 % data | | | |

Do **not**: tune on the held-out set, pick epochs by validation F1, or use test-set pseudo-labels for the paper's numbers.

## 4. Goal B -- paper checklist

- [ ] Main table on the held-out set: ensemble + each backbone, mean +/- CI (patient bootstrap), 5 fold-models bagged.
- [ ] Ablations (5 folds each, same split): aug light/heavy, loss (BCE vs +softmax head), TTA, threshold tuning (none / global / per-class, nested), min-one rule, ensemble size.
- [ ] Leakage experiment: image-level vs patient-level split, 5 folds, report the OOF-vs-holdout gap with CI (earlier result was noisy, std 0.03-0.06).
- [ ] Gate fusion: LayerNorm the two feature vectors before the gate (hypothesis for the collapse, untested); report collapse / no collapse either way; `fixed_avg` and single-branch ablations on 5 folds.
- [ ] Hierarchical loss: 5 folds, tune `group_weight` in {0.1, 0.3, 1.0}; report as negative result if no gain.
- [ ] MURA body-part: add DenseNet-201 / EfficientNet-B0 / ResNet-101 / VGG-19 baselines and evaluate on MURA's public validation set so the comparison with Shah et al. 2026 is like-for-like; use >=3 seeds.
- [ ] External check: train on UNIFESP, test on MURA for the shared classes (Elbow, Finger, Forearm, Hand, Shoulder, Wrist) and the reverse.
- [ ] Datasets: ChestX-ray14 only with the full dataset (or drop it); COVID-19 Radiography: say clearly the split is image-level (no patient ids).
- [ ] Per-class table, calibration (ECE), error analysis of "Others"/Hip/Forearm, Grad-CAM panel, params/FLOPs/latency.
- [ ] Late-Submission score of the final model reported next to the leaderboard, with the caveat about ~60 % of the test set and label noise.
- [ ] Verify every citation by opening the paper (the "Cheng et al. BMC Med Imaging" reference could not be found). Cite the dataset as
      Farina EMJM, Abdala N, Kitamura FC (2022), Kaggle, doi:10.34740/KAGGLE/DSV/3399135 (licence CC BY-NC-SA 4.0).
- [ ] CLAIM 2024 checklist, code + split files on GitHub/Zenodo, seeds, hardware, library versions.

## 5. Stop / claim rules

* Claim "competitive with top Kaggle solutions" if Late-Submission >= ~0.925. Claim "best" only if it exceeds 0.93378 AND you state the noise caveat.
* Claim improvements over a baseline only with a paired-bootstrap CI that excludes 0.
* If the ensemble does not beat the best single model significantly, say so; the contribution is then the protocol + benchmark, not the model.

## 6. What was actually verified (and what was not)

Verified on this laptop (CPU only, Python 3.12, torch 2.15 CPU):
* `evalkit` -- 8 unit tests pass; its `sample_mean_f1` equals the earlier `engine.py` implementation exactly; `macro_f1`/`micro_f1` equal scikit-learn.
* `build_index` -> `make_splits` -> `make_holdout_cv_splits` run on the REAL local DICOMs and reproduce the Kaggle numbers (1,738 train / 743 test images,
  768 / 330 patients, 0 patient overlap, 323-image held-out set / 156 patients).
* `train_v2` runs end-to-end on real DICOMs: a from-scratch ResNet18 at 96 px (CPU, 6-10 epochs, only a code-path/learning sanity check) learned
  (OOF sample-F1 0.25 -> 0.61; always predicting "Chest" would be ~0.42). All backbone names used in the notebook build in timm with 22+22 outputs;
  both augmentation pipelines run.
* `run_ensemble` works on those real `preds.npz` files (5 folds): OOF assembly, holdout report with patient-bootstrap CI, paired test, and a
  `submission.csv` that passes `check_submission` against the real `sample_submission.csv` (743 rows).
* A bug found by this smoke test and fixed: `cv_folds.csv` stores the fold in a column called `fold` (not `cv_fold`).

NOT verified:
* Nothing was run on a GPU or with pretrained weights; no real UNIFESP score from v2 exists yet. Every accuracy printed by the unit tests is **synthetic**.
* Whether `heavy` augmentation, `lam_ce`, EMA 0.998, lr values etc. help -- they are starting points, to be decided by the backlog in section 3.
* Kaggle-side details (GPU memory at batch 32 for the larger backbones, pip/timm versions, weight downloads) -- first real run may need small fixes;
  if a job fails, `logs/<run>.log` has the traceback and the other jobs continue.

## 7. Running as several notebooks (SAME Kaggle account only)

Kaggle's Terms of Use (kaggle.com/terms) forbid operating more than one active account, so everything below uses ONE account; the parts share its weekly GPU quota
(~30 h per the unofficial sources I found -- check your own quota page).

```
python build_notebook.py              # one notebook (default)
python build_notebook.py --parts 2    # UNIFESP_v2_part1of2, part2of2, + UNIFESP_v2_merge
```
1. Run `part1of2` and `part2of2` (each: UNIFESP dataset attached, GPU T4 x2). If your account may run two GPU sessions at once they can overlap -- check Kaggle's session limit.
2. After both finish, **Save Version** of each. Create `UNIFESP_v2_merge`, add both parts' **notebook outputs** as inputs + the UNIFESP dataset, Run all (no GPU).
3. The merge notebook rebuilds the splits and **refuses to merge** if a part used different splits (`split_hashes.json`), skips runs that lack all 5 folds, then runs `run_ensemble`.
Tested locally with simulated parts: matching hashes merge, a tampered hash is refused (exit 1), an incomplete run is skipped.

## 8. Generalization + ablation suite (all notebooks generated by `python v2/build_suite.py`)

Run them ONE AFTER ANOTHER on the same account (each needs the GPU T4 x2). Attach the inputs listed at the top of each notebook; "Save & Run All".

| Order | Notebook | Inputs to attach | Unmeasured time guess | You send me |
|---|---|---|---|---|
| 1 | `UNIFESP_v2_pipeline` (or the 2-part route) | UNIFESP dataset | 2-3 h | `results.json`, `per_run.csv`, Late-Submission score |
| 2 | `UNIFESP_v2_ablation` | UNIFESP dataset | 4-6 h (70 jobs; trim `RUNS` if needed) | `results_ablation/*` (ablation.md, CSVs, gate_weights.csv) |
| 3 | `UNIFESP_v2_mura` | `cjinny/mura-v11` | 3-4 h (+~2x with `RUN_BASELINES=True`) | `results_mura/results.json`, `shah_comparison.json` |
| 4 | `UNIFESP_v2_covid` | `tawsifurrahman/covid19-radiography-database` | 1.5-2 h | `results_covid/results.json` |
| 5 | `UNIFESP_v2_cross` | UNIFESP + MURA + a FracAtlas dataset + outputs of notebooks 1/2 and 3 | < 1 h | `transfer/summary.csv` + JSONs |

Weekly quota is shared (~30 GPU-h per the unofficial sources I found; check yours): notebooks 1+2+3+4 are roughly 10-15 wall-hours = 20-30 GPU-hours, so spread them over two weeks if needed.
Local checks done: every command the notebooks launch ran through `train_v2` in smoke mode (heavy aug, lam_ce 0, leaky folds, fusion variants, DenseNet);
`ablation_report`, `run_ensemble`, `cross_eval` (pure logic + a UNIFESP->UNIFESP integration run), the MURA official split and the leaky split passed their tests.
Not checked: anything touching the real MURA / COVID / FracAtlas files (not available here), GPU memory at batch 32 for the fusion models (two backbones), Kaggle paths.
Bugs found by these checks and fixed: wrong fold column name (`cv_fold` vs `fold`), VGG feature size, a dead `RUN_BASELINES` switch, image-cache on 40k-image PNG sets.

Related documents: `NOVELTY_AND_GAP.md` (what is and is not novel, the ablation matrix), `JOURNAL_SHORTLIST.md` (free-to-publish journals, what to verify).

## 9. Status log
* Main v2 notebook: done (30/30 jobs). Late Submission from `noorabdullah02`: private 0.92563 / public 0.92704 (see `results/stage1_main_v2.md`).
* Ablation suite: imported as **UNIFESP_v2_Ablation_Suite** in `noorabdullah02` (https://www.kaggle.com/code/noorabdullah02/unifesp-v2-ablation-suite), dataset = original
  `felipekitamura/unifesp-xray-bodypart-classification`, GPU T4 x2, Internet on, **Version 1 "v1 - 14 configs x 5 folds ablation" = Save & Run All (Commit)**, started
  (quota shown at start: 30 h/week, ~22 h left). Output to collect when finished: `results_ablation/` (ablation.md, ablation_configs.csv, ablation_decoding.csv, ablation_heads_tta.csv, gate_weights.csv).
* Next in order: mura -> covid -> cross (all in `noorabdullah02`, one notebook at a time; the multi-account route is not used because Kaggle's Terms forbid it).

## 10. Stage 2 -- pushing the Kaggle score (notebook: `UNIFESP_v2_stage2_boost.ipynb`)
Levers, all unmeasured until run (the gain is NOT guaranteed; the private score is noisy at the ~0.003 level):
1. **All labelled data** (1,738 images instead of ~1,415 in stage 1): 5 patient-level folds over everything (`splits/folds.csv`), best configs of stage 1, 2 seeds for the top three.
2. **Pseudo-labels** (competition rules: "pseudo-label is allowed", manual annotation of the test set is not): confident predictions of the 2a ensemble on the 743 unlabelled test images
   are added to every training fold (`--pseudo_csv`), then 2b retrains; `s2u` = ensemble of 2a + 2b.
3. Three submissions are written: `submission_s2a.csv`, `submission_s2b.csv`, `submission_s2u.csv`. Late-submit them (one account) and report ALL scores in the paper, not only the best.
Reference to beat: stage 1 private 0.92563 / public 0.92704; #1 0.93378; #2 0.92787.
Tested locally: no-holdout ensemble, pseudo-label selection, `train_v2 --pseudo_csv` and the notebook's 2a/2b commands in smoke mode. NOT tested: real GPU run, how many test images pass the 0.9 confidence bar.
Comparison with published numbers: `COMPARISON_TABLE.md`.

## 11. Documents and tools added for the paper
* `Q1_PLAN.md` -- claim/evidence matrix, manuscript outline, submission checklist, statistics of "can we beat #1".
* `make_paper_tables.py` (+ `v2/test_paper_tables.py`) -- builds the manuscript tables from the result files: `python make_paper_tables.py --root results --out paper_tables.md --latex`.
  Put the downloaded results in `experiments/results/` with the names listed at the top of the script (`main_results.json`, `results_ablation/`, `results_mura/`, `results_covid/`, `transfer/`, `scores.json`).
* `COMPARISON_TABLE.md`, `NOVELTY_AND_GAP.md`, `JOURNAL_SHORTLIST.md`.

## 12. Explainability (`UNIFESP_v2_explain.ipynb`, code `v2/src/explain.py`, tests `v2/test_explain.py`)
* Trains fold 0 of ConvNeXt-T, Swin-T and ResNet50-heavy (~15-25 min, UNMEASURED) and explains them on the 323 held-out images: Grad-CAM gallery, faithfulness (deletion test vs a random map on the SAME coarse grid),
  randomisation sanity check (tie-aware Spearman), top-confusion table, per-class recall.
* Checked locally: toy model whose score depends on one quadrant (CAM inside/outside 0.57/0.03; deletion CAM 0.778 < random 0.976 < wrong map 0.984), layouts of ResNet50 / ConvNeXt / EfficientNet / DenseNet / VGG (NCHW) and Swin (NHWC), a real DICOM run with a weak model.
* Bugs found and fixed while testing: tie handling in Spearman (invented a correlation), channel-layout heuristic, and an UNFAIR random baseline (pixel scatter vs blocky CAM) -> random map at the CAM's own grid.
* Not checked: real GPU run; results on well-trained models are unknown (they may fail the checks -- then report that).

## 13. CPU work done locally (no GPU needed)
* `eda.py` -> `results/eda/` (dataset_table.md, class_table.csv, eda_summary.json, dicom_tags.csv, 5 figures incl. the 22-class montage). Computed from the real DICOMs in 5 s.
  Findings: all images 256x256; 95 % MONOCHROME1; **BodyPartExamined empty in all 2,481 files** (so the tag-error rate cannot be measured here -- motivation must come from the literature);
  92.4 % single-label; imbalance 738:9 = 82x; top pairs Lumbar+Thoracic Spine (48) and Hand+Wrist (42); held-out set has 1-6 positives for 5 classes.
* `MANUSCRIPT_DRAFT.md` -- Data and Methods written from these facts, Results 5.1-5.3 from stage 1, the rest tagged [RESULT]/[VERIFY].
* Still wanted from the stage-1 output (to do calibration / per-class tables on CPU): `results/ensemble_arrays.npz`, `results/results.json`, `index.csv`, `splits/*.csv` in `experiments/results/stage1/`.
