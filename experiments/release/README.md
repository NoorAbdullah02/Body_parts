# UNIFESP body-part benchmark: code, splits and reproduction guide

Code for the manuscript *Leakage-controlled benchmarking of radiograph body-part classification: reliability, label audit, explainability and cross-dataset transfer on public data*.

## What is here
| Path | Purpose |
|---|---|
| `v2/src/` | all Python modules (data loading, training, ensembling, evaluation, ablation report, MURA/COVID/FracAtlas transfer, frozen-encoder probe, reliability and label audit, Grad-CAM) |
| `v2/build_notebook.py`, `v2/build_suite.py` | generate the Kaggle notebooks from `v2/src` (embedded-file verification) |
| `v2/*.ipynb` | generated Kaggle notebooks (one per experiment) |
| `v2/test_*.py` | unit tests (evaluation kit, ablation report, cross-eval, explain, reliability, stage 2, MURA split) |
| `make_paper_tables.py`, `make_figures.py`, `build_docx.py` | tables, figures and manuscript build from the result files |
| `results/` | printed outputs and CSVs copied from the Kaggle runs (see `RUNBOOK.md`) |

## Data (not redistributed)
**MURA: images, crops or embeddings must NOT be added to this repository** (Stanford Research Use Agreement: non-commercial, no redistribution, no derivative works). Only code, file-name split lists and aggregate results may be shared.
* UNIFESP X-Ray Body Part Classification (Kaggle `felipekitamura/unifesp-xray-bodypart-classification`, CC BY-NC-SA 4.0, doi:10.34740/KAGGLE/DSV/3399135).
* MURA (Stanford; body part read from `XR_*` folder names), COVID-19 Radiography Database, FracAtlas (Sci Data 2023, CC BY 4.0). Check each licence before reuse.

## Protocol in one paragraph
Patient-level multilabel-stratified split (iterative stratification, seed 42): 5 folds, one fold (fold 4) fixed as the held-out set (323 images, 156 patients) and a fresh 5-fold CV on the remaining 1,415 images (612 patients). Head/TTA variant, ensemble weights (Caruana, 40 iterations), per-class thresholds and the min-one rule are chosen on out-of-fold predictions only; the held-out set is used once; CIs by patient-level bootstrap.
Split fingerprints printed by the Stage 2 notebook (SHA-256): `index.csv` 04cc621c95e8a353a665c4a2014b16045a4c6a14af0b944ca9ca9a4ad68a4b8a, `splits/folds.csv` ff44019bf2799d906acbbbe4d8e48f1228f244095a21e768ff9a74a64f36f36a.

## Reproduce
```bash
python -m src.build_index                       # needs UNIFESP_DATA_ROOT
python -m src.make_splits
python -m src.make_holdout_cv_splits --holdout_fold 4
# training / ensembling: see the generated notebooks (UNIFESP_v2_ablation.ipynb etc.) and RUNBOOK.md
python make_paper_tables.py --root results --out results/paper_tables.md
python make_figures.py
```
Run time: single Swin-T fold about 6 min on a T4; the Stage 2 notebook (16 models) took 18,811 s on 2 x T4. Software: Kaggle image with PyTorch 2.11.0+cu130 (as printed by the notebooks); pin timm / albumentations / scikit-learn versions from the run logs before release (`pip freeze` output is not stored in this repository yet).

## Known limitations of this release
* Model weights are not included (3 GB per ablation suite); regenerate with the notebooks.
* The Kaggle notebooks were run from three accounts; the printed outputs in `results/` identify which run each number came from.
* The frozen-probe random-weight control gave a held-out score inconsistent with its OOF score and is excluded from the paper; the ConvNeXt `in1k` tag in `fm_probe.py` was corrected to `fb_in1k` but the corrected run has not been executed.
* Grad-CAM maps and the label-audit suspects have not been reviewed by a clinician.
