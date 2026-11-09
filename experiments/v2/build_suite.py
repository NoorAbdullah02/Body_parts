"""
Generate the generalization / ablation notebooks (all self-contained, rebuilt from src/):

    python build_suite.py                  # all four
    python build_suite.py --which mura     # ablation | mura | covid | cross

    UNIFESP_v2_ablation.ipynb   UNIFESP: recipe, loss-head, split-type, seed and FUSION (gate collapse) ablations + decoding ablation
    UNIFESP_v2_mura.ipynb       MURA body-part with the OFFICIAL protocol (train -> held-out = MURA valid), comparable with Shah 2026
    UNIFESP_v2_covid.ipynb      COVID-19 Radiography (task generalization; image-level split, say so in the paper)
    UNIFESP_v2_cross.ipynb      zero-shot transfer UNIFESP <-> MURA and -> FracAtlas through an explicit class map

Everything runs on ONE Kaggle account (Kaggle's terms forbid more than one account per person), one notebook after the other.
"""
import argparse
import ast
import json
import os

import build_notebook as B

HERE, SRC = B.HERE, B.SRC
SUITE_FILES = B.SRC_FILES + ["fusion", "cross_eval", "ablation_report", "make_leaky_splits_v2", "make_mura_official_splits",
                             "build_index_mura", "build_index_covidcxr", "build_index_chestxray14", "make_pseudo_labels", "explain", "fm_probe", "reliability"]
md, code, writer_cell = B.md, B.code, B.writer_cell

DETECT_MURA = code("""
import os
images_root = None
for cur, dirs, files in os.walk("/kaggle/input"):
    if "XR_ELBOW" in dirs or any(d.startswith("XR_") for d in dirs):
        images_root = os.path.dirname(cur)          # cur is .../train or .../valid -> MURA-v1.1 root
        break
assert images_root is not None, "Attach cjinny/mura-v11 (Notebook Settings -> Data)."
print("Images dataset:", images_root)
IMAGES_DATASET = images_root
""")

DETECT_COVID = code("""
import os
CLASS_DIRS = {"covid", "lung_opacity", "normal", "viral pneumonia", "viral_pneumonia"}
images_root = None
for cur, dirs, files in os.walk("/kaggle/input"):
    if {d.lower() for d in dirs} & CLASS_DIRS:
        images_root = cur
        break
assert images_root is not None, "Attach tawsifurrahman/covid19-radiography-database (Notebook Settings -> Data)."
print("Images dataset:", images_root)
IMAGES_DATASET = images_root
""")


def env_cell(dataset, cache=True):
    lines = ['import sys, os', 'sys.path.insert(0, "/kaggle/working")',
             'os.environ["UNIFESP_DATA_ROOT"] = IMAGES_DATASET', 'os.environ["UNIFESP_INDEX_ROOT"] = "/kaggle/working"',
             'os.environ["UNIFESP_OUTPUT_DIR"] = "/kaggle/working/runs"',
             f'os.environ["UNIFESP_CACHE_DIR"] = "{"/kaggle/working/image_cache" if cache else "none"}"']
    if dataset != "unifesp":
        lines.append(f'os.environ["UNIFESP_DATASET"] = "{dataset}"')
    lines += ['!pip install -q timm albumentations iterative-stratification pydicom', 'import torch',
              'print("CUDA:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count(), "| torch", torch.__version__)']
    return code("\n".join(lines))


def jobs_cell(runs, epochs, holdout=True, runs_expr=None, folds=(0, 1, 2, 3, 4)):
    runs_src = runs_expr if runs_expr else json.dumps(runs, indent=1)      # runs_expr: a Python expression evaluated in the notebook
    return code(f"""
import subprocess, itertools, threading, queue, os, time

EPOCHS, BATCH = {epochs}, 32
RUNS = {runs_src}
FOLDS = {list(folds)}
OUT = "/kaggle/working/runs"
CV = "/kaggle/working/splits/cv_folds.csv"
HOLD = "/kaggle/working/splits/holdout_uids.csv"

jobs = []
for r, f in itertools.product(RUNS, FOLDS):          # config-major order: a cut-off still leaves COMPLETE configs
    name = f"{{r['prefix']}}_f{{f}}"
    if os.path.exists(f"{{OUT}}/{{name}}/preds.npz"):
        continue
    cmd = ["python", "-m", "src.train_v2", "--backbone", r["bb"], "--img_size", str(r["size"]), "--aug", r.get("aug", "light"),
           "--lr", str(r["lr"]), "--epochs", str(r.get("epochs", EPOCHS)), "--batch_size", str(BATCH), "--fold", str(f),
           "--folds_csv", r.get("folds_csv", CV), "--fold_col", "fold", "--run_name", name, "--seed", str(r.get("seed", 42)),
           "--lam_ce", str(r.get("lam_ce", 0.5))]
    {"cmd += ['--holdout_csv', HOLD]" if holdout else "pass"}
    if r.get("pseudo_csv"): cmd += ["--pseudo_csv", r["pseudo_csv"]]
    jobs.append((name, cmd))
print(len(jobs), "jobs to run")

q = queue.Queue()
for j in jobs: q.put(j)
failed = []

def worker(gpu):
    while True:
        try: name, cmd = q.get_nowait()
        except queue.Empty: return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        os.makedirs("/kaggle/working/logs", exist_ok=True)
        with open(f"/kaggle/working/logs/{{name}}.log", "w") as log:
            rc = subprocess.call(cmd, cwd="/kaggle/working", env=env, stdout=log, stderr=subprocess.STDOUT)
        print(("[ok]   " if rc == 0 else "[FAIL] ") + name, time.strftime("%H:%M:%S"))
        if rc != 0: failed.append(name)

ths = [threading.Thread(target=worker, args=(g,)) for g in range(max(1, torch.cuda.device_count()))]
[t.start() for t in ths]; [t.join() for t in ths]
print("failed:", failed)
""")


def keep_weights_cell(prefixes):
    return code(f"""
# keep model weights only for the runs needed later (cross-dataset evaluation); predictions (preds.npz) are always kept
import glob, os
KEEP = {list(prefixes)!r}
freed = 0
for p in glob.glob("/kaggle/working/runs/*/final_ema.pt"):
    if not any(os.path.basename(os.path.dirname(p)).startswith(k) for k in KEEP):
        freed += os.path.getsize(p); os.remove(p)
print(f"removed {{freed / 1e9:.2f}} GB of unneeded weights")
""")


def done_runs_code():
    return ('done = lambda p: all(os.path.exists(f"/kaggle/working/runs/{p}_f{k}/preds.npz") for k in range(5))\n'
            'complete = [r["prefix"] for r in RUNS if done(r["prefix"])]\n'
            'print("complete:", complete); print("incomplete:", [r["prefix"] for r in RUNS if not done(r["prefix"])])\n')


HEAD_FILES = lambda title, body: [md(f"# {title}\n\n{body}")]


def generic_head(title, body, detect):
    return HEAD_FILES(title, body) + [detect, md("## Source files (generated from `experiments/v2/src/` -- do not edit here)")] + \
        [writer_cell(n) for n in SUITE_FILES]


# --------------------------------------------------------------------------------------------- ablation (UNIFESP)
def ablation_notebook():
    S = "swin_tiny_s224"
    runs = [
        {"prefix": S, "bb": "swin_tiny", "size": 224, "lr": 2e-4},                                            # reference
        {"prefix": S + "_heavy", "bb": "swin_tiny", "size": 224, "lr": 2e-4, "aug": "heavy"},
        {"prefix": S + "_ce0", "bb": "swin_tiny", "size": 224, "lr": 2e-4, "lam_ce": 0.0},
        {"prefix": S + "_ce1", "bb": "swin_tiny", "size": 224, "lr": 2e-4, "lam_ce": 1.0},
        {"prefix": S + "_leaky", "bb": "swin_tiny", "size": 224, "lr": 2e-4, "folds_csv": "/kaggle/working/splits/cv_folds_leaky.csv"},
        {"prefix": S + "_seed1", "bb": "swin_tiny", "size": 224, "lr": 2e-4, "seed": 1},
        {"prefix": S + "_seed2", "bb": "swin_tiny", "size": 224, "lr": 2e-4, "seed": 2},
        {"prefix": "hybrid_gate_s224", "bb": "hybrid_gate", "size": 224, "lr": 2e-4},
        {"prefix": "hybrid_gate_ln_s224", "bb": "hybrid_gate_ln", "size": 224, "lr": 2e-4},
        {"prefix": "hybrid_avg_s224", "bb": "hybrid_avg", "size": 224, "lr": 2e-4},
        {"prefix": "hybrid_cnn_s224", "bb": "hybrid_cnn", "size": 224, "lr": 2e-4},
        {"prefix": "hybrid_swin_s224", "bb": "hybrid_swin", "size": 224, "lr": 2e-4},
        {"prefix": "densenet121_s256", "bb": "densenet121", "size": 256, "lr": 3e-4},
        {"prefix": "convnext_tiny_s256", "bb": "convnext_tiny", "size": 256, "lr": 1e-4},
    ]
    cells = generic_head("UNIFESP v2 -- ablation suite (blocks A, B, C, D, E of NOVELTY_AND_GAP.md)", """
**Attach:** `felipekitamura/unifesp-xray-bodypart-classification`, **GPU T4 x2**, Internet on. Same patient-level split / held-out set as the main v2 notebook.

| Block | Configs |
|---|---|
| A recipe | heavy vs light augmentation; softmax-head weight 0 / 0.5 / 1 |
| C model/seeds | 3 seeds of Swin-T (+ their average), DenseNet121, ConvNeXt-T |
| D fusion | gate (raw), gate+LayerNorm, fixed average, DenseNet only, Swin only -> **mean CNN gate weight per fold** (`meta.json`) |
| E split | `_leaky` = image-level folds of the same pool: compare `oof_minus_hold` with the patient-level reference |
| B decoding | computed from the reference run by `ablation_report` (none / min-one / global / per-class honest / per-class leaky) |

14 configs x 5 folds = 70 jobs. UNMEASURED estimate: 4-6 h on T4 x2 (fusion runs are ~2x slower). Config-major order, so if the session is cut you keep the
complete configs. Remove entries from `RUNS` to shorten it. Commit with **Save & Run All**.
""", B.DETECT) + [md("## Environment"), env_cell("unifesp"), md("## Index, splits (patient-level + a deliberately leaky image-level variant), cache"),
        code("""
!cd /kaggle/working && python -m src.build_index
!cd /kaggle/working && python -m src.make_splits
!cd /kaggle/working && python -m src.make_holdout_cv_splits --holdout_fold 4
!cd /kaggle/working && python -m src.make_leaky_splits_v2
!cd /kaggle/working && python -m src.merge_runs --write_hashes --index_csv index.csv --splits_dir splits --out split_hashes.json
"""), B.CACHE, md("## Train all configs on all folds"), jobs_cell(runs, 30),
        keep_weights_cell([S + "_f"]),
        md("## Ablation report (paired bootstrap vs the reference, decoding ablation, head/TTA variants)"),
        code("import json, os\n" + done_runs_code() + f"""
cfg = {{r["prefix"]: [r["prefix"]] for r in RUNS if r["prefix"] in complete}}
if all(p in complete for p in ["{S}", "{S}_seed1", "{S}_seed2"]):
    cfg["swin_3seeds_average"] = ["{S}", "{S}_seed1", "{S}_seed2"]
json.dump(cfg, open("/kaggle/working/ablation_cfg.json", "w"), indent=1)
print(list(cfg))
assert "{S}" in cfg, "reference run incomplete -- see /kaggle/working/logs"
"""),
        code("""
!cd /kaggle/working && python -m src.ablation_report --out_dir runs --config_json ablation_cfg.json --reference swin_tiny_s224 \\
    --index_csv index.csv --cv_folds_csv splits/cv_folds.csv --result_dir results_ablation --n_boot 2000
print(open("/kaggle/working/results_ablation/ablation.md").read())
"""),
        md("## Fusion ablation: how much weight does the CNN branch get? (gate collapse check)"),
        code("""
import glob, json, pandas as pd
rows = []
for m in sorted(glob.glob("/kaggle/working/runs/hybrid_*_f*/meta.json")):
    d = json.load(open(m)); run = os.path.basename(os.path.dirname(m))
    if "gate_cnn_weight_oof_mean" in d:
        rows.append({"run": run, "cnn_gate_weight_mean": d["gate_cnn_weight_oof_mean"], "std": d["gate_cnn_weight_oof_std"]})
g = pd.DataFrame(rows)
g["config"] = g["run"].str.replace(r"_f\\d$", "", regex=True)
print(g.groupby("config")["cnn_gate_weight_mean"].agg(["mean", "std", "min", "max"]).round(4))
g.to_csv("/kaggle/working/results_ablation/gate_weights.csv", index=False)
"""),
        md("## Next\n\nDownload `results_ablation/` (`ablation.md`, the three CSVs, `gate_weights.csv`). Reference = `swin_tiny_s224`. Interpret with `NOVELTY_AND_GAP.md` section 4.")]
    return cells


# --------------------------------------------------------------------------------------------- MURA
def mura_notebook():
    runs = [{"prefix": "mura_swin_tiny_s224", "bb": "swin_tiny", "size": 224, "lr": 2e-4},
            {"prefix": "mura_densenet121_s224", "bb": "densenet121", "size": 224, "lr": 3e-4}]
    baselines = [{"prefix": "mura_densenet201_s224", "bb": "densenet201", "size": 224, "lr": 3e-4},
                 {"prefix": "mura_efficientnet_b0_s224", "bb": "efficientnet_b0", "size": 224, "lr": 3e-4},
                 {"prefix": "mura_resnet101_s224", "bb": "resnet101", "size": 224, "lr": 3e-4},
                 {"prefix": "mura_vgg19_s224", "bb": "vgg19", "size": 224, "lr": 1e-4}]
    return generic_head("UNIFESP v2 -- MURA body-part (official protocol, comparable with Shah et al. 2026)", """
**Attach:** `cjinny/mura-v11` (the notebook looks for the `XR_*` folders), **GPU T4 x2**, Internet on.

Protocol: **held-out set = MURA `valid/` (public validation set, as in Shah 2026); CV pool = `train/` split into 5 patient-level folds**; 7 body-part classes
(the folder name `XR_*` is the label; MURA's own normal/abnormal label is NOT used). Reported: accuracy and macro-F1 of the 5-fold-bagged ensemble with
patient-bootstrap CIs, which Shah 2026 did not report (single seed, no CI).

Compute (UNMEASURED; earlier run: ~3.5 min per epoch per fold on a T4): 2 primary backbones x 5 folds x 12 epochs ~ 3-4 h wall on T4 x2.
Set `RUN_BASELINES = True` to add DenseNet-201 / EfficientNet-B0 / ResNet-101 / VGG-19 (Shah's baselines) -- that roughly triples the time; split it over
several sessions (config-major order keeps complete configs). Caching is disabled (PNG decoding is cheap; a 40k-image cache would fill the disk).
""", DETECT_MURA) + [md("## Environment"), env_cell("mura", cache=False),
        md("## Index + OFFICIAL MURA splits"),
        code("""
!cd /kaggle/working && python -m src.build_index_mura
!cd /kaggle/working && python -m src.make_mura_official_splits
!cd /kaggle/working && python -m src.merge_runs --write_hashes --index_csv index.csv --splits_dir splits --out split_hashes.json
"""), md("## Train"),
        code("RUN_BASELINES = False   # True: also train Shah-2026's baselines (3x the time)"),
        code("BASE = " + json.dumps(runs) + "\nBASELINES = " + json.dumps(baselines) + "\n"),
        jobs_cell(runs, 12, runs_expr="BASE + (BASELINES if RUN_BASELINES else [])"),
        keep_weights_cell(["mura_swin_tiny_s224_f"]),
        md("## Ensemble, honest evaluation, comparison with Shah 2026 (accuracy / macro-F1 of the argmax decision)"),
        code("""
import os
prefixes = ",".join(r["prefix"] for r in RUNS if all(os.path.exists(f"/kaggle/working/runs/{r['prefix']}_f{k}/preds.npz") for k in range(5)))
assert prefixes, "no complete run -- see /kaggle/working/logs"
print("runs:", prefixes)
!cd /kaggle/working && python -m src.run_ensemble --out_dir runs --runs {prefixes} --folds 0,1,2,3,4 --index_csv index.csv --cv_folds_csv splits/cv_folds.csv --holdout_csv splits/holdout_uids.csv --result_dir results_mura --n_boot 2000
"""),
        code("""
import json, numpy as np, pandas as pd
from src import evalkit as E
z = np.load("/kaggle/working/results_mura/ensemble_arrays.npz")
idx = pd.read_csv("/kaggle/working/index.csv").set_index("uid")
lab = [c for c in idx.columns if c.startswith("label_")]
y = idx.loc[z["hold_uid"], lab].to_numpy(int); groups = idx.loc[z["hold_uid"], "patient_id"].astype(str).to_numpy()
pred = np.eye(len(lab), dtype=int)[z["hold"].argmax(1)]                      # single-label decision
acc = lambda a, b: float((a.argmax(1) == b.argmax(1)).mean())
res = {"accuracy": E.cluster_bootstrap(acc, y, pred, groups, n_boot=2000), "macro_f1": E.cluster_bootstrap(E.macro_f1, y, pred, groups, n_boot=2000)}
print(json.dumps(res, indent=1))
print("Shah et al. 2026 (their MURA-valid test, 1 seed, no CI): accuracy 0.9643, macro-F1 0.9574 | DenseNet-201 0.9187 / 0.9093 | EfficientNet-B0 0.9043 / 0.8952 | ResNet-101 0.8934 / 0.8831 | VGG-19 0.8612 / 0.8498")
json.dump(res, open("/kaggle/working/results_mura/shah_comparison.json", "w"), indent=1)
"""),
        md("## Next\n\nKeep this notebook's **output** (it contains `mura_swin_tiny_s224_f*/final_ema.pt`) -- the cross-dataset notebook loads it. Send `results_mura/results.json` and `shah_comparison.json`.")]


# --------------------------------------------------------------------------------------------- COVID
def covid_notebook():
    runs = [{"prefix": "covid_swin_tiny_s224", "bb": "swin_tiny", "size": 224, "lr": 2e-4},
            {"prefix": "covid_densenet121_s224", "bb": "densenet121", "size": 224, "lr": 3e-4}]
    return generic_head("UNIFESP v2 -- COVID-19 Radiography (task generalization, same pipeline)", """
**Attach:** `tawsifurrahman/covid19-radiography-database`, **GPU T4 x2**, Internet on.

4-class chest X-ray diagnosis (COVID / Lung_Opacity / Normal / Viral_Pneumonia). The dataset has **no patient ids** (merged from many sources), so the split
is image-level: the numbers can be optimistic and the paper must say so. Same v2 pipeline (EMA, no checkpoint selection, OOF-only decisions, bootstrap CIs).
UNMEASURED estimate: ~1.5-2 h (earlier single-backbone run took 2h17m for 20 epochs).
""", DETECT_COVID) + [md("## Environment"), env_cell("covidcxr", cache=False), md("## Index + splits"),
        code("""
!cd /kaggle/working && python -m src.build_index_covidcxr
!cd /kaggle/working && python -m src.make_splits
!cd /kaggle/working && python -m src.make_holdout_cv_splits --holdout_fold 4
!cd /kaggle/working && python -m src.merge_runs --write_hashes --index_csv index.csv --splits_dir splits --out split_hashes.json
"""), md("## Train"), jobs_cell(runs, 12), keep_weights_cell([]),
        md("## Ensemble + honest evaluation"),
        code("""
import os
prefixes = ",".join(r["prefix"] for r in RUNS if all(os.path.exists(f"/kaggle/working/runs/{r['prefix']}_f{k}/preds.npz") for k in range(5)))
assert prefixes, "no complete run -- see /kaggle/working/logs"
!cd /kaggle/working && python -m src.run_ensemble --out_dir runs --runs {prefixes} --folds 0,1,2,3,4 --index_csv index.csv --cv_folds_csv splits/cv_folds.csv --holdout_csv splits/holdout_uids.csv --result_dir results_covid --n_boot 2000
import json; print(json.dumps(json.load(open("/kaggle/working/results_covid/results.json"))["holdout"], indent=1)[:3000])
"""),
        md("## Next\n\nSend `results_covid/results.json`. Remember: image-level split => state it as a limitation.")]


# --------------------------------------------------------------------------------------------- cross-dataset
def cross_notebook(frac_only=False):
    title = "UNIFESP v2 -- cross-dataset transfer (block F): " + ("-> FracAtlas only (re-run after a truncated-image crash)" if frac_only else "UNIFESP <-> MURA and -> FracAtlas")
    return generic_head(title, """
**Inputs to add:** (1) `felipekitamura/unifesp-xray-bodypart-classification`, (2) `cjinny/mura-v11`, (3) a **FracAtlas** dataset (CC BY 4.0; search Kaggle for "FracAtlas";
optional -- the cell skips it if not found), (4) the **output of the main/ablation v2 notebook** (needs `swin_tiny_s224_f*/final_ema.pt`), (5) the **output of the MURA notebook**
(needs `mura_swin_tiny_s224_f*/final_ema.pt`). GPU T4 x2.

Zero-shot closed-set transfer through the class map in `src/cross_eval.py` (shared groups: elbow, finger, forearm, hand, shoulder, wrist for UNIFESP<->MURA;
hand, hip, leg, shoulder for FracAtlas). **The map is an assumption** (e.g. FracAtlas "leg" = UNIFESP Lower Leg + Knee + Thigh): have a clinician check it. Humerus (MURA) has no UNIFESP
counterpart and is dropped. FracAtlas column names are not verified: the loader prints them and stops if `image_id/hand/leg/hip/shoulder` are missing.
Bootstrap CIs here are image-level (no patient ids in the targets) -- say so in the paper.
""", B.DETECT) + [md("## Environment, UNIFESP index (target for MURA->UNIFESP)"),
        code("""
import sys, os, glob
sys.path.insert(0, "/kaggle/working")
os.environ["UNIFESP_DATA_ROOT"] = IMAGES_DATASET
os.environ["UNIFESP_INDEX_ROOT"] = "/kaggle/working"
os.environ["UNIFESP_OUTPUT_DIR"] = "/kaggle/working/runs"
os.environ["UNIFESP_CACHE_DIR"] = "none"
!pip install -q timm albumentations iterative-stratification pydicom
import torch; print("CUDA:", torch.cuda.is_available())
!cd /kaggle/working && python -m src.build_index
"""),
        md("## Locate MURA, FracAtlas and the trained weights"),
        code("""
# MURA root
MURA_ROOT = None
for cur, dirs, files in os.walk("/kaggle/input"):
    if "XR_ELBOW" in dirs or any(d.startswith("XR_") for d in dirs):
        MURA_ROOT = os.path.dirname(cur); break
print("MURA_ROOT:", MURA_ROOT)
# FracAtlas: a dataset.csv with an image_id column
FRAC_ROOT = FRAC_CSV = None
for cur, dirs, files in os.walk("/kaggle/input"):
    if "dataset.csv" in files:
        head = open(os.path.join(cur, "dataset.csv")).readline().lower()
        if "image_id" in head and "shoulder" in head:
            FRAC_ROOT, FRAC_CSV = cur, os.path.join(cur, "dataset.csv"); break
print("FracAtlas:", FRAC_CSV, "(None = not attached -> skipped)")
def weights_dir(prefix):
    hits = glob.glob(f"/kaggle/input/**/{prefix}_f0/final_ema.pt", recursive=True)
    assert hits, f"weights for {prefix} not found: add the notebook output that contains runs/{prefix}_f*/final_ema.pt"
    return os.path.dirname(os.path.dirname(hits[0]))
W_UNI, W_MURA = weights_dir("swin_tiny_s224"), weights_dir("mura_swin_tiny_s224")
print(W_UNI, W_MURA, sep="\\n")
"""),
        ] + ([] if frac_only else [md("## UNIFESP -> MURA, MURA -> UNIFESP"),
        code("""
!cd /kaggle/working && python -m src.cross_eval --src_dataset unifesp --target mura --target_root {MURA_ROOT} --weights_dir {W_UNI} --run_prefix swin_tiny_s224 --backbone swin_tiny --img_size 224 --max_images 12000 --out_dir transfer
!cd /kaggle/working && python -m src.cross_eval --src_dataset mura --target unifesp --target_root {IMAGES_DATASET} --target_index index.csv --weights_dir {W_MURA} --run_prefix mura_swin_tiny_s224 --backbone swin_tiny --img_size 224 --out_dir transfer
""")]) + [
        md("## -> FracAtlas (zero-shot, both source models)"),
        code("""
if FRAC_CSV:
    !cd /kaggle/working && python -m src.cross_eval --src_dataset unifesp --target fracatlas --target_root {FRAC_ROOT} --fracatlas_csv {FRAC_CSV} --weights_dir {W_UNI} --run_prefix swin_tiny_s224 --backbone swin_tiny --img_size 224 --out_dir transfer
    !cd /kaggle/working && python -m src.cross_eval --src_dataset mura --target fracatlas --target_root {FRAC_ROOT} --fracatlas_csv {FRAC_CSV} --weights_dir {W_MURA} --run_prefix mura_swin_tiny_s224 --backbone swin_tiny --img_size 224 --out_dir transfer
else:
    print("FracAtlas not attached -- skipped")
"""),
        md("## Summary table"),
        code("""
import glob, json, pandas as pd
rows = []
for p in sorted(glob.glob("/kaggle/working/transfer/transfer_*.json")):
    d = json.load(open(p))
    for head in ("softmax", "sigmoid"):
        h = d[head]; ci = h["accuracy_ci_image_bootstrap"]
        rows.append({"source": d["source"], "target": d["target"], "head": head, "groups": ",".join(d["groups"]), "n": h["n_images"],
                     "accuracy": round(h["accuracy"], 4), "ci": f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]", "macro_f1": round(h["macro_f1"], 4)})
tab = pd.DataFrame(rows); print(tab.to_string(index=False)); tab.to_csv("/kaggle/working/transfer/summary.csv", index=False)
"""),
        md("## Next\n\nSend `transfer/summary.csv` and the JSON files (confusion matrices inside). Compare with the in-distribution numbers of the MURA / main notebooks.")]



# --------------------------------------------------------------------------------------------- stage 2 (boost the Kaggle score)
def stage2_notebook():
    F = "/kaggle/working/splits/folds.csv"            # 5 patient-level folds over ALL 1,738 labelled images (no held-out set)
    base = [("convnext_tiny_s256", "convnext_tiny", 256, 1e-4, "light", [0, 1]),
            ("resnet50_s256_heavy", "resnet50", 256, 3e-4, "heavy", [0, 1]),
            ("efficientnet_b3_s256_hv", "efficientnet_b3", 256, 3e-4, "heavy", [0, 1]),
            ("efficientnetv2_s_s256", "efficientnetv2_s", 256, 3e-4, "light", [0]),
            ("densenet121_s256", "densenet121", 256, 3e-4, "light", [0])]
    def mk(prefix_tag, pseudo):
        out = []
        for p, bb, sz, lr, aug, seeds in base:
            for sd in seeds:
                d = {"prefix": f"{prefix_tag}_{p}_seed{sd}", "bb": bb, "size": sz, "lr": lr, "aug": aug, "seed": sd, "folds_csv": F}
                if pseudo:
                    d["pseudo_csv"] = "/kaggle/working/splits/pseudo.csv"
                out.append(d)
        return out
    a_runs, b_runs = mk("s2", False), mk("s2p", True)
    ens = lambda tag, runs_expr, extra="": code(f"""
import os
_runs = {runs_expr}
done = [p for p in _runs if all(os.path.exists(f"/kaggle/working/runs/{{p}}_f{{k}}/preds.npz") for k in range(5))]
print("complete runs:", len(done), "of", len(_runs)); assert done, "no complete run -- see /kaggle/working/logs"
prefixes = ",".join(done)
!cd /kaggle/working && python -m src.run_ensemble --out_dir runs --runs {{prefixes}} --folds 0,1,2,3,4 --index_csv index.csv --cv_folds_csv splits/folds.csv --sample_submission {{IMAGES_DATASET}}/sample_submission.csv --result_dir results_{tag}
!cp /kaggle/working/results_{tag}/submission.csv /kaggle/working/submission_{tag}.csv
import json; r = json.load(open("/kaggle/working/results_{tag}/results.json"))
print("OOF (ensemble @0.5 + min-one):", {{k: round(v, 4) for k, v in r["oof"]["ens_0.5_min_one"].items()}})
print("weights:", {{k: round(v, 3) for k, v in r["weights"].items() if v > 0}})
""")
    return generic_head("UNIFESP v2 -- stage 2: train on ALL labelled images (+ optional pseudo-labels) to push the Kaggle score", """
**Attach:** `felipekitamura/unifesp-xray-bodypart-classification`, **GPU T4 x2**, Internet on. One account, one notebook at a time.

* **2a** -- the best configs of stage 1 (ConvNeXt-T, ResNet50-heavy, EffNet-B3-heavy, EffNetV2-S, DenseNet121; 2 seeds for the first three) trained on **all 1,738 labelled images**
  in 5 patient-level folds (+25 % data vs stage 1, which kept 323 images aside). Ensemble weights / thresholds from OOF only -> `submission_s2a.csv`.
* **Pseudo-label round** -- confident predictions of the 2a ensemble on the 743 unlabelled test images become extra training targets (the competition rules state
  pseudo-labelling is allowed; no test label is ever touched). **2b** retrains the same configs with them -> `submission_s2b.csv`; **s2u** = ensemble of 2a + 2b.
* UNMEASURED time: 40 jobs per round, ~2.5 min each wall => ~100 min per round, ~3.5-4 h in total.

Honesty notes for the paper: there is NO held-out set here (everything is used for training), so the only internal number is OOF, which for the pseudo-label round is slightly optimistic;
judge the rounds by the **Late-Submission score** and report ALL submissions you made (not just the best), and say that 2b/s2u are transductive (use unlabelled test images).
The paper's primary numbers stay those of stage 1 (held-out set, CIs).
""", B.DETECT) + [md("## Environment, index, ALL-data folds, cache"), env_cell("unifesp"),
        code("""
!cd /kaggle/working && python -m src.build_index
!cd /kaggle/working && python -m src.make_splits
!cd /kaggle/working && python -m src.merge_runs --write_hashes --index_csv index.csv --splits_dir splits --out split_hashes.json
"""), B.CACHE,
        md("## 2a -- all-data folds, no pseudo-labels"), jobs_cell(a_runs, 30, holdout=False), keep_weights_cell([]),
        md("## 2a ensemble (OOF-only decisions) -> submission_s2a.csv"), ens("s2a", "[" + ", ".join(repr(r["prefix"]) for r in a_runs) + "]"),
        md("## Confident pseudo-labels for the unlabelled test images (rules: pseudo-labelling is allowed)"),
        code("""
MIN_CONF = 0.9        # lower (0.8) keeps more images but noisier labels
!cd /kaggle/working && python -m src.make_pseudo_labels --ensemble_arrays results_s2a/ensemble_arrays.npz --index_csv index.csv --out splits/pseudo.csv --min_conf {MIN_CONF}
"""),
        md("## 2b -- same configs, plus the pseudo-labelled test images in every training fold"), jobs_cell(b_runs, 30, holdout=False), keep_weights_cell([]),
        md("## 2b ensemble -> submission_s2b.csv, and 2a+2b -> submission_s2u.csv"),
        ens("s2b", "[" + ", ".join(repr(r["prefix"]) for r in b_runs) + "]"),
        ens("s2u", "[" + ", ".join(repr(r["prefix"]) for r in a_runs + b_runs) + "]"),
        md("## Next\n\nDownload `submission_s2a.csv`, `submission_s2b.csv`, `submission_s2u.csv` and the three `results_*` folders. Late-submit them (ONE account) and send me the "
           "private/public scores. Stage-1 reference: private 0.92563 / public 0.92704; competition #1 = 0.93378.")]


# --------------------------------------------------------------------------------------------- explainability
def explain_notebook():
    models = [("convnext_tiny_s256", "convnext_tiny", 256, 1e-4, "light"),
              ("swin_tiny_s224", "swin_tiny", 224, 2e-4, "light"),
              ("resnet50_s256_heavy", "resnet50", 256, 3e-4, "heavy")]
    runs = [{"prefix": p, "bb": bb, "size": sz, "lr": lr, "aug": aug} for p, bb, sz, lr, aug in models]
    cmds = "\n".join(f"!cd /kaggle/working && python -m src.explain --weights runs/{p}_f0/final_ema.pt --backbone {bb} --img_size {sz} "
                     f"--index_csv index.csv --holdout_csv splits/holdout_uids.csv --out_dir explain_{p} --title {p}" for p, bb, sz, lr, aug in models)
    return generic_head("UNIFESP v2 -- explainability (Grad-CAM + faithfulness + sanity check + error analysis)", """
**Attach:** `felipekitamura/unifesp-xray-bodypart-classification`, **GPU T4 x2**, Internet on.

Trains ONE fold (fold 0 of the patient-level CV pool) of ConvNeXt-T, Swin-T and ResNet50-heavy (a modern CNN, a Transformer and the 2nd-place Kaggle architecture), then explains each on the **323
held-out images** that no model ever saw. UNMEASURED time: ~15-25 min.

Per model (`explain_<run>/`): `gallery_page*.png` (per class: a correct and a missed image; Grad-CAM for the true class and for the wrongly predicted class),
`explain_metrics.json` (**faithfulness**: deleting the most salient pixels first vs a random map on the same coarse grid; **sanity check**: CAM of the trained model vs a randomly initialised
model, Spearman; **tie-aware**), `top_confusions.csv` (error analysis), `per_class_recall.csv`.

Limits to state in the paper: Grad-CAM grids are coarse (7x7-8x8), UNIFESP has no localisation labels (no pointing game / IoU), and no radiologist reviewed the maps unless you add that step
(suggestion: one radiologist rates ~100 maps as "anatomically plausible / not").
""", B.DETECT) + [md("## Environment, index, splits, cache"), env_cell("unifesp"),
        code("""
!cd /kaggle/working && python -m src.build_index
!cd /kaggle/working && python -m src.make_splits
!cd /kaggle/working && python -m src.make_holdout_cv_splits --holdout_fold 4
"""), B.CACHE, md("## Train fold 0 of the three models (two GPUs in parallel)"), jobs_cell(runs, 30, folds=(0,)),
        md("## Explain each model on the held-out set"), code(cmds),
        md("## Summary table + first gallery page of each model"),
        code("""
import json, glob, os, pandas as pd
rows = []
for m in sorted(glob.glob("/kaggle/working/explain_*/explain_metrics.json")):
    d = json.load(open(m)); f = d["faithfulness_deletion"]; r = d["randomization_sanity"]
    rows.append({"model": d["title"], "target_layer": d["target_layer"], "holdout_sample_F1(blend)": round(d["holdout_sample_f1_blend"], 4), "pairs": f["pairs"],
                 "deletion_CAM": round(f["score_cam"]["mean"], 3), "deletion_random_grid": round(f["score_random_order"]["mean"], 3),
                 "gain [95% CI]": f"{f['deletion_gain_random_minus_cam']['mean']:.3f} [{f['deletion_gain_random_minus_cam']['ci_low']:.3f}, {f['deletion_gain_random_minus_cam']['ci_high']:.3f}]",
                 "rho trained-vs-random": round(r["spearman_trained_vs_random_weights"]["mean"], 3), "frac |rho|<0.3": round(r["fraction_abs_rho_below_0.3"], 2)})
tab = pd.DataFrame(rows); print(tab.to_string(index=False)); tab.to_csv("/kaggle/working/explain_summary.csv", index=False)
from IPython.display import Image, display
for g in sorted(glob.glob("/kaggle/working/explain_*/gallery_page1.png")):
    print(g); display(Image(g))
"""),
        md("## Next\n\nDownload `explain_summary.csv` and the three `explain_*` folders (galleries, metrics, confusions). Interpretation rule: **gain > 0 with a CI above 0** and **low rho** = the maps depend on what the model learned; "
           "if a model fails either check, say so in the paper instead of showing only the pretty pictures.")]


# --------------------------------------------------------------------------------------------- frozen foundation-model probe (gap G7)
def fm_notebook():
    return generic_head("UNIFESP v2 -- frozen foundation-model linear probe (gap G7)", """
**Attach:** `felipekitamura/unifesp-xray-bodypart-classification`, **GPU T4 (x1 or x2)**, **Internet ON** (weights come from the Hugging Face hub). Does NOT depend on any other notebook.

Question: how far does a FROZEN pretrained encoder + a linear layer get on UNIFESP, compared with the fine-tuned networks (ablation reference Swin-T: held-out sample-F1 0.9089 [0.8576, 0.9509])?
Encoders: DINOv2-S, DINOv2-B (natural images), BiomedCLIP (biomedical image-text), Swin-T / ConvNeXt-T ImageNet-1k (supervised), plus a random-weights ResNet18 as the floor.
Same patient-level folds and held-out set as the main study; the probe's weight decay is chosen on OOF only; 0.5 threshold + min-one rule (the ablation reference decoding).
Also: label-efficiency curve (10/25/50/100 % of the training patients).
UNMEASURED time: feature extraction of 1,738 images x 6 encoders is a few minutes on a T4; the probes are seconds. Expect 10-30 min in total.
A model that cannot be downloaded is skipped and listed with its error, the others still run.
""", B.DETECT) + [md("## Environment, index, splits, cache"), env_cell("unifesp"),
        code("!pip install -q open_clip_torch"),
        code("""
!cd /kaggle/working && python -m src.build_index
!cd /kaggle/working && python -m src.make_splits
!cd /kaggle/working && python -m src.make_holdout_cv_splits --holdout_fold 4
"""), B.CACHE,
        md("## Probe (5-fold OOF for the weight decay, held-out once, patient-level bootstrap CI)"),
        code("""
!cd /kaggle/working && python -m src.fm_probe --models dinov2_s,dinov2_b,biomedclip,swin_in1k,convnext_in1k,resnet18_random \\
    --cv_folds_csv splits/cv_folds.csv --holdout_csv splits/holdout_uids.csv --out_dir fm_results --n_boot 2000
"""),
        md("## Summary tables\n\nDownload `fm_results/` (fm_summary.csv, fm_label_efficiency.csv, fm_wd_search.csv, fm_report.md, fm_*_probs.npz) and send it / place it in `experiments/results/fm/`.")]


# --------------------------------------------------------------------------------------------- reliability / label audit (CPU only)
def reliability_notebook():
    return generic_head("UNIFESP v2 -- reliability for routing: calibration, selective prediction, per-class CIs, label-error audit (gaps G8, G9, G12)", """
**Attach:** `felipekitamura/unifesp-xray-bodypart-classification` and the **output of `noorabdullah02/unifesp-v2-ablation-suite`** (needs `runs/*_f*/preds.npz`). **No GPU needed** (Accelerator: None), a few minutes.

Uses the saved fold predictions of the ablation runs (same patient-level folds and held-out set as the main study): swin_tiny_s224 (3 seeds), densenet121_s256, convnext_tiny_s256.
Outputs: calibration (ECE before/after temperature scaling fitted on OOF), risk-coverage curve ("abstain" for low-confidence images), per-class recall/precision/F1 with patient-bootstrap CIs,
and a list of SUSPECTED label errors in the training pool (out-of-fold, confident-learning style) for a clinician to review. Suspects are not proven errors.
""", B.DETECT) + [md("## Environment, index, splits"), env_cell("unifesp", cache=False),
        code("""
!cd /kaggle/working && python -m src.build_index
!cd /kaggle/working && python -m src.make_splits
!cd /kaggle/working && python -m src.make_holdout_cv_splits --holdout_fold 4
import glob, os
hits = glob.glob("/kaggle/input/**/swin_tiny_s224_f0/preds.npz", recursive=True)
assert hits, "attach the output of noorabdullah02/unifesp-v2-ablation-suite"
RUNS_DIR = os.path.dirname(os.path.dirname(hits[0])); print("runs dir:", RUNS_DIR)
"""),
        md("## Analysis (OOF-only decisions; held-out set used once)"),
        code("""
RUNS = "swin_tiny_s224,swin_tiny_s224_seed1,swin_tiny_s224_seed2,densenet121_s256,convnext_tiny_s256"
!cd /kaggle/working && python -m src.reliability --out_dir {RUNS_DIR} --runs {RUNS} --index_csv index.csv --cv_folds_csv splits/cv_folds.csv --holdout_csv splits/holdout_uids.csv --result_dir results_reliability --n_boot 1000
"""),
        md("## Files\n\nDownload `results_reliability/` (reliability.json, per_class_holdout.csv, label_audit_suspects.csv) or copy the printed output.")]


BUILDERS = {"ablation": ablation_notebook, "mura": mura_notebook, "covid": covid_notebook, "cross": cross_notebook, "stage2": stage2_notebook, "explain": explain_notebook, "fm": fm_notebook, "cross_frac": lambda: cross_notebook(True), "reliability": reliability_notebook}
FILENAMES = {"ablation": "UNIFESP_v2_ablation.ipynb", "mura": "UNIFESP_v2_mura.ipynb", "covid": "UNIFESP_v2_covid.ipynb",
             "cross": "UNIFESP_v2_cross.ipynb", "stage2": "UNIFESP_v2_stage2_boost.ipynb", "explain": "UNIFESP_v2_explain.ipynb", "fm": "UNIFESP_v2_fm_probe.ipynb", "cross_frac": "UNIFESP_v2_cross_fracatlas.ipynb", "reliability": "UNIFESP_v2_reliability.ipynb"}


def verify(cells):
    for c in cells:
        if c["cell_type"] != "code":
            continue
        keep, cont = [], False
        for l in "".join(c["source"]).splitlines():
            if cont or l.lstrip().startswith(("!", "%")):
                cont = l.rstrip().endswith("\\")
                continue
            keep.append(l)
        # an `if X:` whose only body lines were shell commands would be empty after stripping -> add `pass`
        src = "\n".join(keep)
        ast.parse(src.replace("else:\n", "else:\n    pass\n") if False else _patch_empty_blocks(src))
    for name in SUITE_FILES:
        want = open(os.path.join(SRC, name + ".py"), encoding="utf-8").read()
        hit = False
        for c in cells:
            s = "".join(c["source"])
            if c["cell_type"] == "code" and f'/kaggle/working/src/{name}.py"' in s:
                for node in ast.walk(ast.parse(s)):
                    if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "write":
                        assert ast.literal_eval(node.args[0]) == want, f"{name}.py differs in notebook"
                        hit = True
        assert hit, f"{name}.py not embedded"


def _patch_empty_blocks(src):
    out, lines = [], src.split("\n")
    for i, l in enumerate(lines):
        out.append(l)
        if l.rstrip().endswith(":") and not l.lstrip().startswith("#"):
            nxt = next((x for x in lines[i + 1:] if x.strip()), "")
            if (len(nxt) - len(nxt.lstrip())) <= (len(l) - len(l.lstrip())):
                out.append(" " * (len(l) - len(l.lstrip()) + 4) + "pass")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all", choices=["all"] + list(BUILDERS))
    a = ap.parse_args()
    for k in (BUILDERS if a.which == "all" else [a.which]):
        cells = BUILDERS[k]()
        verify(cells)
        nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
              "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}}
        out = os.path.join(HERE, FILENAMES[k])
        json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
        print(f"wrote {FILENAMES[k]}  ({len(cells)} cells, {os.path.getsize(out) / 1e3:.0f} kB; embedded files verified)")


if __name__ == "__main__":
    main()
