"""
Generate the self-contained Kaggle notebooks from the files in src/.

    python build_notebook.py              # ONE notebook with all runs:  UNIFESP_v2_pipeline.ipynb
    python build_notebook.py --parts 2    # UNIFESP_v2_part1of2.ipynb, ..part2of2.ipynb + UNIFESP_v2_merge.ipynb

Why generate: the notebooks are ALWAYS rebuilt from src/, so what is tested locally is exactly what runs
on Kaggle (the old notebooks warned about "stale code" problems).  Every embedded source file is verified
to round-trip byte-for-byte and every code cell is parsed before anything is written.

All part notebooks belong to the SAME Kaggle account (Kaggle's terms forbid more than one account per person)
and share its weekly GPU quota; splitting only shortens each session and lets two sessions overlap if the
account is allowed to run two at once.
"""
import argparse
import ast
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
shutil.copy(os.path.join(HERE, "..", "evalkit.py"), os.path.join(SRC, "evalkit.py"))   # single source of truth

SRC_FILES = ["__init__", "config", "dataset", "build_index", "make_splits", "make_holdout_cv_splits",
             "cache_images", "evalkit", "train_v2", "run_ensemble", "merge_runs"]

# (prefix, backbone, img_size, aug, lr) -- UNIFESP images are natively 256x256; Swin-T is fixed at 224.
RUNS_ALL = [
    ("swin_tiny_s224",          "swin_tiny",        224, "light", 2e-4),
    ("densenet121_s256",        "densenet121",      256, "light", 3e-4),
    ("efficientnetv2_s_s256",   "efficientnetv2_s", 256, "light", 3e-4),
    ("convnext_tiny_s256",      "convnext_tiny",    256, "light", 1e-4),   # lr lowered: 3e-4 gave 0.73 macro-F1 before
    ("resnet50_s256_heavy",     "resnet50",         256, "heavy", 3e-4),   # 2nd-place Kaggle recipe: ResNet + heavy aug
    ("efficientnet_b3_s256_hv", "efficientnet_b3",  256, "heavy", 3e-4),
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def writer_cell(name):
    body = open(os.path.join(SRC, name + ".py"), encoding="utf-8").read()
    return code(
        'import os\n'
        'os.makedirs("/kaggle/working/src", exist_ok=True)\n'
        f'with open("/kaggle/working/src/{name}.py", "w", encoding="utf-8") as f:\n'
        f'    f.write({body!r})\n'
        f'print("wrote src/{name}.py")\n'
    )


DETECT = code("""
import os
images_root = None
for cur, dirs, files in os.walk("/kaggle/input"):
    if "train" in dirs and "test" in dirs:
        images_root = cur
        dirs[:] = [d for d in dirs if d not in ("train", "test")]
        break
assert images_root is not None, "Attach felipekitamura/unifesp-xray-bodypart-classification (Notebook Settings -> Data)."
print("Images dataset:", images_root)
IMAGES_DATASET = images_root
""")

ENV = code("""
import sys, os
sys.path.insert(0, "/kaggle/working")
os.environ["UNIFESP_DATA_ROOT"] = IMAGES_DATASET
os.environ["UNIFESP_INDEX_ROOT"] = "/kaggle/working"
os.environ["UNIFESP_OUTPUT_DIR"] = "/kaggle/working/runs"
os.environ["UNIFESP_CACHE_DIR"] = "/kaggle/working/image_cache"
!pip install -q timm albumentations iterative-stratification pydicom
import torch
print("CUDA:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count(), "| torch", torch.__version__)
""")

SPLITS = code("""
!cd /kaggle/working && python -m src.build_index
!cd /kaggle/working && python -m src.make_splits
!cd /kaggle/working && python -m src.make_holdout_cv_splits --holdout_fold 4
!cd /kaggle/working && python -m src.merge_runs --write_hashes --index_csv index.csv --splits_dir splits --out split_hashes.json
""")

CACHE = code("!cd /kaggle/working && python -m src.cache_images")


def jobs_cell(runs):
    lines = ",\n".join(f"    ({p!r}, {b!r}, {s}, {a!r}, {lr})" for p, b, s, a, lr in runs)
    return code(f"""
import subprocess, itertools, threading, queue, os, time

EPOCHS, BATCH = 30, 32
RUNS = [  # (prefix, backbone, img_size, aug, lr)
{lines},
]
FOLDS = [0, 1, 2, 3, 4]
OUT = "/kaggle/working/runs"
CV = "/kaggle/working/splits/cv_folds.csv"
HOLD = "/kaggle/working/splits/holdout_uids.csv"

jobs = []
for (prefix, bb, size, aug, lr), f in itertools.product(RUNS, FOLDS):
    name = f"{{prefix}}_f{{f}}"
    if os.path.exists(f"{{OUT}}/{{name}}/preds.npz"):
        continue
    jobs.append((name, ["python", "-m", "src.train_v2", "--backbone", bb, "--img_size", str(size), "--aug", aug,
                        "--lr", str(lr), "--epochs", str(EPOCHS), "--batch_size", str(BATCH), "--fold", str(f),
                        "--folds_csv", CV, "--fold_col", "fold", "--holdout_csv", HOLD, "--run_name", name]))
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


ENSEMBLE = code("""
import os
if "RUNS" in globals():     # single / part notebook: use only runs whose 5 folds all finished
    done = [r[0] for r in RUNS if all(os.path.exists(f"/kaggle/working/runs/{r[0]}_f{k}/preds.npz") for k in range(5))]
    print("complete runs :", done)
    print("incomplete (skipped; see /kaggle/working/logs/*.log):", [r[0] for r in RUNS if r[0] not in done])
    prefixes = ",".join(done)
else:                       # merge notebook: merge_runs already wrote the list of complete runs
    prefixes = open("/kaggle/working/runs/RUNS.txt").read()
assert prefixes, "no complete run -- check /kaggle/working/logs/*.log"
!cd /kaggle/working && python -m src.run_ensemble --out_dir runs --runs {prefixes} --folds 0,1,2,3,4 \\
    --index_csv index.csv --cv_folds_csv splits/cv_folds.csv --holdout_csv splits/holdout_uids.csv \\
    --sample_submission {IMAGES_DATASET}/sample_submission.csv --result_dir results
!cp /kaggle/working/results/submission.csv /kaggle/working/submission.csv
""")

SHOW = code("""
import json, pandas as pd
res = json.load(open("/kaggle/working/results/results.json"))
print(json.dumps({k: res[k] for k in ("weights", "chosen_variant")}, indent=1))
print(json.dumps(res["holdout"], indent=1)[:3500])
pd.read_csv("/kaggle/working/results/per_run.csv")
""")


def single_notebook(runs):
    return [md("""
# UNIFESP v2 -- leakage-free, ensemble-ready pipeline (paper protocol + Kaggle test predictions)

**Attach ONE dataset:** `felipekitamura/unifesp-xray-bodypart-classification`. **Accelerator:** GPU T4 x2. **Internet:** on.

1. Builds `index.csv`, the patient-level 5-fold split, a fixed **20 % held-out set** and a 5-fold **CV pool**.
2. Trains every run on every CV fold (`src/train_v2.py`): fixed schedule + EMA, **no checkpoint selection**, so out-of-fold (OOF)
   predictions are unbiased. Each fold also predicts the held-out set and the Kaggle test set.
3. `src/run_ensemble.py` makes ALL decisions on OOF only, reports the held-out score ONCE with patient-level bootstrap CIs and writes
   `submission.csv` for Kaggle *Late Submission*.

Resumable: finished runs (`preds.npz` exists) are skipped, so after a cut-off session just run all again.
"""), DETECT, md("## Source files (generated from `experiments/v2/src/` -- do not edit here)")] + \
        [writer_cell(n) for n in SRC_FILES] + \
        [md("## Environment"), ENV, md("## Index, patient-level splits, held-out set, image cache"), SPLITS, CACHE,
         md("## Train every run on every CV fold (2 GPUs in parallel, resumable)\n\nUNIFESP images are natively 256x256: nothing to gain by upsampling; Swin-T is fixed at 224."),
         jobs_cell(runs),
         md("## Ensemble + honest evaluation + Kaggle submission\n\nLook at `holdout.ens_tuned_min_one` (the number to report), its CI, and `holdout.paired_ens_vs_single_sample_f1`."),
         ENSEMBLE, SHOW,
         md("## Next\n\nSubmit `submission.csv` at <https://www.kaggle.com/competitions/unifesp-x-ray-body-part-classifier/submit> (Late Submission, ONE account). "
            "Leaderboard: #1 = 0.93378, #2 = 0.92787; the private score uses ~60 % of the test set with the original (noisy) labels, so ~0.003 differences are noise.")]


def part_notebook(k, n, runs):
    names = ", ".join(r[0] for r in runs)
    head = md(f"""
# UNIFESP v2 -- part {k} of {n}

Runs: **{names}**  (5 CV folds each). Same account as the other parts; they share its weekly GPU quota.
**Attach:** `felipekitamura/unifesp-xray-bodypart-classification`, **GPU T4 x2**, Internet on.

After ALL parts have finished: **Save Version** (Save & Run All) of each part, then run `UNIFESP_v2_merge` with the
parts' outputs added as inputs. This notebook also writes `split_hashes.json`, which the merge notebook uses to refuse merging
predictions that came from different splits.
""")
    return [head, DETECT, md("## Source files (generated)")] + [writer_cell(x) for x in SRC_FILES] + \
        [md("## Environment"), ENV, md("## Index, splits, cache"), SPLITS, CACHE,
         md("## Train this part's runs on every CV fold"), jobs_cell(runs),
         md("## Quick look at this part alone (the real ensemble is made in the merge notebook)"), ENSEMBLE, SHOW]


def merge_notebook(n):
    return [md(f"""
# UNIFESP v2 -- merge the {n} parts and make the final ensemble

1. Add the **output of every part notebook** as input (right panel -> Add Input -> Notebook Output) -- and the UNIFESP dataset.
2. No GPU needed. Run all.

The notebook rebuilds the splits (deterministic), **refuses to merge if any part used different splits**, copies the finished runs
(only runs with all 5 folds) and calls `run_ensemble` on all of them.
"""), DETECT, md("## Source files (generated)")] + [writer_cell(x) for x in SRC_FILES] + [
        md("## Environment + splits"),
        code("""
import sys, os
sys.path.insert(0, "/kaggle/working")
os.environ["UNIFESP_DATA_ROOT"] = IMAGES_DATASET
os.environ["UNIFESP_INDEX_ROOT"] = "/kaggle/working"
os.environ["UNIFESP_OUTPUT_DIR"] = "/kaggle/working/runs"
!pip install -q iterative-stratification pydicom
"""),
        SPLITS,
        md("## Collect the parts' predictions (+ split-hash check)"),
        code("""
!cd /kaggle/working && python -m src.merge_runs --input_root /kaggle/input --out_dir runs --index_csv index.csv --splits_dir splits
print("merging:", open("/kaggle/working/runs/RUNS.txt").read())
"""),
        md("## Final ensemble + honest evaluation + submission"),
        ENSEMBLE, SHOW,
        md("## Next\n\nSubmit `submission.csv` (ONE account) as a Late Submission and compare with the leaderboard.")]


def verify(cells):
    for c in cells:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        keep, cont = [], False
        for l in src.splitlines():          # drop shell/magic lines incl. backslash continuations
            if cont or l.lstrip().startswith(("!", "%")):
                cont = l.rstrip().endswith("\\")
                continue
            keep.append(l)
        ast.parse("\n".join(keep))
    for name in SRC_FILES:
        want = open(os.path.join(SRC, name + ".py"), encoding="utf-8").read()
        found = False
        for c in cells:
            s = "".join(c["source"])
            if c["cell_type"] == "code" and f'/kaggle/working/src/{name}.py"' in s:
                for node in ast.walk(ast.parse(s)):
                    if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "write":
                        assert ast.literal_eval(node.args[0]) == want, f"{name}.py differs in notebook"
                        found = True
        assert found, f"{name}.py not embedded"


def save(cells, fname):
    verify(cells)
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}}
    out = os.path.join(HERE, fname)
    json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
    print(f"wrote {fname}  ({len(cells)} cells, {os.path.getsize(out) / 1e3:.0f} kB; embedded files verified)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, default=1)
    a = ap.parse_args()
    if a.parts <= 1:
        save(single_notebook(RUNS_ALL), "UNIFESP_v2_pipeline.ipynb")
        return
    for k in range(a.parts):
        runs = RUNS_ALL[k::a.parts]                 # interleaved -> similar total time per part
        save(part_notebook(k + 1, a.parts, runs), f"UNIFESP_v2_part{k + 1}of{a.parts}.ipynb")
    save(merge_notebook(a.parts), "UNIFESP_v2_merge.ipynb")


if __name__ == "__main__":
    main()
