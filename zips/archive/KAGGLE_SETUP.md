# Kaggle setup (dual GPU, 12-hour session limit)

## 1. Upload this pipeline as a Kaggle Dataset

Zip the whole `MereDelam` project folder (just `src/`, `index.csv`,
`splits/`, `requirements.txt` — you don't need to re-upload `train/`/`test/`
DICOMs, the original UNIFESP Kaggle dataset already has those) and upload it
as a new **private Kaggle Dataset**, e.g. named `unifesp-pipeline`.

## 2. New Kaggle Notebook → attach TWO datasets

- `felipekitamura/unifesp-xray-bodypart-classification` (the original images)
- `<your-username>/unifesp-pipeline` (the code + index.csv + splits/ you just uploaded)

Enable **GPU T4 x2** (or whatever dual-GPU option your Kaggle tier has) in
Notebook Settings → Accelerator.

## 3. First cell — environment setup

```python
import os, shutil, sys

# copy the code so `import src.xxx` works from /kaggle/working
shutil.copytree("/kaggle/input/unifesp-pipeline/src", "/kaggle/working/src", dirs_exist_ok=True)
sys.path.insert(0, "/kaggle/working")

os.environ["UNIFESP_DATA_ROOT"] = "/kaggle/input/unifesp-xray-bodypart-classification"
os.environ["UNIFESP_INDEX_ROOT"] = "/kaggle/input/unifesp-pipeline"
os.environ["UNIFESP_OUTPUT_DIR"] = "/kaggle/working/runs"

!pip install -q timm albumentations iterative-stratification thop grad-cam
```

## 4. Step 2 — architecture stats (cheap, run once)

```python
!cd /kaggle/working && python -m src.compare_backbones
```

## 5. Step 2 — train all 8 baselines, using BOTH GPUs in parallel

Each backbone/fold run is independent, so put one on each GPU and run two
at a time in the background. The dataset is tiny (1738 train images), so
even "slow" backbones (ViT/Swin) finish a 30-epoch run in well under the
12h budget — the parallelism just lets you get through all 8 backbones in
roughly half the wall-clock time.

```python
import subprocess, time

backbones = ["vgg16", "inception_v3", "resnet50", "densenet121",
             "efficientnet_b3", "efficientnetv2_s", "convnext_tiny", "swin_tiny"]

def launch(backbone, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log = open(f"/kaggle/working/{backbone}.log", "w")
    return subprocess.Popen(
        ["python", "-m", "src.train", "--backbone", backbone, "--fold", "0",
         "--epochs", "30", "--batch_size", "32", "--max_hours", "5"],
        cwd="/kaggle/working", env=env, stdout=log, stderr=subprocess.STDOUT,
    )

# run two at a time (one per GPU), wait for both, then the next pair
for i in range(0, len(backbones), 2):
    pair = backbones[i:i+2]
    procs = [launch(b, gpu) for gpu, b in enumerate(pair)]
    for p in procs:
        p.wait()
    print("finished pair:", pair)
```

Watch progress any time with `!tail -f /kaggle/working/<backbone>.log` in another cell.

## 6. If the 12-hour session ends mid-run

Just start a new session (same two datasets attached, same first cell) and
re-run the **exact same command** for whichever backbone didn't finish — it
auto-resumes from `/kaggle/working/runs/<backbone>_fold0/checkpoint_last.pt`.
`--max_hours 5` above is intentionally well under 12h per backbone so a
pair almost always finishes inside one session; lower it further if a
backbone is still running when you want to stop (it checkpoints every
epoch regardless).

**Important:** `/kaggle/working` is wiped when a session ends unless you
"Save Version" the notebook (or download the `runs/` folder) before it
closes — do that at the end of every session so checkpoints survive.

## 7. After training, build the final comparison table

```python
!cd /kaggle/working && python -m src.aggregate_results --fold 0
```

This merges `backbone_architecture_stats.csv` (params/FLOPs/inference time)
with each backbone's `history.json` (best macro-F1/AUC/kappa, training time)
into `comparison_table_fold0.csv` — exactly the Step 2 deliverable.
