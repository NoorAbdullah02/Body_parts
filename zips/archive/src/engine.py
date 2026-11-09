"""
Train/validate loop with checkpoint + auto-resume, built around Kaggle's
12-hour session limit: pass --max_hours (default 11.0, leaving a ~1h buffer).
If the wall-clock budget runs out, we save a checkpoint after the current
epoch finishes and exit cleanly (exit code 0) instead of getting killed
mid-epoch by Kaggle. Re-running the same command resumes from epoch+1.
"""
import os
import time
import json
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score


def _safe_macro_auc(y_true, y_prob):
    """Macro AUC skipping classes with 0 or all-positive ground truth in this split
    (undefined otherwise) -- common for the rarest UNIFESP classes in a single fold."""
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.min() == col.max():
            continue
        aucs.append(roc_auc_score(col, y_prob[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def sample_mean_f1(y_true, y_pred):
    """Per-image F1 averaged over images -- this is the UNIFESP Kaggle
    competition's actual leaderboard metric ("Mean F1-Score": one F1 per
    SOPInstanceUID between predicted and true label sets, then averaged).
    It's NOT the same as macro_f1 (averaged per-CLASS): since ~90% of images
    are single-label and Chest dominates, a model that nails the common
    classes scores high here even if it's weak on rare classes -- which is
    exactly why the real Kaggle leaderboard (top score 0.934, kaggle.com/
    competitions/unifesp-x-ray-body-part-classifier) sits well above the
    86.94% "accuracy" reported in Krishnan & Padmavathi 2025 (Procedia CS) --
    different metric, not actually a stronger model. This function is what
    makes our numbers comparable to that leaderboard.
    """
    f1s = []
    for i in range(y_true.shape[0]):
        t, p = y_true[i], y_pred[i]
        tp = np.sum((t == 1) & (p == 1))
        fp = np.sum((t == 0) & (p == 1))
        fn = np.sum((t == 1) & (p == 0))
        if tp == 0 and fp == 0 and fn == 0:
            f1s.append(1.0)  # both empty -- shouldn't happen here (every image has >=1 label) but defined for safety
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def compute_metrics(y_true, y_prob, threshold=0.5):
    # Every value here is cast to a plain Python float (sklearn/numpy return
    # numpy scalars) -- these end up pickled inside checkpoint_*.pt via the
    # "history" list, and PyTorch >=2.6's default torch.load(weights_only=True)
    # refuses to unpickle numpy scalar types, which crashed evaluate_checkpoints.py
    # with "Unsupported global: GLOBAL numpy._core.multiarray.scalar".
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "sample_mean_f1": float(sample_mean_f1(y_true, y_pred)),  # <- compare this one to the Kaggle leaderboard
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_auc": float(_safe_macro_auc(y_true, y_prob)),
        "subset_accuracy": float((y_pred == y_true).all(axis=1).mean()),
        "kappa_flat": float(cohen_kappa_score(y_true.flatten(), y_pred.flatten())),
    }


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None):
    model.train()
    total_loss = 0.0
    n = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = loss_fn(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / n


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_probs, all_targets = [], []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = loss_fn(logits, targets)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets)
    metrics = compute_metrics(y_true, y_prob)
    metrics["loss"] = total_loss / n
    return metrics, y_prob, y_true


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_metric, history):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
        "history": history,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)  # our own checkpoint, trusted
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt["scheduler_state"] is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt["best_metric"], ckpt["history"]


def run_training(model, train_loader, val_loader, optimizer, scheduler, loss_fn,
                  device, run_dir, n_epochs=30, max_hours=11.0, metric_name="macro_f1"):
    os.makedirs(run_dir, exist_ok=True)
    last_ckpt = os.path.join(run_dir, "checkpoint_last.pt")
    best_ckpt = os.path.join(run_dir, "checkpoint_best.pt")
    history_path = os.path.join(run_dir, "history.json")

    start_epoch = 0
    best_metric = -float("inf")
    history = []
    if os.path.exists(last_ckpt):
        start_epoch, best_metric, history = load_checkpoint(last_ckpt, model, optimizer, scheduler, device)
        start_epoch += 1
        print(f"[resume] loaded checkpoint, continuing from epoch {start_epoch}/{n_epochs}")

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    run_start = time.time()
    budget_seconds = max_hours * 3600

    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        val_metrics, _, _ = validate(model, val_loader, loss_fn, device)
        if scheduler is not None:
            scheduler.step()
        dt = time.time() - t0

        row = {"epoch": epoch, "train_loss": train_loss, "epoch_seconds": dt, **val_metrics}
        history.append(row)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
              f"{metric_name}={val_metrics[metric_name]:.4f} ({dt:.1f}s)")

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        if val_metrics[metric_name] > best_metric:
            best_metric = val_metrics[metric_name]
            save_checkpoint(best_ckpt, epoch, model, optimizer, scheduler, best_metric, history)
            print(f"  -> new best {metric_name}={best_metric:.4f}, saved checkpoint_best.pt")

        save_checkpoint(last_ckpt, epoch, model, optimizer, scheduler, best_metric, history)

        elapsed = time.time() - run_start
        if elapsed > budget_seconds:
            print(f"[time budget] {elapsed/3600:.2f}h elapsed > {max_hours}h limit, "
                  f"stopping after epoch {epoch}. Re-run the same command to resume.")
            break

    return history, best_metric
