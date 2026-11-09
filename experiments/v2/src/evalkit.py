"""
evalkit -- leakage-free evaluation utilities for the UNIFESP body-part project.

Pure numpy (no torch), so it runs anywhere: Kaggle, laptop, CI.

What it fixes compared with the earlier notebooks
-------------------------------------------------
1. Thresholds are tuned on OUT-OF-FOLD (OOF) predictions of the CV pool and then
   FROZEN before they touch the held-out set (``tune_thresholds_oof`` +
   ``nested_evaluate``).  The old ``ensemble_eval.py`` tuned thresholds on the very
   fold it then scored, which inflated sample_mean_f1 (0.9356 / 0.9445).
2. "At least one label per image" decoding (``predict_labels(min_one=True)``): ~92 %
   of UNIFESP images are single-label and an empty prediction scores F1 = 0.
3. Patient-level (cluster) bootstrap confidence intervals and a paired bootstrap
   test for model-vs-model differences (``cluster_bootstrap``, ``paired_bootstrap``).
4. Caruana greedy ensemble selection on OOF predictions (``caruana_ensemble``).
5. Kaggle submission writer + validator (``write_submission``, ``check_submission``).

Metric
------
``sample_mean_f1`` = per-image F1 between predicted and true label sets, averaged over
images.  This is the Kaggle "Mean F1-Score" of the UNIFESP competition.
"""
from __future__ import annotations

import csv
from typing import Iterable, Sequence

import numpy as np

N_CLASSES = 22
DEFAULT_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


# --------------------------------------------------------------------------- metrics
def per_image_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """F1 per image. Both-empty -> 1.0, true-nonempty & pred-empty -> 0.0."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    tp = (y_true & y_pred).sum(1)
    fp = ((1 - y_true) & y_pred).sum(1)
    fn = (y_true & (1 - y_pred)).sum(1)
    denom = 2 * tp + fp + fn
    return np.where(denom == 0, 1.0, 2.0 * tp / np.maximum(denom, 1))


def sample_mean_f1(y_true, y_pred) -> float:
    return float(per_image_f1(y_true, y_pred).mean())


def macro_f1(y_true, y_pred) -> float:
    """Mean of per-class F1 (zero_division=0, same as sklearn's default for this use)."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    tp = (y_true & y_pred).sum(0)
    fp = ((1 - y_true) & y_pred).sum(0)
    fn = (y_true & (1 - y_pred)).sum(0)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom == 0, 0.0, 2.0 * tp / np.maximum(denom, 1))
    return float(f1.mean())


def micro_f1(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    tp = (y_true & y_pred).sum()
    fp = ((1 - y_true) & y_pred).sum()
    fn = (y_true & (1 - y_pred)).sum()
    denom = 2 * tp + fp + fn
    return float(0.0 if denom == 0 else 2.0 * tp / denom)


def subset_accuracy(y_true, y_pred) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).all(axis=1).mean())


def all_metrics(y_true, y_pred) -> dict:
    return {
        "sample_mean_f1": sample_mean_f1(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "micro_f1": micro_f1(y_true, y_pred),
        "subset_accuracy": subset_accuracy(y_true, y_pred),
    }


# --------------------------------------------------------------------------- decoding
def predict_labels(prob: np.ndarray, thresholds=0.5, min_one: bool = True,
                   max_labels: int | None = None) -> np.ndarray:
    """Probabilities -> 0/1 matrix.

    min_one:    every image gets at least its best-scoring class (score = prob / threshold,
                so a class with a low tuned threshold is not unfairly skipped).
    max_labels: keep at most this many labels per image (by the same score).
    """
    prob = np.asarray(prob, dtype=np.float64)
    thr = np.broadcast_to(np.asarray(thresholds, dtype=np.float64), (prob.shape[1],))
    pred = (prob >= thr[None, :]).astype(np.int64)
    score = prob / np.maximum(thr[None, :], 1e-6)
    if min_one:
        empty = pred.sum(1) == 0
        if empty.any():
            best = score[empty].argmax(1)
            pred[np.where(empty)[0], best] = 1
    if max_labels is not None:
        order = np.argsort(-score, axis=1)
        keep = np.zeros_like(pred)
        rows = np.arange(prob.shape[0])[:, None]
        keep[rows, order[:, :max_labels]] = 1
        pred = pred * keep
        if min_one:  # top-1 is always kept, but be explicit
            pred[np.arange(prob.shape[0]), order[:, 0]] = 1
    return pred


# --------------------------------------------------------------------------- thresholds
def tune_thresholds_oof(oof_prob: np.ndarray, y: np.ndarray, grid: Sequence[float] = DEFAULT_GRID,
                        base: float = 0.5, n_rounds: int = 3, min_pos: float = 20.0,
                        min_one: bool = True) -> np.ndarray:
    """Per-class thresholds by coordinate ascent on the REAL objective (sample_mean_f1).

    * Use ONLY out-of-fold predictions of the CV pool here -- never the held-out set and
      never the same fold you later report on.
    * Rare classes are shrunk toward ``base`` with weight n_pos / (n_pos + min_pos), because
      a threshold fitted on 5-10 positives is mostly noise.
    """
    oof_prob = np.asarray(oof_prob, dtype=np.float64)
    y = np.asarray(y).astype(np.int64)
    C = oof_prob.shape[1]
    thr = np.full(C, base, dtype=np.float64)

    def score(t):
        return sample_mean_f1(y, predict_labels(oof_prob, t, min_one=min_one))

    best = score(thr)
    for _ in range(n_rounds):
        changed = False
        for c in range(C):
            cur = thr[c]
            best_t, best_s = cur, best
            for g in grid:
                if g == cur:
                    continue
                thr[c] = g
                s = score(thr)
                # strictly better, or equal but closer to the base threshold (prefer simple)
                if s > best_s + 1e-12 or (abs(s - best_s) <= 1e-12 and abs(g - base) < abs(best_t - base)):
                    best_t, best_s = g, s
            thr[c] = best_t
            if best_t != cur:
                changed = True
            best = best_s
        if not changed:
            break
    n_pos = y.sum(0).astype(np.float64)
    lam = n_pos / (n_pos + min_pos)
    return lam * thr + (1.0 - lam) * base


def nested_evaluate(oof_prob, y_oof, hold_prob, y_hold, **tune_kw) -> dict:
    """The only honest way to report tuned-threshold numbers.

    thresholds <- OOF of the CV pool;  scores <- held-out set (never used for any choice).
    Returns the metrics for three decoding rules so each lever's gain is visible.
    """
    thr = tune_thresholds_oof(oof_prob, y_oof, **tune_kw)
    out = {
        "thresholds": thr,
        "hold_0.5": all_metrics(y_hold, predict_labels(hold_prob, 0.5, min_one=False)),
        "hold_0.5_min_one": all_metrics(y_hold, predict_labels(hold_prob, 0.5, min_one=True)),
        "hold_tuned_min_one": all_metrics(y_hold, predict_labels(hold_prob, thr, min_one=True)),
    }
    return out


# --------------------------------------------------------------------------- bootstrap
def _cluster_indices(groups: np.ndarray):
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    return uniq, members


def cluster_bootstrap(metric_fn, y_true, y_pred, groups, n_boot: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> dict:
    """Patient-level bootstrap: resample PATIENTS (all their images together), recompute metric.

    metric_fn(y_true, y_pred) -> float.  ``groups`` = patient id per image.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    _, members = _cluster_indices(groups)
    P = len(members)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, P, size=P)
        idx = np.concatenate([members[k] for k in pick])
        vals[b] = metric_fn(y_true[idx], y_pred[idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(metric_fn(y_true, y_pred)), "ci_low": float(lo), "ci_high": float(hi),
            "boot_std": float(vals.std(ddof=1)), "n_patients": P, "n_boot": n_boot}


def paired_bootstrap(metric_fn, y_true, pred_a, pred_b, groups, n_boot: int = 2000, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    """Is model A better than B?  Same patient resamples for both; returns CI of (A - B)."""
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    _, members = _cluster_indices(groups)
    P = len(members)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, P, size=P)
        idx = np.concatenate([members[k] for k in pick])
        d[b] = metric_fn(y_true[idx], pred_a[idx]) - metric_fn(y_true[idx], pred_b[idx])
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    # two-sided bootstrap p-value for H0: difference == 0
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return {"diff": float(metric_fn(y_true, pred_a) - metric_fn(y_true, pred_b)),
            "ci_low": float(lo), "ci_high": float(hi), "p_value": float(min(1.0, p)), "n_boot": n_boot}


# --------------------------------------------------------------------------- ensembling
def average_probs(prob_list: Iterable[np.ndarray], weights=None, mode: str = "prob") -> np.ndarray:
    arr = np.stack([np.asarray(p, dtype=np.float64) for p in prob_list])
    w = np.ones(len(arr)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    if mode == "logit":
        eps = 1e-6
        lg = np.log(np.clip(arr, eps, 1 - eps) / (1 - np.clip(arr, eps, 1 - eps)))
        z = np.tensordot(w, lg, axes=1)
        return 1.0 / (1.0 + np.exp(-z))
    return np.tensordot(w, arr, axes=1)


def caruana_ensemble(oof_list: Sequence[np.ndarray], y: np.ndarray, n_iter: int = 40,
                     init_best: bool = True) -> np.ndarray:
    """Greedy forward selection WITH replacement on OOF sample_mean_f1 (0.5 thr, min_one).

    Returns weights (sum 1).  Fit on OOF only; apply the weights unchanged to held-out/test.
    """
    k = len(oof_list)
    counts = np.zeros(k)

    def sc(w):
        p = average_probs(oof_list, w)
        return sample_mean_f1(y, predict_labels(p, 0.5, min_one=True))

    if init_best:
        singles = [sc(np.eye(k)[i]) for i in range(k)]
        counts[int(np.argmax(singles))] = 1
    for _ in range(n_iter):
        best_i, best_s = None, -1.0
        for i in range(k):
            trial = counts.copy()
            trial[i] += 1
            s = sc(trial)
            if s > best_s:
                best_i, best_s = i, s
        counts[best_i] += 1
    return counts / counts.sum()


def combine_heads(p_sigmoid: np.ndarray, p_softmax: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a multilabel (sigmoid) head with a single-label (softmax) head."""
    return alpha * np.asarray(p_sigmoid) + (1.0 - alpha) * np.asarray(p_softmax)


# --------------------------------------------------------------------------- submission
def write_submission(uids: Sequence[str], prob: np.ndarray, thresholds, path: str,
                     min_one: bool = True, max_labels: int | None = None) -> np.ndarray:
    pred = predict_labels(prob, thresholds, min_one=min_one, max_labels=max_labels)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["SOPInstanceUID", "Target"])
        for u, row in zip(uids, pred):
            w.writerow([u, " ".join(str(int(c)) for c in np.where(row == 1)[0])])
    return pred


def check_submission(path: str, sample_path: str, n_classes: int = N_CLASSES) -> dict:
    """Validate a submission against Kaggle's sample_submission.csv. Raises AssertionError."""
    with open(path, newline="") as f:
        sub = list(csv.DictReader(f))
    with open(sample_path, newline="") as f:
        ref = list(csv.DictReader(f))
    assert list(sub[0].keys()) == ["SOPInstanceUID", "Target"], f"bad header {list(sub[0].keys())}"
    assert len(sub) == len(ref), f"row count {len(sub)} != {len(ref)}"
    assert {r["SOPInstanceUID"] for r in sub} == {r["SOPInstanceUID"] for r in ref}, "id set differs"
    assert len({r["SOPInstanceUID"] for r in sub}) == len(sub), "duplicate ids"
    n_labels = []
    for r in sub:
        labs = r["Target"].split()
        assert len(labs) >= 1, f"empty Target for {r['SOPInstanceUID']}"
        ints = [int(x) for x in labs]
        assert all(0 <= i < n_classes for i in ints), f"label out of range: {labs}"
        assert len(set(ints)) == len(ints), f"duplicate labels: {labs}"
        n_labels.append(len(ints))
    return {"rows": len(sub), "mean_labels_per_image": float(np.mean(n_labels)),
            "frac_single_label": float(np.mean(np.array(n_labels) == 1))}
