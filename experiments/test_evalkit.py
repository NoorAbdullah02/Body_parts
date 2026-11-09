"""Self-tests for evalkit.  Run:  python test_evalkit.py   (needs numpy, scikit-learn only)."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit as E  # noqa: E402

# real UNIFESP train label counts (train.csv), used only to make the synthetic data realistic
CLASS_COUNTS = np.array([92, 54, 81, 738, 9, 25, 78, 17, 15, 74, 25, 110, 26, 87, 120, 71, 41, 23, 10, 15, 64, 105], float)


def make_data(n=1738, seed=0, noise=1.0, patients_per_img=0.44):
    """Synthetic probs: ~92 % single-label images, real class imbalance, noisy logits."""
    rng = np.random.default_rng(seed)
    p = CLASS_COUNTS / CLASS_COUNTS.sum()
    y = np.zeros((n, 22), dtype=np.int64)
    for i in range(n):
        k = rng.choice([1, 2, 3], p=[0.924, 0.070, 0.006])
        y[i, rng.choice(22, size=k, replace=False, p=p)] = 1
    n_pat = max(1, int(n * patients_per_img))
    groups = rng.integers(0, n_pat, size=n)
    # class-dependent difficulty: rare classes are noisier -> thresholds matter
    sigma = noise * (1.0 + 0.6 * (1 - CLASS_COUNTS / CLASS_COUNTS.max()))
    logit = np.where(y == 1, 2.2, -3.5) + rng.normal(0, 1, size=y.shape) * sigma[None, :]
    prob = 1.0 / (1.0 + np.exp(-logit))
    return y, prob, groups


def ref_sample_mean_f1(y_true, y_pred):
    """Verbatim logic of src/engine.py::sample_mean_f1 (the earlier notebooks)."""
    f1s = []
    for i in range(y_true.shape[0]):
        t, p = y_true[i], y_pred[i]
        tp = np.sum((t == 1) & (p == 1)); fp = np.sum((t == 0) & (p == 1)); fn = np.sum((t == 1) & (p == 0))
        if tp == 0 and fp == 0 and fn == 0:
            f1s.append(1.0); continue
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0)
    return float(np.mean(f1s))


def test_metrics_match_reference():
    y, prob, _ = make_data(400, seed=1)
    pred = (prob >= 0.5).astype(int)
    assert abs(E.sample_mean_f1(y, pred) - ref_sample_mean_f1(y, pred)) < 1e-12
    from sklearn.metrics import f1_score
    assert abs(E.macro_f1(y, pred) - f1_score(y, pred, average="macro", zero_division=0)) < 1e-12
    assert abs(E.micro_f1(y, pred) - f1_score(y, pred, average="micro", zero_division=0)) < 1e-12
    # edge cases: both empty -> 1, pred empty / true nonempty -> 0
    z = np.zeros((1, 22), int); one = z.copy(); one[0, 3] = 1
    assert E.sample_mean_f1(z, z) == 1.0 and E.sample_mean_f1(one, z) == 0.0 and E.sample_mean_f1(one, one) == 1.0


def test_min_one_and_max_labels():
    y, prob, _ = make_data(500, seed=2)
    hi = np.full(22, 2.0)  # nothing can pass the threshold
    assert E.predict_labels(prob, hi, min_one=False).sum() == 0
    p1 = E.predict_labels(prob, hi, min_one=True)
    assert (p1.sum(1) == 1).all()
    assert (p1.argmax(1) == prob.argmax(1)).all()          # equal thresholds -> the arg-max class
    lo = E.predict_labels(prob, 0.05, min_one=True, max_labels=2)
    assert (lo.sum(1) <= 2).all() and (lo.sum(1) >= 1).all()


def test_min_one_helps_when_many_empty():
    y, prob, _ = make_data(1500, seed=3, noise=1.8)
    a = E.sample_mean_f1(y, E.predict_labels(prob, 0.5, min_one=False))
    b = E.sample_mean_f1(y, E.predict_labels(prob, 0.5, min_one=True))
    assert b >= a, (a, b)
    print(f"  min_one rule: {a:.4f} -> {b:.4f}")


def test_leakage_is_real_and_nested_is_honest():
    """Tuning thresholds on the set you report on inflates the score; nested does not."""
    leaky, nested, base = [], [], []
    for s in range(12):
        y, prob, _ = make_data(1738, seed=100 + s)
        half = len(y) // 2
        y_a, p_a, y_b, p_b = y[:half], prob[:half], y[half:], prob[half:]
        thr_same = E.tune_thresholds_oof(p_b, y_b)             # tuned ON the eval set  (the old bug)
        thr_other = E.tune_thresholds_oof(p_a, y_a)            # tuned elsewhere        (nested)
        leaky.append(E.sample_mean_f1(y_b, E.predict_labels(p_b, thr_same)))
        nested.append(E.sample_mean_f1(y_b, E.predict_labels(p_b, thr_other)))
        base.append(E.sample_mean_f1(y_b, E.predict_labels(p_b, 0.5)))
        # deterministic property: tuning on the same data can never be worse than the start point
        assert leaky[-1] >= base[-1] - 0.02  # (small slack: shrinkage pulls rare thr back to 0.5)
    gap = float(np.mean(leaky) - np.mean(nested))
    print(f"  base={np.mean(base):.4f}  nested={np.mean(nested):.4f}  leaky={np.mean(leaky):.4f}  inflation={gap:+.4f}")
    assert gap > 0.0, "leaky tuning should look better than nested on average"


def test_rare_class_shrinkage():
    y, prob, _ = make_data(1738, seed=7, noise=1.6)
    thr = E.tune_thresholds_oof(prob, y, min_pos=20.0)
    n_pos = y.sum(0)
    rare = int(np.argmin(n_pos)); common = int(np.argmax(n_pos))
    assert abs(thr[rare] - 0.5) <= abs(thr[common] - 0.5) + 0.25
    assert ((thr > 0) & (thr < 1)).all()


def test_bootstrap_ci():
    y, prob, groups = make_data(1200, seed=11)
    pred = E.predict_labels(prob, 0.5)
    r = E.cluster_bootstrap(E.sample_mean_f1, y, pred, groups, n_boot=400, seed=1)
    assert r["ci_low"] <= r["estimate"] <= r["ci_high"], r
    # smaller sample -> wider interval
    r_small = E.cluster_bootstrap(E.sample_mean_f1, y[:300], pred[:300], groups[:300], n_boot=400, seed=1)
    assert (r_small["ci_high"] - r_small["ci_low"]) > (r["ci_high"] - r["ci_low"])
    # identical models: diff 0, p = 1
    d = E.paired_bootstrap(E.sample_mean_f1, y, pred, pred, groups, n_boot=200, seed=2)
    assert d["diff"] == 0.0 and d["p_value"] == 1.0
    # clearly better model is detected
    good = E.predict_labels(prob, 0.5)
    bad = good.copy(); bad[: len(bad) // 3] = 0
    d2 = E.paired_bootstrap(E.sample_mean_f1, y, good, bad, groups, n_boot=300, seed=3)
    assert d2["diff"] > 0 and d2["ci_low"] > 0 and d2["p_value"] < 0.05, d2
    print(f"  CI (n=1200, 12-ish patients clustered): {r['estimate']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")


def test_caruana_and_blend():
    y, p1, _ = make_data(900, seed=21, noise=1.4)
    rng = np.random.default_rng(5)
    mu = np.where(y == 1, 2.2, -3.5)
    p2 = 1 / (1 + np.exp(-(mu + rng.normal(0, 2.2, y.shape))))   # weaker model
    p3 = 1 / (1 + np.exp(-(mu + rng.normal(0, 1.2, y.shape))))   # stronger, independent noise
    w = E.caruana_ensemble([p1, p2, p3], y, n_iter=25)
    assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all()
    ens = E.average_probs([p1, p2, p3], w)
    s_ens = E.sample_mean_f1(y, E.predict_labels(ens, 0.5))
    s_best = max(E.sample_mean_f1(y, E.predict_labels(p, 0.5)) for p in (p1, p2, p3))
    assert s_ens >= s_best - 1e-9, (s_ens, s_best)
    lg = E.average_probs([p1, p3], mode="logit")
    assert lg.shape == p1.shape and ((lg > 0) & (lg < 1)).all()
    bl = E.combine_heads(p1, p3, 0.3)
    assert np.allclose(bl, 0.3 * p1 + 0.7 * p3)


def test_submission_roundtrip():
    y, prob, _ = make_data(60, seed=31)
    uids = [f"1.2.826.0.1.{i}" for i in range(60)]
    with tempfile.TemporaryDirectory() as d:
        sample = os.path.join(d, "sample.csv"); out = os.path.join(d, "sub.csv")
        with open(sample, "w") as f:
            f.write("SOPInstanceUID,Target\n")
            for u in uids:
                f.write(f"{u},0 1 2 3\n")
        pred = E.write_submission(uids, prob, np.full(22, 0.99), out)    # forces the min-one fallback
        info = E.check_submission(out, sample)
        assert info["rows"] == 60 and info["frac_single_label"] == 1.0
        assert open(out).read().splitlines()[0] == "SOPInstanceUID,Target"
        # format of a multi-label row: sorted, space separated ints
        pred = E.write_submission(uids, prob, 0.2, out)
        for line in open(out).read().splitlines()[1:]:
            labs = [int(x) for x in line.split(",")[1].split()]
            assert labs == sorted(labs) and len(labs) >= 1
        # validator must catch problems
        bad = os.path.join(d, "bad.csv")
        open(bad, "w").write("SOPInstanceUID,Target\n" + f"{uids[0]},\n" + "\n".join(f"{u},1" for u in uids[1:]) + "\n")
        try:
            E.check_submission(bad, sample); raise SystemExit("validator missed an empty Target")
        except AssertionError:
            pass


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"[run ] {t.__name__}")
        t()
        print(f"[pass] {t.__name__}")
    print(f"\nall {len(tests)} tests passed")


if __name__ == "__main__":
    main()
