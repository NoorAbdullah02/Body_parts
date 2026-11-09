"""
make_pseudo_labels -- confident pseudo-labels for the UNLABELLED Kaggle test images, from a finished ensemble.

The competition rules (read on the join dialog) say: "Annotating the test set is manually prohibited (pseudo-label is allowed)".
This script does NOT touch any true test label (there are none); it only turns the ensemble's own predictions into training targets.

    python -m src.make_pseudo_labels --ensemble_arrays results_s2a/ensemble_arrays.npz --index_csv index.csv --out splits/pseudo.csv --min_conf 0.9

Selection (per image): decode with the ensemble's frozen thresholds (min-one rule); keep the image only if EVERY predicted class has probability >= min_conf
and EVERY other class has probability <= 1 - min_conf.  Optional --max_images caps the number kept (most confident first).
Output columns: uid + label_* (hard 0/1), compatible with train_v2 --pseudo_csv.

Honest-evaluation note: pseudo-labels come from models that saw all training folds, so the OOF score of a pseudo-label round is slightly optimistic.
Judge such a round by the Late-Submission score, and report that it is a transductive (test-image-using) variant.
"""
import argparse

import numpy as np
import pandas as pd

from . import evalkit as E


def select_confident(prob, thresholds, min_conf=0.9, max_images=0):
    pred = E.predict_labels(prob, thresholds, min_one=True)
    pos_ok = np.where(pred == 1, prob >= min_conf, True).all(1)
    neg_ok = np.where(pred == 0, prob <= 1.0 - min_conf, True).all(1)
    keep = pos_ok & neg_ok
    conf = np.where(pred == 1, prob, 1.0 - prob).min(1)
    idx = np.where(keep)[0]
    if max_images and len(idx) > max_images:
        idx = idx[np.argsort(-conf[idx])[:max_images]]
    return np.sort(idx), pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble_arrays", required=True)
    p.add_argument("--index_csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min_conf", type=float, default=0.9)
    p.add_argument("--max_images", type=int, default=0)
    a = p.parse_args()
    z = np.load(a.ensemble_arrays, allow_pickle=False)
    assert z["test"].size, "ensemble_arrays.npz has no test probabilities"
    cols = [c for c in pd.read_csv(a.index_csv, nrows=1).columns if c.startswith("label_")]
    assert z["test"].shape[1] == len(cols), (z["test"].shape, len(cols))
    idx, pred = select_confident(z["test"], z["thresholds"], a.min_conf, a.max_images)
    out = pd.DataFrame(pred[idx], columns=cols)
    out.insert(0, "uid", z["test_uid"][idx])
    out.to_csv(a.out, index=False)
    n = len(z["test"])
    print(f"kept {len(idx)} / {n} test images (min_conf={a.min_conf}); "
          f"single-label share {float((pred[idx].sum(1) == 1).mean()):.3f}; class counts {dict(zip(cols, pred[idx].sum(0).tolist()))}")
    print("saved", a.out)


if __name__ == "__main__":
    main()
