"""Synthetic end-to-end test of src/reliability.py (no GPU, no real model): fake fold predictions with known label noise."""
import json, os, subprocess, sys, tempfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main(smoke):
    idx = pd.read_csv(f"{smoke}/index.csv"); cv = pd.read_csv(f"{smoke}/splits/cv_folds.csv"); hold = pd.read_csv(f"{smoke}/splits/holdout_uids.csv")
    lab = [c for c in idx.columns if c.startswith("label_")]; L = idx.set_index("uid")
    rng = np.random.default_rng(0); out = tempfile.mkdtemp()
    flipped = set()
    # inject label errors into 12 pool images: the model (trained on the TRUE labels) will disagree confidently with the given label
    pool = cv["uid"].tolist(); bad = set(rng.choice(pool, 12, replace=False))
    def fake(uids, true):
        p = np.clip(true * 0.9 + rng.normal(0, 0.05, true.shape) + 0.02, 0.001, 0.999); return p.astype(np.float32)
    for run in ("a", "b"):
        for f in range(5):
            d = f"{out}/{run}_f{f}"; os.makedirs(d); P = {}
            va = cv[cv["fold"] == f]["uid"].to_numpy(); ytrue = L.loc[va, lab].to_numpy(float)
            for tag, uids in (("oof", va), ("hold", hold["uid"].to_numpy())):
                yy = L.loc[uids, lab].to_numpy(float)
                sig = fake(uids, yy); soft = sig / sig.sum(1, keepdims=True)
                P.update({f"{tag}_uid": np.asarray(uids, dtype=str), f"{tag}_sig": sig, f"{tag}_soft": soft, f"{tag}_sig_tta": sig, f"{tag}_soft_tta": soft})
            np.savez(f"{d}/preds.npz", **P)
    # given labels with errors: swap the label of the 'bad' images in a COPY of the index
    idx2 = idx.copy(); m = idx2["uid"].isin(bad)
    for i in idx2.index[m]:
        row = idx2.loc[i, lab].to_numpy(int); k = int(row.argmax()); row[:] = 0; row[(k + 3) % len(lab)] = 1; idx2.loc[i, lab] = row
    idx2.to_csv(f"{out}/index.csv", index=False)
    r = subprocess.run([sys.executable, "-m", "src.reliability", "--out_dir", out, "--runs", "a,b", "--index_csv", f"{out}/index.csv", "--cv_folds_csv", f"{smoke}/splits/cv_folds.csv",
                        "--holdout_csv", f"{smoke}/splits/holdout_uids.csv", "--result_dir", f"{out}/res", "--n_boot", "50"], cwd=HERE, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=HERE))
    print(r.stdout[-2500:]); print(r.stderr[-1500:])
    assert r.returncode == 0
    j = json.load(open(f"{out}/res/reliability.json"))
    got = set(pd.read_csv(f"{out}/res/label_audit_suspects.csv")["uid"]) if os.path.exists(f"{out}/res/label_audit_suspects.csv") else set()
    hit = len(got & bad)
    print("injected errors:", len(bad), "| flagged:", len(got), "| flagged & injected:", hit)
    assert hit >= 8, "audit failed to recover most injected label errors"
    assert j["calibration"]["ece_holdout_after"] <= j["calibration"]["ece_holdout_before"] + 0.02
    print("OK")


if __name__ == "__main__":
    main(sys.argv[1])
