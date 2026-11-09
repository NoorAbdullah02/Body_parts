"""
merge_runs -- helpers for running the pipeline as several notebooks (same Kaggle account) and merging.

Why a hash check: every part notebook rebuilds index.csv / splits from the same code and seed, so
they SHOULD be identical.  If they are not (different dataset version, library version, ...) the
OOF predictions of the parts refer to different folds and must not be merged.

    python -m src.merge_runs --write_hashes --index_csv index.csv --splits_dir splits --out split_hashes.json
    python -m src.merge_runs --input_root /kaggle/input --out_dir runs --index_csv index.csv --splits_dir splits
"""
import argparse
import hashlib
import json
import os
import re
import shutil

RUN_DIR = re.compile(r"^(?P<prefix>.+)_f(?P<fold>\d+)$")
HASHED = ("cv_folds.csv", "holdout_uids.csv", "folds.csv")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_hashes(index_csv, splits_dir):
    out = {"index.csv": sha256(index_csv)}
    for n in HASHED:
        p = os.path.join(splits_dir, n)
        if os.path.exists(p):
            out[n] = sha256(p)
    return out


def discover(input_root):
    """{prefix: {fold: dir}} for every <prefix>_f<k>/preds.npz found anywhere under input_root."""
    found = {}
    for root, dirs, files in os.walk(input_root):
        if "preds.npz" in files:
            m = RUN_DIR.match(os.path.basename(root))
            if m:
                found.setdefault(m["prefix"], {}).setdefault(int(m["fold"]), root)
    return found


def check_hashes(input_root, local):
    """Compare local split hashes with every split_hashes.json found under input_root."""
    problems, seen = [], 0
    for root, _d, files in os.walk(input_root):
        if "split_hashes.json" in files:
            seen += 1
            theirs = json.load(open(os.path.join(root, "split_hashes.json")))
            for k, v in theirs.items():
                if k in local and local[k] != v:
                    problems.append(f"{root}: {k} differs")
    return seen, problems


def collect(input_root, out_dir, folds=(0, 1, 2, 3, 4)):
    os.makedirs(out_dir, exist_ok=True)
    found = discover(input_root)
    complete, incomplete = [], {}
    for prefix, by_fold in sorted(found.items()):
        missing = [f for f in folds if f not in by_fold]
        if missing:
            incomplete[prefix] = missing
            continue
        for f, src in by_fold.items():
            dst = os.path.join(out_dir, f"{prefix}_f{f}")
            os.makedirs(dst, exist_ok=True)
            for name in ("preds.npz", "meta.json"):
                if os.path.exists(os.path.join(src, name)):
                    shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
        complete.append(prefix)
    return complete, incomplete


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--write_hashes", action="store_true")
    p.add_argument("--index_csv", required=True)
    p.add_argument("--splits_dir", required=True)
    p.add_argument("--out", default="split_hashes.json")
    p.add_argument("--input_root", default="/kaggle/input")
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--allow_hash_mismatch", action="store_true")
    a = p.parse_args()
    local = split_hashes(a.index_csv, a.splits_dir)
    if a.write_hashes:
        json.dump(local, open(a.out, "w"), indent=1)
        print("wrote", a.out, local)
        return
    seen, problems = check_hashes(a.input_root, local)
    print(f"split_hashes.json files found: {seen}")
    if problems and not a.allow_hash_mismatch:
        raise SystemExit("SPLIT MISMATCH -- do not merge:\n  " + "\n  ".join(problems))
    complete, incomplete = collect(a.input_root, a.out_dir, tuple(int(x) for x in a.folds.split(",")))
    print("complete runs :", complete)
    print("incomplete    :", incomplete, "(skipped)")
    if not complete:
        raise SystemExit("no complete run found under " + a.input_root)
    open(os.path.join(a.out_dir, "RUNS.txt"), "w").write(",".join(complete))


if __name__ == "__main__":
    main()
