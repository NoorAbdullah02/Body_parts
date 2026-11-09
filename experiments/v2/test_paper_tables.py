"""Test of ../make_paper_tables.py on synthetic result folders. Run from experiments/v2: python test_paper_tables.py"""
import json, os, shutil, sys, tempfile
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
shutil.copy(os.path.join(HERE, "..", "evalkit.py"), os.path.join(HERE, "src", "evalkit.py"))
import test_run_ensemble as T                       # noqa: E402
from src import run_ensemble as RE, ablation_report as AR   # noqa: E402
import make_paper_tables as M                       # noqa: E402


def main():
    tmp = tempfile.mkdtemp()
    try:
        out = T.build(tmp)
        class A: pass
        a = A(); a.out_dir, a.runs, a.folds = out, "strong_s224,weak_s256,mid_s224", "0,1,2,3,4"
        a.index_csv, a.cv_folds_csv = os.path.join(tmp, "index.csv"), os.path.join(tmp, "cv_folds.csv")
        a.holdout_csv, a.sample_submission = os.path.join(tmp, "holdout.csv"), os.path.join(tmp, "sample.csv")
        a.result_dir, a.n_boot, a.caruana_iters, a.min_pos = os.path.join(tmp, "res_main"), 100, 10, 20.0
        RE.run(a)
        root = os.path.join(tmp, "root"); os.makedirs(os.path.join(root, "results_ablation")); os.makedirs(os.path.join(root, "transfer"))
        os.makedirs(os.path.join(root, "results_mura")); os.makedirs(os.path.join(root, "results_covid"))
        shutil.copy(os.path.join(a.result_dir, "results.json"), os.path.join(root, "main_results.json"))
        shutil.copy(os.path.join(a.result_dir, "per_run.csv"), os.path.join(root, "main_per_run.csv"))
        json.dump({"strong": ["strong_s224"], "weak": ["weak_s256"], "avg": ["strong_s224", "mid_s224"]}, open(os.path.join(tmp, "cfg.json"), "w"))
        b = A(); b.out_dir, b.config_json, b.reference = out, os.path.join(tmp, "cfg.json"), "strong"
        b.index_csv, b.cv_folds_csv, b.result_dir, b.folds, b.n_boot = a.index_csv, a.cv_folds_csv, os.path.join(root, "results_ablation"), "0,1,2,3,4", 100
        AR.run(b)
        pd.DataFrame([{"run": f"hybrid_gate_ln_s224_f{k}", "cnn_gate_weight_mean": 0.4 + 0.01 * k, "std": 0.1} for k in range(5)] +
                     [{"run": f"hybrid_gate_s224_f{k}", "cnn_gate_weight_mean": 0.0001, "std": 0.0} for k in range(5)]).to_csv(os.path.join(root, "results_ablation", "gate_weights.csv"), index=False)
        est = lambda v: {"estimate": v, "ci_low": v - .01, "ci_high": v + .01}
        json.dump({"accuracy": est(0.97), "macro_f1": est(0.96)}, open(os.path.join(root, "results_mura", "shah_comparison.json"), "w"))
        h = json.load(open(os.path.join(root, "main_results.json")))
        json.dump(h, open(os.path.join(root, "results_covid", "results.json"), "w"))
        pd.DataFrame([{"source": "unifesp", "target": "mura", "head": "softmax", "n": 100, "accuracy": 0.8}]).to_csv(os.path.join(root, "transfer", "summary.csv"), index=False)
        pd.DataFrame([{"model": "convnext_tiny_s256", "gain [95% CI]": "0.05 [0.03, 0.07]", "rho trained-vs-random": 0.1}]).to_csv(os.path.join(root, "explain_summary.csv"), index=False)
        json.dump({"stage1": {"private": 0.92563, "public": 0.92704}, "s2a": {"private": 0.931, "public": 0.93}}, open(os.path.join(root, "scores.json"), "w"))
        outp = os.path.join(tmp, "paper_tables.md")
        sections, skipped = M.build(root)
        assert len(sections) == 10 and not skipped, (len(sections), skipped)
        sys.argv = ["x", "--root", root, "--out", outp, "--latex"]; M.main()
        txt = open(outp).read()
        for need in ("Table 1.", "Table 3.", "0.92563", "Table 4.", "Leak gap", "Table 6.", "Table 7.", "Table 10.", "0.9643", "\\begin{table}"):
            assert need in txt, need
        # missing inputs only skip their table
        shutil.rmtree(os.path.join(root, "results_mura")); os.remove(os.path.join(root, "scores.json"))
        s2, sk2 = M.build(root); assert any("shah" in x for x in sk2) and any("scores" in x for x in sk2) and len(s2) == 9, (len(s2), sk2)
        print("paper tables test passed;", len(sections), "tables;", "skip handling OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()
