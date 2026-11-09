"""End-to-end test of src/ablation_report.py on synthetic fold outputs. Run from experiments/v2: python test_ablation_report.py"""
import json, os, shutil, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
shutil.copy(os.path.join(HERE, "..", "evalkit.py"), os.path.join(HERE, "src", "evalkit.py"))
import test_run_ensemble as T            # noqa: E402
from src import ablation_report as AR    # noqa: E402


def main():
    tmp = tempfile.mkdtemp()
    try:
        out = T.build(tmp)
        cfg = {"strong": ["strong_s224"], "weak": ["weak_s256"], "two_run_avg": ["strong_s224", "mid_s224"]}
        json.dump(cfg, open(os.path.join(tmp, "cfg.json"), "w"))
        class A: pass
        a = A(); a.out_dir, a.config_json, a.reference = out, os.path.join(tmp, "cfg.json"), "strong"
        a.index_csv, a.cv_folds_csv = os.path.join(tmp, "index.csv"), os.path.join(tmp, "cv_folds.csv")
        a.result_dir, a.folds, a.n_boot = os.path.join(tmp, "res"), "0,1,2,3,4", 200
        r = AR.run(a)
        for f in ("ablation_configs.csv", "ablation_decoding.csv", "ablation_heads_tta.csv", "ablation.md"):
            assert os.path.exists(os.path.join(a.result_dir, f)), f
        c = r["configs"].set_index("config")
        print(c[["hold_sample_f1", "hold_ci_low", "hold_ci_high", "fold_models_mean", "fold_models_std", "diff_vs_ref", "p_value"]].round(4).to_string())
        print(r["decoding"].round(4).to_string(index=False)); print("leak gap", round(r["leak_gap"], 4))
        assert c.loc["strong", "hold_sample_f1"] > c.loc["weak", "hold_sample_f1"]          # strong beats weak
        assert c.loc["weak", "diff_vs_ref"] < 0 and 0.0 <= c.loc["weak", "p_value"] <= 1.0   # (n=200 images: not necessarily significant)
        assert c.loc["weak", "diff_ci_low"] <= c.loc["weak", "diff_vs_ref"] <= c.loc["weak", "diff_ci_high"]
        assert "diff_vs_ref" in c.columns and c.loc["strong"].isna()["diff_vs_ref"]          # reference has no diff row
        assert c.loc["strong", "hold_ci_low"] <= c.loc["strong", "hold_sample_f1"] <= c.loc["strong", "hold_ci_high"]
        d = r["decoding"].set_index("decoding")
        assert d.iloc[1]["hold_sample_f1"] >= d.iloc[0]["hold_sample_f1"] - 1e-9             # min-one never hurts here
        assert r["leak_gap"] >= -0.02                                                        # leaky tuning cannot look much worse
        assert "Leak gap" in open(os.path.join(a.result_dir, "ablation.md")).read()
        print("\nablation_report test passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()
