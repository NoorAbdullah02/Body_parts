"""Pure-numpy tests of the class-map / transfer-metric logic in src/cross_eval.py (no torch, no data needed).
Run from experiments/v2:   python test_cross_eval.py
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("UNIFESP_DATASET", "unifesp")
from src import cross_eval as X   # noqa: E402


def test_names_and_groups():
    cols = ["label_0_Abdomen", "label_12_Lower_Leg", "label_5_Elbow", "label_21_Wrist"]
    assert X.class_names_from_columns(cols) == ["Abdomen", "Elbow", "Lower Leg", "Wrist"]      # sorted by index
    g = X.shared_groups(X.GROUPS_UNIFESP, X.GROUPS_MURA)
    assert g == ["elbow", "finger", "forearm", "hand", "shoulder", "wrist"], g                 # Humerus / Hip / leg not shared
    g2 = X.shared_groups(X.GROUPS_UNIFESP, X.GROUPS_FRACATLAS)
    assert g2 == ["hand", "hip", "leg", "shoulder"], g2


def test_group_probabilities_and_metrics():
    names = ["Elbow", "Knee", "Lower Leg", "Chest", "Hand"]
    groups = ["elbow", "hand", "leg"]
    prob = np.array([[0.7, 0.1, 0.1, 0.05, 0.05],       # elbow
                     [0.05, 0.1, 0.1, 0.05, 0.7],       # hand
                     [0.05, 0.4, 0.35, 0.1, 0.1],       # leg: Knee + Lower Leg must be SUMMED (0.75) -> leg
                     [0.0, 0.0, 0.0, 1.0, 0.0]])       # Chest only -> no shared mass; must not crash (all zeros)
    pg = X.group_probabilities(prob, names, X.GROUPS_UNIFESP, groups, "sum")
    assert pg.shape == (4, 3) and np.allclose(pg[:3].sum(1), 1) and pg[2].argmax() == 2
    assert np.allclose(pg[3], 0)
    y = np.array([0, 1, 2, 1])                # last image is Chest-only -> argmax of zeros = group 0 -> wrong
    m = X.transfer_metrics(pg, y, groups)
    assert m["n_images"] == 4 and abs(m["accuracy"] - 0.75) < 1e-9, m["accuracy"]
    assert np.array(m["confusion"]).sum() == 4
    assert abs(m["per_group_recall"]["hand"] - 0.5) < 1e-9, m["per_group_recall"]   # 1 of 2 hand images found
    # perfect case
    pg2 = np.eye(3)[[0, 1, 2, 2]]
    m2 = X.transfer_metrics(pg2, np.array([0, 1, 2, 2]), groups)
    assert m2["accuracy"] == 1.0 and m2["macro_f1"] == 1.0


def test_max_head():
    names = ["Knee", "Lower Leg", "Hand"]
    pg = X.group_probabilities(np.array([[0.6, 0.5, 0.1]]), names, X.GROUPS_UNIFESP, ["hand", "leg"], "max")
    assert abs(pg[0, 1] - 0.6 / 0.7) < 1e-9           # sigmoid head: max inside the group, then renormalise


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("[pass]", k)
    print("cross_eval pure tests passed")
