"""
Anatomical grouping of the 22 UNIFESP classes, used by the hierarchical loss
(losses_hierarchical.py). Grouping follows the same idea as Cheng et al.
2025/2026 (BMC Med Imaging) -- broad anatomical region as a coarser level
above the fine-grained label -- adapted to UNIFESP's 22 classes instead of
their 36 SNOMED codes.
"""
from . import config as C

GROUPS = {
    "Head": [17, 18],                       # Sinus, Skull
    "Spine": [2, 13, 20],                   # Cervical, Lumbar, Thoracic Spine
    "Trunk": [0, 3, 15],                     # Abdomen, Chest, Pelvis
    "UpperLimb": [4, 5, 7, 8, 9, 16, 21],    # Clavicles, Elbow, Finger, Forearm, Hand, Shoulder, Wrist
    "LowerLimb": [1, 6, 10, 11, 12, 19],     # Ankle, Feet, Hip, Knee, Lower Leg, Thigh
    "Other": [14],                          # Others
}

GROUP_NAMES = list(GROUPS.keys())
N_GROUPS = len(GROUP_NAMES)

# label_id (0..21) -> group index (0..5)
LABEL_TO_GROUP = {}
for g_idx, (g_name, label_ids) in enumerate(GROUPS.items()):
    for lid in label_ids:
        LABEL_TO_GROUP[lid] = g_idx

assert len(LABEL_TO_GROUP) == C.N_CLASSES, "every one of the 22 classes must map to exactly one group"


def build_label_to_group_matrix():
    """(N_CLASSES, N_GROUPS) 0/1 matrix: row k is a one-hot for class k's group.
    Multiplying a (batch, N_CLASSES) prediction/target by this gives the
    per-group aggregate (used by the hierarchical loss)."""
    import numpy as np
    m = np.zeros((C.N_CLASSES, N_GROUPS), dtype="float32")
    for lid, g_idx in LABEL_TO_GROUP.items():
        m[lid, g_idx] = 1.0
    return m
