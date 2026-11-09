"""
Step 2: plain multilabel loss with inverse-frequency class weighting
(dataset is heavily imbalanced: Chest=738 vs Clavicles=9 images).

Step 3 will add a hierarchical variant in losses_hierarchical.py that groups
labels into anatomical super-categories (spine, upper limb, lower limb, ...)
and penalizes cross-group errors more than within-group ones.
"""
import numpy as np
import torch
import torch.nn as nn


def compute_pos_weight(label_matrix, cap=20.0):
    """label_matrix: (N, n_classes) 0/1 array from the train split only."""
    n = label_matrix.shape[0]
    pos = label_matrix.sum(axis=0)
    pos = np.clip(pos, 1, None)  # avoid div-by-zero for classes with 0 positives in a fold
    weight = (n - pos) / pos
    weight = np.clip(weight, 1.0, cap)  # cap so the rarest class doesn't blow up the loss scale
    return torch.tensor(weight, dtype=torch.float32)


def build_loss(label_matrix=None, device="cpu"):
    pos_weight = compute_pos_weight(label_matrix) if label_matrix is not None else None
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
