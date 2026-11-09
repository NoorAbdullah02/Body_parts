"""
Step 3 novelty component: hierarchical multilabel loss.

Plain BCE (Step 2) treats every misclassification equally -- confusing
Lumbar Spine for Thoracic Spine costs the same as confusing it for Hand.
That's not how these errors matter clinically or visually: adjacent/related
body parts get confused far more often (our EDA's top co-occurring pairs
were Lumbar+Thoracic Spine and Hand+Wrist), and a router that sends a spine
image to the wrong spine model is a much smaller failure than sending it to
a hand model.

We add a second loss term at the coarser anatomical-group level (see
hierarchy.py: Head/Spine/Trunk/UpperLimb/LowerLimb/Other) computed from the
same logits, so the model is pushed to at least get the broad region right
even when it's unsure of the exact class. This follows the same idea as
Cheng et al.'s hierarchical loss (BMC Med Imaging, 2025/2026) adapted to
UNIFESP's 22-class/6-group structure.
"""
import torch
import torch.nn as nn

from .hierarchy import N_GROUPS, build_label_to_group_matrix
from .losses import compute_pos_weight


class HierarchicalBCELoss(nn.Module):
    def __init__(self, pos_weight=None, group_weight=0.3, device="cpu"):
        super().__init__()
        self.group_weight = group_weight
        self.fine_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        label_to_group = torch.tensor(build_label_to_group_matrix())  # (N_CLASSES, N_GROUPS)
        self.register_buffer("label_to_group", label_to_group)

    def forward(self, logits, targets):
        fine = self.fine_loss(logits, targets)

        # Group-level logit = max of the fine LOGITS belonging to that group
        # (not sigmoid probabilities -> plain BCELoss/binary_cross_entropy on
        # probabilities is explicitly unsafe under autocast/fp16, which is
        # what actually crashed on Kaggle: "torch.nn.functional.
        # binary_cross_entropy ... are unsafe to autocast". Staying in logit
        # space and using binary_cross_entropy_with_logits avoids that, and
        # max-of-logits is still a monotonic, differentiable "is any class in
        # this group present" signal.
        not_in_group = (self.label_to_group == 0).T            # (N_GROUPS, N_CLASSES) bool
        group_logit = torch.stack([
            logits.masked_fill(not_in_group[g], float("-inf")).max(dim=1).values
            for g in range(N_GROUPS)
        ], dim=1)                                               # (B, N_GROUPS)
        group_target = (targets @ self.label_to_group).clamp(max=1.0)  # (B, N_GROUPS)

        group = nn.functional.binary_cross_entropy_with_logits(group_logit, group_target)
        return fine + self.group_weight * group


def build_hierarchical_loss(label_matrix=None, group_weight=0.3, device="cpu"):
    pos_weight = compute_pos_weight(label_matrix) if label_matrix is not None else None
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    return HierarchicalBCELoss(pos_weight=pos_weight, group_weight=group_weight).to(device)
