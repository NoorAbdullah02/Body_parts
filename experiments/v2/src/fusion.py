"""
fusion -- DenseNet121 (CNN) + Swin-Tiny (Transformer) fusion variants for the gate-collapse ablation.

Background: in the earlier notebooks the learned gate put ~0.0001 weight on the CNN branch in 5/5 CV folds
(i.e. it silently became a plain Swin-Tiny).  Hypothesis (UNTESTED): the gate sees raw pooled features whose scales
differ (DenseNet: post-ReLU average-pool; Swin: LayerNorm output), so the softmax saturates.  Variants:

    hybrid_gate     original: gate on raw concatenated features             (reproduces the collapse)
    hybrid_gate_ln  LayerNorm both feature vectors before gate AND projections (the proposed fix)
    hybrid_avg      fixed 0.5 / 0.5 average of the projected branches        (no gate at all)
    hybrid_cnn      CNN branch only (same projection + head)
    hybrid_swin     Swin branch only (same projection + head)

All variants return (sigmoid_logits, softmax_logits) like TwoHeadNet, so train_v2 / run_ensemble need no change.
``gate_weights(x)`` returns the (B, 2) branch weights [cnn, swin] for the analysis.
Input size must be 224 (Swin-Tiny window7_224).
"""
import torch
import torch.nn as nn

from . import config as C

N = C.N_CLASSES
VARIANTS = ("hybrid_gate", "hybrid_gate_ln", "hybrid_avg", "hybrid_cnn", "hybrid_swin")


class FusionNet(nn.Module):
    def __init__(self, variant, pretrained=True, fusion_dim=512, drop=0.2,
                 cnn_name="densenet121", swin_name="swin_tiny_patch4_window7_224"):
        super().__init__()
        assert variant in VARIANTS, f"unknown fusion variant {variant}; options {VARIANTS}"
        import timm
        self.variant = variant
        self.cnn = None if variant == "hybrid_swin" else timm.create_model(cnn_name, pretrained=pretrained, num_classes=0)
        self.swin = None if variant == "hybrid_cnn" else timm.create_model(swin_name, pretrained=pretrained, num_classes=0)
        cd = self.cnn.num_features if self.cnn is not None else 0
        sd = self.swin.num_features if self.swin is not None else 0
        self.ln_c = nn.LayerNorm(cd) if (variant == "hybrid_gate_ln" and cd) else None
        self.ln_s = nn.LayerNorm(sd) if (variant == "hybrid_gate_ln" and sd) else None
        self.proj_c = nn.Linear(cd, fusion_dim) if cd else None
        self.proj_s = nn.Linear(sd, fusion_dim) if sd else None
        self.gate = (nn.Sequential(nn.Linear(cd + sd, 128), nn.ReLU(inplace=True), nn.Linear(128, 2))
                     if variant in ("hybrid_gate", "hybrid_gate_ln") else None)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(fusion_dim, 2 * N)

    def _features(self, x):
        fc = self.cnn(x) if self.cnn is not None else None
        fs = self.swin(x) if self.swin is not None else None
        if self.ln_c is not None:
            fc, fs = self.ln_c(fc), self.ln_s(fs)
        return fc, fs

    def _weights(self, fc, fs):
        ref = fc if fc is not None else fs
        B = ref.size(0)
        if self.variant == "hybrid_avg":
            return ref.new_full((B, 2), 0.5)
        if self.variant == "hybrid_cnn":
            return ref.new_tensor([1.0, 0.0]).expand(B, 2)
        if self.variant == "hybrid_swin":
            return ref.new_tensor([0.0, 1.0]).expand(B, 2)
        return torch.softmax(self.gate(torch.cat([fc, fs], dim=1)), dim=1)

    def forward(self, x):
        fc, fs = self._features(x)
        w = self._weights(fc, fs)
        fused = 0
        if fc is not None:
            fused = fused + w[:, 0:1] * self.proj_c(fc)
        if fs is not None:
            fused = fused + w[:, 1:2] * self.proj_s(fs)
        z = self.head(self.drop(fused))
        return z[:, :N], z[:, N:]

    @torch.no_grad()
    def gate_weights(self, x):
        fc, fs = self._features(x)
        return self._weights(fc, fs)
