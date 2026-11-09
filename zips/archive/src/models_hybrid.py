"""
Step 3 novelty component: Hybrid CNN-Transformer with attention-based fusion.

Step 2's two best backbones were complementary: DenseNet121 (best macro_f1
0.875 -- strong on rare classes) and Swin-Tiny (best sample_mean_f1 0.928 --
strong on the common ones). The idea, following the dual-path CNN+Transformer
fusion pattern reported for musculoskeletal radiographs (Shah et al. 2026,
J. Imaging, Xception-Swin): the CNN branch's local conv features (bone edges,
cortical texture) and the Transformer branch's global shifted-window
attention (whole-image context) carry different information, and a learned
gate lets the model weight each per-sample instead of fixing their
contribution.

No prior UNIFESP work (the Krishnan 2025 paper, or any Kaggle competition
solution we found) uses a CNN+Transformer hybrid -- the competition's top
entries used single CNNs (ResNet50/152). This is new for this dataset.
"""
import timm
import torch
import torch.nn as nn

from . import config as C

HYBRID_INPUT_SIZE = 224  # swin_tiny_patch4_window7_224 needs 224; densenet121 is resolution-flexible


class HybridCNNTransformer(nn.Module):
    def __init__(self, n_classes=C.N_CLASSES, cnn_name="densenet121",
                 transformer_name="swin_tiny_patch4_window7_224", fusion_dim=512, pretrained=True):
        super().__init__()
        self.cnn = timm.create_model(cnn_name, pretrained=pretrained, num_classes=0)       # -> pooled feature vector
        self.transformer = timm.create_model(transformer_name, pretrained=pretrained, num_classes=0)

        cnn_dim = self.cnn.num_features
        trans_dim = self.transformer.num_features

        self.cnn_proj = nn.Linear(cnn_dim, fusion_dim)
        self.trans_proj = nn.Linear(trans_dim, fusion_dim)

        # learned per-sample gate over the two branches, from their concatenated raw features
        self.gate = nn.Sequential(
            nn.Linear(cnn_dim + trans_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, n_classes),
        )

    def forward(self, x):
        cnn_feat = self.cnn(x)                  # (B, cnn_dim)
        trans_feat = self.transformer(x)         # (B, trans_dim)

        gate_logits = self.gate(torch.cat([cnn_feat, trans_feat], dim=1))
        gate_weights = torch.softmax(gate_logits, dim=1)  # (B, 2) -- how much to trust CNN vs Transformer

        fused = (gate_weights[:, 0:1] * self.cnn_proj(cnn_feat)
                 + gate_weights[:, 1:2] * self.trans_proj(trans_feat))

        return self.classifier(fused)

    def forward_with_gate(self, x):
        """Same as forward, but also returns the per-sample gate weights --
        useful for Step 5 (explainability): shows whether the model is
        leaning on local texture (CNN) or global context (Transformer) for
        a given image, per class confusion pair."""
        cnn_feat = self.cnn(x)
        trans_feat = self.transformer(x)
        gate_logits = self.gate(torch.cat([cnn_feat, trans_feat], dim=1))
        gate_weights = torch.softmax(gate_logits, dim=1)
        fused = (gate_weights[:, 0:1] * self.cnn_proj(cnn_feat)
                 + gate_weights[:, 1:2] * self.trans_proj(trans_feat))
        return self.classifier(fused), gate_weights
