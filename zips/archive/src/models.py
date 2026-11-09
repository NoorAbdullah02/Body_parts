"""
Model factory.

Step 2 baselines (reproducing Krishnan & Padmavathi 2025's comparison, plus
modern backbones): vgg16, inception_v3, resnet50, densenet121, efficientnet_b3,
efficientnetv2_s, convnext_tiny, swin_tiny.

Step 3 novelty: HybridCNNTransformer (see models_hybrid.py) is registered
separately so baseline runs stay untouched while that's developed.
"""
import timm
import torch.nn as nn

from . import config as C

BASELINE_BACKBONES = {
    "vgg16": "vgg16",
    "inception_v3": "inception_v3",
    "resnet50": "resnet50",
    "densenet121": "densenet121",
    "efficientnet_b3": "efficientnet_b3",
    "efficientnetv2_s": "tf_efficientnetv2_s",
    "convnext_tiny": "convnext_tiny",
    "swin_tiny": "swin_tiny_patch4_window7_224",
}


def build_model(name, n_classes=C.N_CLASSES, pretrained=True):
    if name not in BASELINE_BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Options: {list(BASELINE_BACKBONES)}")
    timm_name = BASELINE_BACKBONES[name]
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=n_classes)
    return model


def input_size_for(name):
    # swin_tiny_patch4_window7_224 is fixed at 224; everything else we train at C.IMG_SIZE (256)
    if name == "swin_tiny":
        return 224
    return C.IMG_SIZE
