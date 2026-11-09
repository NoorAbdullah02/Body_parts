"""Correctness tests for src/explain.py (CPU, torch needed). Run from experiments/v2: python test_explain.py
1. A TOY model whose class-0 score depends ONLY on the top-left quadrant: Grad-CAM must light up that quadrant and the deletion test must
   beat random-order deletion (this proves the metric code points the right way);
2. target-layer / layout detection for the real architectures used in the project (CNN NCHW and Swin NHWC)."""
import os, sys
import numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("UNIFESP_DATASET", "unifesp"); os.environ.setdefault("UNIFESP_OUTPUT_DIR", "/tmp/_explain_test_out")
from src import explain as X   # noqa: E402

N = 22


class Toy(nn.Module):
    """class 0 logit = mean of the top-left quadrant of feature channel 0; every other logit is 0."""
    def __init__(self):
        super().__init__()
        self.feat = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        with torch.no_grad():
            self.feat.weight.zero_(); self.feat.weight[0, :, 1, 1] = 1.0 / 3        # channel 0 = mean of the 3 colour channels
    def forward(self, x):
        f = torch.relu(self.feat(x))                                                 # (B,4,S,S): the "spatial feature map"
        S = f.shape[-1] // 2
        z = torch.zeros(x.shape[0], N)
        z[:, 0] = 10 * f[:, 0, :S, :S].mean((1, 2)) - 2.0
        return z, torch.zeros(x.shape[0], N)


def test_toy_cam_and_deletion():
    torch.manual_seed(0)
    m = Toy().eval(); S = 32
    x = 0.2 + 0.05 * torch.rand(1, 3, S, S)          # dim background ...
    x[..., :12, :12] = 1.0                           # ... and a bright blob inside the top-left quadrant (where the model looks)
    eng = X.GradCAM(m, S)
    assert eng.name == "feat" and not eng.channels_last
    cam, sig, _ = eng(x, [0])
    cam = cam[0]
    inside, outside = cam[:S // 2, :S // 2].mean(), cam[S // 2:, S // 2:].mean()
    assert inside > outside + 0.3, (inside, outside)                                 # map concentrates in the quadrant that matters
    up = X.upsample(cam, S)
    s_cam, s_rnd, p0 = X.deletion_scores(m, x, up, 0, steps=(0, .1, .2, .3, .4, .5), n_random=5, grid=cam.shape)
    assert s_cam < s_rnd - 0.02, (s_cam, s_rnd)                                       # deleting salient pixels first hurts more than random
    # a map pointing at the WRONG quadrant must score worse (metric has the right sign)
    wrong = np.zeros((S, S)); wrong[S // 2:, S // 2:] = 1
    s_wrong, _, _ = X.deletion_scores(m, x, wrong, 0, steps=(0, .1, .2, .3, .4, .5), n_random=5)
    assert s_wrong > s_cam, (s_wrong, s_cam)
    eng.close()
    print(f"toy: CAM inside/outside {inside:.2f}/{outside:.2f}; deletion score CAM {s_cam:.3f} < random {s_rnd:.3f} < wrong-map {s_wrong:.3f}")


def test_spearman_and_ci():
    a = np.arange(100.0)
    assert abs(X.spearman(a, a) - 1) < 1e-9 and abs(X.spearman(a, -a) + 1) < 1e-9 and X.spearman(a, np.zeros(100)) == 0.0
    ci = X.boot_ci(np.random.default_rng(0).normal(0.2, 0.1, 200), n=300)
    assert ci["ci_low"] < ci["mean"] < ci["ci_high"]


def test_layouts():
    from src.train_v2 import build_model
    expect = {"swin_tiny": 224, "convnext_tiny": 256, "resnet50": 256, "efficientnet_b0": 224, "densenet121": 256, "vgg19": 224}
    for name, size in expect.items():
        m = build_model(name, pretrained=False).eval()
        n, cl = X.auto_target(m, size)
        eng = X.GradCAM(m, size)
        cam, sig, soft = eng(torch.randn(2, 3, size, size), [3, 5])
        eng.close()
        assert cam.ndim == 3 and cam.shape[0] == 2 and cam.min() >= 0 and cam.max() <= 1 + 1e-6, (name, cam.shape)
        print(f"{name:16s} target={n:24s} channels_last={cl}  cam grid={cam.shape[1]}x{cam.shape[2]}")
        if name == "swin_tiny":
            assert cl is True, "Swin feature maps are NHWC"


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("[pass]", k)
    print("explain tests passed")
