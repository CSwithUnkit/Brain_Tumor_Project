"""Regression: dashboard inference must match the training input convention.

Training loads every image as 3-channel (cv2.IMREAD_COLOR, FR-007/FR-008).
A grayscale upload used to reach the enhancement stage as a native 2D array,
which takes a *different* enhancement code path (direct) than 3-channel input
(luminance-only LAB) -- a silent train/serve skew that shifted classifier
confidence (observed: 36.3% vs 86.9% on the same synthetic scan).

_run_pipeline must replicate non-RGB inputs to RGB *before* enhancement, so a
grayscale scan is processed exactly like its RGB-replicated twin.
"""

import numpy as np
import torch
from PIL import Image

import dashboard.app as app
from enhancement.pipeline import EnhancementAblationManager


class _StubDevice:
    type = "cpu"


class _StubUNet(torch.nn.Module):
    def forward(self, x):
        b, _, h, w = x.shape
        return torch.zeros(b, 1, h, w)


class _StubClassifier(torch.nn.Module):
    def forward(self, x):
        b = x.shape[0]
        return torch.tensor([[0.1, 2.0, 0.3, 0.2]] * b)


class _RecordingEnhancer(EnhancementAblationManager):
    """Captures the ndim of the array handed to the enhancement stage."""

    last_ndim = None

    def process(self, image, variant=None):
        _RecordingEnhancer.last_ndim = image.ndim
        return super().process(image, variant=variant)


class _StubGradCAM:
    def __init__(self, *args, **kwargs):
        pass

    def generate_heatmap(self, tensor, target_category=None):
        return np.zeros((8, 8), dtype=np.float32)


class _DummyStatus:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, *args, **kwargs):
        pass


def _install_stubs(monkeypatch):
    monkeypatch.setattr(
        app,
        "load_models",
        lambda pipeline: (_StubUNet(), _StubClassifier(), torch.device("cpu")),
    )
    monkeypatch.setattr(app, "EnhancementAblationManager", _RecordingEnhancer)
    monkeypatch.setattr(app, "BrainTumorGradCAM", _StubGradCAM)
    monkeypatch.setattr(app.st, "status", lambda *a, **k: _DummyStatus())
    monkeypatch.setattr(app.st, "write", lambda *a, **k: None)


def _grayscale_scan():
    rng = np.random.default_rng(7)
    return Image.fromarray((rng.random((160, 200)) * 255).astype(np.uint8), mode="L")


def test_grayscale_upload_enhanced_as_three_channel(monkeypatch):
    """The enhancement stage must receive a 3-channel array (training parity)."""
    _install_stubs(monkeypatch)
    app._run_pipeline(_grayscale_scan(), "Exp 2: Enhanced")
    assert _RecordingEnhancer.last_ndim == 3


def test_grayscale_matches_rgb_twin(monkeypatch):
    """A grayscale scan must produce identical probabilities to its RGB twin."""
    _install_stubs(monkeypatch)
    gray = _grayscale_scan()
    r_gray = app._run_pipeline(gray, "Exp 2: Enhanced")
    r_rgb = app._run_pipeline(gray.convert("RGB"), "Exp 2: Enhanced")
    np.testing.assert_allclose(r_gray["probs"], r_rgb["probs"], rtol=0, atol=0)
    assert r_gray["pred_class"] == r_rgb["pred_class"]


def test_grayscale_matches_rgb_twin_all_pipelines(monkeypatch):
    """Parity holds for every experiment path (Exp 1 raw / Exp 2 / Exp 3 masked)."""
    _install_stubs(monkeypatch)
    gray = _grayscale_scan()
    for pipeline in ("Exp 1: Raw", "Exp 2: Enhanced", "Exp 3: Seg-guided"):
        r_gray = app._run_pipeline(gray, pipeline)
        r_rgb = app._run_pipeline(gray.convert("RGB"), pipeline)
        np.testing.assert_allclose(r_gray["probs"], r_rgb["probs"], rtol=0, atol=0)
