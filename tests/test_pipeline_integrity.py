"""Regression tests for the 2026-09-27 pipeline-integrity audit fixes.

Covers: C1 (PMRAM preprocessing must match training), C2 (Exp3 fail-hard
without U-Net), C3 (PMRAM fail-hard without checkpoint + provenance),
M1 (dashboard Exp3 train/serve parity), M2 (reports consume test metrics),
M4 (Phase I protocol documentation), M5/M6/M7 (ingestion determinism,
mask-pairing collisions, case-insensitive dedup), and the cache fail-hard
(exp2/exp3/U-Net must train on enhanced images, never silently raw).
"""
import json
import os
import sys

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from validation.external_pmram import (
    PMRAMValidator,
    _folder_to_class,
    _preprocess_image,
    _sha256_of_file,
)
from classification.run_experiments import load_unet_for_exp3, apply_exp3_guidance
from classification.masking_utils import apply_exp3_guidance_numpy
from data.dataset_preprocessor import find_cached_enhanced
from data.dataset_ingestion import DatasetIngestor


# ── C1: PMRAM preprocessing matches the model's training distribution ────────

def _dummy_bgr():
    rng = np.random.default_rng(3)
    img = np.zeros((300, 300, 3), np.uint8)
    cv2.circle(img, (150, 150), 70, (140, 140, 140), -1)
    return np.clip(img.astype(int) + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)


def test_pmram_preprocess_enhanced_matches_training_pipeline():
    from enhancement.pipeline import EnhancementAblationManager
    img = _dummy_bgr()
    enhancer = EnhancementAblationManager()
    got = _preprocess_image(img, enhanced=True, enhancer=enhancer)

    # Manual replication of the training distribution: RGB → 256 → /255 →
    # WPT→LMMSE→CLAHE → clip → ImageNet normalize.
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rs = cv2.resize(rgb, (256, 256)).astype(np.float32) / 255.0
    enh = np.clip(enhancer.process(rs), 0, 1)
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    expected = torch.from_numpy(((enh - mean) / std).transpose(2, 0, 1)).unsqueeze(0)
    assert got.shape == (1, 3, 256, 256)
    assert torch.allclose(got, expected, atol=1e-6)


def test_pmram_preprocess_enhanced_differs_from_raw():
    from enhancement.pipeline import EnhancementAblationManager
    img = _dummy_bgr()
    enhancer = EnhancementAblationManager()
    raw = _preprocess_image(img, enhanced=False)
    enh = _preprocess_image(img, enhanced=True, enhancer=enhancer)
    # The old (wrong-distribution) behavior must not equal the fixed one.
    assert not torch.allclose(raw, enh, atol=1e-3)


def test_pmram_preprocess_enhanced_without_pipeline_is_a_bug():
    with pytest.raises(RuntimeError):
        _preprocess_image(_dummy_bgr(), enhanced=True, enhancer=None)


def test_pmram_validator_rejects_bad_preprocessing_arg(tmp_path):
    with pytest.raises(ValueError):
        PMRAMValidator("m.pth", "x.csv", "y.json", pmram_root=str(tmp_path),
                       preprocessing="sometimes")


# ── C3: PMRAM fails hard on missing checkpoint; records provenance ───────────

def test_pmram_load_model_fails_hard_on_missing_checkpoint(tmp_path):
    v = PMRAMValidator("/nonexistent/model.pth", "x.csv", "y.json",
                       pmram_root=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        v.load_model()


def test_sha256_of_file(tmp_path):
    p = tmp_path / "ckpt.bin"
    p.write_bytes(b"fake-checkpoint-bytes")
    import hashlib
    assert _sha256_of_file(str(p)) == hashlib.sha256(b"fake-checkpoint-bytes").hexdigest()


def test_folder_to_class_abnormal_is_not_no_tumor():
    # "abnormal" contains the substring "normal" but means the opposite.
    assert _folder_to_class("abnormal") is None
    assert _folder_to_class("ABNORMAL_scans") is None
    # Sanity: real variants still map correctly.
    assert _folder_to_class("512Glioma") == 0
    assert _folder_to_class("Meningioma") == 1
    assert _folder_to_class("pituitary") == 2
    assert _folder_to_class("No_Tumor") == 3
    assert _folder_to_class("notumor") == 3
    assert _folder_to_class("abnormal_glioma") == 0  # specific class wins


# ── C2: Exp3 fails hard without a U-Net checkpoint ───────────────────────────

def test_exp3_unet_loader_fails_hard(tmp_path):
    device = torch.device("cpu")
    with pytest.raises(FileNotFoundError):
        load_unet_for_exp3(str(tmp_path / "nope.pth"), device)


# ── M1: dashboard Exp3 guidance == training guidance (train/serve parity) ───

def _synthetic_case(seed=0, tumor=True):
    rng = np.random.default_rng(seed)
    enh = np.clip(rng.normal(110, 25, (256, 256, 3)), 0, 255).astype(np.uint8)
    mask_prob = np.zeros((256, 256), np.float32)
    if tumor:
        cv2.circle(mask_prob, (128, 128), 40, 0.95, -1)
        mask_prob += rng.normal(0, 0.02, mask_prob.shape).astype(np.float32)
        mask_prob = np.clip(mask_prob, 0, 1)
    return enh, mask_prob


def test_exp3_guidance_numpy_matches_torch_training():
    """The dashboard's numpy path must be bit-identical to training's torch path."""
    from segmentation.unet_model import UNet
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    torch.manual_seed(0)
    unet = UNet(n_channels=3, n_classes=1).eval()

    for seed, tumor in [(0, True), (1, False)]:
        enh, _ = _synthetic_case(seed, tumor)
        tensor = torch.from_numpy(
            ((enh.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)
        ).float().unsqueeze(0)
        with torch.no_grad():
            mask_prob_t = torch.sigmoid(unet(tensor))  # (1,1,256,256)

        torch_out = apply_exp3_guidance(tensor, unet, soft=False)
        numpy_out = apply_exp3_guidance_numpy(
            enh, mask_prob_t.squeeze().numpy(), mean, std)

        assert torch_out.shape == (1, 3, 256, 256)
        assert np.allclose(torch_out.numpy(), numpy_out, atol=1e-5), \
            f"train/serve mismatch (tumor={tumor})"


def test_exp3_guidance_empty_mask_passes_through_unmasked():
    """< 50px masks must leave the image untouched in BOTH paths."""
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    enh, _ = _synthetic_case(seed=5, tumor=False)
    tiny = np.zeros((256, 256), np.float32)
    tiny[120:125, 120:125] = 0.9  # 25 px < 50 px guard

    numpy_out = apply_exp3_guidance_numpy(enh, tiny, mean, std)
    plain = ((enh.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)[None]
    assert np.allclose(numpy_out, plain, atol=1e-6), \
        "dashboard darkened an image with a near-empty mask"

    # Torch path with a stub U-Net emitting the same tiny mask.
    class TinyUNet(torch.nn.Module):
        def forward(self, x):
            with np.errstate(divide="ignore"):
                logit = np.log(tiny / np.clip(1 - tiny, 1e-6, None)).astype(np.float32)
            return torch.from_numpy(logit)[None, None]
    tensor = torch.from_numpy(plain).float()
    torch_out = apply_exp3_guidance(tensor, TinyUNet(), soft=False)
    assert np.allclose(torch_out.numpy(), plain, atol=1e-5), \
        "training darkened an image with a near-empty mask"


# ── M5/M6/M7: ingestion determinism, collisions, dedup ──────────────────────

@pytest.fixture()
def fake_brisc(tmp_path, monkeypatch):
    root = tmp_path / "data" / "brisc"
    (root / "glioma").mkdir(parents=True)
    (root / "segmentation_task" / "t1" / "images").mkdir(parents=True)
    (root / "segmentation_task" / "t1" / "masks").mkdir(parents=True)
    img = np.zeros((16, 16), np.uint8)
    cv2.imwrite(str(root / "glioma" / "a.jpg"), img)
    cv2.imwrite(str(root / "glioma" / "b.jpg"), img)
    cv2.imwrite(str(root / "segmentation_task" / "t1" / "images" / "s1.jpg"), img)
    cv2.imwrite(str(root / "segmentation_task" / "t1" / "masks" / "s1_mask.png"), img)
    ing = DatasetIngestor()
    ing.brisc_dir = root
    monkeypatch.setattr(DatasetIngestor, "MIN_CLASSIFICATION_IMAGES", 0)
    monkeypatch.setattr(DatasetIngestor, "MIN_SEGMENTATION_PAIRS", 0)
    return ing, root


def test_ingestion_classification_sorted_and_deduped(fake_brisc):
    ing, root = fake_brisc
    # Symlink with a different name resolving to the same file (M7: dedup).
    os.symlink(str(root / "glioma" / "a.jpg"), str(root / "glioma" / "alias.jpg"))
    data = ing.ingest_brisc()
    paths = [r["path"] for r in data["classification"]]
    assert len(paths) == len(set(paths)) == 2, f"dedup failed: {paths}"
    assert paths == sorted(paths), "ingestion order is not deterministic"


def test_ingestion_duplicate_mask_stem_fails_loud(fake_brisc):
    ing, root = fake_brisc
    (root / "segmentation_task" / "t2" / "masks").mkdir(parents=True)
    img = np.zeros((16, 16), np.uint8)
    # Same stem "s1" in a nested folder — must NOT pair silently.
    cv2.imwrite(str(root / "segmentation_task" / "t2" / "masks" / "s1_mask.png"), img)
    with pytest.raises(RuntimeError, match="Duplicate mask stem"):
        ing.ingest_brisc()


# ── Cache: legacy flat layout is untrusted; misses fail hard ────────────────

def test_find_cached_enhanced_ignores_legacy_flat(tmp_path, monkeypatch):
    cache = tmp_path / "cached_enhanced"
    cache.mkdir()
    # A legacy flat-layout file must NOT be trusted (cross-class collisions).
    (cache / "img_001.jpg").write_bytes(b"junk")
    monkeypatch.chdir(tmp_path)
    assert find_cached_enhanced("data/brisc/glioma/img_001.jpg", str(cache)) is None


def test_find_cached_enhanced_structured_hit(tmp_path, monkeypatch):
    cache = tmp_path / "cached_enhanced" / "glioma"
    cache.mkdir(parents=True)
    (cache / "img_001.jpg").write_bytes(b"junk")
    monkeypatch.chdir(tmp_path)
    hit = find_cached_enhanced("data/brisc/glioma/img_001.jpg",
                               str(tmp_path / "cached_enhanced"))
    assert hit is not None and hit.endswith(os.path.join("glioma", "img_001.jpg"))


# ── M2/M4: reports consume held-out test metrics; honest provenance ─────────

def _write_hist(path, acc, f1, epoch=3):
    hist = [
        {"epoch": 1, "val_metrics": {"accuracy": acc - 0.02, "macro_f1": f1 - 0.02,
                                     "macro_precision": 0.9, "macro_recall": 0.9}},
        {"epoch": epoch, "val_metrics": {"accuracy": acc, "macro_f1": f1,
                                         "macro_precision": 0.9, "macro_recall": 0.9}},
    ]
    with open(path, "w") as f:
        json.dump(hist, f)


def test_phase2_report_uses_test_metrics_and_clean_provenance(tmp_path):
    from evaluation.consolidate_reports import ReportConsolidator
    r = tmp_path / "results"
    r.mkdir()
    _write_hist(r / "metrics_exp1_baseline.json", 0.97, 0.96)
    _write_hist(r / "metrics_exp2_enhanced.json", 0.96, 0.95)
    _write_hist(r / "metrics_exp3_seg_guided.json", 0.90, 0.89)
    for exp, acc in [("exp1_baseline", 0.965), ("exp2_enhanced", 0.955),
                     ("exp3_seg_guided", 0.895)]:
        with open(r / f"metrics_{exp}_test.json", "w") as f:
            json.dump({"accuracy": acc, "macro_f1": acc - 0.01,
                       "macro_precision": 0.9, "macro_recall": 0.9,
                       "num_test_samples": 100}, f)

    ReportConsolidator(project_root=str(tmp_path)).generate_phase2_report()
    md = (tmp_path / "reports" / "PHASE_II_FINAL_REPORT.md").read_text()
    assert "Test Accuracy" in md
    assert "96.50%" in md  # exp1 held-out test accuracy, not validation
    assert "pre-fix" not in md
    assert "HELD-OUT test set" in md


def test_phase2_report_warns_without_test_metrics(tmp_path):
    from evaluation.consolidate_reports import ReportConsolidator
    r = tmp_path / "results"
    r.mkdir()
    _write_hist(r / "metrics_exp1_baseline.json", 0.97, 0.96)

    ReportConsolidator(project_root=str(tmp_path)).generate_phase2_report()
    md = (tmp_path / "reports" / "PHASE_II_FINAL_REPORT.md").read_text()
    assert "pre-fix" in md
    assert "Do not cite" in md


def test_phase1_report_documents_real_protocol(tmp_path):
    from evaluation.consolidate_reports import ReportConsolidator
    r = tmp_path / "results"
    r.mkdir()
    _write_hist(r / "metrics_exp1_baseline.json", 0.97, 0.96)
    _write_hist(r / "metrics_exp2_enhanced.json", 0.96, 0.95)
    meta_dir = tmp_path / "data" / "brisc"
    meta_dir.mkdir(parents=True)
    with open(meta_dir / "brisc_metadata.json", "w") as f:
        json.dump({"classification_count": 6123, "segmentation_count": 4793,
                   "classification": [], "segmentation": []}, f)

    ReportConsolidator(project_root=str(tmp_path)).generate_phase1_report()
    md = (tmp_path / "reports" / "PHASE_I_EVALUATION_REPORT.md").read_text()
    assert "ElasticTransform" in md
    assert "GridDistortion" not in md
    assert "Epochs 6–25" in md
    assert "6,123 classification" in md  # dynamic, not hardcoded 6,000
    assert "15×" not in md
