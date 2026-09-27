#!/usr/bin/env python3
"""Executable Grad-CAM localization stage (M3).

FR-034/036/037/038: quantifies how well each classifier's attention aligns
with radiologist-annotated tumor masks (pixel-level IoU / Dice), on the
HELD-OUT segmentation test split — the same split `segmentation/train_unet.py`
evaluates on (split_dataset(..., random_state=42)).

For every experiment the Grad-CAM heatmap is generated on the EXACT input the
deployed model sees:
  exp1_baseline   → raw image
  exp2_enhanced   → WPT→LMMSE→CLAHE cached image
  exp3_seg_guided → enhanced image + U-Net soft context guidance
                    (identical to classification/run_experiments.py)

Writes results/gradcam_localization_summary.json, consumed by
evaluation/consolidate_reports.py.

Fail-hard rules (no silent fake science):
  - missing classifier checkpoint → FileNotFoundError (never random weights)
  - missing U-Net checkpoint for exp3_* → FileNotFoundError
  - missing enhancement cache entry for exp2/exp3 → FileNotFoundError

Usage:
    python -m explainability.run_localization_eval
    python -m explainability.run_localization_eval --max_samples 100
    python -m explainability.run_localization_eval --experiments exp2_enhanced
"""

import argparse
import json
import logging
import os

import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("xai_localization")

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

EXPERIMENTS = ["exp1_baseline", "exp2_enhanced", "exp3_seg_guided", "exp3_soft_masked"]


def _normalize(rgb_f32: np.ndarray) -> torch.Tensor:
    """(H, W, C) float [0,1] → (1, C, H, W) ImageNet-normalized tensor."""
    norm = (rgb_f32 - _MEAN) / _STD
    return torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0)


def _load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
    return img


def _load_mask_256(mask_path: str) -> np.ndarray:
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    m = cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.float32)


def build_experiment_input(exp: str, image_path: str, device, unet_model=None) -> torch.Tensor:
    """Reproduce each experiment's exact inference input (train/serve parity)."""
    from data.dataset_preprocessor import find_cached_enhanced

    if exp == "exp1_baseline":
        return _normalize(_load_rgb(image_path)).to(device)

    cached = find_cached_enhanced(image_path)
    if cached is None:
        raise FileNotFoundError(
            f"Enhancement cache miss for {image_path}. "
            f"Experiment {exp} trains on cached enhanced images — "
            f"rebuild the cache first (scripts/rerun_full_pipeline.py stage 'cache')."
        )
    tensor = _normalize(_load_rgb(cached)).to(device)

    if exp in ("exp3_seg_guided", "exp3_soft_masked"):
        if unet_model is None:
            raise FileNotFoundError(
                f"Experiment {exp} needs the trained U-Net for segmentation "
                f"guidance but no checkpoint was provided/loaded."
            )
        from classification.run_experiments import apply_exp3_guidance
        # U-Net is frozen here — no gradients needed through the mask.
        # (Grad-CAM, by contrast, MUST run with grad enabled: it calls
        # backward() internally. The caller's torch.no_grad() must NOT
        # cover cam.generate_heatmap.)
        with torch.no_grad():
            tensor = apply_exp3_guidance(
                tensor, unet_model, soft=(exp == "exp3_soft_masked")
            )
    return tensor


def load_classifier(exp: str, device) -> torch.nn.Module:
    from classification.classifier_model import BrainTumorClassifier

    ckpt = f"checkpoints/classification/best_efficientnet_{exp}.pth"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Classifier checkpoint missing: {ckpt}. "
            f"Train {exp} first — refusing to evaluate random weights."
        )
    model = BrainTumorClassifier(num_classes=4, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()
    logger.info(f"  loaded {ckpt}")
    return model


def load_unet(device):
    from segmentation.unet_model import UNet

    ckpt = "checkpoints/unet/best_unet_enhanced.pth"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"U-Net checkpoint missing: {ckpt}. Train the U-Net first "
            f"(python -m segmentation.train_unet)."
        )
    unet = UNet(n_channels=3, n_classes=1).to(device)
    unet.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    unet.eval()
    logger.info(f"  loaded {ckpt}")
    return unet


def main() -> None:
    ap = argparse.ArgumentParser(description="Grad-CAM localization evaluation driver.")
    ap.add_argument("--experiments", nargs="+", default=EXPERIMENTS, choices=EXPERIMENTS)
    ap.add_argument("--max_samples", type=int, default=200,
                    help="Cap on test samples per experiment (deterministic first-N).")
    ap.add_argument("--output", default="results/gradcam_localization_summary.json")
    args = ap.parse_args()

    from utils.device_config import get_system_execution_profile
    device = get_system_execution_profile()["device"]

    meta_path = "data/brisc/brisc_metadata.json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found. Run ingestion first: python -m data.dataset_ingestion"
        )
    with open(meta_path) as f:
        seg_pairs = json.load(f)["segmentation"]

    # Same deterministic split the U-Net itself is evaluated on.
    from data.dataset_preprocessor import split_dataset
    _, _, test_pairs = split_dataset(seg_pairs, random_state=42)
    test_pairs = test_pairs[: args.max_samples]
    logger.info(f"Held-out segmentation test samples: {len(test_pairs)}")

    from explainability.gradcam_generator import BrainTumorGradCAM
    from explainability.localization_eval import LocalizationEvaluator

    needs_unet = any(e.startswith("exp3") for e in args.experiments)
    unet_model = load_unet(device) if needs_unet else None

    evaluator = LocalizationEvaluator()
    for exp in args.experiments:
        logger.info(f"▶ Experiment: {exp}")
        model = load_classifier(exp, device)
        cam = BrainTumorGradCAM(model)
        done = 0
        # NOTE: no torch.no_grad() here — Grad-CAM calls backward() internally
        # and needs the autograd graph. (The frozen U-Net inside
        # build_experiment_input already guards itself with torch.no_grad().)
        for pair in test_pairs:
            tensor = build_experiment_input(exp, pair["image_path"], device, unet_model)
            heatmap = cam.generate_heatmap(tensor)  # (256, 256), predicted class
            gt_mask = _load_mask_256(pair["mask_path"])
            evaluator.evaluate_sample(exp, heatmap, gt_mask)
            done += 1
        cam.cleanup()
        del model
        logger.info(f"  ✓ {exp}: {done} heatmaps evaluated")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    summary = evaluator.summarize_and_save(args.output)
    for exp, thr in summary.items():
        t05 = thr.get("0.5", {})
        logger.info(
            f"  {exp}: IoU@0.5={t05.get('mean_iou', float('nan')):.4f} "
            f"Dice@0.5={t05.get('mean_dice', float('nan')):.4f} "
            f"(n={t05.get('n_samples', 0)})"
        )
    logger.info(f"✅ Localization summary → {args.output}")


if __name__ == "__main__":
    main()
