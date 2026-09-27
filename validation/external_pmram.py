import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import tqdm

from classification.classifier_model import BrainTumorClassifier
from classification.run_experiments import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ImageNet normalization constants — must match training preprocessing
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Folder-name → class-index mapping (case-insensitive substring match) ──────
# Each tuple: (substring_to_match_lowercase, class_index)
# Checked in order; first match wins.
_FOLDER_CLASS_RULES: list[tuple[str, int]] = [
    ("glioma", 0),  # covers glioma, 512Glioma, Glioma, glioma_tumor …
    ("meningioma", 1),  # covers meningioma, 512Meningioma …
    ("pituitary", 2),  # covers pituitary, 512Pituitary …
    ("no_tumor", 3),  # covers no_tumor, 512No_Tumor …
    ("notumor", 3),  # covers notumor (no separator variant)
    ("normal", 3),  # covers normal / healthy folders
]

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _sha256_of_file(path: str, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file (checkpoint provenance)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _folder_to_class(folder_name: str) -> int | None:
    """
    Map a PMRAM folder name to a class index using case-insensitive substring
    matching. Returns None if no rule matches (folder should be skipped).
    """
    lower = folder_name.lower()
    # Guard: "abnormal" contains the substring "normal" but means the OPPOSITE
    # (a tumor class). Without this, an "abnormal/..." folder would be
    # silently labeled no_tumor (class 3). Skip such ambiguous folders unless
    # a specific tumor class also matches.
    if "abnormal" in lower and not any(s in lower for s, _ in _FOLDER_CLASS_RULES[:3]):
        return None
    for substring, class_idx in _FOLDER_CLASS_RULES:
        if substring in lower:
            return class_idx
    return None


def _is_augmented(path: str) -> bool:
    """Return True if the path string indicates an augmented image."""
    lower = path.lower()
    return "augmented" in lower


def _preprocess_image(img_bgr: np.ndarray, enhanced: bool = True, enhancer=None) -> torch.Tensor:
    """
    BGR uint8 → (1, 3, 256, 256) float32 tensor with ImageNet normalisation.

    CRITICAL (C1): the preprocessing MUST match the distribution the evaluated
    model was trained on. The default model (best_efficientnet_exp2_enhanced)
    was trained on WPT→LMMSE→CLAHE enhanced images, so enhancement is applied
    by default (enhanced=True). Evaluating it on raw images (the old behavior)
    measures the wrong distribution and invalidates the reported accuracy.
    Pass enhanced=False only for models trained on raw images (e.g. exp1).
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rs = cv2.resize(img_rgb, (256, 256)).astype(np.float32) / 255.0
    if enhanced:
        if enhancer is None:
            raise RuntimeError(
                "enhanced=True but no enhancement pipeline supplied. "
                "This is a validator bug, not a data problem."
            )
        img_rs = enhancer.process(img_rs)  # float32 [0, 1], same as training cache
        img_rs = np.clip(img_rs, 0.0, 1.0)
    img_norm = (img_rs - _MEAN) / _STD  # (256, 256, 3)
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,256,256)
    return tensor


class PMRAMValidator:
    """
    FR-039 to FR-042: Evaluates the final model on the PMRAM dataset with ZERO retraining.

    Supports two data-source modes:
      1. Folder-walk mode  (pmram_root is supplied or auto-detected):
         Walks the directory tree, maps sub-folder names to class indices,
         filters augmented paths, and loads real JPEG/PNG images.
      2. Metadata-CSV mode (legacy / unit-test compatible):
         Falls back to reading pmram_meta_path CSV with a 'provenance' column.
    """

    def __init__(
        self,
        model_path: str,
        pmram_meta_path: str,
        brisc_metrics_path: str,
        pmram_root: str | None = None,
        preprocessing: str = "enhanced",
    ):
        if preprocessing not in ("enhanced", "raw"):
            raise ValueError(f"preprocessing must be 'enhanced' or 'raw', got {preprocessing!r}")
        self.model_path = model_path
        self.pmram_meta_path = pmram_meta_path
        self.brisc_metrics_path = brisc_metrics_path
        self.preprocessing = preprocessing

        # Sanity: the preprocessing must match the model's training
        # distribution. exp1_baseline trained on raw images; exp2_enhanced and
        # exp3_* trained on WPT→LMMSE→CLAHE enhanced images.
        _name = os.path.basename(model_path).lower()
        if "exp1" in _name and preprocessing == "enhanced":
            logger.warning(
                "⚠️ Model filename suggests exp1 (trained on RAW images) but "
                "preprocessing='enhanced'. Pass --preprocessing raw for exp1."
            )
        if ("exp2" in _name or "exp3" in _name) and preprocessing == "raw":
            logger.warning(
                "⚠️ Model filename suggests exp2/exp3 (trained on ENHANCED images) "
                "but preprocessing='raw'. This measures the wrong distribution."
            )

        self.enhancer = None
        if preprocessing == "enhanced":
            from enhancement.pipeline import EnhancementAblationManager

            self.enhancer = EnhancementAblationManager()  # WPT→LMMSE→CLAHE, same as training

        from utils.device_config import get_system_execution_profile

        profile = get_system_execution_profile()
        self.device = profile["device"]

        # Auto-detect pmram_root from pmram_meta_path if not supplied
        if pmram_root is not None:
            self.pmram_root = pmram_root
        else:
            # Heuristic: parent of the CSV is the PMRAM root
            self.pmram_root = str(Path(pmram_meta_path).parent)

        if os.path.exists(os.path.join(self.pmram_root, "original")):
            self.pmram_root = os.path.join(self.pmram_root, "original")

    # ------------------------------------------------------------------
    def load_model(self) -> BrainTumorClassifier:
        # CRITICAL (C3, fail-hard): a missing checkpoint used to only log a
        # warning and evaluate RANDOMLY INITIALIZED weights while still
        # writing results/pmram_external_validation.json — i.e. publishing
        # fake external-validation numbers. Refuse instead.
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"PMRAM validation: classifier checkpoint not found at "
                f"{self.model_path}. Refusing to evaluate uninitialized weights."
            )

        model = BrainTumorClassifier(num_classes=4, pretrained=False)
        model.load_state_dict(
            torch.load(self.model_path, map_location=self.device, weights_only=True)
        )
        logger.info(f"Loaded classifier from {self.model_path}")
        model.to(self.device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    def load_pmram_metadata(self) -> pd.DataFrame:
        """FR-042 & FR-005: Restrict to original PMRAM images only (CSV mode)."""
        if not os.path.exists(self.pmram_meta_path):
            raise FileNotFoundError(f"Metadata not found: {self.pmram_meta_path}")

        df = pd.read_csv(self.pmram_meta_path)

        if "provenance" not in df.columns:
            raise ValueError(
                f"PMRAM metadata {self.pmram_meta_path} has no 'provenance' "
                f"column (found: {list(df.columns)}). Cannot separate original "
                "from augmented images."
            )

        original_df = df[df["provenance"] == "original"].copy()
        augmented_count = len(df) - len(original_df)

        if augmented_count > 0:
            logger.warning(
                f"Found {augmented_count} augmented images in PMRAM set. "
                "Excluding them from external validation."
            )

        if len(original_df) != 1600:
            logger.warning(f"Expected 1,600 original images, found {len(original_df)}.")

        return original_df

    # ------------------------------------------------------------------
    def load_pmram_from_folder(self) -> list[tuple[str, int]]:
        """
        Walk self.pmram_root, map sub-folder names to class indices, and
        return a list of (image_path, class_index) for non-augmented images.

        Supports all PMRAM folder naming variants:
          - Glioma      / 512Glioma      → class 0
          - Meningioma  / 512Meningioma  → class 1
          - Pituitary   / 512Pituitary   → class 2
          - No_Tumor / notumor / normal / 512No_Tumor → class 3

        Strictly rejects paths containing 'augmented' (case-insensitive).
        """
        samples: list[tuple[str, int]] = []
        class_counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

        if not os.path.isdir(self.pmram_root):
            logger.warning(f"PMRAM root not found: {self.pmram_root}. Falling back to CSV mode.")
            return []

        for dirpath, _dirnames, filenames in os.walk(self.pmram_root):
            folder_name = os.path.basename(dirpath)
            class_idx = _folder_to_class(folder_name)
            if class_idx is None:
                continue  # skip non-class directories (root, intermediate dirs)

            for fname in filenames:
                ext = Path(fname).suffix.lower()
                if ext not in _IMAGE_EXTS:
                    continue

                full_path = os.path.join(dirpath, fname)

                # Strictly exclude augmented images
                if _is_augmented(full_path):
                    continue

                samples.append((full_path, class_idx))
                class_counts[class_idx] += 1

        logger.info(
            f"PMRAM folder-walk complete | "
            f"glioma={class_counts[0]}, meningioma={class_counts[1]}, "
            f"pituitary={class_counts[2]}, no_tumor={class_counts[3]} | "
            f"total={len(samples)}"
        )

        if len(samples) == 0:
            logger.warning("No images found via folder-walk. Falling back to CSV mode.")

        return samples

    # ------------------------------------------------------------------
    def load_brisc_metrics(self) -> dict[str, float]:
        """
        Load BRISC validation metrics for generalization-gap calculation.

        Priority: the EXPLICIT brisc_metrics_path argument first, then
        known fallbacks. Within a history file, the BEST epoch (by macro F1,
        then accuracy) is used — not the last epoch (previously the gap was
        computed against whatever the final epoch happened to be).
        """
        candidates = [
            self.brisc_metrics_path,
            "results/metrics_exp3_seg_guided.json",
            "results/metrics_exp1_baseline.json",
        ]

        for path in candidates:
            if not path or not os.path.exists(path):
                continue
            try:
                with open(path) as f:
                    history = json.load(f)
                if not history:
                    continue
                # History may be a list of per-epoch dicts (pick best) or a
                # single final-metrics dict (e.g. *_test.json).
                if isinstance(history, list):

                    def _score(h: dict) -> tuple:
                        m = h.get("val_metrics", {}) or {}
                        return (m.get("macro_f1", 0.0), m.get("accuracy", 0.0))

                    best = max(history, key=_score)
                    metrics = best.get("val_metrics", {})
                elif isinstance(history, dict):
                    metrics = {k: v for k, v in history.items() if isinstance(v, (int, float))}
                else:
                    continue
                if metrics:
                    logger.info(f"Loaded BRISC metrics from {path}")
                    return metrics
            except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                logger.warning(f"Could not parse {path}: {e}")
                continue

        logger.warning("No valid BRISC metrics file found. Gap cannot be calculated.")
        return {}

    # ------------------------------------------------------------------
    def validate(self) -> None:
        logger.info("Starting PMRAM External Validation...")

        model = self.load_model()
        brisc_metrics = self.load_brisc_metrics()

        # ── Choose data source ─────────────────────────────────────────
        folder_samples = self.load_pmram_from_folder()
        use_folder_mode = len(folder_samples) > 0

        if use_folder_mode:
            logger.info(f"Using folder-walk mode: {len(folder_samples)} images found.")
            all_preds: list[int] = []
            all_labels: list[int] = []
            errors = 0

            with torch.no_grad():
                for img_path, class_idx in tqdm.tqdm(folder_samples, desc="PMRAM Validation"):
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is None:
                        logger.warning(f"Cannot read image: {img_path}")
                        errors += 1
                        continue

                    tensor = _preprocess_image(
                        img_bgr,
                        enhanced=(self.preprocessing == "enhanced"),
                        enhancer=self.enhancer,
                    ).to(self.device)

                    with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                        logits = model(tensor)

                    pred = torch.argmax(logits, dim=1).item()
                    all_preds.append(pred)
                    all_labels.append(class_idx)

            if errors > 0:
                logger.warning(f"{errors} images could not be read and were skipped.")

        else:
            # ── Legacy CSV / unit-test path ────────────────────────────
            # There are no real images behind a bare CSV, so there is NOTHING
            # honest to evaluate. Previously this branch fabricated
            # all-zero dummy predictions and wrote them to
            # results/pmram_external_validation.json as if they were real
            # external-validation metrics — a research-integrity violation.
            # Fail loudly instead of publishing fake numbers.
            raise RuntimeError(
                "PMRAM folder-walk found no usable images and CSV mode has no "
                "image data to score. Refusing to write fabricated metrics to "
                "results/pmram_external_validation.json. Provide --pmram_root "
                "pointing at the real PMRAM image tree."
            )

        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))

        # ── Generalization gap (Delta = BRISC − PMRAM) ─────────────────
        gap: dict[str, float] = {}
        for k, v in brisc_metrics.items():
            if k in metrics and isinstance(v, (int, float)):
                gap[k] = float(v - metrics[k])

        results = {
            "pmram_metrics": metrics,
            "brisc_metrics": brisc_metrics,
            "generalization_gap": gap,
            "num_samples_evaluated": len(all_labels),
            "data_source": "folder_walk",
            # Provenance (C3): exactly which artifact produced these numbers.
            "checkpoint_path": os.path.abspath(self.model_path),
            "checkpoint_sha256": _sha256_of_file(self.model_path),
            "preprocessing": self.preprocessing,  # 'enhanced' = WPT→LMMSE→CLAHE, must match training
        }

        os.makedirs("results", exist_ok=True)
        out_path = "results/pmram_external_validation.json"
        # Atomic write: never leave a half-written JSON behind on crash.
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(results, f, indent=4)
        os.replace(tmp_path, out_path)

        logger.info(f"Validation completed. Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Results saved to {out_path}")

        if self.device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PMRAM External Validation")
    parser.add_argument(
        "--model_path", default="checkpoints/classification/best_efficientnet_exp2_enhanced.pth"
    )
    parser.add_argument(
        "--pmram_root",
        default="data/pmram",
        help="Root directory of the PMRAM dataset (folder-walk mode)",
    )
    parser.add_argument(
        "--pmram_meta",
        default="data/pmram/pmram_metadata.csv",
        help="CSV metadata fallback path (legacy; folder-walk is authoritative)",
    )
    parser.add_argument("--brisc_metrics", default="results/metrics_exp2_enhanced_test.json")
    parser.add_argument(
        "--preprocessing",
        default="enhanced",
        choices=["enhanced", "raw"],
        help="Input distribution the model was TRAINED on: 'enhanced' "
        "applies WPT→LMMSE→CLAHE (exp2/exp3), 'raw' skips it (exp1).",
    )
    args = parser.parse_args()

    validator = PMRAMValidator(
        model_path=args.model_path,
        pmram_meta_path=args.pmram_meta,
        brisc_metrics_path=args.brisc_metrics,
        pmram_root=args.pmram_root,
        preprocessing=args.preprocessing,
    )
    validator.validate()
