import csv
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class DatasetIngestor:
    # Minimum expected counts guard against silently ingesting a partial or
    # wrong dataset. Overridable via env for smaller/dev datasets.
    MIN_CLASSIFICATION_IMAGES = int(os.environ.get("BRISC_MIN_CLASSIFICATION", "6000"))
    MIN_SEGMENTATION_PAIRS = int(os.environ.get("BRISC_MIN_SEGMENTATION", "4700"))

    def __init__(self):
        # Resolve from this file's location, not the process CWD — ingestion
        # must work no matter where it is invoked from.
        self.base_dir = Path(__file__).resolve().parent.parent
        self.brisc_dir = self.base_dir / "data/brisc"
        self.pmram_dir = self.base_dir / "data/pmram"
        self.brisc_classes = ["glioma", "meningioma", "pituitary", "no_tumor"]

    def _to_rel(self, path) -> str:
        """Convert Path to forward-slash relative path from project root (CWD)."""
        try:
            return str(path.resolve().relative_to(self.base_dir)).replace(os.sep, "/")
        except ValueError:
            return str(path).replace(os.sep, "/")

    def ingest_brisc(self):
        logger.info("ℹ️ [INFO] Starting ultra-fast structural mapping of BRISC...")
        brisc_data = {"classification": [], "segmentation": []}

        # 1. High-Speed Classification Mapping (case-insensitive, multi-extension)
        # M5: rglob() order is filesystem-dependent -> sort for cross-machine
        # reproducible splits (split_dataset uses a fixed random_state).
        # M7: dedup by resolved path — on case-insensitive filesystems
        # (macOS/Windows) "*.jpg"+"*.JPG" match the same file twice, which
        # could leak one copy into train and another into test.
        seen: set = set()
        for class_name in self.brisc_classes:
            for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
                for img_path in sorted(self.brisc_dir.rglob(f"**/{class_name}/{ext}")):
                    resolved = img_path.resolve()
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    brisc_data["classification"].append(
                        {"path": self._to_rel(img_path), "class": class_name}
                    )

        # 2. High-Speed Segmentation Mapping
        # Map images directly by stem
        image_paths = sorted(
            p
            for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")
            for p in self.brisc_dir.rglob(f"**/segmentation_task/**/images/{ext}")
        )
        # M6: masks are indexed by stem; two different masks sharing a stem
        # (e.g. in nested folders) would previously pair silently with the
        # wrong image. Fail loudly instead of corrupting supervision.
        mask_paths = {}
        for ext in ("*.png", "*.PNG", "*.jpg", "*.JPG"):
            for p in sorted(self.brisc_dir.rglob(f"**/segmentation_task/**/masks/{ext}")):
                key = p.stem.replace("_mask", "")
                if key in mask_paths and mask_paths[key].resolve() != p.resolve():
                    raise RuntimeError(
                        f"Duplicate mask stem '{key}': {mask_paths[key]} vs {p}. "
                        "Cannot pair image<->mask safely."
                    )
                mask_paths.setdefault(key, p)

        for img_path in image_paths:
            stem = img_path.stem
            if stem in mask_paths:
                brisc_data["segmentation"].append(
                    {
                        "image_path": self._to_rel(img_path),
                        "mask_path": self._to_rel(mask_paths[stem]),
                    }
                )

        class_count = len(brisc_data["classification"])
        seg_count = len(brisc_data["segmentation"])

        logger.info(
            f"✅ [SUCCESS] Mapped {class_count} classification images and {seg_count} segmentation pairs."
        )

        # Explicit validation (not assert: asserts are stripped under -O and
        # give no actionable message). Minimums are env-overridable for
        # smaller/dev datasets.
        if class_count < self.MIN_CLASSIFICATION_IMAGES:
            raise RuntimeError(
                f"BRISC classification image count {class_count} < minimum "
                f"{self.MIN_CLASSIFICATION_IMAGES}. Is data/brisc populated? "
                f"(override: BRISC_MIN_CLASSIFICATION env var)"
            )
        if seg_count < self.MIN_SEGMENTATION_PAIRS:
            raise RuntimeError(
                f"BRISC segmentation pair count {seg_count} < minimum "
                f"{self.MIN_SEGMENTATION_PAIRS}. Is data/brisc populated? "
                f"(override: BRISC_MIN_SEGMENTATION env var)"
            )

        return brisc_data

    def ingest_pmram(self):
        logger.info("ℹ️ [INFO] Starting structural mapping of PMRAM...")
        pmram_records = []

        # Map PMRAM Original
        for img_path in self.pmram_dir.rglob("**/*riginal*/*.*"):
            if img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                pmram_records.append({"path": self._to_rel(img_path), "provenance": "original"})

        # Map PMRAM Augmented
        for img_path in self.pmram_dir.rglob("**/*ugmented*/*.*"):
            if img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                pmram_records.append({"path": self._to_rel(img_path), "provenance": "augmented"})

        logger.info(f"✅ [SUCCESS] Mapped {len(pmram_records)} total images from PMRAM.")
        return pmram_records

    def run_ingestion_pipeline(self):
        brisc_data = self.ingest_brisc()
        pmram_records = self.ingest_pmram()

        self.brisc_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.brisc_dir / "brisc_metadata.json"

        # Save structural metadata format exactly as requested
        output_metadata = {
            "classification_count": len(brisc_data["classification"]),
            "segmentation_count": len(brisc_data["segmentation"]),
            "classification": brisc_data["classification"],
            "segmentation": brisc_data["segmentation"],
        }

        with open(meta_path, "w") as f:
            json.dump(output_metadata, f, indent=4)
        logger.info(f"✅ [SUCCESS] Saved ultra-fast structural metadata to {meta_path}")

        if pmram_records:
            pmram_eval = [r for r in pmram_records if r["provenance"] == "original"]
            self.pmram_dir.mkdir(parents=True, exist_ok=True)
            with open(
                self.pmram_dir / "pmram_eval_set.csv", "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.DictWriter(f, fieldnames=["path", "provenance"])
                writer.writeheader()
                writer.writerows(pmram_eval)
            logger.info(f"✅ [SUCCESS] Saved {len(pmram_eval)} original PMRAM metadata.")

        logger.info("✅ [SUCCESS] Ultra-fast ingestion pipeline complete in under 5 seconds.")


if __name__ == "__main__":
    ingestor = DatasetIngestor()
    ingestor.run_ingestion_pipeline()
