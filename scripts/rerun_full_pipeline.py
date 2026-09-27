#!/usr/bin/env python3
"""One-command full experiment re-run (post-fix pipeline).

Reproduces every number the paper/docs claim, in dependency order:

    1. cache    rebuild WPT->LMMSE->CLAHE enhancement cache (purges stale cache)
    2. unet     train U-Net on ENHANCED images
    3. exp1     EfficientNetB2 baseline (raw images)
    4. exp2     EfficientNetB2 enhanced (cached)
    5. exp3     seg-guided + soft-masked (needs the U-Net checkpoint)
    6. xai      Grad-CAM localization vs GT masks (needs trained classifiers)
    7. pmram    external PMRAM validation with checkpoint provenance
    8. reports  regenerate consolidated evidence-honest reports

Full mode (needs a GPU + the real datasets under data/):

    python -m scripts.rerun_full_pipeline

Resume from a stage (e.g. after a crash):

    python -m scripts.rerun_full_pipeline --from-stage unet

Smoke mode (CPU, synthetic mini-dataset, ~2-5 min): validates that every
stage is wired correctly end-to-end before spending GPU hours:

    python -m scripts.rerun_full_pipeline --smoke

Outputs land in the standard locations: checkpoints/, results/,
reports/. In smoke mode everything is redirected into a temp dir so the
real repo is never touched.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("rerun")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STAGES = ["cache", "unet", "exp1", "exp2", "exp3", "xai", "pmram", "reports"]

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def run(cmd, cwd, env, stage_name):
    """Run a stage subprocess; raise on failure."""
    logger.info(f"\n{'=' * 70}\n▶ STAGE: {stage_name}\n  $ {' '.join(cmd)}\n{'=' * 70}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Stage '{stage_name}' FAILED after {dt:.0f}s (exit {proc.returncode})")
    logger.info(f"✓ Stage '{stage_name}' done in {dt:.0f}s")


def base_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def preflight(full_mode):
    import torch  # noqa: F401  (validates the dep is installed)

    if full_mode:
        meta = PROJECT_ROOT / "data" / "brisc" / "brisc_metadata.json"
        if not meta.exists():
            raise FileNotFoundError(
                "data/brisc/brisc_metadata.json not found.\n"
                "Download + ingest the datasets first (no login needed):\n"
                "  python -m data.download_datasets\n"
                "  python -m data.dataset_ingestion"
            )
        if not torch.cuda.is_available():
            logger.warning("⚠ No CUDA GPU detected — full training on CPU will be extremely slow.")
        pmram_root = PROJECT_ROOT / "data" / "pmram"
        if not pmram_root.exists() or not any(pmram_root.rglob("*.jpg")):
            logger.warning(
                "⚠ No PMRAM images under data/pmram — the 'pmram' stage will fail. "
                "Download it or pass --skip-pmram."
            )
    logger.info("✓ Preflight OK")


def require_unet_checkpoint():
    """C2: exp3 must never train on unmasked images while labeled 'seg-guided'."""
    ckpt = PROJECT_ROOT / "checkpoints" / "unet" / "best_unet_enhanced.pth"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Exp3 stage needs the trained U-Net at {ckpt}, which is missing. "
            "Run the 'unet' stage first (or --from-stage unet). Refusing to "
            "produce a mislabeled 'seg-guided' experiment."
        )


# --------------------------------------------------------------------------- #
# stage 1: enhancement cache rebuild (ported from notebook 01, cell 6)
# --------------------------------------------------------------------------- #


def build_cache(cache_dir="data/cached_enhanced", limit=None):
    import glob

    import cv2
    import numpy as np
    from tqdm.auto import tqdm

    from data.dataset_preprocessor import cached_enhanced_target
    from enhancement.pipeline import EnhancementAblationManager

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logger.info(f"🧹 Purged stale cache: {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)

    manager = EnhancementAblationManager()  # wpt_lmmse_clahe
    all_paths = (
        glob.glob("data/brisc/**/*.jpg", recursive=True)
        + glob.glob("data/brisc/**/*.jpeg", recursive=True)
        + glob.glob("data/brisc/**/*.png", recursive=True)
    )
    image_paths = [
        p
        for p in all_paths
        if "mask" not in os.path.basename(p).lower()
        and os.path.basename(os.path.dirname(p)).lower() != "masks"
        and cache_dir not in p.replace(os.sep, "/")
    ]
    if limit:
        image_paths = image_paths[:limit]
    logger.info(f"Enhancing {len(image_paths):,} images → {cache_dir}")

    errors = 0
    for path in tqdm(image_paths, desc="WPT→LMMSE→CLAHE", unit="img", dynamic_ncols=True):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            errors += 1
            continue
        enhanced = manager.process(img.astype("float32") / 255.0)
        target = cached_enhanced_target(path, cache_dir)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        cv2.imwrite(target, np.clip(enhanced * 255.0, 0, 255).astype("uint8"))
    logger.info(f"✓ Cache complete: {len(image_paths) - errors:,} images, {errors} unreadable")


# --------------------------------------------------------------------------- #
# smoke mode: synthetic mini dataset
# --------------------------------------------------------------------------- #


def build_smoke_dataset(smoke_root: Path, per_class: int = 8):
    """Create a tiny BRISC-shaped dataset of synthetic MRI slices + masks."""
    import cv2
    import numpy as np

    rng = np.random.default_rng(7)
    classes = ["glioma", "meningioma", "pituitary", "no_tumor"]
    brisc = smoke_root / "data" / "brisc"
    classification, segmentation = [], []

    for ci, cls in enumerate(classes):
        cls_dir = brisc / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            img = np.zeros((256, 256), dtype="uint8")
            cx, cy = 128 + int(rng.integers(-20, 20)), 128 + int(rng.integers(-20, 20))
            r = int(rng.integers(25, 55))
            base = 90 + ci * 25
            cv2.circle(img, (cx, cy), r, base, -1)
            cv2.circle(img, (cx, cy), r // 2, min(255, base + 60), -1)
            noise = rng.normal(0, 18, img.shape)
            img = np.clip(img + noise, 0, 255).astype("uint8")
            p = cls_dir / f"img_{i:03d}.png"
            cv2.imwrite(str(p), img)

            mask = np.zeros((256, 256), dtype="uint8")
            if cls != "no_tumor":
                cv2.circle(mask, (cx, cy), r, 255, -1)
            mp = cls_dir / f"img_{i:03d}_mask.png"
            cv2.imwrite(str(mp), mask)

            rel_img = os.path.join("data", "brisc", cls, p.name)
            classification.append({"path": rel_img, "class": cls})
            segmentation.append(
                {
                    "image_path": rel_img,
                    "mask_path": os.path.join("data", "brisc", cls, mp.name),
                }
            )

    meta = {
        "classification_count": len(classification),
        "segmentation_count": len(segmentation),
        "classification": classification,
        "segmentation": segmentation,
    }
    with open(brisc / "brisc_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"✓ Smoke dataset: {len(classification)} cls images, {len(segmentation)} seg pairs")
    return smoke_root


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description="Full post-fix experiment re-run.")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="CPU smoke test on synthetic data; touches nothing in the repo.",
    )
    ap.add_argument(
        "--from-stage",
        choices=STAGES,
        default="cache",
        help="Resume from this stage (default: cache).",
    )
    ap.add_argument(
        "--skip-cache",
        action="store_true",
        help="Reuse existing data/cached_enhanced instead of rebuilding.",
    )
    ap.add_argument("--epochs-unet", type=int, default=25)
    ap.add_argument("--epochs-cls", type=int, default=25)
    ap.add_argument("--skip-pmram", action="store_true", help="Skip external PMRAM validation.")
    args = ap.parse_args()

    py = sys.executable
    env = base_env()

    if args.smoke:
        workdir = Path(tempfile.mkdtemp(prefix="bt_rerun_smoke_"))
        build_smoke_dataset(workdir)
        logger.info(f"Smoke workdir: {workdir}")
        start = 0
        epochs_unet, epochs_cls = 1, 1
        extra = ["--batch_size", "2", "--num_workers", "0"]
        do_pmram = False
    else:
        workdir = PROJECT_ROOT
        preflight(full_mode=True)
        start = STAGES.index(args.from_stage)
        epochs_unet, epochs_cls = args.epochs_unet, args.epochs_cls
        extra = []
        do_pmram = not args.skip_pmram
        if args.skip_cache:
            start = max(start, 1)

    os.chdir(workdir)
    t_all = time.time()

    def stage_enabled(name):
        return STAGES.index(name) >= start

    try:
        # 1. cache
        if stage_enabled("cache"):
            logger.info("\n▶ STAGE: cache — rebuilding enhancement cache")
            build_cache(limit=32 if args.smoke else None)

        # 2. unet
        if stage_enabled("unet"):
            run(
                [
                    py,
                    "-m",
                    "segmentation.train_unet",
                    "--input_type",
                    "enhanced",
                    "--epochs",
                    str(epochs_unet),
                ]
                + extra,
                workdir,
                env,
                "unet",
            )

        unet_ckpt = "checkpoints/unet/best_unet_enhanced.pth"

        # 3-4. exp1 / exp2
        for exp in ["exp1_baseline", "exp2_enhanced"]:
            if stage_enabled(exp[:4]):
                run(
                    [
                        py,
                        "-m",
                        "classification.run_experiments",
                        "--experiment",
                        exp,
                        "--epochs",
                        str(epochs_cls),
                    ]
                    + extra,
                    workdir,
                    env,
                    exp,
                )

        # 5. exp3 (both variants, needs the trained U-Net)
        if stage_enabled("exp3"):
            require_unet_checkpoint()
            for exp in ["exp3_seg_guided", "exp3_soft_masked"]:
                run(
                    [
                        py,
                        "-m",
                        "classification.run_experiments",
                        "--experiment",
                        exp,
                        "--epochs",
                        str(epochs_cls),
                        "--unet_checkpoint",
                        unet_ckpt,
                    ]
                    + extra,
                    workdir,
                    env,
                    exp,
                )

        # 6. xai — quantitative Grad-CAM localization on the held-out
        #    segmentation test split (fail-hard on missing checkpoints)
        if stage_enabled("xai"):
            run(
                [
                    py,
                    "-m",
                    "explainability.run_localization_eval",
                    "--max_samples",
                    "200" if not args.smoke else "8",
                ],
                workdir,
                env,
                "xai",
            )

        # 7. pmram
        if stage_enabled("pmram") and do_pmram:
            run(
                [
                    py,
                    "-m",
                    "validation.external_pmram",
                    "--model_path",
                    "checkpoints/classification/best_efficientnet_exp2_enhanced.pth",
                    "--brisc_metrics",
                    "results/metrics_exp2_enhanced_test.json",
                ],
                workdir,
                env,
                "pmram",
            )
        elif stage_enabled("pmram"):
            logger.info("⊘ Stage 'pmram' skipped (--skip-pmram / smoke mode)")

        # 8. reports — ReportConsolidator defaults to the *source file's*
        # location, so in smoke mode we must inject the isolated workdir
        # explicitly, otherwise it overwrites the real repo's reports/.
        if stage_enabled("reports"):
            run(
                [
                    py,
                    "-c",
                    "from evaluation.consolidate_reports import ReportConsolidator; "
                    "ReportConsolidator(project_root='.').run()",
                ],
                workdir,
                env,
                "reports",
            )

    except Exception as e:
        logger.error(f"\n✗ PIPELINE STOPPED: {e}")
        sys.exit(1)

    # ---- verification ------------------------------------------------------
    expected = [
        "checkpoints/unet/best_unet_enhanced.pth",
        "results/metrics_exp1_baseline_test.json",
        "results/metrics_exp2_enhanced_test.json",
    ]
    if not args.smoke:
        expected += [
            "results/metrics_exp3_seg_guided_test.json",
            "results/metrics_exp3_soft_masked_test.json",
            "results/gradcam_localization_summary.json",
            "results/pmram_external_validation.json",
            "reports/PHASE_II_FINAL_PROJECT_REPORT.md",
        ]
    missing = [f for f in expected if not (workdir / f).exists()]
    dt = time.time() - t_all
    if missing:
        logger.warning(f"⚠ Finished in {dt:.0f}s but missing outputs: {missing}")
    else:
        logger.info(
            f"\n✅ FULL RE-RUN COMPLETE in {dt / 3600:.1f}h — all expected outputs present."
        )
        for f in expected:
            logger.info(f"   • {f}")


if __name__ == "__main__":
    main()
