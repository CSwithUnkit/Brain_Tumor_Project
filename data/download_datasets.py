"""No-login dataset downloader for the Brain Tumor Project.

BRISC comes from Zenodo, PMRAM (original/raw images only) from Mendeley Data.
Both sources allow anonymous download — no Kaggle account, no API tokens.

Design notes (learned from real failures on flaky networks / Windows):
- Pure standard library (urllib + zipfile): works on Windows, macOS, Linux,
  Colab without curl/unzip binaries.
- Resumable downloads via HTTP Range, with retry loop: the Mendeley and
  Zenodo endpoints have both dropped connections mid-transfer (curl exit 18).
- NEVER prints a fake success: every step verifies on disk (archive size,
  then extracted content counts). Anything unverifiable raises loudly.
"""

from __future__ import annotations

import logging
import os
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Canonical sources (verified 2026-09-27)
# --------------------------------------------------------------------------- #
BRISC_URL = "https://zenodo.org/records/17524350/files/brisc2025.zip?download=1"
BRISC_ARCHIVE = "brisc2025.zip"
BRISC_EXPECTED_SIZE = 260175408  # bytes, as published on Zenodo
BRISC_MIN_CLASSIFICATION = 6000
BRISC_MIN_SEGMENTATION = 4700

PMRAM_URL = (
    "https://data.mendeley.com/public-files/datasets/m7w55sw88b"
    "/files/92239c50-c554-48a5-8e6e-7d01a22ede01/file_downloaded"
)
PMRAM_ARCHIVE = "raw_data.zip"
PMRAM_EXPECTED_SIZE = 37728150  # bytes, as published on Mendeley
# Only the 1,600 original images are used for external validation; the
# augmented derivatives must never leak into validation.
PMRAM_MIN_ORIGINAL = 1500

BRISC_CLASSES = ("glioma", "meningioma", "pituitary", "no_tumor")
_CHUNK = 1024 * 1024  # 1 MiB
_MAX_RETRIES = 6


# --------------------------------------------------------------------------- #
# Resumable download
# --------------------------------------------------------------------------- #
def download_with_resume(
    url: str,
    dest: Path,
    expected_size: int | None = None,
    max_retries: int = _MAX_RETRIES,
) -> Path:
    """Download *url* to *dest*, resuming partial files via HTTP Range.

    Retries transient failures. Raises RuntimeError if the transfer cannot
    be completed. A size mismatch against *expected_size* is a loud warning,
    not a silent pass — the caller must still verify extracted content.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        existing = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=60) as resp:
                status = resp.status
                if status == 206 and existing:
                    mode, offset = "ab", existing
                    total = existing + int(resp.headers.get("Content-Length", 0))
                elif status == 200:
                    if existing:
                        logger.warning(
                            "Server ignored Range request — restarting %s from scratch.",
                            dest.name,
                        )
                    mode, offset, total = "wb", 0, int(resp.headers.get("Content-Length", 0) or 0)
                else:
                    raise RuntimeError(f"Unexpected HTTP status {status} for {url}")

                downloaded = offset
                last_log = 0.0
                with open(dest, mode) as f:
                    while True:
                        chunk = resp.read(_CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if total and now - last_log > 10:
                            logger.info(
                                "%s: %.1f%% (%d/%d MiB)",
                                dest.name,
                                100.0 * downloaded / total,
                                downloaded // _CHUNK,
                                total // _CHUNK,
                            )
                            last_log = now

            final = dest.stat().st_size
            if expected_size and final != expected_size:
                logger.warning(
                    "Size mismatch for %s: got %d bytes, expected %d. "
                    "Continuing — content verification is the final gate.",
                    dest.name,
                    final,
                    expected_size,
                )
            else:
                logger.info("Downloaded %s (%d bytes).", dest.name, final)
            return dest

        except HTTPError as e:
            # 416 = Range unsatisfiable: our partial file is already complete.
            if e.code == 416 and expected_size and dest.exists():
                if dest.stat().st_size >= expected_size:
                    logger.info("%s already fully downloaded.", dest.name)
                    return dest
            logger.warning("Attempt %d/%d failed (HTTP %s): %s", attempt, max_retries, e.code, e)
        except (URLError, TimeoutError, ConnectionError, OSError) as e:
            logger.warning("Attempt %d/%d failed: %s", attempt, max_retries, e)

        if attempt < max_retries:
            time.sleep(min(2**attempt, 30))

    raise RuntimeError(f"Could not download {url} after {max_retries} attempts.")


def extract_zip(archive: Path, dest_dir: Path) -> None:
    """Extract *archive* into *dest_dir* (stdlib zipfile — Windows-safe)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s ...", archive.name)
    try:
        with zipfile.ZipFile(archive) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(f"Corrupt entry in {archive.name}: {bad}")
            zf.extractall(dest_dir)
    except (zipfile.BadZipFile, FileNotFoundError, OSError) as e:
        raise RuntimeError(f"Cannot extract {archive}: {e}") from e
    logger.info("Extracted %s.", archive.name)


# --------------------------------------------------------------------------- #
# Content verification — the final gate. Counts must match the ingestion
# minimums, otherwise the download is treated as failed.
# --------------------------------------------------------------------------- #
def verify_brisc(
    root: Path,
    min_classification: int = BRISC_MIN_CLASSIFICATION,
    min_segmentation: int = BRISC_MIN_SEGMENTATION,
) -> dict:
    root = Path(root)
    base = root / "brisc2025"
    if not base.is_dir():
        raise RuntimeError(f"BRISC content missing: {base} not found after extraction.")

    cls_count = 0
    for split in ("train", "test"):
        for cls in BRISC_CLASSES:
            d = base / "classification_task" / split / cls
            if d.is_dir():
                cls_count += sum(
                    1 for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                )

    seg_pairs = 0
    for split in ("train", "test"):
        img_d = base / "segmentation_task" / split / "images"
        msk_d = base / "segmentation_task" / split / "masks"
        if img_d.is_dir() and msk_d.is_dir():
            imgs = {
                p.stem for p in img_d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            }
            msks = {
                p.stem for p in msk_d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            }
            seg_pairs += len(imgs & msks)

    if cls_count < min_classification:
        raise RuntimeError(
            f"BRISC verification failed: {cls_count} classification images "
            f"(need >= {min_classification})."
        )
    if seg_pairs < min_segmentation:
        raise RuntimeError(
            f"BRISC verification failed: {seg_pairs} segmentation pairs "
            f"(need >= {min_segmentation})."
        )
    logger.info(
        "BRISC verified: %d classification images, %d segmentation pairs.", cls_count, seg_pairs
    )
    return {"classification": cls_count, "segmentation_pairs": seg_pairs}


def verify_pmram(root: Path, min_original: int = PMRAM_MIN_ORIGINAL) -> dict:
    root = Path(root)
    raw = root / "Raw"
    if not raw.is_dir():
        # Fall back to a flat class-dir layout (some mirrors differ).
        candidates = [root]
    else:
        candidates = [raw]

    counts: dict[str, int] = {}
    for base in candidates:
        for d in base.iterdir():
            if not d.is_dir() or "augment" in d.name.lower():
                continue
            n = sum(1 for p in d.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            if n:
                counts[d.name] = n

    total = sum(counts.values())
    if total < min_original:
        raise RuntimeError(
            f"PMRAM verification failed: {total} original images (need >= {min_original}). "
            f"Seen folders: {counts}"
        )
    logger.info("PMRAM verified: %d original images %s.", total, counts)
    return {"original": total, "folders": counts}


# --------------------------------------------------------------------------- #
# Idempotent ensure_* entry points
# --------------------------------------------------------------------------- #
def _resolve(root: Path | None, *parts: str) -> Path:
    base = Path(root) if root else Path.cwd()
    return base.joinpath(*parts)


def ensure_brisc(root: Path | None = None) -> dict:
    """Download + verify BRISC unless the verified content is already present."""
    target = _resolve(root, "data", "brisc")
    try:
        return verify_brisc(target)
    except RuntimeError:
        logger.info("BRISC not present/complete — downloading from Zenodo (no login needed).")

    archive = target / BRISC_ARCHIVE
    download_with_resume(BRISC_URL, archive, expected_size=BRISC_EXPECTED_SIZE)
    extract_zip(archive, target)
    return verify_brisc(target)


def ensure_pmram(root: Path | None = None) -> dict:
    """Download + verify PMRAM originals unless already present."""
    target = _resolve(root, "data", "pmram")
    try:
        return verify_pmram(target)
    except RuntimeError:
        logger.info("PMRAM not present/complete — downloading from Mendeley (no login needed).")

    archive = target / PMRAM_ARCHIVE
    download_with_resume(PMRAM_URL, archive, expected_size=PMRAM_EXPECTED_SIZE)
    extract_zip(archive, target)
    return verify_pmram(target)


# --------------------------------------------------------------------------- #
# Legacy / Colab-compatible surface
# --------------------------------------------------------------------------- #
def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401 — presence probe

        return True
    except ImportError:
        return False


def setup_environment() -> Path:
    base_dir = Path(os.getcwd())
    if is_colab():
        logger.info("Detected Google Colab environment.")
        try:
            from google.colab import drive

            drive_path = "/content/drive"
            if not os.path.exists(drive_path):
                drive.mount(drive_path)
            project_dir = Path("/content/drive/MyDrive/Brain_Tumor_Project")
            project_dir.mkdir(parents=True, exist_ok=True)
            base_dir = project_dir
            os.chdir(base_dir)
            logger.info("Mounted Drive and set base directory to %s", base_dir)
        except Exception as e:  # noqa: BLE001 — Drive mount is best-effort
            logger.error("Failed to mount Google Drive: %s", e)
            logger.info("Falling back to local storage in Colab.")
    else:
        logger.info("Detected local environment. Base directory: %s", base_dir)
    return base_dir


def create_directory_structure(base_dir: Path) -> None:
    for d in [
        base_dir / "data" / "brisc",
        base_dir / "data" / "pmram",
        base_dir / "data" / "cached_enhanced",
        base_dir / "checkpoints" / "unet",
        base_dir / "checkpoints" / "classification",
        base_dir / "logs",
        base_dir / "results",
    ]:
        d.mkdir(parents=True, exist_ok=True)
        logger.info("Created directory: %s", d)


def download_datasets(base_dir: Path) -> None:
    """Download + verify both datasets (no login required)."""
    base_dir = Path(base_dir)
    create_directory_structure(base_dir)
    ensure_brisc(base_dir)
    ensure_pmram(base_dir)
    logger.info("Dataset download complete.")


if __name__ == "__main__":
    print("--- Setup and Download Script (no login required) ---")
    base_directory = setup_environment()
    download_datasets(base_directory)
    print("Setup complete.")
