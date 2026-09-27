import logging
import os
import subprocess
from pathlib import Path

# Canonical on-disk layout — MUST match data/dataset_ingestion.py, which
# globs data/brisc and data/pmram (previously this module created
# datasets/raw/... while ingestion searched data/..., so downloaded data
# was never found).
BRISC_RAW_DIR = Path("data") / "brisc"
PMRAM_RAW_DIR = Path("data") / "pmram"
CACHED_ENHANCED_DIR = Path("data") / "cached_enhanced"

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def is_colab() -> bool:
    """Detects if running in Google Colab."""
    try:
        import google.colab  # noqa: F401 — presence probe, not a used import

        return True
    except ImportError:
        return False


def setup_environment() -> Path:
    """Sets up Google Drive mounting and base directories."""
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

            # Change working directory to drive
            os.chdir(base_dir)
            logger.info(f"Mounted Drive and set base directory to {base_dir}")
        except Exception as e:
            logger.error(f"Failed to mount Google Drive: {e}")
            logger.info("Falling back to local storage in Colab.")
    else:
        logger.info(f"Detected local environment. Base directory: {base_dir}")

    return base_dir


def create_directory_structure(base_dir: Path) -> None:
    """Creates the necessary folder structure (canonical data/ layout)."""
    dirs_to_create = [
        base_dir / BRISC_RAW_DIR,
        base_dir / PMRAM_RAW_DIR,
        base_dir / CACHED_ENHANCED_DIR,
        base_dir / "checkpoints" / "unet",
        base_dir / "checkpoints" / "classification",
        base_dir / "logs",
        base_dir / "results",
    ]

    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {d}")


def download_datasets(base_dir: Path) -> None:
    """
    Download datasets using the Kaggle API into the canonical data/ layout.

    Raises NotImplementedError until real Kaggle dataset IDs are configured —
    previously this function only LOGGED "downloading..." while doing
    nothing, silently leaving an empty dataset behind.
    """
    brisc_id = os.environ.get("BRISC_KAGGLE_ID", "").strip()
    pmram_id = os.environ.get("PMRAM_KAGGLE_ID", "").strip()
    if not brisc_id or not pmram_id:
        raise NotImplementedError(
            "Dataset download is not configured: set the BRISC_KAGGLE_ID and "
            "PMRAM_KAGGLE_ID environment variables to the Kaggle dataset slugs "
            "(e.g. export BRISC_KAGGLE_ID='your-username/brisc-dataset'), "
            "place kaggle.json at ~/.kaggle/kaggle.json (chmod 600), then re-run."
        )

    for dataset_id, target in (
        (brisc_id, base_dir / BRISC_RAW_DIR),
        (pmram_id, base_dir / PMRAM_RAW_DIR),
    ):
        logger.info(f"Downloading {dataset_id} to {target}...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(target), "--unzip"],
            check=True,
        )
    logger.info("Dataset download complete.")


if __name__ == "__main__":
    logger.info("--- Setup and Download Script ---")
    logger.info("Instructions for Kaggle API:")
    logger.info("1. Go to your Kaggle account settings and create a New API Token.")
    logger.info("2. Place kaggle.json in ~/.kaggle/kaggle.json and set permissions (chmod 600).")
    logger.info(
        "   In Colab: Upload kaggle.json and run `mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`"
    )

    base_directory = setup_environment()
    create_directory_structure(base_directory)
    download_datasets(base_directory)
    logger.info("Setup complete.")
