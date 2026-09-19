import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_colab() -> bool:
    """Detects if running in Google Colab."""
    try:
        import google.colab
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
    """Creates the necessary folder structure."""
    dirs_to_create = [
        base_dir / "datasets" / "raw" / "brisc",
        base_dir / "datasets" / "raw" / "pmram",
        base_dir / "datasets" / "processed" / "train",
        base_dir / "datasets" / "processed" / "val",
        base_dir / "datasets" / "processed" / "test",
        base_dir / "checkpoints" / "unet",
        base_dir / "checkpoints" / "efficientnetb2",
        base_dir / "logs",
        base_dir / "configs",
        base_dir / "data"
    ]
    
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {d}")

def download_datasets(base_dir: Path) -> None:
    """
    Automates downloading of datasets using Kaggle API or direct URL.
    """
    logger.info("Starting dataset download process...")
    
    # BRISC Dataset
    brisc_dir = base_dir / "datasets" / "raw" / "brisc"
    logger.info(f"Downloading BRISC dataset to {brisc_dir}...")
    # NOTE: Replace 'brisc-dataset-name' with actual Kaggle dataset ID
    # os.system(f"kaggle datasets download -d brisc-dataset-name -p {brisc_dir} --unzip")
    
    # PMRAM Dataset
    pmram_dir = base_dir / "datasets" / "raw" / "pmram"
    logger.info(f"Downloading PMRAM dataset to {pmram_dir}...")
    # NOTE: Replace 'pmram-dataset-name' with actual Kaggle dataset ID
    # os.system(f"kaggle datasets download -d pmram-dataset-name -p {pmram_dir} --unzip")

if __name__ == "__main__":
    logger.info("--- Setup and Download Script ---")
    logger.info("Instructions for Kaggle API:")
    logger.info("1. Go to your Kaggle account settings and create a New API Token.")
    logger.info("2. Place kaggle.json in ~/.kaggle/kaggle.json and set permissions (chmod 600).")
    logger.info("   In Colab: Upload kaggle.json and run `mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`")
    
    base_directory = setup_environment()
    create_directory_structure(base_directory)
    download_datasets(base_directory)
    logger.info("Setup complete.")
