# ── Cell 1: Environment Setup & Hardware Profile ───────────────────────────
from __future__ import annotations
import os, sys

# ── Google Drive mount (Colab only — skipped automatically when running locally) ─
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    PROJECT_PATH = "/content/drive/MyDrive/Brain_Tumor_Project/MRI_Project"
except ImportError:
    # Running locally: resolve project root from this notebook's location
    _cwd = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
    PROJECT_PATH = os.path.abspath(os.path.join(_cwd, '..') if os.path.basename(_cwd) == 'notebooks' else _cwd)

assert os.path.exists(PROJECT_PATH), (
    f"Project not found at {PROJECT_PATH}.\n"
    "Colab: upload MRI_Project to Google Drive under Brain_Tumor_Project/.\n"
    "Local: run this notebook from the MRI_Project directory."
)
os.chdir(PROJECT_PATH)
sys.path.insert(0, PROJECT_PATH)

# Dependencies loaded successfully (pandas and albumentations removed per security policy).

print(f"✅ Working Directory: {os.getcwd()}")
print(f"✅ Python Path configured")

# ── Dynamic hardware profile ────────────────────────────────────────────────
from utils.device_config import get_system_execution_profile
profile = get_system_execution_profile()
print(f"🖥️ System Profile: {profile['gpu_name']} ({profile['vram_gb']} GB VRAM) | RAM: {profile['total_ram_gb']} GB | Workers: {profile['num_workers']}")
print(f"   Device: {profile['device']} | AMP: {profile['use_amp']} | Batch size: {profile['batch_size']}")

# ── Cell 2: Install Dependencies ────────────────────────────────────────────
import subprocess, sys

print("Installing dependencies...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q",
     "--only-binary=:all:"],
    capture_output=True, text=True
)
if result.returncode != 0:
    # Fallback: install without binary constraint
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
        check=True
    )

# Verify critical imports
import torch, cv2, pywt, albumentations
print(f"✅ PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
print(f"✅ OpenCV {cv2.__version__} | PyWavelets {pywt.__version__}")
print(f"✅ All dependencies installed successfully")

# ── Cell 3: Kaggle API Configuration ────────────────────────────────────────
import os, shutil, json

KAGGLE_DIR = os.path.expanduser("~/.kaggle")
KAGGLE_JSON = os.path.join(KAGGLE_DIR, "kaggle.json")
os.makedirs(KAGGLE_DIR, exist_ok=True)

local_kaggle = "kaggle.json"
if os.path.exists(local_kaggle):
    shutil.copy(local_kaggle, KAGGLE_JSON)
    os.chmod(KAGGLE_JSON, 0o600)
    with open(KAGGLE_JSON) as f:
        creds = json.load(f)
    print(f"✅ Kaggle API configured for user: {creds.get('username', 'unknown')}")
else:
    print("⚠️  kaggle.json not found in project root.")
    print("   Upload it manually or place it at: kaggle.json")
    print("   Get yours from: https://www.kaggle.com/settings → API → Create New Token")

# ── Cell 4: Dataset Download & Extraction ───────────────────────────────────
import os

# ─── BRISC Dataset ───
# Replace the dataset ID below with your actual Kaggle BRISC dataset slug
BRISC_DATASET_ID = "your-username/brisc-brain-tumor-dataset"  # ← UPDATE THIS
PMRAM_DATASET_ID = "your-username/pmram-brain-mri"            # ← UPDATE THIS

if not os.path.exists("data/brisc/brisc2025"):
    print("Downloading BRISC dataset...")
    os.makedirs("data/brisc", exist_ok=True)
    os.system(f"kaggle datasets download -d {BRISC_DATASET_ID} -p data/brisc --unzip -q")
    print("✅ BRISC downloaded and extracted")
else:
    print("✅ BRISC already present — skipping download")

if not os.path.exists("data/pmram"):
    print("Downloading PMRAM dataset...")
    os.makedirs("data/pmram", exist_ok=True)
    os.system(f"kaggle datasets download -d {PMRAM_DATASET_ID} -p data/pmram --unzip -q")
    print("✅ PMRAM downloaded and extracted")
else:
    print("✅ PMRAM already present — skipping download")

# Verify structure
import subprocess
result = subprocess.run(["find", "data/brisc", "-mindepth", "2", "-maxdepth", "2", "-type", "d"],
                        capture_output=True, text=True)
print("\nBRISC directory structure:")
print(result.stdout[:1000] or "  (empty — check dataset ID above)")

# ── Cell 5: High-Speed Structural Dataset Ingestion (<5 seconds) ────────────
import time

print("Running structural ingestion (no MD5 hashing — ultra-fast)...")
t0 = time.time()

# Run ingestion module
import subprocess, sys
result = subprocess.run(
    [sys.executable, "-m", "data.dataset_ingestion"],
    capture_output=True, text=True
)
elapsed = time.time() - t0

print(result.stdout)
if result.returncode != 0:
    print("⚠️  STDERR:", result.stderr[-2000:])
    raise RuntimeError("dataset_ingestion failed — check dataset paths above")

print(f"⏱  Completed in {elapsed:.1f}s")

# Load and display the produced metadata
import json
with open("data/brisc/brisc_metadata.json") as f:
    meta = json.load(f)

print(f"\n{'='*50}")
print(f"  Classification images : {meta['classification_count']:,}")
print(f"  Segmentation pairs    : {meta['segmentation_count']:,}")
print(f"{'='*50}")
assert meta["classification_count"] >= 6000, "Too few classification images!"
assert meta["segmentation_count"] >= 4700,   "Too few segmentation pairs!"
print("✅ Dataset integrity verified")

# ── Cell 6: Offline WPT→LMMSE→CLAHE Enhancement Caching ──────────────────
# Eliminates all on-the-fly CPU enhancement overhead during GPU training
# Expected: ~4,793 segmentation images + 6,000 classification images
import os, cv2, glob, numpy as np, shutil
from tqdm.auto import tqdm
from enhancement.pipeline import EnhancementAblationManager

CACHE_DIR = "data/cached_enhanced"

# ── PURGE stale cache so old corrupted files never mix with new ones ─────────
if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)
    print(f"\U0001f9f9 Purged stale cache: {CACHE_DIR}")
os.makedirs(CACHE_DIR, exist_ok=True)
print(f"\u2705 Clean cache directory created: {CACHE_DIR}")

manager = EnhancementAblationManager()   # defaults to wpt_lmmse_clahe

# ── Strictly include only MRI scan images; exclude ALL mask files ───────────
all_paths = (
    glob.glob("data/brisc/**/*.jpg", recursive=True) +
    glob.glob("data/brisc/**/*.png", recursive=True)
)
image_paths = [
    p for p in all_paths
    if "mask" not in os.path.basename(p).lower()          # no mask_ filenames
    and os.path.basename(os.path.dirname(p)).lower() != "masks"  # not inside masks/
    and CACHE_DIR not in p.replace(os.sep, "/")           # not already cached
]

print(f"Total MRI scan images : {len(image_paths):,}")
print(f"To process now        : {len(image_paths):,} (full fresh run)")

errors = 0
for path in tqdm(image_paths, desc="WPT\u2192LMMSE\u2192CLAHE", unit="img", dynamic_ncols=True):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        errors += 1
        continue
    img_float = img.astype(np.float32) / 255.0
    enhanced  = manager.process(img_float)   # returns float32 in [0, 1]
    enhanced_uint8 = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(CACHE_DIR, os.path.basename(path)), enhanced_uint8)

total_cached = len(os.listdir(CACHE_DIR))
print(f"\n\u2705 Enhancement caching complete!")
print(f"   Total cached images : {total_cached:,}")
print(f"   Errors (unreadable) : {errors}")

# ── Cell 7: Enhancement Ablation Quality Metrics Table ──────────────────────
import subprocess, sys
result = subprocess.run(
    [sys.executable, "-m", "enhancement.pipeline"],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-500:])

# ── Cell 8: Visual QC — Raw vs Enhanced Side-by-Side (Diverse Percentile Sampling) ──
import cv2, numpy as np, matplotlib.pyplot as plt, glob, os

CACHE_DIR = "data/cached_enhanced"

# ── Cache health check: auto-delete any near-zero (corrupted) entries ────────
stale = []
for p in glob.glob(f"{CACHE_DIR}/*.png") + glob.glob(f"{CACHE_DIR}/*.jpg"):
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None or img.max() <= 5:
        stale.append(p)
if stale:
    for p in stale:
        os.remove(p)
    print(f"\U0001f9f9 Auto-deleted {len(stale)} corrupted cache entries")

# ── Collect all valid cached images (JPG and PNG) ────────────────────────
cached_paths = sorted(
    glob.glob(f"{CACHE_DIR}/*.jpg") + glob.glob(f"{CACHE_DIR}/*.png")
)
print(f"\u2705 {len(cached_paths):,} valid enhanced images available for QC")

if not cached_paths:
    print("No cached images found. Re-run Cell 6.")
else:
    # ── Sample 3 diverse slices at 25%, 50%, 75% index positions ─────────────
    n = len(cached_paths)
    indices = [int(n * 0.25), int(n * 0.50), int(n * 0.75)]
    samples = [cached_paths[i] for i in indices]

    fig, axes = plt.subplots(len(samples), 2, figsize=(12, 4 * len(samples)))
    if len(samples) == 1:
        axes = [axes]

    for row, cached_path in enumerate(samples):
        basename = os.path.basename(cached_path)
        raw_candidates = glob.glob(f"data/brisc/**/{basename}", recursive=True)

        enhanced_img = cv2.imread(cached_path, cv2.IMREAD_GRAYSCALE)

        ax_raw = axes[row][0]
        ax_enh = axes[row][1]

        if raw_candidates:
            raw_img = cv2.imread(raw_candidates[0], cv2.IMREAD_GRAYSCALE)
            ax_raw.imshow(raw_img, cmap="gray", vmin=0, vmax=255)
            ax_raw.set_title(
                f"Raw MRI\n[{raw_img.min()}, {raw_img.max()}] — {basename}",
                fontsize=9
            )
        else:
            ax_raw.text(0.5, 0.5, "Raw not found", ha="center", va="center",
                       transform=ax_raw.transAxes)
            ax_raw.set_title(f"Raw MRI ({basename})")
        ax_raw.axis("off")

        ax_enh.imshow(enhanced_img, cmap="gray", vmin=0, vmax=255)
        ax_enh.set_title(
            f"WPT→LMMSE→CLAHE — [{enhanced_img.min()}, {enhanced_img.max()}]",
            fontsize=10, color="teal"
        )
        ax_enh.axis("off")

    plt.suptitle(
        "Enhancement QC: Raw vs Processed MRI Scans\n"
        "(Samples at 25%, 50%, 75% of sorted cache)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()
    print(f"\u2705 Visualized {len(samples)} diverse brain slices")
    print("   Expected: Enhanced pixel range should be [~0, ~200\u2013255] for valid MRI scans.")

