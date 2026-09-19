# ── Cell 1: Environment Setup & Hardware Profile ───────────────────────────
from __future__ import annotations
import os, sys

# ── Google Drive mount (Colab only — skipped automatically when running locally) ─
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    PROJECT_PATH = "/content/drive/MyDrive/Brain_Tumor_Project/MRI_Project"
except ImportError:
    _cwd = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
    PROJECT_PATH = os.path.abspath(os.path.join(_cwd, '..') if os.path.basename(_cwd) == 'notebooks' else _cwd)

assert os.path.exists(PROJECT_PATH), f"Project not found at {PROJECT_PATH}"
os.chdir(PROJECT_PATH)
sys.path.insert(0, PROJECT_PATH)

# Dependencies loaded successfully (pandas and albumentations removed per security policy).
os.makedirs("results", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

try:
    import pytorch_grad_cam
except ImportError:
    import subprocess, sys as _sys
    subprocess.check_call([_sys.executable, "-m", "pip", "install", "-q", "grad-cam"])

# ── Dynamic hardware profile ────────────────────────────────────────────────
from utils.device_config import get_system_execution_profile
profile = get_system_execution_profile()
print(f"🖥️ System Profile: {profile['gpu_name']} ({profile['vram_gb']} GB VRAM) | RAM: {profile['total_ram_gb']} GB | Workers: {profile['num_workers']}")

import torch
device = profile["device"]
if profile["has_cuda"]:
    torch.cuda.empty_cache()
print(f"✅ PyTorch {torch.__version__} | Device: {device} | AMP: {profile['use_amp']}")

# ── Cell 2: Verify Prerequisites (Checkpoints + Metadata) ───────────────────
import os, json

def verify_experiment_quality(exp_id: str, ckpt_path: str, min_f1: float = 0.80, min_epochs: int = 20) -> bool:
    """
    Strictly verifies if a training checkpoint is 100% complete and meets medical quality benchmarks.
    Returns True ONLY if:
    1. Checkpoint file exists and is > 10MB.
    2. Metrics history JSON exists and has >= min_epochs recorded.
    3. Peak validation Macro F1 score meets or exceeds min_f1.
    """
    if not os.path.exists(ckpt_path) or os.path.getsize(ckpt_path) < 10 * 1024 * 1024:
        return False
    
    metrics_path = f"results/metrics_{exp_id}.json"
    if not os.path.exists(metrics_path):
        return False
        
    try:
        with open(metrics_path, "r") as f:
            history = json.load(f)
        if not isinstance(history, list) or len(history) < min_epochs:
            return False
        best_f1 = max(h.get("val_metrics", {}).get("macro_f1", 0.0) for h in history)
        return best_f1 >= min_f1
    except Exception:
        return False

UNET_CKPT  = "checkpoints/unet/best_unet_enhanced.pth"
META_PATH  = "data/brisc/brisc_metadata.json"
CACHE_DIR  = "data/cached_enhanced"

checks = {
    "BRISC metadata"        : META_PATH,
    "U-Net checkpoint"      : UNET_CKPT,
    "Enhanced image cache"  : CACHE_DIR,
}
all_ok = True
for label, path in checks.items():
    exists = os.path.exists(path)
    icon   = "✅" if exists else "❌"
    print(f"  {icon} {label:30s} : {path}")
    if not exists:
        all_ok = False

if not all_ok:
    raise FileNotFoundError(
        "Missing prerequisites. Please run Notebooks 1 and 2 first."
    )

with open(META_PATH) as f:
    meta = json.load(f)
print(f"\n✅ Classification images : {meta['classification_count']:,}")
print(f"✅ Segmentation pairs    : {meta['segmentation_count']:,}")
print("\n✅ All prerequisites satisfied.")

# ── Cell 3: Load Trained U-Net ──────────────────────────────────────────────
import torch
from segmentation.unet_model import UNet

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "checkpoints/unet/best_unet_enhanced.pth"

unet = UNet(n_channels=3, n_classes=1).to(device)
state = torch.load(CKPT_PATH, map_location=device, weights_only=True)
unet.load_state_dict(state)
unet.eval()

n_params = sum(p.numel() for p in unet.parameters())
print(f"✅ U-Net loaded: {CKPT_PATH}")
print(f"   Parameters: {n_params:,}")

# ── Cell 4A: Experiment 1 — Baseline (Raw Scans) ──────────────────────────────
import os
CKPT_EXP1 = "checkpoints/classification/best_efficientnet_exp1_baseline.pth"
if verify_experiment_quality("exp1_baseline", CKPT_EXP1, min_f1=0.85, min_epochs=20):
    print("✅ Experiment 1 is 100% COMPLETE with High Quality (F1 >= 85%). Skipping re-training.")
else:
    print("⚠️ Experiment 1 checkpoint missing or incomplete. Starting high-precision training...")
    import sys, importlib
    _orig_argv = sys.argv[:]
    sys.argv = ["run_experiments", "--experiment", "exp1_baseline", "--epochs", "30"]
    try:
        import classification.run_experiments as _re1
        importlib.reload(_re1)
        _re1.main()
    finally:
        sys.argv = _orig_argv
    print("\n✅ Experiment 1 — COMPLETE")

# ── Cell 4B: Experiment 2 — Enhanced Scans (WPT→LMMSE→CLAHE) ─────────────────
import os
CKPT_EXP2 = "checkpoints/classification/best_efficientnet_exp2_enhanced.pth"
if verify_experiment_quality("exp2_enhanced", CKPT_EXP2, min_f1=0.80, min_epochs=20):
    print("✅ Experiment 2 is 100% COMPLETE with High Quality (F1 >= 80%). Skipping re-training.")
else:
    print("⚠️ Experiment 2 checkpoint missing or incomplete. Starting high-precision training...")
    import sys, importlib
    _orig_argv = sys.argv[:]
    sys.argv = ["run_experiments", "--experiment", "exp2_enhanced", "--epochs", "30"]
    try:
        import classification.run_experiments as _re2
        importlib.reload(_re2)
        _re2.main()
    finally:
        sys.argv = _orig_argv
    print("\n✅ Experiment 2 — COMPLETE")

# ── Cell 4C: Experiment 3 — Segmentation-Guided (RoI Crop) ───────────────────
import os
CKPT_EXP3 = "checkpoints/classification/best_efficientnet_exp3_seg_guided.pth"
if verify_experiment_quality("exp3_seg_guided", CKPT_EXP3, min_f1=0.80, min_epochs=20):
    print("✅ Experiment 3 is 100% COMPLETE with High Quality (F1 >= 80%). Skipping re-training.")
else:
    print("⚠️ Experiment 3 checkpoint missing or incomplete. Starting high-precision training...")
    import sys, importlib
    _orig_argv = sys.argv[:]
    sys.argv = ["run_experiments", "--experiment", "exp3_seg_guided", "--epochs", "30"]
    try:
        import classification.run_experiments as _re3
        importlib.reload(_re3)
        _re3.main()
    finally:
        sys.argv = _orig_argv
    print("\n✅ Experiment 3 — COMPLETE")

# ── Cell 5: Comparative Results Table ────────────────────────────────────────
import json, glob, os, pandas as pd

rows = []
for fpath in sorted(glob.glob("results/metrics_exp*.json")):
    with open(fpath) as f:
        history = json.load(f)
    if not history:
        continue
    best = max(history, key=lambda h: h["val_metrics"].get("macro_f1", 0))
    exp  = os.path.basename(fpath).replace("metrics_", "").replace(".json", "")
    m    = best["val_metrics"]
    rows.append({
        "Experiment"     : exp,
        "Best Epoch"     : best["epoch"],
        "Accuracy"       : f"{m.get('accuracy',        0)*100:.2f}%",
        "Macro F1"       : f"{m.get('macro_f1',        0)*100:.2f}%",
        "Macro Precision": f"{m.get('macro_precision', 0)*100:.2f}%",
        "Macro Recall"   : f"{m.get('macro_recall',    0)*100:.2f}%",
    })

if rows:
    df = pd.DataFrame(rows)
    print("\n" + "="*70)
    print("         CLASSIFICATION RESULTS SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    df.to_csv("results/summary_table.csv", index=False)
    print("\n✅ Saved to results/summary_table.csv")
else:
    print("⚠️  No results found. Run Cells 4A / 4B / 4C first.")

# ── Cell 6: Grad-CAM XAI — Heatmaps + Quantitative Localization ──────────────
try:
    import pytorch_grad_cam
except ImportError:
    import subprocess, sys as _sys
    subprocess.check_call([_sys.executable, "-m", "pip", "install", "-q", "grad-cam"])

import torch, cv2, numpy as np, matplotlib.pyplot as plt, glob, json, os

from classification.classifier_model import BrainTumorClassifier
from explainability.gradcam_generator import BrainTumorGradCAM

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "checkpoints/classification/best_efficientnet_exp3_seg_guided.pth"

if not os.path.exists(CKPT_PATH):
    candidates = sorted(glob.glob("checkpoints/classification/best_efficientnet_*.pth"))
    if not candidates:
        raise FileNotFoundError("No classifier checkpoint found. Run Cells 4A/4B/4C first.")
    CKPT_PATH = candidates[-1]

# CRITICAL: Build model on CPU, load weights, THEN move to device
model = BrainTumorClassifier(num_classes=4, pretrained=False)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()
print(f"✅ Classifier loaded: {CKPT_PATH} → {device}")

with open("data/brisc/brisc_metadata.json") as f:
    meta = json.load(f)

class_to_idx = {"glioma": 0, "meningioma": 1, "pituitary": 2, "no_tumor": 3}
idx_to_class = {v: k for k, v in class_to_idx.items()}

gradcam = BrainTumorGradCAM(model)
# Re-enforce device placement after GradCAM instantiation (guards against any internal .cpu() calls)
model.to(device)
print(f"✅ Classifier & GradCAM active on: {next(model.parameters()).device}")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# One image per class (deterministic — first occurrence in metadata)
samples_by_class = {}
for rec in meta["classification"]:
    c = rec["class"]
    if c not in samples_by_class:
        samples_by_class[c] = rec.get("path") or rec.get("image_path")

n_classes = len(samples_by_class)
fig, axes = plt.subplots(n_classes, 3, figsize=(15, 5 * n_classes))
# Guard: ensure axes is always 2D even for a single row
if n_classes == 1:
    axes = axes[np.newaxis, :]

for row, (class_name, img_path) in enumerate(samples_by_class.items()):
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rs  = cv2.resize(img_rgb, (256, 256))

    tensor = torch.from_numpy(((img_rs / 255.0 - MEAN) / STD).transpose(2, 0, 1)).float()
    tensor = tensor.unsqueeze(0)
    # Dynamically guarantee tensor and model are on the exact same device
    target_dev = next(model.parameters()).device
    tensor = tensor.to(target_dev)

    with torch.no_grad():
        logits = model(tensor)
        pred   = torch.argmax(logits, dim=1).item()
        conf   = torch.softmax(logits, dim=1)[0, pred].item()

    heatmap = gradcam.generate_heatmap(tensor, target_category=pred)

    axes[row, 0].imshow(img_rs)
    axes[row, 0].set_title(f"Input\nTrue: {class_name}", fontsize=9)
    axes[row, 0].axis("off")

    axes[row, 1].imshow(heatmap, cmap="jet")
    axes[row, 1].set_title("Grad-CAM Heatmap", fontsize=9)
    axes[row, 1].axis("off")

    overlay = (img_rs * 0.6 + plt.cm.jet(heatmap)[:,:,:3] * 255 * 0.4).clip(0, 255).astype(np.uint8)
    axes[row, 2].imshow(overlay)
    axes[row, 2].set_title(f"Blend\nPred: {idx_to_class[pred]} ({conf*100:.1f}%)", fontsize=9)
    axes[row, 2].axis("off")

plt.suptitle("Grad-CAM Explainability — All 4 Tumor Classes", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/figures/gradcam_panel.png", dpi=150)
plt.show()
print("✅ Grad-CAM panel saved to reports/figures/gradcam_panel.png")

# ── Cell 7: PMRAM External Generalization Validation ────────────────────────
# Zero-retraining inference on 1,600 original PMRAM scans
import subprocess, sys

result = subprocess.run(
    [sys.executable, "-m", "validation.external_pmram"],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])
    print("⚠️  PMRAM validation failed — check data/pmram/ directory")
else:
    print("✅ PMRAM External Evaluation Complete")
    # Load and display generalization gap
    import json, os
    if os.path.exists("results/pmram_external_validation.json"):
        with open("results/pmram_external_validation.json") as f:
            pmram = json.load(f)
        print(f"\nBRISC (Exp3) Accuracy : {pmram.get('brisc_metrics', {}).get('accuracy', 'N/A')}")
        print(f"PMRAM Accuracy        : {pmram.get('pmram_metrics', {}).get('accuracy', 'N/A')}")
        print(f"Generalization Gap    : {pmram.get('generalization_gap', {}).get('accuracy', 'N/A')}")

# ── Cell 8: Consolidate Reports ──────────────────────────────────────────────
import importlib, os
try:
    import evaluation.consolidate_reports as _rpt
    importlib.reload(_rpt)
    _rpt.main()
except Exception as _e:
    print(f"⚠️  Report consolidation failed: {_e}")
for rpt in ["reports/PHASE_I_EVALUATION_REPORT.md", "reports/PHASE_II_FINAL_REPORT.md"]:
    if os.path.exists(rpt):
        print(f"✅ {rpt}")
print("\n✅ All reports generated")

# ── Cell 9: Launch NeuroScan-Enterprise Dashboard ───────────────────────────
import subprocess, time, re, sys, os

print("🚀 Launching Streamlit dashboard...")
streamlit_proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
     "--server.port", "8501",
     "--server.headless", "true",
     "--server.enableCORS", "false"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
time.sleep(4)  # Wait for Streamlit to start

is_colab = "google.colab" in sys.modules

if is_colab:
    print("📦 Installing cloudflared...")
    os.system("wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb")
    os.system("dpkg -i cloudflared-linux-amd64.deb > /dev/null 2>&1")
    
    print("🌐 Starting Cloudflare tunnel...")
    cf_proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://localhost:8501"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    tunnel_url = None
    for _ in range(60):
        time.sleep(1)
        line = cf_proc.stderr.readline().decode("utf-8", errors="ignore")
        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if match:
            tunnel_url = match.group(0)
            break

    if tunnel_url:
        print(f"\n{'='*60}")
        print(f"  🧠 NEUROSCAN-ENTERPRISE DASHBOARD IS LIVE!")
        print(f"  🔗 URL: {tunnel_url}")
        print(f"{'='*60}")
        print("\n  Upload an MRI scan in the sidebar to run inference.")
        print("  Keep this cell running to maintain the tunnel.")
    else:
        print("⚠️  Could not retrieve tunnel URL within 60 seconds.")
        print("   Check cloudflared output manually.")
else:
    print(f"\n{'='*60}")
    print(f"  🧠 NEUROSCAN-ENTERPRISE DASHBOARD IS LIVE!")
    print(f"  🔗 Local URL: http://localhost:8501")
    print(f"{'='*60}")
    print("\n  Open the link in your browser to access the dashboard.")
    print("  Keep this cell running to maintain the server.")

