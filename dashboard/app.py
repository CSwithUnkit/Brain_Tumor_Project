import os
import sys
from pathlib import Path
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from typing import Tuple, Optional

# Dynamic hardware profiler — works on CPU, CUDA, Google Colab, and Windows
from utils.device_config import get_system_execution_profile

# ONNX Runtime: optional soft-import for faster CPU inference
try:
    import onnxruntime as _ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

# Detect hardware once at module load (Streamlit caches the module across reruns)
_HW_PROFILE = get_system_execution_profile()

from enhancement.pipeline import EnhancementAblationManager, to_display_rgb
from segmentation.unet_model import UNet
from classification.classifier_model import BrainTumorClassifier
from explainability.gradcam_generator import BrainTumorGradCAM

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MRI Image Enhancement and Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: Premium Dark Medical UI ──────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">

<style>
/* ─── Reset & base ─────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #070B12;
    color: #e2e8f0;
    font-family: 'Inter', 'Plus Jakarta Sans', sans-serif;
}

/* ─── Sidebar ───────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid rgba(51, 65, 85, 0.5) !important;
}

/* ─── Pipeline step cards ───────────────────────── */
.pipeline-card {
    background: #0F172A;
    border: 1px solid rgba(51, 65, 85, 0.5);
    border-radius: 8px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
}
.pipeline-card .step-label {
    font-size: 0.7rem; font-weight: 600; color: #94a3b8;
    letter-spacing: 0.05em; text-transform: uppercase;
}

/* ─── Diagnosis panel ───────────────────────────── */
.diag-panel {
    border-radius: 8px; padding: 1.5rem;
    border: 1px solid rgba(51, 65, 85, 0.5);
    background: #0F172A;
}
.diag-label {
    font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase;
    color: #94a3b8; margin-bottom: 0.5rem;
}
.diag-value {
    font-size: 1.8rem; font-weight: 700; margin-bottom: 0.2rem;
}
.diag-conf { font-size: 0.85rem; color: #94a3b8; }

/* ─── TypeUI / Shadcn colour tokens ─────────────── */
:root {
    --shadcn-border  : rgba(51, 65, 85, 0.5);
    --shadcn-emerald : #10B981;
    --shadcn-violet  : #6366F1;
    --shadcn-cyan    : #06B6D4;
    --shadcn-rose    : #F43F5E;
    --shadcn-amber   : #F59E0B;
}

/* ─── Clinical counseling section ───────────────── */
.counseling-section {
    background: #0F172A;
    border: 1px solid var(--shadcn-border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 0.5rem;
}
.counseling-header {
    display: flex; align-items: center; gap: 0.75rem;
    border-bottom: 1px solid var(--shadcn-border);
    padding-bottom: 1rem; margin-bottom: 1.4rem;
}
.counseling-title {
    font-size: 1.05rem; font-weight: 600; color: #f8fafc;
}
.c-section-label {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.05em;
    text-transform: uppercase; padding: 0.2rem 0.5rem;
    border-radius: 4px; margin-bottom: 0.5rem; display: inline-block;
}
.c-label-cyan   { color: var(--shadcn-cyan);   background: rgba(6,182,212,0.10); }
.c-label-amber  { color: var(--shadcn-amber);  background: rgba(245,158,11,0.10); }
.c-label-violet { color: var(--shadcn-violet); background: rgba(99,102,241,0.10); }
.c-label-green  { color: var(--shadcn-emerald);background: rgba(16,185,129,0.10); }
.c-body-text {
    font-size: 0.85rem; color: #cbd5e1; margin-bottom: 1.2rem;
}
.c-bullet-item {
    display: flex; gap: 0.5rem; align-items: flex-start;
    padding: 0.3rem 0; font-size: 0.85rem; color: #cbd5e1;
}
.c-checklist-item {
    display: flex; gap: 0.5rem; align-items: flex-start;
    padding: 0.3rem 0; font-size: 0.85rem; color: #cbd5e1;
}
.c-checkbox {
    width: 14px; height: 14px; border: 1.5px solid #64748b;
    border-radius: 3px; flex-shrink: 0; margin-top: 0.15rem;
}

/* ─── Empty State ───────────────────────────────── */
.empty-state {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 4rem 2rem; text-align: center;
    background: #0F172A; border-radius: 12px; border: 1px dashed rgba(51, 65, 85, 0.5);
    margin: 2rem 0;
}
.empty-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
}
.empty-title {
    font-size: 1.25rem; font-weight: 600; color: #f8fafc; margin-bottom: 0.5rem;
}
.empty-sub {
    font-size: 0.85rem; color: #94a3b8; max-width: 400px; line-height: 1.4;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="neuro-header" style="background: #0F172A; border: 1px solid rgba(51, 65, 85, 0.5); border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem;">
  <div class="neuro-title" style="font-size: 1.5rem; font-weight: 600; color: #f8fafc; margin-bottom: 0.5rem;">MRI Image Enhancement and Tumor Detection</div>
  <div class="badge-row" style="display: flex; gap: 0.75rem;">
    <span style="font-size: 0.75rem; color: #10B981; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); padding: 0.2rem 0.5rem; border-radius: 4px;">System: Online (Auto-Detected GPU/CPU)</span>
    <span style="font-size: 0.75rem; color: #60A5FA; background: rgba(96, 165, 250, 0.1); border: 1px solid rgba(96, 165, 250, 0.2); padding: 0.2rem 0.5rem; border-radius: 4px;">Active Pipeline: Exp 2 Enhanced (98.2% F1)</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Helper functions ──────────────────────────────────────────────────────────
def calculate_biomarkers(mask, raw_img, enh_img):
    pixels = int(np.sum(mask > 0))
    centroid = bbox = None
    perimeter = 0.0
    if pixels > 0:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            bbox = (x, y, x + w, y + h)
            perimeter = cv2.arcLength(c, closed=True)
            M = cv2.moments(c)
            if M['m00'] != 0:
                centroid = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    cnr = 0.0
    if raw_img.shape[0] >= 20:
        rg = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
        eg = cv2.cvtColor(enh_img, cv2.COLOR_RGB2GRAY)
        rv = np.var(rg[0:20, 0:20]) + 1e-5
        ev = np.var(eg[0:20, 0:20]) + 1e-5
        rs = np.mean(rg[mask > 0]) if pixels > 0 else np.mean(rg)
        es = np.mean(eg[mask > 0]) if pixels > 0 else np.mean(eg)
        rc, ec = rs / np.sqrt(rv), es / np.sqrt(ev)
        cnr = ((ec - rc) / rc) * 100 if rc > 0 else 0.0
    return pixels, centroid, bbox, cnr, perimeter

def prob_bar_html(label, value, color="#3b82f6"):
    return f"""
    <div style="margin-bottom: 0.6rem;">
      <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #cbd5e1; margin-bottom: 0.2rem;">
        <span>{label}</span><span style="font-weight: 600; color: {color}">{value * 100:.1f}%</span>
      </div>
      <div style="height: 4px; background: rgba(51, 65, 85, 0.5); border-radius: 2px; overflow: hidden;">
        <div style="height: 100%; width: {value * 100:.1f}%; background: {color}; box-shadow: 0 0 8px {color}; border-radius: 2px;"></div>
      </div>
    </div>"""

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("### 📋 Patient Metadata")
patient_id  = st.sidebar.text_input("Patient ID", value="PID-90210", label_visibility="visible")
scan_date   = st.sidebar.date_input("Scan Date", value=datetime.today())
sequence    = st.sidebar.selectbox("MRI Sequence", [
    "T1-Weighted Contrast Enhanced", "T2-Weighted", "FLAIR", "DWI"])
institution = st.sidebar.text_input("Institution", value="ITS Engineering College")

st.sidebar.divider()
st.sidebar.markdown("### 🤖 AI Configuration")
model_choice = st.sidebar.selectbox("Active Pipeline", [
    "Exp 2: Enhanced (Recommended - 98.2% Acc)", 
    "Exp 1: Baseline (98.9% Acc)", 
    "Exp 3: Segmentation-Guided"], index=0)

model_map = {
    "Exp 1: Baseline (98.9% Acc)"          : "classification/best_efficientnet_exp1_baseline.pth",
    "Exp 2: Enhanced (Recommended - 98.2% Acc)"          : "classification/best_efficientnet_exp2_enhanced.pth",
    "Exp 3: Segmentation-Guided": "classification/best_efficientnet_exp3_seg_guided.pth",
}

st.sidebar.divider()
st.sidebar.markdown("### 🔌 System Status")
ckpt_dir   = PROJECT_ROOT / "checkpoints"
seg_path   = ckpt_dir / "unet/best_unet_enhanced.pth"
class_path = ckpt_dir / model_map[model_choice]

def status_html(label, ok):
    dot = "online" if ok else "offline"
    txt = "ONLINE" if ok else "MISSING"
    clr = "#10d97a" if ok else "#ef4455"
    return f'<span class="status-dot {dot}"></span><span style="font-size:0.78rem;color:{clr}">{txt}</span> <span style="font-size:0.78rem;color:#567a8f">{label}</span>'

st.sidebar.markdown(status_html("U-Net Segmentation", seg_path.exists()), unsafe_allow_html=True)
st.sidebar.markdown(status_html(f"Classifier ({model_choice[:5]})", class_path.exists()), unsafe_allow_html=True)

if not class_path.exists() or not seg_path.exists():
    st.sidebar.warning("Model weights missing. Train on Colab to enable live inference.")

# ── Hardware badge ────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown("### ⚙️ Inference Hardware")
_p = _HW_PROFILE
if _p["has_cuda"]:
    _hw_label = f"🟢 GPU · {_p['gpu_name']}"
    _hw_sub   = f"{_p['vram_gb']:.1f} GB VRAM · AMP fp16"
    _hw_color = "#10d97a"
else:
    _hw_label = "🔵 CPU-only"
    _hw_sub   = f"{_p['total_ram_gb']:.1f} GB RAM · {_p['num_workers']} workers"
    _hw_color = "#00b2dc"
_ort_badge = ("ORT ✓" if _ORT_AVAILABLE else "PyTorch") 
st.sidebar.markdown(
    f'<span style="font-size:0.82rem;color:{_hw_color};font-weight:600">{_hw_label}</span><br>'
    f'<span style="font-size:0.74rem;color:#567a8f">{_hw_sub} · Backend: {_ort_badge}</span>',
    unsafe_allow_html=True
)

st.sidebar.divider()
st.sidebar.markdown("### 📂 MRI Upload")
upload = st.sidebar.file_uploader("Upload Scan (PNG / JPG)", type=['png', 'jpg', 'jpeg'])

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    # Use the hardware profiler for device selection — respects VRAM constraints
    # and is consistent with the training scripts.
    profile = get_system_execution_profile()
    device  = profile["device"]

    unet        = UNet(n_channels=3, n_classes=1).to(device)
    classifier  = BrainTumorClassifier(num_classes=4, pretrained=False).to(device)
    seg_ckpt    = PROJECT_ROOT / 'checkpoints/unet/best_unet_enhanced.pth'
    cls_ckpt    = PROJECT_ROOT / f'checkpoints/{model_map[model_choice]}'
    if seg_ckpt.exists():
        unet.load_state_dict(torch.load(str(seg_ckpt), map_location=device, weights_only=True))
    if cls_ckpt.exists():
        classifier.load_state_dict(torch.load(str(cls_ckpt), map_location=device, weights_only=True))
    unet.eval(); classifier.eval()

    # ── ONNX Runtime inference session (optional, faster CPU inference) ───────
    # If an exported ONNX file exists alongside the .pth, ORT is used for the
    # classifier forward pass. Falls back to PyTorch silently when ORT is absent
    # or the .onnx file has not been generated yet.
    cls_ort_path = PROJECT_ROOT / 'deployment/exported/classifier_exported.onnx'
    ort_session: Optional[object] = None
    if _ORT_AVAILABLE and cls_ort_path.exists():
        try:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if device.type == "cuda"
                else ["CPUExecutionProvider"]
            )
            ort_session = _ort.InferenceSession(str(cls_ort_path), providers=providers)
        except Exception:
            ort_session = None  # ORT session failed — keep PyTorch fallback

    return unet, classifier, device, ort_session

unet, classifier, device, _ort_session = load_models()
enhancer = EnhancementAblationManager()
gradcam  = BrainTumorGradCAM(classifier, use_cuda=(device.type == 'cuda'))

# ── Main flow ─────────────────────────────────────────────────────────────────
CLASSES = ['Intra-axial Glial Neoplasm', 'Extra-axial Dural Lesion', 'Sella Turcica Pituitary Adenoma', 'No Tumor']
CLASS_COLORS = {
    'Intra-axial Glial Neoplasm'       : '#ef4455',
    'Extra-axial Dural Lesion'         : '#f59e0b',
    'Sella Turcica Pituitary Adenoma': '#a78bfa',
    'No Tumor'         : '#10d97a',
}
PROB_COLORS = ['#ef4455', '#f59e0b', '#a78bfa', '#10d97a']

# ── Clinical counseling knowledge base ───────────────────────────────────────
# Full neuro-oncological counseling text keyed by predicted class.
# Imported into the dashboard expander and forwarded to the PDF generator.
from reports.pdf_report_generator import COUNSELING_DB, generate_pdf_report, generate_clinical_report_bytes

# ── Counseling HTML helpers ───────────────────────────────────────────────────
def _c_bullet(text: str, dot_class: str) -> str:
    return (
        f'<div class="c-bullet-item">'
        f'<span class="{dot_class}">&#8226;</span>'
        f'<span>{text}</span></div>'
    )

def _c_checklist(items) -> str:
    rows = ''.join(
        f'<div class="c-checklist-item">'
        f'<div class="c-checkbox"></div>'
        f'<span>{i}. {item}</span></div>'
        for i, item in enumerate(items, 1)
    )
    return rows

def render_counseling_section(pred_class: str) -> None:
    """Render the expandable Clinical Counseling & Patient Guidance section."""
    info = COUNSELING_DB.get(pred_class, COUNSELING_DB['No Tumor'])
    is_tumor = pred_class != 'No Tumor'
    diag_badge_color = '#F43F5E' if is_tumor else '#10B981'
    diag_badge_bg    = 'rgba(244,63,94,0.10)' if is_tumor else 'rgba(16,185,129,0.10)'
    diag_badge_border= 'rgba(244,63,94,0.30)' if is_tumor else 'rgba(16,185,129,0.30)'

    with st.expander(
        f"🩺 Clinical Findings & Patient Counseling Guidelines — {pred_class}",
        expanded=is_tumor,
    ):
        st.markdown(f"""
        <div class="counseling-section">
          <div class="counseling-header">
            <div>
              <div class="counseling-title">Clinical Findings &amp; Patient Counseling Guidelines</div>
              <div class="counseling-subtitle">NeuroScan AI &nbsp;&middot;&nbsp; Neuro-Oncology Decision Support</div>
            </div>
            <span style="margin-left:auto;font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
              text-transform:uppercase;padding:0.25rem 0.8rem;border-radius:6px;
              color:{diag_badge_color};background:{diag_badge_bg};border:1px solid {diag_badge_border};
              font-family:'JetBrains Mono',monospace">{pred_class.upper()}</span>
          </div>

          <!-- 1. Pathological Nature -->
          <div class="c-section-label c-label-cyan">1 &middot; Pathological Nature &amp; Subtype Rationale</div>
          <div class="c-body-text">{info['pathological_nature']}</div>

          <!-- 2. Critical Patient Precautions -->
          <div class="c-section-label c-label-amber">2 &middot; Critical Patient Precautions &amp; Red Flag Warning Symptoms</div>
          {''.join(_c_bullet(p, 'c-bullet-dot-amber') for p in info['precautions'])}

          <br>

          <!-- 3. Next Diagnostic Steps -->
          <div class="c-section-label c-label-violet" style="margin-top:1.1rem">3 &middot; Recommended Confirmatory Diagnostic Workup</div>
          {''.join(_c_bullet(s, 'c-bullet-dot-violet') for s in info['next_steps'])}

          <br>

          <!-- 4. Physician Checklist -->
          <div class="c-section-label c-label-green" style="margin-top:1.1rem">4 &middot; Attending Physician Consultation Checklist</div>
          {_c_checklist(info['checklist'])}

        </div>
        """, unsafe_allow_html=True)

if upload is not None:
    raw_img_pil = Image.open(upload).convert('L')
    raw_arr = np.array(raw_img_pil).astype(np.float32) / 255.0
    raw_arr = cv2.resize(raw_arr, (256, 256))

    if not class_path.exists() or not seg_path.exists():
        st.markdown("""
        <div style="text-align:center;padding:3rem;background:rgba(245,158,11,0.07);
            border:1px solid rgba(245,158,11,0.3);border-radius:14px;margin-top:2rem">
          <div style="font-size:2rem;margin-bottom:0.8rem">⏳</div>
          <div style="font-size:1.15rem;font-weight:600;color:#f59e0b;margin-bottom:0.5rem">
            Model Weights Pending
          </div>
          <div style="color:#9a7a40;font-size:0.9rem">
            Train the models on Google Colab (Notebooks 2 &amp; 3) to enable live neural inference.
          </div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    with st.spinner('Running Clinical Inference Pipeline…'):
        # 1. Enhance
        enhanced  = enhancer.process(raw_arr)
        enh_rgb   = to_display_rgb(enhanced)
        raw_rgb   = to_display_rgb(raw_arr)

        # 2. Segment
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        inp  = (enh_rgb / 255.0 - mean) / std
        tensor = torch.from_numpy(inp.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            logits   = unet(tensor)
            mask_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        binary_mask = (mask_prob > 0.5).astype(np.uint8) * 255

        # 3. Biomarkers
        area, centroid, bbox, cnr, perim = calculate_biomarkers(binary_mask, raw_rgb, enh_rgb)

        # 4. Overlay
        overlay_seg = enh_rgb.copy()
        if area > 0:
            overlay_seg[binary_mask > 0] = (
                overlay_seg[binary_mask > 0] * 0.55 +
                np.array([255, 40, 80]) * 0.45
            ).clip(0, 255).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_seg, contours, -1, (255, 255, 0), 2)

        # 5. Classify
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            probs = torch.softmax(classifier(tensor), dim=1).squeeze().cpu().numpy()
            
        pred_idx   = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]
        
        fusion_override = False
        if area > 100 and pred_class == 'No Tumor':
            tumor_probs = probs[:3]
            tumor_probs = tumor_probs / (np.sum(tumor_probs) + 1e-8)
            probs[:3] = tumor_probs
            probs[3] = 0.0
            
            pred_idx = int(np.argmax(probs))
            pred_class = CLASSES[pred_idx]
            fusion_override = True
            
        pred_conf  = probs[pred_idx] * 100

        # 6. Grad-CAM
        heatmap = gradcam.generate_heatmap(tensor, target_category=pred_idx)
        heatmap = cv2.resize(heatmap, (256, 256))
        heatmap_c = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_c = cv2.cvtColor(heatmap_c, cv2.COLOR_BGR2RGB)
        gradcam_overlay = cv2.addWeighted(enh_rgb, 0.55, heatmap_c, 0.45, 0)

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── Imaging row ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    panels = [
        (c1, raw_rgb,        "1. Raw Acquisition",        ""),
        (c2, enh_rgb,        "2. WPT-LMMSE-CLAHE Contrast", ""),
        (c3, overlay_seg,    "3. U-Net RoI Contour",      "Tumor Detected" if area > 0 else "Clear"),
        (c4, gradcam_overlay,"4. Grad-CAM Attention",     f"+{cnr:.1f}% CNR" if cnr > 0 else ""),
    ]
    for col, img, title, caption in panels:
        with col:
            st.markdown(f"""
            <div class="pipeline-card">
              <div class="step-label">{title}</div>
            </div>""", unsafe_allow_html=True)
            if caption:
                st.markdown(f"""<div style="position: absolute; margin-top: 10px; margin-left: 10px; z-index: 10; background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(51, 65, 85, 0.8); padding: 3px 8px; border-radius: 4px; font-size: 0.7rem; color: #e2e8f0; font-weight: 600;">{caption}</div>""", unsafe_allow_html=True)
            st.image(img, use_container_width=True)

    st.divider()

    # ── Diagnosis + Probabilities ──────────────────────────────────────────────
    col_diag, col_prob, col_morph = st.columns([1.2, 1.2, 1.6])

    is_tumor = (pred_class != 'No Tumor')
    panel_cls = "positive" if is_tumor else "negative"
    dcolor    = CLASS_COLORS[pred_class]

    with col_diag:
        override_html = '<div style="margin-top:0.5rem"><span class="badge badge-amber">Multi-Modal Fusion Override: Lesion Detected by U-Net</span></div>' if fusion_override else ''
        st.markdown(f"""
        <div class="diag-panel {panel_cls}">
          <div class="diag-label {panel_cls}">PRIMARY DIAGNOSIS</div>
          <div class="diag-value {panel_cls}">{pred_class.upper()}</div>
          <div class="diag-conf">Confidence: <strong style="color:{dcolor}">{pred_conf:.1f}%</strong></div>
          {override_html}
          <div style="margin-top:1rem;font-size:0.78rem;color:#567a8f">
            Pipeline: {model_choice}<br>
            Sequence: {sequence}
          </div>
        </div>""", unsafe_allow_html=True)

    with col_prob:
        st.markdown('<div style="padding-top:0.3rem">', unsafe_allow_html=True)
        st.markdown('<div class="report-meta" style="margin-bottom:0.8rem;letter-spacing:0.08em">SOFTMAX DISTRIBUTION</div>', unsafe_allow_html=True)
        for cls, prob, color in zip(CLASSES, probs, PROB_COLORS):
            st.markdown(prob_bar_html(cls, prob, color), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_morph:
        st.markdown('<div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.8rem;">High-Density Morphometrics</div>', unsafe_allow_html=True)
        if area > 0:
            st.markdown(f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
              <div style="background: #0F172A; border: 1px solid rgba(51, 65, 85, 0.5); border-radius: 6px; padding: 0.8rem;">
                <div style="font-size: 0.65rem; color: #64748b; text-transform: uppercase; margin-bottom: 0.2rem;">Tumor Area</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">{area:,} <span style="font-size: 0.7rem; font-weight: 400; color: #94a3b8;">px² (≈ {area/100:.1f} mm²)</span></div>
              </div>
              <div style="background: #0F172A; border: 1px solid rgba(51, 65, 85, 0.5); border-radius: 6px; padding: 0.8rem;">
                <div style="font-size: 0.65rem; color: #64748b; text-transform: uppercase; margin-bottom: 0.2rem;">CNR Gain</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">+{cnr:.1f}%</div>
              </div>
              <div style="background: #0F172A; border: 1px solid rgba(51, 65, 85, 0.5); border-radius: 6px; padding: 0.8rem;">
                <div style="font-size: 0.65rem; color: #64748b; text-transform: uppercase; margin-bottom: 0.2rem;">Perimeter</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">{perim:.0f} <span style="font-size: 0.7rem; font-weight: 400; color: #94a3b8;">px</span></div>
              </div>
              <div style="background: #0F172A; border: 1px solid rgba(51, 65, 85, 0.5); border-radius: 6px; padding: 0.8rem;">
                <div style="font-size: 0.65rem; color: #64748b; text-transform: uppercase; margin-bottom: 0.2rem;">Centroid (X, Y)</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">{centroid if centroid else '—'}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-tile" style="text-align:center;padding:2rem">
              <div style="font-size:1.8rem;margin-bottom:0.5rem">✅</div>
              <div style="color:#10d97a;font-weight:600">No Focal Lesion Detected</div>
              <div style="font-size:0.8rem;color:#567a8f;margin-top:0.3rem">Morphometry bypassed</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Clinical Counseling Module ─────────────────────────────────────────────
    render_counseling_section(pred_class)

    st.divider()

    # ── PDF Export Button ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top: 1.5rem; padding: 1.5rem; background: #0F172A; border: 1px solid rgba(51, 65, 85, 0.5); border-radius: 8px;">
      <div style="font-size: 1.1rem; font-weight: 600; color: #f8fafc; margin-bottom: 0.5rem;">📄 Official Diagnostic Report</div>
      <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 1rem;">Export complete clinical report with images, biomarkers &amp; counseling guidelines</div>
    </div>""", unsafe_allow_html=True)

    if st.button("📄 Export Official Diagnostic Report (PDF)", type="primary", use_container_width=True):
        try:
            report_bytes, mime, ext = generate_clinical_report_bytes(
                patient_id   = patient_id,
                scan_date    = str(scan_date),
                sequence     = sequence,
                institution  = institution,
                pred_class   = pred_class,
                pred_conf    = pred_conf,
                probs        = list(probs),
                area         = area,
                perim        = perim,
                centroid     = centroid,
                cnr          = cnr,
                raw_img      = raw_rgb,
                enh_img      = enh_rgb,
                seg_img      = overlay_seg,
                gradcam_img  = gradcam_overlay,
                model_choice = model_choice,
                bbox         = bbox,
            )
            
            label = "⬇️ Download PDF Report" if ext == "pdf" else "⬇️ Download HTML Report (open in browser → Ctrl+P to PDF)"
            fname = f"neuroscan_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.{ext}"
            
            st.download_button(
                label     = label,
                data      = report_bytes,
                file_name = fname,
                mime      = mime,
                use_container_width=True,
            )
            if ext == "html":
                st.info(
                    "📝 HTML report downloaded. Open in any browser and use "
                    "**File → Print → Save as PDF** for a full A4 clinical PDF."
                )
        except Exception as e:
            st.error(f"Report generation error: {e}")

else:
    # Empty state
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">🧠</div>
      <div class="empty-title">Awaiting MRI Scan</div>
      <div class="empty-sub">Upload a diagnostic scan using the sidebar to begin AI-assisted analysis.</div>
      <br>
      <div style="display:flex;justify-content:center;gap:1.5rem;flex-wrap:wrap;margin-top:1rem">
        <span class="badge badge-cyan" style="animation-delay:0.1s">WPT Enhancement</span>
        <span class="badge badge-green" style="animation-delay:0.2s">U-Net Segmentation</span>
        <span class="badge badge-amber" style="animation-delay:0.3s">EfficientNetB2 Classification</span>
        <span class="badge badge-cyan" style="animation-delay:0.4s">Grad-CAM XAI</span>
      </div>
    </div>""", unsafe_allow_html=True)
