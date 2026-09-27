"""NeuroScan AI — medical-grade MRI brain tumor decision-support dashboard.

Design principles: clean + minimal, only necessary controls, easy to use,
responsive, fast (system fonts, cached inference), subtle animations,
plain-language results. All inference logic is unchanged from the
validated pipeline; only presentation was redesigned.
"""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from classification.classifier_model import BrainTumorClassifier
from classification.masking_utils import apply_exp3_guidance_numpy
from dashboard.mri_reader import load_medical_image, volume_slice_to_pil
from enhancement.pipeline import EnhancementAblationManager, to_display_rgb
from explainability.gradcam_generator import BrainTumorGradCAM
from reports.pdf_report_generator import COUNSELING_DB, generate_clinical_report_bytes
from segmentation.unet_model import UNet
from utils.device_config import get_system_execution_profile

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI — MRI Brain Tumor Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal clinical theme ────────────────────────────────────────────────────
# System font stack (no webfont download → faster first paint, works offline).
# One accent (clinical teal); color is used sparingly and never as the only
# signal for a diagnosis.
st.markdown(
    """
<style>
:root{
  --bg:#F5F7FA; --card:#FFFFFF; --line:#E2E8F2;
  --ink:#17233B; --muted:#5A6B84; --faint:#8B98AC;
  --teal:#0E7C8C; --teal-dk:#0A5E6B; --teal-bg:#E7F4F6;
  --good:#1E7F4F; --good-bg:#EAF6EF;
  --bad:#B23A2A;  --bad-bg:#FBEEE9;
  --radius:14px;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"]{ background:var(--bg) !important; }
[data-testid="stHeader"]{ border-bottom:1px solid var(--line) !important; }
section[data-testid="stSidebar"]{ background:var(--card) !important; border-right:1px solid var(--line) !important; }
section[data-testid="stSidebar"] .block-container{ padding-top:1.2rem; }

/* top bar */
.topbar{ display:flex; justify-content:space-between; align-items:center;
  background:var(--card); border:1px solid var(--line); border-radius:var(--radius);
  padding:12px 20px; margin-bottom:18px; }
.brand{ font-size:17px; color:var(--ink); font-weight:700; display:flex; align-items:center; gap:10px; }
.ai-badge{ font-size:10.5px; font-weight:700; letter-spacing:.08em; color:var(--teal-dk);
  background:var(--teal-bg); border:1px solid #BFE3E8; padding:3px 10px; border-radius:20px; }
.topnote{ font-size:12px; color:var(--muted); }

/* hero / empty state */
.hero{ text-align:center; background:var(--card); border:1px solid var(--line);
  border-radius:20px; padding:56px 32px 40px; margin:8px 0 18px; }
.hero-icon{ font-size:52px; display:inline-block; animation:pulseSoft 3.2s ease-in-out infinite; }
.hero h1{ font-size:24px; color:var(--ink); margin:14px 0 6px; font-weight:700; }
.hero p{ font-size:14.5px; color:var(--muted); max-width:520px; margin:0 auto; line-height:1.65; }
.steps{ display:flex; gap:12px; justify-content:center; margin-top:26px; flex-wrap:wrap; }
.step{ background:var(--bg); border:1px solid var(--line); border-radius:12px;
  padding:14px 18px; min-width:150px; max-width:210px; text-align:left; }
.step .n{ display:inline-flex; width:24px; height:24px; border-radius:50%;
  background:var(--teal); color:#fff; font-size:12.5px; font-weight:700;
  align-items:center; justify-content:center; margin-bottom:8px; }
.step .t{ font-size:13.5px; font-weight:600; color:var(--ink); }
.step .d{ font-size:12px; color:var(--muted); margin-top:2px; line-height:1.5; }
.chips{ display:flex; gap:8px; justify-content:center; flex-wrap:wrap; margin-top:20px; }
.chip{ font-size:12px; color:var(--teal-dk); background:var(--teal-bg);
  border:1px solid #CBE7EB; padding:4px 12px; border-radius:20px; }

/* diagnosis card */
.dx{ background:var(--card); border:1px solid var(--line); border-left:5px solid var(--teal);
  border-radius:var(--radius); padding:22px 26px; margin:4px 0 18px; }
.dx-ok{ border-left-color:var(--good); }
.dx-found{ border-left-color:var(--bad); }
.dx-kicker{ font-size:11px; font-weight:700; letter-spacing:.07em; text-transform:uppercase;
  color:var(--faint); margin-bottom:6px; }
.dx-name{ font-size:26px; font-weight:700; color:var(--ink); line-height:1.25; }
.dx-clin{ font-size:13px; color:var(--muted); margin-top:2px; }
.dx-conf{ display:inline-block; margin-top:10px; font-size:13px; font-weight:700; color:var(--ink);
  background:var(--bg); border:1px solid var(--line); border-radius:8px; padding:5px 12px; }
.dx-plain{ font-size:13.5px; color:var(--muted); margin:10px 0 0; line-height:1.6; max-width:640px; }
.dx-note{ margin-top:12px; font-size:12px; color:var(--faint); }
.dx-fusion{ margin-top:10px; font-size:12.5px; color:#8a5a00; background:#fef6e0;
  border:1px solid #f0dfae; border-radius:8px; padding:8px 12px; }

/* confidence bars */
.cf-row{ margin:10px 0; }
.cf-top{ display:flex; justify-content:space-between; font-size:13px; color:var(--ink); margin-bottom:5px; }
.cf-pct{ font-weight:700; font-variant-numeric:tabular-nums; }
.cf-track{ height:8px; background:#EDF1F7; border-radius:6px; overflow:hidden; }
.cf-fill{ height:100%; background:var(--teal); border-radius:6px; transition:width .6s ease; }
.cf-row:first-child .cf-fill{ background:var(--teal-dk); }

/* section titles */
.sec-t{ font-size:14px; font-weight:700; color:var(--ink); margin:22px 0 4px; }
.sec-d{ font-size:12.5px; color:var(--muted); margin-bottom:6px; }

/* metric cards */
[data-testid="stMetric"]{ background:var(--card); border:1px solid var(--line);
  border-radius:12px; padding:12px 16px; }

/* sidebar step labels */
.sb-step{ font-size:12px; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
  color:var(--teal-dk); margin:14px 0 6px; }
.dot{ display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:6px; vertical-align:1px; }
.status-line{ font-size:12.5px; color:var(--muted); margin:3px 0; }

/* animations — subtle, calm, medical */
@keyframes fadeUp{ from{ opacity:0; transform:translateY(10px);} to{ opacity:1; transform:none;} }
@keyframes pulseSoft{ 0%,100%{ transform:scale(1); opacity:.9;} 50%{ transform:scale(1.07); opacity:1;} }
.anim{ animation:fadeUp .45s ease both; }
.d1{ animation-delay:.06s; } .d2{ animation-delay:.14s; } .d3{ animation-delay:.22s; }
@media (prefers-reduced-motion: reduce){ *{ animation:none !important; transition:none !important; } }

/* small screens */
@media (max-width:640px){
  .topbar{ flex-direction:column; gap:6px; align-items:flex-start; }
  .hero{ padding:36px 20px 28px; }
  .dx-name{ font-size:22px; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Helpers ───────────────────────────────────────────────────────────────────
CLASSES = [
    "Intra-axial Glial Neoplasm",
    "Extra-axial Dural Lesion",
    "Sella Turcica Pituitary Adenoma",
    "No Tumor",
]

# Plain-language mapping: clinical label → (patient-friendly name, one-line meaning)
PLAIN_INFO = {
    "Intra-axial Glial Neoplasm": (
        "Glioma",
        "A tumor arising from glial cells inside the brain tissue. "
        "The exact grade can only be confirmed by biopsy.",
    ),
    "Extra-axial Dural Lesion": (
        "Meningioma",
        "A tumor of the membrane surrounding the brain. Usually benign — "
        "the grade needs histology to confirm.",
    ),
    "Sella Turcica Pituitary Adenoma": (
        "Pituitary adenoma",
        "A usually benign tumor of the pituitary gland. Hormone blood tests "
        "are typically needed alongside imaging.",
    ),
    "No Tumor": (
        "No tumor detected",
        "No focal tumor was found on this scan. Please discuss any ongoing "
        "symptoms with your doctor.",
    ),
}


def calculate_biomarkers(mask, raw_img, enh_img):
    pixels = int(np.sum(mask > 0))
    centroid = bbox = None
    perimeter = 0.0
    if pixels > 0:
        # .copy(): findContours mutates its input on some OpenCV builds;
        # never let contour extraction corrupt the caller's mask.
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            bbox = (x, y, x + w, y + h)
            perimeter = cv2.arcLength(c, closed=True)
            M = cv2.moments(c)
            if M["m00"] != 0:
                centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
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


def conf_bar_html(label: str, pct: float) -> str:
    return f"""
    <div class="cf-row" role="img" aria-label="{label}: {pct:.1f} percent">
      <div class="cf-top"><span>{label}</span><span class="cf-pct">{pct:.1f}%</span></div>
      <div class="cf-track"><div class="cf-fill" style="width:{max(pct, 2):.1f}%"></div></div>
    </div>"""


# ── Sidebar: 3 numbered steps, nothing else ───────────────────────────────────
st.sidebar.markdown("## 🧠 NeuroScan AI")
st.sidebar.caption("AI-assisted MRI brain tumor analysis")

st.sidebar.markdown('<div class="sb-step">1 · Scan</div>', unsafe_allow_html=True)
upload = st.sidebar.file_uploader(
    "MRI scan",
    type=["png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp", "dcm", "nii", "gz"],
    help="Image slice (PNG/JPG/BMP/TIFF/WebP) or a full volume: DICOM (.dcm), NIfTI (.nii/.nii.gz).",
)

# Parse once per file; 3-D volumes get a slice picker. Parse errors are
# shown here in the sidebar (human-readable, from mri_reader).
scan = None
slice_idx = 0
if upload is not None:
    file_key = (getattr(upload, "file_id", None), upload.name, upload.size)
    if st.session_state.get("scan_key") != file_key:
        try:
            st.session_state["scan"] = load_medical_image(upload.name, upload.getvalue())
        except ValueError as e:
            st.sidebar.error(str(e))
            st.session_state["scan"] = None
        st.session_state["scan_key"] = file_key
    scan = st.session_state.get("scan")
    if scan is not None and scan["kind"] == "volume":
        n = int(scan["meta"]["slice_count"])
        slice_idx = st.sidebar.slider(
            "Slice", 0, n - 1, n // 2, help="Pick the axial slice to analyze."
        )
        st.sidebar.caption(f"Volume: {scan['meta']['format']} · {n} slices")
    elif scan is not None:
        st.sidebar.caption(f"Image: {scan['meta'].get('notes', scan['meta']['format'])}")

st.sidebar.markdown('<div class="sb-step">2 · Patient</div>', unsafe_allow_html=True)
patient_id = st.sidebar.text_input("Patient ID", placeholder="e.g. PID-1024")
scan_date = st.sidebar.date_input("Scan date", value=datetime.today())
with st.sidebar.expander("Report details"):
    sequence = st.selectbox("MRI sequence", ["T1-Weighted CE", "T2-Weighted", "FLAIR", "DWI"])
    institution = st.text_input("Institution", placeholder="Hospital / clinic name")

st.sidebar.markdown('<div class="sb-step">3 · Analysis</div>', unsafe_allow_html=True)
model_choice = st.sidebar.selectbox(
    "AI pipeline",
    ["Exp 2: Enhanced", "Exp 1: Baseline", "Exp 3: Seg-Guided"],
    index=0,
    help="Which trained model analyzes the scan. Enhanced uses the WPT→LMMSE→CLAHE pipeline.",
)
model_map = {
    "Exp 1: Baseline": "classification/best_efficientnet_exp1_baseline.pth",
    "Exp 2: Enhanced": "classification/best_efficientnet_exp2_enhanced.pth",
    "Exp 3: Seg-Guided": "classification/best_efficientnet_exp3_seg_guided.pth",
}

ckpt_dir = PROJECT_ROOT / "checkpoints"
seg_path = ckpt_dir / "unet/best_unet_enhanced.pth"
class_path = ckpt_dir / model_map[model_choice]


def _dot(ok: bool) -> str:
    return f'<span class="dot" style="background:{"#1E7F4F" if ok else "#B23A2A"}"></span>'


st.sidebar.markdown(
    f'<div class="status-line">{_dot(seg_path.exists())} U-Net segmentation</div>'
    f'<div class="status-line">{_dot(class_path.exists())} Classifier · {model_choice}</div>',
    unsafe_allow_html=True,
)
if not class_path.exists() or not seg_path.exists():
    st.sidebar.warning(
        "Model weights missing — train on Colab (notebooks 2 & 3) to enable analysis."
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    "Decision support only. Not a standalone diagnosis — "
    "a qualified radiologist must review every finding."
)


# ── Model loading (cached per pipeline) ───────────────────────────────────────
@st.cache_resource
def load_models(active_pipeline: str):
    # active_pipeline is the cache KEY: switching pipelines reloads the
    # right classifier instead of silently reusing the previous one.
    profile = get_system_execution_profile()
    device = profile["device"]
    unet = UNet(n_channels=3, n_classes=1).to(device)
    classifier = BrainTumorClassifier(num_classes=4, pretrained=False).to(device)
    seg_ckpt = PROJECT_ROOT / "checkpoints/unet/best_unet_enhanced.pth"
    cls_ckpt = PROJECT_ROOT / f"checkpoints/{model_map[active_pipeline]}"
    if seg_ckpt.exists():
        unet.load_state_dict(torch.load(str(seg_ckpt), map_location=device, weights_only=True))
    if cls_ckpt.exists():
        classifier.load_state_dict(
            torch.load(str(cls_ckpt), map_location=device, weights_only=True)
        )
    unet.eval()
    classifier.eval()
    return unet, classifier, device


# ── Top bar ───────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="topbar anim">
  <div class="brand">🧠 NeuroScan AI <span class="ai-badge">AI-ASSISTED</span></div>
  <div class="topnote">Radiology decision support · For specialist review</div>
</div>
""",
    unsafe_allow_html=True,
)


# ── Inference (staged, cached in session so UI interactions don't recompute) ───
def _run_pipeline(pil_img: Image.Image, pipeline: str) -> dict:
    """Full validated pipeline. Returns plain arrays/scalars for the UI."""
    unet, classifier, device = load_models(pipeline)
    enhancer = EnhancementAblationManager()
    gradcam = BrainTumorGradCAM(classifier, use_cuda=(device.type == "cuda"))

    raw_arr = np.array(pil_img).astype(np.float32) / 255.0
    raw_arr = cv2.resize(raw_arr, (256, 256))

    with st.status("Analyzing scan…", expanded=True) as status:
        st.write("✨ Enhancing image quality")
        enhanced = enhancer.process(raw_arr)
        enh_rgb = to_display_rgb(enhanced)
        raw_rgb = to_display_rgb(raw_arr)

        status.update(label="Mapping tumor region…")
        st.write("🧠 Segmenting with U-Net")
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        inp = (enh_rgb / 255.0 - mean) / std
        tensor = torch.from_numpy(inp.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = unet(tensor)
            mask_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        binary_mask = (mask_prob > 0.5).astype(np.uint8) * 255

        area, centroid, bbox, cnr, perim = calculate_biomarkers(binary_mask, raw_rgb, enh_rgb)

        overlay_seg = enh_rgb.copy()
        if area > 0:
            overlay_seg[binary_mask > 0] = (
                (overlay_seg[binary_mask > 0] * 0.55 + np.array([255, 40, 80]) * 0.45)
                .clip(0, 255)
                .astype(np.uint8)
            )
            contours, _ = cv2.findContours(
                binary_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay_seg, contours, -1, (255, 255, 0), 2)

        status.update(label="Classifying…")
        st.write("🔬 Classifying with EfficientNetB2")
        # The classifier input MUST match the training distribution of the
        # selected experiment (Exp1: raw, Exp3: soft seg-guided, else enhanced).
        if pipeline.startswith("Exp 1"):
            cls_rgb = raw_rgb
            cls_inp = (cls_rgb / 255.0 - mean) / std
            cls_tensor = (
                torch.from_numpy(cls_inp.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            )
        elif pipeline.startswith("Exp 3"):
            # PARITY WITH TRAINING (classification/run_experiments.py ::
            # apply_exp3_guidance, via masking_utils.apply_exp3_guidance_numpy):
            # guidance in NORMALIZED space — not uint8 space — with the same
            # 50px empty-mask guard, so a healthy scan is never darkened.
            cls_tensor = torch.from_numpy(
                apply_exp3_guidance_numpy(enh_rgb, mask_prob, mean, std)
            ).to(device)
        else:
            cls_rgb = enh_rgb
            cls_inp = (cls_rgb / 255.0 - mean) / std
            cls_tensor = (
                torch.from_numpy(cls_inp.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            )
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            probs = torch.softmax(classifier(cls_tensor), dim=1).squeeze().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]

        # Model disagreement check (no output is altered): the segmentation
        # and classification heads were trained independently, so when they
        # disagree the only honest action is to ask for human review.
        model_disagreement = (area > 100 and pred_class == "No Tumor") or (
            area <= 100 and pred_class != "No Tumor" and float(probs[pred_idx]) >= 0.80
        )

        status.update(label="Explaining…")
        st.write("🎯 Generating visual explanation")
        heatmap = gradcam.generate_heatmap(cls_tensor, target_category=pred_idx)
        heatmap = cv2.resize(heatmap, (256, 256))
        heatmap_c = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_c = cv2.cvtColor(heatmap_c, cv2.COLOR_BGR2RGB)
        gradcam_overlay = cv2.addWeighted(enh_rgb, 0.55, heatmap_c, 0.45, 0)

        status.update(label="Analysis complete", state="complete", expanded=False)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "raw_rgb": raw_rgb,
        "enh_rgb": enh_rgb,
        "overlay_seg": overlay_seg,
        "gradcam_overlay": gradcam_overlay,
        "probs": probs,
        "pred_idx": pred_idx,
        "pred_class": pred_class,
        "model_disagreement": model_disagreement,
        "area": area,
        "centroid": centroid,
        "bbox": bbox,
        "cnr": cnr,
        "perim": perim,
    }


# ── Clinical guidance: 6.1 = this scan's findings, 6.2 = general info ──────────
# The split never changes, so the reader can tell at a glance what came from
# their image vs what is generic category information.
def render_guidance(pred_class: str, r: dict) -> None:
    info = COUNSELING_DB.get(pred_class, COUNSELING_DB["No Tumor"])
    plain_name, _ = PLAIN_INFO[pred_class]
    is_tumor = pred_class != "No Tumor"
    pred_conf = float(r["probs"][r["pred_idx"]] * 100)
    area, centroid, bbox, perim = r["area"], r["centroid"], r["bbox"], r["perim"]
    with st.expander(f"📋 Clinical guidance — {plain_name}", expanded=is_tumor):
        st.caption("6.1 is always this scan's own findings; 6.2 is always general information.")
        st.markdown("**6.1 · About this scan**")
        st.caption("Findings measured from this scan only.")
        st.markdown(f"- **AI finding:** {plain_name} ({pred_conf:.1f}% confidence)")
        if area > 0:
            st.markdown(f"- **Tumor area:** {area:,} px²")
            st.markdown(f"- **Tumor perimeter:** {perim:.1f} px")
            st.markdown(f"- **Location (centroid):** {centroid}")
            st.markdown(f"- **Bounding box:** {bbox}")
        else:
            st.markdown("- **Tumor area:** No focal lesion segmented")
        st.markdown("**6.2 · General information**")
        st.caption(
            "General information for this finding category — not personalised medical advice."
        )
        st.markdown("**About this finding**")
        st.write(info["pathological_nature"])
        st.markdown("**Precautions & red-flag symptoms**")
        for p in info["precautions"]:
            st.markdown(f"- {p}")
        st.markdown("**Recommended workup**")
        for s in info["next_steps"]:
            st.markdown(f"- {s}")
        st.markdown("**Questions for your doctor**")
        for i, item in enumerate(info["checklist"]):
            st.checkbox(item, key=f"guidance_{pred_class}_{i}")


# ── Main flow ─────────────────────────────────────────────────────────────────
if upload is None:
    st.markdown(
        """
    <div class="hero anim">
      <div class="hero-icon">🧠</div>
      <h1>Analyze an MRI scan</h1>
      <p>Upload a brain MRI slice and NeuroScan AI will enhance it, map any tumor
      region, classify the finding and explain its reasoning — in under a minute.</p>
      <div class="steps">
        <div class="step anim d1"><div class="n">1</div><div class="t">Upload</div>
          <div class="d">Add the MRI from the sidebar — image, DICOM or NIfTI.</div></div>
        <div class="step anim d2"><div class="n">2</div><div class="t">AI analysis</div>
          <div class="d">Enhancement → segmentation → classification → explanation.</div></div>
        <div class="step anim d3"><div class="n">3</div><div class="t">Review</div>
          <div class="d">Read the finding, check guidance, export the report.</div></div>
      </div>
      <div class="chips">
        <span class="chip">WPT enhancement</span>
        <span class="chip">U-Net segmentation</span>
        <span class="chip">EfficientNetB2</span>
        <span class="chip">Grad-CAM</span>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.info(
        "👈 Start by uploading a scan in the sidebar. Nothing is stored — analysis runs in memory.",
        icon="🔒",
    )
else:
    if not class_path.exists() or not seg_path.exists():
        st.warning(
            "**Model weights missing.** Train the models on Google Colab "
            "(notebooks 2 & 3) to enable analysis.",
            icon="⚠️",
        )
        st.stop()

    if scan is None:
        st.stop()  # unreadable file: the reason is already shown in the sidebar.
    # Cache inference in session state: ticking checkboxes / switching tabs
    # must not recompute the neural pipeline.
    inf_key = (getattr(upload, "file_id", None), upload.size, model_choice, slice_idx)
    if st.session_state.get("inf_key") != inf_key:
        if scan["kind"] == "volume":
            pil_img = volume_slice_to_pil(scan["volume"], slice_idx)
        else:
            pil_img = scan["image"]
        st.session_state["inf_result"] = _run_pipeline(pil_img, model_choice)
        st.session_state["inf_key"] = inf_key
    r = st.session_state["inf_result"]

    probs, pred_idx, pred_class = r["probs"], r["pred_idx"], r["pred_class"]
    pred_conf = float(probs[pred_idx] * 100)
    is_tumor = pred_class != "No Tumor"
    plain_name, plain_desc = PLAIN_INFO[pred_class]

    # ── Diagnosis card ──
    dx_cls = "dx-found" if is_tumor else "dx-ok"
    fusion_html = (
        (
            '<div class="dx-fusion">⚠️ The segmentation and classification models disagree '
            "on this scan — please review manually before any decision.</div>"
        )
        if r["model_disagreement"]
        else ""
    )
    st.markdown(
        f"""
    <div class="dx {dx_cls} anim">
      <div class="dx-kicker">AI finding · requires radiologist review</div>
      <div class="dx-name">{plain_name}</div>
      <div class="dx-clin">{pred_class}</div>
      <div class="dx-conf">Confidence {pred_conf:.1f}%</div>
      <p class="dx-plain">{plain_desc}</p>
      {fusion_html}
      <div class="dx-note">This is decision support, not a diagnosis. Grade and treatment
      can only be determined by a qualified specialist with the full clinical picture.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Scan views (tabs keep it clean; captions teach what each view means) ──
    t1, t2, t3, t4 = st.tabs(["Original", "Enhanced", "Tumor map", "AI attention"])
    with t1:
        st.image(r["raw_rgb"], use_container_width=True)
        st.caption("The scan as uploaded.")
    with t2:
        st.image(r["enh_rgb"], use_container_width=True)
        st.caption("After WPT → LMMSE → CLAHE contrast enhancement.")
    with t3:
        st.image(r["overlay_seg"], use_container_width=True)
        st.caption(
            "U-Net's predicted tumor region — outlined in yellow."
            if r["area"] > 0
            else "No tumor region segmented."
        )
    with t4:
        st.image(r["gradcam_overlay"], use_container_width=True)
        st.caption("Grad-CAM heatmap — the regions the classifier focused on.")

    # ── Key figures ──
    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence", f"{pred_conf:.1f}%")
    m2.metric("Tumor area", f"{r['area']:,} px²" if r["area"] > 0 else "—")
    m3.metric("Contrast gain", f"{r['cnr']:+.1f}%")

    # ── Confidence breakdown ──
    st.markdown(
        '<div class="sec-t">Confidence breakdown</div>'
        '<div class="sec-d">How strongly the model considers each possibility.</div>',
        unsafe_allow_html=True,
    )
    for cls, prob in sorted(zip(CLASSES, probs, strict=True), key=lambda x: x[1], reverse=True):
        st.markdown(conf_bar_html(PLAIN_INFO[cls][0], prob * 100), unsafe_allow_html=True)

    # ── Clinical guidance ──
    st.markdown('<div class="sec-t">Guidance</div>', unsafe_allow_html=True)
    render_guidance(pred_class, r)

    # ── Report export ──
    st.markdown("---")
    if st.button("⬇ Download assessment report (PDF)", type="primary", use_container_width=True):
        try:
            report_bytes, mime, ext = generate_clinical_report_bytes(
                patient_id=patient_id or "—",
                scan_date=str(scan_date),
                sequence=sequence,
                institution=institution or "—",
                pred_class=pred_class,
                pred_conf=pred_conf,
                probs=list(probs),
                area=r["area"],
                perim=r["perim"],
                centroid=r["centroid"],
                cnr=r["cnr"],
                raw_img=r["raw_rgb"],
                enh_img=r["enh_rgb"],
                seg_img=r["overlay_seg"],
                gradcam_img=r["gradcam_overlay"],
                model_choice=model_choice,
                bbox=r["bbox"],
            )
            label = (
                "Download PDF report"
                if ext == "pdf"
                else "Download HTML report — open in a browser, then File → Print → Save as PDF"
            )
            st.download_button(
                label=label,
                data=report_bytes,
                file_name=f"neuroscan_{patient_id or 'scan'}_{datetime.now().strftime('%Y%m%d_%H%M')}.{ext}",
                mime=mime,
                use_container_width=True,
            )
            if ext == "html":
                st.info(
                    "HTML report downloaded. Open it in any browser and use "
                    "File → Print → Save as PDF for a full A4 report."
                )
        except Exception as e:
            st.error(f"Report generation error: {e}")

    st.caption("AI-assisted assessment. All findings require review by a qualified radiologist.")
