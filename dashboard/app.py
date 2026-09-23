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

# ── CSS: Clinical Light Theme ──────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Figtree:wght@400;600;700&family=Noto+Sans:ital,wght@0,400;0,500;0,600;1,400&family=Noto+Sans+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
/* ─── Design tokens: Taste-Skill Premium ────────── */
:root {
    /* surfaces */
    --bg-page    : #F4F4F5;
    --bg-card    : #FFFFFF;
    --bg-sidebar : #FAFAFA;
    --border     : #E4E4E7;
    --border-mid : #D4D4D8;
    
    /* text scale */
    --text-hi    : #09090B;
    --text-body  : #3F3F46;
    --text-muted : #71717A;
    --text-faint : #A1A1AA;
    
    /* clinical palette — focused */
    --primary    : #0891B2;
    --primary-dk : #0E7490;
    --primary-lt : #ECFEFF;
    --danger     : #E11D48;
    --success    : #059669;
    
    /* spacing */
    --sp-2: 8px;
    --sp-4: 16px;
    --sp-6: 24px;
    --sp-8: 32px;
    
    /* elevations */
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.03);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.025);
    
    /* radii */
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 16px;
}

/* ─── Reset & base ─────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: var(--bg-page) !important;
    color: var(--text-body);
    font-family: 'Noto Sans', system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
}
[data-testid="stAppViewContainer"] { background: var(--bg-page); }
[data-testid="stHeader"] { background: var(--bg-page) !important; border-bottom: 1px solid var(--border) !important; }

/* ─── Sidebar ───────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: var(--text-body) !important;
}
/* inputs, date pickers, selects — premium treatment */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] [data-baseweb="input"] > div,
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="base-input"],
section[data-testid="stSidebar"] [data-testid="stDateInput"] input,
section[data-testid="stSidebar"] [data-testid="stDateInput"] > div > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-hi) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stSidebar"] input:focus,
section[data-testid="stSidebar"] [data-baseweb="input"] > div:focus-within,
section[data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 1px var(--primary) !important;
}
/* file uploader drop zone */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] > div,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] > div {
    background: var(--bg-card) !important;
    border: 1.5px dashed var(--border-mid) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-muted) !important;
    transition: border-color 0.2s ease, background 0.2s ease !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--primary) !important;
    background: var(--primary-lt) !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
    color: var(--text-muted) !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background: var(--bg-page) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-hi) !important;
    box-shadow: var(--shadow-sm) !important;
    border-radius: var(--radius-sm) !important;
}
/* select dropdown popover (opens outside sidebar) */
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] ul {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-lg) !important;
}
[data-baseweb="popover"] [role="option"] {
    color: var(--text-body) !important;
    font-family: 'Noto Sans', sans-serif !important;
    font-size: 14px !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
    background: var(--primary-lt) !important;
    color: var(--primary-dk) !important;
}
/* date picker calendar */
[data-baseweb="calendar"],
[data-baseweb="datepicker"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-lg) !important;
}
[data-baseweb="calendar"] * {
    color: var(--text-body) !important;
    font-family: 'Noto Sans', sans-serif !important;
}
[data-baseweb="calendar"] [aria-selected="true"] > div {
    background: var(--primary) !important;
    color: #fff !important;
}
section[data-testid="stSidebar"] hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ─── Expander: enforce light theme ────────────── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stExpander"] details {
    background: var(--bg-card) !important;
}
[data-testid="stExpander"] details > summary {
    background: var(--bg-card) !important;
    color: var(--text-hi) !important;
    font-size: 15px;
    font-weight: 600;
    font-family: 'Figtree', sans-serif;
    padding: 12px 16px !important;
}
[data-testid="stExpander"] details[open] > summary {
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stExpander"] details > summary:hover {
    background: var(--bg-sidebar) !important;
}
[data-testid="stExpander"] details > div {
    background: var(--bg-card) !important;
}

/* ─── Primary CTA → clinical teal (flat, no glow) ─── */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: var(--primary) !important;
    border: 1px solid var(--primary-dk) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.01em !important;
    box-shadow: var(--shadow-sm) !important;
    border-radius: var(--radius-sm) !important;
    transition: all 0.2s ease !important;
    font-family: 'Noto Sans', sans-serif !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    background: var(--primary-dk) !important;
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px) !important;
}

/* ─── Download button ─────────────────────────── */
.stDownloadButton > button {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--primary) !important;
    font-weight: 600 !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    background: var(--bg-sidebar) !important;
    border-color: var(--border-mid) !important;
}

/* ─── Image grid: uniform 1:1 aspect ratio ──────── */
[data-testid="stImage"] {
    border-radius: 0 0 var(--radius-md) var(--radius-md);
    overflow: hidden;
    border: 1px solid var(--border);
    border-top: none;
    display: block;
    background: var(--bg-page);
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.3s ease;
}
[data-testid="stImage"]:hover {
    box-shadow: var(--shadow-md);
}
[data-testid="stImage"] > img {
    width: 100% !important;
    height: 100% !important;
    aspect-ratio: 1 / 1;
    object-fit: cover !important;
    border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
    display: block !important;
}

/* ─── Streamlit native element resets ──────────── */
[data-testid="stDivider"] hr, hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-mid); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-faint); }

/* ─── Letterhead header ────────────────────────── */
.rpt-header {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    overflow: hidden;
    margin-bottom: var(--sp-6);
    box-shadow: var(--shadow-sm);
}
.rpt-header-top {
    padding: var(--sp-6) var(--sp-8);
    border-bottom: 1px solid var(--border);
    background: var(--bg-card);
}
.rpt-title {
    font-family: 'Figtree', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: var(--text-hi);
    letter-spacing: -0.02em;
}
.rpt-subtitle {
    font-size: 14px;
    color: var(--text-muted);
    margin-top: 4px;
    font-family: 'Noto Sans', sans-serif;
}
.rpt-meta-row {
    padding: var(--sp-4) var(--sp-8);
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: var(--sp-4) var(--sp-6);
    background: var(--bg-sidebar);
}
.rpt-meta-cell { display: flex; flex-direction: column; gap: 4px; }
.rpt-meta-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-faint);
    font-family: 'Noto Sans', sans-serif;
}
.rpt-meta-value {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-hi);
    font-family: 'Noto Sans', sans-serif;
}

/* ─── Pipeline step cards ───────────────────────── */
.pipeline-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-bottom: 1px solid var(--border-mid);
    border-radius: var(--radius-md) var(--radius-md) 0 0;
    padding: 12px 16px;
    margin-bottom: 0;
    min-height: 48px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--sp-2);
}
.pipeline-card .step-label {
    font-size: 12px;
    font-weight: 700;
    color: var(--text-hi);
    letter-spacing: 0.02em;
    font-family: 'Figtree', sans-serif;
}
.step-badge {
    font-size: 11px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 12px;
    letter-spacing: 0.05em;
    white-space: nowrap;
    flex-shrink: 0;
    text-transform: uppercase;
    font-family: 'Noto Sans', sans-serif;
}
.badge-tumor { color: #9F1239; background: #FFE4E6; border: 1px solid #FECDD3; }
.badge-clear { color: #065F46; background: #D1FAE5; border: 1px solid #A7F3D0; }
.badge-cnr   { color: #075985; background: #E0F2FE; border: 1px solid #BAE6FD; }

/* ─── Diagnosis panel (The Focal Point) ─────────── */
.diag-panel {
    border-radius: var(--radius-lg);
    padding: var(--sp-6) var(--sp-8);
    border: 1px solid var(--border);
    border-left: 6px solid var(--primary);
    background: var(--bg-card);
    box-shadow: var(--shadow-lg);
    height: 100%;
    position: relative;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.diag-panel:hover {
    transform: translateY(-2px);
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
}
.diag-panel.diag-positive { border-left-color: var(--danger); }
.diag-panel.diag-negative { border-left-color: var(--success); }
.diag-label {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: var(--sp-2);
    font-family: 'Noto Sans', sans-serif;
}
.diag-value {
    font-family: 'Figtree', sans-serif;
    font-size: clamp(24px, 3vw, 32px);
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -0.02em;
    word-break: break-word;
    overflow-wrap: anywhere;
    max-width: 100%;
    margin-bottom: var(--sp-4);
    color: var(--text-hi);
}
.diag-conf {
    font-size: 15px;
    color: var(--text-body);
    font-family: 'Noto Sans', sans-serif;
    padding: 8px 12px;
    background: var(--bg-sidebar);
    border-radius: var(--radius-sm);
    display: inline-block;
    border: 1px solid var(--border);
}

/* ─── Softmax confidence bars ───────────────────── */
.prob-section-label {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: var(--sp-4);
    font-family: 'Figtree', sans-serif;
}
.prob-row { margin-bottom: var(--sp-4); }
.prob-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 6px;
}
.prob-label {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-hi);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 76%;
    font-family: 'Noto Sans', sans-serif;
}
.prob-pct {
    font-size: 13px;
    font-weight: 600;
    font-family: 'Noto Sans Mono', monospace;
    color: var(--text-hi);
    flex-shrink: 0;
    text-align: right;
    min-width: 44px;
}
.prob-track {
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}

/* ─── Morphometrics data table ──────────────────── */
.morph-table-wrap {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-sm);
    overflow: hidden;
}
.morph-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    font-family: 'Noto Sans', sans-serif;
}
.morph-table th {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    text-align: left;
    background: var(--bg-sidebar);
}
.morph-table td {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    color: var(--text-body);
    vertical-align: middle;
}
.morph-table tr:hover td {
    background: var(--bg-page);
}
.morph-table tr:last-child td { border-bottom: none; }
.morph-table td:last-child {
    font-family: 'Noto Sans Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-hi);
    text-align: right;
}

/* ─── Sidebar section headers ───────────────────── */
.sb-head {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    font-weight: 700;
    color: var(--text-hi);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 4px 0 8px 0;
    margin-bottom: 8px;
    border-bottom: 1px solid var(--border);
    font-family: 'Figtree', sans-serif;
}
.sb-head svg { opacity: 0.7; flex-shrink: 0; }

/* ─── Document section divider (counseling) ─────── */
.c-doc-divider {
    display: flex;
    align-items: center;
    gap: var(--sp-4);
    margin: var(--sp-6) 0 var(--sp-4) 0;
}
.c-doc-divider hr {
    flex: 1;
    border: none;
    border-top: 1px dashed var(--border-mid);
    margin: 0;
}
.c-doc-divider span {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    white-space: nowrap;
    flex-shrink: 0;
    font-family: 'Noto Sans', sans-serif;
}

/* ─── Clinical counseling section ───────────────── */
.counseling-section {
    background: var(--bg-card);
    border-radius: var(--radius-md);
    padding: var(--sp-6) var(--sp-8);
}
.counseling-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--sp-4);
    padding-bottom: var(--sp-4);
    margin-bottom: var(--sp-4);
    border-bottom: 1px solid var(--border);
}
.counseling-title {
    font-family: 'Figtree', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: var(--text-hi);
    letter-spacing: -0.01em;
}
.counseling-subtitle {
    font-size: 13px;
    color: var(--text-muted);
    margin-top: 4px;
    font-family: 'Noto Sans', sans-serif;
}
.counseling-badge {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 12px;
    white-space: nowrap;
    flex-shrink: 0;
    font-family: 'Noto Sans', sans-serif;
}
.c-body-text {
    font-size: 14px;
    color: var(--text-body);
    margin-bottom: var(--sp-4);
    line-height: 1.7;
    font-family: 'Noto Sans', sans-serif;
}
.c-bullet-item {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    padding: 6px 0 6px 8px;
    font-size: 14px;
    color: var(--text-body);
    line-height: 1.6;
    font-family: 'Noto Sans', sans-serif;
}
.c-checklist-item {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    padding: 6px 0 6px 8px;
    font-size: 14px;
    color: var(--text-body);
    line-height: 1.6;
    font-family: 'Noto Sans', sans-serif;
}
.c-checkbox {
    width: 16px;
    height: 16px;
    border: 2px solid var(--border-mid);
    border-radius: 4px;
    flex-shrink: 0;
    margin-top: 3px;
}

/* ─── Empty State ───────────────────────────────── */
@keyframes brain-pulse {
    0%, 100% { opacity: 0.6; transform: scale(1); }
    50%       { opacity: 1; transform: scale(1.05); }
}
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 80px 40px;
    text-align: center;
    background: linear-gradient(180deg, var(--bg-card) 0%, var(--bg-page) 100%);
    border-radius: var(--radius-lg);
    border: 2px dashed var(--border-mid);
    margin: var(--sp-8) 0;
    box-shadow: var(--shadow-sm);
    transition: border-color 0.3s ease;
}
.empty-state:hover {
    border-color: var(--text-faint);
}
.empty-brain-icon {
    display: block;
    margin-bottom: 24px;
    animation: brain-pulse 4s ease-in-out infinite;
    transform-origin: center;
    color: var(--primary);
}
.empty-title {
    font-family: 'Figtree', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: var(--text-hi);
    margin-bottom: 8px;
}
.empty-sub {
    font-size: 15px;
    color: var(--text-muted);
    max-width: 480px;
    line-height: 1.6;
    font-family: 'Noto Sans', sans-serif;
}
.empty-cap-list {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 24px;
}
.empty-cap {
    font-size: 13px;
    color: var(--text-hi);
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 12px;
    font-family: 'Noto Sans', sans-serif;
    box-shadow: var(--shadow-sm);
}

/* ─── Streamlit alert boxes → light theme ───────── */
[data-testid="stAlert"] {
    background: #FFFBEB !important;
    border: 1px solid #FDE68A !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-sm) !important;
}
/* Info variant */
[data-testid="stAlert"][data-type="info"],
.stAlert[data-type="info"],
[data-testid="stAlert"].element-container {
    background: var(--primary-lt) !important;
    border-color: #BAE6FD !important;
}
[data-testid="stAlert"] p,
[data-testid="stAlert"] li,
[data-testid="stAlert"] div {
    color: var(--text-body) !important;
    font-family: 'Noto Sans', sans-serif !important;
    font-size: 14px !important;
}
section[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: #FFFBEB !important;
    border-color: #FDE68A !important;
}

/* ─── Spinner text ──────────────────────────────── */
[data-testid="stSpinner"] p,
[data-testid="stSpinner"] span {
    color: var(--text-muted) !important;
    font-family: 'Noto Sans', sans-serif !important;
    font-size: 14px !important;
}

/* ─── Counseling bullet dot ─────────────────────── */
.c-bullet-dot {
    color: var(--primary);
    font-size: 20px;
    line-height: 1.4;
    flex-shrink: 0;
    margin-top: -3px;
    display: inline-block;
    width: 16px;
    text-align: center;
}

/* ─── Force main content area background ───────── */
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main .block-container {
    background: var(--bg-page) !important;
}
</style>
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

def prob_bar_html(label: str, value: float, rank: int = 0) -> str:
    """Render a confidence bar using a single teal opacity scale.
    rank=0 → highest confidence (solid teal); higher rank → lighter fill.
    Sorted by caller (descending confidence).
    """
    pct = value * 100
    # Teal opacity: rank 0 = 1.0, rank 1 = 0.72, rank 2 = 0.48, rank 3 = 0.28
    opacity = max(0.25, 1.0 - rank * 0.25)
    return f"""
    <div class="prob-row">
      <div class="prob-header">
        <span class="prob-label">{label}</span>
        <span class="prob-pct">{pct:.1f}%</span>
      </div>
      <div class="prob-track">
        <div class="prob-fill" style="width:{pct:.2f}%; background:rgba(8,145,178,{opacity:.2f});"></div>
      </div>
    </div>"""

# ── Sidebar SVG icon helpers ──────────────────────────────────────────────────
_SB_ICO = {
    "upload" : '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
    "patient": '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
    "ai"     : '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg>',
    "status" : '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "hw"     : '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
}

def _sbhead(icon_key: str, title: str) -> str:
    """Return a styled sidebar section header with an inline SVG icon."""
    return f'<div class="sb-head">{_SB_ICO[icon_key]}<span>{title}</span></div>'

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown(_sbhead("upload", "MRI Upload"), unsafe_allow_html=True)
upload = st.sidebar.file_uploader("Upload Scan (PNG / JPG)", type=['png', 'jpg', 'jpeg'])

st.sidebar.divider()
st.sidebar.markdown(_sbhead("patient", "Patient Metadata"), unsafe_allow_html=True)
patient_id  = st.sidebar.text_input("Patient ID", value="PID-90210", label_visibility="visible")
scan_date   = st.sidebar.date_input("Scan Date", value=datetime.today())
sequence    = st.sidebar.selectbox("MRI Sequence", [
    "T1-Weighted CE", "T2-Weighted", "FLAIR", "DWI"])
institution = st.sidebar.text_input("Institution", value="ITS Engineering College")

st.sidebar.divider()
st.sidebar.markdown(_sbhead("ai", "AI Configuration"), unsafe_allow_html=True)
model_choice = st.sidebar.selectbox("Active Pipeline", [
    "Exp 2: Enhanced (98.2%)",
    "Exp 1: Baseline (98.9%)",
    "Exp 3: Seg-Guided"], index=0)

model_map = {
    "Exp 1: Baseline (98.9%)"  : "classification/best_efficientnet_exp1_baseline.pth",
    "Exp 2: Enhanced (98.2%)"  : "classification/best_efficientnet_exp2_enhanced.pth",
    "Exp 3: Seg-Guided"        : "classification/best_efficientnet_exp3_seg_guided.pth",
}

st.sidebar.divider()
st.sidebar.markdown(_sbhead("status", "System Status"), unsafe_allow_html=True)
ckpt_dir   = PROJECT_ROOT / "checkpoints"
seg_path   = ckpt_dir / "unet/best_unet_enhanced.pth"
class_path = ckpt_dir / model_map[model_choice]

def status_html(label, ok):
    txt = "Online" if ok else "Missing"
    clr = "#16A34A" if ok else "#DC2626"
    dot = f'<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:{clr};margin-right:5px;vertical-align:middle"></span>'
    return (f'{dot}<span style="font-size:13px;color:{clr};font-weight:600;font-family:Noto Sans,sans-serif">{txt}</span>'
            f' <span style="font-size:13px;color:#64748B;font-family:Noto Sans,sans-serif">{label}</span>')

st.sidebar.markdown(status_html("U-Net Segmentation", seg_path.exists()), unsafe_allow_html=True)
st.sidebar.markdown(status_html(f"Classifier ({model_choice[:5]})", class_path.exists()), unsafe_allow_html=True)

if not class_path.exists() or not seg_path.exists():
    st.sidebar.warning("Model weights missing. Train on Colab to enable live inference.")

# ── Hardware badge ────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown(_sbhead("hw", "Inference Hardware"), unsafe_allow_html=True)
_p = _HW_PROFILE
if _p["has_cuda"]:
    _hw_label = f"GPU · {_p['gpu_name']}"
    _hw_sub   = f"{_p['vram_gb']:.1f} GB VRAM · AMP fp16"
    _hw_color = "#16A34A"
else:
    _hw_label = "CPU"
    _hw_sub   = f"{_p['total_ram_gb']:.1f} GB RAM · {_p['num_workers']} workers"
    _hw_color = "#0891B2"
_ort_badge = ("ORT ✓" if _ORT_AVAILABLE else "PyTorch")
st.sidebar.markdown(
    f'<span style="font-size:13px;color:{_hw_color};font-weight:600;font-family:Noto Sans,sans-serif">{_hw_label}</span><br>'
    f'<span style="font-size:12px;color:#64748B;font-family:Noto Sans,sans-serif">{_hw_sub} · {_ort_badge}</span>',
    unsafe_allow_html=True
)

# ── Letterhead Header ──────────────────────────────────────────────────────────
st.markdown(f"""
<div class="rpt-header">
  <div class="rpt-header-top">
    <div>
      <div class="rpt-title">NeuroScan AI &mdash; Radiology Decision Support</div>
      <div class="rpt-subtitle">MRI Neuro-Oncology Inference Platform &nbsp;&middot;&nbsp; {institution}</div>
    </div>
  </div>
  <div class="rpt-meta-row">
    <div class="rpt-meta-cell"><span class="rpt-meta-label">Patient ID</span><span class="rpt-meta-value">{patient_id}</span></div>
    <div class="rpt-meta-cell"><span class="rpt-meta-label">Scan Date</span><span class="rpt-meta-value">{scan_date}</span></div>
    <div class="rpt-meta-cell"><span class="rpt-meta-label">MRI Sequence</span><span class="rpt-meta-value">{sequence}</span></div>
    <div class="rpt-meta-cell"><span class="rpt-meta-label">Active Pipeline</span><span class="rpt-meta-value">{model_choice}</span></div>
    <div class="rpt-meta-cell"><span class="rpt-meta-label">Inference Engine</span><span class="rpt-meta-value">EfficientNetB2 + U-Net</span></div>
    <div class="rpt-meta-cell"><span class="rpt-meta-label">Generated</span><span class="rpt-meta-value">{datetime.now().strftime('%d %b %Y, %H:%M')}</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

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
# CLASS_COLORS removed — diagnosis state is conveyed by .diag-positive/.diag-negative
# CSS left-border classes only. Text uses --text-hi token throughout.

# ── Clinical counseling knowledge base ───────────────────────────────────────
# Full neuro-oncological counseling text keyed by predicted class.
# Imported into the dashboard expander and forwarded to the PDF generator.
from reports.pdf_report_generator import COUNSELING_DB, generate_pdf_report, generate_clinical_report_bytes

# ── Counseling HTML helpers ───────────────────────────────────────────────────
def _c_bullet(text: str, dot_class: str = "") -> str:
    return (
        f'<div class="c-bullet-item">'
        f'<span class="c-bullet-dot">&#8226;</span>'
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
    if is_tumor:
        badge_style = "color:#DC2626;background:#FEF2F2;border:1px solid #FECACA"
    else:
        badge_style = "color:#16A34A;background:#F0FDF4;border:1px solid #BBF7D0"

    with st.expander(
        f"Clinical Findings & Patient Counseling — {pred_class}",
        expanded=is_tumor,
    ):
        st.markdown(f"""
        <div class="counseling-section">
          <div class="counseling-header">
            <div>
              <div class="counseling-title">Clinical Findings &amp; Patient Counseling Guidelines</div>
              <div class="counseling-subtitle">NeuroScan AI &nbsp;&middot;&nbsp; Neuro-Oncology Decision Support</div>
            </div>
            <span class="counseling-badge" style="{badge_style}">{pred_class.upper()}</span>
          </div>

          <div class="c-doc-divider"><span>1 &middot; Pathological Nature &amp; Subtype Rationale</span><hr></div>
          <div class="c-body-text">{info['pathological_nature']}</div>

          <div class="c-doc-divider"><span>2 &middot; Critical Patient Precautions &amp; Red Flag Symptoms</span><hr></div>
          {''.join(_c_bullet(p, '') for p in info['precautions'])}

          <div class="c-doc-divider"><span>3 &middot; Recommended Confirmatory Diagnostic Workup</span><hr></div>
          {''.join(_c_bullet(s, '') for s in info['next_steps'])}

          <div class="c-doc-divider"><span>4 &middot; Attending Physician Consultation Checklist</span><hr></div>
          {_c_checklist(info['checklist'])}

        </div>
        """, unsafe_allow_html=True)

if upload is not None:
    raw_img_pil = Image.open(upload).convert('L')
    raw_arr = np.array(raw_img_pil).astype(np.float32) / 255.0
    raw_arr = cv2.resize(raw_arr, (256, 256))

    if not class_path.exists() or not seg_path.exists():
        st.markdown("""
        <div style="text-align:center;padding:32px;background:#FFFBEB;border:1px solid #FDE68A;border-radius:8px;margin-top:24px">
          <div style="font-size:15px;font-weight:600;color:#92400E;font-family:Figtree,Noto Sans,sans-serif;margin-bottom:8px">Model Weights Pending</div>
          <div style="color:#78350F;font-size:13px;font-family:Noto Sans,sans-serif;line-height:1.6">Train the models on Google Colab (Notebooks 2 &amp; 3) to enable live neural inference.</div>
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
        (c4, gradcam_overlay,"4. Grad-CAM Attention",     f"{cnr:+.1f}% CNR" if cnr != 0 else ""),
    ]
    for col, img, title, caption in panels:
        with col:
            if caption:
                badge_cls = ("badge-tumor" if "Tumor" in caption
                             else "badge-cnr" if "CNR" in caption
                             else "badge-clear")
                badge_html = f'<span class="step-badge {badge_cls}">{caption}</span>'
            else:
                badge_html = ""
            st.markdown(f"""
            <div class="pipeline-card">
              <span class="step-label">{title}</span>
              {badge_html}
            </div>""", unsafe_allow_html=True)
            st.image(img, use_container_width=True)

    st.divider()

    # ── Diagnosis + Probabilities ──────────────────────────────────────────────
    col_diag, col_prob, col_morph = st.columns([1.2, 1.2, 1.6])

    is_tumor = (pred_class != 'No Tumor')
    panel_side = "diag-positive" if is_tumor else "diag-negative"

    with col_diag:
        override_html = (
            '<div style="margin-top:8px;font-size:12px;color:#0891B2;font-weight:500;'
            'padding:6px 8px;background:#E0F2FE;border-radius:4px;border:1px solid #BAE6FD;'
            'font-family:Noto Sans,sans-serif">'
            'Multi-Modal Fusion Override: Lesion Detected by U-Net</div>'
        ) if fusion_override else ''
        st.markdown(f"""
        <div class="diag-panel {panel_side}">
          <div class="diag-label">Primary Diagnosis</div>
          <div class="diag-value">{pred_class}</div>
          <div class="diag-conf">Confidence: <strong style="font-family:'Noto Sans Mono',monospace;color:var(--text-hi)">{pred_conf:.1f}%</strong></div>
          {override_html}
          <div style="margin-top:16px;font-size:13px;color:var(--text-muted);line-height:1.7;font-family:'Noto Sans',sans-serif">
            Pipeline: {model_choice}<br>
            Sequence: {sequence}
          </div>
        </div>""", unsafe_allow_html=True)

    with col_prob:
        st.markdown('<div style="padding-top:4px">', unsafe_allow_html=True)
        st.markdown('<div class="prob-section-label">Confidence Distribution</div>', unsafe_allow_html=True)
        # Sorted descending; rank drives teal opacity (rank 0 = darkest = highest confidence)
        sorted_probs = sorted(zip(CLASSES, probs), key=lambda x: x[1], reverse=True)
        for rank, (cls, prob) in enumerate(sorted_probs):
            st.markdown(prob_bar_html(cls, prob, rank), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_morph:
        st.markdown('<div class="prob-section-label">Morphometric Analysis</div>', unsafe_allow_html=True)
        if area > 0:
            st.markdown(f"""
            <div class="morph-table-wrap">
              <table class="morph-table">
                <thead>
                  <tr><th>Metric</th><th style="text-align:right">Value</th></tr>
                </thead>
                <tbody>
                  <tr><td>Tumor Area</td><td>{area:,} px&sup2; &asymp; {area/100:.1f} mm&sup2;</td></tr>
                  <tr><td>CNR Gain</td><td>{cnr:+.1f}%</td></tr>
                  <tr><td>Perimeter</td><td>{perim:.0f} px</td></tr>
                  <tr><td>Centroid (X, Y)</td><td>{centroid if centroid else '&mdash;'}</td></tr>
                </tbody>
              </table>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="morph-table-wrap" style="padding:24px;text-align:center">
              <div style="font-size:13px;color:#16A34A;font-weight:600;font-family:Noto Sans,sans-serif;margin-bottom:4px">No Focal Lesion Detected</div>
              <div style="font-size:12px;color:#64748B;font-family:Noto Sans,sans-serif">Morphometric analysis bypassed &mdash; segmentation mask empty</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Clinical Counseling Module ─────────────────────────────────────────────
    render_counseling_section(pred_class)

    st.divider()

    # ── PDF Export Button ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:24px;padding:16px 24px;background:var(--bg-card);border:1px solid var(--border);border-left:4px solid var(--primary);border-radius:8px">
      <div style="font-size:15px;font-weight:600;color:var(--text-hi);font-family:'Figtree','Noto Sans',sans-serif;margin-bottom:4px">Export Official Diagnostic Report</div>
      <div style="font-size:13px;color:var(--text-muted);font-family:'Noto Sans',sans-serif;line-height:1.6">Generate a complete PDF clinical report including scan images, morphometric biomarkers, confidence distribution, and patient counseling guidelines.</div>
    </div>""", unsafe_allow_html=True)

    if st.button("Export Official Diagnostic Report (PDF)", type="primary", use_container_width=True):
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
            
            label = "Download PDF Report" if ext == "pdf" else "Download HTML Report — open in browser and use File → Print → Save as PDF"
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
                    "HTML report downloaded. Open it in any browser and use "
                    "**File → Print → Save as PDF** to produce a full A4 clinical PDF."
                )
        except Exception as e:
            st.error(f"Report generation error: {e}")

else:
    # Empty state
    st.markdown("""
    <div class="empty-state">
      <div class="empty-brain-icon">
        <svg width="72" height="72" viewBox="0 0 24 24" fill="none"
             stroke="#94A3B8" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"
             xmlns="http://www.w3.org/2000/svg">
          <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/>
          <path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/>
          <path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/>
          <path d="M17.599 6.5a3 3 0 0 0 .399-1.375"/>
          <path d="M6.003 5.125A3 3 0 0 0 6.401 6.5"/>
          <path d="M3.477 10.896a4 4 0 0 1 .585-.396"/>
          <path d="M19.938 10.5a4 4 0 0 1 .585.396"/>
          <path d="M6 18a4 4 0 0 1-1.967-.516"/>
          <path d="M19.967 17.484A4 4 0 0 1 18 18"/>
        </svg>
      </div>
      <div class="empty-title">Awaiting MRI Scan</div>
      <div class="empty-sub">Upload a DICOM-compatible diagnostic scan using the sidebar to begin AI-assisted neuro-oncology analysis.</div>
      <div class="empty-cap-list">
        <span class="empty-cap">WPT Enhancement</span>
        <span class="empty-cap">U-Net Segmentation</span>
        <span class="empty-cap">EfficientNetB2 Classification</span>
        <span class="empty-cap">Grad-CAM Explainability</span>
      </div>
    </div>""", unsafe_allow_html=True)
