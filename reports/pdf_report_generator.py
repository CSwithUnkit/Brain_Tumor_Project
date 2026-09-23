"""
reports/pdf_report_generator.py
================================
Hospital-grade clinical PDF report generator for the NeuroScan AI workstation.

Uses fpdf2 (pure-Python, no LaTeX) to produce an A4-format diagnostic PDF with:
  1. Institutional header & metadata bar
  2. 2x2 visual quad-panel (Raw / Enhanced / Segmentation / Grad-CAM)
  3. Confidence table + morphometric biomarkers
  4. Neuro-oncological counseling sections (Pathological Impression,
     Patient Precautions, Diagnostic Workup, Physician Checklist)
  5. AI medical disclaimer + radiologist signature block

Entry point:
    from reports.pdf_report_generator import generate_pdf_report
    pdf_bytes = generate_pdf_report(...)  # -> bytes, ready for st.download_button
"""

from __future__ import annotations

import io
import os
import tempfile
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

# -- fpdf2 soft-import -------------------------------------------------------
try:
    from fpdf import FPDF, XPos, YPos
    _FPDF_AVAILABLE = True
except ImportError:
    _FPDF_AVAILABLE = False
    FPDF = object  # type: ignore[assignment,misc]

# -- Clinical counseling knowledge base ---------------------------------------
COUNSELING_DB: dict = {
    "Intra-axial Glial Neoplasm": {
        "pathological_nature": (
            "Infiltrative intra-axial neuroepithelial malignancy involving cerebral white matter. "
            "Gliomas arise from glial progenitor cells and demonstrate varying degrees of invasion "
            "along white matter tracts. WHO grading (I-IV) is determined by histology and molecular "
            "markers. High-grade variants (GBM, WHO Grade IV) carry median survival of 14-16 months "
            "with standard Stupp protocol chemoradiation."
        ),
        "precautions": [
            "Seizure precautions: avoid driving, operating heavy machinery, or unsupervised swimming; "
            "keep emergency anticonvulsant (e.g., lorazepam) readily accessible.",
            "Monitor for raised intracranial pressure (ICP) symptoms: early-morning headache that "
            "worsens on Valsalva, projectile vomiting, papilledema, or progressive focal neurological deficit.",
            "Avoid significant head trauma; use helmet during activities with fall risk.",
            "Report any sudden change in speech, motor function, or seizure pattern immediately to "
            "emergency services or treating neurosurgeon.",
            "Corticosteroids (dexamethasone) may be prescribed for cerebral oedema -- monitor blood "
            "glucose and blood pressure; do not abruptly discontinue.",
        ],
        "next_steps": [
            "MR Spectroscopy: Evaluate Choline/NAA ratio (elevated Cho:NAA > 2 supports high-grade glioma).",
            "Thin-slice 3D T1 contrast-enhanced MRI (1mm isotropic MPRAGE) for surgical planning.",
            "Stereotactic neuro-navigation biopsy or maximal safe surgical resection consultation.",
            "Molecular panel (tissue): IDH1/IDH2 mutation status, 1p/19q codeletion, "
            "MGMT promoter methylation, TERT promoter, EGFR amplification.",
            "Functional MRI (fMRI) and DTI tractography if lesion is near eloquent cortex or "
            "corticospinal tract.",
            "Radiation Oncology and Medical Oncology (neuro-oncology) multidisciplinary tumour board review.",
        ],
        "checklist": [
            "Discussed diagnosis and WHO grade uncertainty pending tissue biopsy with patient/family.",
            "Seizure precautions counselled; anticonvulsant prescription issued if indicated.",
            "Corticosteroid therapy initiated for symptomatic cerebral oedema (if applicable).",
            "Neurosurgery referral placed for stereotactic biopsy/resection.",
            "Molecular testing panel ordered.",
            "Multidisciplinary tumour board (MDT) referral submitted.",
            "Follow-up MRI date scheduled (typically 6-8 weeks post-op or per MDT guidance).",
            "Palliative care / psycho-oncology referral offered.",
        ],
    },
    "Extra-axial Dural Lesion": {
        "pathological_nature": (
            "Extra-axial, dural-based lesion arising from arachnoid cap cells, most commonly benign "
            "(WHO Grade I). Meningiomas represent ~37% of primary CNS tumours. WHO Grade II (atypical) "
            "and Grade III (anaplastic) variants have higher recurrence rates. They may cause symptoms "
            "via mass effect on adjacent brain, cranial nerves, or dural venous sinuses. The characteristic "
            "MRI appearance is a homogeneously enhancing extra-axial mass with a 'dural tail' sign."
        ),
        "precautions": [
            "Monitor for cranial nerve compression: sudden visual field changes (chiasmal compression), "
            "diplopia, facial numbness, or hearing loss -- present to emergency if acute.",
            "Avoid vigorous neck manipulation (chiropractic, high-impact sports with neck rotation) "
            "if tumour is in the skull base or cavernous sinus region.",
            "Report any new-onset focal motor deficit, word-finding difficulty, or sudden severe headache.",
            "Anticonvulsants are NOT routinely indicated for meningioma unless seizures have occurred.",
            "If on anticoagulation for other conditions, discuss with neurosurgeon before any procedural "
            "intervention given vascular supply of meningioma from external carotid branches.",
        ],
        "next_steps": [
            "High-resolution CT with bone windows: assess bone hyperostosis, intratumoral calcification, "
            "and skull base involvement.",
            "MR Angiography or DSA: evaluate vascular supply (middle meningeal artery) and proximity to "
            "dural venous sinuses (superior sagittal, transverse) -- critical for resection planning.",
            "Formal ophthalmology / neuro-ophthalmology review if near optic nerve, chiasm, or cavernous sinus.",
            "Simpson Grade resection planning (neurosurgical consultation) -- Simpson Grade I/II associated "
            "with lowest recurrence.",
            "Consider Stereotactic Radiosurgery (Gamma Knife / CyberKnife) if lesion is small (<3 cm), "
            "in eloquent location, or unresectable (skull base, cavernous sinus).",
            "Annual MRI surveillance for small, asymptomatic, incidentally discovered meningiomas.",
        ],
        "checklist": [
            "Communicated benign vs atypical/anaplastic grade uncertainty pending post-op histology.",
            "Cranial nerve examination performed and documented.",
            "Ophthalmology referral placed if visual symptoms present.",
            "Neurosurgery consulted for Simpson Grade resection or SRS suitability.",
            "CT bone windows ordered to assess skull base involvement.",
            "MR angiography requested if near major venous sinuses.",
            "Surveillance MRI protocol discussed if 'wait and scan' approach adopted.",
            "Patient education provided regarding symptom red flags requiring emergency review.",
        ],
    },
    "Sella Turcica Pituitary Adenoma": {
        "pathological_nature": (
            "Sella turcica neuroendocrine neoplasm arising from anterior pituitary gland cells, adjacent "
            "to the optic chiasm. Classified as microadenoma (<10 mm) or macroadenoma (>=10 mm). "
            "Functional adenomas (secreting: prolactinoma, GH-secreting acromegaly, ACTH-secreting "
            "Cushing's disease) require endocrine therapy in addition to surgical/radiosurgical management. "
            "Non-functioning adenomas cause symptoms primarily via mass effect on the optic chiasm "
            "(bitemporal hemianopia) and pituitary stalk compression."
        ),
        "precautions": [
            "Urgent visual field assessment: bitemporal hemianopia (superior field defect first) "
            "indicates chiasmal compression -- this is a surgical urgency.",
            "Watch for adrenal crisis symptoms: severe fatigue, dizziness, hypotension, nausea, "
            "hypoglycemia -- especially post-surgical or with concurrent illness (sick-day rules).",
            "Avoid medications that may elevate prolactin (antipsychotics, metoclopramide, domperidone) "
            "in prolactinoma patients unless medically essential.",
            "Pituitary apoplexy risk: sudden severe headache, acute visual loss, or altered consciousness "
            "requires immediate emergency assessment -- may represent haemorrhage into the adenoma.",
            "Hormonal replacement (hydrocortisone, levothyroxine, testosterone/oestrogen) must not be "
            "abruptly discontinued.",
        ],
        "next_steps": [
            "Formal Goldman visual field perimetry (Humphrey 24-2 or 30-2) -- quantify bitemporal defect "
            "and establish baseline for monitoring.",
            "Complete anterior pituitary endocrine blood panel: Prolactin (PRL), ACTH, Morning Cortisol "
            "(8 AM), Growth Hormone (GH), IGF-1, TSH, Free T4, LH, FSH, Testosterone/Oestradiol.",
            "Dedicated pituitary MRI protocol: coronal 3mm T1 pre/post gadolinium + dynamic sequence "
            "(if not already performed) for precise delineation of adenoma vs normal gland.",
            "Endocrine / Endocrinology consultation for functional adenoma management "
            "(dopamine agonist for prolactinoma; somatostatin analogue for acromegaly).",
            "Endoscopic Transsphenoidal Surgery (ETS) consultation for macroadenomas with "
            "chiasmal compression or functional adenomas refractory to medical therapy.",
            "Post-operative cortisol and pituitary function assessment (Day 1 morning cortisol) if surgery performed.",
        ],
        "checklist": [
            "Visual field testing (Goldman/Humphrey) requested urgently if chiasmal compression suspected.",
            "Full anterior pituitary hormone panel ordered.",
            "Ophthalmology referral placed for formal perimetry.",
            "Endocrinology consultation requested.",
            "Transsphenoidal surgery consultation placed for macroadenoma with visual compromise.",
            "Hydrocortisone stress dosing protocol explained to patient (sick-day rules).",
            "Patient advised re: pituitary apoplexy warning symptoms.",
            "Dopamine agonist therapy initiated for prolactinoma if appropriate (Endocrinology-led).",
        ],
    },
    "No Tumor": {
        "pathological_nature": (
            "Unremarkable cerebral parenchyma with normal ventricular symmetry, no focal mass effect, "
            "no abnormal enhancement, and no evidence of midline shift. White matter signal is within "
            "normal limits for age. No evidence of an intracranial neoplasm on this study."
        ),
        "precautions": [
            "Reassure patient: no intracranial tumour identified on AI-assisted MRI analysis.",
            "Evaluate non-neoplastic causes of presenting symptoms: tension headache, migraine (with aura), "
            "cervical spondylosis / cervicogenic headache, or age-related microvascular changes.",
            "Alert patient to 'red flag' headache symptoms warranting urgent re-imaging: thunderclap onset "
            "(SAH), progressive headache worsening over weeks, headache with fever/neck stiffness, "
            "or new neurological deficit.",
            "Consider vascular pathology if symptoms suggest: MR Angiography for suspected aneurysm or AVM.",
            "Psychiatric / psychological evaluation if headache/neurological symptoms are functional in origin.",
        ],
        "next_steps": [
            "Clinical follow-up as symptoms dictate -- no urgent neurosurgical intervention required.",
            "If headaches persist: Neurology referral for headache classification and pharmacological management.",
            "Cervical spine MRI if cervicogenic headache is suspected.",
            "EEG if seizure-like episodes reported.",
            "Consider ophthalmology review for visual symptoms (migraine equivalent, papilledema screening).",
        ],
        "checklist": [
            "Patient reassured regarding absence of neoplastic pathology on this study.",
            "Non-neoplastic differential diagnoses discussed.",
            "Red-flag headache warning criteria explained.",
            "Neurology referral offered if headaches are disabling or progressive.",
            "Follow-up plan documented (clinical review vs re-imaging timeline).",
            "GP / primary care physician informed of normal scan result.",
        ],
    },
}


class _ClinicalPDF(FPDF if _FPDF_AVAILABLE else object):  # type: ignore[misc]
    """fpdf2-based A4 clinical report with institutional header/footer on every page."""

    C_DARK   = (9,   13,  22)
    C_SLATE  = (15,  23,  42)
    C_CYAN   = (6,   182, 212)
    C_GREEN  = (16,  185, 129)
    C_RED    = (244, 63,  94)
    C_AMBER  = (245, 158, 11)
    C_VIOLET = (99,  102, 241)
    C_WHITE  = (255, 255, 255)
    C_LIGHT  = (248, 250, 252)
    C_BORDER = (226, 232, 240)
    C_MID    = (71,  85,  105)
    C_TEXT   = (15,  23,  42)
    C_FAINT  = (148, 163, 184)

    def __init__(self, institution: str = ""):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.institution = institution
        self.set_auto_page_break(auto=True, margin=20)
        self.add_page()

    def header(self):
        self.set_fill_color(*self.C_DARK)
        self.rect(0, 0, 210, 26, style="F")
        self.set_fill_color(*self.C_CYAN)
        self.rect(0, 26, 210, 0.8, style="F")
        self.set_xy(10, 5)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*self.C_WHITE)
        self.cell(0, 6, "NEUROSCAN CLINICAL ONCOLOGY WORKSTATION", ln=False)
        self.set_xy(10, 12)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*self.C_CYAN)
        self.cell(0, 5,
            "MRI Image Enhancement and Tumor Detection  --  AI-Assisted Diagnostic Report  |  "
            + self.institution)
        self.set_xy(130, 7)
        self.set_font("Helvetica", "B", 7.5)
        self.set_text_color(*self.C_WHITE)
        self.cell(70, 5, "DIAGNOSTIC IMAGING REPORT", align="R")
        self.set_xy(0, 27)

    def footer(self):
        self.set_y(-13)
        self.set_fill_color(*self.C_SLATE)
        self.rect(0, self.get_y(), 210, 13, style="F")
        self.set_xy(10, self.get_y() + 3)
        self.set_font("Helvetica", "I", 6.5)
        self.set_text_color(*self.C_FAINT)
        self.cell(0, 4,
            "AI-assisted decision support only (SaMD). Requires clinical correlation "
            "by a certified neuro-radiologist. Not for independent diagnostic use.",
            align="L")
        self.set_xy(0, self.get_y())
        self.cell(200, 4, f"Page {self.page_no()}", align="R")

    def section_title(self, text: str, color: tuple = None):
        color = color or self.C_CYAN
        self.ln(4)
        self.set_fill_color(*color)
        self.rect(10, self.get_y(), 190, 0.5, style="F")
        self.ln(2)
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(*color)
        self.set_x(10)
        self.cell(0, 6, "  " + text.upper(), ln=True)
        self.set_fill_color(*self.C_BORDER)
        self.rect(10, self.get_y(), 190, 0.3, style="F")
        self.ln(3)

    def kv_row(self, key: str, value: str, shade: bool = False):
        if shade:
            self.set_fill_color(*self.C_LIGHT)
            self.rect(10, self.get_y(), 190, 6.5, style="F")
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*self.C_MID)
        self.set_x(12)
        self.cell(58, 6.5, key, ln=False)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*self.C_TEXT)
        self.multi_cell(128, 6.5, value, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def bullet(self, text: str, color: tuple = None, indent: float = 14):
        color = color or self.C_TEXT
        self.set_x(indent)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*color)
        self.cell(5, 5.5, chr(8226), ln=False)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*self.C_TEXT)
        self.set_x(indent + 5)
        self.multi_cell(185 - indent, 5.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def checklist_item(self, text: str, idx: int):
        self.set_x(14)
        self.set_fill_color(*self.C_BORDER)
        self.rect(14, self.get_y() + 1, 4, 4, style="FD")
        self.set_x(20)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*self.C_TEXT)
        self.multi_cell(176, 5.5, f"{idx}. {text}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(0.5)


def _np_to_pil_tmp(arr: np.ndarray) -> str:
    """Save a numpy uint8 RGB array to a temporary PNG file; return path."""
    img = Image.fromarray(arr.astype(np.uint8))
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img.save(path)
    return path


def generate_pdf_report(
    patient_id: str,
    scan_date: str,
    sequence: str,
    institution: str,
    pred_class: str,
    pred_conf: float,
    probs: List[float],
    area: int,
    perim: float,
    centroid,
    cnr: float,
    raw_img: np.ndarray,
    enh_img: np.ndarray,
    seg_img: np.ndarray,
    gradcam_img: np.ndarray,
    model_choice: str = "Exp 2: Enhanced",
    bbox=None,
) -> bytes:
    """
    Generate a hospital-grade A4 clinical PDF report and return raw bytes.

    Parameters
    ----------
    patient_id   : Patient / Study identifier
    scan_date    : Scan date string
    sequence     : MRI sequence name
    institution  : Referring institution name
    pred_class   : Predicted tumour class name (Glioma / Meningioma / Pituitary Adenoma / No Tumor)
    pred_conf    : Model confidence 0-100 float
    probs        : Softmax probability list [Glioma, Meningioma, Pituitary, No Tumor]
    area         : Tumour area in pixels squared
    perim        : Tumour perimeter in pixels
    centroid     : (x, y) tuple or None
    cnr          : CNR enhancement ratio percent
    raw_img      : 256x256 RGB uint8 ndarray - original MRI
    enh_img      : 256x256 RGB uint8 ndarray - WPT->LMMSE->CLAHE enhanced
    seg_img      : 256x256 RGB uint8 ndarray - segmentation overlay
    gradcam_img  : 256x256 RGB uint8 ndarray - Grad-CAM heatmap overlay
    model_choice : Active model pipeline label
    bbox         : Bounding box tuple or None

    Returns
    -------
    bytes  -- Raw PDF content ready for st.download_button(mime="application/pdf")
    """
    if not _FPDF_AVAILABLE:
        # Graceful fallback: generate print-ready HTML as bytes instead
        # fpdf2's fonttools DLL may be blocked by system Application Control policy.
        # The HTML fallback is fully functional and can be printed to PDF from any browser.
        return generate_html_clinical_report(
            patient_id=patient_id, scan_date=scan_date, sequence=sequence,
            institution=institution, pred_class=pred_class, pred_conf=pred_conf,
            probs=probs, area=area, perim=perim, centroid=centroid, cnr=cnr,
            raw_img=raw_img, enh_img=enh_img, seg_img=seg_img, gradcam_img=gradcam_img,
            model_choice=model_choice, bbox=bbox,
        ).encode("utf-8")

    CLASSES    = ["Intra-axial Glial Neoplasm", "Extra-axial Dural Lesion", "Sella Turcica Pituitary Adenoma", "No Tumor"]
    ts_full    = datetime.now().strftime("%d %B %Y, %H:%M")
    report_id  = f"NSR-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    counseling = COUNSELING_DB.get(pred_class, COUNSELING_DB["No Tumor"])
    is_tumor   = pred_class != "No Tumor"

    # Save quad images to temp files
    tmp_raw  = _np_to_pil_tmp(raw_img)
    tmp_enh  = _np_to_pil_tmp(enh_img)
    tmp_seg  = _np_to_pil_tmp(seg_img)
    tmp_gcam = _np_to_pil_tmp(gradcam_img)
    tmp_files = [tmp_raw, tmp_enh, tmp_seg, tmp_gcam]

    try:
        pdf = _ClinicalPDF(institution=institution)

        # -- Metadata bar ---------------------------------------------------
        pdf.set_fill_color(*_ClinicalPDF.C_LIGHT)
        pdf.rect(10, pdf.get_y(), 190, 7, style="F")
        pdf.set_x(12)
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(*_ClinicalPDF.C_MID)
        pdf.cell(0, 7,
            f"Patient: {patient_id}   |   Date: {scan_date}   |   "
            f"Report ID: {report_id}   |   Generated: {ts_full}",
            ln=True)

        # -- Section 1: Patient & Examination Details ----------------------
        pdf.section_title("1. Patient & Examination Details")
        details = [
            ("Patient ID",         patient_id),
            ("Scan Date",          scan_date),
            ("Report Generated",   ts_full),
            ("Report ID",          report_id),
            ("MRI Sequence",       sequence),
            ("Modality",           "MRI Brain - 1.5T / 3.0T T1-Contrast Enhanced"),
            ("Referring Dept",     "Neurology / Neuro-Oncology"),
            ("Institution",        institution),
            ("AI Pipeline",        model_choice),
            ("Enhancement",        "WPT -> LMMSE -> CLAHE (+CNR)"),
            ("Segmentation Model", "U-Net (best_unet_enhanced.pth)"),
            ("Classifier",         "EfficientNetB2 -- BRISC 2025 (4-class)"),
            ("Explainability",     "Gradient-weighted Class Activation Maps (Grad-CAM)"),
        ]
        for i, (k, v) in enumerate(details):
            pdf.kv_row(k, v, shade=(i % 2 == 0))
        pdf.ln(2)

        # -- Section 2: Primary Diagnosis ----------------------------------
        diag_color = _ClinicalPDF.C_RED if is_tumor else _ClinicalPDF.C_GREEN
        pdf.section_title("2. Primary AI Diagnosis", color=diag_color)
        fill_rgb = (255, 242, 244) if is_tumor else (240, 253, 244)
        pdf.set_fill_color(*fill_rgb)
        pdf.rect(10, pdf.get_y(), 190, 20, style="F")
        pdf.set_fill_color(*diag_color)
        pdf.rect(10, pdf.get_y(), 2, 20, style="F")
        y0 = pdf.get_y()
        pdf.set_xy(15, y0 + 2)
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(*diag_color)
        pdf.cell(100, 8, pred_class.upper(), ln=False)
        pdf.set_xy(15, y0 + 12)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(*_ClinicalPDF.C_MID)
        pdf.cell(0, 6,
            f"Model Confidence: {pred_conf:.1f}%   |   Pipeline: {model_choice}   |   Sequence: {sequence}",
            ln=True)
        pdf.ln(3)

        # -- Section 3: Softmax Distribution --------------------------------
        pdf.section_title("3. Differential Diagnosis -- Softmax Distribution")
        col_colors = [
            _ClinicalPDF.C_RED,
            _ClinicalPDF.C_AMBER,
            _ClinicalPDF.C_VIOLET,
            _ClinicalPDF.C_GREEN,
        ]
        col_w = [62, 38, 90]
        # Header
        pdf.set_fill_color(*_ClinicalPDF.C_SLATE)
        pdf.set_x(10)
        for h, w in zip(["Pathology", "Probability (%)", "Confidence Bar"], col_w):
            pdf.set_font("Helvetica", "B", 7.5)
            pdf.set_text_color(*_ClinicalPDF.C_WHITE)
            pdf.cell(w, 7, f"  {h}", border=0, fill=True, ln=False)
        pdf.ln(7)
        for i, (cls, prob) in enumerate(zip(CLASSES, probs)):
            y_row = pdf.get_y()
            if i % 2 == 0:
                pdf.set_fill_color(*_ClinicalPDF.C_LIGHT)
                pdf.rect(10, y_row, 190, 7, style="F")
            pdf.set_x(10)
            pdf.set_font("Helvetica", "B" if cls == pred_class else "", 8)
            pdf.set_text_color(*(col_colors[i] if cls == pred_class else _ClinicalPDF.C_TEXT))
            pdf.cell(col_w[0], 7, f"  {cls}", ln=False)
            pdf.set_font("Helvetica", "B" if cls == pred_class else "", 8)
            pdf.cell(col_w[1], 7, f"  {prob * 100:.1f}%", ln=False)
            bar_x   = pdf.get_x() + 2
            bar_y   = y_row + 1.8
            bar_max = 84
            pdf.set_fill_color(*_ClinicalPDF.C_BORDER)
            pdf.rect(bar_x, bar_y, bar_max, 3.5, style="F")
            pdf.set_fill_color(*col_colors[i])
            pdf.rect(bar_x, bar_y, bar_max * prob, 3.5, style="F")
            pdf.ln(7)
        pdf.ln(3)

        # -- Section 4: Visual Quad-Panel -----------------------------------
        pdf.section_title("4. Visual Examination -- Diagnostic Quad-Panel")
        labels   = [
            "1. Raw MRI Input",
            "2. WPT->LMMSE->CLAHE Enhanced",
            "3. U-Net Lesion Segmentation",
            "4. Grad-CAM XAI Attribution",
        ]
        captions = [
            "Original acquisition",
            f"{cnr:+.1f}% CNR gain",
            "Crimson tumour RoI / contour",
            "Model attention heatmap",
        ]
        imgs  = [tmp_raw, tmp_enh, tmp_seg, tmp_gcam]
        iw, ih, gap = 43, 43, 4
        for row in range(2):
            y_row = pdf.get_y()
            for col in range(2):
                idx   = row * 2 + col
                x_img = 10 + col * (iw + gap)
                pdf.set_fill_color(*_ClinicalPDF.C_BORDER)
                pdf.rect(x_img - 0.5, y_row - 0.5, iw + 1, ih + 8.5, style="F")
                pdf.set_xy(x_img, y_row)
                pdf.set_font("Helvetica", "B", 6.5)
                pdf.set_text_color(*_ClinicalPDF.C_CYAN)
                pdf.cell(iw, 5, labels[idx], align="C", ln=False)
                try:
                    pdf.image(imgs[idx], x=x_img, y=y_row + 5, w=iw, h=ih - 5)
                except Exception:
                    pass
                pdf.set_xy(x_img, y_row + ih + 1)
                pdf.set_font("Helvetica", "I", 6)
                pdf.set_text_color(*_ClinicalPDF.C_FAINT)
                pdf.cell(iw, 4, captions[idx], align="C", ln=False)
            pdf.set_y(y_row + ih + 6)
            if row == 0:
                pdf.ln(2)
        pdf.ln(4)

        # -- Section 5: Morphometric Biomarkers ----------------------------
        pdf.section_title("5. Quantitative Morphometric Biomarkers")
        mm2_est = round(area * 0.25, 1) if area > 0 else 0
        morpho  = [
            ("Tumour Area",          f"{area:,} px2" if area > 0 else "No focal lesion"),
            ("Estimated Area (mm2)", f"{mm2_est} mm2" if area > 0 else "N/A"),
            ("Lesion Perimeter",     f"{perim:.1f} px" if area > 0 else "N/A"),
            ("Centroid (X, Y)",      str(centroid) if centroid else "N/A"),
            ("Bounding Box (XYXY)",  str(bbox) if bbox else "N/A"),
            ("CNR Enhancement",      f"{cnr:+.1f}% (WPT->LMMSE->CLAHE pipeline)"),
        ]
        for i, (k, v) in enumerate(morpho):
            pdf.kv_row(k, v, shade=(i % 2 == 0))
        pdf.ln(4)

        # -- Section 6: Clinical Counseling ---------------------------------
        pdf.section_title("6. Physician's Clinical Counseling & Next Steps")

        # 6.1 Pathological Impression
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(*_ClinicalPDF.C_CYAN)
        pdf.set_x(10)
        pdf.cell(0, 6, "6.1  Pathological Impression & Subtype Rationale", ln=True)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*_ClinicalPDF.C_TEXT)
        pdf.set_x(14)
        pdf.multi_cell(182, 5.5, counseling["pathological_nature"],
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

        # 6.2 Precautions
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(*_ClinicalPDF.C_AMBER)
        pdf.set_x(10)
        pdf.cell(0, 6, "6.2  Essential Patient Precautions & Red Flag Warning Symptoms", ln=True)
        for prec in counseling["precautions"]:
            pdf.bullet(prec, color=_ClinicalPDF.C_AMBER)
        pdf.ln(3)

        # 6.3 Diagnostic Workup
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(*_ClinicalPDF.C_VIOLET)
        pdf.set_x(10)
        pdf.cell(0, 6, "6.3  Recommended Confirmatory Diagnostic Workup", ln=True)
        for step in counseling["next_steps"]:
            pdf.bullet(step, color=_ClinicalPDF.C_VIOLET)
        pdf.ln(3)

        # 6.4 Physician Checklist
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(*_ClinicalPDF.C_GREEN)
        pdf.set_x(10)
        pdf.cell(0, 6, "6.4  Attending Physician Consultation Checklist", ln=True)
        for i, item in enumerate(counseling["checklist"], start=1):
            pdf.checklist_item(item, i)
        pdf.ln(4)

        # -- Section 7: Disclaimer -----------------------------------------
        pdf.section_title("7. Medical Disclaimer & Regulatory Status",
                          color=_ClinicalPDF.C_AMBER)
        pdf.set_fill_color(255, 252, 235)
        pdf.rect(10, pdf.get_y(), 190, 22, style="F")
        pdf.set_fill_color(*_ClinicalPDF.C_AMBER)
        pdf.rect(10, pdf.get_y(), 2, 22, style="F")
        y_d = pdf.get_y()
        pdf.set_xy(14, y_d + 2)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*_ClinicalPDF.C_AMBER)
        pdf.cell(0, 5, "WARNING: AI-Assisted Diagnostic Decision Support Tool (SaMD)", ln=True)
        pdf.set_xy(14, pdf.get_y())
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(*_ClinicalPDF.C_TEXT)
        pdf.multi_cell(182, 5,
            "This report is generated by an AI decision-support system and must NOT be used as "
            "a standalone clinical diagnosis. All findings require review and confirmation by a "
            "qualified neuro-radiologist or neuro-oncologist before any patient management "
            "decision. Classified as Software as a Medical Device (SaMD) -- research and "
            "educational use only. Not CE/FDA cleared for independent diagnostic use.",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(6)

        # -- Section 8: Signature Block ------------------------------------
        pdf.section_title("8. Verification & Authorisation")
        sig_rows = [
            ("Reporting Radiologist / Neuro-Oncologist", "_" * 40),
            ("Medical Registration Number",              "_" * 20),
            ("Verification Date",                        "_" * 20),
            ("Department / Unit",                        "_" * 28),
            ("Institution Stamp / Seal",                 "(official stamp here)"),
        ]
        for i, (k, v) in enumerate(sig_rows):
            pdf.kv_row(k, v, shade=(i % 2 == 0))
        pdf.ln(4)
        pdf.set_x(10)
        pdf.set_font("Helvetica", "I", 6.5)
        pdf.set_text_color(*_ClinicalPDF.C_FAINT)
        pdf.multi_cell(190, 5,
            f"Report auto-generated by NeuroScan AI Clinical Workstation -- "
            f"MRI Image Enhancement and Tumor Detection System v1.0  |  "
            f"EfficientNetB2 + U-Net  |  BRISC 2025 Dataset  |  "
            f"{institution}  |  {report_id}",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        return bytes(pdf.output())

    finally:
        for p in tmp_files:
            try:
                os.unlink(p)
            except OSError:
                pass


# ── Pure-Python HTML fallback (no native DLL required) ────────────────────────

def _img_to_base64(arr: np.ndarray) -> str:
    """Convert numpy uint8 RGB array to base64 PNG data URI."""
    import base64
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def generate_html_clinical_report(
    patient_id: str,
    scan_date: str,
    sequence: str,
    institution: str,
    pred_class: str,
    pred_conf: float,
    probs: List[float],
    area: int,
    perim: float,
    centroid,
    cnr: float,
    raw_img: np.ndarray,
    enh_img: np.ndarray,
    seg_img: np.ndarray,
    gradcam_img: np.ndarray,
    model_choice: str = "Exp 2: Enhanced",
    bbox=None,
) -> str:
    """
    Generate a fully-styled, print-ready HTML clinical report.
    This is the fallback used when fpdf2 is unavailable (e.g. DLL policy blocks).
    Open in any browser and use Ctrl+P -> Save as PDF for a perfect A4 layout.
    Returns the HTML string.
    """
    CLASSES = ["Intra-axial Glial Neoplasm", "Extra-axial Dural Lesion", "Sella Turcica Pituitary Adenoma", "No Tumor"]
    ts_full   = datetime.now().strftime("%d %B %Y, %H:%M")
    report_id = f"NSR-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    counseling = COUNSELING_DB.get(pred_class, COUNSELING_DB["No Tumor"])
    is_tumor   = pred_class != "No Tumor"

    # Encode images as base64
    img_raw  = _img_to_base64(raw_img)
    img_enh  = _img_to_base64(enh_img)
    img_seg  = _img_to_base64(seg_img)
    img_gcam = _img_to_base64(gradcam_img)

    diag_color  = "#dc2626" if is_tumor else "#059669"
    diag_bg     = "#fef2f2" if is_tumor else "#f0fdf4"
    diag_border = "#ef4444" if is_tumor else "#10b981"

    COLORS = ["#ef4444", "#f59e0b", "#8b5cf6", "#10b981"]
    prob_rows = "".join(
        f"<tr><td>{cls}</td>"
        f"<td style='font-weight:700;color:{col}'>{p*100:.1f}%</td>"
        f"<td><div style='background:#e2e8f0;border-radius:3px;height:10px;width:100%'>"
        f"<div style='background:{col};border-radius:3px;height:10px;width:{p*100:.1f}%'></div>"
        f"</div></td></tr>"
        for cls, p, col in zip(CLASSES, probs, COLORS)
    )
    mm2_est = round(area * 0.25, 1) if area > 0 else 0
    morph_rows = "".join(
        f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>"
        for k, v in [
            ("Tumour Area", f"{area:,} px2" if area > 0 else "No focal lesion"),
            ("Estimated Area (mm2)", f"{mm2_est} mm2" if area > 0 else "N/A"),
            ("Lesion Perimeter", f"{perim:.1f} px" if area > 0 else "N/A"),
            ("Centroid (X, Y)", str(centroid) if centroid else "N/A"),
            ("Bounding Box (XYXY)", str(bbox) if bbox else "N/A"),
            ("CNR Enhancement", f"{cnr:+.1f}%"),
        ]
    )
    prec_items  = "".join(f"<li>{p}</li>" for p in counseling["precautions"])
    steps_items = "".join(f"<li>{s}</li>" for s in counseling["next_steps"])
    check_items = "".join(
        f"<li><input type='checkbox'> {item}</li>" for item in counseling["checklist"]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeuroScan Clinical Report - {patient_id}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Inter',sans-serif;background:#f8fafc;color:#0f172a;font-size:13px;line-height:1.6}}
  .page{{max-width:900px;margin:2rem auto;background:#fff;box-shadow:0 4px 24px rgba(0,0,0,.12);border-radius:4px;overflow:hidden}}
  .letterhead{{background:linear-gradient(135deg,#090d16,#0f172a);color:#fff;padding:1.8rem 2.5rem;display:flex;justify-content:space-between;align-items:flex-start}}
  .lh-title{{font-size:1.3rem;font-weight:700;letter-spacing:-.02em;margin-bottom:.2rem}}
  .lh-sub{{font-size:.75rem;color:#06b6d4;letter-spacing:.03em}}
  .lh-right{{text-align:right;font-size:.75rem;color:#94a3b8;line-height:1.8}}
  .meta-bar{{background:#f0f4f8;border-bottom:1px solid #e2e8f0;padding:.5rem 2.5rem;font-size:.72rem;color:#64748b;display:flex;justify-content:space-between}}
  .body{{padding:1.8rem 2.5rem}}
  .sh{{font-size:.65rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#64748b;border-bottom:2px solid #e2e8f0;padding-bottom:.3rem;margin:1.6rem 0 .9rem}}
  .diag-box{{background:{diag_bg};border-left:4px solid {diag_border};border-radius:0 6px 6px 0;padding:1.2rem 1.5rem;margin:.5rem 0 1rem}}
  .diag-label{{font-size:.65rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:{diag_color};margin-bottom:.35rem}}
  .diag-value{{font-size:1.8rem;font-weight:700;color:{diag_color};line-height:1;margin-bottom:.3rem}}
  .diag-conf{{font-size:.82rem;color:#64748b}}
  table{{width:100%;border-collapse:collapse;font-size:.82rem;margin-bottom:.5rem}}
  th{{background:#f0f4f8;text-align:left;padding:.45rem .7rem;font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#64748b}}
  td{{padding:.45rem .7rem;border-bottom:1px solid #edf0f4;color:#1e293b}}
  tr:last-child td{{border-bottom:none}}
  tr:hover td{{background:#f8fafc}}
  .quad-grid{{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:.5rem 0 1rem}}
  .quad-card{{border:1px solid #e2e8f0;border-radius:6px;overflow:hidden}}
  .quad-label{{background:#0f172a;color:#06b6d4;font-size:.65rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;padding:.4rem .7rem}}
  .quad-card img{{width:100%;display:block}}
  .quad-cap{{font-size:.68rem;color:#94a3b8;padding:.3rem .7rem;background:#f8fafc;font-style:italic}}
  ul{{padding-left:1.5rem;margin:.3rem 0 .8rem}}
  li{{margin:.25rem 0;color:#334155}}
  .disc{{background:#fffbeb;border:1px solid #fcd34d;border-radius:4px;padding:.9rem 1.1rem;font-size:.75rem;color:#78350f;margin-top:1.5rem}}
  .sig-grid{{display:grid;grid-template-columns:1fr 1fr;gap:.5rem 1.5rem;margin:.5rem 0}}
  .sig-field .sig-label{{font-size:.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.2rem}}
  .sig-field .sig-line{{border-bottom:1.5px solid #cbd5e1;padding-bottom:.3rem;font-size:.85rem;color:#e2e8f0}}
  .footer{{background:#f0f4f8;border-top:1px solid #e2e8f0;padding:.6rem 2.5rem;font-size:.68rem;color:#94a3b8;display:flex;justify-content:space-between}}
  @media print{{body{{background:#fff}}.page{{box-shadow:none;margin:0;border-radius:0}}}}
</style>
</head>
<body>
<div class="page">

<div class="letterhead">
  <div>
    <div class="lh-title">NEUROSCAN CLINICAL ONCOLOGY WORKSTATION</div>
    <div class="lh-sub">MRI Image Enhancement and Tumor Detection &nbsp;|&nbsp; AI-Assisted Diagnostic Report</div>
    <div class="lh-sub" style="margin-top:.3rem">{institution}</div>
  </div>
  <div class="lh-right">
    <div><strong style="color:#fff;font-size:.85rem">DIAGNOSTIC IMAGING REPORT</strong></div>
    <div>Generated: {ts_full}</div>
    <div>Report ID: {report_id}</div>
  </div>
</div>

<div class="meta-bar">
  <span><strong>{patient_id}</strong> &nbsp;|&nbsp; {sequence} &nbsp;|&nbsp; {scan_date}</span>
  <span>EfficientNetB2 + U-Net &nbsp;|&nbsp; {model_choice}</span>
</div>

<div class="body">

  <div class="sh">1. Patient &amp; Examination Details</div>
  <table><tbody>
    <tr><td>Patient ID</td><td><strong>{patient_id}</strong></td><td>Scan Date</td><td><strong>{scan_date}</strong></td></tr>
    <tr><td>Report ID</td><td>{report_id}</td><td>Report Generated</td><td>{ts_full}</td></tr>
    <tr><td>MRI Sequence</td><td>{sequence}</td><td>Modality</td><td>MRI Brain &mdash; 1.5T/3.0T T1-CE</td></tr>
    <tr><td>Institution</td><td>{institution}</td><td>Referring Dept</td><td>Neurology / Neuro-Oncology</td></tr>
    <tr><td>AI Pipeline</td><td>{model_choice}</td><td>Enhancement</td><td>WPT &rarr; LMMSE &rarr; CLAHE</td></tr>
    <tr><td>Classifier</td><td>EfficientNetB2 (BRISC 2025)</td><td>Explainability</td><td>Grad-CAM (layer4)</td></tr>
  </tbody></table>

  <div class="sh">2. Primary AI Diagnosis</div>
  <div class="diag-box">
    <div class="diag-label">Primary Finding</div>
    <div class="diag-value">{pred_class}</div>
    <div class="diag-conf">Model Confidence: <strong>{pred_conf:.1f}%</strong> &nbsp;|&nbsp; Pipeline: {model_choice} &nbsp;|&nbsp; Sequence: {sequence}</div>
  </div>

  <div class="sh">3. Differential Diagnosis &mdash; Softmax Distribution</div>
  <table><thead><tr><th>Pathology</th><th>Probability (%)</th><th style="width:40%">Confidence Bar</th></tr></thead>
  <tbody>{prob_rows}</tbody></table>

  <div class="sh">4. Visual Examination &mdash; Diagnostic Quad-Panel</div>
  <div class="quad-grid">
    <div class="quad-card"><div class="quad-label">1. Raw MRI Input</div><img src="{img_raw}"><div class="quad-cap">Original acquisition</div></div>
    <div class="quad-card"><div class="quad-label">2. WPT&rarr;LMMSE&rarr;CLAHE Enhanced</div><img src="{img_enh}"><div class="quad-cap">{cnr:+.1f}% CNR gain</div></div>
    <div class="quad-card"><div class="quad-label">3. U-Net Lesion Segmentation</div><img src="{img_seg}"><div class="quad-cap">Crimson tumour RoI / yellow contour</div></div>
    <div class="quad-card"><div class="quad-label">4. Grad-CAM XAI Attribution</div><img src="{img_gcam}"><div class="quad-cap">Model attention heatmap</div></div>
  </div>

  <div class="sh">5. Quantitative Morphometric Biomarkers</div>
  <table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>
  <tbody>{morph_rows}</tbody></table>

  <div class="sh">6. Physician's Clinical Counseling &amp; Next Steps</div>

  <p style="font-size:.75rem;font-weight:700;color:#0891b2;margin:.8rem 0 .4rem">6.1 Pathological Impression &amp; Subtype Rationale</p>
  <p style="font-size:.82rem;color:#334155;margin-bottom:1rem">{counseling['pathological_nature']}</p>

  <p style="font-size:.75rem;font-weight:700;color:#d97706;margin:.8rem 0 .4rem">6.2 Critical Patient Precautions &amp; Red Flag Warning Symptoms</p>
  <ul>{prec_items}</ul>

  <p style="font-size:.75rem;font-weight:700;color:#7c3aed;margin:.8rem 0 .4rem">6.3 Recommended Confirmatory Diagnostic Workup</p>
  <ul>{steps_items}</ul>

  <p style="font-size:.75rem;font-weight:700;color:#059669;margin:.8rem 0 .4rem">6.4 Attending Physician Consultation Checklist</p>
  <ul style="list-style:none;padding-left:0">{check_items}</ul>

  <div class="sh">7. Medical Disclaimer &amp; Regulatory Status</div>
  <div class="disc">
    <strong>&#9888; AI-Assisted Diagnostic Decision Support Tool (SaMD):</strong>
    This report is generated by an AI decision-support system and must NOT be used as a standalone
    clinical diagnosis. All findings require review and confirmation by a qualified neuro-radiologist
    or neuro-oncologist before any patient management decision. Classified as Software as a Medical
    Device (SaMD) &mdash; research and educational use only. Not CE/FDA cleared for independent diagnostic use.
  </div>

  <div class="sh">8. Verification &amp; Authorisation</div>
  <div class="sig-grid">
    <div class="sig-field"><div class="sig-label">Reporting Radiologist / Neuro-Oncologist</div><div class="sig-line">&nbsp;</div></div>
    <div class="sig-field"><div class="sig-label">Medical Registration Number</div><div class="sig-line">&nbsp;</div></div>
    <div class="sig-field"><div class="sig-label">Verification Date</div><div class="sig-line">&nbsp;</div></div>
    <div class="sig-field"><div class="sig-label">Department / Unit</div><div class="sig-line">&nbsp;</div></div>
    <div class="sig-field" style="grid-column:span 2"><div class="sig-label">Institution Stamp / Seal</div><div class="sig-line" style="min-height:2.5rem">&nbsp;</div></div>
  </div>

</div><!-- /body -->

<div class="footer">
  <span>NeuroScan AI &nbsp;|&nbsp; MRI Image Enhancement and Tumor Detection v1.0 &nbsp;|&nbsp; {institution}</span>
  <span>Report ID: {report_id} &nbsp;|&nbsp; Page 1 of 1</span>
</div>

</div><!-- /page -->
</body></html>"""


def generate_clinical_report_bytes(
    *,
    prefer_pdf: bool = True,
    **kwargs,
) -> tuple:
    """
    Convenience wrapper that generates a clinical report and returns (bytes, mime_type, ext).

    Returns a (content_bytes, mime_type, file_extension) tuple.
    - When fpdf2 is available and prefer_pdf=True: returns (pdf_bytes, "application/pdf", "pdf")
    - Otherwise: returns (html_bytes, "text/html", "html")  [print to PDF from browser]

    Parameters
    ----------
    prefer_pdf : bool  Try to generate a PDF first (default True)
    **kwargs         : All arguments forwarded to generate_pdf_report() / generate_html_clinical_report()

    Returns
    -------
    (bytes, str, str) -- (content, mime_type, file_extension)
    """
    if _FPDF_AVAILABLE and prefer_pdf:
        try:
            data = generate_pdf_report(**kwargs)
            if data[:4] == b"%PDF":
                return data, "application/pdf", "pdf"
        except Exception:
            pass
    # Fallback to HTML
    html = generate_html_clinical_report(**kwargs)
    return html.encode("utf-8"), "text/html", "html"
