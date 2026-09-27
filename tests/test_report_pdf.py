"""Regression test for the report-export bug: generate_clinical_report_bytes
must return a real PDF (not silently fall back to HTML).

Root cause (2026-09-27): bullet() used chr(8226) which Helvetica cannot encode,
so generate_pdf_report raised FPDFUnicodeEncodingException and the wrapper's
bare `except Exception: pass` silently degraded to HTML on Streamlit Cloud.
"""

import numpy as np

from reports.pdf_report_generator import (
    generate_clinical_report_bytes,
    generate_pdf_report,
)


def _demo_images(size=64):
    rng = np.random.default_rng(0)
    return {
        k: rng.integers(0, 255, (size, size, 3)).astype(np.uint8)
        for k in ("raw", "enh", "seg", "grad")
    }


def _kwargs(pred_class="Intra-axial Glial Neoplasm"):
    imgs = _demo_images()
    probs = [0.90, 0.04, 0.03, 0.03] if pred_class != "No Tumor" else [0.01, 0.01, 0.01, 0.97]
    return dict(
        patient_id="PID-TEST",
        scan_date="2026-09-27",
        sequence="T1-Weighted CE",
        institution="Test Hospital",
        pred_class=pred_class,
        pred_conf=90.0,
        probs=probs,
        area=500,
        perim=90.0,
        centroid=(32, 30),
        cnr=12.5,
        raw_img=imgs["raw"],
        enh_img=imgs["enh"],
        seg_img=imgs["seg"],
        gradcam_img=imgs["grad"],
        model_choice="Exp 2: Enhanced",
        bbox=(10, 10, 50, 50),
    )


def test_report_returns_real_pdf_for_tumor():
    data, mime, ext = generate_clinical_report_bytes(**_kwargs())
    assert ext == "pdf", f"expected PDF, got {ext} (silent HTML fallback?)"
    assert mime == "application/pdf"
    assert data[:4] == b"%PDF"


def test_report_returns_real_pdf_for_no_tumor():
    data, mime, ext = generate_clinical_report_bytes(**_kwargs("No Tumor"))
    assert ext == "pdf", f"expected PDF, got {ext} (silent HTML fallback?)"
    assert data[:4] == b"%PDF"


def test_pdf_report_direct_call():
    data = generate_pdf_report(**_kwargs())
    assert data[:4] == b"%PDF"
    assert len(data) > 10_000  # images + multi-page content embedded


def test_guidance_split_personalized_before_general():
    """Every 6.x guidance topic must split: 6.x.1 personalized, 6.x.2 general.

    The ordering and the split are a UX contract — the reader must be able to
    tell at a glance what came from their image vs what is generic.
    """
    from reports.pdf_report_generator import generate_html_clinical_report

    html = generate_html_clinical_report(**_kwargs())
    for topic, title in [
        ("6.1", "About This Finding"),
        ("6.2", "Precautions"),
        ("6.3", "Recommended Workup"),
        ("6.4", "Patient Counseling Points"),
    ]:
        assert title in html, f"{topic} '{title}' missing"
        i1 = html.find(f"{topic}.1")
        i2 = html.find(f"{topic}.2")
        assert i1 != -1 and i2 != -1, f"{topic}.1/{topic}.2 subsections missing"
        assert i1 < i2, f"{topic}.1 (personalized) must come before {topic}.2 (general)"
    # 6.1.1 carries measured values from this inference run ...
    seg_611 = html[html.find("6.1.1") : html.find("6.1.2")]
    assert "500 px2" in seg_611
    assert "(32, 30)" in seg_611
    # ... and the personalized questions embed the scan's numbers.
    seg_641 = html[html.find("6.4.1") : html.find("6.4.2")]
    assert "500 px2" in seg_641
