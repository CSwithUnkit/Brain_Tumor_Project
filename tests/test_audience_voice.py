"""Audience-voice contract tests.

The clinical surfaces (dashboard guidance, clinical PDF/HTML report) are
written FOR the treating doctor. Patient voice ("ask your doctor") must never
appear there -- it belongs only in the separate patient handout.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, ".")

from reports.pdf_report_generator import (  # noqa: E402
    COUNSELING_DB,
    DOCTOR_GUIDANCE,
    _patient_handout_items,
    generate_html_clinical_report,
    generate_patient_handout_pdf,
    personalized_guidance,
)

# Phrases that address the reader as the patient. Forbidden in doctor surfaces.
_PATIENT_PHRASES = [
    "ask your doctor",
    "your doctor will",
    "your doctor can",
    "your doctor may",
    "your specialist",
    "your endocrinologist",
    "your neurosurgeon",
    "contact your doctor",
]


def _demo_images(size=64):
    rng = np.random.default_rng(0)
    return {
        k: rng.integers(0, 255, (size, size, 3)).astype(np.uint8)
        for k in ("raw", "enh", "seg", "grad")
    }


def _doctor_texts():
    texts = []
    for cls, entry in DOCTOR_GUIDANCE.items():
        for key in ("precautions", "workup", "counseling"):
            texts.extend(entry[key])
        pg = personalized_guidance(cls, 72.5, 1500, 160.0, (100, 80), (10, 10, 190, 170))
        texts += [pg["about"], pg["precautions"], pg["workup"]] + pg["questions"]
    # No-lesion path too.
    pg = personalized_guidance("No Tumor", 95.0, 0, 0.0, None, None)
    texts += [pg["about"], pg["precautions"], pg["workup"]] + pg["questions"]
    return texts


def test_doctor_guidance_covers_all_classes():
    assert set(DOCTOR_GUIDANCE) == set(COUNSELING_DB)
    for cls, entry in DOCTOR_GUIDANCE.items():
        assert set(entry) == {"precautions", "workup", "counseling"}, cls
        for key, items in entry.items():
            assert len(items) >= 4, f"{cls}.{key} too short"
            assert all(isinstance(i, str) and len(i) > 20 for i in items)


def test_doctor_voice_has_no_patient_phrasing():
    for text in _doctor_texts():
        low = text.lower()
        for phrase in _PATIENT_PHRASES:
            assert phrase not in low, f"patient phrasing in doctor voice: {phrase!r} :: {text[:80]}"


def test_clinical_html_report_has_no_patient_phrasing():
    imgs = _demo_images()
    html = generate_html_clinical_report(
        patient_id="TEST-01",
        scan_date="2026-09-27",
        sequence="T1",
        institution="Test",
        pred_class="Extra-axial Dural Lesion",
        pred_conf=36.3,
        probs=[0.1, 0.363, 0.2, 0.337],
        area=1772,
        perim=168.4,
        centroid=(172, 92),
        cnr=12.5,
        raw_img=imgs["raw"],
        enh_img=imgs["enh"],
        seg_img=imgs["seg"],
        gradcam_img=imgs["grad"],
    )
    low = html.lower()
    assert "6.4 patient counseling points" in low
    for phrase in _PATIENT_PHRASES:
        assert phrase not in low, f"patient phrasing leaked into clinical report: {phrase!r}"


def test_personalized_guidance_audience_contract():
    doc = personalized_guidance("Extra-axial Dural Lesion", 36.3, 1772, 168.4, (172, 92), None)
    pat = personalized_guidance(
        "Extra-axial Dural Lesion", 36.3, 1772, 168.4, (172, 92), None, audience="patient"
    )
    # Same keys, different voice.
    assert set(doc) == set(pat) == {"about", "precautions", "workup", "questions"}
    assert doc["questions"] != pat["questions"]
    # Doctor 6.4.1 items embed this scan's numbers (counseling points, not questions).
    joined = " ".join(doc["questions"])
    assert "1,772 px2" in joined
    assert "36.3%" in joined
    # Patient 6.4.1 items are questions for the doctor.
    assert any("my scan" in q for q in pat["questions"])
    with pytest.raises(ValueError):
        personalized_guidance("No Tumor", 90.0, 0, 0.0, None, None, audience="nurse")


def test_patient_handout_pdf_valid():
    data = generate_patient_handout_pdf(
        patient_id="TEST-01",
        scan_date="2026-09-27",
        pred_class="Extra-axial Dural Lesion",
        pred_conf=36.3,
        area=1772,
    )
    assert data[:4] == b"%PDF"
    assert len(data) > 3_000  # text-only handout, no embedded images


def test_patient_handout_items_patient_voiced():
    items = _patient_handout_items("Extra-axial Dural Lesion", 36.3, 1772)
    assert items["plain"] == "Meningioma"
    assert "usually benign" in items["explain"]
    # Patient checklist = scan-specific questions + "ask your doctor" checklist.
    joined = " ".join(items["questions"]).lower()
    assert "ask your doctor" in joined or "ask whether" in joined
    assert len(items["precautions"]) >= 4
    # No-tumor handout works too.
    items_nt = _patient_handout_items("No Tumor", 95.0, 0)
    assert items_nt["plain"] == "No tumor detected"
    assert len(items_nt["questions"]) >= 4
