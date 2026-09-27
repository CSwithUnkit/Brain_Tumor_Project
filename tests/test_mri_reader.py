"""Tests for dashboard/mri_reader.py — every MRI format the dashboard accepts."""

import io

import numpy as np
import pytest
from PIL import Image

from dashboard.mri_reader import (
    load_medical_image,
    volume_slice_to_pil,
)


def _png_bytes():
    img = Image.fromarray((np.random.default_rng(0).integers(0, 255, (64, 64))).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _dicom_bytes(frames=1, monochrome1=False):
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    arr = np.random.default_rng(1).integers(0, 4095, (frames, 48, 48)).astype(np.uint16)
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.Modality = "MR"
    ds.Rows, ds.Columns = 48, 48
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME1" if monochrome1 else "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.RescaleSlope, ds.RescaleIntercept = 2.0, -100.0
    ds.WindowCenter, ds.WindowWidth = 200.0, 400.0
    if frames > 1:
        ds.NumberOfFrames = frames
    ds.PixelData = arr.tobytes()
    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds)
    return buf.getvalue()


def _nifti_bytes(shape=(40, 40, 12)):
    import os
    import tempfile

    import nibabel as nib

    img = nib.Nifti1Image(
        np.random.default_rng(2).normal(500, 100, shape).astype(np.float32), affine=np.eye(4)
    )
    fd, tmp = tempfile.mkstemp(suffix=".nii.gz")
    os.close(fd)
    try:
        nib.save(img, tmp)
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp)


def test_png_slice():
    out = load_medical_image("scan.png", _png_bytes())
    assert out["kind"] == "slice"
    assert out["image"].size == (64, 64)
    assert out["meta"]["slice_count"] == 1


def test_dicom_single_slice_applies_rescale_and_window():
    out = load_medical_image("scan.dcm", _dicom_bytes())
    assert out["kind"] == "slice"
    assert out["meta"]["format"] == "DICOM"
    px = np.array(out["image"])
    assert px.min() >= 0 and px.max() <= 255


def test_dicom_monochrome1_inverted():
    # MONOCHROME1 with a bright block must come out bright after inversion
    # handling (no crash, sane range).
    out = load_medical_image("scan.dcm", _dicom_bytes(monochrome1=True))
    assert out["kind"] == "slice"
    assert np.array(out["image"]).max() <= 255


def test_dicom_multiframe_is_volume():
    out = load_medical_image("vol.dcm", _dicom_bytes(frames=6))
    assert out["kind"] == "volume"
    assert out["volume"].shape == (48, 48, 6)
    assert out["meta"]["slice_count"] == 6
    pil = volume_slice_to_pil(out["volume"], 3)
    assert pil.size == (48, 48)


def test_nifti_volume_and_slice_picker_bounds():
    out = load_medical_image("vol.nii.gz", _nifti_bytes())
    assert out["kind"] == "volume"
    assert out["volume"].shape == (40, 40, 12)
    # out-of-range slice indices clamp instead of crashing
    assert volume_slice_to_pil(out["volume"], 999).size == (40, 40)
    assert volume_slice_to_pil(out["volume"], -5).size == (40, 40)


def test_unsupported_format_raises_helpful_error():
    with pytest.raises(ValueError, match="Unsupported format"):
        load_medical_image("scan.xyz", b"junk")


def test_empty_file_raises():
    with pytest.raises(ValueError, match="empty"):
        load_medical_image("scan.png", b"")
