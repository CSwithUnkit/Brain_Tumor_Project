"""Medical image reader for the dashboard.

Accepts the formats an MRI scan actually comes in:
  - Standard images : PNG, JPG/JPEG, BMP, TIFF, WebP  (single 2-D slice)
  - DICOM           : .dcm — applies Rescale Slope/Intercept, VOI windowing
                      (WindowCenter/WindowWidth) and MONOCHROME1 inversion;
                      multi-frame objects are returned as a volume.
  - NIfTI           : .nii / .nii.gz — 3-D volumes (4-D reduced to the middle
                      timepoint); returned as a volume for slice selection.

Every output slice is normalised to uint8 grayscale with 1–99 percentile
windowing so scans from different scanners map to a comparable range
before entering the enhancement pipeline.

Public API
----------
load_medical_image(filename, data) -> dict with keys:
    kind   : "slice" | "volume"
    image  : PIL.Image (grayscale)            -- when kind == "slice"
    volume : np.ndarray float32, shape (H, W, D) -- when kind == "volume"
    meta   : dict with format, shape, spacing, slice_count, notes
"""
from __future__ import annotations

import io
import os

import numpy as np
from PIL import Image

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
_DICOM_EXTS = {".dcm", ".dicom"}
_NIFTI_EXTS = {".nii", ".nii.gz"}


def _ext_of(filename: str) -> str:
    name = filename.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return os.path.splitext(name)[1]


def _window_normalize(arr: np.ndarray) -> np.ndarray:
    """Percentile (1–99) windowing -> uint8. Robust to outlier intensities."""
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(arr[np.isfinite(arr)], (1, 99))
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.uint8)
    out = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def _read_standard_image(data: bytes) -> dict:
    img = Image.open(io.BytesIO(data))
    img = img.convert("L")  # grayscale; pipeline works on single-channel slices
    meta = {"format": "image", "shape": (img.height, img.width),
            "slice_count": 1, "notes": f"{img.width}x{img.height} {img.mode}"}
    return {"kind": "slice", "image": img, "meta": meta}


def _apply_dicom_windowing(ds, arr: np.ndarray) -> np.ndarray:
    """Apply modality rescale + VOI LUT windowing + MONOCHROME1 inversion."""
    arr = arr.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    def _first(v):
        # WindowCenter/Width may be multi-valued; use the first.
        # Deliberate two-attempt parse: indexed access first, plain float()
        # as fallback for scalar tags. A genuinely malformed tag raises here
        # (loud) instead of silently picking a wrong window.
        try:
            return float(v[0] if len(v) > 1 else v)
        except Exception:
            return float(v)

    if "WindowCenter" in ds and "WindowWidth" in ds:
        try:
            c, w = _first(ds.WindowCenter), _first(ds.WindowWidth)
            lo, hi = c - w / 2.0, c + w / 2.0
            if hi > lo:
                arr = np.clip((arr - lo) / (hi - lo), 0, 1) * 255.0
        except Exception:
            pass  # fall through to percentile windowing

    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr  # inverted grayscale -> standard
    return arr


def _read_dicom(data: bytes, filename: str) -> dict:
    import pydicom

    ds = pydicom.dcmread(io.BytesIO(data), force=True)
    if "PixelData" not in ds:
        raise ValueError(f"{filename}: DICOM object has no pixel data.")
    arr = ds.pixel_array  # (H, W) or (Frames, H, W)
    arr = _apply_dicom_windowing(ds, arr)

    # Voxel-spacing metadata is informational only (we never convert px→mm);
    # if the tags are missing/malformed, continue without it rather than
    # rejecting an otherwise readable scan.
    spacing = None
    try:
        ps = [float(x) for x in ds.PixelSpacing]
        st = float(getattr(ds, "SliceThickness", 0.0))
        spacing = (*ps, st) if st else tuple(ps)
    except Exception:
        pass  # deliberate: spacing is optional metadata, not required

    if arr.ndim == 2:
        img = Image.fromarray(_window_normalize(arr))
        meta = {"format": "DICOM", "shape": arr.shape, "slice_count": 1,
                "spacing": spacing,
                "notes": f"{getattr(ds, 'Modality', '?')} {arr.shape[1]}x{arr.shape[0]}"}
        return {"kind": "slice", "image": img, "meta": meta}
    if arr.ndim == 3:
        # (Frames, H, W) -> (H, W, D)
        vol = np.transpose(arr, (1, 2, 0)).astype(np.float32)
        meta = {"format": "DICOM multi-frame", "shape": vol.shape,
                "slice_count": int(vol.shape[2]), "spacing": spacing,
                "notes": f"{vol.shape[2]} frames"}
        return {"kind": "volume", "volume": vol, "meta": meta}
    raise ValueError(f"{filename}: unsupported DICOM pixel shape {arr.shape}.")


def _read_nifti(data: bytes, filename: str) -> dict:
    import nibabel as nib
    import tempfile

    # nibabel works on file paths: spill to a temp file (deleted after load).
    suffix = ".nii.gz" if filename.lower().endswith(".gz") else ".nii"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        img = nib.load(tmp)
        arr = np.asarray(img.dataobj, dtype=np.float32)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    # Voxel spacing is informational only (we never convert px→mm);
    # missing/malformed headers must not reject a readable scan.
    try:
        spacing = tuple(float(x) for x in img.header.get_zooms())
    except Exception:
        spacing = None  # deliberate: spacing is optional metadata

    if arr.ndim == 4:
        arr = arr[..., arr.shape[3] // 2]  # middle timepoint
    if arr.ndim != 3:
        raise ValueError(f"{filename}: expected a 3-D NIfTI volume, got {arr.shape}.")
    # nibabel volumes are (X, Y, Z); present axial slices as (H, W, D).
    vol = np.transpose(arr, (1, 0, 2))
    meta = {"format": "NIfTI", "shape": vol.shape,
            "slice_count": int(vol.shape[2]), "spacing": spacing,
            "notes": f"{vol.shape[2]} axial slices"}
    return {"kind": "volume", "volume": vol, "meta": meta}


def load_medical_image(filename: str, data: bytes) -> dict:
    """Detect format from the filename and return a slice or volume dict.

    Raises ValueError with a human-readable message for unsupported or
    corrupt inputs (the dashboard surfaces this directly to the user).
    """
    if not data:
        raise ValueError("The uploaded file is empty.")
    ext = _ext_of(filename or "")
    try:
        if ext in _IMAGE_EXTS:
            return _read_standard_image(data)
        if ext in _DICOM_EXTS or (len(data) > 132 and data[128:132] == b"DICM"):
            # DICM magic lives at byte offset 128 (after the 128-byte preamble).
            return _read_dicom(data, filename)
        if ext in _NIFTI_EXTS:
            return _read_nifti(data, filename)
    except (ValueError, IOError, OSError) as e:
        raise ValueError(f"Could not read {filename or 'the file'}: {e}") from e
    raise ValueError(
        f"Unsupported format '{ext or '?'}'. "
        "Upload PNG, JPG, BMP, TIFF, WebP, DICOM (.dcm) or NIfTI (.nii/.nii.gz).")


def volume_slice_to_pil(volume: np.ndarray, index: int) -> Image.Image:
    """Extract one axial slice from a (H, W, D) volume as a PIL image."""
    d = int(volume.shape[2])
    idx = max(0, min(d - 1, int(index)))
    return Image.fromarray(_window_normalize(volume[:, :, idx]))
