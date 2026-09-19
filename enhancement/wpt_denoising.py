from typing import Optional
import cv2
import numpy as np
import pywt


def reconstruct_wpt(
    image: np.ndarray, wavelet: str = "db4", level: int = 2
) -> np.ndarray:
  """2D Multi-resolution Wavelet Sub-band Denoising (WPT/2D DWT) for Brain MRI.

  Decomposes into sub-bands, applies BayesShrink soft-thresholding strictly to
  high-frequency detail bands, and fully preserves the low-frequency
  approximation sub-band (cA) to guarantee anatomical contrast.
  """
  if image is None:
    return image

  orig_shape = image.shape[:2]
  img = image.astype(np.float32)
  if img.ndim == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  img_min = float(img.min())
  img_max = float(img.max())
  if (img_max - img_min) < 1e-5:
    return np.clip(img, 0.0, 1.0).astype(np.float32)

  # Normalize locally to [0.0, 1.0]
  norm = (img - img_min) / (img_max - img_min)

  # 2-level 2D Wavelet Sub-band Decomposition
  # coeffs[0] = cA2 (Approximation sub-band - 95% brain structure energy)
  # coeffs = (cH2, cV2, cD2)
  # coeffs = (cH1, cV1, cD1)
  coeffs = pywt.wavedec2(norm, wavelet=wavelet, level=level, mode="symmetric")

  cA = coeffs[0]  # NEVER modify approximation sub-band!
  denoised_coeffs = [cA]

  # Soft-threshold only detail sub-bands
  for detail_tuple in coeffs[1:]:
    new_details = []
    for d in detail_tuple:
      non_zero = d[np.abs(d) > 1e-4]
      if len(non_zero) > 10:
        sigma = float(np.median(np.abs(non_zero)) / 0.6745)
      else:
        sigma = 0.005

      thresh = sigma * np.sqrt(2.0 * np.log(norm.size)) * 0.15
      d_thresh = pywt.threshold(d, value=thresh, mode="soft")
      new_details.append(d_thresh)
    denoised_coeffs.append(tuple(new_details))

  # Reconstruct
  rec = pywt.waverec2(denoised_coeffs, wavelet=wavelet, mode="symmetric")
  if rec.shape != orig_shape:
    rec = cv2.resize(rec, (orig_shape[1], orig_shape[0]))

  # Restore original dynamic range and return normalized float32
  rec = rec * (img_max - img_min) + img_min
  rec = np.clip(rec, 0.0, 1.0).astype(np.float32)

  # Fail-safe check: if output collapsed, fall back to input
  if rec.max() - rec.min() < 1e-3:
    return np.clip(norm, 0.0, 1.0).astype(np.float32)

  return rec
