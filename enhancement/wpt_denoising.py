import cv2
import numpy as np
import pywt


def reconstruct_wpt(
    image: np.ndarray, wavelet: str = "db4", level: int = 2
) -> np.ndarray:
  """2D Multi-resolution Wavelet Sub-band Denoising (WPT/2D DWT) for Brain MRI.

  Decomposes into sub-bands, applies adaptive soft-thresholding strictly to
  high-frequency detail bands, and fully preserves the low-frequency
  approximation sub-band (cA) to guarantee anatomical contrast.

  The detail threshold is a scaled universal threshold
  ``t = 0.15 * sigma * sqrt(2 * log(N))`` where ``sigma`` is the MAD noise
  estimate of the sub-band. The 0.15 attenuation factor keeps the filter
  conservative on MRI texture.

  Returns:
      Denoised image as float32 in the **same dynamic range as the input**.
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
    # Flat image: denoising is the identity, preserve input range.
    return img.astype(np.float32)

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
      # MAD noise estimate over the FULL detail sub-band (unbiased).
      # (Previous code dropped near-zero coefficients first, which biased
      # the median upward and over-thresholded fine detail.)
      sigma = float(np.median(np.abs(d)) / 0.6745)
      sigma = max(sigma, 1e-6)

      thresh = sigma * np.sqrt(2.0 * np.log(norm.size)) * 0.15
      d_thresh = pywt.threshold(d, value=thresh, mode="soft")
      new_details.append(d_thresh)
    denoised_coeffs.append(tuple(new_details))

  # Reconstruct
  rec = pywt.waverec2(denoised_coeffs, wavelet=wavelet, mode="symmetric")
  if rec.shape != orig_shape:
    rec = cv2.resize(rec, (orig_shape[1], orig_shape[0]))

  # Restore original dynamic range. NOTE: clip bounds must be the input's
  # own [img_min, img_max] — clipping a [0, 255] restoration to [0, 1]
  # squashed every enhanced image to near-white (fixed).
  rec = rec * (img_max - img_min) + img_min
  rec = np.clip(rec, img_min, img_max).astype(np.float32)

  # Fail-safe check: if output collapsed, fall back to input
  if rec.max() - rec.min() < 1e-3:
    return np.clip(norm * (img_max - img_min) + img_min,
                   img_min, img_max).astype(np.float32)

  return rec
