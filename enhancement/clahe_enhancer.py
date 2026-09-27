import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    FR-014: Contrast Limited Adaptive Histogram Equalization using OpenCV.

    Args:
        image: 2D numpy array (grayscale image, can be float or uint8)
        clip_limit: Threshold for contrast limiting (default: 2.0)
        tile_grid_size: Size of grid for histogram equalization (default: (8, 8))

    Returns:
        Enhanced image with the same dtype and dynamic range as the input
        (previous versions silently remapped float [0, 255] inputs to [0, 1]).
    """
    is_float = image.dtype in (np.float32, np.float64)
    if is_float:
        in_min, in_max = float(image.min()), float(image.max())
        span = in_max - in_min
        if span < 1e-8:
            # Flat image: CLAHE is the identity.
            return image.astype(np.float32)
        # Map to the uint8 working range OpenCV CLAHE requires.
        img_uint8 = np.clip((image - in_min) / span * 255.0, 0, 255).astype(np.uint8)
    else:
        img_uint8 = np.clip(image, 0, 255).astype(np.uint8)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img_uint8)

    if is_float:
        # Map back to the input's original dynamic range.
        out = enhanced.astype(np.float32) / 255.0 * span + in_min
        return np.clip(out, in_min, in_max).astype(np.float32)
    return enhanced
