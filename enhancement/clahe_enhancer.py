import cv2
import numpy as np

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    FR-014: Contrast Limited Adaptive Histogram Equalization using OpenCV.
    
    Args:
        image: 2D numpy array (grayscale image, can be float or uint8)
        clip_limit: Threshold for contrast limiting (default: 2.0)
        tile_grid_size: Size of grid for histogram equalization (default: (8, 8))
        
    Returns:
        Enhanced image of the same type and range as input.
    """
    if image.dtype in [np.float32, np.float64]:
        # Ensure range is strictly [0, 255] before uint8 cast
        if image.max() <= 1.0 + 1e-5:
            img_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img_uint8)
    
    if image.dtype in [np.float32, np.float64]:
        return (enhanced.astype(np.float32) / 255.0) # Return normalized
    return enhanced
