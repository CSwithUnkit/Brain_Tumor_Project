import numpy as np
from scipy.ndimage import uniform_filter

def apply_lmmse(image: np.ndarray, kernel_size: int = 5, noise_var: float = None) -> np.ndarray:
    """
    FR-013: Linear Minimum Mean Square Error (LMMSE) spatial filter.
    
    formula: x_hat = mean_x + (var_x / (var_x + noise_var)) * (y - mean_x)
    where var_x = max(0, var_y - noise_var)
    
    Args:
        image: 2D numpy array (grayscale image)
        kernel_size: Sliding window size (default: 5)
        noise_var: Estimated noise variance. If None, estimated from homogeneous regions.
        
    Returns:
        Filtered image as a 2D numpy array of type float.
    """
    img_float = image.astype(np.float64)
    
    # Local mean (\bar{x})
    local_mean = uniform_filter(img_float, size=kernel_size)
    
    # Local variance (\sigma_y^2)
    # var(X) = E[X^2] - (E[X])^2
    local_sqr_mean = uniform_filter(img_float**2, size=kernel_size)
    local_var = np.maximum(0, local_sqr_mean - local_mean**2)
    
    if noise_var is None:
        # Estimate noise variance as the mean of the 10% lowest local variances
        # Assuming these regions are flat and mainly contain noise
        noise_var_est = np.percentile(local_var, 10)
        noise_var = noise_var_est if noise_var_est > 0 else 1e-5
        
    # Signal variance (\sigma_x^2)
    signal_var = np.maximum(0, local_var - noise_var)
    
    # Avoid division by zero
    denominator = signal_var + noise_var
    denominator[denominator < 1e-10] = 1e-10
    
    # LMMSE Filter
    ratio = signal_var / denominator
    filtered_img = local_mean + ratio * (img_float - local_mean)
    
    # Clip to input range and return as float32
    in_min = float(img_float.min())
    in_max = float(img_float.max())
    return np.clip(filtered_img, in_min, in_max).astype(np.float32)
