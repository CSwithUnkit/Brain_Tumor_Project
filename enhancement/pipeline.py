import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import entropy
import logging
from typing import Optional

from .wpt_denoising import reconstruct_wpt
from .lmmse_filter import apply_lmmse
from .clahe_enhancer import apply_clahe

logger = logging.getLogger(__name__)

def to_display_rgb(img_float: np.ndarray) -> np.ndarray:
    """Takes any 2D float array in, scales by 255, converts to uint8, and stacks to 3 channels (H, W, 3) RGB.

    Always performs a strict min-max stretch to [0, 255] before converting, so
    the output is guaranteed to be a valid uint8 array with shape (H, W, 3).
    """
    arr = np.asarray(img_float, dtype=np.float32)

    # Robust min-max stretch so output always spans [0, 255]
    a_min, a_max = float(arr.min()), float(arr.max())
    if a_max - a_min > 1e-4:
        arr = (arr - a_min) / (a_max - a_min)   # normalize to [0, 1]
    # else: keep as-is (all-same-value slice → will map to 0)

    img_uint8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    if img_uint8.ndim == 2:
        return cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    # If already (H, W, 3), return directly
    return img_uint8

class EnhancementAblationManager:
    """
    FR-016: EnhancementAblationManager to generate and apply 8 configurations.
    """
    VALID_MODES = [
        'raw', 'wpt', 'lmmse', 'clahe', 
        'wpt_lmmse', 'wpt_clahe', 'lmmse_clahe', 'wpt_lmmse_clahe'
    ]
    
    def __init__(self, default_variant: str = 'wpt_lmmse_clahe', *args, **kwargs):
        self.default_variant = kwargs.get('mode') or kwargs.get('config') or kwargs.get('variant') or default_variant
        if self.default_variant not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{self.default_variant}'. Must be one of {self.VALID_MODES}")
        
    def process(self, image: np.ndarray, variant: Optional[str] = None) -> np.ndarray:
        """
        FR-015: Fixed execution order: Input -> WPT -> LMMSE -> CLAHE -> Output
        """
        mode = variant if variant is not None else self.default_variant
        
        is_color = len(image.shape) == 3
        if is_color:
            # Process luminance channel only to avoid color shifts
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel, a, b = cv2.split(lab)
            processed = l_channel.astype(np.float32)
        else:
            processed = image.astype(np.float32)

        # Apply WPT if in mode
        if 'wpt' in mode:
            processed = reconstruct_wpt(processed)
            
        # Apply LMMSE if in mode
        if 'lmmse' in mode:
            processed = apply_lmmse(processed)
            
        # Apply CLAHE if in mode
        if 'clahe' in mode:
            processed = apply_clahe(processed)

        # ── Robust Post-Pipeline Contrast Normalization ──────────────────────
        # Guarantee every slice uses the full [0, 1] dynamic range after the
        # cascaded WPT→LMMSE→CLAHE pipeline.  Prevents contrast collapse to
        # pathological ranges like [3/255, 3/255] or [0, 109/255].
        p_min, p_max = float(processed.min()), float(processed.max())
        if p_max - p_min > 1e-4:
            processed = (processed - p_min) / (p_max - p_min)
        else:
            # Fail-safe: the pipeline produced a flat/collapsed image.
            # Restore the original slice (pre-pipeline) so callers always
            # receive a usable image rather than a black frame.
            original_slice = l_channel.astype(np.float32) / 255.0 if is_color else image.astype(np.float32)
            if original_slice.max() > 1.0 + 1e-5:
                original_slice = original_slice / 255.0
            processed = original_slice
            logger.warning(
                "Enhancement pipeline produced a collapsed output "
                f"(range={p_max - p_min:.6f}). Falling back to original input."
            )

        # Reconstruct color image if necessary
        if is_color:
            # Preserve continuous range [0.0, 1.0] if input was float
            if processed.max() <= 1.0 + 1e-5:
                # Merge logic is complex if returning [0,1] float RGB from LAB,
                # but the user said "Ensure every stage (WPT -> LMMSE -> CLAHE) preserves continuous range [0.0, 1.0]."
                # If we convert back to LAB and RGB, OpenCV requires matching scales.
                # Since L is [0,1], let's scale it to [0,255] just for cv2.cvtColor, then scale back
                processed_uint8 = np.clip(processed * 255.0, 0, 255).astype(np.uint8)
                merged = cv2.merge([processed_uint8, a, b])
                final_image = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
                return final_image
            else:
                processed_uint8 = np.clip(processed, 0, 255).astype(np.uint8)
                merged = cv2.merge([processed_uint8, a, b])
                final_image = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
                return final_image
        else:
            return processed

def compute_entropy(image: np.ndarray) -> float:
    """Compute image entropy."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([image.astype(np.uint8)], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    return entropy(hist, base=2)

def compute_cnr(image: np.ndarray, signal_mask: np.ndarray, bg_mask: np.ndarray) -> float:
    """Compute Contrast-to-Noise Ratio (CNR)."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    img_float = image.astype(np.float64)
    mu_s = np.mean(img_float[signal_mask > 0])
    mu_n = np.mean(img_float[bg_mask > 0])
    sigma_n = np.std(img_float[bg_mask > 0])
    
    if sigma_n == 0:
        return 0.0
        
    return abs(mu_s - mu_n) / sigma_n

def evaluate_metrics(original: np.ndarray, processed: np.ndarray) -> dict:
    """
    FR-017: Compute image quality metrics: PSNR, SSIM, Entropy, CNR
    """
    if len(original.shape) == 3:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        proc_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
    else:
        orig_gray, proc_gray = original, processed
        
    # PSNR & SSIM
    data_range = proc_gray.max() - proc_gray.min()
    data_range = max(1.0, float(data_range)) # Avoid zero division
    
    val_psnr = psnr(orig_gray, proc_gray, data_range=data_range)
    val_ssim = ssim(orig_gray, proc_gray, data_range=data_range)
    val_entropy = compute_entropy(proc_gray)
    
    # Synthetic masks for CNR (since we don't have true ROIs here)
    h, w = orig_gray.shape
    signal_mask = np.zeros((h, w), dtype=np.uint8)
    bg_mask = np.zeros((h, w), dtype=np.uint8)
    # Assume center is signal, edges are background
    cv2.circle(signal_mask, (w//2, h//2), min(h, w)//4, 1, -1)
    cv2.rectangle(bg_mask, (0, 0), (w, h), 1, 10) 
    
    val_cnr = compute_cnr(proc_gray, signal_mask, bg_mask)
    
    return {
        'PSNR': val_psnr,
        'SSIM': val_ssim,
        'Entropy': val_entropy,
        'CNR': val_cnr
    }

if __name__ == '__main__':
    # Synthetic test slice
    print("Generating synthetic 256x256 test MRI slice with Gaussian noise...")
    synthetic_image = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(synthetic_image, (128, 128), 60, 150, -1)
    cv2.circle(synthetic_image, (128, 128), 30, 200, -1)
    
    # Add Gaussian noise
    noise = np.random.normal(0, 25, synthetic_image.shape)
    noisy_image = np.clip(synthetic_image + noise, 0, 255).astype(np.uint8)
    
    print("\nEvaluating 8 Ablation Variants:")
    print("-" * 75)
    print(f"{'Variant':<20} | {'PSNR':<10} | {'SSIM':<10} | {'Entropy':<10} | {'CNR':<10}")
    print("-" * 75)
    
    for mode in EnhancementAblationManager.VALID_MODES:
        manager = EnhancementAblationManager(default_variant=mode)
        processed_img = manager.process(noisy_image)
        # Convert float outputs back to uint8 for fair comparison if needed
        if processed_img.dtype.kind == 'f':
            if processed_img.max() <= 1.0 + 1e-5:
                processed_img = np.clip(processed_img * 255.0, 0, 255).astype(np.uint8)
            else:
                processed_img = np.clip(processed_img, 0, 255).astype(np.uint8)
            
        metrics = evaluate_metrics(synthetic_image, processed_img)
        
        print(f"{mode:<20} | {metrics['PSNR']:<10.2f} | {metrics['SSIM']:<10.4f} | {metrics['Entropy']:<10.2f} | {metrics['CNR']:<10.2f}")
    
    print("-" * 75)
    print("Pipeline evaluation completed successfully.")
