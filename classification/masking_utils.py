import torch
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def apply_hard_mask(image: torch.Tensor, mask_prob: torch.Tensor, padding: float = 0.15) -> torch.Tensor:
    """
    FR-028 / FR-029: Soft context-preserving segmentation guidance.
    Instead of zeroing out the background (hard mask), softly retains
    anatomical landmarks at 40% intensity so EfficientNet can still see
    skull base (sella turcica for pituitary) and dural margins (meningioma).

    Formula for tumor-positive samples:
        I_guided = I * (0.40 + 0.60 * M_prob)

    For empty masks (no_tumor / failed segmentation):
        Returns the original image unchanged.

    Args:
        image: (B, C, H, W)
        mask_prob: (B, 1, H, W) probabilities from U-Net
        padding: (unused, kept for API compat with callers)

    Returns:
        Context-guided image tensor (B, C, H, W)
    """
    batch_size = image.shape[0]
    guided_images = []

    for b in range(batch_size):
        img_b = image[b]               # (C, H, W)
        m_prob = mask_prob[b]           # (1, H, W)

        # Check if mask is empty — return image unchanged, no noisy warning
        if m_prob.sum() < 1e-6:
            logger.debug(f"Batch index {b}: Empty predicted mask. Returning full image.")
            guided_images.append(img_b)
            continue

        # Soft context-preserving attention: retain 40% background, boost tumor to 100%
        weight = 0.40 + 0.60 * m_prob   # (1, H, W), broadcasts to (C, H, W)
        guided_images.append(img_b * weight)

    return torch.stack(guided_images)

def apply_soft_mask(image: torch.Tensor, mask_prob: torch.Tensor, gamma: float = 0.4) -> torch.Tensor:
    """
    FR-029: Soft-masked attention-weighted variant.
    Formula: I_soft = I * (gamma + (1 - gamma) * M_prob)
    
    Args:
        image: (B, C, H, W)
        mask_prob: (B, 1, H, W) probabilities from U-Net
        gamma: Base intensity weight to retain anatomical context (default 0.4 = 40%)
        
    Returns:
        Soft-masked image tensor (B, C, H, W)
    """
    # Ensure mask_prob is broadcastable (B, 1, H, W) to (B, C, H, W)
    weight = gamma + (1.0 - gamma) * mask_prob
    soft_masked_img = image * weight
    return soft_masked_img
