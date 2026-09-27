import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def apply_soft_context_mask(image: torch.Tensor, mask_prob: torch.Tensor) -> torch.Tensor:
    """
    FR-028 / FR-029: Soft context-preserving segmentation guidance.
    Instead of zeroing out the background (hard mask), softly retains
    anatomical landmarks at 40% intensity so EfficientNet can still see
    skull base (sella turcica for pituitary) and dural margins (meningioma).

    Formula for tumor-positive samples:
        I_guided = I * (0.40 + 0.60 * M_prob)

    For empty masks (no_tumor / failed segmentation):
        Returns the original image unchanged.

    NOTE: this was previously (mis)named ``apply_hard_mask`` even though it
    never crops or zeroes anything — the name now says what it does.

    Args:
        image: (B, C, H, W)
        mask_prob: (B, 1, H, W) probabilities from U-Net

    Returns:
        Context-guided image tensor (B, C, H, W)
    """
    batch_size = image.shape[0]
    guided_images = []

    for b in range(batch_size):
        img_b = image[b]  # (C, H, W)
        m_prob = mask_prob[b]  # (1, H, W)

        # Check if mask is empty — return image unchanged, no noisy warning
        if m_prob.sum() < 1e-6:
            logger.debug(f"Batch index {b}: Empty predicted mask. Returning full image.")
            guided_images.append(img_b)
            continue

        # Soft context-preserving attention: retain 40% background, boost tumor to 100%
        weight = 0.40 + 0.60 * m_prob  # (1, H, W), broadcasts to (C, H, W)
        guided_images.append(img_b * weight)

    return torch.stack(guided_images)


def apply_hard_mask(
    image: torch.Tensor, mask_prob: torch.Tensor, padding: float = 0.15
) -> torch.Tensor:
    """
    Deprecated alias of :func:`apply_soft_context_mask` (kept for backward
    compatibility). Despite the name it never performed hard masking —
    ``padding`` is accepted and ignored.
    """
    import warnings

    warnings.warn(
        "apply_hard_mask is deprecated; use apply_soft_context_mask "
        "(the function never performed hard masking).",
        DeprecationWarning,
        stacklevel=2,
    )
    return apply_soft_context_mask(image, mask_prob)


def apply_exp3_guidance_numpy(
    enh_rgb: np.ndarray, mask_prob: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """
    NumPy twin of :func:`apply_exp3_guidance` for single-image inference
    (dashboard). MUST stay numerically identical to the torch training path:

      - guidance applied in NORMALIZED space, not uint8 space
      - 50px empty-mask guard: masks with < 50 active pixels pass through
        unmasked (a healthy scan is never darkened)

    Args:
        enh_rgb:   (H, W, 3) uint8 enhanced image
        mask_prob: (H, W) float sigmoid mask from the U-Net (same input space)
        mean/std:  (3,) ImageNet normalization constants

    Returns:
        (1, C, H, W) float32 normalized tensor, ready for the classifier.
    """
    cls_f = enh_rgb.astype(np.float32) / 255.0
    cls_norm = (cls_f - mean) / std  # (H, W, C)
    area = int((mask_prob > 0.5).sum())
    if area >= 50:
        w = 0.40 + 0.60 * mask_prob  # (H, W)
        cls_norm = cls_norm * w[..., None]
    return np.ascontiguousarray(cls_norm.transpose(2, 0, 1))[None].astype(np.float32)


def apply_soft_mask(
    image: torch.Tensor, mask_prob: torch.Tensor, gamma: float = 0.4
) -> torch.Tensor:
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
