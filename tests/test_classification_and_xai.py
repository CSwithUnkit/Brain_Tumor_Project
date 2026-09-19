import pytest
import torch
import numpy as np

from classification.classifier_model import BrainTumorClassifier
from classification.masking_utils import apply_hard_mask, apply_soft_mask
from explainability.gradcam_generator import BrainTumorGradCAM
from explainability.localization_eval import compute_localization_metrics, binarize_heatmap

def test_classifier_forward_pass():
    """Verify EfficientNetB2 outputs shape (B, 4)."""
    model = BrainTumorClassifier(num_classes=4, pretrained=False)
    model.eval()
    dummy_input = torch.randn(2, 3, 256, 256)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    assert output.shape == (2, 4), f"Expected (2, 4), got {output.shape}"

def test_hard_masking_behavior():
    """Verify soft context-preserving guidance and empty-mask fallback."""
    image = torch.ones(2, 3, 64, 64)
    # Batch 1: normal mask (center square)
    mask1 = torch.zeros(1, 1, 64, 64)
    mask1[:, :, 20:40, 20:40] = 1.0
    
    # Batch 2: empty mask
    mask2 = torch.zeros(1, 1, 64, 64)
    
    mask_prob = torch.cat([mask1, mask2], dim=0)
    
    masked = apply_hard_mask(image, mask_prob, padding=0.1)
    
    assert masked.shape == (2, 3, 64, 64)
    
    # Soft context-preserving: background gets 0.40 intensity (not zero)
    # Formula: I * (0.40 + 0.60 * M_prob)
    # Where mask=0 → weight=0.40, where mask=1 → weight=1.0
    assert torch.allclose(masked[0, :, 0:19, 0:19], torch.tensor(0.4), atol=1e-4), \
        "Background should be 40% intensity, not zero"
    assert torch.allclose(masked[0, :, 25:35, 25:35], torch.tensor(1.0), atol=1e-4), \
        "Tumor region should be at full (100%) intensity"
    
    # Batch 2 should be unmasked (fallback) -> entirely ones
    assert torch.all(masked[1] == 1.0), "Fallback failed for empty mask"

def test_soft_masking_behavior():
    image = torch.ones(1, 3, 64, 64)
    mask_prob = torch.zeros(1, 1, 64, 64)
    mask_prob[:, :, 32:64, 32:64] = 1.0
    
    gamma = 0.2
    masked = apply_soft_mask(image, mask_prob, gamma=gamma)
    
    # Where mask is 0, value should be gamma (0.2)
    assert torch.all(masked[:, :, 0:30, 0:30] == gamma)
    
    # Where mask is 1, value should be gamma + (1-gamma)*1 = 1.0
    assert torch.all(masked[:, :, 35:60, 35:60] == 1.0)

def test_gradcam_generation():
    """Verify Grad-CAM generates valid heatmap without runtime exceptions."""
    model = BrainTumorClassifier(num_classes=4, pretrained=False)
    gradcam = BrainTumorGradCAM(model, use_cuda=False)
    
    dummy_input = torch.randn(1, 3, 256, 256)
    heatmap = gradcam.generate_heatmap(dummy_input, target_category=1)
    
    assert heatmap.shape == (256, 256), "Heatmap shape mismatch"
    assert np.min(heatmap) >= 0.0 and np.max(heatmap) <= 1.0, "Heatmap not properly normalized"

def test_localization_metrics():
    """Test IoU/Dice quantitative evaluation."""
    gt_mask = np.zeros((10, 10))
    gt_mask[2:8, 2:8] = 1.0 # 6x6 square
    
    heatmap = np.zeros((10, 10))
    heatmap[3:9, 3:9] = 0.8 # Offset 6x6 square
    
    bin_heatmap = binarize_heatmap(heatmap, 0.5)
    metrics = compute_localization_metrics(bin_heatmap, gt_mask)
    
    # Intersection is 5x5 = 25
    # Area_h is 36, Area_gt is 36
    # Union is 36 + 36 - 25 = 47
    # IoU = 25/47 ~ 0.5319
    # Dice = 50/72 ~ 0.6944
    assert np.isclose(metrics['iou'], 25/47, atol=1e-3)
    assert np.isclose(metrics['dice'], 50/72, atol=1e-3)
