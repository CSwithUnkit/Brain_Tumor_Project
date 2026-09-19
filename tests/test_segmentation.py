import pytest
import torch
import numpy as np
from segmentation.unet_model import UNet
from segmentation.metrics import (
    compute_dice_coefficient,
    compute_iou_score,
    compute_hausdorff_distance,
    CombinedBCEDiceLoss
)

@pytest.fixture
def dummy_batch():
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
    y = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    return x, y

def test_unet_shape_invariance(dummy_batch):
    """Test U-Net input-output tensor shape invariance."""
    x, y = dummy_batch
    model = UNet(n_channels=3, n_classes=1)
    
    out = model(x)
    assert out.shape == (2, 1, 256, 256), f"Expected (2, 1, 256, 256), got {out.shape}"

def test_metrics_perfect_prediction():
    """Test metrics with perfect prediction."""
    y = torch.ones(2, 1, 64, 64)
    # Logits > 0 implies sigmoid(logits) > 0.5 (True)
    logits = torch.ones(2, 1, 64, 64) * 10 
    
    dice = compute_dice_coefficient(logits, y)
    iou = compute_iou_score(logits, y)
    hd = compute_hausdorff_distance(logits, y)
    
    assert np.isclose(dice, 1.0, atol=1e-4)
    assert np.isclose(iou, 1.0, atol=1e-4)
    assert hd == 0.0

def test_metrics_empty_masks():
    """Test metrics with empty masks fallback."""
    y = torch.zeros(2, 1, 64, 64)
    logits = torch.ones(2, 1, 64, 64) * -10 # Predicts 0
    
    # Hausdorff distance should be gracefully handled as 0.0 for matching empty
    hd = compute_hausdorff_distance(logits, y)
    assert hd == 0.0

def test_loss_backward_pass(dummy_batch):
    """Test loss backward pass for NaN/Inf."""
    x, y = dummy_batch
    model = UNet()
    criterion = CombinedBCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    out = model(x)
    loss = criterion(out, y)
    
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    has_nan = False
    for param in model.parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                has_nan = True
                
    assert not has_nan, "Gradients contain NaN or Inf values"
