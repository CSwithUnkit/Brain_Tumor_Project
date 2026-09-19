import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

def compute_dice_coefficient(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> float:
    preds = (torch.sigmoid(preds) > 0.5).float().view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice.item()

def compute_iou_score(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> float:
    preds = (torch.sigmoid(preds) > 0.5).float().view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def compute_precision_recall(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> Tuple[float, float]:
    preds = (torch.sigmoid(preds) > 0.5).float().view(-1)
    targets = targets.view(-1)
    
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return precision.item(), recall.item()

def compute_hausdorff_distance(preds: torch.Tensor, targets: torch.Tensor, percentile: float = 95.0) -> float:
    """Compute Hausdorff Distance at the given percentile (default: HD95)."""
    preds = (torch.sigmoid(preds) > 0.5).float()
    batch_size = preds.size(0)
    total_hd = 0.0
    
    for i in range(batch_size):
        p_np = preds[i].squeeze().cpu().numpy()
        t_np = targets[i].squeeze().cpu().numpy()
        
        p_coords = np.argwhere(p_np > 0)
        t_coords = np.argwhere(t_np > 0)
        
        if len(p_coords) == 0 and len(t_coords) == 0:
            hd = 0.0
        elif len(p_coords) == 0 or len(t_coords) == 0:
            hd = np.sqrt(p_np.shape[0]**2 + p_np.shape[1]**2)
        else:
            from scipy.spatial import cKDTree
            tree_t = cKDTree(t_coords)
            d_p_to_t, _ = tree_t.query(p_coords)
            
            tree_p = cKDTree(p_coords)
            d_t_to_p, _ = tree_p.query(t_coords)
            
            all_distances = np.concatenate([d_p_to_t, d_t_to_p])
            hd = float(np.percentile(all_distances, percentile))
            
        total_hd += hd
        
    return total_hd / batch_size if batch_size > 0 else 0.0

class CombinedBCEDiceLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        
        probs = torch.sigmoid(logits).view(-1)
        flat_targets = targets.view(-1)
        intersection = (probs * flat_targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + flat_targets.sum() + self.smooth)
        dice_loss = 1 - dice
        
        return self.alpha * bce_loss + self.beta * dice_loss

class TverskyFocalLoss(nn.Module):
    """
    Tversky Focal Loss for small-tumor binary segmentation.
    alpha > beta penalizes false negatives (missed tumors) more than FPs.
    gamma provides focal modulation similar to Focal Loss.
    Ref: Salehi et al. (2017), Abraham & Khan (2019).
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3,
                 gamma: float = 0.75, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha   # weight on FN
        self.beta = beta     # weight on FP
        self.gamma = gamma   # focal exponent
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        flat_probs = probs.view(-1)
        flat_targets = targets.view(-1)
        tp = (flat_probs * flat_targets).sum()
        fn = ((1 - flat_probs) * flat_targets).sum()
        fp = (flat_probs * (1 - flat_targets)).sum()
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return (1 - tversky_index) ** self.gamma
