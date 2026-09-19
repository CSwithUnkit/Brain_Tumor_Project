import json
import logging
import os
import numpy as np
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def binarize_heatmap(heatmap: np.ndarray, threshold: float) -> np.ndarray:
    """Binarize Grad-CAM heatmap at a given activation threshold."""
    return (heatmap >= threshold).astype(np.float32)

def compute_localization_metrics(binarized_heatmap: np.ndarray, ground_truth_mask: np.ndarray) -> Dict[str, float]:
    """
    FR-034: Computes quantitative IoU and Dice overlap against BRISC ground-truth tumor masks.
    """
    intersection = np.sum(binarized_heatmap * ground_truth_mask)
    area_h = np.sum(binarized_heatmap)
    area_gt = np.sum(ground_truth_mask)
    
    union = area_h + area_gt - intersection
    
    # Smooth to avoid zero division
    smooth = 1e-6
    iou = (intersection + smooth) / (union + smooth)
    dice = (2.0 * intersection + smooth) / (area_h + area_gt + smooth)
    
    return {'iou': float(iou), 'dice': float(dice)}

class LocalizationEvaluator:
    """
    FR-036, FR-037, FR-038: Evaluates models side-by-side to determine 
    if enhancement or segmentation guidance improves localization quality.
    """
    def __init__(self, thresholds: List[float] = [0.3, 0.5, 0.7]):
        self.thresholds = thresholds
        self.results = {}
        
    def evaluate_sample(self, experiment_name: str, heatmap: np.ndarray, gt_mask: np.ndarray):
        """Accumulate metrics for a specific experiment variant on a single sample."""
        if experiment_name not in self.results:
            self.results[experiment_name] = {str(t): {'iou': [], 'dice': []} for t in self.thresholds}
            
        for t in self.thresholds:
            bin_h = binarize_heatmap(heatmap, t)
            metrics = compute_localization_metrics(bin_h, gt_mask)
            
            self.results[experiment_name][str(t)]['iou'].append(metrics['iou'])
            self.results[experiment_name][str(t)]['dice'].append(metrics['dice'])
            
    def summarize_and_save(self, save_path: str = 'results/gradcam_localization_summary.json'):
        """Computes means and saves statistics to JSON."""
        summary = {}
        for exp, thresh_data in self.results.items():
            summary[exp] = {}
            for t_str, metrics in thresh_data.items():
                summary[exp][t_str] = {
                    'mean_iou': float(np.mean(metrics['iou'])),
                    'std_iou': float(np.std(metrics['iou'])),
                    'mean_dice': float(np.mean(metrics['dice'])),
                    'std_dice': float(np.std(metrics['dice']))
                }
                
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=4)
            
        logger.info(f"Saved localization summary to {save_path}")
        return summary
