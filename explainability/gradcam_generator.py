import torch
import cv2
import numpy as np
import logging
from typing import List
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "grad-cam", "--quiet"])
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

logger = logging.getLogger(__name__)

class BrainTumorGradCAM:
    """
    FR-033, FR-035: Grad-CAM heatmap generator for Explainability.
    """
    def __init__(self, model: torch.nn.Module, use_cuda: bool = False):
        self.model = model
        # Detect device directly from where the model parameters already live.
        # NEVER override with use_cuda flag — that would silently move a CUDA
        # model back to CPU if the caller forgot to pass use_cuda=True.
        self.device = next(model.parameters()).device
        # Ensure the model stays on the correct device (idempotent if already there)
        self.model.to(self.device)
        self.model.eval()
        
        # Target layer for EfficientNetB2
        self.target_layers = [model.model.features[-1]]
        
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)

    def generate_heatmap(self, input_tensor: torch.Tensor, target_category: int = None, target_class: int = None) -> np.ndarray:
        """
        Generate normalized activation heatmap.
        Args:
            input_tensor: (1, C, H, W) tensor
            target_category: class index to generate CAM for (None = highest scoring class)
            target_class: alias for target_category
        """
        input_tensor = input_tensor.to(self.device)
        category = target_category if target_category is not None else target_class
        targets = None if category is None else [ClassifierOutputTarget(category)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        
        # cam(..) returns a batch of heatmaps. We assume batch size 1.
        return grayscale_cam[0]

    def generate(self, input_tensor: torch.Tensor, target_category: int = None, target_class: int = None) -> np.ndarray:
        """Alias for generate_heatmap to prevent AttributeError"""
        return self.generate_heatmap(input_tensor, target_category=target_category, target_class=target_class)

    def overlay_heatmap(self, original_image_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """
        Overlays heatmap onto original MRI slice with OpenCV Jet colormap.
        original_image_rgb: (H, W, 3) float32 array in [0, 1]
        heatmap: (H, W) float32 array in [0, 1]
        """
        # Ensure image is in correct format for show_cam_on_image
        if original_image_rgb.max() > 1.0:
            original_image_rgb = original_image_rgb.astype(np.float32) / 255.0
            
        visualization = show_cam_on_image(original_image_rgb, heatmap, use_rgb=True, colormap=cv2.COLORMAP_JET)
        return visualization

    def generate_visualization_grid(self, raw: np.ndarray, enhanced: np.ndarray, 
                                    mask: np.ndarray, heatmap: np.ndarray, overlay: np.ndarray, 
                                    save_path: str = None) -> np.ndarray:
        """
        Creates and optionally saves a high-res grid of images.
        All inputs should be resized to the same spatial dimensions.
        """
        # Ensure all are 3-channel uint8 for concatenation
        def to_rgb_uint8(img):
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            return img
            
        images = [to_rgb_uint8(img) for img in [raw, enhanced, mask, heatmap, overlay]]
        grid = np.concatenate(images, axis=1) # Horizontally stack
        
        if save_path:
            # OpenCV writes in BGR
            cv2.imwrite(save_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved visualization grid to {save_path}")
            
        return grid
        
    def cleanup(self):
        """Free memory."""
        del self.cam
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
