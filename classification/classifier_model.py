import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from typing import Optional

class BrainTumorClassifier(nn.Module):
    """
    FR-025: EfficientNetB2 classifier for 4 classes: 
    ['glioma', 'meningioma', 'pituitary', 'no_tumor']
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        
        if pretrained:
            self.model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
        else:
            self.model = efficientnet_b2(weights=None)
            
        # Get the input features for the classifier head
        in_features = self.model.classifier[1].in_features
        
        # Modify the classification head for 4 classes
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def freeze_feature_extractor(self, freeze: bool = True) -> None:
        """
        Provides method to freeze/unfreeze the feature extractor (all layers except classifier).
        """
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = not freeze
                
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the layer before the classifier.
        """
        return self.model.features(x)
