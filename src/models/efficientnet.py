"""
EfficientNet model with transfer learning
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B0 based classifier with transfer learning
    
    Uses pre-trained EfficientNet-B0 backbone and custom classification head
    """
    
    def __init__(
        self,
        num_classes: int = 120,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False
    ):
        """
        Initialize EfficientNet classifier
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout probability in classifier head
            freeze_backbone: Whether to freeze backbone weights
        """
        super(EfficientNetClassifier, self).__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Get feature dimension
        feature_dim = self.backbone.classifier[1].in_features
        
        # Replace classifier head
        self.backbone.classifier = nn.Identity()
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # Freeze backbone if needed
        if freeze_backbone:
            self.freeze_backbone()
    
    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 3, 224, 224]
            
        Returns:
            Logits tensor [B, num_classes]
        """
        # Extract features
        features = self.backbone(x)
        
        # Classify
        logits = self.classifier(features)
        
        return logits
