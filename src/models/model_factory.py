"""
Model factory for creating models
"""

import torch.nn as nn
from typing import Dict, Any
import logging

from .baseline_cnn import BaselineCNN
from .efficientnet import EfficientNetClassifier

logger = logging.getLogger(__name__)


def create_model(
    model_name: str,
    num_classes: int,
    config: Dict[str, Any],
    **kwargs
) -> nn.Module:
    """
    Create model based on name and configuration
    
    Args:
        model_name: Name of model ('baseline_cnn' or 'efficientnet')
        num_classes: Number of output classes
        config: Model configuration dictionary
        **kwargs: Additional model-specific arguments
        
    Returns:
        Initialized model
    """
    model_name = model_name.lower()
    
    if model_name == 'baseline_cnn' or model_name == 'baseline':
        model_config = config.get('baseline', {})
        model = BaselineCNN(
            num_classes=num_classes,
            conv_channels=model_config.get('conv_channels', [32, 64, 128]),
            dropout=model_config.get('dropout', 0.5),
            fc_hidden=model_config.get('fc_hidden', 512)
        )
        logger.info("Created BaselineCNN model")
    
    elif model_name == 'efficientnet' or model_name == 'efficientnet_b0':
        model_config = config.get('efficientnet', {})
        model = EfficientNetClassifier(
            num_classes=num_classes,
            pretrained=model_config.get('pretrained', True),
            dropout=model_config.get('dropout', 0.3),
            freeze_backbone=kwargs.get('freeze_backbone', False)
        )
        logger.info("Created EfficientNetClassifier model")
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model
