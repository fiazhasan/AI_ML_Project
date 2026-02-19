"""
Tests for model definitions
"""

import pytest
import torch

from src.models.baseline_cnn import BaselineCNN
from src.models.efficientnet import EfficientNetClassifier
from src.models.model_factory import create_model


def test_baseline_cnn():
    """Test baseline CNN model"""
    model = BaselineCNN(num_classes=120)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    assert output.shape == (2, 120)
    assert not torch.isnan(output).any()


def test_efficientnet():
    """Test EfficientNet model"""
    model = EfficientNetClassifier(num_classes=120, pretrained=False)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    assert output.shape == (2, 120)
    assert not torch.isnan(output).any()


def test_efficientnet_freeze():
    """Test EfficientNet freezing"""
    model = EfficientNetClassifier(num_classes=120, pretrained=False, freeze_backbone=True)
    
    # Check that backbone is frozen
    for param in model.backbone.parameters():
        assert not param.requires_grad
    
    # Check that classifier is not frozen
    for param in model.classifier.parameters():
        assert param.requires_grad


def test_model_factory():
    """Test model factory"""
    config = {
        'baseline': {'dropout': 0.5},
        'efficientnet': {'pretrained': False, 'dropout': 0.3}
    }
    
    # Test baseline creation
    baseline = create_model('baseline', 120, config)
    assert isinstance(baseline, BaselineCNN)
    
    # Test EfficientNet creation
    efficientnet = create_model('efficientnet', 120, config)
    assert isinstance(efficientnet, EfficientNetClassifier)
