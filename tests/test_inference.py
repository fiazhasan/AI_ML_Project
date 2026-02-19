"""
Tests for inference pipeline
"""

import pytest
import torch
from PIL import Image
import numpy as np

from src.inference.preprocessor import InferencePreprocessor
from src.inference.predictor import Predictor


def test_preprocessor():
    """Test inference preprocessor"""
    preprocessor = InferencePreprocessor()
    
    # Create dummy image
    img = Image.new('RGB', (256, 256), color='red')
    
    # Preprocess
    tensor = preprocessor.preprocess(img)
    
    assert tensor.shape == (1, 3, 224, 224)
    assert isinstance(tensor, torch.Tensor)


def test_preprocessor_batch():
    """Test batch preprocessing"""
    preprocessor = InferencePreprocessor()
    
    # Create dummy images
    images = [
        Image.new('RGB', (256, 256), color='red'),
        Image.new('RGB', (256, 256), color='blue')
    ]
    
    # Preprocess batch
    tensor = preprocessor.preprocess_batch(images)
    
    assert tensor.shape == (2, 3, 224, 224)


def test_preprocessor_numpy():
    """Test preprocessing numpy array"""
    preprocessor = InferencePreprocessor()
    
    # Create numpy image
    img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Preprocess
    tensor = preprocessor.preprocess(img_array)
    
    assert tensor.shape == (1, 3, 224, 224)
