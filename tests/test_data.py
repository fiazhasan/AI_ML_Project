"""
Tests for data processing modules
"""

import pytest
import torch
from pathlib import Path

from src.data.preprocessing import ImagePreprocessor
from src.data.dataset import DogBreedDataset


def test_preprocessor():
    """Test image preprocessor"""
    preprocessor = ImagePreprocessor(image_size=224)
    
    # Test transform creation
    train_transform = preprocessor.get_train_transform({'horizontal_flip': 0.5})
    val_transform = preprocessor.get_val_transform()
    
    assert train_transform is not None
    assert val_transform is not None


def test_preprocessor_normalization():
    """Test preprocessor normalization"""
    preprocessor = ImagePreprocessor(
        image_size=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    transform = preprocessor.get_val_transform()
    assert transform is not None


# Note: Dataset tests require actual data, so they're skipped if data doesn't exist
@pytest.mark.skipif(
    not Path("data/raw/Images").exists(),
    reason="Dataset not available"
)
def test_dataset_loading():
    """Test dataset loading"""
    dataset = DogBreedDataset(
        data_dir="data/raw",
        split='train'
    )
    
    assert len(dataset) > 0
    assert dataset.num_classes > 0
