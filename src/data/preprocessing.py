"""
Image preprocessing utilities
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Tuple, List, Optional


class ImagePreprocessor:
    """Image preprocessing pipeline"""
    
    def __init__(
        self,
        image_size: int = 224,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        normalize: bool = True
    ):
        """
        Initialize preprocessor
        
        Args:
            image_size: Target image size (square)
            mean: Normalization mean (ImageNet default)
            std: Normalization std (ImageNet default)
            normalize: Whether to normalize images
        """
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.normalize = normalize
    
    def get_train_transform(self, augmentation_config: dict) -> transforms.Compose:
        """
        Get training transforms with augmentation
        
        Args:
            augmentation_config: Augmentation configuration dictionary
            
        Returns:
            Composition of transforms
        """
        transform_list = [
            transforms.Resize((self.image_size + 32, self.image_size + 32)),
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip(p=augmentation_config.get('horizontal_flip', 0.5)),
            transforms.RandomRotation(degrees=augmentation_config.get('rotation', 15)),
            transforms.ColorJitter(
                brightness=augmentation_config.get('brightness', 0.2),
                contrast=augmentation_config.get('contrast', 0.2),
                saturation=augmentation_config.get('color_jitter', 0.1),
                hue=0.1
            ),
            transforms.ToTensor(),
        ]
        
        if self.normalize:
            transform_list.append(transforms.Normalize(mean=self.mean, std=self.std))
        
        return transforms.Compose(transform_list)
    
    def get_val_transform(self) -> transforms.Compose:
        """
        Get validation/test transforms (no augmentation)
        
        Returns:
            Composition of transforms
        """
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]
        
        if self.normalize:
            transform_list.append(transforms.Normalize(mean=self.mean, std=self.std))
        
        return transforms.Compose(transform_list)
    
    def preprocess_image(self, image: Image.Image, is_training: bool = False) -> torch.Tensor:
        """
        Preprocess single image
        
        Args:
            image: PIL Image
            is_training: Whether this is for training (applies augmentation)
            
        Returns:
            Preprocessed tensor
        """
        if is_training:
            transform = self.get_train_transform({})
        else:
            transform = self.get_val_transform()
        
        return transform(image)


def denormalize(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """
    Denormalize tensor for visualization
    
    Args:
        tensor: Normalized tensor [C, H, W]
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean
