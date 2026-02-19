"""
Advanced data augmentation using Albumentations
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any


def get_augmentation_pipeline(config: Dict[str, Any], is_training: bool = True) -> A.Compose:
    """
    Get augmentation pipeline using Albumentations
    
    Args:
        config: Augmentation configuration
        is_training: Whether this is for training
        
    Returns:
        Albumentations compose object
    """
    image_size = config.get('image_size', 224)
    aug_config = config.get('augmentation', {})
    
    if is_training:
        # Training augmentations
        transform_list = [
            A.Resize(image_size + 32, image_size + 32),
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=aug_config.get('horizontal_flip', 0.5)),
            A.Rotate(limit=aug_config.get('rotation', 15), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=aug_config.get('brightness', 0.2),
                contrast_limit=aug_config.get('contrast', 0.2),
                p=0.5
            ),
            A.ColorJitter(
                brightness=aug_config.get('brightness', 0.2),
                contrast=aug_config.get('contrast', 0.2),
                saturation=aug_config.get('color_jitter', 0.1),
                hue=0.1,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        ]
    else:
        # Validation/test - only resize
        transform_list = [
            A.Resize(image_size, image_size),
        ]
    
    # Normalization (ImageNet stats)
    transform_list.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    return A.Compose(transform_list)
