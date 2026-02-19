"""
Inference preprocessing pipeline
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Union
import logging

logger = logging.getLogger(__name__)


class InferencePreprocessor:
    """Preprocessing for inference"""
    
    def __init__(
        self,
        image_size: int = 224,
        mean: list = [0.485, 0.456, 0.406],
        std: list = [0.229, 0.224, 0.225]
    ):
        """
        Initialize inference preprocessor
        
        Args:
            image_size: Target image size
            mean: Normalization mean
            std: Normalization std
        """
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def preprocess(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess image for inference
        
        Args:
            image: Image path, PIL Image, or numpy array
            
        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        # Load image if path provided
        if isinstance(image, str):
            try:
                image = Image.open(image).convert('RGB')
            except Exception as e:
                logger.error(f"Error loading image from {image}: {e}")
                raise
        
        # Convert numpy array to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def preprocess_batch(self, images: list) -> torch.Tensor:
        """
        Preprocess batch of images
        
        Args:
            images: List of images
            
        Returns:
            Preprocessed tensor [B, 3, H, W]
        """
        tensors = [self.preprocess(img).squeeze(0) for img in images]
        return torch.stack(tensors)
