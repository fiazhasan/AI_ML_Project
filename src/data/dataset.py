"""
Dataset class for dog breed classification
"""

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
import pandas as pd
from pathlib import Path
from typing import Optional, Callable, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class DogBreedDataset(Dataset):
    """Dataset class for Stanford Dogs"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        annotations_file: Optional[str] = None
    ):
        """
        Initialize dataset
        
        Args:
            data_dir: Root directory containing images
            split: Dataset split ('train', 'val', 'test')
            transform: Image transforms
            annotations_file: Path to annotations file (optional)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load image paths and labels
        self.samples = self._load_samples(annotations_file)
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self, annotations_file: Optional[str]) -> list:
        """
        Load image paths and labels
        
        Args:
            annotations_file: Path to annotations file
            
        Returns:
            List of (image_path, label) tuples
        """
        samples = []
        
        # Stanford Dogs structure: Images/{class_name}/{image_name}.jpg
        images_dir = self.data_dir / "Images"
        
        if not images_dir.exists():
            logger.warning(f"Images directory not found: {images_dir}")
            return samples
        
        # Get all class directories
        class_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
        
        # Create label to index mapping
        self.class_to_idx = {cls.name: idx for idx, cls in enumerate(class_dirs)}
        self.idx_to_class = {idx: cls.name for idx, cls in enumerate(class_dirs)}
        self.num_classes = len(class_dirs)
        
        # Load all samples
        all_samples = []
        for class_dir in class_dirs:
            class_name = class_dir.name
            label = self.class_to_idx[class_name]
            
            # Get all images in this class
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            
            for img_path in image_files:
                all_samples.append((str(img_path), label))
        
        # Split dataset if needed
        if self.split in ['train', 'val', 'test']:
            # Use stratified split to maintain class distribution
            labels = [s[1] for s in all_samples]
            paths = [s[0] for s in all_samples]
            
            # First split: train vs (val+test)
            train_paths, temp_paths, train_labels, temp_labels = train_test_split(
                paths, labels, test_size=0.3, stratify=labels, random_state=42
            )
            
            # Second split: val vs test
            val_paths, test_paths, val_labels, test_labels = train_test_split(
                temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
            )
            
            if self.split == 'train':
                samples = list(zip(train_paths, train_labels))
            elif self.split == 'val':
                samples = list(zip(val_paths, val_labels))
            else:  # test
                samples = list(zip(test_paths, test_labels))
        else:
            # Use all samples
            samples = all_samples
        
        return samples
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get sample by index
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, label)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            # Handle both torchvision transforms and albumentations
            if hasattr(self.transform, '__call__'):
                try:
                    # Try albumentations format first
                    if 'image' in str(type(self.transform)):
                        image = self.transform(image=np.array(image))['image']
                    else:
                        # torchvision transforms
                        image = self.transform(image)
                except:
                    # Fallback to torchvision
                    image = self.transform(image)
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced dataset
        
        Returns:
            Tensor of class weights
        """
        labels = [s[1] for s in self.samples]
        class_counts = np.bincount(labels)
        total = len(labels)
        
        # Inverse frequency weighting
        class_weights = total / (self.num_classes * class_counts)
        return torch.FloatTensor(class_weights)
    
    def get_sampler_weights(self) -> torch.Tensor:
        """
        Get sample weights for WeightedRandomSampler
        
        Returns:
            Tensor of sample weights
        """
        labels = [s[1] for s in self.samples]
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label] for label in labels]
        return torch.FloatTensor(sample_weights)
