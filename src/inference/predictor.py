"""
Prediction pipeline
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np
import logging

from src.utils.helpers import get_device, load_checkpoint
from src.inference.preprocessor import InferencePreprocessor

logger = logging.getLogger(__name__)


class Predictor:
    """Model predictor for inference"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        model_path: str,
        class_names: List[str],
        device: str = None,
        top_k: int = 5
    ):
        """
        Initialize predictor
        
        Args:
            model: Model instance
            model_path: Path to model checkpoint
            class_names: List of class names
            device: Device to run inference on
            top_k: Number of top predictions to return
        """
        self.device = get_device(device)
        self.class_names = class_names
        self.top_k = top_k
        self.preprocessor = InferencePreprocessor()
        
        # Load model
        self.model = model.to(self.device)
        checkpoint = load_checkpoint(model_path, self.model, device=self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Running inference on {self.device}")
    
    def predict(self, image) -> List[Dict]:
        """
        Predict class for single image
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            
        Returns:
            List of predictions with confidence scores
        """
        # Preprocess
        tensor = self.preprocessor.preprocess(image).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_probs, top_indices = probabilities.topk(self.top_k, dim=1)
        
        # Format results
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            predictions.append({
                'class': self.class_names[idx.item()],
                'class_index': idx.item(),
                'confidence': prob.item()
            })
        
        return predictions
    
    def predict_batch(self, images: List) -> List[List[Dict]]:
        """
        Predict classes for batch of images
        
        Args:
            images: List of input images
            
        Returns:
            List of prediction lists
        """
        # Preprocess batch
        tensor = self.preprocessor.preprocess_batch(images).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_probs, top_indices = probabilities.topk(self.top_k, dim=1)
        
        # Format results
        batch_predictions = []
        for probs, indices in zip(top_probs, top_indices):
            predictions = []
            for prob, idx in zip(probs, indices):
                predictions.append({
                    'class': self.class_names[idx.item()],
                    'class_index': idx.item(),
                    'confidence': prob.item()
                })
            batch_predictions.append(predictions)
        
        return batch_predictions
