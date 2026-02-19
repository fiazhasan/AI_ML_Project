"""
Error analysis utilities
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """Error analysis and visualization"""
    
    def __init__(self, class_names: List[str]):
        """
        Initialize error analyzer
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def analyze_errors(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        images: List = None,
        top_k: int = 20
    ) -> Dict:
        """
        Analyze prediction errors
        
        Args:
            predictions: Predicted labels
            labels: Ground truth labels
            images: List of image paths or tensors (optional)
            top_k: Number of top errors to return
            
        Returns:
            Dictionary with error analysis
        """
        errors = []
        
        for idx, (pred, label) in enumerate(zip(predictions, labels)):
            if pred != label:
                errors.append({
                    'index': idx,
                    'true_label': label,
                    'predicted_label': pred,
                    'true_class': self.class_names[label],
                    'predicted_class': self.class_names[pred],
                    'image': images[idx] if images else None
                })
        
        # Sort by frequency of error type
        error_types = defaultdict(int)
        for error in errors:
            error_key = (error['true_label'], error['predicted_label'])
            error_types[error_key] += 1
        
        # Get most common errors
        most_common_errors = sorted(
            error_types.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        analysis = {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(predictions),
            'errors': errors[:top_k],
            'most_common_errors': [
                {
                    'true_class': self.class_names[true],
                    'predicted_class': self.class_names[pred],
                    'count': count
                }
                for (true, pred), count in most_common_errors
            ]
        }
        
        return analysis
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        save_path: str = None,
        top_n: int = 20
    ):
        """
        Plot confusion matrix
        
        Args:
            confusion_matrix: Confusion matrix array
            save_path: Path to save plot
            top_n: Show top N classes by frequency
        """
        # Select top N classes
        class_counts = confusion_matrix.sum(axis=1)
        top_indices = np.argsort(class_counts)[-top_n:]
        
        cm_subset = confusion_matrix[np.ix_(top_indices, top_indices)]
        class_names_subset = [self.class_names[i] for i in top_indices]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_subset,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names_subset,
            yticklabels=class_names_subset
        )
        plt.title(f'Confusion Matrix (Top {top_n} Classes)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_errors(
        self,
        errors: List[Dict],
        save_dir: str = "logs/error_analysis",
        num_samples: int = 20
    ):
        """
        Visualize error samples
        
        Args:
            errors: List of error dictionaries
            save_dir: Directory to save visualizations
            num_samples: Number of samples to visualize
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Select diverse error samples
        samples = errors[:num_samples]
        
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()
        
        for idx, error in enumerate(samples):
            ax = axes[idx]
            ax.axis('off')
            
            # Show image if available
            if error.get('image') is not None:
                # Handle different image formats
                img = error['image']
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                    if img.shape[0] == 3:  # CHW format
                        img = img.transpose(1, 2, 0)
                    img = np.clip(img, 0, 1)
                
                ax.imshow(img)
            
            # Add text annotation
            title = f"True: {error['true_class']}\nPred: {error['predicted_class']}"
            ax.set_title(title, fontsize=8)
        
        plt.suptitle('Error Analysis - Misclassified Samples', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/error_samples.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Error visualizations saved to {save_dir}")
