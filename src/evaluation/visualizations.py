"""
Visualization utilities for evaluation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def plot_training_history(history: Dict, save_path: str = None):
    """
    Plot training history
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_accuracy(
    f1_scores: np.ndarray,
    class_names: List[str],
    save_path: str = None,
    top_n: int = 30
):
    """
    Plot per-class accuracy/F1 scores
    
    Args:
        f1_scores: F1 scores per class
        class_names: List of class names
        save_path: Path to save plot
        top_n: Show top N classes
    """
    # Sort by score
    sorted_indices = np.argsort(f1_scores)[::-1][:top_n]
    sorted_scores = f1_scores[sorted_indices]
    sorted_names = [class_names[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(sorted_scores)), sorted_scores)
    plt.yticks(range(len(sorted_scores)), sorted_names)
    plt.xlabel('F1 Score')
    plt.title(f'Per-Class F1 Scores (Top {top_n})')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
