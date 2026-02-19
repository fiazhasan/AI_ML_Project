"""
Evaluation metrics
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor, top_k: int = 1) -> float:
    """
    Calculate top-k accuracy
    
    Args:
        outputs: Model outputs [B, num_classes]
        labels: Ground truth labels [B]
        top_k: Top-k accuracy (1 or 5)
        
    Returns:
        Accuracy value
    """
    _, pred = outputs.topk(top_k, dim=1)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    correct_k = correct[:top_k].reshape(-1).float().sum(0)
    return (correct_k / labels.size(0)).item()


def calculate_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: List[str]
) -> Dict:
    """
    Calculate comprehensive metrics
    
    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        class_names: List of class names
        
    Returns:
        Dictionary of metrics
    """
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    
    # F1 scores
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    f1_per_class = f1_score(labels, predictions, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Classification report
    report = classification_report(
        labels, predictions,
        target_names=class_names,
        output_dict=True
    )
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return metrics


def calculate_top_k_accuracy(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    k_values: List[int] = [1, 5]
) -> Dict[int, float]:
    """
    Calculate top-k accuracies for multiple k values
    
    Args:
        outputs: Model outputs [B, num_classes]
        labels: Ground truth labels [B]
        k_values: List of k values
        
    Returns:
        Dictionary mapping k to accuracy
    """
    results = {}
    for k in k_values:
        acc = calculate_accuracy(outputs, labels, top_k=k)
        results[k] = acc
    return results
