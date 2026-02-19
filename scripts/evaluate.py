"""
Evaluate a trained model on the test set.

Loads config and model checkpoint, runs inference on the test split,
computes metrics (accuracy, top-k, F1, etc.) and optionally runs
error analysis and saves visualizations.
"""

import sys
from pathlib import Path

# Ensure project root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging

from src.utils.config import get_config
from src.utils.logger import setup_logger
from src.utils.helpers import set_seed, get_device, load_checkpoint
from src.data.dataset import DogBreedDataset
from src.data.preprocessing import ImagePreprocessor
from src.models.model_factory import create_model
from src.evaluation.metrics import calculate_metrics, calculate_top_k_accuracy
from src.evaluation.error_analysis import ErrorAnalyzer
from src.evaluation.visualizations import plot_training_history, plot_per_class_accuracy

logger = setup_logger()


def evaluate_model(model_path: str, config_path: str = "config.yaml"):
    """Evaluate model on test set"""
    # Load configuration
    config = get_config(config_path)
    
    # Set random seed
    set_seed(config.get('data.random_seed', 42))
    
    # Setup device
    device = get_device(config.get('training.device'))
    logger.info(f"Using device: {device}")
    
    # Create preprocessor
    preprocessor = ImagePreprocessor(
        image_size=config.get('data.image_size', 224),
        mean=config.get('data.normalization.mean'),
        std=config.get('data.normalization.std')
    )
    
    # Create test dataset
    logger.info("Loading test dataset...")
    test_dataset = DogBreedDataset(
        data_dir=config.get('data.data_dir'),
        split='test',
        transform=preprocessor.get_val_transform()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('data.batch_size', 32),
        shuffle=False,
        num_workers=config.get('data.num_workers', 4)
    )
    
    # Create model
    logger.info("Loading model...")
    model_name = 'efficientnet'  # Infer from checkpoint or config
    model = create_model(
        model_name=model_name,
        num_classes=config.get('data.num_classes', 120),
        config=config['model']
    )
    
    # Load checkpoint
    checkpoint = load_checkpoint(model_path, model, device=device)
    model.eval()
    
    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Evaluate
    logger.info("Evaluating model...")
    all_predictions = []
    all_labels = []
    all_outputs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_outputs.append(outputs.cpu())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_outputs = torch.cat(all_outputs, dim=0)
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    class_names = [test_dataset.idx_to_class[i] for i in range(test_dataset.num_classes)]
    
    metrics = calculate_metrics(all_predictions, all_labels, class_names)
    
    # Top-k accuracy
    top_k_acc = calculate_top_k_accuracy(all_outputs, torch.tensor(all_labels), k_values=[1, 5])
    
    # Print results
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Top-1 Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"Top-5 Accuracy: {top_k_acc[5]:.4f} ({top_k_acc[5]*100:.2f}%)")
    logger.info(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    logger.info(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
    
    # Error analysis
    if config.get('evaluation.visualize_errors', True):
        logger.info("\nPerforming error analysis...")
        error_analyzer = ErrorAnalyzer(class_names)
        error_analysis = error_analyzer.analyze_errors(
            all_predictions,
            all_labels,
            top_k=config.get('evaluation.num_error_samples', 20)
        )
        
        logger.info(f"Total Errors: {error_analysis['total_errors']}")
        logger.info(f"Error Rate: {error_analysis['error_rate']:.4f}")
        
        # Save visualizations
        output_dir = config.get('logging.log_dir', 'logs')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        error_analyzer.plot_confusion_matrix(
            metrics['confusion_matrix'],
            save_path=f"{output_dir}/confusion_matrix.png"
        )
        
        plot_per_class_accuracy(
            metrics['f1_per_class'],
            class_names,
            save_path=f"{output_dir}/per_class_f1.png"
        )
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    logger.info("\nEvaluation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate dog breed classification model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    
    args = parser.parse_args()
    evaluate_model(args.model_path, args.config)
