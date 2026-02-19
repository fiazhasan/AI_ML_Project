"""
Single-image inference from the command line.

Loads the trained model and runs prediction on one image path.
Prints top-k breed predictions with confidence scores.
"""

import sys
from pathlib import Path

# Ensure project root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path
from PIL import Image
import logging

from src.utils.config import get_config
from src.utils.logger import setup_logger
from src.models.model_factory import create_model
from src.inference.predictor import Predictor

logger = setup_logger()


def test_inference(image_path: str, model_path: str, config_path: str = "config.yaml"):
    """Test inference on single image"""
    # Load configuration
    config = get_config(config_path)
    
    # Create model
    model = create_model(
        model_name='efficientnet',
        num_classes=config.get('data.num_classes', 120),
        config=config['model']
    )
    
    # Create class names (simplified)
    num_classes = config.get('data.num_classes', 120)
    class_names = [f"Breed_{i}" for i in range(num_classes)]
    
    # Create predictor
    predictor = Predictor(
        model=model,
        model_path=model_path,
        class_names=class_names,
        device=config.get('training.device'),
        top_k=5
    )
    
    # Load and predict
    logger.info(f"Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    logger.info("Running prediction...")
    predictions = predictor.predict(image)
    
    # Print results
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    for i, pred in enumerate(predictions, 1):
        print(f"{i}. {pred['class']}: {pred['confidence']:.4f} ({pred['confidence']*100:.2f}%)")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test inference on image")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image file"
    )
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
    test_inference(args.image, args.model_path, args.config)
