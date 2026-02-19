"""
Local training script for dog breed classification.

Supports baseline CNN and EfficientNet. Loads config from config.yaml;
creates train/val loaders, model, optimizer, scheduler; runs training
with checkpointing and optional early stopping.
"""

import sys
from pathlib import Path

# Ensure project root is on Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler
import logging
from pathlib import Path

from src.utils.config import get_config
from src.utils.logger import setup_logger
from src.utils.helpers import set_seed, get_device, save_checkpoint
from src.data.dataset import DogBreedDataset
from src.data.preprocessing import ImagePreprocessor
from src.models.model_factory import create_model
from src.training.trainer import Trainer
from src.training.callbacks import TensorBoardCallback, ModelCheckpoint

logger = setup_logger()


def create_data_loaders(config, preprocessor):
    """Create data loaders"""
    # Create datasets
    train_dataset = DogBreedDataset(
        data_dir=config.get('data.data_dir'),
        split='train',
        transform=preprocessor.get_train_transform(config.get('data.augmentation', {}))
    )
    
    val_dataset = DogBreedDataset(
        data_dir=config.get('data.data_dir'),
        split='val',
        transform=preprocessor.get_val_transform()
    )
    
    # Create samplers for class imbalance
    train_sampler = None
    if config.get('data.use_weighted_sampling', False):
        sample_weights = train_dataset.get_sampler_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('data.batch_size', 32),
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.get('data.num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('data.batch_size', 32),
        shuffle=False,
        num_workers=config.get('data.num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset


def create_loss_function(config, train_dataset):
    """Create loss function with class weights"""
    criterion = nn.CrossEntropyLoss()
    
    if config.get('data.use_class_weights', False):
        class_weights = train_dataset.get_class_weights()
        device = get_device(config.get('training.device'))
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info("Using weighted CrossEntropyLoss")
    
    return criterion


def create_optimizer(model, config, phase: int = 1):
    """Create optimizer"""
    lr = config.get('training.learning_rate', 0.001)
    
    if phase == 2 and 'phase2' in config.get('training', {}):
        # Phase 2: Different learning rates for backbone and classifier
        backbone_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                classifier_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': config.get('training.phase2.backbone_lr', 1e-5)},
            {'params': classifier_params, 'lr': config.get('training.phase2.classifier_lr', 1e-4)}
        ], weight_decay=config.get('training.weight_decay', 0.01))
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=config.get('training.weight_decay', 0.01)
        )
    
    return optimizer


def create_scheduler(optimizer, config, num_epochs):
    """Create learning rate scheduler"""
    scheduler_type = config.get('training.scheduler', 'cosine')
    
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs
        )
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
    else:
        scheduler = None
    
    return scheduler


def train_model(model_name: str, config_path: str = "config.yaml"):
    """Main training function"""
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
    
    # Create data loaders
    logger.info("Loading datasets...")
    train_loader, val_loader, train_dataset = create_data_loaders(config, preprocessor)
    
    # Create model
    logger.info(f"Creating {model_name} model...")
    model = create_model(
        model_name=model_name,
        num_classes=config.get('data.num_classes', 120),
        config=config['model'],
        freeze_backbone=(model_name == 'efficientnet' and config.get('model.efficientnet.freeze_backbone_epochs', 0) > 0)
    )
    
    # Create loss function
    criterion = create_loss_function(config, train_dataset)
    
    # Phase 1: Train classifier only (for EfficientNet)
    if model_name == 'efficientnet' and config.get('model.efficientnet.freeze_backbone_epochs', 0) > 0:
        logger.info("Phase 1: Training classifier only...")
        freeze_epochs = config.get('model.efficientnet.freeze_backbone_epochs', 5)
        
        optimizer = create_optimizer(model, config, phase=1)
        scheduler = create_scheduler(optimizer, config, freeze_epochs)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            use_amp=config.get('training.use_amp', True),
            gradient_accumulation_steps=config.get('training.gradient_accumulation_steps', 1)
        )
        
        model_dir = Path(config.get('paths.models_dir', 'models')) / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        trainer.train(
            num_epochs=freeze_epochs,
            early_stopping=config.get('training.early_stopping'),
            save_path=str(model_dir / "phase1_best_model.pth")
        )
        
        # Unfreeze backbone
        logger.info("Unfreezing backbone for fine-tuning...")
        model.unfreeze_backbone()
    
    # Phase 2: Fine-tune entire model
    logger.info("Phase 2: Fine-tuning entire model...")
    optimizer = create_optimizer(model, config, phase=2)
    
    num_epochs = config.get('training.num_epochs', 30)
    if model_name == 'efficientnet':
        num_epochs -= config.get('model.efficientnet.freeze_backbone_epochs', 5)
    
    scheduler = create_scheduler(optimizer, config, num_epochs)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=config.get('training.use_amp', True),
        gradient_accumulation_steps=config.get('training.gradient_accumulation_steps', 1)
    )
    
    # Setup callbacks
    tensorboard_callback = TensorBoardCallback(
        log_dir=config.get('logging.tensorboard_dir', 'logs/tensorboard')
    )
    
    model_dir = Path(config.get('paths.models_dir', 'models')) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    history = trainer.train(
        num_epochs=num_epochs,
        early_stopping=config.get('training.early_stopping'),
        save_path=str(model_dir / "best_model.pth")
    )
    
    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dog breed classification model")
    parser.add_argument(
        "--model",
        type=str,
        choices=['baseline', 'efficientnet'],
        default='efficientnet',
        help="Model to train"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    
    args = parser.parse_args()
    train_model(args.model, args.config)
