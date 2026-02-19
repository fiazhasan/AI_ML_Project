"""
Training logic and trainer class
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Callable
import logging
from tqdm import tqdm
import time

from src.utils.helpers import get_device, count_parameters, format_time

logger = logging.getLogger(__name__)


class Trainer:
    """Model trainer with training and validation loops"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1
    ):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Training device
            use_amp: Use automatic mixed precision
            gradient_accumulation_steps: Gradient accumulation steps
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or get_device()
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Move model to device
        self.model.to(self.device)
        
        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {count_parameters(model):,}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights (with gradient accumulation)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Statistics
            running_loss += loss.item() * self.gradient_accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{running_loss/(batch_idx+1):.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                with autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f"{running_loss/(len(pbar)+1):.4f}",
                    'acc': f"{100.*correct/total:.2f}%"
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def train(
        self,
        num_epochs: int,
        early_stopping: Optional[Dict] = None,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Train model for multiple epochs
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping: Early stopping configuration
            save_path: Path to save best model
            
        Returns:
            Training history dictionary
        """
        best_val_acc = 0.0
        patience_counter = 0
        early_stopping_patience = early_stopping.get('patience', 5) if early_stopping else None
        early_stopping_min_delta = early_stopping.get('min_delta', 0.001) if early_stopping else None
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            
            # Validate
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%"
            )
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_acc': val_metrics['accuracy'],
                        'val_loss': val_metrics['loss'],
                    }, save_path)
                    logger.info(f"Saved best model (val_acc: {best_val_acc:.2f}%)")
            else:
                patience_counter += 1
            
            # Early stopping
            if early_stopping and patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        total_time = time.time() - start_time
        logger.info(f"\nTraining completed in {format_time(total_time)}")
        logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        return self.history
