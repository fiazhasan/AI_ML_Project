"""
Training callbacks for monitoring and logging
"""

import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TensorBoardCallback:
    """TensorBoard logging callback"""
    
    def __init__(self, log_dir: str = "logs/tensorboard"):
        """
        Initialize TensorBoard callback
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        logger.info(f"TensorBoard logging to {self.log_dir}")
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float], prefix: str = ""):
        """
        Log metrics to TensorBoard
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
            prefix: Prefix for metric names
        """
        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            self.writer.add_scalar(tag, value, epoch)
    
    def log_model_graph(self, model: torch.nn.Module, sample_input: torch.Tensor):
        """
        Log model graph to TensorBoard
        
        Args:
            model: Model to log
            sample_input: Sample input tensor
        """
        try:
            self.writer.add_graph(model, sample_input)
        except Exception as e:
            logger.warning(f"Could not log model graph: {e}")
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()


class ModelCheckpoint:
    """Model checkpoint callback"""
    
    def __init__(self, save_dir: str, monitor: str = 'val_acc', mode: str = 'max'):
        """
        Initialize checkpoint callback
        
        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'max' or 'min'
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_value = float('-inf') if mode == 'max' else float('inf')
    
    def __call__(self, epoch: int, model: torch.nn.Module, metrics: Dict[str, float]):
        """
        Save checkpoint if metric improved
        
        Args:
            epoch: Current epoch
            model: Model to save
            metrics: Current metrics
        """
        if self.monitor not in metrics:
            logger.warning(f"Metric {self.monitor} not found in metrics")
            return
        
        current_value = metrics[self.monitor]
        improved = False
        
        if self.mode == 'max':
            improved = current_value > self.best_value
        else:
            improved = current_value < self.best_value
        
        if improved:
            self.best_value = current_value
            checkpoint_path = self.save_dir / f"best_model_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path} ({{self.monitor}}: {current_value:.4f})")
