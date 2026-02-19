"""
Baseline CNN model for dog breed classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class BaselineCNN(nn.Module):
    """
    Simple CNN baseline model
    
    Architecture:
    - 3 Convolutional blocks (Conv2D → BatchNorm → ReLU → MaxPool)
    - 2 Fully connected layers
    - Dropout for regularization
    """
    
    def __init__(
        self,
        num_classes: int = 120,
        conv_channels: List[int] = [32, 64, 128],
        dropout: float = 0.5,
        fc_hidden: int = 512
    ):
        """
        Initialize baseline CNN
        
        Args:
            num_classes: Number of output classes
            conv_channels: Number of channels in each conv block
            dropout: Dropout probability
            fc_hidden: Hidden units in FC layer
        """
        super(BaselineCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional blocks
        self.conv1 = self._make_conv_block(3, conv_channels[0])
        self.conv2 = self._make_conv_block(conv_channels[0], conv_channels[1])
        self.conv3 = self._make_conv_block(conv_channels[1], conv_channels[2])
        
        # Calculate flattened size (assuming 224x224 input)
        # After 3 maxpool operations: 224 -> 112 -> 56 -> 28
        self.flattened_size = conv_channels[2] * 28 * 28
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, fc_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, num_classes)
    
    def _make_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Create a convolutional block
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            
        Returns:
            Sequential block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 3, 224, 224]
            
        Returns:
            Logits tensor [B, num_classes]
        """
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
