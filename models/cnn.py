"""
Convolutional Neural Network models for neural cryptanalysis.

Includes:
- Standard CNN for 1D bit sequences
- Residual CNN for deeper networks
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class CNN(nn.Module):
    """
    1D Convolutional Neural Network for cryptanalysis.
    
    Designed for bit-level or word-level representations where
    spatial/sequential structure matters.
    
    Best for: R1_raw_pair, R4_bit_sliced representations
    """
    
    def __init__(
        self,
        input_dim: int,
        input_channels: int = 1,
        conv_filters: List[int] = [32, 64],
        kernel_size: int = 3,
        fc_layers: List[int] = [128, 64],
        dropout: float = 0.1,
        pool_type: str = 'max'
    ):
        """
        Initialize CNN.
        
        Args:
            input_dim: Length of input sequence (e.g., block_size)
            input_channels: Number of input channels (1 for single, 2 for pairs)
            conv_filters: Number of filters per conv layer
            kernel_size: Convolution kernel size
            fc_layers: Fully connected layer sizes
            dropout: Dropout probability
            pool_type: Pooling type ('max' or 'avg')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.input_channels = input_channels
        
        # Build convolutional layers
        conv_layers = []
        in_channels = input_channels
        current_length = input_dim
        
        for filters in conv_filters:
            conv_layers.extend([
                nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
            ])
            
            # Pooling
            if pool_type == 'max':
                conv_layers.append(nn.MaxPool1d(2))
            else:
                conv_layers.append(nn.AvgPool1d(2))
            
            current_length = current_length // 2
            in_channels = filters
        
        self.conv = nn.Sequential(*conv_layers)
        
        # Calculate flattened size
        self.flat_size = conv_filters[-1] * current_length
        
        # Build fully connected layers
        fc = []
        prev_dim = self.flat_size
        
        for dim in fc_layers:
            fc.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        fc.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        
        self.fc = nn.Sequential(*fc)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input of shape (batch, channels, length) or (batch, length)
            
        Returns:
            Output probabilities of shape (batch, 1)
        """
        # Add channel dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Ensure (batch, channels, length) format
        if x.dim() == 3 and x.size(2) > x.size(1):
            pass  # Already correct
        elif x.dim() == 3:
            x = x.transpose(1, 2)  # (batch, length, channels) -> (batch, channels, length)
        
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResidualCNN(nn.Module):
    """
    Residual CNN for deeper networks.
    
    Uses skip connections to enable training deeper architectures.
    """
    
    def __init__(
        self,
        input_dim: int,
        input_channels: int = 1,
        num_blocks: int = 3,
        base_filters: int = 32,
        fc_layers: List[int] = [128, 64],
        dropout: float = 0.1
    ):
        """
        Initialize Residual CNN.
        
        Args:
            input_dim: Input sequence length
            input_channels: Number of input channels
            num_blocks: Number of residual blocks
            base_filters: Base number of filters (doubled each block)
            fc_layers: FC layer sizes
            dropout: Dropout rate
        """
        super().__init__()
        
        # Input projection
        self.input_conv = nn.Sequential(
            nn.Conv1d(input_channels, base_filters, 3, padding=1),
            nn.BatchNorm1d(base_filters),
            nn.ReLU()
        )
        
        # Residual blocks with increasing filters
        self.blocks = nn.ModuleList()
        in_filters = base_filters
        current_length = input_dim
        
        for i in range(num_blocks):
            out_filters = base_filters * (2 ** i)
            self.blocks.append(
                ResidualConvBlock(in_filters, out_filters, downsample=(i > 0))
            )
            if i > 0:
                current_length = current_length // 2
            in_filters = out_filters
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # FC layers
        fc = []
        prev_dim = in_filters
        
        for dim in fc_layers:
            fc.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        fc.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        
        self.fc = nn.Sequential(*fc)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add channel dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.input_conv(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        return self.fc(x)


class ResidualConvBlock(nn.Module):
    """Residual block for CNN."""
    
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False):
        super().__init__()
        
        stride = 2 if downsample else 1
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        
        # Skip connection
        if in_channels != out_channels or downsample:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.skip = nn.Identity()
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.skip(x))


class CNN2D(nn.Module):
    """
    2D CNN for bit-sliced representations.
    
    Treats the ciphertext pair as a 2D image where:
    - Height: words
    - Width: bits per word
    - Channels: 2 (C and C')
    
    Best for: R4_bit_sliced representation
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int],  # (channels, height, width)
        conv_filters: List[int] = [32, 64, 128],
        fc_layers: List[int] = [128, 64],
        dropout: float = 0.1
    ):
        """
        Initialize 2D CNN.
        
        Args:
            input_shape: (channels, height, width) of input
            conv_filters: Filters per conv layer
            fc_layers: FC layer sizes
            dropout: Dropout rate
        """
        super().__init__()
        
        channels, height, width = input_shape
        
        conv_layers = []
        in_channels = channels
        
        for filters in conv_filters:
            conv_layers.extend([
                nn.Conv2d(in_channels, filters, 3, padding=1),
                nn.BatchNorm2d(filters),
                nn.ReLU(),
                nn.MaxPool2d(2, ceil_mode=True)
            ])
            height = (height + 1) // 2
            width = (width + 1) // 2
            in_channels = filters
        
        self.conv = nn.Sequential(*conv_layers)
        self.flat_size = conv_filters[-1] * height * width
        
        fc = []
        prev_dim = self.flat_size
        
        for dim in fc_layers:
            fc.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        fc.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        
        self.fc = nn.Sequential(*fc)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
