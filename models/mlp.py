"""
Multi-Layer Perceptron models for neural cryptanalysis.

Includes:
- Standard MLP with configurable layers
- GohrMLP: Architecture from Gohr's seminal paper on neural cryptanalysis
"""

import torch
import torch.nn as nn
from typing import List, Optional


class MLP(nn.Module):
    """
    Configurable Multi-Layer Perceptron for binary classification.
    
    Features:
    - Configurable hidden layer sizes
    - Batch normalization
    - Dropout
    - Various activation functions
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [512, 256, 128, 64],
        dropout: float = 0.1,
        batch_norm: bool = True,
        activation: str = 'relu'
    ):
        """
        Initialize MLP.
        
        Args:
            input_dim: Input feature dimension
            hidden_layers: List of hidden layer sizes
            dropout: Dropout probability
            batch_norm: Use batch normalization
            activation: Activation function ('relu', 'gelu', 'selu')
        """
        super().__init__()
        
        self.input_dim = input_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'selu':
                layers.append(nn.SELU())
            else:
                layers.append(nn.ReLU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, input_dim) or (batch, ...)
            
        Returns:
            Output probabilities of shape (batch, 1)
        """
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        return self.network(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate features before final layer."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Forward through all but last 2 layers (Linear + Sigmoid)
        for layer in list(self.network.children())[:-2]:
            x = layer(x)
        return x


class GohrMLP(nn.Module):
    """
    MLP architecture from Gohr's neural cryptanalysis paper.
    
    Reference: "Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning"
    
    Features:
    - Specific layer configuration proven effective for SPECK
    - Batch normalization after each hidden layer
    - Designed for bit-level input representations
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.0
    ):
        """
        Initialize Gohr-style MLP.
        
        Args:
            input_dim: Input dimension (typically 32 or 64 for bit representations)
            hidden_dims: Hidden layer dimensions (default: Gohr's architecture)
            dropout: Dropout rate
        """
        super().__init__()
        
        if hidden_dims is None:
            # Gohr's original architecture
            hidden_dims = [512, 512, 256, 128, 64, 32]
        
        self.input_dim = input_dim
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # Output
        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


class ResidualMLP(nn.Module):
    """
    MLP with residual connections for deeper networks.
    
    Useful when training very deep distinguishers.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize Residual MLP.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension for all residual blocks
            num_blocks: Number of residual blocks
            dropout: Dropout rate
        """
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x)
        
        return self.output(x)


class ResidualBlock(nn.Module):
    """Single residual block for ResidualMLP."""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))
