"""
DBitNet: Deep Bit-level Network for Neural Cryptanalysis.
Based on: Bellini et al. "Cipher-Agnostic Neural Training Pipeline with
Automated Finding of Good Input Differences" (ToSC 2023).

Architecture:
  1. Input: bit-level representation (block_size,)
  2. Reshape to (1, block_size) for 1D convolution
  3. Initial convolution expanding to `num_filters` channels
  4. N dilated residual blocks with increasing dilation (1, 2, 4, 8, ...)
     Each block has: Conv1d → BN → ReLU → Conv1d → BN + skip
     Followed by a Squeeze-and-Excitation (SE) channel attention layer
  5. Global Average Pooling
  6. Dense prediction head

The key insight from Bellini et al.: dilated convolutions capture
multi-scale bit patterns without increasing parameter count, and SE
blocks let the model dynamically weight which scales are informative.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention.
    
    Learns to recalibrate channel-wise feature responses by:
      1. Squeeze: Global average pooling (C,L) → (C,1)
      2. Excitation: FC(C/r) → ReLU → FC(C) → Sigmoid → channel weights
      3. Scale: channel_weights × input
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)  # Floor of 8 to avoid degenerate layers
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, length)
        w = self.se(x).unsqueeze(-1)  # (batch, channels, 1)
        return x * w


class DilatedResBlock(nn.Module):
    """Dilated residual block with SE attention.
    
    Two dilated Conv1d layers with BN+ReLU, residual skip connection,
    followed by SE channel attention.
    """

    def __init__(self, channels: int, dilation: int = 1, reduction: int = 4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3,
                      padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3,
                      padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.se = SqueezeExcitation(channels, reduction)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        out = self.se(out)
        return self.relu(out + residual)


class DBitNet(nn.Module):
    """DBitNet architecture for neural differential cryptanalysis.
    
    Args:
        input_dim: Number of input features (bits).
        num_filters: Number of convolution channels (default 64).
        num_blocks: Number of dilated residual blocks (default 6).
        reduction: SE reduction ratio (default 4).
        head_dims: Dense head hidden dimensions (default [128, 64]).
    """

    def __init__(
        self,
        input_dim: int,
        num_filters: int = 64,
        num_blocks: int = 6,
        reduction: int = 4,
        head_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        if head_dims is None:
            head_dims = [128, 64]

        self.input_dim = input_dim

        # Initial projection: (batch, 1, input_dim) → (batch, num_filters, input_dim)
        self.initial = nn.Sequential(
            nn.Conv1d(1, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
        )

        # Dilated residual blocks with exponentially increasing dilation
        # Dilation pattern: 1, 2, 4, 8, 1, 2, ... (wraps every 4)
        blocks = []
        for i in range(num_blocks):
            dilation = 2 ** (i % 4)  # 1, 2, 4, 8, 1, 2, ...
            blocks.append(DilatedResBlock(num_filters, dilation, reduction))
        self.res_blocks = nn.Sequential(*blocks)

        # Global Average Pooling → Dense head
        head_layers = []
        prev = num_filters
        for dim in head_dims:
            head_layers.extend([
                nn.Linear(prev, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
            ])
            prev = dim
        head_layers.extend([
            nn.Linear(prev, 1),
            nn.Sigmoid(),
        ])
        self.head = nn.Sequential(*head_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim) → (batch, 1, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.initial(x)         # (batch, num_filters, input_dim)
        x = self.res_blocks(x)      # (batch, num_filters, input_dim)

        # Global Average Pooling
        x = x.mean(dim=2)           # (batch, num_filters)

        return self.head(x)
