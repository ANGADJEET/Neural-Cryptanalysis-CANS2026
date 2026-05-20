"""
Gohr's Residual Network for neural cryptanalysis.
Based on: "Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning" (CRYPTO 2019)

The architecture uses 1D convolutions followed by residual blocks and a prediction head.
Adapted to handle arbitrary input dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GohrResidualBlock(nn.Module):
    """Residual block with two conv layers and batch normalization."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class GohrResNet(nn.Module):
    """
    Gohr's deep residual network for differential cryptanalysis.

    Architecture (from the CRYPTO 2019 paper):
      1. Initial convolution expanding to `num_filters` channels
      2. `depth` residual blocks (each with two 3x1 convolutions)
      3. Flatten + Dense prediction head

    Handles arbitrary input_dim by treating each input feature as a
    position in a 1D sequence with 1 channel.
    """

    def __init__(
        self,
        input_dim: int,
        num_filters: int = 32,
        depth: int = 10,
    ):
        super().__init__()

        self.input_dim = input_dim

        # Treat input as (batch, 1, input_dim) — 1 channel, input_dim positions
        # Initial convolution to expand channels
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[GohrResidualBlock(num_filters) for _ in range(depth)]
        )

        # Prediction head
        flat_dim = num_filters * input_dim
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

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
        # x: (batch, input_dim) → (batch, 1, input_dim) for Conv1d
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)

        x = self.initial_conv(x)   # (batch, num_filters, input_dim)
        x = self.res_blocks(x)     # (batch, num_filters, input_dim)
        return self.head(x)
