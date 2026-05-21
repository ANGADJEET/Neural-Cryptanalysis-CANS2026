"""
SENet: Squeeze-and-Excitation enhanced ResNet for Neural Cryptanalysis.
Based on: Bao et al. "Enhancing Neural Distinguishers with SE Attention" (2022).

This extends the GohrResNet architecture by adding a Squeeze-and-Excitation
block after each residual block. The SE block learns to recalibrate channel
responses, letting the network dynamically emphasize informative channels.

The key modification vs GohrResNet:
  GohrResNet:  x → ResBlock → ResBlock → ... → head
  SENet:       x → ResBlock → SE → ResBlock → SE → ... → head
"""

import torch
import torch.nn as nn
from typing import Optional


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for 1D convolutions.
    
    Squeeze:    Global Average Pool (C,L) → (C,)
    Excitation: FC(C → C/r) → ReLU → FC(C/r → C) → Sigmoid
    Scale:      channel_weights × input
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1)  # (batch, C, 1)
        return x * w


class SEResidualBlock(nn.Module):
    """Residual block with SE attention.
    
    Two Conv1d + BN layers with skip connection, followed by SE.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )
        self.se = SEBlock(channels, reduction)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_block(x)
        out = self.se(out)
        return self.relu(out + residual)


class SENet(nn.Module):
    """SE-enhanced deep residual network for differential cryptanalysis.
    
    Architecture (based on Gohr's ResNet + Bao's SE enhancement):
      1. Initial 1D convolution: (1, input_dim) → (num_filters, input_dim)
      2. `depth` SE-enhanced residual blocks
      3. Flatten + Dense prediction head (same as GohrResNet)
    
    Args:
        input_dim: Number of input features (bits).
        num_filters: Number of convolution channels (default 32).
        depth: Number of SE residual blocks (default 10).
        reduction: SE reduction ratio (default 4).
    """

    def __init__(
        self,
        input_dim: int,
        num_filters: int = 32,
        depth: int = 10,
        reduction: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim

        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
        )

        self.se_res_blocks = nn.Sequential(
            *[SEResidualBlock(num_filters, reduction) for _ in range(depth)]
        )

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
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        x = self.initial_conv(x)
        x = self.se_res_blocks(x)
        return self.head(x)
