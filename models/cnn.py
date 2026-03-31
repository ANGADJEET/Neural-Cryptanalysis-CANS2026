
import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class CNN(nn.Module):
    
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
        super().__init__()
        
        self.input_dim = input_dim
        self.input_channels = input_channels
        
        conv_layers = []
        in_channels = input_channels
        current_length = input_dim
        
        for filters in conv_filters:
            conv_layers.extend([
                nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
            ])
            
            if pool_type == 'max':
                conv_layers.append(nn.MaxPool1d(2))
            else:
                conv_layers.append(nn.AvgPool1d(2))
            
            current_length = current_length // 2
            in_channels = filters
        
        self.conv = nn.Sequential(*conv_layers)
        
        self.flat_size = conv_filters[-1] * current_length
        
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
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        if x.dim() == 3 and x.size(2) > x.size(1):
            pass
        elif x.dim() == 3:
            x = x.transpose(1, 2)
        
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResidualCNN(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        input_channels: int = 1,
        num_blocks: int = 3,
        base_filters: int = 32,
        fc_layers: List[int] = [128, 64],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_conv = nn.Sequential(
            nn.Conv1d(input_channels, base_filters, 3, padding=1),
            nn.BatchNorm1d(base_filters),
            nn.ReLU()
        )
        
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
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
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
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.input_conv(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        return self.fc(x)


class ResidualConvBlock(nn.Module):
    
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
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        conv_filters: List[int] = [32, 64, 128],
        fc_layers: List[int] = [128, 64],
        dropout: float = 0.1
    ):
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
