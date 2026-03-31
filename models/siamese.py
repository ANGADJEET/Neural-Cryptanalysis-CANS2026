
import torch
import torch.nn as nn
from typing import List, Optional


class SiameseNetwork(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        encoder_type: str = 'mlp',
        encoder_dims: List[int] = [256, 128, 64],
        classifier_dims: List[int] = [64, 32],
        distance_type: str = 'concat',
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.distance_type = distance_type
        
        branch_dim = input_dim // 2
        
        if encoder_type == 'mlp':
            self.encoder = self._build_mlp_encoder(branch_dim, encoder_dims, dropout)
        else:
            self.encoder = self._build_cnn_encoder(branch_dim, encoder_dims, dropout)
        
        self.embed_dim = encoder_dims[-1]
        
        if distance_type == 'concat':
            classifier_input = 2 * self.embed_dim
        elif distance_type in ['diff', 'mult']:
            classifier_input = self.embed_dim
        elif distance_type == 'all':
            classifier_input = 4 * self.embed_dim
        else:
            classifier_input = 2 * self.embed_dim
        
        self.classifier = self._build_classifier(classifier_input, classifier_dims, dropout)
    
    def _build_mlp_encoder(self, input_dim: int, dims: List[int], dropout: float) -> nn.Module:
        layers = []
        prev_dim = input_dim
        
        for dim in dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        return nn.Sequential(*layers)
    
    def _build_cnn_encoder(self, input_dim: int, dims: List[int], dropout: float) -> nn.Module:
        layers = [
            nn.Conv1d(1, dims[0], 3, padding=1),
            nn.BatchNorm1d(dims[0]),
            nn.ReLU(),
            nn.MaxPool1d(2)
        ]
        
        for i in range(1, len(dims)):
            layers.extend([
                nn.Conv1d(dims[i-1], dims[i], 3, padding=1),
                nn.BatchNorm1d(dims[i]),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ])
        
        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())
        
        return nn.Sequential(*layers)
    
    def _build_classifier(self, input_dim: int, dims: List[int], dropout: float) -> nn.Module:
        layers = []
        prev_dim = input_dim
        
        for dim in dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        
        return nn.Sequential(*layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.encoder(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            half = x.size(1) // 2
            x1, x2 = x[:, :half], x[:, half:]
        elif x.dim() == 3:
            x1, x2 = x[:, 0], x[:, 1]
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        e1 = self.encode(x1)
        e2 = self.encode(x2)
        
        if self.distance_type == 'concat':
            combined = torch.cat([e1, e2], dim=1)
        elif self.distance_type == 'diff':
            combined = torch.abs(e1 - e2)
        elif self.distance_type == 'mult':
            combined = e1 * e2
        elif self.distance_type == 'all':
            combined = torch.cat([
                e1, e2,
                torch.abs(e1 - e2),
                e1 * e2
            ], dim=1)
        else:
            combined = torch.cat([e1, e2], dim=1)
        
        return self.classifier(combined)
    
    def get_embeddings(self, x: torch.Tensor) -> tuple:
        if x.dim() == 2:
            half = x.size(1) // 2
            x1, x2 = x[:, :half], x[:, half:]
        elif x.dim() == 3:
            x1, x2 = x[:, 0], x[:, 1]
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        return self.encode(x1), self.encode(x2)


class ContrastiveSiamese(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        encoder_dims: List[int] = [256, 128, 64],
        margin: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.margin = margin
        
        layers = []
        prev_dim = input_dim
        
        for dim in encoder_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        self.encoder = nn.Sequential(*layers)
        self.embed_dim = encoder_dims[-1]
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.encoder(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            half = x.size(1) // 2
            x1, x2 = x[:, :half], x[:, half:]
        else:
            x1, x2 = x[:, 0], x[:, 1]
        
        e1 = self.encode(x1)
        e2 = self.encode(x2)
        
        distance = torch.sqrt(torch.sum((e1 - e2) ** 2, dim=1, keepdim=True) + 1e-8)
        return distance
    
    def contrastive_loss(self, distance: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = labels * distance.pow(2) + \
               (1 - labels) * torch.clamp(self.margin - distance, min=0).pow(2)
        return loss.mean()
