"""
Siamese Network for neural cryptanalysis.

Learns to compare ciphertext pairs by mapping them to an embedding
space where cipher-generated pairs cluster differently from random pairs.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class SiameseNetwork(nn.Module):
    """
    Siamese Network for pairwise comparison.
    
    Uses a shared encoder to embed both C and C', then combines
    embeddings to classify as cipher or random.
    
    Best for: R1_raw_pair representation (processes C and C' separately)
    """
    
    def __init__(
        self,
        input_dim: int,
        encoder_type: str = 'mlp',
        encoder_dims: List[int] = [256, 128, 64],
        classifier_dims: List[int] = [64, 32],
        distance_type: str = 'concat',
        dropout: float = 0.1
    ):
        """
        Initialize Siamese Network.
        
        Args:
            input_dim: Dimension of each input (C or C')
            encoder_type: 'mlp' or 'cnn'
            encoder_dims: Encoder hidden dimensions
            classifier_dims: Classifier hidden dimensions
            distance_type: How to combine embeddings:
                - 'concat': Concatenate [e1, e2]
                - 'diff': |e1 - e2|
                - 'mult': e1 * e2
                - 'all': Concatenate all above
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.distance_type = distance_type
        
        # Each branch processes half the input (C or C')
        branch_dim = input_dim // 2
        
        # Build shared encoder
        if encoder_type == 'mlp':
            self.encoder = self._build_mlp_encoder(branch_dim, encoder_dims, dropout)
        else:
            self.encoder = self._build_cnn_encoder(branch_dim, encoder_dims, dropout)
        
        self.embed_dim = encoder_dims[-1]
        
        # Calculate classifier input size based on distance type
        if distance_type == 'concat':
            classifier_input = 2 * self.embed_dim
        elif distance_type in ['diff', 'mult']:
            classifier_input = self.embed_dim
        elif distance_type == 'all':
            classifier_input = 4 * self.embed_dim  # concat + diff + mult + (e1+e2)/2
        else:
            classifier_input = 2 * self.embed_dim
        
        # Build classifier
        self.classifier = self._build_classifier(classifier_input, classifier_dims, dropout)
    
    def _build_mlp_encoder(self, input_dim: int, dims: List[int], dropout: float) -> nn.Module:
        """Build MLP encoder."""
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
        """Build 1D CNN encoder."""
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
        """Build classifier head."""
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
        """Encode a single input to embedding space."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.encoder(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input of shape (batch, 2, dim) where x[:, 0] is C and x[:, 1] is C'
               or shape (batch, 2*dim) which will be split
            
        Returns:
            Output probabilities of shape (batch, 1)
        """
        # Handle different input formats
        if x.dim() == 2:
            # Assume concatenated: split in half
            half = x.size(1) // 2
            x1, x2 = x[:, :half], x[:, half:]
        elif x.dim() == 3:
            # Shape: (batch, 2, dim)
            x1, x2 = x[:, 0], x[:, 1]
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Encode both inputs using shared encoder
        e1 = self.encode(x1)
        e2 = self.encode(x2)
        
        # Combine embeddings
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
        """Get embeddings for both inputs (for visualization)."""
        if x.dim() == 2:
            half = x.size(1) // 2
            x1, x2 = x[:, :half], x[:, half:]
        elif x.dim() == 3:
            x1, x2 = x[:, 0], x[:, 1]
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        return self.encode(x1), self.encode(x2)


class ContrastiveSiamese(nn.Module):
    """
    Siamese Network trained with contrastive loss.
    
    Instead of directly classifying, learns an embedding space where
    cipher pairs are close and random pairs are far.
    """
    
    def __init__(
        self,
        input_dim: int,
        encoder_dims: List[int] = [256, 128, 64],
        margin: float = 1.0,
        dropout: float = 0.1
    ):
        """
        Initialize Contrastive Siamese.
        
        Args:
            input_dim: Input dimension per element
            encoder_dims: Encoder hidden dimensions
            margin: Margin for contrastive loss
            dropout: Dropout rate
        """
        super().__init__()
        
        self.margin = margin
        
        # Build encoder
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
        """Encode input to embedding."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.encoder(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning distance between embeddings.
        
        Args:
            x: Shape (batch, 2, dim)
            
        Returns:
            Euclidean distance between embeddings
        """
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
        """
        Compute contrastive loss.
        
        Args:
            distance: Pairwise distances
            labels: 1 for same class (cipher), 0 for different (random)
            
        Returns:
            Contrastive loss value
        """
        # For cipher pairs (label=1): minimize distance
        # For random pairs (label=0): maximize distance (up to margin)
        loss = labels * distance.pow(2) + \
               (1 - labels) * torch.clamp(self.margin - distance, min=0).pow(2)
        return loss.mean()
