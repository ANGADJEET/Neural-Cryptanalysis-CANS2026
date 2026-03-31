"""
Recurrent Neural Network models for neural cryptanalysis.

For sequential/round-wise representations (R7_sequential).
Captures dependencies across cipher rounds.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class CryptoLSTM(nn.Module):
    """
    LSTM for sequential round differences.
    
    Processes round-by-round differences to capture temporal
    patterns in differential propagation.
    
    Best for: R7_sequential representation
    """
    
    def __init__(
        self,
        input_dim: int,  # bits per round
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        fc_layers: List[int] = [64],
        dropout: float = 0.1
    ):
        """
        Initialize LSTM.
        
        Args:
            input_dim: Input dimension per timestep (block_size)
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            bidirectional: Use bidirectional LSTM
            fc_layers: FC layer sizes after LSTM
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer (optional but improves performance)
        lstm_output_dim = hidden_size * self.num_directions
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Classifier
        fc = []
        prev_dim = lstm_output_dim
        
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
        
        self.classifier = nn.Sequential(*fc)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input of shape (batch, seq_len, input_dim)
               where seq_len is number of rounds
            
        Returns:
            Output probabilities of shape (batch, 1)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*directions)
        
        # Attention mechanism
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*directions)
        
        return self.classifier(context)
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for interpretability."""
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        return attn_weights.squeeze(-1)


class CryptoGRU(nn.Module):
    """
    GRU variant for sequential round differences.
    
    Lighter than LSTM, often performs similarly for cryptanalysis.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        fc_layers: List[int] = [64],
        dropout: float = 0.1
    ):
        """
        Initialize GRU.
        
        Args:
            input_dim: Input dimension per timestep
            hidden_size: GRU hidden state size
            num_layers: Number of GRU layers
            bidirectional: Use bidirectional GRU
            fc_layers: FC layer sizes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Use final hidden state
        fc = []
        prev_dim = hidden_size * self.num_directions * num_layers
        
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
        
        self.classifier = nn.Sequential(*fc)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        _, hidden = self.gru(x)  # hidden: (num_layers*directions, batch, hidden)
        
        # Concatenate all hidden states
        hidden = hidden.permute(1, 0, 2).contiguous()  # (batch, layers*dirs, hidden)
        hidden = hidden.view(hidden.size(0), -1)  # (batch, layers*dirs*hidden)
        
        return self.classifier(hidden)


class TransformerCrypto(nn.Module):
    """
    Transformer encoder for round-wise differences.
    
    Uses self-attention to capture long-range dependencies
    across all rounds simultaneously.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 32  # Maximum rounds
    ):
        """
        Initialize Transformer.
        
        Args:
            input_dim: Input dimension per round
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input of shape (batch, seq_len, input_dim)
            
        Returns:
            Output probabilities
        """
        batch_size, seq_len, _ = x.shape
        
        # Project and add positional encoding
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Use CLS-style aggregation (mean over sequence)
        x = x.mean(dim=1)
        
        return self.classifier(x)
    
    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get attention maps from all layers (for visualization).
        
        Returns list of attention tensors of shape (batch, heads, seq, seq).
        """
        # This requires modifying the transformer to store attention weights
        # For now, return empty list - can be implemented with hooks
        return []
