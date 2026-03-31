
import torch
import torch.nn as nn
from typing import List, Optional


class CryptoLSTM(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        fc_layers: List[int] = [64],
        dropout: float = 0.1
    ):
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
        
        lstm_output_dim = hidden_size * self.num_directions
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
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
        lstm_out, _ = self.lstm(x)
        
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        return self.classifier(context)
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        return attn_weights.squeeze(-1)


class CryptoGRU(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        fc_layers: List[int] = [64],
        dropout: float = 0.1
    ):
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
        _, hidden = self.gru(x)
        
        hidden = hidden.permute(1, 0, 2).contiguous()
        hidden = hidden.view(hidden.size(0), -1)
        
        return self.classifier(hidden)


class TransformerCrypto(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 32
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        x = self.transformer(x)
        
        x = x.mean(dim=1)
        
        return self.classifier(x)
    
    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        return []
