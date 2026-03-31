"""
Mutual Information Neural Estimator (MINE) for cryptanalysis.

Estimates mutual information I(X; Y) between representations
and oracle labels to quantify cryptographic signal.

Reference: Belghazi et al., "MINE: Mutual Information Neural Estimation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np


class MINE(nn.Module):
    """
    Mutual Information Neural Estimator.
    
    Uses the Donsker-Varadhan representation to estimate
    mutual information between representation X and label Y.
    
    I(X; Y) >= E[T(X, Y)] - log(E[exp(T(X', Y))])
    
    where (X, Y) are joint samples and (X', Y) are marginal samples.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        estimator: str = 'dv'  # 'dv' or 'nwj'
    ):
        """
        Initialize MINE.
        
        Args:
            input_dim: Input representation dimension
            hidden_dims: Hidden layer dimensions
            estimator: MI estimation method:
                - 'dv': Donsker-Varadhan (tighter but higher variance)
                - 'nwj': Nguyen-Wainwright-Jordan (lower variance)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.estimator = estimator
        
        # Statistics network T(x, y)
        # Input: concatenation of representation x and label y
        layers = []
        prev_dim = input_dim + 1  # +1 for label
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.T = nn.Sequential(*layers)
        
        # Exponential moving average for stable training
        self.register_buffer('ema_exp', torch.tensor(1.0))
        self.ema_alpha = 0.01
    
    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing MI lower bound.
        
        Args:
            x: Input representations (batch, input_dim)
            y: Labels (batch, 1) or (batch,)
            
        Returns:
            (mi_estimate, statistics_output)
        """
        if y.dim() == 1:
            y = y.unsqueeze(1)
        
        batch_size = x.size(0)
        
        # Joint samples (x, y)
        xy_joint = torch.cat([x, y], dim=1)
        t_joint = self.T(xy_joint)
        
        # Marginal samples (x', y) - shuffle x
        perm = torch.randperm(batch_size, device=x.device)
        x_shuffled = x[perm]
        xy_marginal = torch.cat([x_shuffled, y], dim=1)
        t_marginal = self.T(xy_marginal)
        
        # MI estimation
        if self.estimator == 'dv':
            # Donsker-Varadhan
            mi = t_joint.mean() - torch.log(torch.exp(t_marginal).mean() + 1e-8)
        else:
            # NWJ (f-divergence)
            mi = t_joint.mean() - (torch.exp(t_marginal - 1)).mean()
        
        return mi, t_joint
    
    def mi_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute negative MI for training (we maximize MI, so minimize -MI).
        
        Uses exponential moving average for stable gradients.
        """
        if y.dim() == 1:
            y = y.unsqueeze(1)
        
        batch_size = x.size(0)
        
        # Joint
        xy_joint = torch.cat([x, y], dim=1)
        t_joint = self.T(xy_joint)
        
        # Marginal
        perm = torch.randperm(batch_size, device=x.device)
        xy_marginal = torch.cat([x[perm], y], dim=1)
        t_marginal = self.T(xy_marginal)
        
        if self.estimator == 'dv':
            # Use EMA for denominator
            exp_t = torch.exp(t_marginal)
            
            if self.training:
                self.ema_exp = (1 - self.ema_alpha) * self.ema_exp + self.ema_alpha * exp_t.mean().detach()
            
            # Biased gradient correction
            mi = t_joint.mean() - torch.log(exp_t.mean() + 1e-8)
        else:
            mi = t_joint.mean() - (torch.exp(t_marginal - 1)).mean()
        
        return -mi  # Negative because we minimize


class MutualInfoEstimator:
    """
    High-level interface for estimating mutual information.
    
    Handles training and evaluation of MINE for cryptanalysis experiments.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        estimator: str = 'dv',
        device: str = 'cuda'
    ):
        """
        Initialize MI estimator.
        
        Args:
            input_dim: Input dimension
            hidden_dims: MINE hidden dimensions
            estimator: 'dv' or 'nwj'
            device: Compute device
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.mine = MINE(input_dim, hidden_dims, estimator).to(self.device)
        self.optimizer = torch.optim.Adam(self.mine.parameters(), lr=0.001)
    
    def estimate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_epochs: int = 100,
        batch_size: int = 5000,
        verbose: bool = False
    ) -> float:
        """
        Estimate mutual information I(X; Y).
        
        Args:
            X: Input array (n_samples, input_dim)
            Y: Labels (n_samples,)
            n_epochs: Training epochs
            batch_size: Batch size
            verbose: Print progress
            
        Returns:
            Estimated mutual information in nats
        """
        # Convert to tensors
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        Y_tensor = torch.from_numpy(Y.astype(np.float32)).to(self.device)
        
        n_samples = len(X)
        mi_history = []
        
        for epoch in range(n_epochs):
            # Shuffle
            perm = torch.randperm(n_samples, device=self.device)
            X_shuffled = X_tensor[perm]
            Y_shuffled = Y_tensor[perm]
            
            epoch_mi = []
            
            for i in range(0, n_samples, batch_size):
                batch_x = X_shuffled[i:i+batch_size]
                batch_y = Y_shuffled[i:i+batch_size]
                
                self.optimizer.zero_grad()
                loss = self.mine.mi_loss(batch_x, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.mine.parameters(), 1.0)
                
                self.optimizer.step()
                
                epoch_mi.append(-loss.item())
            
            avg_mi = np.mean(epoch_mi)
            mi_history.append(avg_mi)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, MI estimate: {avg_mi:.4f}")
        
        # Return average of last 10 epochs
        return np.mean(mi_history[-10:])
    
    def estimate_conditional(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        n_epochs: int = 100,
        batch_size: int = 5000
    ) -> float:
        """
        Estimate conditional mutual information I(X; Y | Z).
        
        Uses chain rule: I(X; Y | Z) = I(X, Z; Y) - I(Z; Y)
        
        Args:
            X: Variable 1
            Y: Labels
            Z: Conditioning variable
            n_epochs: Training epochs
            batch_size: Batch size
            
        Returns:
            Conditional MI estimate
        """
        # I(X, Z; Y)
        XZ = np.concatenate([X, Z], axis=1)
        estimator_xz = MutualInfoEstimator(
            XZ.shape[1], self.mine.T[0].in_features - 1, 
            device=self.device
        )
        mi_xz_y = estimator_xz.estimate(XZ, Y, n_epochs, batch_size)
        
        # I(Z; Y)
        estimator_z = MutualInfoEstimator(
            Z.shape[1], device=self.device
        )
        mi_z_y = estimator_z.estimate(Z, Y, n_epochs, batch_size)
        
        return mi_xz_y - mi_z_y


class InfoNCE(nn.Module):
    """
    InfoNCE contrastive MI bound.
    
    Alternative to MINE with better stability for some cases.
    Uses contrastive learning formulation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        temperature: float = 0.1
    ):
        """
        Initialize InfoNCE estimator.
        
        Args:
            input_dim: Input dimension
            hidden_dims: Encoder hidden dimensions
            temperature: Temperature for softmax
        """
        super().__init__()
        
        self.temperature = temperature
        
        # Encoder network
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU()
            ])
            prev_dim = dim
        
        self.encoder = nn.Sequential(*layers)
        self.embed_dim = hidden_dims[-1]
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss (negative of MI lower bound).
        
        Args:
            x: Input representations (batch, input_dim)
            y: Labels (batch,)
            
        Returns:
            InfoNCE loss
        """
        # Encode
        z = self.encoder(x)
        z = F.normalize(z, dim=1)
        
        # Compute similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature
        
        # Positive pairs: same label
        labels = y.view(-1, 1)
        mask_pos = (labels == labels.t()).float()
        mask_pos.fill_diagonal_(0)  # Exclude self
        
        # Negative pairs: different label
        mask_neg = (labels != labels.t()).float()
        
        # InfoNCE loss
        exp_sim = torch.exp(sim)
        
        # For each anchor, compute log prob of positive vs all
        pos_sim = (exp_sim * mask_pos).sum(dim=1)
        neg_sim = (exp_sim * mask_neg).sum(dim=1)
        
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8) + 1e-8)
        
        return loss.mean()
