
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np


class MINE(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        estimator: str = 'dv'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.estimator = estimator
        
        layers = []
        prev_dim = input_dim + 1
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.T = nn.Sequential(*layers)
        
        self.register_buffer('ema_exp', torch.tensor(1.0))
        self.ema_alpha = 0.01
    
    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if y.dim() == 1:
            y = y.unsqueeze(1)
        
        batch_size = x.size(0)
        
        xy_joint = torch.cat([x, y], dim=1)
        t_joint = self.T(xy_joint)
        
        perm = torch.randperm(batch_size, device=x.device)
        x_shuffled = x[perm]
        xy_marginal = torch.cat([x_shuffled, y], dim=1)
        t_marginal = self.T(xy_marginal)
        
        if self.estimator == 'dv':
            mi = t_joint.mean() - torch.log(torch.exp(t_marginal).mean() + 1e-8)
        else:
            mi = t_joint.mean() - (torch.exp(t_marginal - 1)).mean()
        
        return mi, t_joint
    
    def mi_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.dim() == 1:
            y = y.unsqueeze(1)
        
        batch_size = x.size(0)
        
        xy_joint = torch.cat([x, y], dim=1)
        t_joint = self.T(xy_joint)
        
        perm = torch.randperm(batch_size, device=x.device)
        xy_marginal = torch.cat([x[perm], y], dim=1)
        t_marginal = self.T(xy_marginal)
        
        if self.estimator == 'dv':
            exp_t = torch.exp(t_marginal)
            
            if self.training:
                self.ema_exp = (1 - self.ema_alpha) * self.ema_exp + self.ema_alpha * exp_t.mean().detach()
            
            mi = t_joint.mean() - torch.log(exp_t.mean() + 1e-8)
        else:
            mi = t_joint.mean() - (torch.exp(t_marginal - 1)).mean()
        
        return -mi


class MutualInfoEstimator:
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        estimator: str = 'dv',
        device: str = 'cuda'
    ):
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
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        Y_tensor = torch.from_numpy(Y.astype(np.float32)).to(self.device)
        
        n_samples = len(X)
        mi_history = []
        
        for epoch in range(n_epochs):
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
                
                torch.nn.utils.clip_grad_norm_(self.mine.parameters(), 1.0)
                
                self.optimizer.step()
                
                epoch_mi.append(-loss.item())
            
            avg_mi = np.mean(epoch_mi)
            mi_history.append(avg_mi)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, MI estimate: {avg_mi:.4f}")
        
        return np.mean(mi_history[-10:])
    
    def estimate_conditional(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        n_epochs: int = 100,
        batch_size: int = 5000
    ) -> float:
        XZ = np.concatenate([X, Z], axis=1)
        estimator_xz = MutualInfoEstimator(
            XZ.shape[1], self.mine.T[0].in_features - 1, 
            device=self.device
        )
        mi_xz_y = estimator_xz.estimate(XZ, Y, n_epochs, batch_size)
        
        estimator_z = MutualInfoEstimator(
            Z.shape[1], device=self.device
        )
        mi_z_y = estimator_z.estimate(Z, Y, n_epochs, batch_size)
        
        return mi_xz_y - mi_z_y


class InfoNCE(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        temperature: float = 0.1
    ):
        super().__init__()
        
        self.temperature = temperature
        
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
        z = self.encoder(x)
        z = F.normalize(z, dim=1)
        
        sim = torch.mm(z, z.t()) / self.temperature
        
        labels = y.view(-1, 1)
        mask_pos = (labels == labels.t()).float()
        mask_pos.fill_diagonal_(0)
        
        mask_neg = (labels != labels.t()).float()
        
        exp_sim = torch.exp(sim)
        
        pos_sim = (exp_sim * mask_pos).sum(dim=1)
        neg_sim = (exp_sim * mask_neg).sum(dim=1)
        
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8) + 1e-8)
        
        return loss.mean()
