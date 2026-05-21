
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
    """Wrapper for MINE-based mutual information estimation.
    
    Improvements over vanilla MINE:
      - Learning rate warmup: first `warmup_epochs` use lr/10 for stability
        near zero MI (avoids initial overshoot that causes false positives).
      - Default 500 epochs (was 100) — MINE's gradient estimator is high-variance
        and needs many epochs to converge, especially for near-zero MI.
      - EMA-smoothed final estimate over the last `avg_window` epochs.
      - validate_calibration() method: tests against known-MI Gaussian pairs
        to prove the estimator is working before using it on cipher data.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        estimator: str = 'dv',
        device: str = 'cuda',
        lr: float = 0.001,
        warmup_epochs: int = 50,
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.estimator_type = estimator
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self._build()
    
    def _build(self):
        """(Re)initialize the MINE network and optimizer."""
        self.mine = MINE(self.input_dim, self.hidden_dims, self.estimator_type).to(self.device)
        self.optimizer = torch.optim.Adam(self.mine.parameters(), lr=self.lr)
    
    def estimate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_epochs: int = 500,
        batch_size: int = 5000,
        verbose: bool = False,
        avg_window: int = 20,
    ) -> float:
        """Estimate I(X; Y) using MINE with LR warmup.
        
        Args:
            X: (n_samples, input_dim) array.
            Y: (n_samples,) or (n_samples, 1) array.
            n_epochs: Number of training epochs (default 500).
            batch_size: Mini-batch size.
            verbose: Print progress every 50 epochs.
            avg_window: Average MI over last `avg_window` epochs for stability.
        
        Returns:
            Estimated mutual information in nats.
        """
        # Re-initialize network for each call to avoid stale weights
        self._build()
        
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        Y_tensor = torch.from_numpy(Y.astype(np.float32)).to(self.device)
        
        n_samples = len(X)
        mi_history = []
        
        # Set warmup LR
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.lr / 10.0
        
        for epoch in range(n_epochs):
            # Transition from warmup to full LR
            if epoch == self.warmup_epochs:
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.lr
            
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
            
            if verbose and (epoch + 1) % 50 == 0:
                recent = np.mean(mi_history[-avg_window:]) if len(mi_history) >= avg_window else avg_mi
                print(f"  Epoch {epoch+1}/{n_epochs}, MI (recent avg): {recent:.4f} nats")
        
        # Return average over last avg_window epochs for stability
        window = min(avg_window, len(mi_history))
        return float(np.mean(mi_history[-window:]))
    
    def validate_calibration(
        self,
        rho: float = 0.7,
        n_samples: int = 100000,
        n_epochs: int = 500,
        tolerance: float = 0.15,
        verbose: bool = True,
    ) -> dict:
        """Run MINE on bivariate Gaussians with known MI as a positive control.
        
        The true MI for bivariate Gaussian (X, Y) with correlation rho is:
            I(X; Y) = -0.5 * ln(1 - rho^2)
        
        This validates that MINE is correctly calibrated before using it
        on cipher data where the true MI is unknown.
        
        Args:
            rho: Correlation coefficient. Default 0.7 gives MI ≈ 0.357 nats.
            n_samples: Number of Gaussian samples.
            n_epochs: MINE training epochs.
            tolerance: Acceptable absolute error from true MI (in nats).
            verbose: Print results.
        
        Returns:
            dict with 'true_mi', 'estimated_mi', 'abs_error', 'calibrated' (bool).
        """
        true_mi = -0.5 * np.log(1 - rho**2)
        
        # Generate correlated Gaussians: Y = rho*X + sqrt(1-rho^2)*Z
        rng = np.random.RandomState(12345)  # Fixed seed for reproducibility
        X_ctrl = rng.randn(n_samples, 1).astype(np.float32)
        noise = rng.randn(n_samples, 1).astype(np.float32)
        Y_ctrl = (rho * X_ctrl + np.sqrt(1 - rho**2) * noise).astype(np.float32)
        
        # Create a fresh estimator for the control (1-dim input)
        ctrl_estimator = MutualInfoEstimator(
            input_dim=1,
            hidden_dims=self.hidden_dims,
            estimator=self.estimator_type,
            device=self.device,
            lr=self.lr,
            warmup_epochs=self.warmup_epochs,
        )
        estimated_mi = ctrl_estimator.estimate(
            X_ctrl, Y_ctrl, n_epochs=n_epochs, verbose=False
        )
        
        abs_error = abs(estimated_mi - true_mi)
        calibrated = abs_error < tolerance
        
        if verbose:
            status = '✓ PASS' if calibrated else '✗ FAIL'
            print(f"  MINE calibration ({status}): "
                  f"true={true_mi:.4f}, estimated={estimated_mi:.4f}, "
                  f"error={abs_error:.4f} nats (tol={tolerance})")
        
        return {
            'true_mi': float(true_mi),
            'estimated_mi': float(estimated_mi),
            'abs_error': float(abs_error),
            'rho': float(rho),
            'calibrated': bool(calibrated),
        }
    
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
            input_dim=XZ.shape[1],
            hidden_dims=[256, 128, 64],
            device=self.device
        )
        mi_xz_y = estimator_xz.estimate(XZ, Y, n_epochs, batch_size)
        
        estimator_z = MutualInfoEstimator(
            input_dim=Z.shape[1],
            hidden_dims=[256, 128, 64],
            device=self.device
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
