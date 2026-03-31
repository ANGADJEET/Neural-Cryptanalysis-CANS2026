"""
Training pipeline for neural cryptanalysis models.

Features:
- Wandb integration for experiment tracking
- Configurable training loop
- Multiple evaluation metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Callable, Any
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Trainer:
    """
    Training manager for neural cryptanalysis models.
    
    Handles training loop, evaluation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        criterion: Optional[nn.Module] = None,
        device: str = 'cuda',
        config: Optional[Dict] = None,
        use_wandb: bool = True,
        project_name: str = 'neural-cryptanalysis',
        run_name: Optional[str] = None,
        save_dir: str = './checkpoints'
    ):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer (default: Adam)
            scheduler: Learning rate scheduler (optional)
            criterion: Loss function (default: BCELoss)
            device: Compute device
            config: Configuration dict for logging
            use_wandb: Enable wandb logging
            project_name: Wandb project name
            run_name: Wandb run name
            save_dir: Directory for checkpoints
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        # Loss function
        if criterion is None:
            self.criterion = nn.BCELoss()
        else:
            self.criterion = criterion
        
        self.config = config or {}
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Wandb setup
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=project_name,
                name=run_name,
                config=self.config
            )
            wandb.watch(self.model, log='all', log_freq=100)
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_advantage': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, (X, y) in enumerate(pbar):
            X, y = X.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(X).squeeze()
            loss = self.criterion(outputs, y)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item() * X.size(0)
            predictions = (outputs > 0.5).float()
            correct += (predictions == y).sum().item()
            total += X.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': correct / total
            })
        
        metrics = {
            'train_loss': total_loss / total,
            'train_acc': correct / total
        }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                outputs = self.model(X).squeeze()
                loss = self.criterion(outputs, y)
                
                total_loss += loss.item() * X.size(0)
                predictions = (outputs > 0.5).float()
                correct += (predictions == y).sum().item()
                total += X.size(0)
                
                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        accuracy = correct / total
        advantage = 2 * abs(accuracy - 0.5)  # Cryptographic advantage
        
        # AUC-ROC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_labels, all_outputs)
        except:
            auc = 0.0
        
        metrics = {
            'val_loss': total_loss / total,
            'val_acc': accuracy,
            'val_advantage': advantage,
            'val_auc': auc
        }
        
        return metrics
    
    def train(
        self,
        n_epochs: int = 50,
        early_stopping_patience: int = 5,
        save_best: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            n_epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            save_best: Save best model checkpoint
            
        Returns:
            Training history
        """
        patience_counter = 0
        
        for epoch in range(n_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Update history
            for key, value in train_metrics.items():
                self.history[key].append(value)
            for key, value in val_metrics.items():
                if key in self.history:
                    self.history[key].append(value)
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    **train_metrics,
                    **val_metrics,
                    'epoch': epoch,
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            
            # Print progress
            print(f"Epoch {epoch + 1}/{n_epochs} | "
                  f"Train Loss: {train_metrics['train_loss']:.4f} | "
                  f"Train Acc: {train_metrics['train_acc']:.4f} | "
                  f"Val Acc: {val_metrics['val_acc']:.4f} | "
                  f"Advantage: {val_metrics['val_advantage']:.4f}")
            
            # Save best model
            if val_metrics['val_acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['val_acc']
                patience_counter = 0
                
                if save_best:
                    self.save_checkpoint('best_model.pt')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Finish wandb run
        if self.use_wandb:
            wandb.finish()
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)


def train_model(
    model: nn.Module,
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    representation: str = 'R2_xor_diff',
    block_size: int = 32,
    batch_size: int = 5000,
    n_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cuda',
    use_wandb: bool = True,
    **kwargs
) -> Dict[str, List[float]]:
    """
    Convenience function to train a model.
    
    Args:
        model: Model to train
        train_data: Training data dict
        val_data: Validation data dict
        representation: Representation name
        block_size: Cipher block size
        batch_size: Batch size
        n_epochs: Number of epochs
        learning_rate: Learning rate
        device: Compute device
        use_wandb: Enable wandb
        **kwargs: Additional trainer arguments
        
    Returns:
        Training history
    """
    from data.dataloader import CryptoDataset
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = CryptoDataset(train_data, representation, block_size)
    val_dataset = CryptoDataset(val_data, representation, block_size)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_wandb=use_wandb,
        config={
            'representation': representation,
            'block_size': block_size,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'n_epochs': n_epochs,
            **kwargs
        }
    )
    
    # Train
    return trainer.train(n_epochs=n_epochs)
