"""
Training callbacks for neural cryptanalysis.
"""

import torch
from typing import Optional, Dict, Any
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class EarlyStopping:
    """
    Early stopping callback to stop training when validation metric stops improving.
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'max',
        verbose: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' depending on metric
            verbose: Print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        
        if mode == 'min':
            self.is_better = lambda x, best: x < best - min_delta
        else:
            self.is_better = lambda x, best: x > best + min_delta
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False


class ModelCheckpoint:
    """
    Callback to save model checkpoints.
    """
    
    def __init__(
        self,
        save_dir: str = './checkpoints',
        filename: str = 'model_{epoch}.pt',
        monitor: str = 'val_acc',
        mode: str = 'max',
        save_best_only: bool = True,
        verbose: bool = True
    ):
        """
        Initialize model checkpoint.
        
        Args:
            save_dir: Directory to save checkpoints
            filename: Checkpoint filename pattern
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save best model
            verbose: Print save messages
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        self.best_score = None
        
        if mode == 'min':
            self.is_better = lambda x, best: x < best
        else:
            self.is_better = lambda x, best: x > best
    
    def __call__(
        self,
        model: torch.nn.Module,
        metrics: Dict[str, float],
        epoch: int
    ) -> Optional[str]:
        """
        Maybe save checkpoint.
        
        Args:
            model: Model to save
            metrics: Current metrics
            epoch: Current epoch
            
        Returns:
            Path to saved checkpoint if saved, None otherwise
        """
        current_score = metrics.get(self.monitor, 0.0)
        
        should_save = False
        if self.best_score is None:
            self.best_score = current_score
            should_save = True
        elif self.is_better(current_score, self.best_score):
            self.best_score = current_score
            should_save = True
        elif not self.save_best_only:
            should_save = True
        
        if should_save:
            filepath = self.save_dir / self.filename.format(epoch=epoch)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'metrics': metrics
            }, filepath)
            
            if self.verbose:
                print(f"Saved checkpoint: {filepath}")
            
            return str(filepath)
        
        return None


class WandbCallback:
    """
    Callback for Weights & Biases logging.
    """
    
    def __init__(
        self,
        project: str = 'neural-cryptanalysis',
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        log_model: bool = True,
        log_freq: int = 100
    ):
        """
        Initialize wandb callback.
        
        Args:
            project: Wandb project name
            name: Run name
            config: Configuration dict
            log_model: Log model architecture
            log_freq: Gradient logging frequency
        """
        self.project = project
        self.name = name
        self.config = config or {}
        self.log_model = log_model
        self.log_freq = log_freq
        
        self.initialized = False
    
    def on_train_begin(self, model: torch.nn.Module):
        """Called at start of training."""
        if not WANDB_AVAILABLE:
            print("Warning: wandb not available, skipping logging")
            return
        
        wandb.init(
            project=self.project,
            name=self.name,
            config=self.config
        )
        
        if self.log_model:
            wandb.watch(model, log='all', log_freq=self.log_freq)
        
        self.initialized = True
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Called at end of each epoch."""
        if not self.initialized:
            return
        
        wandb.log({
            'epoch': epoch,
            **metrics
        })
    
    def on_train_end(self, model: torch.nn.Module = None):
        """Called at end of training."""
        if not self.initialized:
            return
        
        if model is not None and self.log_model:
            # Save model artifact
            artifact = wandb.Artifact('model', type='model')
            torch.save(model.state_dict(), 'model.pt')
            artifact.add_file('model.pt')
            wandb.log_artifact(artifact)
        
        wandb.finish()
    
    def log(self, data: Dict[str, Any]):
        """Log custom data."""
        if not self.initialized:
            return
        wandb.log(data)
    
    def log_figure(self, key: str, figure):
        """Log matplotlib figure."""
        if not self.initialized:
            return
        wandb.log({key: wandb.Image(figure)})


class LearningRateLogger:
    """
    Callback to log learning rate.
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def on_epoch_end(self) -> Dict[str, float]:
        """Return LR for logging."""
        return {'learning_rate': self.get_lr()}
