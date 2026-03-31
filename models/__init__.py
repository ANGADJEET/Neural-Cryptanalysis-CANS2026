"""
Neural network models for cryptanalysis.
"""

from .mlp import MLP, GohrMLP
from .cnn import CNN, ResidualCNN
from .siamese import SiameseNetwork
from .rnn import CryptoLSTM, CryptoGRU
from .mine import MINE, MutualInfoEstimator

__all__ = [
    'MLP',
    'GohrMLP',
    'CNN',
    'ResidualCNN',
    'SiameseNetwork',
    'CryptoLSTM',
    'CryptoGRU',
    'MINE',
    'MutualInfoEstimator',
]


def get_model(name: str, input_dim: int, **kwargs):
    """
    Factory function to get model by name.
    
    Args:
        name: Model name ('mlp', 'cnn', 'siamese', 'lstm', 'mine')
        input_dim: Input dimension
        **kwargs: Model-specific arguments
        
    Returns:
        Model instance
    """
    models = {
        'mlp': MLP,
        'gohr_mlp': GohrMLP,
        'cnn': CNN,
        'residual_cnn': ResidualCNN,
        'siamese': SiameseNetwork,
        'lstm': CryptoLSTM,
        'gru': CryptoGRU,
        'mine': MINE,
    }
    
    if name.lower() not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    
    return models[name.lower()](input_dim=input_dim, **kwargs)
