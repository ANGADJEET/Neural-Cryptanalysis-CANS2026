
from .mlp import MLP, GohrMLP
from .cnn import CNN, ResidualCNN
from .siamese import SiameseNetwork
from .rnn import CryptoLSTM, CryptoGRU
from .mine import MINE, MutualInfoEstimator
from .gohr_resnet import GohrResNet

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
    'GohrResNet',
]


def get_model(name: str, input_dim: int, **kwargs):
    models = {
        'mlp': MLP,
        'gohr_mlp': GohrMLP,
        'cnn': CNN,
        'residual_cnn': ResidualCNN,
        'siamese': SiameseNetwork,
        'lstm': CryptoLSTM,
        'gru': CryptoGRU,
        'mine': MINE,
        'gohr_resnet': GohrResNet,
    }
    
    if name.lower() not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    
    return models[name.lower()](input_dim=input_dim, **kwargs)
