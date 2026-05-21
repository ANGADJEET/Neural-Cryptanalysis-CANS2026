
from .base import BaseCipher
from .speck32 import Speck32
from .simon32 import Simon32
from .present import Present
from .random_permutation import RandomPermutation
from .simon32_irk import Simon32IRK

__all__ = [
    'BaseCipher',
    'Speck32',
    'Simon32', 
    'Present',
    'RandomPermutation',
    'Simon32IRK',
]


def get_cipher(name: str) -> BaseCipher:
    ciphers = {
        'speck32': Speck32,
        'simon32': Simon32,
        'present': Present,
        'random': RandomPermutation,
        'simon32_irk': Simon32IRK,
    }
    if name.lower() not in ciphers:
        raise ValueError(f"Unknown cipher: {name}. Available: {list(ciphers.keys())}")
    return ciphers[name.lower()]()

