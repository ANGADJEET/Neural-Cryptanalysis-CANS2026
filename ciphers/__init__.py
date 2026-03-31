"""
Cipher implementations for neural differential cryptanalysis.
Supports SPECK32/64, SIMON32/64, and PRESENT.
"""

from .base import BaseCipher
from .speck32 import Speck32
from .simon32 import Simon32
from .present import Present
from .random_permutation import RandomPermutation

__all__ = [
    'BaseCipher',
    'Speck32',
    'Simon32', 
    'Present',
    'RandomPermutation',
]


def get_cipher(name: str) -> BaseCipher:
    """Factory function to get cipher by name."""
    ciphers = {
        'speck32': Speck32,
        'simon32': Simon32,
        'present': Present,
        'random': RandomPermutation,
    }
    if name.lower() not in ciphers:
        raise ValueError(f"Unknown cipher: {name}. Available: {list(ciphers.keys())}")
    return ciphers[name.lower()]()
