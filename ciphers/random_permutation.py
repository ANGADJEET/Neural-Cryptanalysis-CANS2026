"""
Random permutation baseline for distinguisher experiments.

Generates random ciphertext pairs that serve as the negative class
(label=0) in the distinguisher training.
"""

import numpy as np
from typing import Tuple, List
from .base import BaseCipher


class RandomPermutation(BaseCipher):
    """
    Random permutation generator for baseline comparison.
    
    This generates uniformly random ciphertext pairs that have no
    cryptographic structure. Used as the negative class in distinguisher
    experiments.
    """
    
    def __init__(self, block_size: int = 32):
        """
        Initialize random permutation generator.
        
        Args:
            block_size: Block size in bits (default 32 for SPECK/SIMON compatibility)
        """
        super().__init__(
            block_size=block_size,
            key_size=0,  # No key needed
            max_rounds=0  # No rounds
        )
    
    def encrypt(
        self, 
        plaintext: np.ndarray, 
        n_rounds: int = 0, 
        key: np.ndarray = None
    ) -> np.ndarray:
        """
        Generate random "ciphertext" (ignores plaintext).
        
        This simulates a random permutation by returning uniformly
        random values independent of the input.
        
        Args:
            plaintext: Ignored (shape used for output size)
            n_rounds: Ignored
            key: Ignored
            
        Returns:
            Random array of same shape as plaintext
        """
        n_samples = len(plaintext)
        if self.block_size <= 32:
            return np.random.randint(0, 2**self.block_size, size=n_samples, dtype=np.uint32)
        else:
            # For 64-bit blocks
            return np.random.randint(0, 2**63, size=n_samples, dtype=np.uint64) * 2 + \
                   np.random.randint(0, 2, size=n_samples, dtype=np.uint64)
    
    def encrypt_with_trace(
        self, 
        plaintext: np.ndarray, 
        n_rounds: int = 0, 
        key: np.ndarray = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Generate random output (no meaningful intermediate states).
        
        Args:
            plaintext: Shape reference
            n_rounds: Ignored
            key: Ignored
            
        Returns:
            (random_ciphertext, []) - empty trace list
        """
        ciphertext = self.encrypt(plaintext, n_rounds, key)
        return ciphertext, []
    
    def random_key(self) -> np.ndarray:
        """Return empty array (no key needed)."""
        return np.array([], dtype=np.uint16)
    
    def generate_random_pairs(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate independent random ciphertext pairs.
        
        For random permutation baseline, C and C' are completely
        independent (no differential structure).
        
        Args:
            n_samples: Number of pairs to generate
            
        Returns:
            (C, C') tuple of random arrays
        """
        if self.block_size <= 32:
            dtype = np.uint32
            max_val = 2**self.block_size
            C = np.random.randint(0, max_val, size=n_samples, dtype=dtype)
            C_prime = np.random.randint(0, max_val, size=n_samples, dtype=dtype)
        else:
            dtype = np.uint64
            C = np.random.randint(0, 2**63, size=n_samples, dtype=dtype) * 2 + \
                np.random.randint(0, 2, size=n_samples, dtype=dtype)
            C_prime = np.random.randint(0, 2**63, size=n_samples, dtype=dtype) * 2 + \
                      np.random.randint(0, 2, size=n_samples, dtype=dtype)
        
        return C, C_prime


def create_random_baseline(block_size: int) -> RandomPermutation:
    """
    Factory function to create random permutation with specified block size.
    
    Args:
        block_size: Block size in bits
        
    Returns:
        RandomPermutation instance
    """
    return RandomPermutation(block_size=block_size)
