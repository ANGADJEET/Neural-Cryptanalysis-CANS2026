
import numpy as np
from typing import Tuple, List
from .base import BaseCipher


class RandomPermutation(BaseCipher):
    
    def __init__(self, block_size: int = 32):
        super().__init__(
            block_size=block_size,
            key_size=0,
            max_rounds=0
        )
    
    def encrypt(
        self, 
        plaintext: np.ndarray, 
        n_rounds: int = 0, 
        key: np.ndarray = None
    ) -> np.ndarray:
        n_samples = len(plaintext)
        if self.block_size <= 32:
            return np.random.randint(0, 2**self.block_size, size=n_samples, dtype=np.uint32)
        else:
            return np.random.randint(0, 2**63, size=n_samples, dtype=np.uint64) * 2 + \
                   np.random.randint(0, 2, size=n_samples, dtype=np.uint64)
    
    def encrypt_with_trace(
        self, 
        plaintext: np.ndarray, 
        n_rounds: int = 0, 
        key: np.ndarray = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        ciphertext = self.encrypt(plaintext, n_rounds, key)
        return ciphertext, []
    
    def random_key(self) -> np.ndarray:
        return np.array([], dtype=np.uint16)
    
    def generate_random_pairs(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
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
    return RandomPermutation(block_size=block_size)
