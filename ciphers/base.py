
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import numpy as np


class BaseCipher(ABC):
    
    def __init__(self, block_size: int, key_size: int, max_rounds: int):
        self.block_size = block_size
        self.key_size = key_size
        self.max_rounds = max_rounds
    
    @abstractmethod
    def encrypt(
        self, 
        plaintext: np.ndarray, 
        n_rounds: int, 
        key: np.ndarray
    ) -> np.ndarray:
        pass
    
    @abstractmethod
    def encrypt_with_trace(
        self, 
        plaintext: np.ndarray, 
        n_rounds: int, 
        key: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        pass
    
    @abstractmethod
    def random_key(self) -> np.ndarray:
        pass
    
    def random_plaintexts(self, n_samples: int) -> np.ndarray:
        return np.random.randint(0, 2**self.block_size, size=n_samples, dtype=np.uint64)
    
    def apply_difference(
        self, 
        plaintext: np.ndarray, 
        delta_p: int
    ) -> np.ndarray:
        return plaintext ^ delta_p
    
    def to_bits(self, values: np.ndarray) -> np.ndarray:
        n_samples = len(values)
        bits = np.zeros((n_samples, self.block_size), dtype=np.uint8)
        for i in range(self.block_size):
            bits[:, self.block_size - 1 - i] = (values >> i) & 1
        return bits
    
    def from_bits(self, bits: np.ndarray) -> np.ndarray:
        n_samples = bits.shape[0]
        values = np.zeros(n_samples, dtype=np.uint64)
        for i in range(self.block_size):
            values += bits[:, self.block_size - 1 - i].astype(np.uint64) << i
        return values
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(block={self.block_size}, key={self.key_size}, rounds={self.max_rounds})"
