"""
Abstract base class for all cipher implementations.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import numpy as np


class BaseCipher(ABC):
    """
    Abstract base class for block cipher implementations.
    
    All ciphers must implement:
    - encrypt: Encrypt plaintext for specified rounds
    - encrypt_with_trace: Encrypt and return intermediate states
    - random_key: Generate a random key
    """
    
    def __init__(self, block_size: int, key_size: int, max_rounds: int):
        """
        Initialize cipher with parameters.
        
        Args:
            block_size: Block size in bits
            key_size: Key size in bits
            max_rounds: Maximum number of rounds
        """
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
        """
        Encrypt plaintext for specified number of rounds.
        
        Args:
            plaintext: Plaintext array of shape (N, block_size) as bits
                       or (N,) as integers depending on implementation
            n_rounds: Number of rounds to encrypt
            key: Encryption key
            
        Returns:
            Ciphertext array of same shape as plaintext
        """
        pass
    
    @abstractmethod
    def encrypt_with_trace(
        self, 
        plaintext: np.ndarray, 
        n_rounds: int, 
        key: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Encrypt plaintext and return intermediate round states.
        
        Args:
            plaintext: Plaintext array
            n_rounds: Number of rounds to encrypt
            key: Encryption key
            
        Returns:
            Tuple of (final_ciphertext, list_of_intermediate_states)
            Each intermediate state has shape (N, block_size)
        """
        pass
    
    @abstractmethod
    def random_key(self) -> np.ndarray:
        """
        Generate a random key for this cipher.
        
        Returns:
            Random key array
        """
        pass
    
    def random_plaintexts(self, n_samples: int) -> np.ndarray:
        """
        Generate random plaintexts.
        
        Args:
            n_samples: Number of plaintexts to generate
            
        Returns:
            Array of random plaintexts
        """
        # Default: generate as integers
        return np.random.randint(0, 2**self.block_size, size=n_samples, dtype=np.uint64)
    
    def apply_difference(
        self, 
        plaintext: np.ndarray, 
        delta_p: int
    ) -> np.ndarray:
        """
        Apply input difference to plaintext.
        
        Args:
            plaintext: Original plaintext
            delta_p: Input difference (XOR)
            
        Returns:
            Modified plaintext P' = P XOR delta_p
        """
        return plaintext ^ delta_p
    
    def to_bits(self, values: np.ndarray) -> np.ndarray:
        """
        Convert integer values to bit representation.
        
        Args:
            values: Array of integers
            
        Returns:
            Array of shape (N, block_size) with bit values
        """
        n_samples = len(values)
        bits = np.zeros((n_samples, self.block_size), dtype=np.uint8)
        for i in range(self.block_size):
            bits[:, self.block_size - 1 - i] = (values >> i) & 1
        return bits
    
    def from_bits(self, bits: np.ndarray) -> np.ndarray:
        """
        Convert bit representation to integers.
        
        Args:
            bits: Array of shape (N, block_size)
            
        Returns:
            Array of integers
        """
        n_samples = bits.shape[0]
        values = np.zeros(n_samples, dtype=np.uint64)
        for i in range(self.block_size):
            values += bits[:, self.block_size - 1 - i].astype(np.uint64) << i
        return values
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(block={self.block_size}, key={self.key_size}, rounds={self.max_rounds})"
