"""
SIMON32/64 block cipher implementation.

SIMON is a Feistel cipher designed by NSA.
- Block size: 32 bits (2 x 16-bit words)
- Key size: 64 bits (4 x 16-bit words)
- Rounds: 32

Reference: https://eprint.iacr.org/2013/404.pdf
"""

import numpy as np
from typing import Tuple, List
from .base import BaseCipher


class Simon32(BaseCipher):
    """
    SIMON32/64 implementation optimized for batch operations.
    
    Uses numpy vectorization for efficient encryption of large batches.
    SIMON has slower diffusion than SPECK (Feistel vs ARX).
    """
    
    # SIMON32/64 parameters
    BLOCK_SIZE = 32
    KEY_SIZE = 64
    WORD_SIZE = 16
    ROUNDS = 32
    
    # Key schedule constants
    Z_SEQ = [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 
             0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1,
             1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1,
             0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0]  # z0 sequence
    
    def __init__(self):
        super().__init__(
            block_size=self.BLOCK_SIZE,
            key_size=self.KEY_SIZE,
            max_rounds=self.ROUNDS
        )
        self.word_mask = (1 << self.WORD_SIZE) - 1  # 0xFFFF
    
    def _rol(self, x: np.ndarray, r: int) -> np.ndarray:
        """Left rotation for 16-bit words (vectorized)."""
        r = r % self.WORD_SIZE
        return ((x << r) | (x >> (self.WORD_SIZE - r))) & self.word_mask
    
    def _ror_scalar(self, x: int, r: int) -> int:
        """Right rotation for scalar 16-bit value."""
        r = r % self.WORD_SIZE
        return ((x >> r) | (x << (self.WORD_SIZE - r))) & self.word_mask
    
    def _expand_key(self, key: np.ndarray, n_rounds: int) -> np.ndarray:
        """
        Expand key to round keys using SIMON key schedule.
        
        For SIMON32/64, m=4 key words.
        
        Args:
            key: Master key as 4 x 16-bit words [k3, k2, k1, k0]
            n_rounds: Number of rounds
            
        Returns:
            Array of round keys
        """
        m = 4  # Number of key words for SIMON32/64
        c = 0xFFFC  # (2^n - 4) for n=16
        
        # Initialize with key words: k = [k3, k2, k1, k0] means k[0]=k3, k[1]=k2, etc.
        # But we want k[i] = K[m-1-i], so k = [k0, k1, k2, k3]
        k = [int(key[3]), int(key[2]), int(key[1]), int(key[0])]
        
        for i in range(m, n_rounds):
            tmp = self._ror_scalar(k[i-1], 3)
            tmp = tmp ^ k[i-3]
            tmp = tmp ^ self._ror_scalar(tmp, 1)
            
            z_bit = self.Z_SEQ[(i - m) % 62]
            k_new = (~k[i-m] & self.word_mask) ^ tmp ^ z_bit ^ c
            k.append(k_new & self.word_mask)
        
        return np.array(k[:n_rounds], dtype=np.uint16)
    
    def _f(self, x: np.ndarray) -> np.ndarray:
        """
        SIMON round function f(x) = (x <<< 1) & (x <<< 8) XOR (x <<< 2)
        """
        return (self._rol(x, 1) & self._rol(x, 8)) ^ self._rol(x, 2)
    
    def _round_function(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single Feistel round of SIMON.
        
        Args:
            x: Left word (batch)
            y: Right word (batch)
            k: Round key
            
        Returns:
            Updated (x, y) = (y XOR f(x) XOR k, x)
        """
        new_x = (y ^ self._f(x) ^ k) & self.word_mask
        new_y = x
        return new_x, new_y
    
    def encrypt(
        self, 
        plaintext: np.ndarray, 
        n_rounds: int, 
        key: np.ndarray
    ) -> np.ndarray:
        """
        Encrypt plaintext for specified number of rounds.
        
        Args:
            plaintext: Array of shape (N,) with 32-bit integers,
                      or (N, 2) with 16-bit word pairs [x, y]
            n_rounds: Number of rounds (1 to 32)
            key: Key as (4,) array of 16-bit words
            
        Returns:
            Ciphertext array of same shape as input
        """
        n_rounds = min(n_rounds, self.ROUNDS)
        round_keys = self._expand_key(key, n_rounds)
        
        # Handle different input formats
        if plaintext.ndim == 1:
            # Convert 32-bit integers to word pairs
            x = ((plaintext >> 16) & self.word_mask).astype(np.uint16)
            y = (plaintext & self.word_mask).astype(np.uint16)
            return_packed = True
        else:
            x = plaintext[:, 0].astype(np.uint16)
            y = plaintext[:, 1].astype(np.uint16)
            return_packed = False
        
        # Apply rounds
        for i in range(n_rounds):
            x, y = self._round_function(x, y, round_keys[i])
        
        # Return in same format as input
        if return_packed:
            return (x.astype(np.uint32) << 16) | y.astype(np.uint32)
        else:
            return np.stack([x, y], axis=1)
    
    def encrypt_with_trace(
        self, 
        plaintext: np.ndarray, 
        n_rounds: int, 
        key: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Encrypt and return intermediate states after each round.
        
        Args:
            plaintext: Array of shape (N,) with 32-bit integers
            n_rounds: Number of rounds
            key: Encryption key
            
        Returns:
            (final_ciphertext, [state_after_round_1, ..., state_after_round_n])
        """
        n_rounds = min(n_rounds, self.ROUNDS)
        round_keys = self._expand_key(key, n_rounds)
        
        # Convert to word pairs
        x = ((plaintext >> 16) & self.word_mask).astype(np.uint16)
        y = (plaintext & self.word_mask).astype(np.uint16)
        
        intermediate_states = []
        
        # Apply rounds and save states
        for i in range(n_rounds):
            x, y = self._round_function(x, y, round_keys[i])
            # Pack state as 32-bit value
            state = (x.astype(np.uint32) << 16) | y.astype(np.uint32)
            intermediate_states.append(state)
        
        final = intermediate_states[-1] if intermediate_states else plaintext
        return final, intermediate_states
    
    def random_key(self) -> np.ndarray:
        """Generate random 64-bit key as 4 x 16-bit words."""
        return np.random.randint(0, 2**16, size=4, dtype=np.uint16)
    
    def random_plaintexts(self, n_samples: int) -> np.ndarray:
        """Generate random 32-bit plaintexts."""
        return np.random.randint(0, 2**32, size=n_samples, dtype=np.uint32)
    
    def get_default_delta_p(self) -> int:
        """Return recommended input difference for differential analysis."""
        # Single-bit difference in right word
        return 0x00000001


# Convenience function for quick testing
def test_simon32():
    """Test SIMON32 implementation with known test vectors."""
    cipher = Simon32()
    
    # Test vector from SIMON paper
    # Key: 0x1918 1110 0908 0100
    key = np.array([0x1918, 0x1110, 0x0908, 0x0100], dtype=np.uint16)
    plaintext = np.array([0x65656877], dtype=np.uint32)
    
    # Expected ciphertext after 32 rounds: 0xc69b e9bb
    ciphertext = cipher.encrypt(plaintext, 32, key)
    expected = 0xc69be9bb
    
    print(f"Plaintext:  0x{plaintext[0]:08x}")
    print(f"Ciphertext: 0x{ciphertext[0]:08x}")
    print(f"Expected:   0x{expected:08x}")
    print(f"Match: {ciphertext[0] == expected}")
    
    return ciphertext[0] == expected


if __name__ == "__main__":
    test_simon32()
