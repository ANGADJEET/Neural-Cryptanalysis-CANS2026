"""
SPECK32/64 block cipher implementation.

SPECK is an ARX (Add-Rotate-XOR) cipher designed by NSA.
- Block size: 32 bits (2 x 16-bit words)
- Key size: 64 bits (4 x 16-bit words)
- Rounds: 22

Reference: https://eprint.iacr.org/2013/404.pdf
"""

import numpy as np
from typing import Tuple, List
from .base import BaseCipher


class Speck32(BaseCipher):
    """
    SPECK32/64 implementation optimized for batch operations.
    
    Uses numpy vectorization for efficient encryption of large batches.
    """
    
    # SPECK32/64 parameters
    BLOCK_SIZE = 32
    KEY_SIZE = 64
    WORD_SIZE = 16
    ROUNDS = 22
    ALPHA = 7  # Right rotation
    BETA = 2   # Left rotation
    
    def __init__(self):
        super().__init__(
            block_size=self.BLOCK_SIZE,
            key_size=self.KEY_SIZE,
            max_rounds=self.ROUNDS
        )
        self.word_mask = (1 << self.WORD_SIZE) - 1  # 0xFFFF
    
    def _ror(self, x: np.ndarray, r: int) -> np.ndarray:
        """Right rotation for 16-bit words."""
        r = r % self.WORD_SIZE
        return ((x >> r) | (x << (self.WORD_SIZE - r))) & self.word_mask
    
    def _rol(self, x: np.ndarray, r: int) -> np.ndarray:
        """Left rotation for 16-bit words."""
        r = r % self.WORD_SIZE
        return ((x << r) | (x >> (self.WORD_SIZE - r))) & self.word_mask
    
    def _expand_key(self, key: np.ndarray, n_rounds: int) -> np.ndarray:
        """
        Expand key to round keys.
        
        For SPECK32/64, m=4 key words.
        Key format: [k3, k2, k1, k0] where words are 16-bit.
        Initial state: l = [k1, k2, k3], k[0] = k0
        
        Args:
            key: Master key as 4 x 16-bit words [k3, k2, k1, k0]
            n_rounds: Number of rounds
            
        Returns:
            Array of round keys
        """
        m = 4  # Number of key words
        
        # Initialize: l[0..m-2] = K[m-1]..K[1], k[0] = K[0]
        # key array is [k3, k2, k1, k0] 
        l = [int(key[2]), int(key[1]), int(key[0])]  # l[0]=k1, l[1]=k2, l[2]=k3
        k = int(key[3])  # k[0] = k0
        
        round_keys = [k]
        
        for i in range(n_rounds - 1):
            # l[i+m-1] = (k[i] + ROR(l[i], alpha)) XOR i
            l_idx = i % (m - 1)
            l_new = ((k + self._ror_scalar(l[l_idx], self.ALPHA)) & self.word_mask) ^ i
            
            # k[i+1] = ROL(k[i], beta) XOR l[i+m-1]
            k = self._rol_scalar(k, self.BETA) ^ l_new
            
            l[l_idx] = l_new
            round_keys.append(k)
        
        return np.array(round_keys, dtype=np.uint16)
    
    def _ror_scalar(self, x: int, r: int) -> int:
        """Right rotation for scalar 16-bit value."""
        r = r % self.WORD_SIZE
        return ((x >> r) | (x << (self.WORD_SIZE - r))) & self.word_mask
    
    def _rol_scalar(self, x: int, r: int) -> int:
        """Left rotation for scalar 16-bit value."""
        r = r % self.WORD_SIZE
        return ((x << r) | (x >> (self.WORD_SIZE - r))) & self.word_mask
    
    def _round_function(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single round of SPECK.
        
        Args:
            x: Left word (batch)
            y: Right word (batch)
            k: Round key
            
        Returns:
            Updated (x, y)
        """
        # x = ((ROR(x, alpha) + y) & mask) XOR k
        x = ((self._ror(x, self.ALPHA) + y) & self.word_mask) ^ k
        # y = ROL(y, beta) XOR x
        y = self._rol(y, self.BETA) ^ x
        return x, y
    
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
            n_rounds: Number of rounds (1 to 22)
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
        # 0x0040/0000 is a known good differential for SPECK32
        return 0x00400000


# Convenience function for quick testing
def test_speck32():
    """Test SPECK32 implementation with known test vectors."""
    cipher = Speck32()
    
    # Test vector from SPECK paper
    # Key: 0x1918 1110 0908 0100
    key = np.array([0x1918, 0x1110, 0x0908, 0x0100], dtype=np.uint16)
    plaintext = np.array([0x6574694c], dtype=np.uint32)  # "Lite"
    
    # Expected ciphertext after 22 rounds: 0xa868 42f2
    ciphertext = cipher.encrypt(plaintext, 22, key)
    expected = 0xa86842f2
    
    print(f"Plaintext:  0x{plaintext[0]:08x}")
    print(f"Ciphertext: 0x{ciphertext[0]:08x}")
    print(f"Expected:   0x{expected:08x}")
    print(f"Match: {ciphertext[0] == expected}")
    
    return ciphertext[0] == expected


if __name__ == "__main__":
    test_speck32()
