"""
PRESENT block cipher implementation.

PRESENT is an ultra-lightweight SPN (Substitution-Permutation Network) cipher.
- Block size: 64 bits
- Key size: 80 bits
- Rounds: 31

Reference: https://www.iacr.org/archive/ches2007/47270450/47270450.pdf
"""

import numpy as np
from typing import Tuple, List
from .base import BaseCipher


class Present(BaseCipher):
    """
    PRESENT-80 implementation optimized for batch operations.
    
    Uses numpy vectorization. PRESENT has medium diffusion (SPN structure).
    """
    
    # PRESENT parameters
    BLOCK_SIZE = 64
    KEY_SIZE = 80
    ROUNDS = 31
    
    # S-box (4-bit)
    SBOX = np.array([
        0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
        0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2
    ], dtype=np.uint8)
    
    # Inverse S-box
    SBOX_INV = np.array([
        0x5, 0xE, 0xF, 0x8, 0xC, 0x1, 0x2, 0xD,
        0xB, 0x4, 0x6, 0x3, 0x0, 0x7, 0x9, 0xA
    ], dtype=np.uint8)
    
    # Permutation layer (bit positions)
    PERM = np.array([
        0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51,
        4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55,
        8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
        12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63
    ], dtype=np.uint8)
    
    def __init__(self):
        super().__init__(
            block_size=self.BLOCK_SIZE,
            key_size=self.KEY_SIZE,
            max_rounds=self.ROUNDS
        )
    
    def _sbox_layer(self, state: np.ndarray) -> np.ndarray:
        """
        Apply S-box to all 16 nibbles of 64-bit state.
        
        Args:
            state: Array of 64-bit states (N,)
            
        Returns:
            S-box substituted states
        """
        result = np.zeros_like(state)
        for i in range(16):
            nibble = (state >> (4 * i)) & 0xF
            result |= self.SBOX[nibble.astype(np.int64)].astype(np.uint64) << (4 * i)
        return result
    
    def _perm_layer(self, state: np.ndarray) -> np.ndarray:
        """
        Apply bit permutation layer.
        
        Args:
            state: Array of 64-bit states (N,)
            
        Returns:
            Permuted states
        """
        result = np.zeros_like(state)
        for i in range(64):
            bit = (state >> i) & 1
            result |= bit << self.PERM[i]
        return result
    
    def _expand_key(self, key: np.ndarray, n_rounds: int) -> np.ndarray:
        """
        Expand 80-bit key to round keys.
        
        Args:
            key: 80-bit key as (2,) array [high_16, low_64]
            n_rounds: Number of rounds
            
        Returns:
            Array of 64-bit round keys
        """
        # Key register: 80 bits
        # key[0] = high 16 bits, key[1] = low 64 bits
        key_high = int(key[0])
        key_low = int(key[1])
        
        round_keys = []
        
        for i in range(n_rounds + 1):
            # Round key is high 64 bits
            rk = (key_high << 48) | (key_low >> 16)
            round_keys.append(rk & ((1 << 64) - 1))
            
            # Key schedule update
            # 1. Rotate left by 61
            combined = (key_high << 64) | key_low
            combined = ((combined << 61) | (combined >> 19)) & ((1 << 80) - 1)
            
            # 2. S-box on leftmost 4 bits
            top_nibble = (combined >> 76) & 0xF
            combined = (combined & ((1 << 76) - 1)) | (int(self.SBOX[top_nibble]) << 76)
            
            # 3. XOR round counter to bits [19:15]
            combined ^= (i + 1) << 15
            
            key_high = (combined >> 64) & 0xFFFF
            key_low = combined & ((1 << 64) - 1)
        
        return np.array(round_keys, dtype=np.uint64)
    
    def encrypt(
        self, 
        plaintext: np.ndarray, 
        n_rounds: int, 
        key: np.ndarray
    ) -> np.ndarray:
        """
        Encrypt plaintext for specified number of rounds.
        
        Args:
            plaintext: Array of shape (N,) with 64-bit integers
            n_rounds: Number of rounds (1 to 31)
            key: Key as (2,) array [high_16_bits, low_64_bits]
            
        Returns:
            Ciphertext array
        """
        n_rounds = min(n_rounds, self.ROUNDS)
        round_keys = self._expand_key(key, n_rounds)
        
        state = plaintext.astype(np.uint64)
        
        for i in range(n_rounds):
            # Add round key
            state = state ^ round_keys[i]
            # S-box layer
            state = self._sbox_layer(state)
            # Permutation layer
            state = self._perm_layer(state)
        
        # Final round key addition
        state = state ^ round_keys[n_rounds]
        
        return state
    
    def encrypt_with_trace(
        self, 
        plaintext: np.ndarray, 
        n_rounds: int, 
        key: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Encrypt and return intermediate states after each round.
        
        Args:
            plaintext: Array of shape (N,) with 64-bit integers
            n_rounds: Number of rounds
            key: Encryption key
            
        Returns:
            (final_ciphertext, [state_after_round_1, ..., state_after_round_n])
        """
        n_rounds = min(n_rounds, self.ROUNDS)
        round_keys = self._expand_key(key, n_rounds)
        
        state = plaintext.astype(np.uint64)
        intermediate_states = []
        
        for i in range(n_rounds):
            # Add round key
            state = state ^ round_keys[i]
            # S-box layer  
            state = self._sbox_layer(state)
            # Permutation layer
            state = self._perm_layer(state)
            # Save state after this round
            intermediate_states.append(state.copy())
        
        # Final round key addition
        state = state ^ round_keys[n_rounds]
        
        return state, intermediate_states
    
    def random_key(self) -> np.ndarray:
        """Generate random 80-bit key as [high_16, low_64]."""
        return np.array([
            np.random.randint(0, 2**16),
            np.random.randint(0, 2**64, dtype=np.uint64)
        ], dtype=np.uint64)
    
    def random_plaintexts(self, n_samples: int) -> np.ndarray:
        """Generate random 64-bit plaintexts."""
        return np.random.randint(0, 2**63, size=n_samples, dtype=np.uint64) * 2 + \
               np.random.randint(0, 2, size=n_samples, dtype=np.uint64)
    
    def get_default_delta_p(self) -> int:
        """Return recommended input difference for differential analysis."""
        # Single-bit difference
        return 0x0000000000000001


# Convenience function for quick testing
def test_present():
    """Test PRESENT implementation with known test vectors."""
    cipher = Present()
    
    # Test vector from PRESENT paper
    # Key: 0x00000000000000000000 (80 bits)
    key = np.array([0x0000, 0x0000000000000000], dtype=np.uint64)
    plaintext = np.array([0x0000000000000000], dtype=np.uint64)
    
    # Expected ciphertext after 31 rounds: 0x5579C1387B228445
    ciphertext = cipher.encrypt(plaintext, 31, key)
    expected = 0x5579C1387B228445
    
    print(f"Plaintext:  0x{plaintext[0]:016x}")
    print(f"Ciphertext: 0x{ciphertext[0]:016x}")
    print(f"Expected:   0x{expected:016x}")
    print(f"Match: {ciphertext[0] == expected}")
    
    return ciphertext[0] == expected


if __name__ == "__main__":
    test_present()
