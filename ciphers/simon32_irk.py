"""
SIMON32 with Independent Round Keys (SIMON32-IRK)

A variant of SIMON32/64 where each round key is independently and
uniformly random, rather than derived from a master key via the
SIMON key schedule.

Purpose: Test whether Gohr et al.'s theorem — that Feistel distinguishers
learn only DDT features, implying Markovian composition — holds when its
"independent round keys" assumption is satisfied.

If SIMON32-IRK shows positive transfer (while standard SIMON32 shows
anti-transfer), the anti-transfer in standard SIMON is caused by key
schedule dependence. If SIMON32-IRK still shows anti-transfer, the
theorem's assumption may be insufficient for SIMON's specific nonlinearity.

The round function is IDENTICAL to standard SIMON32:
  f(x) = (x <<< 1) & (x <<< 8) ^ (x <<< 2)
  (x_new, y_new) = (y ^ f(x) ^ k_i, x)

The ONLY difference: k_i are independent uniform random 16-bit values,
not derived from a 64-bit master key.
"""

import numpy as np
from typing import Tuple, List
from .base import BaseCipher


class Simon32IRK(BaseCipher):
    """SIMON32 with truly independent round keys.
    
    Same round function as Simon32, but random_key() returns
    n_rounds independently sampled 16-bit subkeys.
    """
    
    BLOCK_SIZE = 32
    KEY_SIZE = 0  # Not applicable — keys are per-round
    WORD_SIZE = 16
    ROUNDS = 32  # Maximum rounds
    
    def __init__(self, default_rounds: int = 32):
        super().__init__(
            block_size=self.BLOCK_SIZE,
            key_size=self.ROUNDS * self.WORD_SIZE,  # Total bits across all round keys
            max_rounds=self.ROUNDS
        )
        self.word_mask = (1 << self.WORD_SIZE) - 1
        self.default_rounds = default_rounds
    
    def _rol(self, x: np.ndarray, r: int) -> np.ndarray:
        r = r % self.WORD_SIZE
        return ((x << r) | (x >> (self.WORD_SIZE - r))) & self.word_mask
    
    def _f(self, x: np.ndarray) -> np.ndarray:
        """SIMON round function: f(x) = (x<<<1 & x<<<8) ^ x<<<2"""
        return (self._rol(x, 1) & self._rol(x, 8)) ^ self._rol(x, 2)
    
    def _round_function(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        new_x = (y ^ self._f(x) ^ k) & self.word_mask
        new_y = x
        return new_x, new_y
    
    def _expand_key(self, key: np.ndarray, n_rounds: int) -> np.ndarray:
        """For IRK, key IS the expanded round keys — no schedule needed."""
        # key is already an array of n_rounds 16-bit subkeys
        if len(key) < n_rounds:
            raise ValueError(
                f"Key has {len(key)} subkeys but {n_rounds} rounds requested. "
                f"Generate key with random_key(n_rounds={n_rounds})."
            )
        return key[:n_rounds].astype(np.uint16)
    
    def encrypt(
        self, 
        plaintext: np.ndarray, 
        n_rounds: int, 
        key: np.ndarray
    ) -> np.ndarray:
        n_rounds = min(n_rounds, self.ROUNDS)
        round_keys = self._expand_key(key, n_rounds)
        
        x = ((plaintext >> 16) & self.word_mask).astype(np.uint16)
        y = (plaintext & self.word_mask).astype(np.uint16)
        
        for i in range(n_rounds):
            x, y = self._round_function(x, y, round_keys[i])
        
        return (x.astype(np.uint32) << 16) | y.astype(np.uint32)
    
    def encrypt_with_trace(
        self, 
        plaintext: np.ndarray, 
        n_rounds: int, 
        key: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        n_rounds = min(n_rounds, self.ROUNDS)
        round_keys = self._expand_key(key, n_rounds)
        
        x = ((plaintext >> 16) & self.word_mask).astype(np.uint16)
        y = (plaintext & self.word_mask).astype(np.uint16)
        
        intermediate_states = []
        
        for i in range(n_rounds):
            x, y = self._round_function(x, y, round_keys[i])
            state = (x.astype(np.uint32) << 16) | y.astype(np.uint32)
            intermediate_states.append(state)
        
        final = intermediate_states[-1] if intermediate_states else plaintext
        return final, intermediate_states
    
    def random_key(self, n_rounds: int = None) -> np.ndarray:
        """Generate independently random 16-bit subkeys for each round.
        
        This is the critical difference from standard SIMON32: there is
        NO key schedule. Each round key is uniformly and independently
        sampled.
        """
        if n_rounds is None:
            n_rounds = self.default_rounds
        return np.random.randint(0, 2**16, size=n_rounds, dtype=np.uint16)
    
    def random_plaintexts(self, n_samples: int) -> np.ndarray:
        return np.random.randint(0, 2**32, size=n_samples, dtype=np.uint32)
    
    def get_default_delta_p(self) -> int:
        return 0x00000001  # Same as standard SIMON32


def test_simon32_irk():
    """Verify that SIMON32-IRK produces valid (non-trivial) ciphertexts."""
    cipher = Simon32IRK()
    
    # Test that different keys produce different ciphertexts
    key1 = cipher.random_key(n_rounds=10)
    key2 = cipher.random_key(n_rounds=10)
    plaintext = np.array([0x12345678], dtype=np.uint32)
    
    ct1 = cipher.encrypt(plaintext, 10, key1)
    ct2 = cipher.encrypt(plaintext, 10, key2)
    
    print(f"Plaintext:   0x{plaintext[0]:08x}")
    print(f"CT (key 1):  0x{ct1[0]:08x}")
    print(f"CT (key 2):  0x{ct2[0]:08x}")
    print(f"Different:   {ct1[0] != ct2[0]}")
    
    # Test encrypt_with_trace consistency
    ct_direct = cipher.encrypt(plaintext, 10, key1)
    ct_trace, trace = cipher.encrypt_with_trace(plaintext, 10, key1)
    print(f"Trace match: {ct_direct[0] == ct_trace[0]}")
    
    # Test that differential pairs produce non-trivial output diffs
    n = 10000
    P = cipher.random_plaintexts(n)
    delta_p = cipher.get_default_delta_p()
    P_prime = P ^ delta_p
    key = cipher.random_key(n_rounds=8)
    
    C = cipher.encrypt(P, 8, key)
    C_prime = cipher.encrypt(P_prime, 8, key)
    diff = C ^ C_prime
    
    # Should NOT be all zeros (that would mean no diffusion)
    nonzero_frac = np.mean(diff != 0)
    print(f"Non-zero diffs (8r): {nonzero_frac:.4f} (should be ~1.0)")
    
    return ct1[0] != ct2[0] and ct_direct[0] == ct_trace[0]


if __name__ == "__main__":
    test_simon32_irk()
