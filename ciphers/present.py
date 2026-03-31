
import numpy as np
from typing import Tuple, List
from .base import BaseCipher


class Present(BaseCipher):
    
    BLOCK_SIZE = 64
    KEY_SIZE = 80
    ROUNDS = 31
    
    SBOX = np.array([
        0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
        0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2
    ], dtype=np.uint8)
    
    SBOX_INV = np.array([
        0x5, 0xE, 0xF, 0x8, 0xC, 0x1, 0x2, 0xD,
        0xB, 0x4, 0x6, 0x3, 0x0, 0x7, 0x9, 0xA
    ], dtype=np.uint8)
    
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
        result = np.zeros_like(state)
        for i in range(16):
            nibble = (state >> (4 * i)) & 0xF
            result |= self.SBOX[nibble.astype(np.int64)].astype(np.uint64) << (4 * i)
        return result
    
    def _perm_layer(self, state: np.ndarray) -> np.ndarray:
        result = np.zeros_like(state)
        for i in range(64):
            bit = (state >> i) & 1
            result |= bit << self.PERM[i]
        return result
    
    def _expand_key(self, key: np.ndarray, n_rounds: int) -> np.ndarray:
        key_high = int(key[0])
        key_low = int(key[1])
        
        round_keys = []
        
        for i in range(n_rounds + 1):
            rk = (key_high << 48) | (key_low >> 16)
            round_keys.append(rk & ((1 << 64) - 1))
            
            combined = (key_high << 64) | key_low
            combined = ((combined << 61) | (combined >> 19)) & ((1 << 80) - 1)
            
            top_nibble = (combined >> 76) & 0xF
            combined = (combined & ((1 << 76) - 1)) | (int(self.SBOX[top_nibble]) << 76)
            
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
        n_rounds = min(n_rounds, self.ROUNDS)
        round_keys = self._expand_key(key, n_rounds)
        
        state = plaintext.astype(np.uint64)
        
        for i in range(n_rounds):
            state = state ^ round_keys[i]
            state = self._sbox_layer(state)
            state = self._perm_layer(state)
        
        state = state ^ round_keys[n_rounds]
        
        return state
    
    def encrypt_with_trace(
        self, 
        plaintext: np.ndarray, 
        n_rounds: int, 
        key: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        n_rounds = min(n_rounds, self.ROUNDS)
        round_keys = self._expand_key(key, n_rounds)
        
        state = plaintext.astype(np.uint64)
        intermediate_states = []
        
        for i in range(n_rounds):
            state = state ^ round_keys[i]
            state = self._sbox_layer(state)
            state = self._perm_layer(state)
            intermediate_states.append(state.copy())
        
        state = state ^ round_keys[n_rounds]
        
        return state, intermediate_states
    
    def random_key(self) -> np.ndarray:
        return np.array([
            np.random.randint(0, 2**16),
            np.random.randint(0, 2**64, dtype=np.uint64)
        ], dtype=np.uint64)
    
    def random_plaintexts(self, n_samples: int) -> np.ndarray:
        return np.random.randint(0, 2**63, size=n_samples, dtype=np.uint64) * 2 + \
               np.random.randint(0, 2, size=n_samples, dtype=np.uint64)
    
    def get_default_delta_p(self) -> int:
        return 0x0000000000000001


def test_present():
    cipher = Present()
    
    key = np.array([0x0000, 0x0000000000000000], dtype=np.uint64)
    plaintext = np.array([0x0000000000000000], dtype=np.uint64)
    
    ciphertext = cipher.encrypt(plaintext, 31, key)
    expected = 0x5579C1387B228445
    
    print(f"Plaintext:  0x{plaintext[0]:016x}")
    print(f"Ciphertext: 0x{ciphertext[0]:016x}")
    print(f"Expected:   0x{expected:016x}")
    print(f"Match: {ciphertext[0] == expected}")
    
    return ciphertext[0] == expected


if __name__ == "__main__":
    test_present()
