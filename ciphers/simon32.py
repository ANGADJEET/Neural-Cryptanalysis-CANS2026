
import numpy as np
from typing import Tuple, List
from .base import BaseCipher


class Simon32(BaseCipher):
    
    BLOCK_SIZE = 32
    KEY_SIZE = 64
    WORD_SIZE = 16
    ROUNDS = 32
    
    Z_SEQ = [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 
             0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1,
             1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1,
             0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
    
    def __init__(self):
        super().__init__(
            block_size=self.BLOCK_SIZE,
            key_size=self.KEY_SIZE,
            max_rounds=self.ROUNDS
        )
        self.word_mask = (1 << self.WORD_SIZE) - 1
    
    def _rol(self, x: np.ndarray, r: int) -> np.ndarray:
        r = r % self.WORD_SIZE
        return ((x << r) | (x >> (self.WORD_SIZE - r))) & self.word_mask
    
    def _ror_scalar(self, x: int, r: int) -> int:
        r = r % self.WORD_SIZE
        return ((x >> r) | (x << (self.WORD_SIZE - r))) & self.word_mask
    
    def _expand_key(self, key: np.ndarray, n_rounds: int) -> np.ndarray:
        m = 4
        c = 0xFFFC
        
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
    
    def encrypt(
        self, 
        plaintext: np.ndarray, 
        n_rounds: int, 
        key: np.ndarray
    ) -> np.ndarray:
        n_rounds = min(n_rounds, self.ROUNDS)
        round_keys = self._expand_key(key, n_rounds)
        
        if plaintext.ndim == 1:
            x = ((plaintext >> 16) & self.word_mask).astype(np.uint16)
            y = (plaintext & self.word_mask).astype(np.uint16)
            return_packed = True
        else:
            x = plaintext[:, 0].astype(np.uint16)
            y = plaintext[:, 1].astype(np.uint16)
            return_packed = False
        
        for i in range(n_rounds):
            x, y = self._round_function(x, y, round_keys[i])
        
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
    
    def random_key(self) -> np.ndarray:
        return np.random.randint(0, 2**16, size=4, dtype=np.uint16)
    
    def random_plaintexts(self, n_samples: int) -> np.ndarray:
        return np.random.randint(0, 2**32, size=n_samples, dtype=np.uint32)
    
    def get_default_delta_p(self) -> int:
        return 0x00000001


def test_simon32():
    cipher = Simon32()
    
    key = np.array([0x1918, 0x1110, 0x0908, 0x0100], dtype=np.uint16)
    plaintext = np.array([0x65656877], dtype=np.uint32)
    
    ciphertext = cipher.encrypt(plaintext, 32, key)
    expected = 0xc69be9bb
    
    print(f"Plaintext:  0x{plaintext[0]:08x}")
    print(f"Ciphertext: 0x{ciphertext[0]:08x}")
    print(f"Expected:   0x{expected:08x}")
    print(f"Match: {ciphertext[0] == expected}")
    
    return ciphertext[0] == expected


if __name__ == "__main__":
    test_simon32()
