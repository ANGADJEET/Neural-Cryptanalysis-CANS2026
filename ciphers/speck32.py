
import numpy as np
from typing import Tuple, List
from .base import BaseCipher


class Speck32(BaseCipher):
    
    BLOCK_SIZE = 32
    KEY_SIZE = 64
    WORD_SIZE = 16
    ROUNDS = 22
    ALPHA = 7
    BETA = 2
    
    def __init__(self):
        super().__init__(
            block_size=self.BLOCK_SIZE,
            key_size=self.KEY_SIZE,
            max_rounds=self.ROUNDS
        )
        self.word_mask = (1 << self.WORD_SIZE) - 1
    
    def _ror(self, x: np.ndarray, r: int) -> np.ndarray:
        r = r % self.WORD_SIZE
        return ((x >> r) | (x << (self.WORD_SIZE - r))) & self.word_mask
    
    def _rol(self, x: np.ndarray, r: int) -> np.ndarray:
        r = r % self.WORD_SIZE
        return ((x << r) | (x >> (self.WORD_SIZE - r))) & self.word_mask
    
    def _expand_key(self, key: np.ndarray, n_rounds: int) -> np.ndarray:
        m = 4
        
        l = [int(key[2]), int(key[1]), int(key[0])]
        k = int(key[3])
        
        round_keys = [k]
        
        for i in range(n_rounds - 1):
            l_idx = i % (m - 1)
            l_new = ((k + self._ror_scalar(l[l_idx], self.ALPHA)) & self.word_mask) ^ i
            
            k = self._rol_scalar(k, self.BETA) ^ l_new
            
            l[l_idx] = l_new
            round_keys.append(k)
        
        return np.array(round_keys, dtype=np.uint16)
    
    def _ror_scalar(self, x: int, r: int) -> int:
        r = r % self.WORD_SIZE
        return ((x >> r) | (x << (self.WORD_SIZE - r))) & self.word_mask
    
    def _rol_scalar(self, x: int, r: int) -> int:
        r = r % self.WORD_SIZE
        return ((x << r) | (x >> (self.WORD_SIZE - r))) & self.word_mask
    
    def _round_function(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = ((self._ror(x, self.ALPHA) + y) & self.word_mask) ^ k
        y = self._rol(y, self.BETA) ^ x
        return x, y
    
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
        return 0x00400000


def test_speck32():
    cipher = Speck32()
    
    key = np.array([0x1918, 0x1110, 0x0908, 0x0100], dtype=np.uint16)
    plaintext = np.array([0x6574694c], dtype=np.uint32)
    
    ciphertext = cipher.encrypt(plaintext, 22, key)
    expected = 0xa86842f2
    
    print(f"Plaintext:  0x{plaintext[0]:08x}")
    print(f"Ciphertext: 0x{ciphertext[0]:08x}")
    print(f"Expected:   0x{expected:08x}")
    print(f"Match: {ciphertext[0] == expected}")
    
    return ciphertext[0] == expected


if __name__ == "__main__":
    test_speck32()
