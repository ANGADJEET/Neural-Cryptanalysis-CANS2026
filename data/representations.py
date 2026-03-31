
import numpy as np
from typing import Dict, Optional, Tuple, Callable, List
from dataclasses import dataclass


@dataclass
class RepresentationInfo:
    name: str
    description: str
    shape_fn: Callable[[int], Tuple[int, ...]]
    requires_plaintext: bool = False
    requires_trace: bool = False
    model_affinity: List[str] = None
    
    def __post_init__(self):
        if self.model_affinity is None:
            self.model_affinity = ['mlp']


REPRESENTATION_REGISTRY: Dict[str, RepresentationInfo] = {}


def register_representation(name: str, description: str, shape_fn: Callable,
                           requires_plaintext: bool = False,
                           requires_trace: bool = False,
                           model_affinity: List[str] = None):
    def decorator(fn):
        REPRESENTATION_REGISTRY[name] = RepresentationInfo(
            name=name,
            description=description,
            shape_fn=shape_fn,
            requires_plaintext=requires_plaintext,
            requires_trace=requires_trace,
            model_affinity=model_affinity or ['mlp']
        )
        return fn
    return decorator


class RepresentationFactory:
    
    def __init__(self, block_size: int, word_size: int = 16):
        self.block_size = block_size
        self.word_size = word_size
        self.n_words = block_size // word_size
    
    def _to_bits(self, values: np.ndarray) -> np.ndarray:
        n_samples = len(values)
        bits = np.zeros((n_samples, self.block_size), dtype=np.float32)
        for i in range(self.block_size):
            bits[:, self.block_size - 1 - i] = (values >> i) & 1
        return bits
    
    def _to_words(self, values: np.ndarray) -> np.ndarray:
        n_samples = len(values)
        words = np.zeros((n_samples, self.n_words), dtype=np.float32)
        mask = (1 << self.word_size) - 1
        for i in range(self.n_words):
            words[:, self.n_words - 1 - i] = (values >> (i * self.word_size)) & mask
        words = words / (2**self.word_size - 1)
        return words
    
    def get_representation(
        self,
        name: str,
        C: np.ndarray,
        C_prime: np.ndarray,
        P: Optional[np.ndarray] = None,
        P_prime: Optional[np.ndarray] = None,
        intermediates: Optional[np.ndarray] = None,
        intermediates_prime: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        method_name = f'_repr_{name.lower()}'
        if hasattr(self, method_name):
            return getattr(self, method_name)(
                C, C_prime, P, P_prime, intermediates, intermediates_prime, **kwargs
            )
        else:
            raise ValueError(f"Unknown representation: {name}. Available: {self.list_representations()}")
    
    def list_representations(self) -> List[str]:
        return [
            'R1_raw_pair', 'R2_xor_diff', 'R3_concat', 'R4_bit_sliced',
            'R5_word_level', 'R6_joint_pc', 'R7_sequential', 'R8_statistical',
            'R9_masked'
        ]
    
    
    def _repr_r1_raw_pair(self, C, C_prime, P=None, P_prime=None, 
                          intermediates=None, intermediates_prime=None, **kwargs) -> np.ndarray:
        C_bits = self._to_bits(C)
        C_prime_bits = self._to_bits(C_prime)
        return np.stack([C_bits, C_prime_bits], axis=1)
    
    def _repr_r2_xor_diff(self, C, C_prime, P=None, P_prime=None,
                          intermediates=None, intermediates_prime=None, **kwargs) -> np.ndarray:
        diff = C ^ C_prime
        return self._to_bits(diff)
    
    def _repr_r3_concat(self, C, C_prime, P=None, P_prime=None,
                        intermediates=None, intermediates_prime=None, **kwargs) -> np.ndarray:
        C_bits = self._to_bits(C)
        C_prime_bits = self._to_bits(C_prime)
        return np.concatenate([C_bits, C_prime_bits], axis=1)
    
    def _repr_r4_bit_sliced(self, C, C_prime, P=None, P_prime=None,
                            intermediates=None, intermediates_prime=None, **kwargs) -> np.ndarray:
        C_bits = self._to_bits(C).reshape(-1, self.n_words, self.word_size)
        C_prime_bits = self._to_bits(C_prime).reshape(-1, self.n_words, self.word_size)
        return np.stack([C_bits, C_prime_bits], axis=1)
    
    def _repr_r5_word_level(self, C, C_prime, P=None, P_prime=None,
                            intermediates=None, intermediates_prime=None, **kwargs) -> np.ndarray:
        C_words = self._to_words(C)
        C_prime_words = self._to_words(C_prime)
        return np.concatenate([C_words, C_prime_words], axis=1)
    
    def _repr_r6_joint_pc(self, C, C_prime, P=None, P_prime=None,
                          intermediates=None, intermediates_prime=None, **kwargs) -> np.ndarray:
        if P is None or P_prime is None:
            raise ValueError("R6_joint_pc requires plaintext (P, P_prime)")
        
        P_bits = self._to_bits(P)
        C_bits = self._to_bits(C)
        P_prime_bits = self._to_bits(P_prime)
        C_prime_bits = self._to_bits(C_prime)
        
        return np.concatenate([P_bits, C_bits, P_prime_bits, C_prime_bits], axis=1)
    
    def _repr_r7_sequential(self, C, C_prime, P=None, P_prime=None,
                            intermediates=None, intermediates_prime=None, **kwargs) -> np.ndarray:
        if intermediates is None or intermediates_prime is None:
            raise ValueError("R7_sequential requires intermediate states")
        
        n_samples, n_rounds = intermediates.shape
        
        diffs = intermediates ^ intermediates_prime
        
        result = np.zeros((n_samples, n_rounds, self.block_size), dtype=np.float32)
        for r in range(n_rounds):
            result[:, r, :] = self._to_bits(diffs[:, r])
        
        return result
    
    def _repr_r8_statistical(self, C, C_prime, P=None, P_prime=None,
                             intermediates=None, intermediates_prime=None, **kwargs) -> np.ndarray:
        from .statistics import compute_statistical_features
        return compute_statistical_features(C, C_prime, self.block_size)
    
    def _repr_r9_masked(self, C, C_prime, P=None, P_prime=None,
                        intermediates=None, intermediates_prime=None,
                        mask_prob: float = 0.1, noise_std: float = 0.0, **kwargs) -> np.ndarray:
        diff = C ^ C_prime
        bits = self._to_bits(diff)
        
        if mask_prob > 0:
            mask = np.random.random(bits.shape) < mask_prob
            bits = np.where(mask, 0.5, bits)
        
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, bits.shape)
            bits = np.clip(bits + noise, 0, 1)
        
        return bits.astype(np.float32)
    
    
    def get_output_shape(self, name: str, n_rounds: int = 1) -> Tuple[int, ...]:
        shapes = {
            'R1_raw_pair': (2, self.block_size),
            'R2_xor_diff': (self.block_size,),
            'R3_concat': (2 * self.block_size,),
            'R4_bit_sliced': (2, self.n_words, self.word_size),
            'R5_word_level': (2 * self.n_words,),
            'R6_joint_pc': (4 * self.block_size,),
            'R7_sequential': (n_rounds, self.block_size),
            'R8_statistical': (self._get_stat_feature_count(),),
            'R9_masked': (self.block_size,),
        }
        return shapes.get(name.upper(), shapes.get(name, None))
    
    def _get_stat_feature_count(self) -> int:
        return 1 + self.block_size + 4


def get_representation(
    name: str,
    C: np.ndarray,
    C_prime: np.ndarray,
    block_size: int = 32,
    **kwargs
) -> np.ndarray:
    factory = RepresentationFactory(block_size=block_size)
    return factory.get_representation(name, C, C_prime, **kwargs)
