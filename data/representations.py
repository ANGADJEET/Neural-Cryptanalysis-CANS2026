"""
Input representation factory for neural cryptanalysis.

Implements all 9 representation types from the project specification:
R1: Raw Pair (C, C')
R2: XOR Difference (C ⊕ C')
R3: Concatenated (C || C')
R4: Bit-Sliced (for CNN)
R5: Word-Level (for ARX ciphers)
R6: Joint P-C (P || C || P' || C')
R7: Sequential (round-wise differences)
R8: Statistical (Hamming weight, correlations)
R9: Masked (with random noise/masking)
"""

import numpy as np
from typing import Dict, Optional, Tuple, Callable, List
from dataclasses import dataclass


@dataclass
class RepresentationInfo:
    """Metadata about a representation type."""
    name: str
    description: str
    shape_fn: Callable[[int], Tuple[int, ...]]  # block_size -> output shape
    requires_plaintext: bool = False
    requires_trace: bool = False
    model_affinity: List[str] = None  # Suggested models
    
    def __post_init__(self):
        if self.model_affinity is None:
            self.model_affinity = ['mlp']


# Registry of all representations
REPRESENTATION_REGISTRY: Dict[str, RepresentationInfo] = {}


def register_representation(name: str, description: str, shape_fn: Callable,
                           requires_plaintext: bool = False,
                           requires_trace: bool = False,
                           model_affinity: List[str] = None):
    """Decorator to register a representation."""
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
    """
    Factory for converting raw data to various input representations.
    
    Usage:
        factory = RepresentationFactory(block_size=32)
        X = factory.get_representation('R2_xor_diff', C, C_prime)
    """
    
    def __init__(self, block_size: int, word_size: int = 16):
        """
        Initialize factory.
        
        Args:
            block_size: Cipher block size in bits
            word_size: Word size for word-level representations
        """
        self.block_size = block_size
        self.word_size = word_size
        self.n_words = block_size // word_size
    
    def _to_bits(self, values: np.ndarray) -> np.ndarray:
        """Convert integer values to bit array (N, block_size)."""
        n_samples = len(values)
        bits = np.zeros((n_samples, self.block_size), dtype=np.float32)
        for i in range(self.block_size):
            bits[:, self.block_size - 1 - i] = (values >> i) & 1
        return bits
    
    def _to_words(self, values: np.ndarray) -> np.ndarray:
        """Convert integer values to word array (N, n_words)."""
        n_samples = len(values)
        words = np.zeros((n_samples, self.n_words), dtype=np.float32)
        mask = (1 << self.word_size) - 1
        for i in range(self.n_words):
            words[:, self.n_words - 1 - i] = (values >> (i * self.word_size)) & mask
        # Normalize words to [0, 1]
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
        """
        Get specified representation of the data.
        
        Args:
            name: Representation name (e.g., 'R2_xor_diff')
            C: Ciphertext array (N,)
            C_prime: Paired ciphertext array (N,)
            P: Plaintext (optional, for R6)
            P_prime: Paired plaintext (optional, for R6)
            intermediates: Round states for P (optional, for R7)
            intermediates_prime: Round states for P' (optional, for R7)
            **kwargs: Additional arguments for specific representations
            
        Returns:
            Representation array
        """
        method_name = f'_repr_{name.lower()}'
        if hasattr(self, method_name):
            return getattr(self, method_name)(
                C, C_prime, P, P_prime, intermediates, intermediates_prime, **kwargs
            )
        else:
            raise ValueError(f"Unknown representation: {name}. Available: {self.list_representations()}")
    
    def list_representations(self) -> List[str]:
        """List all available representations."""
        return [
            'R1_raw_pair', 'R2_xor_diff', 'R3_concat', 'R4_bit_sliced',
            'R5_word_level', 'R6_joint_pc', 'R7_sequential', 'R8_statistical',
            'R9_masked'
        ]
    
    # ==================== Representation Implementations ====================
    
    def _repr_r1_raw_pair(self, C, C_prime, P=None, P_prime=None, 
                          intermediates=None, intermediates_prime=None, **kwargs) -> np.ndarray:
        """
        R1: Raw Pair (C, C')
        Shape: (N, 2, block_size) as bits
        Model affinity: CNN, Siamese
        """
        C_bits = self._to_bits(C)
        C_prime_bits = self._to_bits(C_prime)
        return np.stack([C_bits, C_prime_bits], axis=1)
    
    def _repr_r2_xor_diff(self, C, C_prime, P=None, P_prime=None,
                          intermediates=None, intermediates_prime=None, **kwargs) -> np.ndarray:
        """
        R2: XOR Difference (C ⊕ C')
        Shape: (N, block_size) as bits
        Model affinity: MLP, CNN
        """
        diff = C ^ C_prime
        return self._to_bits(diff)
    
    def _repr_r3_concat(self, C, C_prime, P=None, P_prime=None,
                        intermediates=None, intermediates_prime=None, **kwargs) -> np.ndarray:
        """
        R3: Concatenated (C || C')
        Shape: (N, 2*block_size) as bits
        Model affinity: MLP
        """
        C_bits = self._to_bits(C)
        C_prime_bits = self._to_bits(C_prime)
        return np.concatenate([C_bits, C_prime_bits], axis=1)
    
    def _repr_r4_bit_sliced(self, C, C_prime, P=None, P_prime=None,
                            intermediates=None, intermediates_prime=None, **kwargs) -> np.ndarray:
        """
        R4: Bit-Sliced for CNN
        Shape: (N, 2, n_words, word_size)
        Model affinity: CNN
        """
        C_bits = self._to_bits(C).reshape(-1, self.n_words, self.word_size)
        C_prime_bits = self._to_bits(C_prime).reshape(-1, self.n_words, self.word_size)
        return np.stack([C_bits, C_prime_bits], axis=1)
    
    def _repr_r5_word_level(self, C, C_prime, P=None, P_prime=None,
                            intermediates=None, intermediates_prime=None, **kwargs) -> np.ndarray:
        """
        R5: Word-Level (normalized words)
        Shape: (N, 2*n_words)
        Model affinity: MLP, RNN
        """
        C_words = self._to_words(C)
        C_prime_words = self._to_words(C_prime)
        return np.concatenate([C_words, C_prime_words], axis=1)
    
    def _repr_r6_joint_pc(self, C, C_prime, P=None, P_prime=None,
                          intermediates=None, intermediates_prime=None, **kwargs) -> np.ndarray:
        """
        R6: Joint P-C (P || C || P' || C')
        Shape: (N, 4*block_size) as bits
        Model affinity: MLP (upper bound analysis)
        
        Note: Requires plaintexts. For random samples, P and P' are zeros.
        """
        if P is None or P_prime is None:
            raise ValueError("R6_joint_pc requires plaintext (P, P_prime)")
        
        P_bits = self._to_bits(P)
        C_bits = self._to_bits(C)
        P_prime_bits = self._to_bits(P_prime)
        C_prime_bits = self._to_bits(C_prime)
        
        return np.concatenate([P_bits, C_bits, P_prime_bits, C_prime_bits], axis=1)
    
    def _repr_r7_sequential(self, C, C_prime, P=None, P_prime=None,
                            intermediates=None, intermediates_prime=None, **kwargs) -> np.ndarray:
        """
        R7: Sequential (round-wise differences)
        Shape: (N, n_rounds, block_size) as bits
        Model affinity: RNN, LSTM
        
        Note: Requires intermediate states from white-box access.
        """
        if intermediates is None or intermediates_prime is None:
            raise ValueError("R7_sequential requires intermediate states")
        
        # intermediates shape: (N, n_rounds)
        n_samples, n_rounds = intermediates.shape
        
        # Compute XOR differences at each round
        diffs = intermediates ^ intermediates_prime  # (N, n_rounds)
        
        # Convert each round difference to bits
        result = np.zeros((n_samples, n_rounds, self.block_size), dtype=np.float32)
        for r in range(n_rounds):
            result[:, r, :] = self._to_bits(diffs[:, r])
        
        return result
    
    def _repr_r8_statistical(self, C, C_prime, P=None, P_prime=None,
                             intermediates=None, intermediates_prime=None, **kwargs) -> np.ndarray:
        """
        R8: Statistical features
        Shape: (N, k) where k depends on features computed
        Model affinity: MLP
        
        Features:
        - Hamming weight of ΔC
        - Bit correlation statistics
        - Empirical probabilities
        """
        from .statistics import compute_statistical_features
        return compute_statistical_features(C, C_prime, self.block_size)
    
    def _repr_r9_masked(self, C, C_prime, P=None, P_prime=None,
                        intermediates=None, intermediates_prime=None,
                        mask_prob: float = 0.1, noise_std: float = 0.0, **kwargs) -> np.ndarray:
        """
        R9: Masked/Noisy representation for robustness testing
        Shape: (N, block_size) as bits
        Model affinity: MLP, CNN
        
        Args:
            mask_prob: Probability of masking each bit (setting to 0.5)
            noise_std: Standard deviation of Gaussian noise to add
        """
        # Start with XOR difference
        diff = C ^ C_prime
        bits = self._to_bits(diff)
        
        # Apply random masking
        if mask_prob > 0:
            mask = np.random.random(bits.shape) < mask_prob
            bits = np.where(mask, 0.5, bits)  # Set masked bits to 0.5 (uninformative)
        
        # Add Gaussian noise
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, bits.shape)
            bits = np.clip(bits + noise, 0, 1)
        
        return bits.astype(np.float32)
    
    # ==================== Utility Methods ====================
    
    def get_output_shape(self, name: str, n_rounds: int = 1) -> Tuple[int, ...]:
        """Get output shape for a representation (excluding batch dimension)."""
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
        """Get number of statistical features."""
        # HW(ΔC) + per-bit correlations + block-wise features
        return 1 + self.block_size + 4


# Convenience functions
def get_representation(
    name: str,
    C: np.ndarray,
    C_prime: np.ndarray,
    block_size: int = 32,
    **kwargs
) -> np.ndarray:
    """
    Convenience function to get a representation.
    
    Args:
        name: Representation name
        C, C_prime: Ciphertext arrays
        block_size: Cipher block size
        **kwargs: Additional arguments
        
    Returns:
        Representation array
    """
    factory = RepresentationFactory(block_size=block_size)
    return factory.get_representation(name, C, C_prime, **kwargs)
