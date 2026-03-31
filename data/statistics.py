"""
Statistical feature computation for neural cryptanalysis.

Computes various statistical features from ciphertext pairs:
- Hamming weight of differences
- Bit correlations
- Empirical probability estimates
"""

import numpy as np
from typing import Tuple


def hamming_weight(x: np.ndarray) -> np.ndarray:
    """
    Compute Hamming weight (population count) of integer values.
    
    Args:
        x: Array of integers
        
    Returns:
        Array of Hamming weights
    """
    # Use lookup table for efficiency
    count = np.zeros_like(x, dtype=np.int32)
    temp = x.copy()
    while np.any(temp > 0):
        count += (temp & 1).astype(np.int32)
        temp >>= 1
    return count


def bit_correlation(x: np.ndarray, y: np.ndarray, block_size: int) -> np.ndarray:
    """
    Compute bit-wise correlation between two ciphertext arrays.
    
    Args:
        x: First ciphertext array (N,)
        y: Second ciphertext array (N,)
        block_size: Number of bits
        
    Returns:
        Correlation coefficient for each bit position (block_size,)
    """
    n_samples = len(x)
    correlations = np.zeros(block_size, dtype=np.float32)
    
    for i in range(block_size):
        x_bit = ((x >> i) & 1).astype(np.float32)
        y_bit = ((y >> i) & 1).astype(np.float32)
        
        # Pearson correlation
        x_mean = x_bit.mean()
        y_mean = y_bit.mean()
        
        x_centered = x_bit - x_mean
        y_centered = y_bit - y_mean
        
        numerator = np.sum(x_centered * y_centered)
        denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
        
        if denominator > 1e-10:
            correlations[block_size - 1 - i] = numerator / denominator
        else:
            correlations[block_size - 1 - i] = 0.0
    
    return correlations


def compute_statistical_features(
    C: np.ndarray,
    C_prime: np.ndarray,
    block_size: int
) -> np.ndarray:
    """
    Compute comprehensive statistical features from ciphertext pairs.
    
    Features computed:
    1. Normalized Hamming weight of ΔC (1 feature)
    2. Per-bit values of ΔC (block_size features)
    3. Block-wise statistics (4 features):
       - Left half HW, Right half HW
       - Cross-half XOR HW
       - Bit balance (difference in HW between halves)
    
    Args:
        C: First ciphertext array (N,)
        C_prime: Second ciphertext array (N,)
        block_size: Number of bits in block
        
    Returns:
        Feature array of shape (N, 1 + block_size + 4)
    """
    n_samples = len(C)
    diff = C ^ C_prime
    
    features = []
    
    # 1. Normalized Hamming weight of ΔC
    hw = hamming_weight(diff).astype(np.float32) / block_size
    features.append(hw.reshape(-1, 1))
    
    # 2. Per-bit values of ΔC
    bits = np.zeros((n_samples, block_size), dtype=np.float32)
    for i in range(block_size):
        bits[:, block_size - 1 - i] = (diff >> i) & 1
    features.append(bits)
    
    # 3. Block-wise statistics
    half_size = block_size // 2
    half_mask = (1 << half_size) - 1
    
    # Left and right halves of difference
    left_half = (diff >> half_size) & half_mask
    right_half = diff & half_mask
    
    # Left half HW (normalized)
    left_hw = hamming_weight(left_half).astype(np.float32) / half_size
    features.append(left_hw.reshape(-1, 1))
    
    # Right half HW (normalized)
    right_hw = hamming_weight(right_half).astype(np.float32) / half_size
    features.append(right_hw.reshape(-1, 1))
    
    # Cross-half XOR HW
    cross_xor = left_half ^ right_half
    cross_hw = hamming_weight(cross_xor).astype(np.float32) / half_size
    features.append(cross_hw.reshape(-1, 1))
    
    # Bit balance (difference in HW between halves, normalized to [-1, 1])
    balance = (left_hw - right_hw)
    features.append(balance.reshape(-1, 1))
    
    return np.concatenate(features, axis=1)


def compute_differential_probability(
    diff_in: int,
    diff_out: int,
    cipher,
    n_samples: int = 100000,
    n_rounds: int = 1
) -> float:
    """
    Empirically estimate differential probability P(diff_out | diff_in).
    
    Args:
        diff_in: Input difference
        diff_out: Output difference
        cipher: Cipher instance
        n_samples: Number of samples for estimation
        n_rounds: Number of rounds
        
    Returns:
        Estimated probability
    """
    key = cipher.random_key()
    P = cipher.random_plaintexts(n_samples)
    P_prime = P ^ diff_in
    
    C = cipher.encrypt(P, n_rounds, key)
    C_prime = cipher.encrypt(P_prime, n_rounds, key)
    
    actual_diff = C ^ C_prime
    matches = np.sum(actual_diff == diff_out)
    
    return matches / n_samples


def estimate_bias(
    C: np.ndarray,
    C_prime: np.ndarray,
    block_size: int
) -> Tuple[np.ndarray, float]:
    """
    Estimate linear approximation bias for each bit position.
    
    For random permutation, each bit of ΔC should be unbiased (p=0.5).
    For cipher, there may be exploitable biases.
    
    Args:
        C: First ciphertext array
        C_prime: Second ciphertext array
        block_size: Number of bits
        
    Returns:
        (per_bit_bias, overall_bias)
        per_bit_bias: Array of biases for each bit position
        overall_bias: Maximum absolute bias across all bits
    """
    diff = C ^ C_prime
    n_samples = len(diff)
    
    biases = np.zeros(block_size, dtype=np.float32)
    
    for i in range(block_size):
        bit = (diff >> i) & 1
        prob_one = np.mean(bit)
        biases[block_size - 1 - i] = abs(prob_one - 0.5)
    
    return biases, np.max(biases)
