
import numpy as np
from typing import Tuple


def hamming_weight(x: np.ndarray) -> np.ndarray:
    count = np.zeros_like(x, dtype=np.int32)
    temp = x.copy()
    while np.any(temp > 0):
        count += (temp & 1).astype(np.int32)
        temp >>= 1
    return count


def bit_correlation(x: np.ndarray, y: np.ndarray, block_size: int) -> np.ndarray:
    n_samples = len(x)
    correlations = np.zeros(block_size, dtype=np.float32)
    
    for i in range(block_size):
        x_bit = ((x >> i) & 1).astype(np.float32)
        y_bit = ((y >> i) & 1).astype(np.float32)
        
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
    n_samples = len(C)
    diff = C ^ C_prime
    
    features = []
    
    hw = hamming_weight(diff).astype(np.float32) / block_size
    features.append(hw.reshape(-1, 1))
    
    bits = np.zeros((n_samples, block_size), dtype=np.float32)
    for i in range(block_size):
        bits[:, block_size - 1 - i] = (diff >> i) & 1
    features.append(bits)
    
    half_size = block_size // 2
    half_mask = (1 << half_size) - 1
    
    left_half = (diff >> half_size) & half_mask
    right_half = diff & half_mask
    
    left_hw = hamming_weight(left_half).astype(np.float32) / half_size
    features.append(left_hw.reshape(-1, 1))
    
    right_hw = hamming_weight(right_half).astype(np.float32) / half_size
    features.append(right_hw.reshape(-1, 1))
    
    cross_xor = left_half ^ right_half
    cross_hw = hamming_weight(cross_xor).astype(np.float32) / half_size
    features.append(cross_hw.reshape(-1, 1))
    
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
    diff = C ^ C_prime
    n_samples = len(diff)
    
    biases = np.zeros(block_size, dtype=np.float32)
    
    for i in range(block_size):
        bit = (diff >> i) & 1
        prob_one = np.mean(bit)
        biases[block_size - 1 - i] = abs(prob_one - 0.5)
    
    return biases, np.max(biases)
