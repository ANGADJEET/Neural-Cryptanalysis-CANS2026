
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
    """Count exact output diff matches. Only useful for 1-2 rounds.
    For multi-round comparison, use compute_classical_distinguisher_accuracy().
    """
    key = cipher.random_key()
    P = cipher.random_plaintexts(n_samples)
    P_prime = P ^ diff_in
    
    C = cipher.encrypt(P, n_rounds, key)
    C_prime = cipher.encrypt(P_prime, n_rounds, key)
    
    actual_diff = C ^ C_prime
    matches = np.sum(actual_diff == diff_out)
    
    return matches / n_samples


def compute_classical_distinguisher_accuracy(
    cipher,
    diff_in: int,
    n_rounds: int,
    n_samples: int = 100000,
    n_keys: int = 5
) -> float:
    """Classical distinguisher using best per-bit bias of ΔC.
    
    Compares the bit-level statistics of real differential pairs
    (E_k(P), E_k(P⊕ΔP)) against random pairs (E_k(Q₁), E_k(Q₂))
    and finds the single bit with maximum separating power.
    
    Returns the best achievable accuracy over all bit positions.
    """
    accs = []
    for _ in range(n_keys):
        key = cipher.random_key()
        half = n_samples // 2

        # Positive: real differential pairs
        P = cipher.random_plaintexts(half)
        C = cipher.encrypt(P, n_rounds, key)
        C_prime = cipher.encrypt((P ^ diff_in).astype(P.dtype), n_rounds, key)
        diff_pos = C ^ C_prime

        # Negative: independent random pairs (same key, no differential)
        Q1 = cipher.random_plaintexts(half)
        Q2 = cipher.random_plaintexts(half)
        diff_neg = cipher.encrypt(Q1, n_rounds, key) ^ cipher.encrypt(Q2, n_rounds, key)

        # Find the single bit with maximum distinguishing power
        best_acc = 0.5
        for bit in range(cipher.block_size):
            pos_bit = ((diff_pos >> bit) & 1).astype(np.float32)
            neg_bit = ((diff_neg >> bit) & 1).astype(np.float32)

            pos_mean = pos_bit.mean()
            neg_mean = neg_bit.mean()

            if abs(pos_mean - neg_mean) > 0.001:
                # Use the midpoint as threshold
                threshold = (pos_mean + neg_mean) / 2
                all_bits = np.concatenate([pos_bit, neg_bit])
                all_labels = np.concatenate([np.ones(half), np.zeros(half)])
                # Predict based on which side of threshold the bit falls
                if pos_mean > neg_mean:
                    preds = (all_bits > threshold).astype(float)
                else:
                    preds = (all_bits < threshold).astype(float)
                acc = float(np.mean(preds == all_labels))
                best_acc = max(best_acc, acc)
        accs.append(best_acc)
    return float(np.mean(accs))


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

