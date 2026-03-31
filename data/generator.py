"""
Dataset generator for neural differential cryptanalysis.

Generates labeled datasets of cipher/random ciphertext pairs
for training neural distinguishers.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from ciphers import get_cipher, BaseCipher
from ciphers.random_permutation import RandomPermutation


class CipherDataGenerator:
    """
    Main dataset generator for neural cryptanalysis experiments.
    
    Generates balanced datasets with:
    - Label 1: Cipher-generated (C, C') pairs from (P, P⊕ΔP)
    - Label 0: Random permutation pairs
    
    Also supports white-box mode with intermediate round states.
    """
    
    def __init__(
        self,
        cipher: Union[str, BaseCipher],
        n_rounds: int,
        delta_p: int,
        key: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize data generator.
        
        Args:
            cipher: Cipher name ('speck32', 'simon32', 'present') or instance
            n_rounds: Number of encryption rounds
            delta_p: Input difference (XOR value)
            key: Fixed encryption key (random if None)
            seed: Random seed for reproducibility
        """
        if isinstance(cipher, str):
            self.cipher = get_cipher(cipher)
        else:
            self.cipher = cipher
        
        self.n_rounds = n_rounds
        self.delta_p = delta_p
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate or set key
        if key is None:
            self.key = self.cipher.random_key()
        else:
            self.key = key
        
        # Random baseline
        self.random_perm = RandomPermutation(block_size=self.cipher.block_size)
    
    def generate_cipher_samples(
        self,
        n_samples: int,
        include_plaintext: bool = False,
        include_trace: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate cipher-encrypted samples (label=1).
        
        Args:
            n_samples: Number of samples to generate
            include_plaintext: Include P, P' in output
            include_trace: Include intermediate round states (white-box)
            
        Returns:
            Dictionary with keys: 'C', 'C_prime', 'labels'
            Optionally: 'P', 'P_prime', 'intermediates', 'intermediates_prime'
        """
        # Generate random plaintexts
        P = self.cipher.random_plaintexts(n_samples)
        P_prime = P ^ self.delta_p
        
        # Encrypt
        if include_trace:
            C, states = self.cipher.encrypt_with_trace(P, self.n_rounds, self.key)
            C_prime, states_prime = self.cipher.encrypt_with_trace(P_prime, self.n_rounds, self.key)
        else:
            C = self.cipher.encrypt(P, self.n_rounds, self.key)
            C_prime = self.cipher.encrypt(P_prime, self.n_rounds, self.key)
        
        result = {
            'C': C,
            'C_prime': C_prime,
            'labels': np.ones(n_samples, dtype=np.uint8)
        }
        
        if include_plaintext:
            result['P'] = P
            result['P_prime'] = P_prime
        
        if include_trace:
            # Stack intermediate states: (n_samples, n_rounds)
            result['intermediates'] = np.stack(states, axis=1)
            result['intermediates_prime'] = np.stack(states_prime, axis=1)
        
        return result
    
    def generate_random_samples(
        self,
        n_samples: int,
        include_plaintext: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate random permutation samples (label=0).
        
        Args:
            n_samples: Number of samples to generate
            include_plaintext: Include random P, P' values (for R6 compatibility)
            
        Returns:
            Dictionary with keys: 'C', 'C_prime', 'labels'
            Optionally: 'P', 'P_prime' (random values, not semantically meaningful)
        """
        C, C_prime = self.random_perm.generate_random_pairs(n_samples)
        
        result = {
            'C': C,
            'C_prime': C_prime,
            'labels': np.zeros(n_samples, dtype=np.uint8)
        }
        
        if include_plaintext:
            # Use random values instead of zeros so R6 representation
            # doesn't create a trivially detectable signal
            result['P'] = self.cipher.random_plaintexts(n_samples)
            result['P_prime'] = self.cipher.random_plaintexts(n_samples)
        
        return result
    
    def generate_balanced_dataset(
        self,
        n_samples: int,
        include_plaintext: bool = False,
        include_trace: bool = False,
        shuffle: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate balanced dataset with equal cipher and random samples.
        
        Args:
            n_samples: Total number of samples (half cipher, half random)
            include_plaintext: Include plaintexts
            include_trace: Include intermediate states
            shuffle: Shuffle the dataset
            
        Returns:
            Dictionary with combined dataset
        """
        n_each = n_samples // 2
        
        # Generate both classes
        cipher_data = self.generate_cipher_samples(
            n_each, include_plaintext, include_trace
        )
        random_data = self.generate_random_samples(n_each, include_plaintext)
        
        # Combine
        result = {
            'C': np.concatenate([cipher_data['C'], random_data['C']]),
            'C_prime': np.concatenate([cipher_data['C_prime'], random_data['C_prime']]),
            'labels': np.concatenate([cipher_data['labels'], random_data['labels']])
        }
        
        if include_plaintext:
            # Random samples now get random plaintexts (not zeros)
            # to avoid trivially detectable signal in R6 representation
            result['P'] = np.concatenate([cipher_data['P'], random_data['P']])
            result['P_prime'] = np.concatenate([cipher_data['P_prime'], random_data['P_prime']])
        
        if include_trace:
            # Random samples have no trace - fill with zeros
            dummy_trace = np.zeros_like(cipher_data['intermediates'])
            result['intermediates'] = np.concatenate([cipher_data['intermediates'], dummy_trace])
            result['intermediates_prime'] = np.concatenate([cipher_data['intermediates_prime'], dummy_trace])
        
        # Shuffle
        if shuffle:
            perm = np.random.permutation(len(result['labels']))
            for key in result:
                result[key] = result[key][perm]
        
        return result
    
    def generate_round_sweep(
        self,
        n_samples_per_round: int,
        min_rounds: int = 1,
        max_rounds: Optional[int] = None
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Generate datasets for multiple round counts.
        
        Useful for accuracy vs rounds experiments.
        
        Args:
            n_samples_per_round: Samples per round configuration
            min_rounds: Minimum rounds
            max_rounds: Maximum rounds (default: cipher's max)
            
        Returns:
            Dict mapping round count to dataset
        """
        if max_rounds is None:
            max_rounds = self.cipher.max_rounds
        
        datasets = {}
        for r in tqdm(range(min_rounds, max_rounds + 1), desc="Generating round sweep"):
            # Temporarily change rounds
            old_rounds = self.n_rounds
            self.n_rounds = r
            
            datasets[r] = self.generate_balanced_dataset(n_samples_per_round)
            
            self.n_rounds = old_rounds
        
        return datasets


def generate_dataset(
    cipher: str,
    n_rounds: int,
    delta_p: int,
    n_train: int,
    n_val: int,
    n_test: int,
    output_dir: str,
    include_plaintext: bool = False,
    include_trace: bool = False,
    seed: int = 42
) -> Path:
    """
    Generate and save complete dataset to CSV files.
    
    Args:
        cipher: Cipher name
        n_rounds: Number of rounds
        delta_p: Input difference
        n_train: Training samples
        n_val: Validation samples
        n_test: Test samples
        output_dir: Output directory
        include_plaintext: Include plaintexts
        include_trace: Include intermediate states
        seed: Random seed
        
    Returns:
        Path to output directory containing CSV files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create generator
    generator = CipherDataGenerator(
        cipher=cipher,
        n_rounds=n_rounds,
        delta_p=delta_p,
        seed=seed
    )
    
    # Generate splits
    print(f"Generating {cipher} dataset with {n_rounds} rounds, delta_p=0x{delta_p:08x}")
    
    splits = {
        'train': generator.generate_balanced_dataset(n_train, include_plaintext, include_trace),
        'val': generator.generate_balanced_dataset(n_val, include_plaintext, include_trace),
        'test': generator.generate_balanced_dataset(n_test, include_plaintext, include_trace)
    }
    
    # Create subdirectory for this dataset
    dataset_name = f"{cipher}_r{n_rounds}_delta{delta_p:08x}"
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each split as CSV
    for split_name, data in splits.items():
        # Build dataframe from data dict
        df = pd.DataFrame({
            'C': data['C'],
            'C_prime': data['C_prime'],
            'label': data['labels']
        })
        
        if include_plaintext and 'P' in data:
            df['P'] = data['P']
            df['P_prime'] = data['P_prime']
        
        # Save to CSV
        csv_path = dataset_dir / f"{split_name}.csv"
        df.to_csv(csv_path, index=False)
    
    # Save metadata
    metadata = {
        'cipher': cipher,
        'n_rounds': n_rounds,
        'delta_p': delta_p,
        'seed': seed,
        'key': [int(k) for k in generator.key]
    }
    import json
    with open(dataset_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved dataset to {dataset_dir}")
    return dataset_dir


def load_dataset(filepath: Union[str, Path]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load dataset from CSV files.
    
    Args:
        filepath: Path to dataset directory containing train.csv, val.csv, test.csv
        
    Returns:
        Dictionary with 'train', 'val', 'test' splits
    """
    filepath = Path(filepath)
    result = {}
    
    for split_name in ['train', 'val', 'test']:
        csv_path = filepath / f"{split_name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            result[split_name] = {
                'C': df['C'].values,
                'C_prime': df['C_prime'].values,
                'labels': df['label'].values.astype(np.uint8)
            }
            # Optional fields
            if 'P' in df.columns:
                result[split_name]['P'] = df['P'].values
                result[split_name]['P_prime'] = df['P_prime'].values
    
    return result


def validate_dataset(data: Dict[str, np.ndarray]) -> Dict[str, any]:
    """
    Validate dataset integrity.
    
    Checks:
    - Balanced labels (within 1% tolerance)
    - Correct shapes (all arrays same length)
    - No NaN values
    - Non-trivial values (not all zeros)
    
    Args:
        data: Dataset dictionary with 'C', 'C_prime', 'labels' keys
        
    Returns:
        Dictionary with validation results and any issues found
    """
    issues = []
    n_samples = len(data['labels'])
    
    # Check all arrays have same length
    for key, arr in data.items():
        if hasattr(arr, '__len__') and len(arr) != n_samples:
            issues.append(f"Array '{key}' has length {len(arr)}, expected {n_samples}")
    
    # Check label balance
    n_positive = np.sum(data['labels'] == 1)
    n_negative = np.sum(data['labels'] == 0)
    balance_ratio = n_positive / n_samples if n_samples > 0 else 0
    if abs(balance_ratio - 0.5) > 0.01:
        issues.append(f"Imbalanced labels: {n_positive} positive, {n_negative} negative (ratio: {balance_ratio:.3f})")
    
    # Check for NaN values
    for key in ['C', 'C_prime']:
        if key in data and np.any(np.isnan(data[key].astype(np.float64))):
            issues.append(f"NaN values found in '{key}'")
    
    # Check for all-zero arrays (suspicious)
    for key in ['C', 'C_prime']:
        if key in data and np.all(data[key] == 0):
            issues.append(f"All-zero values in '{key}' (suspicious)")
    
    # Check plaintext fields if present
    if 'P' in data:
        if np.all(data['P'] == 0) and n_positive < n_samples:
            issues.append("All-zero plaintexts detected — R6 representation may be trivially distinguishable")
    
    return {
        'valid': len(issues) == 0,
        'n_samples': n_samples,
        'n_positive': int(n_positive),
        'n_negative': int(n_negative),
        'balance_ratio': float(balance_ratio),
        'issues': issues
    }

