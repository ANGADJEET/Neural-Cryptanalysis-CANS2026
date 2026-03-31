
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
    
    def __init__(
        self,
        cipher: Union[str, BaseCipher],
        n_rounds: int,
        delta_p: int,
        key: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ):
        if isinstance(cipher, str):
            self.cipher = get_cipher(cipher)
        else:
            self.cipher = cipher
        
        self.n_rounds = n_rounds
        self.delta_p = delta_p
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        if key is None:
            self.key = self.cipher.random_key()
        else:
            self.key = key
        
        self.random_perm = RandomPermutation(block_size=self.cipher.block_size)
    
    def generate_cipher_samples(
        self,
        n_samples: int,
        include_plaintext: bool = False,
        include_trace: bool = False
    ) -> Dict[str, np.ndarray]:
        P = self.cipher.random_plaintexts(n_samples)
        P_prime = P ^ self.delta_p
        
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
            result['intermediates'] = np.stack(states, axis=1)
            result['intermediates_prime'] = np.stack(states_prime, axis=1)
        
        return result
    
    def generate_random_samples(
        self,
        n_samples: int,
        include_plaintext: bool = False
    ) -> Dict[str, np.ndarray]:
        C, C_prime = self.random_perm.generate_random_pairs(n_samples)
        
        result = {
            'C': C,
            'C_prime': C_prime,
            'labels': np.zeros(n_samples, dtype=np.uint8)
        }
        
        if include_plaintext:
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
        n_each = n_samples // 2
        
        cipher_data = self.generate_cipher_samples(
            n_each, include_plaintext, include_trace
        )
        random_data = self.generate_random_samples(n_each, include_plaintext)
        
        result = {
            'C': np.concatenate([cipher_data['C'], random_data['C']]),
            'C_prime': np.concatenate([cipher_data['C_prime'], random_data['C_prime']]),
            'labels': np.concatenate([cipher_data['labels'], random_data['labels']])
        }
        
        if include_plaintext:
            result['P'] = np.concatenate([cipher_data['P'], random_data['P']])
            result['P_prime'] = np.concatenate([cipher_data['P_prime'], random_data['P_prime']])
        
        if include_trace:
            dummy_trace = np.zeros_like(cipher_data['intermediates'])
            result['intermediates'] = np.concatenate([cipher_data['intermediates'], dummy_trace])
            result['intermediates_prime'] = np.concatenate([cipher_data['intermediates_prime'], dummy_trace])
        
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
        if max_rounds is None:
            max_rounds = self.cipher.max_rounds
        
        datasets = {}
        for r in tqdm(range(min_rounds, max_rounds + 1), desc="Generating round sweep"):
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = CipherDataGenerator(
        cipher=cipher,
        n_rounds=n_rounds,
        delta_p=delta_p,
        seed=seed
    )
    
    print(f"Generating {cipher} dataset with {n_rounds} rounds, delta_p=0x{delta_p:08x}")
    
    splits = {
        'train': generator.generate_balanced_dataset(n_train, include_plaintext, include_trace),
        'val': generator.generate_balanced_dataset(n_val, include_plaintext, include_trace),
        'test': generator.generate_balanced_dataset(n_test, include_plaintext, include_trace)
    }
    
    dataset_name = f"{cipher}_r{n_rounds}_delta{delta_p:08x}"
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, data in splits.items():
        df = pd.DataFrame({
            'C': data['C'],
            'C_prime': data['C_prime'],
            'label': data['labels']
        })
        
        if include_plaintext and 'P' in data:
            df['P'] = data['P']
            df['P_prime'] = data['P_prime']
        
        csv_path = dataset_dir / f"{split_name}.csv"
        df.to_csv(csv_path, index=False)
    
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
            if 'P' in df.columns:
                result[split_name]['P'] = df['P'].values
                result[split_name]['P_prime'] = df['P_prime'].values
    
    return result


def validate_dataset(data: Dict[str, np.ndarray]) -> Dict[str, any]:
    issues = []
    n_samples = len(data['labels'])
    
    for key, arr in data.items():
        if hasattr(arr, '__len__') and len(arr) != n_samples:
            issues.append(f"Array '{key}' has length {len(arr)}, expected {n_samples}")
    
    n_positive = np.sum(data['labels'] == 1)
    n_negative = np.sum(data['labels'] == 0)
    balance_ratio = n_positive / n_samples if n_samples > 0 else 0
    if abs(balance_ratio - 0.5) > 0.01:
        issues.append(f"Imbalanced labels: {n_positive} positive, {n_negative} negative (ratio: {balance_ratio:.3f})")
    
    for key in ['C', 'C_prime']:
        if key in data and np.any(np.isnan(data[key].astype(np.float64))):
            issues.append(f"NaN values found in '{key}'")
    
    for key in ['C', 'C_prime']:
        if key in data and np.all(data[key] == 0):
            issues.append(f"All-zero values in '{key}' (suspicious)")
    
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
