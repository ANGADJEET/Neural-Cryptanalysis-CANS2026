"""
PyTorch DataLoader utilities for neural cryptanalysis.

Provides Dataset classes and DataLoader factories for efficient
training and evaluation.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Tuple, List, Union
from pathlib import Path

from .representations import RepresentationFactory


class CryptoDataset(Dataset):
    """
    PyTorch Dataset for neural cryptanalysis.
    
    Supports loading from dict and on-the-fly representation conversion.
    """
    
    def __init__(
        self,
        data: Dict[str, np.ndarray],
        representation: str = 'R2_xor_diff',
        block_size: int = 32,
        transform: Optional[callable] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data: Dictionary with 'C', 'C_prime', 'labels' and optionally
                  'P', 'P_prime', 'intermediates', 'intermediates_prime'
            representation: Representation name to use
            block_size: Cipher block size
            transform: Optional transform to apply
        """
        self.C = data['C']
        self.C_prime = data['C_prime']
        self.labels = data['labels']
        
        # Optional fields
        self.P = data.get('P', None)
        self.P_prime = data.get('P_prime', None)
        self.intermediates = data.get('intermediates', None)
        self.intermediates_prime = data.get('intermediates_prime', None)
        
        self.representation = representation
        self.block_size = block_size
        self.transform = transform
        
        # Pre-compute representations if dataset is small enough
        self.factory = RepresentationFactory(block_size=block_size)
        self._precomputed = None
        
        if len(self.C) <= 10_000_000:  # Pre-compute for datasets < 10M
            self._precompute_representations()
    
    def _precompute_representations(self):
        """Pre-compute all representations for efficiency."""
        self._precomputed = self.factory.get_representation(
            self.representation,
            self.C,
            self.C_prime,
            P=self.P,
            P_prime=self.P_prime,
            intermediates=self.intermediates,
            intermediates_prime=self.intermediates_prime
        )
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            (representation, label) as torch tensors
        """
        if self._precomputed is not None:
            X = self._precomputed[idx]
        else:
            # Compute on-the-fly
            X = self.factory.get_representation(
                self.representation,
                self.C[idx:idx+1],
                self.C_prime[idx:idx+1],
                P=self.P[idx:idx+1] if self.P is not None else None,
                P_prime=self.P_prime[idx:idx+1] if self.P_prime is not None else None,
                intermediates=self.intermediates[idx:idx+1] if self.intermediates is not None else None,
                intermediates_prime=self.intermediates_prime[idx:idx+1] if self.intermediates_prime is not None else None
            )[0]
        
        X = torch.from_numpy(X).float()
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        if self.transform is not None:
            X = self.transform(X)
        
        return X, y


class CSVCryptoDataset(Dataset):
    """
    Dataset that loads from CSV file.
    
    Loads data into memory for efficient access.
    """
    
    def __init__(
        self,
        csv_path: Union[str, Path],
        representation: str = 'R2_xor_diff',
        block_size: int = 32
    ):
        """
        Initialize dataset from CSV file.
        
        Args:
            csv_path: Path to CSV file
            representation: Representation name
            block_size: Cipher block size
        """
        self.csv_path = Path(csv_path)
        self.representation = representation
        self.block_size = block_size
        self.factory = RepresentationFactory(block_size=block_size)
        
        # Load data into memory
        df = pd.read_csv(self.csv_path)
        self.C = df['C'].values
        self.C_prime = df['C_prime'].values
        self.labels = df['label'].values.astype(np.uint8)
        
        # Optional fields
        self.P = df['P'].values if 'P' in df.columns else None
        self.P_prime = df['P_prime'].values if 'P_prime' in df.columns else None
        
        self.length = len(self.labels)
        
        # Pre-compute representations
        self._precomputed = self.factory.get_representation(
            self.representation,
            self.C,
            self.C_prime,
            P=self.P,
            P_prime=self.P_prime
        )
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        X = self._precomputed[idx]
        X = torch.from_numpy(X).float()
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return X, y


def get_dataloaders(
    data: Dict[str, Dict[str, np.ndarray]],
    representation: str = 'R2_xor_diff',
    block_size: int = 32,
    batch_size: int = 5000,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for all splits.
    
    Args:
        data: Dictionary with 'train', 'val', 'test' splits
        representation: Representation name
        block_size: Cipher block size
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for GPU transfer
        
    Returns:
        Dictionary with DataLoaders for each split
    """
    loaders = {}
    
    for split_name, split_data in data.items():
        dataset = CryptoDataset(
            data=split_data,
            representation=representation,
            block_size=block_size
        )
        
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == 'train'),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split_name == 'train')
        )
    
    return loaders


def get_input_dim(representation: str, block_size: int = 32, n_rounds: int = 1) -> int:
    """
    Get the input dimension for a representation.
    
    Args:
        representation: Representation name
        block_size: Cipher block size
        n_rounds: Number of rounds (for sequential representations)
        
    Returns:
        Flattened input dimension
    """
    factory = RepresentationFactory(block_size=block_size)
    shape = factory.get_output_shape(representation, n_rounds=n_rounds)
    
    if shape is None:
        raise ValueError(f"Unknown representation: {representation}")
    
    return int(np.prod(shape))
