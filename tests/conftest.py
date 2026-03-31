"""
Shared fixtures for neural cryptanalysis tests.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def speck_cipher():
    """SPECK32 cipher instance."""
    from ciphers.speck32 import Speck32
    return Speck32()


@pytest.fixture
def simon_cipher():
    """SIMON32 cipher instance."""
    from ciphers.simon32 import Simon32
    return Simon32()


@pytest.fixture
def present_cipher():
    """PRESENT cipher instance."""
    from ciphers.present import Present
    return Present()


@pytest.fixture
def sample_data_32bit():
    """Small sample dataset for 32-bit ciphers."""
    np.random.seed(42)
    n = 1000
    return {
        'C': np.random.randint(0, 2**32, size=n, dtype=np.uint32),
        'C_prime': np.random.randint(0, 2**32, size=n, dtype=np.uint32),
        'labels': np.concatenate([np.ones(n // 2), np.zeros(n // 2)]).astype(np.uint8),
    }


@pytest.fixture
def sample_data_64bit():
    """Small sample dataset for 64-bit ciphers."""
    np.random.seed(42)
    n = 1000
    return {
        'C': np.random.randint(0, 2**63, size=n, dtype=np.uint64),
        'C_prime': np.random.randint(0, 2**63, size=n, dtype=np.uint64),
        'labels': np.concatenate([np.ones(n // 2), np.zeros(n // 2)]).astype(np.uint8),
    }


@pytest.fixture
def small_generator():
    """Small data generator for SPECK32."""
    from data.generator import CipherDataGenerator
    return CipherDataGenerator(
        cipher='speck32',
        n_rounds=5,
        delta_p=0x00400000,
        seed=42
    )
