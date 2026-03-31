
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def speck_cipher():
    from ciphers.speck32 import Speck32
    return Speck32()


@pytest.fixture
def simon_cipher():
    from ciphers.simon32 import Simon32
    return Simon32()


@pytest.fixture
def present_cipher():
    from ciphers.present import Present
    return Present()


@pytest.fixture
def sample_data_32bit():
    np.random.seed(42)
    n = 1000
    return {
        'C': np.random.randint(0, 2**32, size=n, dtype=np.uint32),
        'C_prime': np.random.randint(0, 2**32, size=n, dtype=np.uint32),
        'labels': np.concatenate([np.ones(n // 2), np.zeros(n // 2)]).astype(np.uint8),
    }


@pytest.fixture
def sample_data_64bit():
    np.random.seed(42)
    n = 1000
    return {
        'C': np.random.randint(0, 2**63, size=n, dtype=np.uint64),
        'C_prime': np.random.randint(0, 2**63, size=n, dtype=np.uint64),
        'labels': np.concatenate([np.ones(n // 2), np.zeros(n // 2)]).astype(np.uint8),
    }


@pytest.fixture
def small_generator():
    from data.generator import CipherDataGenerator
    return CipherDataGenerator(
        cipher='speck32',
        n_rounds=5,
        delta_p=0x00400000,
        seed=42
    )
