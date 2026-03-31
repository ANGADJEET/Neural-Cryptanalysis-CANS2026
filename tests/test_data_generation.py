
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.representations import RepresentationFactory
from data.generator import CipherDataGenerator, validate_dataset


class TestRepresentationShapes:
    
    def setup_method(self):
        np.random.seed(42)
        self.n = 100
        self.C_32 = np.random.randint(0, 2**32, size=self.n, dtype=np.uint32)
        self.C_prime_32 = np.random.randint(0, 2**32, size=self.n, dtype=np.uint32)
        self.P_32 = np.random.randint(0, 2**32, size=self.n, dtype=np.uint32)
        self.P_prime_32 = np.random.randint(0, 2**32, size=self.n, dtype=np.uint32)
        self.factory_32 = RepresentationFactory(block_size=32)
        
        self.C_64 = np.random.randint(0, 2**63, size=self.n, dtype=np.uint64)
        self.C_prime_64 = np.random.randint(0, 2**63, size=self.n, dtype=np.uint64)
        self.factory_64 = RepresentationFactory(block_size=64)
    
    def test_r1_raw_pair_shape(self):
        X = self.factory_32.get_representation('R1_raw_pair', self.C_32, self.C_prime_32)
        assert X.shape == (self.n, 2, 32)
        assert X.dtype == np.float32
    
    def test_r2_xor_diff_shape(self):
        X = self.factory_32.get_representation('R2_xor_diff', self.C_32, self.C_prime_32)
        assert X.shape == (self.n, 32)
    
    def test_r3_concat_shape(self):
        X = self.factory_32.get_representation('R3_concat', self.C_32, self.C_prime_32)
        assert X.shape == (self.n, 64)
    
    def test_r4_bit_sliced_shape(self):
        X = self.factory_32.get_representation('R4_bit_sliced', self.C_32, self.C_prime_32)
        assert X.shape == (self.n, 2, 2, 16)
    
    def test_r5_word_level_shape(self):
        X = self.factory_32.get_representation('R5_word_level', self.C_32, self.C_prime_32)
        assert X.shape == (self.n, 4)
    
    def test_r6_joint_pc_shape(self):
        X = self.factory_32.get_representation(
            'R6_joint_pc', self.C_32, self.C_prime_32,
            P=self.P_32, P_prime=self.P_prime_32
        )
        assert X.shape == (self.n, 128)
    
    def test_r6_requires_plaintext(self):
        with pytest.raises(ValueError, match="requires plaintext"):
            self.factory_32.get_representation('R6_joint_pc', self.C_32, self.C_prime_32)
    
    def test_r8_statistical_shape(self):
        X = self.factory_32.get_representation('R8_statistical', self.C_32, self.C_prime_32)
        assert X.shape == (self.n, 1 + 32 + 4)
    
    def test_r9_masked_shape(self):
        X = self.factory_32.get_representation('R9_masked', self.C_32, self.C_prime_32)
        assert X.shape == (self.n, 32)
    
    def test_r2_64bit(self):
        X = self.factory_64.get_representation('R2_xor_diff', self.C_64, self.C_prime_64)
        assert X.shape == (self.n, 64)
    
    def test_r2_values_are_binary(self):
        X = self.factory_32.get_representation('R2_xor_diff', self.C_32, self.C_prime_32)
        assert np.all((X == 0) | (X == 1))
    
    def test_r5_values_normalized(self):
        X = self.factory_32.get_representation('R5_word_level', self.C_32, self.C_prime_32)
        assert np.all(X >= 0) and np.all(X <= 1)
    
    def test_unknown_representation_raises(self):
        with pytest.raises(ValueError, match="Unknown representation"):
            self.factory_32.get_representation('R99_nonexistent', self.C_32, self.C_prime_32)


class TestDataGenerator:
    
    def test_balanced_dataset(self, small_generator):
        data = small_generator.generate_balanced_dataset(1000)
        assert len(data['labels']) == 1000
        assert np.sum(data['labels'] == 1) == 500
        assert np.sum(data['labels'] == 0) == 500
    
    def test_cipher_samples_shape(self, small_generator):
        data = small_generator.generate_cipher_samples(500)
        assert len(data['C']) == 500
        assert len(data['C_prime']) == 500
        assert np.all(data['labels'] == 1)
    
    def test_random_samples_shape(self, small_generator):
        data = small_generator.generate_random_samples(500)
        assert len(data['C']) == 500
        assert np.all(data['labels'] == 0)
    
    def test_include_plaintext(self, small_generator):
        data = small_generator.generate_balanced_dataset(100, include_plaintext=True)
        assert 'P' in data
        assert 'P_prime' in data
        assert len(data['P']) == 100
    
    def test_r6_no_zeros_for_random(self, small_generator):
        data = small_generator.generate_balanced_dataset(1000, include_plaintext=True)
        
        random_mask = data['labels'] == 0
        random_P = data['P'][random_mask]
        random_P_prime = data['P_prime'][random_mask]
        
        assert not np.all(random_P == 0), "Random sample plaintexts are all zeros — R6 bug!"
        assert not np.all(random_P_prime == 0), "Random sample P_prime are all zeros — R6 bug!"
    
    def test_differential_correctness(self, small_generator):
        data = small_generator.generate_cipher_samples(100, include_plaintext=True)
        delta = data['P'] ^ data['P_prime']
        assert np.all(delta == 0x00400000)
    
    def test_round_sweep(self, small_generator):
        datasets = small_generator.generate_round_sweep(100, min_rounds=1, max_rounds=3)
        assert set(datasets.keys()) == {1, 2, 3}
        for r, data in datasets.items():
            assert len(data['labels']) == 100


class TestValidateDataset:
    
    def test_valid_dataset(self):
        data = {
            'C': np.random.randint(0, 2**32, 100, dtype=np.uint32),
            'C_prime': np.random.randint(0, 2**32, 100, dtype=np.uint32),
            'labels': np.array([1]*50 + [0]*50, dtype=np.uint8)
        }
        result = validate_dataset(data)
        assert result['valid'] is True
        assert len(result['issues']) == 0
    
    def test_imbalanced_labels(self):
        data = {
            'C': np.random.randint(0, 2**32, 100, dtype=np.uint32),
            'C_prime': np.random.randint(0, 2**32, 100, dtype=np.uint32),
            'labels': np.array([1]*90 + [0]*10, dtype=np.uint8)
        }
        result = validate_dataset(data)
        assert result['valid'] is False
        assert any('Imbalanced' in issue for issue in result['issues'])
    
    def test_all_zero_ciphertexts(self):
        data = {
            'C': np.zeros(100, dtype=np.uint32),
            'C_prime': np.random.randint(0, 2**32, 100, dtype=np.uint32),
            'labels': np.array([1]*50 + [0]*50, dtype=np.uint8)
        }
        result = validate_dataset(data)
        assert result['valid'] is False
        assert any('All-zero' in issue for issue in result['issues'])
