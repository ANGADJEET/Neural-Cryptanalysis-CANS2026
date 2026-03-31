"""
Tests for model architectures.

Verifies forward pass shapes, output ranges, and model factory for all architectures.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_model


class TestModelFactory:
    """Test the get_model factory function."""
    
    VALID_MODELS = ['mlp', 'gohr_mlp', 'cnn', 'residual_cnn', 'siamese', 'lstm', 'gru']
    
    def test_all_valid_models_instantiate(self):
        for name in self.VALID_MODELS:
            model = get_model(name, input_dim=32)
            assert model is not None, f"Failed to instantiate {name}"
    
    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model('nonexistent_model', input_dim=32)
    
    def test_mine_instantiates(self):
        model = get_model('mine', input_dim=32)
        assert model is not None


class TestMLPModels:
    """Test MLP-based models."""
    
    def test_mlp_forward_shape(self):
        model = get_model('mlp', input_dim=32)
        x = torch.randn(16, 32)
        out = model(x)
        assert out.shape == (16, 1)
    
    def test_mlp_output_range(self):
        model = get_model('mlp', input_dim=32)
        x = torch.randn(16, 32)
        out = model(x)
        assert torch.all(out >= 0) and torch.all(out <= 1), "MLP output not in [0, 1]"
    
    def test_gohr_mlp_forward(self):
        model = get_model('gohr_mlp', input_dim=64)
        x = torch.randn(8, 64)
        out = model(x)
        assert out.shape == (8, 1)
        assert torch.all(out >= 0) and torch.all(out <= 1)
    
    def test_mlp_flattens_multidim_input(self):
        model = get_model('mlp', input_dim=64)
        x = torch.randn(8, 2, 32)  # Multi-dim input
        out = model(x)
        assert out.shape == (8, 1)


class TestCNNModels:
    """Test CNN-based models."""
    
    def test_cnn_forward_shape(self):
        model = get_model('cnn', input_dim=32)
        x = torch.randn(16, 32)
        out = model(x)
        assert out.shape == (16, 1)
    
    def test_cnn_output_range(self):
        model = get_model('cnn', input_dim=32)
        x = torch.randn(16, 32)
        out = model(x)
        assert torch.all(out >= 0) and torch.all(out <= 1)
    
    def test_residual_cnn_forward(self):
        model = get_model('residual_cnn', input_dim=32)
        x = torch.randn(8, 32)
        out = model(x)
        assert out.shape == (8, 1)
        assert torch.all(out >= 0) and torch.all(out <= 1)


class TestRNNModels:
    """Test RNN-based models."""
    
    def test_lstm_forward_3d(self):
        model = get_model('lstm', input_dim=32)
        # 3D input: (batch, seq_len, features) — required by LSTM
        x = torch.randn(8, 5, 32)
        out = model(x)
        assert out.shape == (8, 1)
    
    def test_lstm_output_range(self):
        model = get_model('lstm', input_dim=32)
        x = torch.randn(8, 5, 32)
        out = model(x)
        assert torch.all(out >= 0) and torch.all(out <= 1)
    
    def test_lstm_single_step(self):
        model = get_model('lstm', input_dim=32)
        # Single timestep: (batch, 1, features)
        x = torch.randn(8, 1, 32)
        out = model(x)
        assert out.shape == (8, 1)
    
    def test_gru_forward(self):
        model = get_model('gru', input_dim=32)
        x = torch.randn(8, 5, 32)
        out = model(x)
        assert out.shape == (8, 1)
        assert torch.all(out >= 0) and torch.all(out <= 1)


class TestSiameseModel:
    """Test Siamese network."""
    
    def test_siamese_forward_2d(self):
        model = get_model('siamese', input_dim=32)
        # Siamese expects (batch, 2*dim) which it splits
        x = torch.randn(8, 64)  # 2 * 32
        out = model(x)
        assert out.shape == (8, 1)
    
    def test_siamese_forward_3d(self):
        model = get_model('siamese', input_dim=32)
        x = torch.randn(8, 2, 32)
        out = model(x)
        assert out.shape == (8, 1)
    
    def test_siamese_output_range(self):
        model = get_model('siamese', input_dim=32)
        x = torch.randn(8, 64)
        out = model(x)
        assert torch.all(out >= 0) and torch.all(out <= 1)


class TestModelParameterCounts:
    """Sanity checks on model sizes."""
    
    def test_models_have_parameters(self):
        for name in ['mlp', 'gohr_mlp', 'cnn', 'residual_cnn', 'lstm', 'gru', 'siamese']:
            model = get_model(name, input_dim=32)
            n_params = sum(p.numel() for p in model.parameters())
            assert n_params > 0, f"{name} has no parameters"
    
    def test_models_are_trainable(self):
        for name in ['mlp', 'cnn']:
            model = get_model(name, input_dim=32)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert trainable > 0, f"{name} has no trainable parameters"
