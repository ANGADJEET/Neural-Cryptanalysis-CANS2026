
import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMiniPipeline:
    
    def test_full_pipeline_mlp(self):
        from data.generator import CipherDataGenerator
        from data.dataloader import CryptoDataset, get_input_dim
        from models import get_model
        from training.trainer import Trainer
        from evaluation.metrics import compute_accuracy
        from torch.utils.data import DataLoader
        
        gen = CipherDataGenerator('speck32', n_rounds=5, delta_p=0x00400000, seed=42)
        train_data = gen.generate_balanced_dataset(500)
        val_data = gen.generate_balanced_dataset(200)
        
        representation = 'R2_xor_diff'
        train_ds = CryptoDataset(train_data, representation=representation, block_size=32)
        val_ds = CryptoDataset(val_data, representation=representation, block_size=32)
        
        train_loader = DataLoader(train_ds, batch_size=100, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=100)
        
        input_dim = get_input_dim(representation, block_size=32)
        model = get_model('mlp', input_dim=input_dim, hidden_layers=[64, 32])
        
        device = 'cpu'
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            use_wandb=False,
            save_dir='./test_checkpoints'
        )
        history = trainer.train(n_epochs=2, early_stopping_patience=5, save_best=False)
        
        acc = compute_accuracy(model, val_loader, device)
        
        assert acc > 0, "Accuracy should be > 0"
        assert acc <= 1.0, "Accuracy should be <= 1.0"
        assert 'train_loss' in history
        
        import shutil
        if Path('./test_checkpoints').exists():
            shutil.rmtree('./test_checkpoints')
    
    def test_cnn_pipeline(self):
        from data.generator import CipherDataGenerator
        from data.dataloader import CryptoDataset, get_input_dim
        from models import get_model
        from evaluation.metrics import compute_accuracy
        from torch.utils.data import DataLoader
        
        gen = CipherDataGenerator('speck32', n_rounds=5, delta_p=0x00400000, seed=42)
        data = gen.generate_balanced_dataset(200)
        
        representation = 'R2_xor_diff'
        ds = CryptoDataset(data, representation=representation, block_size=32)
        loader = DataLoader(ds, batch_size=100)
        
        input_dim = get_input_dim(representation, block_size=32)
        model = get_model('cnn', input_dim=input_dim)
        
        for X, y in loader:
            out = model(X)
            assert out.shape[0] == X.shape[0]
            assert out.shape[1] == 1
            break
    
    def test_multiple_representations(self):
        from data.generator import CipherDataGenerator
        from data.dataloader import CryptoDataset
        from torch.utils.data import DataLoader
        
        gen = CipherDataGenerator('speck32', n_rounds=5, delta_p=0x00400000, seed=42)
        data = gen.generate_balanced_dataset(100, include_plaintext=True)
        
        for repr_name in ['R1_raw_pair', 'R2_xor_diff', 'R3_concat', 
                          'R5_word_level', 'R6_joint_pc', 'R8_statistical']:
            ds = CryptoDataset(data, representation=repr_name, block_size=32)
            loader = DataLoader(ds, batch_size=50)
            
            for X, y in loader:
                assert X.shape[0] == 50, f"Bad batch size for {repr_name}"
                assert y.shape[0] == 50
                assert not torch.isnan(X).any(), f"NaN in {repr_name}"
                break
    
    def test_validate_generated_data(self):
        from data.generator import CipherDataGenerator, validate_dataset
        
        gen = CipherDataGenerator('speck32', n_rounds=5, delta_p=0x00400000, seed=42)
        data = gen.generate_balanced_dataset(1000)
        
        result = validate_dataset(data)
        assert result['valid'] is True, f"Validation failed: {result['issues']}"
        assert result['n_positive'] == 500
        assert result['n_negative'] == 500
