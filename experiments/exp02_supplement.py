"""
E02 Supplement: Detailed evaluation of R6 (Joint P-C), R7 (Sequential), and R9 (Masked) representations.
Tests R9 under three noise settings, R6 with plaintext access, and R7 with intermediate round traces.
"""
import sys, json, os
sys.path.insert(0, '.')
import numpy as np
import torch
from experiments.experiment_utils import set_seed, get_device
from ciphers import get_cipher
from data.generator import CipherDataGenerator
from data.representations import RepresentationFactory
from models import get_model
from training.trainer import Trainer
from evaluation.metrics import evaluate_model
from torch.utils.data import DataLoader

class FakeArgs:
    device = 'cuda'

def main():
    args = FakeArgs()
    device = get_device(args)
    cipher = get_cipher('speck32')
    results = {}

    # Test R9 (Masked) at round 5
    print('Testing R9_masked...')
    for noise_setting in ['mask_only', 'noise_only', 'both']:
        seed_accs = []
        for seed in [42, 43, 44, 45, 46]:
            set_seed(seed)
            gen = CipherDataGenerator(cipher='speck32', n_rounds=5, delta_p=cipher.get_default_delta_p())
            train_data = gen.generate_balanced_dataset(500000)
            val_data = gen.generate_balanced_dataset(50000)
            test_data = gen.generate_balanced_dataset(50000)

            factory = RepresentationFactory(block_size=32)
            kwargs = {}
            if noise_setting == 'mask_only':
                kwargs = {'mask_prob': 0.1, 'noise_std': 0.0}
            elif noise_setting == 'noise_only':
                kwargs = {'mask_prob': 0.0, 'noise_std': 0.05}
            else:
                kwargs = {'mask_prob': 0.05, 'noise_std': 0.02}

            X_train = factory.get_representation('R9_masked', train_data['C'], train_data['C_prime'], **kwargs)
            X_val   = factory.get_representation('R9_masked', val_data['C'], val_data['C_prime'], **kwargs)
            X_test  = factory.get_representation('R9_masked', test_data['C'], test_data['C_prime'], **kwargs)

            input_dim = X_train.shape[-1]
            model = get_model('gohr_mlp', input_dim=input_dim)

            train_ds = torch.utils.data.TensorDataset(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(train_data['labels']).float()
            )
            val_ds = torch.utils.data.TensorDataset(
                torch.from_numpy(X_val).float(),
                torch.from_numpy(val_data['labels']).float()
            )
            test_ds = torch.utils.data.TensorDataset(
                torch.from_numpy(X_test).float(),
                torch.from_numpy(test_data['labels']).float()
            )

            train_loader = DataLoader(train_ds, batch_size=5000, shuffle=True)
            val_loader   = DataLoader(val_ds, batch_size=5000)
            test_loader  = DataLoader(test_ds, batch_size=5000)

            trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, device=device, use_wandb=False)
            trainer.train(n_epochs=30, early_stopping_patience=5, save_best=False)
            metrics = evaluate_model(model, test_loader, device)
            seed_accs.append(float(metrics['accuracy']))
            print(f'  R9 {noise_setting} seed {seed}: {metrics["accuracy"]:.4f}')

        results[f'R9_{noise_setting}'] = {
            'mean': float(np.mean(seed_accs)),
            'std': float(np.std(seed_accs)),
            'values': seed_accs,
        }

    # Test R6 (Joint P-C) at round 5 — needs plaintext in data
    print('Testing R6_joint_pc...')
    seed_accs_r6 = []
    for seed in [42, 43, 44, 45, 46]:
        set_seed(seed)
        gen = CipherDataGenerator(cipher='speck32', n_rounds=5, delta_p=cipher.get_default_delta_p())
        train_data = gen.generate_balanced_dataset(500000, include_plaintext=True)
        val_data = gen.generate_balanced_dataset(50000, include_plaintext=True)
        test_data = gen.generate_balanced_dataset(50000, include_plaintext=True)

        factory = RepresentationFactory(block_size=32)
        X_train = factory.get_representation('R6_joint_pc', train_data['C'], train_data['C_prime'],
                                              P=train_data.get('P'), P_prime=train_data.get('P_prime'))
        X_val   = factory.get_representation('R6_joint_pc', val_data['C'], val_data['C_prime'],
                                              P=val_data.get('P'), P_prime=val_data.get('P_prime'))
        X_test  = factory.get_representation('R6_joint_pc', test_data['C'], test_data['C_prime'],
                                              P=test_data.get('P'), P_prime=test_data.get('P_prime'))

        input_dim = X_train.shape[-1]
        model = get_model('gohr_mlp', input_dim=input_dim)

        train_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(train_data['labels']).float()
        )
        val_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(val_data['labels']).float()
        )
        test_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_test).float(),
            torch.from_numpy(test_data['labels']).float()
        )

        train_loader = DataLoader(train_ds, batch_size=5000, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=5000)
        test_loader  = DataLoader(test_ds, batch_size=5000)

        trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, device=device, use_wandb=False)
        trainer.train(n_epochs=30, early_stopping_patience=5, save_best=False)
        metrics = evaluate_model(model, test_loader, device)
        seed_accs_r6.append(float(metrics['accuracy']))
        print(f'  R6 seed {seed}: {metrics["accuracy"]:.4f}')

    results['R6_joint_pc'] = {
        'mean': float(np.mean(seed_accs_r6)),
        'std': float(np.std(seed_accs_r6)),
        'values': seed_accs_r6,
    }

    # Test R7 (Sequential / round-wise differences) at round 5
    # R7 requires white-box intermediate states via include_trace=True
    # Output shape: (N, n_rounds, block_size) — use LSTM for sequential data
    print('Testing R7_sequential...')
    seed_accs_r7 = []
    for seed in [42, 43, 44, 45, 46]:
        set_seed(seed)
        gen = CipherDataGenerator(cipher='speck32', n_rounds=5, delta_p=cipher.get_default_delta_p())
        train_data = gen.generate_balanced_dataset(500000, include_trace=True)
        val_data = gen.generate_balanced_dataset(50000, include_trace=True)
        test_data = gen.generate_balanced_dataset(50000, include_trace=True)

        # CRITICAL FIX for R7 Target Leakage:
        # The generator fills the negative class (label=0) intermediate traces with all zeros natively.
        # This allows the neural network to trivially achieve 100% accuracy by just looking for zeros.
        # We must replace those zeros with completely random numbers to force it to learn true crypto sequences.
        for dataset in [train_data, val_data, test_data]:
            mask_0 = dataset['labels'] == 0
            shape = dataset['intermediates'][mask_0].shape
            dataset['intermediates'][mask_0] = np.random.randint(0, 4294967295, size=shape, dtype=np.uint32)
            dataset['intermediates_prime'][mask_0] = np.random.randint(0, 4294967295, size=shape, dtype=np.uint32)

        factory = RepresentationFactory(block_size=32)
        X_train = factory.get_representation('R7_sequential', train_data['C'], train_data['C_prime'],
                                              intermediates=train_data['intermediates'],
                                              intermediates_prime=train_data['intermediates_prime'])
        X_val   = factory.get_representation('R7_sequential', val_data['C'], val_data['C_prime'],
                                              intermediates=val_data['intermediates'],
                                              intermediates_prime=val_data['intermediates_prime'])
        X_test  = factory.get_representation('R7_sequential', test_data['C'], test_data['C_prime'],
                                              intermediates=test_data['intermediates'],
                                              intermediates_prime=test_data['intermediates_prime'])

        print(f'  R7 shape: {X_train.shape}')
        seq_len = X_train.shape[1]
        feat_dim = X_train.shape[2]
        input_dim = feat_dim

        model = get_model('lstm', input_dim=input_dim)

        train_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(train_data['labels']).float()
        )
        val_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(val_data['labels']).float()
        )
        test_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_test).float(),
            torch.from_numpy(test_data['labels']).float()
        )

        train_loader = DataLoader(train_ds, batch_size=5000, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=5000)
        test_loader  = DataLoader(test_ds, batch_size=5000)

        trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, device=device, use_wandb=False)
        trainer.train(n_epochs=30, early_stopping_patience=5, save_best=False)
        metrics = evaluate_model(model, test_loader, device)
        seed_accs_r7.append(float(metrics['accuracy']))
        print(f'  R7 seed {seed}: {metrics["accuracy"]:.4f}')

    results['R7_sequential'] = {
        'mean': float(np.mean(seed_accs_r7)),
        'std': float(np.std(seed_accs_r7)),
        'values': seed_accs_r7,
    }

    # Save supplemental representation results
    os.makedirs('results/e02_representation', exist_ok=True)
    with open('results/e02_representation/e02_supplement_r5_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('✓ Supplemental representation results saved (R6, R7, R9)')

if __name__ == '__main__':
    main()
