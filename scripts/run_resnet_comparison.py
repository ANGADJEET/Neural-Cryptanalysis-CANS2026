#!/usr/bin/env python3
"""
ResNet vs MLP comparison for all 3 ciphers.
Trains Gohr's ResNet (depth=10, 32 filters) and compares with MLP.

Usage:
  python scripts/run_resnet_comparison.py --device cuda
"""

import argparse
import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from ciphers import get_cipher
from data.generator import CipherDataGenerator
from data.dataloader import CryptoDataset, get_input_dim
from models import get_model
from training.trainer import Trainer
from evaluation.metrics import evaluate_model
from torch.utils.data import DataLoader
from experiments.experiment_utils import set_seed


def train_and_eval(cipher_name, n_rounds, model_name, device, n_samples=500000,
                   batch_size=5000, n_epochs=30, seed=42):
    """Train a model and return validation accuracy."""
    set_seed(seed)
    cipher = get_cipher(cipher_name)
    gen = CipherDataGenerator(cipher_name, n_rounds=n_rounds,
                               delta_p=cipher.get_default_delta_p())
    train_data = gen.generate_balanced_dataset(n_samples, negative_type='gohr')
    val_data = gen.generate_balanced_dataset(n_samples // 10, negative_type='gohr')

    input_dim = get_input_dim('R2_xor_diff', cipher.block_size)
    model = get_model(model_name, input_dim=input_dim)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())

    train_ds = CryptoDataset(train_data, 'R2_xor_diff', cipher.block_size)
    val_ds = CryptoDataset(val_data, 'R2_xor_diff', cipher.block_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    trainer = Trainer(model=model, train_loader=train_loader,
                      val_loader=val_loader, device=device, use_wandb=False)
    trainer.train(n_epochs=n_epochs, early_stopping_patience=5, save_best=False)

    metrics = evaluate_model(model, val_loader, device)
    return float(metrics['accuracy']), n_params


def main():
    parser = argparse.ArgumentParser(description='ResNet vs MLP Comparison')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    output_dir = Path('./results/resnet_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ('speck32', [4, 5, 6, 7]),
        ('simon32', [5, 6, 7, 8]),
        ('present', [3, 4, 5]),
    ]

    results = {}
    t0 = time.time()

    for cipher_name, rounds_list in configs:
        print(f"\n{'═'*60}")
        print(f"  ResNet vs MLP — {cipher_name.upper()}")
        print(f"{'═'*60}")

        cipher_results = {}
        for n_rounds in rounds_list:
            print(f"\n  Round {n_rounds}:")

            # MLP (3 seeds)
            mlp_accs = []
            for seed in [42, 43, 44]:
                acc, n_params_mlp = train_and_eval(
                    cipher_name, n_rounds, 'gohr_mlp', args.device, seed=seed)
                mlp_accs.append(acc)
            mlp_mean = np.mean(mlp_accs)
            mlp_std = np.std(mlp_accs)

            # ResNet (3 seeds)
            resnet_accs = []
            for seed in [42, 43, 44]:
                acc, n_params_resnet = train_and_eval(
                    cipher_name, n_rounds, 'gohr_resnet', args.device, seed=seed)
                resnet_accs.append(acc)
            resnet_mean = np.mean(resnet_accs)
            resnet_std = np.std(resnet_accs)

            gap = resnet_mean - mlp_mean
            print(f"    MLP:    {mlp_mean:.4f} ± {mlp_std:.4f} ({n_params_mlp:,} params)")
            print(f"    ResNet: {resnet_mean:.4f} ± {resnet_std:.4f} ({n_params_resnet:,} params)")
            print(f"    Gap:    {gap:+.4f}")

            cipher_results[str(n_rounds)] = {
                'mlp_mean': float(mlp_mean),
                'mlp_std': float(mlp_std),
                'mlp_params': n_params_mlp,
                'resnet_mean': float(resnet_mean),
                'resnet_std': float(resnet_std),
                'resnet_params': n_params_resnet,
                'gap': float(gap),
            }

        results[cipher_name] = cipher_results

    with open(output_dir / 'resnet_vs_mlp.json', 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'═'*60}")
    print(f"  ✓ ResNet comparison complete ({elapsed:.0f}s)")
    print(f"  Results saved to {output_dir}")
    print(f"{'═'*60}")


if __name__ == '__main__':
    main()
