#!/usr/bin/env python
"""
E13: Model Architecture Comparison

Train all model architectures on the same task and compare:
accuracy ± std, parameter count, training time, inference throughput.

Usage:
    python experiments/exp13_model_comparison.py --cipher speck32 --rounds 5
    python experiments/exp13_model_comparison.py --cipher speck32 --rounds 5 --n-seeds 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
import matplotlib.pyplot as plt

from ciphers import get_cipher
from data.generator import CipherDataGenerator
from data.dataloader import CryptoDataset, get_input_dim
from experiments.experiment_utils import (
    set_seed, add_common_args, save_results, get_device, quick_train_eval
)

# All model architectures to test (feedforward / 2D input)
MODELS = ['gohr_mlp', 'mlp', 'cnn', 'residual_cnn', 'siamese']
# LSTM/GRU need special handling (3D input), test separately
RNN_MODELS = ['lstm', 'gru']


def main():
    parser = argparse.ArgumentParser(description='E13: Model Comparison')
    add_common_args(parser)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--representation', default='R2_xor_diff')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = './results/e13_model_comparison'

    print("=" * 60)
    print("  E13: Model Architecture Comparison")
    print("=" * 60)

    cipher = get_cipher(args.cipher)
    device = get_device(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed + i for i in range(args.n_seeds)]
    all_results = {}

    for model_name in MODELS:
        print(f"\n{'━' * 50}")
        print(f"  Model: {model_name}")
        print(f"{'━' * 50}")

        seed_results = []
        for seed in seeds:
            set_seed(seed)
            print(f"  Seed {seed}...", end=' ')

            gen = CipherDataGenerator(
                cipher=args.cipher, n_rounds=args.rounds,
                delta_p=cipher.get_default_delta_p()
            )
            train_data = gen.generate_balanced_dataset(args.samples)
            val_data = gen.generate_balanced_dataset(args.samples // 10)
            test_data = gen.generate_balanced_dataset(args.samples // 10)

            input_dim = get_input_dim(args.representation, cipher.block_size)

            metrics = quick_train_eval(
                model_name=model_name,
                input_dim=input_dim,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                representation=args.representation,
                block_size=cipher.block_size,
                batch_size=args.batch_size,
                n_epochs=args.epochs,
                device=device,
            )
            seed_results.append(metrics)
            print(f"acc={metrics['accuracy']:.4f}, time={metrics['train_time']:.1f}s")

        # Aggregate
        accs = [r['accuracy'] for r in seed_results]
        all_results[model_name] = {
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'accuracy_values': [float(a) for a in accs],
            'n_params': seed_results[0]['n_params'],
            'train_time_mean': float(np.mean([r['train_time'] for r in seed_results])),
            'infer_throughput': seed_results[0]['infer_throughput'],
        }
        print(f"  → {model_name}: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    # Handle RNN models (need 3D input)
    for model_name in RNN_MODELS:
        print(f"\n{'━' * 50}")
        print(f"  Model: {model_name} (sequence input)")
        print(f"{'━' * 50}")

        seed_results = []
        for seed in seeds:
            set_seed(seed)
            print(f"  Seed {seed}...", end=' ')

            gen = CipherDataGenerator(
                cipher=args.cipher, n_rounds=args.rounds,
                delta_p=cipher.get_default_delta_p()
            )
            train_data = gen.generate_balanced_dataset(args.samples)
            val_data = gen.generate_balanced_dataset(args.samples // 10)
            test_data = gen.generate_balanced_dataset(args.samples // 10)

            input_dim = get_input_dim(args.representation, cipher.block_size)

            # Build 3D data for RNN: (batch, seq_len=1, features)
            from data.representations import RepresentationFactory
            from torch.utils.data import DataLoader, TensorDataset
            import time as _time

            factory = RepresentationFactory(block_size=cipher.block_size)
            X_train = factory.get_representation(args.representation,
                                                  train_data['C'], train_data['C_prime'])
            X_val = factory.get_representation(args.representation,
                                                val_data['C'], val_data['C_prime'])
            X_test = factory.get_representation(args.representation,
                                                 test_data['C'], test_data['C_prime'])

            # Reshape to 3D: (batch, 1, features)
            X_train_t = torch.from_numpy(X_train).float().unsqueeze(1)
            X_val_t = torch.from_numpy(X_val).float().unsqueeze(1)
            X_test_t = torch.from_numpy(X_test).float().unsqueeze(1)

            Y_train = torch.from_numpy(train_data['labels']).float()
            Y_val = torch.from_numpy(val_data['labels']).float()
            Y_test = torch.from_numpy(test_data['labels']).float()

            train_loader = DataLoader(TensorDataset(X_train_t, Y_train),
                                      batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val_t, Y_val),
                                    batch_size=args.batch_size)
            test_loader = DataLoader(TensorDataset(X_test_t, Y_test),
                                     batch_size=args.batch_size)

            from models import get_model
            from training.trainer import Trainer
            from evaluation.metrics import evaluate_model

            model = get_model(model_name, input_dim=input_dim)
            n_params = sum(p.numel() for p in model.parameters())

            trainer = Trainer(model=model, train_loader=train_loader,
                              val_loader=val_loader, device=device, use_wandb=False)

            t0 = _time.time()
            trainer.train(n_epochs=args.epochs, early_stopping_patience=5, save_best=False)
            train_time = _time.time() - t0

            metrics = evaluate_model(model, test_loader, device)

            seed_results.append({
                'accuracy': float(metrics['accuracy']),
                'n_params': n_params,
                'train_time': round(train_time, 2),
            })
            print(f"acc={metrics['accuracy']:.4f}, time={train_time:.1f}s")

        accs = [r['accuracy'] for r in seed_results]
        all_results[model_name] = {
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'accuracy_values': [float(a) for a in accs],
            'n_params': seed_results[0]['n_params'],
            'train_time_mean': float(np.mean([r['train_time'] for r in seed_results])),
            'infer_throughput': 0,
        }
        print(f"  → {model_name}: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    # Print comparison table
    print(f"\n{'═' * 70}")
    print(f"  {'Model':<12} {'Params':>10} {'Accuracy':>16} {'Train (s)':>10}")
    print(f"{'─' * 70}")
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]['accuracy_mean']):
        print(f"  {name:<12} {r['n_params']:>10,} "
              f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f} "
              f"{r['train_time_mean']:>10.1f}")
    print(f"{'═' * 70}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    names = list(all_results.keys())
    means = [all_results[n]['accuracy_mean'] for n in names]
    stds = [all_results[n]['accuracy_std'] for n in names]
    params = [all_results[n]['n_params'] for n in names]

    # Accuracy bar chart with error bars
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars = axes[0].bar(names, means, yerr=stds, capsize=5, color=colors,
                       edgecolor='black', alpha=0.85)
    axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.4)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'Model Comparison — {args.cipher.upper()} ({args.rounds}r)')
    axes[0].set_ylim(0.45, 1.0)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Accuracy vs Parameters scatter
    axes[1].scatter(params, means, s=120, c=colors, edgecolors='black', zorder=5)
    for i, name in enumerate(names):
        axes[1].annotate(name, (params[i], means[i]),
                         textcoords="offset points", xytext=(5, 5), fontsize=9)
    axes[1].set_xlabel('Parameters')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy vs Model Size')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'e13_{args.cipher}_r{args.rounds}.png', dpi=300)
    plt.close()

    save_results(all_results, str(output_dir),
                 f'e13_{args.cipher}_r{args.rounds}_results.json')

    print(f"\n✓ Results saved to {output_dir}")


if __name__ == '__main__':
    main()
