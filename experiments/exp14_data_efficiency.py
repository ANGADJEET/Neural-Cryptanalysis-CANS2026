#!/usr/bin/env python
"""
E14: Data Efficiency Curve

Measure how accuracy scales with training set size.
Critical for understanding the sample complexity of neural distinguishers.

Usage:
    python experiments/exp14_data_efficiency.py --cipher speck32 --rounds 5
    python experiments/exp14_data_efficiency.py --cipher speck32 --rounds 5 --n-seeds 5
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
from models import get_model
from training.trainer import Trainer
from evaluation.metrics import evaluate_model
from experiments.experiment_utils import (
    set_seed, add_common_args, save_results, get_device
)

DATASET_SIZES = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]


def single_run(seed, args):
    """One seed: accuracy at each dataset size."""
    set_seed(seed)
    cipher = get_cipher(args.cipher)
    device = get_device(args)

    # Generate the largest dataset, then subsample
    gen = CipherDataGenerator(
        cipher=args.cipher, n_rounds=args.rounds,
        delta_p=cipher.get_default_delta_p()
    )
    max_size = max(args.sizes)
    full_data = gen.generate_balanced_dataset(max_size)
    test_data = gen.generate_balanced_dataset(50_000)  # Fixed test set

    input_dim = get_input_dim('R2_xor_diff', cipher.block_size)
    test_ds = CryptoDataset(test_data, 'R2_xor_diff', cipher.block_size)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    results = {}
    for size in args.sizes:
        print(f"    N={size:>10,}...", end=' ')

        # Subsample
        if size <= max_size:
            sub_data = {k: v[:size] if hasattr(v, '__len__') and len(v) >= size else v
                        for k, v in full_data.items()}
        else:
            sub_data = gen.generate_balanced_dataset(size)

        val_size = min(size // 5, 50_000)
        val_data = gen.generate_balanced_dataset(val_size)

        train_ds = CryptoDataset(sub_data, 'R2_xor_diff', cipher.block_size)
        val_ds = CryptoDataset(val_data, 'R2_xor_diff', cipher.block_size)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

        model = get_model('gohr_mlp', input_dim=input_dim)
        trainer = Trainer(model=model, train_loader=train_loader,
                          val_loader=val_loader, device=device, use_wandb=False)
        trainer.train(n_epochs=args.epochs, early_stopping_patience=5, save_best=False)

        metrics = evaluate_model(model, test_loader, device)
        results[str(size)] = float(metrics['accuracy'])
        print(f"acc={metrics['accuracy']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='E14: Data Efficiency')
    add_common_args(parser)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--sizes', type=int, nargs='+', default=None,
                        help='Training set sizes to test')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = './results/e14_data_efficiency'
    if args.sizes is None:
        args.sizes = DATASET_SIZES

    print("=" * 60)
    print("  E14: Data Efficiency Curve")
    print("=" * 60)

    seeds = [args.seed + i for i in range(args.n_seeds)]
    all_runs = []

    for i, seed in enumerate(seeds):
        print(f"\n┌─ Seed {seed} ({i+1}/{len(seeds)}) ─────────────────┐")
        result = single_run(seed, args)
        all_runs.append(result)
        print(f"└─ Done ─────────────────────────────────────┘")

    # Aggregate: mean ± std per size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sizes = args.sizes
    aggregated = {}
    for s in sizes:
        key = str(s)
        vals = [run[key] for run in all_runs if key in run]
        if vals:
            aggregated[key] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'values': [float(v) for v in vals],
            }

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    x = sorted([int(k) for k in aggregated.keys()])
    means = [aggregated[str(s)]['mean'] for s in x]
    stds = [aggregated[str(s)]['std'] for s in x]

    ax.errorbar(x, means, yerr=stds, fmt='bo-', linewidth=2,
                markersize=8, capsize=5, capthick=2, label='Gohr MLP')
    ax.fill_between(x, np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds), alpha=0.15, color='blue')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.4, label='Random baseline')
    ax.set_xscale('log')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Test Accuracy')
    ax.set_title(f'Data Efficiency — {args.cipher.upper()} ({args.rounds}r)')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.45, 1.0)

    plt.tight_layout()
    plt.savefig(output_dir / f'e14_{args.cipher}_r{args.rounds}.png', dpi=300)
    plt.close()

    results = {
        'sizes': aggregated,
        '_seeds': seeds,
        '_n_seeds': len(seeds),
        'cipher': args.cipher,
        'rounds': args.rounds,
    }
    save_results(results, str(output_dir),
                 f'e14_{args.cipher}_r{args.rounds}_results.json')

    # Summary table
    print(f"\n{'═' * 50}")
    print(f"  {'Size':>12}  {'Accuracy':>16}")
    print(f"{'─' * 50}")
    for s in x:
        a = aggregated[str(s)]
        print(f"  {s:>12,}  {a['mean']:.4f} ± {a['std']:.4f}")
    print(f"{'═' * 50}")

    print(f"\n✓ Done")


if __name__ == '__main__':
    main()
