#!/usr/bin/env python
"""
E02: Representation Analysis (Multi-Seed)

Compare all input representations on the same cipher and round count.

Usage:
    python experiments/exp02_representation.py --cipher speck32 --rounds 5
    python experiments/exp02_representation.py --cipher speck32 --rounds 5 --n-seeds 5
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

REPRESENTATIONS = [
    'R1_raw_pair', 'R2_xor_diff', 'R3_concat',
    'R4_bit_sliced', 'R5_word_level', 'R8_statistical',
]


def single_run(seed, args):
    """One seed: accuracy for each representation."""
    set_seed(seed)
    cipher = get_cipher(args.cipher)
    device = get_device(args)

    gen = CipherDataGenerator(
        cipher=args.cipher, n_rounds=args.rounds,
        delta_p=cipher.get_default_delta_p()
    )
    train_data = gen.generate_balanced_dataset(args.samples)
    val_data = gen.generate_balanced_dataset(args.samples // 10)
    test_data = gen.generate_balanced_dataset(args.samples // 10)

    results = {}
    for repr_name in REPRESENTATIONS:
        print(f"    {repr_name}...", end=' ')
        try:
            input_dim = get_input_dim(repr_name, cipher.block_size)
            train_ds = CryptoDataset(train_data, repr_name, cipher.block_size)
            val_ds = CryptoDataset(val_data, repr_name, cipher.block_size)
            test_ds = CryptoDataset(test_data, repr_name, cipher.block_size)

            from torch.utils.data import DataLoader
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size)

            model = get_model('mlp', input_dim=input_dim)
            trainer = Trainer(model=model, train_loader=train_loader,
                              val_loader=val_loader, device=device, use_wandb=False)
            trainer.train(n_epochs=args.epochs, early_stopping_patience=3, save_best=False)

            metrics = evaluate_model(model, test_loader, device)
            results[repr_name] = float(metrics['accuracy'])
            print(f"acc={metrics['accuracy']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
            results[repr_name] = 0.5

    return results


def main():
    parser = argparse.ArgumentParser(description='E02: Representation Analysis')
    add_common_args(parser)
    parser.add_argument('--rounds', type=int, default=5)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = './results/e02_representation'

    print("=" * 60)
    print("  E02: Representation Analysis")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed + i for i in range(args.n_seeds)]
    all_runs = []

    for i, seed in enumerate(seeds):
        print(f"\n┌─ Seed {seed} ({i+1}/{len(seeds)}) ─────────────────┐")
        result = single_run(seed, args)
        all_runs.append(result)
        print(f"└─ Done ─────────────────────────────────────┘")

    # Aggregate
    aggregated = {}
    for repr_name in REPRESENTATIONS:
        vals = [run[repr_name] for run in all_runs if repr_name in run]
        aggregated[repr_name] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'values': [float(v) for v in vals],
        }

    # Plot with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(aggregated.keys())
    means = [aggregated[n]['mean'] for n in names]
    stds = [aggregated[n]['std'] for n in names]

    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=5,
                  color=colors, edgecolor='black', alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=9)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.4)
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Representation Comparison — {args.cipher.upper()} ({args.rounds}r, {args.n_seeds} seeds)')
    ax.set_ylim(0.45, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / f'e02_{args.cipher}_r{args.rounds}.png', dpi=300)
    plt.close()

    results = {'representations': aggregated, '_seeds': seeds}
    save_results(results, str(output_dir),
                 f'e02_{args.cipher}_r{args.rounds}_results.json')

    # Summary
    print(f"\n{'═' * 55}")
    for n in names:
        a = aggregated[n]
        print(f"  {n:<20} {a['mean']:.4f} ± {a['std']:.4f}")
    print(f"{'═' * 55}")
    print(f"\n✓ Done")


if __name__ == '__main__':
    main()
