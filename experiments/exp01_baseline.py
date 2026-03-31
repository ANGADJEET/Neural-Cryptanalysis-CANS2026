#!/usr/bin/env python
"""
E01: Baseline Distinguisher (Multi-Seed)

Train a neural distinguisher and measure accuracy ± std across round counts.

Usage:
    python experiments/exp01_baseline.py --cipher speck32
    python experiments/exp01_baseline.py --cipher speck32 --n-seeds 5 --rounds 3 4 5 6 7 8 9
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
from visualization.plots import plot_accuracy_vs_rounds
from experiments.experiment_utils import (
    set_seed, add_common_args, save_results, get_device
)


def train_one_round(cipher_name, cipher, n_rounds, model_name, representation,
                     n_samples, batch_size, n_epochs, device):
    """Train and evaluate a distinguisher for a single round count."""
    gen = CipherDataGenerator(
        cipher=cipher_name, n_rounds=n_rounds,
        delta_p=cipher.get_default_delta_p()
    )
    train_data = gen.generate_balanced_dataset(n_samples)
    val_data = gen.generate_balanced_dataset(n_samples // 10)
    test_data = gen.generate_balanced_dataset(n_samples // 10)

    input_dim = get_input_dim(representation, cipher.block_size)
    train_ds = CryptoDataset(train_data, representation, cipher.block_size)
    val_ds = CryptoDataset(val_data, representation, cipher.block_size)
    test_ds = CryptoDataset(test_data, representation, cipher.block_size)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = get_model(model_name, input_dim=input_dim)
    trainer = Trainer(model=model, train_loader=train_loader,
                      val_loader=val_loader, device=device, use_wandb=False)
    trainer.train(n_epochs=n_epochs, early_stopping_patience=5, save_best=False)

    metrics = evaluate_model(model, test_loader, device)
    return {
        'accuracy': float(metrics['accuracy']),
        'advantage': float(metrics['advantage']),
    }


def single_run(seed, args):
    """One seed: accuracy at each round count."""
    set_seed(seed)
    cipher = get_cipher(args.cipher)
    device = get_device(args)

    results = {}
    for n_rounds in args.round_list:
        print(f"    Round {n_rounds}...", end=' ')
        r = train_one_round(
            args.cipher, cipher, n_rounds, args.model,
            args.representation, args.samples, args.batch_size,
            args.epochs, device
        )
        results[str(n_rounds)] = r['accuracy']
        print(f"acc={r['accuracy']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='E01: Baseline Distinguisher')
    add_common_args(parser)
    parser.add_argument('--rounds', type=int, nargs='+', default=None)
    parser.add_argument('--model', default='gohr_mlp',
                        choices=['gohr_mlp', 'mlp', 'cnn', 'resnet'])
    parser.add_argument('--representation', default='R2_xor_diff')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = './results/e01_baseline'

    print("=" * 60)
    print("  E01: Baseline Distinguisher Experiment")
    print("=" * 60)

    cipher = get_cipher(args.cipher)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.rounds is None:
        if args.cipher == 'speck32':
            args.round_list = list(range(3, 10))
        elif args.cipher == 'simon32':
            args.round_list = list(range(4, 12))
        else:
            args.round_list = list(range(2, 8))
    else:
        args.round_list = args.rounds

    # Multi-seed runs
    seeds = [args.seed + i for i in range(args.n_seeds)]
    all_runs = []

    for i, seed in enumerate(seeds):
        print(f"\n┌─ Seed {seed} ({i+1}/{len(seeds)}) ─────────────────┐")
        result = single_run(seed, args)
        all_runs.append(result)
        print(f"└─ Done ─────────────────────────────────────┘")

    # Aggregate
    aggregated = {}
    for r in args.round_list:
        key = str(r)
        vals = [run[key] for run in all_runs if key in run]
        aggregated[key] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'values': [float(v) for v in vals],
        }

    # Plot with error bars
    fig, ax = plt.subplots(figsize=(9, 6))
    rounds = sorted([int(k) for k in aggregated.keys()])
    means = [aggregated[str(r)]['mean'] for r in rounds]
    stds = [aggregated[str(r)]['std'] for r in rounds]

    ax.errorbar(rounds, means, yerr=stds, fmt='bo-', linewidth=2,
                markersize=8, capsize=5, capthick=2, label=args.model)
    ax.fill_between(rounds, np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds), alpha=0.15, color='blue')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.4, label='Random')
    ax.set_xlabel('Number of Rounds', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title(f'Baseline Distinguisher — {args.cipher.upper()} '
                 f'({args.n_seeds} seeds)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.45, 1.0)
    ax.set_xticks(rounds)

    plt.tight_layout()
    plt.savefig(output_dir / f'e01_{args.cipher}.png', dpi=300)
    plt.close()

    results = {
        'per_round': aggregated,
        '_seeds': seeds,
        '_n_seeds': len(seeds),
    }
    save_results(results, str(output_dir), f'e01_{args.cipher}_results.json')

    # Summary
    print(f"\n{'═' * 50}")
    print(f"  {'Round':>5}  {'Accuracy':>16}")
    print(f"{'─' * 50}")
    for r in rounds:
        a = aggregated[str(r)]
        print(f"  {r:>5}  {a['mean']:.4f} ± {a['std']:.4f}")
    print(f"{'═' * 50}")

    print(f"\n✓ Results saved to {output_dir}")


if __name__ == '__main__':
    main()
