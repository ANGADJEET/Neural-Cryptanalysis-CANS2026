#!/usr/bin/env python
"""
E08: Saliency Maps (Multi-Seed)

Compute gradient-based bit importance for a trained distinguisher.

Usage:
    python experiments/exp08_saliency.py --cipher speck32 --rounds 5 --n-seeds 3
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


def compute_saliency(model, data_loader, device, n_batches=10):
    """Compute input × gradient saliency map."""
    model.eval()
    all_saliency = []
    count = 0

    for X, Y in data_loader:
        if count >= n_batches:
            break
        # Only use positive class (real cipher text)
        mask = Y == 1
        if mask.sum() == 0:
            continue

        X_pos = X[mask].to(device).requires_grad_(True)
        output = model(X_pos).squeeze()
        output.sum().backward()

        # input × gradient (absolute)
        saliency = (X_pos * X_pos.grad).abs().detach().cpu().numpy()
        all_saliency.append(saliency)
        count += 1

    all_saliency = np.concatenate(all_saliency, axis=0)
    mean_saliency = all_saliency.mean(axis=0)

    # Normalize to [0, 1]
    if mean_saliency.max() > 0:
        mean_saliency = mean_saliency / mean_saliency.max()

    return mean_saliency


def single_run(seed, args):
    """One seed: train model, compute saliency."""
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

    input_dim = get_input_dim('R2_xor_diff', cipher.block_size)
    train_ds = CryptoDataset(train_data, 'R2_xor_diff', cipher.block_size)
    val_ds = CryptoDataset(val_data, 'R2_xor_diff', cipher.block_size)
    test_ds = CryptoDataset(test_data, 'R2_xor_diff', cipher.block_size)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = get_model('gohr_mlp', input_dim=input_dim)
    trainer = Trainer(model=model, train_loader=train_loader,
                      val_loader=val_loader, device=device, use_wandb=False)
    trainer.train(n_epochs=args.epochs, early_stopping_patience=5, save_best=False)

    metrics = evaluate_model(model, test_loader, device)
    saliency = compute_saliency(model, test_loader, device, n_batches=10)

    top_bits = np.argsort(saliency)[-5:][::-1]
    print(f"    acc={metrics['accuracy']:.4f}, top bits: {list(top_bits)}")

    return {
        'accuracy': float(metrics['accuracy']),
        'saliency': [float(s) for s in saliency],
        'top_5_bits': [int(b) for b in top_bits],
    }


def main():
    parser = argparse.ArgumentParser(description='E08: Saliency Maps')
    add_common_args(parser)
    parser.add_argument('--rounds', type=int, default=5)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = './results/e08_saliency'

    print("=" * 60)
    print("  E08: Saliency Maps")
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

    # Average saliency across seeds
    saliencies = np.array([r['saliency'] for r in all_runs])
    mean_sal = saliencies.mean(axis=0)
    std_sal = saliencies.std(axis=0)
    top_bits = np.argsort(mean_sal)[-5:][::-1]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    n_bits = len(mean_sal)
    colors = plt.cm.coolwarm(mean_sal)
    ax.bar(range(n_bits), mean_sal, yerr=std_sal, capsize=2,
           color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
    for b in top_bits:
        ax.annotate(f'b{b}', (b, mean_sal[b]),
                    textcoords="offset points", xytext=(0, 8),
                    ha='center', fontsize=8, fontweight='bold')
    ax.set_xlabel('Bit Position')
    ax.set_ylabel('Saliency (normalized)')
    ax.set_title(f'Saliency Map — {args.cipher.upper()} ({args.rounds}r, {args.n_seeds} seeds)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / f'e08_{args.cipher}_r{args.rounds}.png', dpi=300)
    plt.close()

    results = {
        'mean_saliency': [float(s) for s in mean_sal],
        'std_saliency': [float(s) for s in std_sal],
        'top_5_bits': [int(b) for b in top_bits],
        '_seeds': seeds,
    }
    save_results(results, str(output_dir),
                 f'e08_{args.cipher}_r{args.rounds}_results.json')
    print(f"\n  Top-5 bits: {list(top_bits)}")
    print(f"✓ Done")


if __name__ == '__main__':
    main()
