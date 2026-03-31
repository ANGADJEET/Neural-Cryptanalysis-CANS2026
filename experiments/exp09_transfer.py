#!/usr/bin/env python
"""
E09: Transfer Learning (Multi-Seed)

Test cross-round and cross-cipher transfer of trained distinguishers.

Usage:
    python experiments/exp09_transfer.py --cipher speck32 --n-seeds 3
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


def single_run(seed, args):
    """One seed: train source, evaluate on targets."""
    set_seed(seed)
    cipher = get_cipher(args.cipher)
    device = get_device(args)

    # Train on source rounds
    gen = CipherDataGenerator(
        cipher=args.cipher, n_rounds=args.source_rounds,
        delta_p=cipher.get_default_delta_p()
    )
    train_data = gen.generate_balanced_dataset(args.samples)
    val_data = gen.generate_balanced_dataset(args.samples // 10)

    input_dim = get_input_dim('R2_xor_diff', cipher.block_size)
    train_ds = CryptoDataset(train_data, 'R2_xor_diff', cipher.block_size)
    val_ds = CryptoDataset(val_data, 'R2_xor_diff', cipher.block_size)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = get_model('gohr_mlp', input_dim=input_dim)
    trainer = Trainer(model=model, train_loader=train_loader,
                      val_loader=val_loader, device=device, use_wandb=False)
    trainer.train(n_epochs=args.epochs, early_stopping_patience=5, save_best=False)

    source_metrics = evaluate_model(model, val_loader, device)
    print(f"    Source ({args.cipher}, {args.source_rounds}r): {source_metrics['accuracy']:.4f}")

    # Cross-round transfer
    cross_round = {}
    target_rounds = [r for r in args.target_rounds if r != args.source_rounds]
    for tr in target_rounds:
        tgen = CipherDataGenerator(
            cipher=args.cipher, n_rounds=tr,
            delta_p=cipher.get_default_delta_p()
        )
        tdata = tgen.generate_balanced_dataset(args.samples // 10)
        tds = CryptoDataset(tdata, 'R2_xor_diff', cipher.block_size)
        tloader = DataLoader(tds, batch_size=args.batch_size)
        tmetrics = evaluate_model(model, tloader, device)
        cross_round[str(tr)] = float(tmetrics['accuracy'])
        print(f"    → {args.cipher} {tr}r: {tmetrics['accuracy']:.4f}")

    # Cross-cipher transfer (same block size only)
    cross_cipher = {}
    target_ciphers = [c for c in ['speck32', 'simon32'] if c != args.cipher]
    for tc in target_ciphers:
        try:
            tc_cipher = get_cipher(tc)
            if tc_cipher.block_size != cipher.block_size:
                continue
            tcgen = CipherDataGenerator(
                cipher=tc, n_rounds=args.source_rounds,
                delta_p=tc_cipher.get_default_delta_p()
            )
            tcdata = tcgen.generate_balanced_dataset(args.samples // 10)
            tcds = CryptoDataset(tcdata, 'R2_xor_diff', tc_cipher.block_size)
            tcloader = DataLoader(tcds, batch_size=args.batch_size)
            tcm = evaluate_model(model, tcloader, device)
            cross_cipher[tc] = float(tcm['accuracy'])
            print(f"    → {tc} {args.source_rounds}r: {tcm['accuracy']:.4f}")
        except Exception as e:
            print(f"    → {tc}: ERROR - {e}")

    return {
        'source_accuracy': float(source_metrics['accuracy']),
        'cross_round': cross_round,
        'cross_cipher': cross_cipher,
    }


def main():
    parser = argparse.ArgumentParser(description='E09: Transfer Learning')
    add_common_args(parser)
    parser.add_argument('--source-rounds', type=int, default=5)
    parser.add_argument('--target-rounds', type=int, nargs='+',
                        default=[3, 4, 5, 6, 7, 8])
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = './results/e09_transfer'

    print("=" * 60)
    print("  E09: Transfer Learning")
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

    # Aggregate cross-round
    target_rounds = [r for r in args.target_rounds if r != args.source_rounds]
    cr_agg = {}
    for tr in target_rounds:
        key = str(tr)
        vals = [r['cross_round'].get(key, 0.5) for r in all_runs]
        cr_agg[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Cross-round
    src_mean = float(np.mean([r['source_accuracy'] for r in all_runs]))
    r_keys = sorted([int(k) for k in cr_agg.keys()])
    r_means = [cr_agg[str(r)]['mean'] for r in r_keys]
    r_stds = [cr_agg[str(r)]['std'] for r in r_keys]

    axes[0].errorbar(r_keys, r_means, yerr=r_stds, fmt='bo-',
                     linewidth=2, markersize=8, capsize=5, label='Transfer')
    axes[0].axhline(y=src_mean, color='g', linestyle='--',
                    label=f'Source ({args.source_rounds}r)={src_mean:.3f}')
    axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    axes[0].set_xlabel('Target Rounds')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Cross-Round Transfer')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cross-cipher
    cc_names = list(set(k for r in all_runs for k in r['cross_cipher']))
    if cc_names:
        cc_means = [float(np.mean([r['cross_cipher'].get(c, 0.5) for r in all_runs])) for c in cc_names]
        cc_stds = [float(np.std([r['cross_cipher'].get(c, 0.5) for r in all_runs])) for c in cc_names]
        axes[1].bar(cc_names, cc_means, yerr=cc_stds, capsize=5,
                    color='coral', edgecolor='black', alpha=0.85)
        axes[1].axhline(y=src_mean, color='g', linestyle='--')
        axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Cross-Cipher Transfer')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Transfer — {args.cipher.upper()} ({args.n_seeds} seeds)')
    plt.tight_layout()
    plt.savefig(output_dir / f'e09_{args.cipher}.png', dpi=300)
    plt.close()

    save_results({'cross_round': cr_agg, 'source_accuracy': src_mean, '_seeds': seeds},
                 str(output_dir), f'e09_{args.cipher}_results.json')
    print(f"\n✓ Done")


if __name__ == '__main__':
    main()
