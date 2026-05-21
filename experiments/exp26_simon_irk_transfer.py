#!/usr/bin/env python3
"""
E26: SIMON32-IRK vs Standard SIMON32 Transfer Comparison

Tests Gohr et al.'s theorem directly: if Feistel distinguishers learn only DDT
features (which requires independent round keys), then SIMON32-IRK should show
positive or neutral transfer. Standard SIMON32 may show anti-transfer due to
key schedule dependence.

Design:
  For each variant in {SIMON32, SIMON32-IRK}:
    1. Train distinguisher at 6 rounds
    2. Evaluate on 4, 5, 7, 8 rounds
    3. Record transfer polarity with t-test
    4. Compare polarity between variants

Usage:
  python experiments/exp26_simon_irk_transfer.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from ciphers import get_cipher
from ciphers.simon32_irk import Simon32IRK
from data.generator import CipherDataGenerator
from data.dataloader import CryptoDataset, get_input_dim
from data.representations import RepresentationFactory
from models import get_model
from training.trainer import Trainer
from evaluation.metrics import evaluate_model
from experiments.experiment_utils import (
    set_seed, save_results, get_device
)


SOURCE_ROUNDS = 6
TARGET_ROUNDS = [4, 5, 7, 8]


def train_and_transfer_irk(
    cipher_obj,
    cipher_label: str,
    source_rounds: int,
    target_rounds: list,
    n_samples: int,
    n_epochs: int,
    batch_size: int,
    device: str,
    seed: int,
) -> dict:
    """Train on source round, evaluate transfer to target rounds.
    
    Works with both standard SIMON32 and SIMON32-IRK because it
    uses the cipher object directly (not the string-based factory
    for data generation).
    """
    set_seed(seed)
    block_size = cipher_obj.block_size
    delta_p = cipher_obj.get_default_delta_p()

    # Generate training data manually (CipherDataGenerator uses string names,
    # but SIMON32-IRK isn't registered in the cipher factory)
    key = cipher_obj.random_key(n_rounds=max(source_rounds, max(target_rounds)) + 1) \
        if hasattr(cipher_obj, 'default_rounds') else cipher_obj.random_key()

    half = n_samples // 2
    factory = RepresentationFactory(block_size=block_size)

    def make_dataset(n_rounds, n_total):
        h = n_total // 2
        # Positive pairs
        P = cipher_obj.random_plaintexts(h)
        P_prime = (P ^ delta_p).astype(P.dtype)
        C_pos = cipher_obj.encrypt(P, n_rounds, key)
        C_prime_pos = cipher_obj.encrypt(P_prime, n_rounds, key)

        # Negative pairs (Gohr-style: random P1, P2 with P1^P2 != delta_p)
        Q1 = cipher_obj.random_plaintexts(h)
        Q2 = cipher_obj.random_plaintexts(h)
        mask = (Q1 ^ Q2) == delta_p
        while np.any(mask):
            Q2[mask] = cipher_obj.random_plaintexts(np.sum(mask))
            mask = (Q1 ^ Q2) == delta_p
        C_neg = cipher_obj.encrypt(Q1, n_rounds, key)
        C_prime_neg = cipher_obj.encrypt(Q2, n_rounds, key)

        C_all = np.concatenate([C_pos, C_neg])
        C_prime_all = np.concatenate([C_prime_pos, C_prime_neg])
        labels = np.concatenate([np.ones(h, dtype=np.uint8), np.zeros(h, dtype=np.uint8)])

        perm = np.random.permutation(len(labels))
        return {
            'C': C_all[perm],
            'C_prime': C_prime_all[perm],
            'labels': labels[perm],
        }

    train_data = make_dataset(source_rounds, n_samples)
    val_data = make_dataset(source_rounds, n_samples // 10)

    input_dim = get_input_dim('R2_xor_diff', block_size)
    model = get_model('gohr_mlp', input_dim=input_dim)

    train_ds = CryptoDataset(train_data, 'R2_xor_diff', block_size)
    val_ds = CryptoDataset(val_data, 'R2_xor_diff', block_size)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        device=device, use_wandb=False
    )
    trainer.train(n_epochs=n_epochs, early_stopping_patience=5, save_best=False)

    src_metrics = evaluate_model(model, val_loader, device)
    source_acc = float(src_metrics['accuracy'])

    transfer_accs = {}
    for tr in target_rounds:
        tdata = make_dataset(tr, n_samples // 5)
        tds = CryptoDataset(tdata, 'R2_xor_diff', block_size)
        tloader = DataLoader(tds, batch_size=batch_size)
        tmetrics = evaluate_model(model, tloader, device)
        transfer_accs[tr] = float(tmetrics['accuracy'])

    return {'source_accuracy': source_acc, 'transfer_accs': transfer_accs}


def main():
    parser = argparse.ArgumentParser(
        description='E26: SIMON32-IRK vs Standard Transfer Comparison'
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-seeds', type=int, default=5)
    parser.add_argument('--samples', type=int, default=500000)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=5000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', default='./results/e26_simon_irk')
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("⚠ CUDA not available, using CPU")

    print("=" * 65)
    print("  E26: SIMON32-IRK vs Standard SIMON32 Transfer")
    print(f"  Source: {SOURCE_ROUNDS}r, Targets: {TARGET_ROUNDS}")
    print(f"  Seeds: {args.n_seeds}")
    print("=" * 65)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed + i for i in range(args.n_seeds)]
    variants = {
        'simon32_standard': get_cipher('simon32'),
        'simon32_irk': Simon32IRK(default_rounds=32),
    }

    all_results = {}

    for variant_name, cipher_obj in variants.items():
        print(f"\n{'━' * 60}")
        print(f"  Variant: {variant_name}")
        print(f"{'━' * 60}")

        seed_results = []
        for i, seed in enumerate(seeds):
            print(f"  Seed {seed} ({i+1}/{len(seeds)}): ", end='', flush=True)
            result = train_and_transfer_irk(
                cipher_obj=cipher_obj,
                cipher_label=variant_name,
                source_rounds=SOURCE_ROUNDS,
                target_rounds=TARGET_ROUNDS,
                n_samples=args.samples,
                n_epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                seed=seed,
            )
            seed_results.append(result)
            src = result['source_accuracy']
            transfers = ' '.join(
                f"{tr}r={result['transfer_accs'][tr]:.3f}"
                for tr in TARGET_ROUNDS
            )
            print(f"src={src:.3f}  {transfers}")

        # Aggregate
        variant_summary = {'per_target': {}}
        for tr in TARGET_ROUNDS:
            accs = np.array([r['transfer_accs'][tr] for r in seed_results])
            t_stat, p_value = stats.ttest_1samp(accs, 0.5) if len(accs) > 1 else (0, 1)
            polarity = 'anti' if accs.mean() < 0.49 and p_value < 0.05 else \
                       'positive' if accs.mean() > 0.51 and p_value < 0.05 else 'neutral'
            variant_summary['per_target'][str(tr)] = {
                'mean': float(accs.mean()),
                'std': float(accs.std()),
                'accs': accs.tolist(),
                't_stat': float(t_stat),
                'p_value': float(p_value),
                'polarity': polarity,
            }
        variant_summary['source_accs'] = [r['source_accuracy'] for r in seed_results]
        all_results[variant_name] = variant_summary

    # ── Comparison Summary ──────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print(f"  Comparison: Standard SIMON32 vs SIMON32-IRK")
    print(f"{'═' * 65}")
    print(f"  {'Round':<8} {'Standard':<20} {'IRK':<20} {'Polarity Change?'}")
    print(f"{'─' * 65}")

    for tr in TARGET_ROUNDS:
        std = all_results['simon32_standard']['per_target'][str(tr)]
        irk = all_results['simon32_irk']['per_target'][str(tr)]
        changed = std['polarity'] != irk['polarity']
        marker = '★ DIFFERENT' if changed else '  same'
        print(f"  {tr}r{'':<5} {std['mean']:.3f} ({std['polarity']:<8})  "
              f"{irk['mean']:.3f} ({irk['polarity']:<8})  {marker}")

    print(f"{'═' * 65}")

    # ── Plot ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for variant_name, color, marker in [
        ('simon32_standard', '#e74c3c', 'o'),
        ('simon32_irk', '#3498db', 's'),
    ]:
        means = [all_results[variant_name]['per_target'][str(tr)]['mean']
                 for tr in TARGET_ROUNDS]
        stds = [all_results[variant_name]['per_target'][str(tr)]['std']
                for tr in TARGET_ROUNDS]
        label = 'Standard SIMON32' if 'standard' in variant_name else 'SIMON32-IRK'
        ax.errorbar(TARGET_ROUNDS, means, yerr=stds, fmt=f'{marker}-',
                     color=color, linewidth=2, markersize=8, capsize=5, label=label)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Target Rounds', fontsize=12)
    ax.set_ylabel('Transfer Accuracy', fontsize=12)
    ax.set_title(f'SIMON32: Key Schedule vs Independent Round Keys\n'
                 f'(trained on {SOURCE_ROUNDS}r, {args.n_seeds} seeds)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'e26_simon_irk_comparison.png', dpi=300)
    plt.close()

    save_results(all_results, str(output_dir), 'e26_results.json')
    print(f"\n✓ Results saved to {output_dir}")


if __name__ == '__main__':
    main()
