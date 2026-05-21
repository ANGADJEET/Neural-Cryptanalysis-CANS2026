#!/usr/bin/env python3
"""
E09b: Input Difference Invariance of Transfer Polarity

Tests whether anti-transfer (ARX/Feistel) and positive transfer (SPN)
persist across different input differences ΔP. If transfer polarity is
ΔP-specific, the compositionality claim is weakened. If it persists
across all tested ΔP, it's a cipher-family property.

Key design decisions:
  - Each ΔP gets its own independently-trained distinguisher (no reuse)
  - Statistical significance via one-sample t-test against chance (0.5)
  - Reports polarity as: anti (<49%), neutral (49-51%), positive (>51%)
  - Uses Gohr negatives to match the main E09 protocol

Usage:
  python experiments/exp09b_delta_invariance.py --cipher speck32
  python experiments/exp09b_delta_invariance.py --cipher simon32
  python experiments/exp09b_delta_invariance.py --cipher present
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
from data.generator import CipherDataGenerator
from data.dataloader import CryptoDataset, get_input_dim
from models import get_model
from training.trainer import Trainer
from evaluation.metrics import evaluate_model
from experiments.experiment_utils import (
    set_seed, add_common_args, save_results, get_device
)


# ─── Cipher-specific ΔP sets ───────────────────────────────────────────
# Each set includes the default ΔP plus 4 alternatives chosen for
# structural diversity (different active word positions, weights).
DELTA_P_SETS = {
    'speck32': {
        'source_rounds': 5,
        'target_rounds': [3, 4, 6, 7],
        'deltas': [
            (0x00400000, 'default: bit 22'),
            (0x00800000, 'bit 23'),
            (0x00200000, 'bit 21'),
            (0x00008000, 'bit 15 (right word MSB)'),
            (0x00100000, 'bit 20'),
        ],
    },
    'simon32': {
        'source_rounds': 6,
        'target_rounds': [4, 5, 7, 8],
        'deltas': [
            (0x00000001, 'default: bit 0'),
            (0x00000004, 'bit 2'),
            (0x00000040, 'bit 6'),
            (0x00000100, 'bit 8 (left word bit 0)'),
            (0x00001000, 'bit 12'),
        ],
    },
    'present': {
        'source_rounds': 4,
        'target_rounds': [2, 3, 5, 6],
        'deltas': [
            (0x0000000000000001, 'default: bit 0'),
            (0x0000000000000002, 'bit 1'),
            (0x0000000000000010, 'bit 4 (S-box 1)'),
            (0x0000000000000100, 'bit 8 (S-box 2)'),
        ],
    },
}


def classify_polarity(accuracy: float, p_value: float, alpha: float = 0.05) -> str:
    """Classify transfer polarity with statistical rigor."""
    if p_value >= alpha:
        return 'neutral'  # Cannot reject H0: acc = 0.5
    return 'anti' if accuracy < 0.5 else 'positive'


def train_and_transfer(
    cipher_name: str,
    delta_p: int,
    source_rounds: int,
    target_rounds: list,
    n_samples: int,
    n_epochs: int,
    batch_size: int,
    device: str,
    seed: int,
) -> dict:
    """Train a distinguisher with given ΔP and evaluate cross-round transfer.
    
    Returns dict mapping target_round -> accuracy.
    """
    set_seed(seed)
    cipher = get_cipher(cipher_name)

    # Train on source round with this specific ΔP
    gen = CipherDataGenerator(
        cipher=cipher_name, n_rounds=source_rounds, delta_p=delta_p, seed=seed
    )
    train_data = gen.generate_balanced_dataset(n_samples, negative_type='gohr')
    val_data = gen.generate_balanced_dataset(n_samples // 10, negative_type='gohr')

    input_dim = get_input_dim('R2_xor_diff', cipher.block_size)
    model = get_model('gohr_mlp', input_dim=input_dim)

    train_ds = CryptoDataset(train_data, 'R2_xor_diff', cipher.block_size)
    val_ds = CryptoDataset(val_data, 'R2_xor_diff', cipher.block_size)

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

    # Evaluate on each target round (same ΔP)
    transfer_accs = {}
    for tr in target_rounds:
        tgen = CipherDataGenerator(
            cipher=cipher_name, n_rounds=tr, delta_p=delta_p, seed=seed + tr * 1000
        )
        tdata = tgen.generate_balanced_dataset(n_samples // 5, negative_type='gohr')
        tds = CryptoDataset(tdata, 'R2_xor_diff', cipher.block_size)
        tloader = DataLoader(tds, batch_size=batch_size)
        tmetrics = evaluate_model(model, tloader, device)
        transfer_accs[tr] = float(tmetrics['accuracy'])

    return {
        'source_accuracy': source_acc,
        'transfer_accs': transfer_accs,
    }


def main():
    parser = argparse.ArgumentParser(
        description='E09b: ΔP-Invariance of Transfer Polarity'
    )
    add_common_args(parser)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = f'./results/e09b_delta_invariance'

    if args.cipher not in DELTA_P_SETS:
        print(f"No ΔP set configured for {args.cipher}. "
              f"Available: {list(DELTA_P_SETS.keys())}")
        return

    config = DELTA_P_SETS[args.cipher]
    source_rounds = config['source_rounds']
    target_rounds = config['target_rounds']
    deltas = config['deltas']
    device = get_device(args)

    print("=" * 65)
    print(f"  E09b: ΔP-Invariance of Transfer Polarity — {args.cipher.upper()}")
    print(f"  Source: {source_rounds}r, Targets: {target_rounds}")
    print(f"  Testing {len(deltas)} input differences × {args.n_seeds} seeds")
    print("=" * 65)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed + i for i in range(args.n_seeds)]
    all_results = {}

    for delta_p, delta_desc in deltas:
        delta_hex = f"0x{delta_p:08x}" if delta_p < 2**32 else f"0x{delta_p:016x}"
        print(f"\n{'━' * 60}")
        print(f"  ΔP = {delta_hex}  ({delta_desc})")
        print(f"{'━' * 60}")

        seed_results = []
        for i, seed in enumerate(seeds):
            print(f"  Seed {seed} ({i+1}/{len(seeds)}): ", end='', flush=True)
            result = train_and_transfer(
                cipher_name=args.cipher,
                delta_p=delta_p,
                source_rounds=source_rounds,
                target_rounds=target_rounds,
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
                for tr in sorted(result['transfer_accs'].keys())
            )
            print(f"src={src:.3f}  {transfers}")

        # Aggregate per-target-round statistics
        delta_summary = {
            'delta_p': delta_p,
            'delta_hex': delta_hex,
            'delta_desc': delta_desc,
            'source_accs': [r['source_accuracy'] for r in seed_results],
            'source_mean': float(np.mean([r['source_accuracy'] for r in seed_results])),
            'per_target': {},
        }

        for tr in target_rounds:
            accs = np.array([r['transfer_accs'][tr] for r in seed_results])
            mean_acc = float(accs.mean())
            std_acc = float(accs.std())

            # One-sample t-test: H0 = accuracy is 0.5 (no transfer)
            if len(accs) > 1:
                t_stat, p_value = stats.ttest_1samp(accs, 0.5)
            else:
                t_stat, p_value = 0.0, 1.0

            polarity = classify_polarity(mean_acc, p_value)

            delta_summary['per_target'][str(tr)] = {
                'accs': accs.tolist(),
                'mean': mean_acc,
                'std': std_acc,
                't_stat': float(t_stat),
                'p_value': float(p_value),
                'polarity': polarity,
            }

        all_results[delta_hex] = delta_summary

    # ─── Summary Table ──────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  Summary: Transfer Polarity Across Input Differences")
    print(f"{'═' * 70}")

    header = f"  {'ΔP':<18}"
    for tr in target_rounds:
        header += f" {tr}r{'':>8}"
    header += "  Consistent?"
    print(header)
    print(f"{'─' * 70}")

    consistent_count = 0
    for delta_hex, summary in all_results.items():
        row = f"  {delta_hex:<18}"
        polarities = []
        for tr in target_rounds:
            d = summary['per_target'][str(tr)]
            pol_symbol = {'anti': '↓', 'positive': '↑', 'neutral': '─'}[d['polarity']]
            sig = '*' if d['p_value'] < 0.05 else ' '
            row += f" {d['mean']:.3f}{pol_symbol}{sig}   "
            polarities.append(d['polarity'])

        # Check consistency: all non-neutral entries have same polarity
        non_neutral = [p for p in polarities if p != 'neutral']
        if non_neutral and len(set(non_neutral)) == 1:
            row += f"  ✓ all {non_neutral[0]}"
            consistent_count += 1
        elif not non_neutral:
            row += "  ─ all neutral"
        else:
            row += "  ✗ MIXED"
        print(row)

    print(f"{'═' * 70}")
    print(f"  Consistent ΔP: {consistent_count}/{len(deltas)}")

    # ─── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(deltas)))

    for idx, (delta_hex, summary) in enumerate(all_results.items()):
        means = [summary['per_target'][str(tr)]['mean'] for tr in target_rounds]
        stds = [summary['per_target'][str(tr)]['std'] for tr in target_rounds]
        label = f"ΔP={delta_hex}"
        ax.errorbar(target_rounds, means, yerr=stds, fmt='o-',
                     color=colors[idx], linewidth=2, markersize=6,
                     capsize=4, label=label)

    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, label='Chance')
    ax.axhline(y=summary['source_mean'], color='green', linestyle='--',
               alpha=0.4, label=f'Source ({source_rounds}r)')
    ax.set_xlabel('Target Rounds', fontsize=12)
    ax.set_ylabel('Transfer Accuracy', fontsize=12)
    ax.set_title(f'{args.cipher.upper()} — Transfer Polarity vs ΔP ({args.n_seeds} seeds)',
                 fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'e09b_{args.cipher}_delta_invariance.png', dpi=300)
    plt.close()

    # Save full results
    save_results(
        all_results,
        str(output_dir),
        f'e09b_{args.cipher}_results.json'
    )
    print(f"\n✓ Results saved to {output_dir}")


if __name__ == '__main__':
    main()
