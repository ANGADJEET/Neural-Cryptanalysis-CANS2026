
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
    set_seed(seed)
    cipher = get_cipher(args.cipher)
    device = get_device(args)

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

    cross_cipher = {}
    cross_cipher_errors = {}
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
            # Recorded rather than swallowed, so a failed evaluation shows up
            # in the results file instead of quietly reducing the sample count.
            cross_cipher_errors[tc] = f"{type(e).__name__}: {e}"
            print(f"    → {tc}: ERROR - {e}")

    return {
        'source_accuracy': float(source_metrics['accuracy']),
        'cross_round': cross_round,
        'cross_cipher': cross_cipher,
        'cross_cipher_errors': cross_cipher_errors,
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

    target_rounds = [r for r in args.target_rounds if r != args.source_rounds]
    cr_agg = {}
    for tr in target_rounds:
        key = str(tr)
        # Aggregate only over runs that actually produced this target round.
        # Imputing a missing run as 0.5 would silently pull the mean toward
        # chance and shrink the reported std.
        vals = [r['cross_round'][key] for r in all_runs if key in r['cross_round']]
        if not vals:
            raise RuntimeError(
                f"no run produced a cross-round result for target {key}; "
                f"{len(all_runs)} run(s) attempted"
            )
        if len(vals) < len(all_runs):
            print(f"  WARNING: target {key} aggregated over {len(vals)}/{len(all_runs)} runs")
        cr_agg[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                       'n_runs': len(vals), 'n_attempted': len(all_runs)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

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

    cc_names = sorted(set(k for r in all_runs for k in r['cross_cipher']))
    # Same rule as above: a target cipher whose evaluation raised is left out
    # of its own bar rather than counted as chance.
    cc_vals = {c: [r['cross_cipher'][c] for r in all_runs if c in r['cross_cipher']]
               for c in cc_names}
    for c, vals in cc_vals.items():
        if len(vals) < len(all_runs):
            print(f"  WARNING: cross-cipher {c} aggregated over {len(vals)}/{len(all_runs)} runs")
    if cc_names:
        cc_means = [float(np.mean(cc_vals[c])) for c in cc_names]
        cc_stds = [float(np.std(cc_vals[c])) for c in cc_names]
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

    cc_agg = {c: {'mean': float(np.mean(cc_vals[c])), 'std': float(np.std(cc_vals[c])),
                  'n_runs': len(cc_vals[c]), 'n_attempted': len(all_runs)}
              for c in cc_names}
    err_names = sorted(set(k for r in all_runs for k in r.get('cross_cipher_errors', {})))
    cc_errors = {c: [r['cross_cipher_errors'][c] for r in all_runs
                     if c in r.get('cross_cipher_errors', {})]
                 for c in err_names}

    save_results({'cross_round': cr_agg, 'cross_cipher': cc_agg,
                  'cross_cipher_errors': {c: e for c, e in cc_errors.items() if e},
                  'source_accuracy': src_mean, '_seeds': seeds},
                 str(output_dir), f'e09_{args.cipher}_results.json')
    print(f"\n✓ Done")


if __name__ == '__main__':
    main()
