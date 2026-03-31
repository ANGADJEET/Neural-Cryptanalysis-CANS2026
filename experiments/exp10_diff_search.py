
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

    default_dp = cipher.get_default_delta_p()
    candidates = set([default_dp])
    for i in range(cipher.block_size):
        candidates.add(1 << i)
    candidates.update([0x00400000, 0x00800000, 0x00C00000,
                       0x40000000, 0x80000000,
                       0x00010000, 0x00020000])
    candidates = sorted(c for c in candidates if c > 0 and c < (1 << cipher.block_size))
    candidates = candidates[:args.max_candidates]

    results = {}
    for dp in candidates:
        dp_hex = f"0x{dp:08x}"
        print(f"    Δp={dp_hex}...", end=' ')
        try:
            gen = CipherDataGenerator(
                cipher=args.cipher, n_rounds=args.rounds, delta_p=dp
            )
            train_data = gen.generate_balanced_dataset(args.quick_samples)
            val_data = gen.generate_balanced_dataset(args.quick_samples // 10)
            test_data = gen.generate_balanced_dataset(args.quick_samples // 10)

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
            trainer.train(n_epochs=args.quick_epochs, early_stopping_patience=3,
                          save_best=False)

            metrics = evaluate_model(model, test_loader, device)
            results[dp_hex] = float(metrics['accuracy'])
            print(f"acc={metrics['accuracy']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
            results[dp_hex] = 0.5

    return results


def main():
    parser = argparse.ArgumentParser(description='E10: Difference Search')
    add_common_args(parser)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--max-candidates', type=int, default=20)
    parser.add_argument('--quick-samples', type=int, default=100_000)
    parser.add_argument('--quick-epochs', type=int, default=10)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = './results/e10_diff_search'

    print("=" * 60)
    print("  E10: Difference Search")
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

    all_keys = sorted(set(k for run in all_runs for k in run))
    aggregated = {}
    for k in all_keys:
        vals = [run.get(k, 0.5) for run in all_runs]
        aggregated[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    ranked = sorted(aggregated.items(), key=lambda x: -x[1]['mean'])

    top_n = min(15, len(ranked))
    names = [r[0] for r in ranked[:top_n]]
    means = [r[1]['mean'] for r in ranked[:top_n]]
    stds = [r[1]['std'] for r in ranked[:top_n]]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))
    ax.barh(range(len(names)), means, xerr=stds, capsize=4,
            color=colors, edgecolor='black', alpha=0.85, height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Accuracy')
    ax.set_title(f'Difference Search — {args.cipher.upper()} ({args.rounds}r, {args.n_seeds} seeds)')
    ax.axvline(x=0.5, color='r', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / f'e10_{args.cipher}_r{args.rounds}.png', dpi=300)
    plt.close()

    results = {'diffs': {k: v for k, v in aggregated.items()},
               'ranking': [r[0] for r in ranked], '_seeds': seeds}
    save_results(results, str(output_dir),
                 f'e10_{args.cipher}_r{args.rounds}_results.json')

    print(f"\n{'═' * 55}")
    print(f"  Top-5 Δp:")
    for name, data in ranked[:5]:
        print(f"    {name}: {data['mean']:.4f} ± {data['std']:.4f}")
    print(f"{'═' * 55}")
    print(f"\n✓ Done")


if __name__ == '__main__':
    main()
