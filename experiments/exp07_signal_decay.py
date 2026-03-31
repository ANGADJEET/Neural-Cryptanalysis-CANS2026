
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

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

    rounds_range = list(range(args.min_rounds, args.max_rounds + 1))
    default_dp = cipher.get_default_delta_p()
    single_bit_diffs = [1 << i for i in range(0, cipher.block_size, 4)]
    diffs = sorted(set([default_dp] + single_bit_diffs[:6]))[:8]

    results = {}
    for n_rounds in rounds_range:
        for dp in diffs:
            key = f"{n_rounds}_{dp}"
            print(f"    r={n_rounds}, dp=0x{dp:08x}...", end=' ')
            try:
                gen = CipherDataGenerator(
                    cipher=args.cipher, n_rounds=n_rounds, delta_p=dp
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
                results[key] = float(metrics['accuracy'])
                print(f"acc={metrics['accuracy']:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")
                results[key] = 0.5

    return results


def main():
    parser = argparse.ArgumentParser(description='E07: Signal Decay Heatmap')
    add_common_args(parser)
    parser.add_argument('--min-rounds', type=int, default=2)
    parser.add_argument('--max-rounds', type=int, default=8)
    parser.add_argument('--quick-samples', type=int, default=100_000,
                        help='Smaller sample count per cell for speed')
    parser.add_argument('--quick-epochs', type=int, default=10)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = './results/e07_signal_decay'

    print("=" * 60)
    print("  E07: Signal Decay Heatmap")
    print("=" * 60)

    cipher = get_cipher(args.cipher)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed + i for i in range(args.n_seeds)]
    all_runs = []

    for i, seed in enumerate(seeds):
        print(f"\n┌─ Seed {seed} ({i+1}/{len(seeds)}) ─────────────────┐")
        result = single_run(seed, args)
        all_runs.append(result)
        print(f"└─ Done ─────────────────────────────────────┘")

    rounds_range = list(range(args.min_rounds, args.max_rounds + 1))
    default_dp = cipher.get_default_delta_p()
    single_bit_diffs = [1 << i for i in range(0, cipher.block_size, 4)]
    diffs = sorted(set([default_dp] + single_bit_diffs[:6]))[:8]

    heatmap = np.zeros((len(diffs), len(rounds_range)))
    for i, dp in enumerate(diffs):
        for j, r in enumerate(rounds_range):
            key = f"{r}_{dp}"
            vals = [run.get(key, 0.5) for run in all_runs]
            heatmap[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(10, 6))
    diff_labels = [f'0x{dp:08x}' for dp in diffs]
    sns.heatmap(heatmap, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=rounds_range, yticklabels=diff_labels,
                vmin=0.5, vmax=1.0, ax=ax)
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Input Difference (Δp)')
    ax.set_title(f'Signal Decay — {args.cipher.upper()} ({args.n_seeds} seeds)')

    plt.tight_layout()
    plt.savefig(output_dir / f'e07_{args.cipher}.png', dpi=300)
    plt.close()

    results = {
        'heatmap': heatmap.tolist(),
        'rounds': rounds_range,
        'diffs': [f'0x{dp:08x}' for dp in diffs],
        '_seeds': seeds,
    }
    save_results(results, str(output_dir), f'e07_{args.cipher}_results.json')
    print(f"\n✓ Done")


if __name__ == '__main__':
    main()
