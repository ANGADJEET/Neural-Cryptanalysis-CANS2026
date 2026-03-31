
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

    baseline_metrics = evaluate_model(model, test_loader, device)
    baseline_acc = float(baseline_metrics['accuracy'])
    print(f"    Baseline accuracy: {baseline_acc:.4f}")

    model.eval()
    perm_accs = []
    for t in range(args.n_trials):
        perm = np.random.permutation(input_dim)
        correct = 0
        total = 0
        with torch.no_grad():
            for X, Y in test_loader:
                X = X[:, perm].to(device)
                Y = Y.to(device)
                out = model(X).squeeze()
                pred = (out > 0.5).float()
                correct += (pred == Y).sum().item()
                total += Y.shape[0]
        perm_accs.append(correct / total)

    mean_perm = float(np.mean(perm_accs))
    drop = baseline_acc - mean_perm
    print(f"    Permuted accuracy: {mean_perm:.4f} (drop={drop:.4f})")

    return {
        'baseline_accuracy': baseline_acc,
        'permuted_accuracy_mean': mean_perm,
        'permuted_accuracy_std': float(np.std(perm_accs)),
        'accuracy_drop': drop,
    }


def main():
    parser = argparse.ArgumentParser(description='E03: Model Invariance')
    add_common_args(parser)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--n-trials', type=int, default=20,
                        help='Number of random permutations per seed')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = './results/e03_invariance'

    print("=" * 60)
    print("  E03: Model Invariance Test")
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

    bl_vals = [r['baseline_accuracy'] for r in all_runs]
    pm_vals = [r['permuted_accuracy_mean'] for r in all_runs]
    dr_vals = [r['accuracy_drop'] for r in all_runs]

    aggregated = {
        'baseline': {'mean': float(np.mean(bl_vals)), 'std': float(np.std(bl_vals)), 'values': bl_vals},
        'permuted': {'mean': float(np.mean(pm_vals)), 'std': float(np.std(pm_vals)), 'values': pm_vals},
        'drop': {'mean': float(np.mean(dr_vals)), 'std': float(np.std(dr_vals)), 'values': dr_vals},
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    x = ['Baseline', 'Permuted']
    means = [aggregated['baseline']['mean'], aggregated['permuted']['mean']]
    stds = [aggregated['baseline']['std'], aggregated['permuted']['std']]
    colors = ['steelblue', 'coral']

    bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors,
                  edgecolor='black', alpha=0.85, width=0.5)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.4)
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Invariance Test — {args.cipher.upper()} ({args.rounds}r)\n'
                 f'Drop = {aggregated["drop"]["mean"]:.4f} ± {aggregated["drop"]["std"]:.4f}')
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / f'e03_{args.cipher}_r{args.rounds}.png', dpi=300)
    plt.close()

    save_results(aggregated, str(output_dir),
                 f'e03_{args.cipher}_r{args.rounds}_results.json')
    print(f"\n✓ Done. Drop = {aggregated['drop']['mean']:.4f} ± {aggregated['drop']['std']:.4f}")


if __name__ == '__main__':
    main()
