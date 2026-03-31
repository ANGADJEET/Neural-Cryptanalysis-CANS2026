
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


def eval_with_noise(model, test_loader, device, noise_std):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, Y in test_loader:
            X = X.to(device) + torch.randn_like(X.to(device)) * noise_std
            Y = Y.to(device)
            out = model(X).squeeze()
            pred = (out > 0.5).float()
            correct += (pred == Y).sum().item()
            total += Y.shape[0]
    return correct / total


def eval_with_bitflip(model, test_loader, device, flip_prob):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, Y in test_loader:
            X = X.to(device)
            mask = (torch.rand_like(X) < flip_prob).float()
            X = torch.abs(X - mask)
            Y = Y.to(device)
            out = model(X).squeeze()
            pred = (out > 0.5).float()
            correct += (pred == Y).sum().item()
            total += Y.shape[0]
    return correct / total


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

    baseline = evaluate_model(model, test_loader, device)
    baseline_acc = float(baseline['accuracy'])

    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    noise_results = {}
    for sigma in noise_levels:
        acc = eval_with_noise(model, test_loader, device, sigma)
        noise_results[str(sigma)] = acc

    flip_probs = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
    flip_results = {}
    for p in flip_probs:
        acc = eval_with_bitflip(model, test_loader, device, p)
        flip_results[str(p)] = acc

    mismatch_gen = CipherDataGenerator(
        cipher=args.cipher, n_rounds=args.rounds,
        delta_p=cipher.get_default_delta_p()
    )
    mismatch_data = mismatch_gen.generate_balanced_dataset(args.samples // 10)
    mismatch_ds = CryptoDataset(mismatch_data, 'R2_xor_diff', cipher.block_size)
    mismatch_loader = DataLoader(mismatch_ds, batch_size=args.batch_size)
    mismatch_metrics = evaluate_model(model, mismatch_loader, device)

    return {
        'baseline_accuracy': baseline_acc,
        'noise': noise_results,
        'bitflip': flip_results,
        'key_mismatch_accuracy': float(mismatch_metrics['accuracy']),
    }


def main():
    parser = argparse.ArgumentParser(description='E04: Robustness Testing')
    add_common_args(parser)
    parser.add_argument('--rounds', type=int, default=5)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = './results/e04_robustness'

    print("=" * 60)
    print("  E04: Robustness Testing")
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

    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    flip_probs = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]

    noise_means = [float(np.mean([r['noise'][str(s)] for r in all_runs])) for s in noise_levels]
    noise_stds = [float(np.std([r['noise'][str(s)] for r in all_runs])) for s in noise_levels]
    flip_means = [float(np.mean([r['bitflip'][str(p)] for r in all_runs])) for p in flip_probs]
    flip_stds = [float(np.std([r['bitflip'][str(p)] for r in all_runs])) for p in flip_probs]

    bl_mean = float(np.mean([r['baseline_accuracy'] for r in all_runs]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].errorbar(noise_levels, noise_means, yerr=noise_stds,
                     fmt='bo-', linewidth=2, markersize=6, capsize=4)
    axes[0].axhline(y=bl_mean, color='g', linestyle='--', alpha=0.5, label=f'Baseline={bl_mean:.3f}')
    axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    axes[0].set_xlabel('Noise σ')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Gaussian Noise Robustness')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(flip_probs, flip_means, yerr=flip_stds,
                     fmt='r^-', linewidth=2, markersize=6, capsize=4)
    axes[1].axhline(y=bl_mean, color='g', linestyle='--', alpha=0.5, label=f'Baseline={bl_mean:.3f}')
    axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('Flip Probability')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Bit Flip Robustness')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'{args.cipher.upper()} ({args.rounds}r) — Robustness ({args.n_seeds} seeds)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'e04_{args.cipher}_r{args.rounds}.png', dpi=300)
    plt.close()

    results = {
        'noise': {str(s): {'mean': m, 'std': st} for s, m, st in zip(noise_levels, noise_means, noise_stds)},
        'bitflip': {str(p): {'mean': m, 'std': st} for p, m, st in zip(flip_probs, flip_means, flip_stds)},
        'baseline': {'mean': bl_mean},
        '_seeds': seeds,
    }
    save_results(results, str(output_dir),
                 f'e04_{args.cipher}_r{args.rounds}_results.json')
    print(f"\n✓ Done")


if __name__ == '__main__':
    main()
