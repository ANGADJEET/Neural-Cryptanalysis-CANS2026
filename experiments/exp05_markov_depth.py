#!/usr/bin/env python
"""
E05: Memory Depth Ablation / Markov (Multi-Seed)

Train LSTM with varying sequence lengths from round traces.

Usage:
    python experiments/exp05_markov_depth.py --cipher speck32 --rounds 7 --n-seeds 3
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
from data.representations import RepresentationFactory
from models import get_model
from training.trainer import Trainer
from evaluation.metrics import evaluate_model
from experiments.experiment_utils import (
    set_seed, add_common_args, save_results, get_device
)


def generate_trace_data(cipher_name, cipher, n_rounds, delta_p, n_samples):
    """Generate data with round-by-round traces.

    Positive: (P, P ^ delta_p) encrypted under the same key — has differential structure.
    Negative: (Q, R) two INDEPENDENT random plaintexts encrypted under the same key —
              has the same temporal correlation structure as positives but NO differential signal.
    """
    key = cipher.random_key()
    half = n_samples // 2
    factory = RepresentationFactory(block_size=cipher.block_size)

    # --- Positive samples: differential pair (P, P ^ delta_p) ---
    P = cipher.random_plaintexts(half)
    P_prime = (P ^ delta_p).astype(P.dtype)
    _, trace1 = cipher.encrypt_with_trace(P, n_rounds, key)
    _, trace2 = cipher.encrypt_with_trace(P_prime, n_rounds, key)

    pos_traces = []
    for r in range(n_rounds):
        diff = factory.get_representation('R2_xor_diff', trace1[r], trace2[r])
        pos_traces.append(diff)
    pos_traces = np.stack(pos_traces, axis=1)  # (half, n_rounds, block_size)

    # --- Negative samples: independent random pair (Q, R) through the SAME cipher ---
    # This ensures temporal correlation structure matches positives,
    # but there is no differential relationship between Q and R.
    Q = cipher.random_plaintexts(half)
    R = cipher.random_plaintexts(half)
    _, trace_q = cipher.encrypt_with_trace(Q, n_rounds, key)
    _, trace_r = cipher.encrypt_with_trace(R, n_rounds, key)

    neg_traces = []
    for r in range(n_rounds):
        diff = factory.get_representation('R2_xor_diff', trace_q[r], trace_r[r])
        neg_traces.append(diff)
    neg_traces = np.stack(neg_traces, axis=1)  # (half, n_rounds, block_size)

    X = np.concatenate([pos_traces, neg_traces], axis=0)
    Y = np.concatenate([np.ones(half), np.zeros(half)])

    return X, Y


def single_run(seed, args):
    """One seed: accuracy at each depth."""
    set_seed(seed)
    cipher = get_cipher(args.cipher)
    device = get_device(args)

    X_full, Y_full = generate_trace_data(
        args.cipher, cipher, args.rounds,
        cipher.get_default_delta_p(), args.samples
    )
    n_total = X_full.shape[0]
    perm = np.random.permutation(n_total)
    X_full = X_full[perm]
    Y_full = Y_full[perm]

    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    X_train, Y_train = X_full[:n_train], Y_full[:n_train]
    X_val, Y_val = X_full[n_train:n_train + n_val], Y_full[n_train:n_train + n_val]
    X_test, Y_test = X_full[n_train + n_val:], Y_full[n_train + n_val:]

    results = {}
    for depth in args.depths:
        depth = min(depth, args.rounds)
        print(f"    Depth {depth}...", end=' ')

        # Take only last `depth` rounds
        X_tr = torch.from_numpy(X_train[:, -depth:, :]).float()
        X_vl = torch.from_numpy(X_val[:, -depth:, :]).float()
        X_ts = torch.from_numpy(X_test[:, -depth:, :]).float()
        Y_tr = torch.from_numpy(Y_train).float()
        Y_vl = torch.from_numpy(Y_val).float()
        Y_ts = torch.from_numpy(Y_test).float()

        from torch.utils.data import DataLoader, TensorDataset
        train_loader = DataLoader(TensorDataset(X_tr, Y_tr),
                                  batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_vl, Y_vl),
                                batch_size=args.batch_size)
        test_loader = DataLoader(TensorDataset(X_ts, Y_ts),
                                 batch_size=args.batch_size)

        input_dim = X_tr.shape[2]
        model = get_model('lstm', input_dim=input_dim)

        trainer = Trainer(model=model, train_loader=train_loader,
                          val_loader=val_loader, device=device, use_wandb=False)
        trainer.train(n_epochs=args.epochs, early_stopping_patience=5, save_best=False)

        metrics = evaluate_model(model, test_loader, device)
        results[str(depth)] = float(metrics['accuracy'])
        print(f"acc={metrics['accuracy']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='E05: Memory Depth')
    add_common_args(parser)
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--depths', type=int, nargs='+', default=[1, 2, 3, 5, 7])
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = './results/e05_markov_depth'

    print("=" * 60)
    print("  E05: Memory Depth Ablation (Markov)")
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

    # Aggregate
    depths = [min(d, args.rounds) for d in args.depths]
    agg = {}
    for d in depths:
        key = str(d)
        vals = [r[key] for r in all_runs if key in r]
        agg[key] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'values': [float(v) for v in vals],
        }

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x = sorted([int(k) for k in agg.keys()])
    means = [agg[str(d)]['mean'] for d in x]
    stds = [agg[str(d)]['std'] for d in x]

    ax.errorbar(x, means, yerr=stds, fmt='go-', linewidth=2,
                markersize=8, capsize=5, capthick=2)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    ax.set_xlabel('Sequence Depth (rounds)')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Memory Depth — {args.cipher.upper()} ({args.rounds}r, {args.n_seeds} seeds)')
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'e05_{args.cipher}_r{args.rounds}.png', dpi=300)
    plt.close()

    save_results({'depths': agg, '_seeds': seeds}, str(output_dir),
                 f'e05_{args.cipher}_r{args.rounds}_results.json')
    print(f"\n✓ Done")


if __name__ == '__main__':
    main()
