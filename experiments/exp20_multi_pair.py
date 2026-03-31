#!/usr/bin/env python
"""
E20: Multi-Ciphertext Pair Distinguisher

Instead of feeding 1 ciphertext pair to the distinguisher, feed N pairs
encrypted under the same key. Plot accuracy vs N.

This tests whether statistical aggregation over multiple pairs can
push the distinguisher boundary to higher rounds — directly comparable
to Papers 3 & 5 from the literature (2025).

Key insight: with N pairs, the model has N independent samples of the
output distribution, enabling it to detect weaker statistical biases.

Usage:
    python experiments/exp20_multi_pair.py --cipher speck32 --n-seeds 3
    python experiments/exp20_multi_pair.py --cipher speck32 --pair-counts 1 2 4 8 16
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ciphers import get_cipher
from data.representations import RepresentationFactory
from experiments.experiment_utils import (
    set_seed, add_common_args, save_results, get_device
)


# ═══════════════════════════════════════════════════════════════════
#  Multi-Pair Distinguisher Model
# ═══════════════════════════════════════════════════════════════════

class MultiPairDistinguisher(nn.Module):
    """
    Processes N ciphertext pair differences independently through a
    shared encoder, then aggregates via attention pooling before
    classification.

    Input shape: (batch, N_pairs, bit_dim)
    Output: (batch,) — probability of being a differential pair
    """

    def __init__(self, bit_dim, hidden=256):
        super().__init__()

        # Shared per-pair encoder
        self.pair_encoder = nn.Sequential(
            nn.Linear(bit_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )

        # Attention pooling over pairs
        self.attn_score = nn.Linear(hidden, 1)

        # Classifier on pooled representation
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch, n_pairs, bit_dim)
        batch_size, n_pairs, bit_dim = x.shape

        # Encode each pair independently
        x_flat = x.view(batch_size * n_pairs, bit_dim)
        encoded = self.pair_encoder(x_flat)  # (B*N, hidden)
        encoded = encoded.view(batch_size, n_pairs, -1)  # (B, N, hidden)

        # Attention pooling
        attn_logits = self.attn_score(encoded).squeeze(-1)  # (B, N)
        attn_weights = torch.softmax(attn_logits, dim=1)  # (B, N)
        pooled = (encoded * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, hidden)

        return self.classifier(pooled).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════
#  Data Generation
# ═══════════════════════════════════════════════════════════════════

def generate_multi_pair_data(cipher_name, cipher, n_rounds, n_samples,
                             n_pairs, seed=42):
    """
    Generate multi-pair dataset.

    For each sample:
      - Positive: generate n_pairs differential pairs (same key, same Δp)
      - Negative: generate n_pairs random pairs (same key, random P)

    Returns X: (n_samples, n_pairs, bit_dim), Y: (n_samples,)
    """
    np.random.seed(seed)
    key = cipher.random_key()
    delta_p = cipher.get_default_delta_p()
    factory = RepresentationFactory(block_size=cipher.block_size)

    half = n_samples // 2
    X_all = []
    Y_all = []

    # Positive samples (differential pairs)
    for _ in range(half):
        pairs = []
        for _ in range(n_pairs):
            P = cipher.random_plaintexts(1)
            P_prime = (P ^ delta_p).astype(P.dtype)
            C = cipher.encrypt(P, n_rounds, key)
            C_prime = cipher.encrypt(P_prime, n_rounds, key)
            diff = factory.get_representation('R2_xor_diff', C, C_prime)
            pairs.append(diff[0])
        X_all.append(np.stack(pairs))
        Y_all.append(1.0)

    # Negative samples (random pairs)
    for _ in range(half):
        pairs = []
        for _ in range(n_pairs):
            P = cipher.random_plaintexts(1)
            R = cipher.random_plaintexts(1)
            C = cipher.encrypt(P, n_rounds, key)
            C_prime = cipher.encrypt(R, n_rounds, key)
            diff = factory.get_representation('R2_xor_diff', C, C_prime)
            pairs.append(diff[0])
        X_all.append(np.stack(pairs))
        Y_all.append(0.0)

    X = np.stack(X_all)  # (n_samples, n_pairs, bit_dim)
    Y = np.array(Y_all)

    # Shuffle
    perm = np.random.permutation(len(Y))
    return X[perm], Y[perm]


# ═══════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════

def train_multi_pair(model, X, Y, n_epochs, batch_size, device, lr=1e-3):
    """Train multi-pair distinguisher."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    n_train = int(0.8 * len(Y))
    X_train, X_val = X[:n_train], X[n_train:]
    Y_train, Y_val = Y[:n_train], Y[n_train:]

    X_train_t = torch.from_numpy(X_train).float()
    Y_train_t = torch.from_numpy(Y_train).float()
    dataset = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_acc = 0.5
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            X_val_t = torch.from_numpy(X_val).float().to(device)
            preds = model(X_val_t).cpu().numpy()
            acc = float(np.mean((preds > 0.5) == Y_val))

        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 5:
            break

    return best_acc


# ═══════════════════════════════════════════════════════════════════
#  Experiment
# ═══════════════════════════════════════════════════════════════════

def single_run(seed, args):
    set_seed(seed)
    cipher = get_cipher(args.cipher)
    device = get_device(args)
    bit_dim = cipher.block_size

    results = {}

    for n_rounds in args.rounds:
        results[str(n_rounds)] = {}

        for n_pairs in args.pair_counts:
            print(f"      {n_rounds}r, {n_pairs} pairs: ", end='', flush=True)

            X, Y = generate_multi_pair_data(
                args.cipher, cipher, n_rounds, args.samples,
                n_pairs, seed=seed + n_rounds * 100 + n_pairs
            )

            model = MultiPairDistinguisher(bit_dim, hidden=256)
            acc = train_multi_pair(
                model, X, Y,
                n_epochs=args.epochs, batch_size=args.batch_size, device=device
            )

            results[str(n_rounds)][str(n_pairs)] = float(acc)
            print(f"acc={acc:.4f}", flush=True)

    return results


def plot_results(all_results, args, output_dir):
    rounds = sorted([int(k) for k in all_results[0].keys()])
    pair_counts = sorted([int(k) for k in all_results[0][str(rounds[0])].keys()])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Accuracy vs pair count for each round
    colors = plt.cm.viridis(np.linspace(0, 1, len(rounds)))
    for i, r in enumerate(rounds):
        means = []
        stds = []
        for pc in pair_counts:
            vals = [res[str(r)][str(pc)] for res in all_results]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        axes[0].errorbar(pair_counts, means, yerr=stds,
                         fmt='o-', color=colors[i], linewidth=2,
                         markersize=6, capsize=3, label=f'{r}r')

    axes[0].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    axes[0].set_xlabel('Number of Ciphertext Pairs (N)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'Multi-Pair Distinguisher — {args.cipher.upper()}')
    axes[0].legend(title='Rounds')
    axes[0].set_xscale('log', base=2)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Accuracy gain from multi-pair (relative to N=1)
    for i, r in enumerate(rounds):
        baseline = np.mean([res[str(r)][str(pair_counts[0])] for res in all_results])
        gains = []
        for pc in pair_counts:
            val = np.mean([res[str(r)][str(pc)] for res in all_results])
            gains.append(val - baseline)
        axes[1].plot(pair_counts, gains, 'o-', color=colors[i],
                     linewidth=2, markersize=6, label=f'{r}r')

    axes[1].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Number of Pairs (N)')
    axes[1].set_ylabel('Accuracy Gain over N=1')
    axes[1].set_title('Marginal Benefit of Additional Pairs')
    axes[1].legend(title='Rounds')
    axes[1].set_xscale('log', base=2)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'E20: Multi-Ciphertext Pair Analysis — {args.cipher.upper()}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'e20_{args.cipher}_multi_pair.png', dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='E20: Multi-Ciphertext Pair Distinguisher'
    )
    add_common_args(parser)
    parser.add_argument('--rounds', type=int, nargs='+', default=[5, 6, 7, 8])
    parser.add_argument('--pair-counts', type=int, nargs='+',
                        default=[1, 2, 4, 8, 16])
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = './results/e20_multi_pair'

    print("=" * 60)
    print("  E20: Multi-Ciphertext Pair Distinguisher")
    print("  How does accuracy scale with N pairs?")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed + i for i in range(args.n_seeds)]
    all_results = []

    for i, seed in enumerate(seeds):
        print(f"\n┌─ Seed {seed} ({i+1}/{len(seeds)}) ─────────────────┐")
        result = single_run(seed, args)
        all_results.append(result)
        print(f"└─ Done ─────────────────────────────────────┘")

    # Summary
    rounds = sorted([int(k) for k in all_results[0].keys()])
    pair_counts = sorted([int(k) for k in all_results[0][str(rounds[0])].keys()])

    print(f"\n{'═' * 60}")
    header = f"  {'Round':<8}" + "".join(f"{'N='+str(pc):<10}" for pc in pair_counts)
    print(header)
    print(f"{'─' * 60}")
    for r in rounds:
        row = f"  {r:<8}"
        for pc in pair_counts:
            mean = np.mean([res[str(r)][str(pc)] for res in all_results])
            row += f"{mean:<10.4f}"
        print(row)
    print(f"{'═' * 60}")

    save_results(
        {'runs': all_results, '_seeds': seeds},
        str(output_dir), f'e20_{args.cipher}_results.json'
    )
    plot_results(all_results, args, output_dir)
    print(f"\n✓ Results saved to {output_dir}")


if __name__ == '__main__':
    main()
