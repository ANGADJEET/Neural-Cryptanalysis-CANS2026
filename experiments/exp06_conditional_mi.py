#!/usr/bin/env python
"""
E06: Conditional MI — Markov Property Test (Strengthened)

Two analyses:
1. Marginal MI: I(ΔC_r ; Y) per round — how much total signal at each round
2. Conditional MI: I(ΔR_r ; Y | ΔR_{r-1}) — information gain from each round,
   properly testing the Markov assumption

If conditional MI ≈ 0 for all rounds > some threshold, the cipher's differential
distribution satisfies the Markov property for the neural distinguisher.

Usage:
    python experiments/exp06_conditional_mi.py --cipher speck32
    python experiments/exp06_conditional_mi.py --cipher speck32 --n-seeds 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
import matplotlib.pyplot as plt

from ciphers import get_cipher
from data.representations import RepresentationFactory
from evaluation.metrics import estimate_mutual_information
from experiments.experiment_utils import (
    set_seed, add_common_args, save_results, get_device
)


def compute_round_diffs(cipher, cipher_name, n_rounds, delta_p, n_samples):
    """
    Generate round-by-round XOR differences using encrypt_with_trace.

    Returns:
        List of arrays: round_diffs[r] = ΔState after round r
        Labels: 1 for real cipher pairs, 0 for (one random) pair
    """
    key = cipher.random_key()
    half = n_samples // 2

    # Positive class: real differential pairs
    P = cipher.random_plaintexts(half)
    P_prime = (P ^ delta_p).astype(P.dtype)

    _, trace1 = cipher.encrypt_with_trace(P, n_rounds, key)
    _, trace2 = cipher.encrypt_with_trace(P_prime, n_rounds, key)

    # XOR diff at each round
    pos_diffs = []
    for r in range(n_rounds):
        diff = trace1[r] ^ trace2[r]
        pos_diffs.append(diff)

    # Negative class: random pair (no differential relation)
    Q = cipher.random_plaintexts(half)
    Q_prime = cipher.random_plaintexts(half)  # independent random, NOT Q^delta_p

    _, trace_q1 = cipher.encrypt_with_trace(Q, n_rounds, key)
    _, trace_q2 = cipher.encrypt_with_trace(Q_prime, n_rounds, key)

    neg_diffs = []
    for r in range(n_rounds):
        diff = trace_q1[r] ^ trace_q2[r]
        neg_diffs.append(diff)

    # Combine and create labels
    round_diffs = []
    for r in range(n_rounds):
        combined = np.concatenate([pos_diffs[r], neg_diffs[r]])
        round_diffs.append(combined)

    labels = np.concatenate([np.ones(half), np.zeros(half)])

    return round_diffs, labels


def diff_to_bits(diff_array, block_size):
    """Convert integer differences to binary feature vectors."""
    n = len(diff_array)
    bits = np.zeros((n, block_size), dtype=np.float32)
    for b in range(block_size):
        bits[:, b] = (diff_array >> b) & 1
    return bits


def main():
    parser = argparse.ArgumentParser(description='E06: Conditional MI (Strengthened)')
    add_common_args(parser)
    parser.add_argument('--rounds', type=int, default=8,
                        help='Max rounds to analyze')
    parser.add_argument('--mine-epochs', type=int, default=50)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = './results/e06_conditional_mi'

    print("=" * 60)
    print("  E06: Conditional MI — Markov Property Test")
    print("=" * 60)

    cipher = get_cipher(args.cipher)
    device = get_device(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    n_rounds = min(args.rounds, cipher.max_rounds)
    round_diffs, labels = compute_round_diffs(
        cipher, args.cipher, n_rounds,
        cipher.get_default_delta_p(), args.samples
    )

    # ── Part 1: Marginal MI per round ────────────────────────
    print(f"\n{'━' * 50}")
    print("  Part 1: Marginal MI — I(ΔR_r ; Y)")
    print(f"{'━' * 50}")

    marginal_mi = {}
    for r in range(n_rounds):
        X = diff_to_bits(round_diffs[r], cipher.block_size)
        mi = estimate_mutual_information(X, labels, device=device,
                                          n_epochs=args.mine_epochs)
        marginal_mi[r + 1] = float(mi)
        print(f"  Round {r+1}: I(ΔR ; Y) = {mi:.4f} nats")

    # ── Part 2: Conditional MI ──────────────────────────────
    print(f"\n{'━' * 50}")
    print("  Part 2: Conditional MI — I(ΔR_r ; Y | ΔR_{r-1})")
    print(f"{'━' * 50}")

    conditional_mi = {}
    for r in range(1, n_rounds):
        # Features: current round diff CONDITIONED on previous round diff
        # Approximation: concatenate (ΔR_r, ΔR_{r-1}) and subtract MI of ΔR_{r-1} alone
        X_curr = diff_to_bits(round_diffs[r], cipher.block_size)
        X_prev = diff_to_bits(round_diffs[r - 1], cipher.block_size)
        X_joint = np.concatenate([X_curr, X_prev], axis=1)

        mi_joint = estimate_mutual_information(X_joint, labels, device=device,
                                                n_epochs=args.mine_epochs)
        mi_prev = marginal_mi[r]  # I(ΔR_{r-1} ; Y)

        # I(ΔR_r ; Y | ΔR_{r-1}) ≈ I(ΔR_r, ΔR_{r-1} ; Y) - I(ΔR_{r-1} ; Y)
        cmi = max(0, mi_joint - mi_prev)
        conditional_mi[r + 1] = float(cmi)
        print(f"  Round {r+1}: I(ΔR_r ; Y | ΔR_{{r-1}}) ≈ {cmi:.4f} nats "
              f"(joint={mi_joint:.4f}, prev={mi_prev:.4f})")

    # ── Part 3: Markov gap ───────────────────────────────────
    # If I_cond ≈ 0, the round added no new distinguishing info beyond
    # what was already in the previous round → Markov property holds
    markov_gaps = {}
    for r in range(2, n_rounds + 1):
        if r in conditional_mi and (r - 1) in marginal_mi:
            gap = conditional_mi[r]  # small gap = Markov
            markov_gaps[r] = gap

    # ── Plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Marginal MI
    rounds_m = sorted(marginal_mi.keys())
    axes[0].bar(rounds_m, [marginal_mi[r] for r in rounds_m],
                color='steelblue', edgecolor='black', alpha=0.8)
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('MI (nats)')
    axes[0].set_title('Marginal MI: I(ΔR_r ; Y)')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Conditional MI
    rounds_c = sorted(conditional_mi.keys())
    axes[1].bar(rounds_c, [conditional_mi[r] for r in rounds_c],
                color='coral', edgecolor='black', alpha=0.8)
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Conditional MI (nats)')
    axes[1].set_title('Conditional MI: I(ΔR_r ; Y | ΔR_{r-1})')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Markov gap
    rounds_g = sorted(markov_gaps.keys())
    colors = ['green' if markov_gaps[r] < 0.01 else 'orange' if markov_gaps[r] < 0.05 else 'red'
              for r in rounds_g]
    axes[2].bar(rounds_g, [markov_gaps[r] for r in rounds_g],
                color=colors, edgecolor='black', alpha=0.8)
    axes[2].axhline(y=0.01, color='green', linestyle='--', alpha=0.5, label='Markov threshold')
    axes[2].set_xlabel('Round')
    axes[2].set_ylabel('Markov Gap')
    axes[2].set_title('Markov Gap (low = memoryless)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'{args.cipher.upper()} — MI Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f'e06_{args.cipher}.png', dpi=300, bbox_inches='tight')
    plt.close()

    results = {
        'marginal_mi': {str(k): v for k, v in marginal_mi.items()},
        'conditional_mi': {str(k): v for k, v in conditional_mi.items()},
        'markov_gaps': {str(k): v for k, v in markov_gaps.items()},
    }
    save_results(results, str(output_dir), f'e06_{args.cipher}_results.json')

    print(f"\n✓ Results saved to {output_dir}")


if __name__ == '__main__':
    main()
