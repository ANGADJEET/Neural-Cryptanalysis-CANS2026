
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
    key = cipher.random_key()
    half = n_samples // 2

    P = cipher.random_plaintexts(half)
    P_prime = (P ^ delta_p).astype(P.dtype)

    _, trace1 = cipher.encrypt_with_trace(P, n_rounds, key)
    _, trace2 = cipher.encrypt_with_trace(P_prime, n_rounds, key)

    pos_diffs = []
    for r in range(n_rounds):
        diff = trace1[r] ^ trace2[r]
        pos_diffs.append(diff)

    Q = cipher.random_plaintexts(half)
    Q_prime = cipher.random_plaintexts(half)

    _, trace_q1 = cipher.encrypt_with_trace(Q, n_rounds, key)
    _, trace_q2 = cipher.encrypt_with_trace(Q_prime, n_rounds, key)

    neg_diffs = []
    for r in range(n_rounds):
        diff = trace_q1[r] ^ trace_q2[r]
        neg_diffs.append(diff)

    round_diffs = []
    for r in range(n_rounds):
        combined = np.concatenate([pos_diffs[r], neg_diffs[r]])
        round_diffs.append(combined)

    labels = np.concatenate([np.ones(half), np.zeros(half)])

    return round_diffs, labels


def diff_to_bits(diff_array, block_size):
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
    parser.add_argument('--mine-epochs', type=int, default=200)
    parser.add_argument('--skip-calibration', action='store_true',
                        help='Skip MINE calibration check (not recommended)')
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

    # ── MINE Calibration Positive Control ─────────────────────────────
    # Before trusting MINE on cipher data (where true MI is unknown),
    # verify it works on data with known MI.
    if not args.skip_calibration:
        print(f"\n{'━' * 50}")
        print("  Calibration: MINE positive control (ρ=0.7 Gaussians)")
        print(f"{'━' * 50}")
        from models.mine import MutualInfoEstimator
        calibrator = MutualInfoEstimator(
            input_dim=1, device=device
        )
        cal_result = calibrator.validate_calibration(
            rho=0.7, n_epochs=args.mine_epochs, verbose=True
        )
        if not cal_result['calibrated']:
            print("  ⚠ WARNING: MINE failed calibration. Results may be unreliable.")
            print("  Consider increasing --mine-epochs or checking GPU availability.")
    else:
        cal_result = {'skipped': True}

    n_rounds = min(args.rounds, cipher.max_rounds)
    round_diffs, labels = compute_round_diffs(
        cipher, args.cipher, n_rounds,
        cipher.get_default_delta_p(), args.samples
    )

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

    print(f"\n{'━' * 50}")
    print("  Part 2: Conditional MI — I(ΔR_r ; Y | ΔR_{r-1})")
    print(f"{'━' * 50}")

    conditional_mi = {}
    for r in range(1, n_rounds):
        X_curr = diff_to_bits(round_diffs[r], cipher.block_size)
        X_prev = diff_to_bits(round_diffs[r - 1], cipher.block_size)
        X_joint = np.concatenate([X_curr, X_prev], axis=1)

        mi_joint = estimate_mutual_information(X_joint, labels, device=device,
                                                n_epochs=args.mine_epochs)
        mi_prev = marginal_mi[r]

        cmi = max(0, mi_joint - mi_prev)
        conditional_mi[r + 1] = float(cmi)
        print(f"  Round {r+1}: I(ΔR_r ; Y | ΔR_{{r-1}}) ≈ {cmi:.4f} nats "
              f"(joint={mi_joint:.4f}, prev={mi_prev:.4f})")

    markov_gaps = {}
    for r in range(2, n_rounds + 1):
        if r in conditional_mi and (r - 1) in marginal_mi:
            gap = conditional_mi[r]
            markov_gaps[r] = gap

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    rounds_m = sorted(marginal_mi.keys())
    axes[0].bar(rounds_m, [marginal_mi[r] for r in rounds_m],
                color='steelblue', edgecolor='black', alpha=0.8)
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('MI (nats)')
    axes[0].set_title('Marginal MI: I(ΔR_r ; Y)')
    axes[0].grid(True, alpha=0.3, axis='y')

    rounds_c = sorted(conditional_mi.keys())
    axes[1].bar(rounds_c, [conditional_mi[r] for r in rounds_c],
                color='coral', edgecolor='black', alpha=0.8)
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Conditional MI (nats)')
    axes[1].set_title('Conditional MI: I(ΔR_r ; Y | ΔR_{r-1})')
    axes[1].grid(True, alpha=0.3, axis='y')

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
        'mine_calibration': cal_result,
        'mine_epochs': args.mine_epochs,
    }
    save_results(results, str(output_dir), f'e06_{args.cipher}_results.json')

    print(f"\n✓ Results saved to {output_dir}")


if __name__ == '__main__':
    main()
