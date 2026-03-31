#!/usr/bin/env python
"""
E18: Gradient-Based Differential Search

Can an optimizer discover good input differences (Δp) for SPECK32
WITHOUT expert knowledge?

Method:
  - Parameterize Δp as 32 independent Bernoulli logits
  - For each sampled Δp, encrypt N pairs and measure output bias
  - Use REINFORCE (policy gradient) to push toward high-bias Δp
  - Compare discovered Δp against the known best (0x00400000)

If the optimizer rediscovers 0x00400000 or finds comparably good Δp,
this demonstrates that neural/gradient methods can automate the
traditionally expert-driven step of differential search.

Usage:
    python experiments/exp18_diff_search.py --cipher speck32 --rounds 5
    python experiments/exp18_diff_search.py --cipher speck32 --rounds 3 --steps 2000
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
#  Bias Measurement
# ═══════════════════════════════════════════════════════════════════

def measure_output_bias(cipher, delta_p_int, n_rounds, n_samples, key=None):
    """
    Measure the output bias for a given input difference.

    Encrypts N pairs (P, P⊕Δp) and checks how biased the output
    XOR difference is (how far from uniform).

    Returns a scalar bias score — higher = better differential.
    """
    if key is None:
        key = cipher.random_key()

    P = cipher.random_plaintexts(n_samples)
    P_prime = (P ^ delta_p_int).astype(P.dtype)

    C = cipher.encrypt(P, n_rounds, key)
    C_prime = cipher.encrypt(P_prime, n_rounds, key)

    output_diff = C ^ C_prime

    # Measure bias: count how often each output bit is 0
    # For a good differential, some bits will be heavily biased
    bits = np.unpackbits(
        output_diff.view(np.uint8).reshape(-1, cipher.block_size // 8),
        axis=1
    ).astype(np.float32)

    # Bias per bit: |P(bit=1) - 0.5|
    bit_means = np.mean(bits, axis=0)
    bit_bias = np.abs(bit_means - 0.5)

    # Overall bias score: sum of squared biases (like chi-squared)
    score = float(np.sum(bit_bias ** 2))

    return score


def measure_distinguishing_accuracy(cipher, delta_p_int, n_rounds, n_samples, key=None):
    """
    Quick distinguishing test: can a simple threshold separate
    differential pairs from random pairs based on output XOR?

    Returns accuracy (0.5 = no signal, 1.0 = perfect).
    """
    if key is None:
        key = cipher.random_key()
    half = n_samples // 2

    # Positive: differential pairs
    P = cipher.random_plaintexts(half)
    P_prime = (P ^ delta_p_int).astype(P.dtype)
    C_pos = cipher.encrypt(P, n_rounds, key)
    C_prime_pos = cipher.encrypt(P_prime, n_rounds, key)
    diff_pos = C_pos ^ C_prime_pos

    # Negative: random pairs
    Q = cipher.random_plaintexts(half)
    R = cipher.random_plaintexts(half)
    C_neg = cipher.encrypt(Q, n_rounds, key)
    C_prime_neg = cipher.encrypt(R, n_rounds, key)
    diff_neg = C_neg ^ C_prime_neg

    # Simple feature: popcount (number of 1-bits in XOR diff)
    pop_pos = np.array([bin(int(x)).count('1') for x in diff_pos])
    pop_neg = np.array([bin(int(x)).count('1') for x in diff_neg])

    # Find best threshold
    all_pops = np.concatenate([pop_pos, pop_neg])
    all_labels = np.concatenate([np.ones(half), np.zeros(half)])

    best_acc = 0.5
    for threshold in range(0, 33):
        preds = (all_pops <= threshold).astype(float)
        acc = float(np.mean(preds == all_labels))
        best_acc = max(best_acc, acc, 1 - acc)

    return best_acc


# ═══════════════════════════════════════════════════════════════════
#  REINFORCE Optimizer
# ═══════════════════════════════════════════════════════════════════

def bits_to_int32(bits):
    """Convert a 32-element binary array to a uint32 integer."""
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return np.uint32(val)


def int32_to_bits(val):
    """Convert uint32 to 32-element binary array."""
    return np.array([(val >> (31 - i)) & 1 for i in range(32)], dtype=np.float32)


def reinforce_search(cipher, n_rounds, n_steps, n_eval_samples,
                     lr=0.05, baseline_momentum=0.9, entropy_weight=0.05, seed=42):
    """
    Use REINFORCE to search for good input differences.

    Parameterize Δp as 32 independent Bernoulli logits.
    Sample, evaluate bias, update logits via policy gradient.
    """
    set_seed(seed)

    # Learnable logits for each of the 32 bits of Δp
    logits = torch.zeros(32, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=lr)

    baseline = 0.0
    history = []
    best_score = -1
    best_delta = 0
    best_bits = None

    key = cipher.random_key()  # fix key for consistency

    for step in range(n_steps):
        # Sample Δp from current distribution
        probs = torch.sigmoid(logits)
        dist = torch.distributions.Bernoulli(probs)
        sample = dist.sample()

        # Ensure Δp ≠ 0 (zero difference is trivial)
        if sample.sum() == 0:
            idx = torch.randint(0, 32, (1,))
            sample[idx] = 1.0

        delta_p_int = bits_to_int32(sample.detach().numpy())

        # Evaluate
        score = measure_output_bias(
            cipher, delta_p_int, n_rounds, n_eval_samples, key
        )

        # Update baseline
        baseline = baseline_momentum * baseline + (1 - baseline_momentum) * score

        # REINFORCE gradient with entropy regularization
        advantage = score - baseline
        log_prob = dist.log_prob(sample).sum()
        entropy = dist.entropy().mean()  # Encourage exploration
        
        loss = -(advantage * log_prob) - (entropy_weight * entropy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track best
        if score > best_score:
            best_score = score
            best_delta = int(delta_p_int)
            best_bits = sample.detach().numpy().copy()

        if (step + 1) % 50 == 0 or step == 0:
            hw = int(sample.sum().item())
            print(f"      step {step+1}/{n_steps} | "
                  f"score={score:.6f} | best={best_score:.6f} | "
                  f"Δp=0x{int(delta_p_int):08x} (hw={hw}) | "
                  f"best_Δp=0x{best_delta:08x}",
                  flush=True)

        history.append({
            'step': step,
            'score': float(score),
            'delta_p': f"0x{int(delta_p_int):08x}",
            'hamming_weight': int(sample.sum().item()),
        })

    return {
        'best_delta_p': f"0x{best_delta:08x}",
        'best_delta_p_int': int(best_delta),
        'best_score': float(best_score),
        'best_bits': best_bits.tolist() if best_bits is not None else None,
        'final_probs': torch.sigmoid(logits).detach().numpy().tolist(),
        'history': history,
    }


# ═══════════════════════════════════════════════════════════════════
#  Random Baseline
# ═══════════════════════════════════════════════════════════════════

def random_search(cipher, n_rounds, n_candidates, n_eval_samples, seed=42):
    """Random baseline: try n_candidates random Δp values."""
    set_seed(seed)
    key = cipher.random_key()

    best_score = -1
    best_delta = 0

    for i in range(n_candidates):
        # Random non-zero 32-bit difference
        delta_p = np.random.randint(1, 2**32, dtype=np.uint32)
        score = measure_output_bias(cipher, delta_p, n_rounds, n_eval_samples, key)

        if score > best_score:
            best_score = score
            best_delta = int(delta_p)

    return {
        'best_delta_p': f"0x{best_delta:08x}",
        'best_score': float(best_score),
    }


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def single_run(seed, args):
    cipher = get_cipher(args.cipher)
    known_delta = int(cipher.get_default_delta_p())

    print(f"    Known best Δp: 0x{known_delta:08x}")

    # Measure known Δp score for reference
    key = cipher.random_key()
    known_score = measure_output_bias(
        cipher, known_delta, args.rounds, args.eval_samples, key
    )
    known_acc = measure_distinguishing_accuracy(
        cipher, known_delta, args.rounds, 10000, key
    )
    print(f"    Known Δp bias score: {known_score:.6f}")
    print(f"    Known Δp distinguishing acc: {known_acc:.4f}")

    # REINFORCE search
    print(f"\n    REINFORCE search ({args.steps} steps):")
    rl_result = reinforce_search(
        cipher, args.rounds, args.steps, args.eval_samples,
        lr=args.lr, seed=seed
    )

    # Evaluate discovered Δp
    disc_delta = rl_result['best_delta_p_int']
    disc_acc = measure_distinguishing_accuracy(
        cipher, disc_delta, args.rounds, 10000, key
    )

    # Random baseline (same number of evaluations)
    print(f"\n    Random baseline ({args.steps} candidates):")
    rand_result = random_search(
        cipher, args.rounds, args.steps, args.eval_samples, seed=seed
    )
    rand_delta = int(rand_result['best_delta_p'].replace('0x', ''), 16)
    rand_acc = measure_distinguishing_accuracy(
        cipher, rand_delta, args.rounds, 10000, key
    )

    # Summary
    print(f"\n    ╔{'═'*50}╗")
    print(f"    ║  {'Method':<18} {'Δp':<14} {'Bias':<10} {'Acc':<8} ║")
    print(f"    ╠{'═'*50}╣")
    print(f"    ║  {'Known (expert)':<18} 0x{known_delta:08x}   "
          f"{known_score:<10.6f} {known_acc:<8.4f} ║")
    print(f"    ║  {'REINFORCE':<18} 0x{disc_delta:08x}   "
          f"{rl_result['best_score']:<10.6f} {disc_acc:<8.4f} ║")
    print(f"    ║  {'Random':<18} 0x{rand_delta:08x}   "
          f"{rand_result['best_score']:<10.6f} {rand_acc:<8.4f} ║")
    print(f"    ╚{'═'*50}╝")

    rediscovered = (disc_delta == known_delta)
    if rediscovered:
        print(f"    ⚡ REINFORCE rediscovered the known best differential!")
    elif disc_acc >= known_acc * 0.95:
        print(f"    ✓ REINFORCE found a comparably good differential.")
    else:
        print(f"    ✗ REINFORCE did not match expert differential.")

    return {
        'known_delta': f"0x{known_delta:08x}",
        'known_score': float(known_score),
        'known_acc': float(known_acc),
        'rl_delta': rl_result['best_delta_p'],
        'rl_score': float(rl_result['best_score']),
        'rl_acc': float(disc_acc),
        'rl_rediscovered': int(rediscovered),
        'random_delta': rand_result['best_delta_p'],
        'random_score': float(rand_result['best_score']),
        'random_acc': float(rand_acc),
        'rl_final_probs': rl_result['final_probs'],
    }


def plot_results(all_results, args, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Score comparison across seeds
    methods = ['Known', 'REINFORCE', 'Random']
    known_scores = [r['known_score'] for r in all_results]
    rl_scores = [r['rl_score'] for r in all_results]
    rand_scores = [r['random_score'] for r in all_results]

    x = np.arange(len(methods))
    means = [np.mean(known_scores), np.mean(rl_scores), np.mean(rand_scores)]
    stds = [np.std(known_scores), np.std(rl_scores), np.std(rand_scores)]
    colors = ['#4CAF50', '#2196F3', '#FF9800']

    axes[0].bar(x, means, yerr=stds, capsize=5, color=colors,
                edgecolor='black', alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylabel('Output Bias Score')
    axes[0].set_title(f'Differential Search — {args.cipher.upper()} ({args.rounds}r)')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Plot 2: Final learned bit probabilities (from last seed)
    probs = all_results[-1].get('rl_final_probs', [])
    if probs:
        axes[1].bar(range(len(probs)), probs, color='#9C27B0', alpha=0.7)
        axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Bit Position')
        axes[1].set_ylabel('P(bit = 1)')
        axes[1].set_title('Learned Δp Bit Probabilities')
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)

        # Mark the known Δp bits
        known_delta = int(all_results[-1]['known_delta'].replace('0x', ''), 16)
        known_bits = int32_to_bits(known_delta)
        for i, b in enumerate(known_bits):
            if b > 0:
                axes[1].axvline(x=i, color='red', alpha=0.4, linestyle='-',
                               label='Known Δp bit' if i == 0 else None)

    plt.tight_layout()
    plt.savefig(output_dir / f'e18_{args.cipher}_diff_search.png', dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='E18: Gradient-Based Differential Search'
    )
    add_common_args(parser)
    parser.add_argument('--rounds', type=int, default=5,
                        help='Cipher rounds for evaluation')
    parser.add_argument('--steps', type=int, default=1000,
                        help='REINFORCE optimization steps')
    parser.add_argument('--eval-samples', type=int, default=5000,
                        help='Samples per bias evaluation')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate for logits')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = './results/e18_diff_search'

    print("=" * 60)
    print("  E18: Gradient-Based Differential Search")
    print("  REINFORCE vs Random vs Known Expert")
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
    rl_accs = [r['rl_acc'] for r in all_results]
    known_accs = [r['known_acc'] for r in all_results]
    rediscovered = sum(r['rl_rediscovered'] for r in all_results)

    print(f"\n{'═' * 55}")
    print(f"  REINFORCE avg acc: {np.mean(rl_accs):.4f} ± {np.std(rl_accs):.4f}")
    print(f"  Known Δp avg acc:  {np.mean(known_accs):.4f} ± {np.std(known_accs):.4f}")
    print(f"  Exact rediscovery: {rediscovered}/{len(seeds)} seeds")
    print(f"{'═' * 55}")

    save_data = {
        'rl_acc_mean': float(np.mean(rl_accs)),
        'known_acc_mean': float(np.mean(known_accs)),
        'rediscovery_rate': rediscovered / len(seeds),
        'runs': [{k: v for k, v in r.items() if k != 'rl_final_probs'}
                 for r in all_results],
        '_seeds': seeds,
    }
    save_results(save_data, str(output_dir), f'e18_{args.cipher}_results.json')
    plot_results(all_results, args, output_dir)
    print(f"\n✓ Results saved to {output_dir}")


if __name__ == '__main__':
    main()
