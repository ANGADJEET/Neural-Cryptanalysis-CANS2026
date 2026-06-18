"""
Experiment 31: Analytical Validation of Condition C3 (Bias Sign Heatmap)
========================================================================
Computes the per-bit differential bias β_j^(r) = E[(-1)^{ΔC_j} | real] - E[(-1)^{ΔC_j} | random]
for every bit j at each round r, using purely analytical encryption (no ML).

Generates:
  1. Sign heatmaps for PRESENT, SIMON, SPECK — the "smoking gun" for (C3).
  2. Magnitude decay curves showing how |β_j| shrinks with rounds.
  3. Saves raw bias arrays for use by exp32 and exp33.

Expected results:
  - PRESENT: uniform sign across rounds (C3 satisfied → positive transfer).
  - SIMON:   checkerboard sign pattern (C3 violated → anti-transfer).
  - SPECK:   mixed/unstable signs (non-DDT features, C2 violated).

Runtime: ~2 minutes on CPU with 10M samples.
"""

import argparse
import json
import time
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ciphers.speck32 import Speck32
from ciphers.simon32 import Simon32
from ciphers.present import Present


def compute_per_bit_bias(cipher, n_rounds, n_samples, delta_p, n_keys=10):
    """
    Compute per-bit differential bias β_j^(r) for a given cipher and round count.
    
    β_j = E[(-1)^{ΔC_j} | real differential] - 0
    
    For a real differential pair (P, P⊕ΔP) encrypted under the same key,
    ΔC = C ⊕ C'. The bias at bit j is:
        β_j = 2 * Pr[ΔC_j = 0] - 1
    
    We average over multiple keys to get the expected bias under key randomness.
    """
    block_size = cipher.block_size
    bias_accum = np.zeros(block_size, dtype=np.float64)
    
    for _ in range(n_keys):
        key = cipher.random_key()
        P = cipher.random_plaintexts(n_samples)
        
        if block_size == 64:
            P_prime = P ^ np.uint64(delta_p)
        else:
            P_prime = P ^ np.uint32(delta_p)
        
        C = cipher.encrypt(P, n_rounds, key)
        C_prime = cipher.encrypt(P_prime, n_rounds, key)
        
        delta_C = C ^ C_prime
        
        # Vectorized per-bit bias: unpack all bits at once
        # For each bit j: β_j = 2 * Pr[bit_j(ΔC) == 0] - 1 = 1 - 2*mean(bit_j)
        if block_size == 64:
            # Unpack 64 bits: shift by each position and mask
            shifts = np.arange(block_size, dtype=np.uint64)
            # bits shape: (block_size, n_samples)
            bits = ((delta_C[np.newaxis, :] >> shifts[:, np.newaxis]) & np.uint64(1)).astype(np.float64)
        else:
            shifts = np.arange(block_size, dtype=np.uint32)
            bits = ((delta_C[np.newaxis, :] >> shifts[:, np.newaxis]) & np.uint32(1)).astype(np.float64)
        
        # bias_j = 1 - 2 * mean(bit_j)
        bias_accum += 1.0 - 2.0 * np.mean(bits, axis=1)
    
    return bias_accum / n_keys


def run_experiment(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = args.samples
    n_keys = args.n_keys
    
    cipher_configs = {
        'present': {
            'cipher': Present(),
            'rounds': list(range(2, 8)),
            'label': 'PRESENT (SPN)',
        },
        'simon32': {
            'cipher': Simon32(),
            'rounds': list(range(4, 12)),
            'label': 'SIMON32/64 (Feistel)',
        },
        'speck32': {
            'cipher': Speck32(),
            'rounds': list(range(3, 9)),
            'label': 'SPECK32/64 (ARX)',
        },
    }
    
    all_biases = {}
    
    for cipher_name, cfg in cipher_configs.items():
        cipher = cfg['cipher']
        delta_p = cipher.get_default_delta_p()
        rounds = cfg['rounds']
        block_size = cipher.block_size
        
        print(f"\n{'='*60}")
        print(f"  Computing biases for {cfg['label']}")
        print(f"  ΔP = 0x{delta_p:0{block_size//4}x}, {n_samples} samples × {n_keys} keys")
        print(f"{'='*60}")
        
        bias_matrix = np.zeros((len(rounds), block_size))
        
        for i, r in enumerate(rounds):
            t0 = time.time()
            bias_matrix[i] = compute_per_bit_bias(cipher, r, n_samples, delta_p, n_keys)
            elapsed = time.time() - t0
            
            n_positive = np.sum(bias_matrix[i] > 0)
            n_negative = np.sum(bias_matrix[i] < 0)
            max_abs = np.max(np.abs(bias_matrix[i]))
            print(f"  Round {r:2d}: +{n_positive} / -{n_negative} signs, "
                  f"|β|_max = {max_abs:.6f}, {elapsed:.1f}s")
        
        all_biases[cipher_name] = {
            'rounds': rounds,
            'block_size': block_size,
            'bias_matrix': bias_matrix.tolist(),
            'sign_matrix': np.sign(bias_matrix).astype(int).tolist(),
            'delta_p': delta_p,
        }
    
    # ── Save raw data ──
    with open(output_dir / 'e31_bias_data.json', 'w') as f:
        json.dump(all_biases, f, indent=2)
    print(f"\n  Raw data saved to {output_dir / 'e31_bias_data.json'}")
    
    # ── Generate sign heatmaps ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    cmap = mcolors.ListedColormap(['#e74c3c', '#ecf0f1', '#2ecc71'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    for ax, (cipher_name, data) in zip(axes, all_biases.items()):
        sign_matrix = np.array(data['sign_matrix'])
        rounds = data['rounds']
        block_size = data['block_size']
        
        im = ax.imshow(sign_matrix, cmap=cmap, norm=norm, aspect='auto',
                       interpolation='nearest')
        
        ax.set_xlabel('Bit position $j$', fontsize=11)
        ax.set_ylabel('Round $r$', fontsize=11)
        ax.set_title(cipher_configs[cipher_name]['label'], fontsize=13, fontweight='bold')
        ax.set_yticks(range(len(rounds)))
        ax.set_yticklabels(rounds)
        
        if block_size <= 32:
            ax.set_xticks(range(0, block_size, 4))
        else:
            ax.set_xticks(range(0, block_size, 8))
        
        # Count sign flips across rounds
        n_flips = 0
        for col in range(sign_matrix.shape[1]):
            for row in range(1, sign_matrix.shape[0]):
                if sign_matrix[row, col] != 0 and sign_matrix[row-1, col] != 0:
                    if sign_matrix[row, col] != sign_matrix[row-1, col]:
                        n_flips += 1
        total_possible = (sign_matrix.shape[0] - 1) * sign_matrix.shape[1]
        ax.text(0.02, 0.98, f'Sign flips: {n_flips}/{total_possible}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    cbar = fig.colorbar(im, ax=axes, ticks=[-1, 0, 1], shrink=0.8)
    cbar.ax.set_yticklabels(['Negative', 'Zero', 'Positive'])
    
    fig.suptitle('Per-Bit Differential Bias Sign: Analytical Validation of Condition (C3)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'fig_bias_sign_heatmap.png', dpi=200, bbox_inches='tight')
    fig.savefig(output_dir / 'fig_bias_sign_heatmap.pdf', dpi=200, bbox_inches='tight')
    print(f"  Sign heatmap saved to {output_dir / 'fig_bias_sign_heatmap.png'}")
    plt.close(fig)
    
    # ── Generate magnitude decay curves ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (cipher_name, data) in zip(axes, all_biases.items()):
        bias_matrix = np.array(data['bias_matrix'])
        rounds = data['rounds']
        block_size = data['block_size']
        
        mean_abs_bias = np.mean(np.abs(bias_matrix), axis=1)
        max_abs_bias = np.max(np.abs(bias_matrix), axis=1)
        
        ax.semilogy(rounds, mean_abs_bias, 'o-', color='#3498db', 
                     label='Mean $|\\beta_j|$', linewidth=2, markersize=6)
        ax.semilogy(rounds, max_abs_bias, 's--', color='#e74c3c',
                     label='Max $|\\beta_j|$', linewidth=2, markersize=6)
        ax.set_xlabel('Round $r$', fontsize=11)
        ax.set_ylabel('Bias magnitude', fontsize=11)
        ax.set_title(cipher_configs[cipher_name]['label'], fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Differential Bias Magnitude Decay Across Rounds',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'fig_bias_magnitude_decay.png', dpi=200, bbox_inches='tight')
    fig.savefig(output_dir / 'fig_bias_magnitude_decay.pdf', dpi=200, bbox_inches='tight')
    print(f"  Magnitude decay saved to {output_dir / 'fig_bias_magnitude_decay.png'}")
    plt.close(fig)
    
    # ── Print summary statistics (significance-aware) ──
    # Noise floor: for n_samples * n_keys effective samples, 
    # a bias of |β| < threshold is indistinguishable from zero.
    noise_threshold = 3.0 / np.sqrt(args.samples)  # ~3σ significance
    
    print(f"\n{'='*60}")
    print(f"  SUMMARY: Condition (C3) Analytical Validation")
    print(f"  (significance threshold: |beta| > {noise_threshold:.4f})")
    print(f"{'='*60}")
    for cipher_name, data in all_biases.items():
        bias_matrix = np.array(data['bias_matrix'])
        sign_matrix = np.array(data['sign_matrix'])
        n_flips_sig = 0
        n_total_sig = 0
        n_flips_all = 0
        n_total_all = (sign_matrix.shape[0] - 1) * sign_matrix.shape[1]
        for col in range(sign_matrix.shape[1]):
            for row in range(1, sign_matrix.shape[0]):
                # All flips (raw)
                if sign_matrix[row, col] != 0 and sign_matrix[row-1, col] != 0:
                    if sign_matrix[row, col] != sign_matrix[row-1, col]:
                        n_flips_all += 1
                # Significance-aware: only count where BOTH rounds exceed threshold
                if (abs(bias_matrix[row, col]) > noise_threshold and 
                    abs(bias_matrix[row-1, col]) > noise_threshold):
                    n_total_sig += 1
                    if sign_matrix[row, col] != sign_matrix[row-1, col]:
                        n_flips_sig += 1
        
        flip_rate_sig = n_flips_sig / n_total_sig if n_total_sig > 0 else 0
        verdict = "C3 SATISFIED" if flip_rate_sig < 0.05 else "C3 VIOLATED"
        print(f"  {cipher_configs[cipher_name]['label']:30s}: "
              f"sig. flip rate = {flip_rate_sig:.3f} ({n_flips_sig}/{n_total_sig}) "
              f"[raw: {n_flips_all}/{n_total_all}] -> {verdict}")
    
    return all_biases


def main():
    parser = argparse.ArgumentParser(description='Exp31: Bias Sign Heatmap (C3 Validation)')
    parser.add_argument('--samples', type=int, default=2_000_000,
                        help='Number of plaintext pairs per key (default: 2M)')
    parser.add_argument('--n-keys', type=int, default=10,
                        help='Number of random keys to average over')
    parser.add_argument('--output-dir', default='results/e31_bias_sign',
                        help='Output directory')
    args = parser.parse_args()
    
    print(f"\n{'═'*60}")
    print(f"  Experiment 31: Analytical C3 Validation (Bias Sign Heatmap)")
    print(f"  {args.samples:,} samples × {args.n_keys} keys per cipher/round")
    print(f"{'═'*60}")
    
    t0 = time.time()
    run_experiment(args)
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
