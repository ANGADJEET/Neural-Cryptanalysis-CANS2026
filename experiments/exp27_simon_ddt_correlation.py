#!/usr/bin/env python3
"""
E27: Simon32/64 Empirical DDT Bit-Bias Correlation

This script addresses the "anti-transfer paradox" for Simon by measuring the empirical
bit-biases of the output differences at rounds 4, 5, and 6. It calculates the Pearson
correlation of these biases between adjacent rounds to demonstrate that the DDT marginal
distributions oscillate in sign due to the AND-rotation nonlinearity.

This explains why a DDT-only classifier experiences anti-transfer: a feature that
is highly indicative of the "real" distribution at round 6 may be indicative of the
"random" (or opposite) distribution at round 4.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
plt.style.use('ggplot')

from ciphers import get_cipher

def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)

def get_empirical_biases(cipher_name, n_rounds, n_samples=5000000, seed=42):
    set_seed(seed)
    cipher = get_cipher(cipher_name)
    delta_p = cipher.get_default_delta_p()
    
    # Generate positive samples only (real differential pairs)
    key = cipher.random_key()
    P = cipher.random_plaintexts(n_samples)
    P_prime = P ^ delta_p
    
    C = cipher.encrypt(P, n_rounds, key)
    C_prime = cipher.encrypt(P_prime, n_rounds, key)
    
    diff = C ^ C_prime
    block_size = cipher.block_size
    
    biases = np.zeros(block_size)
    for i in range(block_size):
        bit = (diff >> i) & 1
        prob_one = bit.mean()
        # Bias from uniform (0.5)
        biases[block_size - 1 - i] = prob_one - 0.5
        
    return biases

def main():
    print("=" * 60)
    print("  E27: Simon Empirical DDT Bit-Bias Correlation")
    print("=" * 60)
    
    cipher_name = 'simon32'
    rounds = [4, 5, 6]
    n_samples = 5000000
    
    results = {}
    biases_dict = {}
    
    for r in rounds:
        print(f"Computing empirical biases for {cipher_name} at {r} rounds ({n_samples} samples)...")
        biases = get_empirical_biases(cipher_name, r, n_samples=n_samples)
        biases_dict[r] = biases
        
    print("\nCorrelations:")
    correlations = {}
    pairs = [(4, 5), (5, 6), (4, 6)]
    for r1, r2 in pairs:
        corr, pval = pearsonr(biases_dict[r1], biases_dict[r2])
        correlations[f"{r1}r_vs_{r2}r"] = {'r': float(corr), 'p_value': float(pval)}
        print(f"  {r1}r vs {r2}r: r = {corr:.4f} (p = {pval:.4e})")
        
    # Find top bits at 6r to track their evolution
    top_6r_idx = np.argsort(np.abs(biases_dict[6]))[-5:][::-1]
    
    print("\nEvolution of top 5 bits (ranked by absolute bias at 6r):")
    for idx in top_6r_idx:
        b4 = biases_dict[4][idx]
        b5 = biases_dict[5][idx]
        b6 = biases_dict[6][idx]
        print(f"  Bit {31-idx:2d}: 4r = {b4:+.4f}, 5r = {b5:+.4f}, 6r = {b6:+.4f}")
        
    results['correlations'] = correlations
    results['top_bits_evolution'] = {
        int(31-idx): {
            '4r': float(biases_dict[4][idx]),
            '5r': float(biases_dict[5][idx]),
            '6r': float(biases_dict[6][idx])
        } for idx in top_6r_idx
    }
    
    # Save results
    out_dir = Path('./results/e27_simon_ddt_correlation')
    ensure_dir(out_dir)
    with open(out_dir / 'simon_ddt_correlation.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    # Plotting
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot: 4r vs 5r vs 6r
    ax = axes[0]
    ax.scatter(biases_dict[4], biases_dict[5], label='4r vs 5r', alpha=0.7, color='blue')
    ax.scatter(biases_dict[5], biases_dict[6], label='5r vs 6r', alpha=0.7, color='red')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Bias at Source Round (prob - 0.5)')
    ax.set_ylabel('Bias at Target Round (prob - 0.5)')
    ax.set_title('Bit-Bias Oscillation in Simon32/64')
    ax.legend()
    
    # Bar chart of top bits
    ax = axes[1]
    bits = [str(31-idx) for idx in top_6r_idx]
    x = np.arange(len(bits))
    width = 0.25
    
    ax.bar(x - width, [biases_dict[4][idx] for idx in top_6r_idx], width, label='4 Rounds', color='#4c72b0')
    ax.bar(x, [biases_dict[5][idx] for idx in top_6r_idx], width, label='5 Rounds', color='#dd8452')
    ax.bar(x + width, [biases_dict[6][idx] for idx in top_6r_idx], width, label='6 Rounds', color='#55a868')
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(bits)
    ax.set_xlabel('Bit Index')
    ax.set_ylabel('Empirical Bias (prob - 0.5)')
    ax.set_title('Evolution of Top 6r Features')
    ax.legend()
    
    plt.tight_layout()
    fig_path = Path('paper_cans/figures/fig14_simon_ddt.pdf')
    ensure_dir(fig_path.parent)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved results to {out_dir}")
    print(f"Saved figure to {fig_path}")

if __name__ == '__main__':
    main()
