"""
Experiment 32: Theoretical vs. Empirical Transfer Prediction
=============================================================
Validates that the mathematical framework precisely predicts ML model behavior.

Method:
  1. Load theoretical bias vectors β^(r) from exp31 results.
  2. Compute cosine similarity cos(β^(source), β^(target)) for each transfer pair.
  3. Load empirical transfer accuracies from existing e09/multi_cipher results.
  4. Plot theoretical cosine similarity vs. empirical transfer accuracy.

Expected result:
  Strong positive correlation — when cos(β^(r), β^(r')) < 0, transfer accuracy < 50%.

Runtime: < 1 minute CPU (loads pre-computed data).
"""

import argparse
import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-15 or norm_b < 1e-15:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def load_empirical_transfer(results_dir):
    """
    Load empirical transfer results from existing experiment outputs.
    Returns dict: {cipher_name: {(source_r, target_r): accuracy}}.
    """
    results_dir = Path(results_dir)
    empirical = {}
    
    # SPECK: source=5r, targets in e09_transfer
    speck_path = results_dir / 'e09_transfer' / 'e09_speck32_results.json'
    if speck_path.exists():
        with open(speck_path) as f:
            data = json.load(f)
        empirical['speck32'] = {
            'source_round': 5,
            'transfers': {}
        }
        for target_r, vals in data['cross_round'].items():
            empirical['speck32']['transfers'][int(target_r)] = vals['mean']
    
    # SIMON: source=6r, from multi_cipher
    simon_path = results_dir / 'multi_cipher' / 'simon32_results.json'
    if simon_path.exists():
        with open(simon_path) as f:
            data = json.load(f)
        if 'e09' in data:
            e09 = data['e09']
            empirical['simon32'] = {
                'source_round': e09.get('source_rounds', 6),
                'transfers': {}
            }
            for key, val in e09.items():
                if key.startswith('cross_round_'):
                    target_r = int(key.replace('cross_round_', ''))
                    empirical['simon32']['transfers'][target_r] = val['mean']
    
    # PRESENT: source=4r, from multi_cipher
    present_path = results_dir / 'multi_cipher' / 'all_ciphers_results.json'
    if present_path.exists():
        with open(present_path) as f:
            data = json.load(f)
        if 'present' in data and 'e09' in data['present']:
            e09 = data['present']['e09']
            empirical['present'] = {
                'source_round': e09.get('source_rounds', 4),
                'transfers': {}
            }
            for key, val in e09.items():
                if key.startswith('cross_round_'):
                    target_r = int(key.replace('cross_round_', ''))
                    empirical['present']['transfers'][target_r] = val['mean']
    
    return empirical


def run_experiment(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Load theoretical biases from exp31 ──
    bias_path = Path(args.bias_data)
    if not bias_path.exists():
        print(f"  ERROR: Bias data not found at {bias_path}")
        print(f"  Please run exp31_bias_sign_heatmap.py first.")
        return
    
    with open(bias_path) as f:
        bias_data = json.load(f)
    
    # ── Load empirical transfer results ──
    empirical = load_empirical_transfer(args.results_dir)
    
    print(f"\n  Loaded empirical transfer data for: {list(empirical.keys())}")
    
    # ── Compute cosine similarities and pair with empirical accuracies ──
    all_points = []  # (cipher, source_r, target_r, cos_sim, empirical_acc)
    
    cipher_markers = {
        'present': ('o', '#2ecc71', 'PRESENT'),
        'simon32': ('s', '#e74c3c', 'SIMON32/64'),
        'speck32': ('^', '#3498db', 'SPECK32/64'),
    }
    
    for cipher_name in ['present', 'simon32', 'speck32']:
        if cipher_name not in bias_data or cipher_name not in empirical:
            print(f"  Skipping {cipher_name}: missing data")
            continue
        
        bd = bias_data[cipher_name]
        emp = empirical[cipher_name]
        
        rounds_list = bd['rounds']
        bias_matrix = np.array(bd['bias_matrix'])
        source_r = emp['source_round']
        
        if source_r not in rounds_list:
            print(f"  Warning: source round {source_r} not in bias rounds {rounds_list}")
            continue
        
        source_idx = rounds_list.index(source_r)
        source_bias = bias_matrix[source_idx]
        
        for target_r, emp_acc in emp['transfers'].items():
            if target_r in rounds_list:
                target_idx = rounds_list.index(target_r)
                target_bias = bias_matrix[target_idx]
                cos_sim = cosine_similarity(source_bias, target_bias)
                all_points.append((cipher_name, source_r, target_r, cos_sim, emp_acc))
                print(f"  {cipher_markers[cipher_name][2]:15s}: "
                      f"{source_r}r→{target_r}r  cos={cos_sim:+.4f}  acc={emp_acc:.4f}")
    
    if not all_points:
        print("  ERROR: No matching data points found.")
        return
    
    # ── Compute overall correlation ──
    cos_vals = np.array([p[3] for p in all_points])
    acc_vals = np.array([p[4] for p in all_points])
    
    if len(cos_vals) >= 3:
        pearson_r, pearson_p = stats.pearsonr(cos_vals, acc_vals)
        spearman_rho, spearman_p = stats.spearmanr(cos_vals, acc_vals)
    else:
        pearson_r, pearson_p = 0, 1
        spearman_rho, spearman_p = 0, 1
    
    print(f"\n  Pearson  r = {pearson_r:.4f}  (p = {pearson_p:.2e})")
    print(f"  Spearman ρ = {spearman_rho:.4f}  (p = {spearman_p:.2e})")
    
    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for cipher_name in ['present', 'simon32', 'speck32']:
        marker, color, label = cipher_markers[cipher_name]
        pts = [(p[3], p[4]) for p in all_points if p[0] == cipher_name]
        if pts:
            x, y = zip(*pts)
            ax.scatter(x, y, marker=marker, c=color, s=100, label=label,
                       edgecolors='black', linewidth=0.5, zorder=3)
    
    # Reference lines
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance (50%)')
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    
    # Shade anti-transfer quadrant
    ax.axhspan(0, 0.5, xmin=0, xmax=0.5, alpha=0.05, color='red')
    ax.text(-0.7, 0.46, 'Anti-transfer\nquadrant', fontsize=9, color='#e74c3c',
            ha='center', style='italic')
    
    # Linear fit
    if len(cos_vals) >= 3:
        z = np.polyfit(cos_vals, acc_vals, 1)
        x_fit = np.linspace(cos_vals.min() - 0.1, cos_vals.max() + 0.1, 100)
        y_fit = np.polyval(z, x_fit)
        ax.plot(x_fit, y_fit, '--', color='#7f8c8d', alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Theoretical Cosine Similarity $\\cos(\\beta^{(r_{src})}, \\beta^{(r_{tgt})})$',
                   fontsize=12)
    ax.set_ylabel('Empirical Transfer Accuracy', fontsize=12)
    ax.set_title(f'Theory Predicts Practice: Pearson $r$ = {pearson_r:.3f}, '
                 f'Spearman $\\rho$ = {spearman_rho:.3f}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'fig_transfer_prediction.png', dpi=200, bbox_inches='tight')
    fig.savefig(output_dir / 'fig_transfer_prediction.pdf', dpi=200, bbox_inches='tight')
    print(f"\n  Plot saved to {output_dir / 'fig_transfer_prediction.png'}")
    plt.close(fig)
    
    # ── Save results ──
    results = {
        'points': [
            {
                'cipher': p[0], 'source_round': p[1], 'target_round': p[2],
                'cosine_similarity': p[3], 'empirical_accuracy': p[4]
            }
            for p in all_points
        ],
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_rho': spearman_rho,
        'spearman_p': spearman_p,
    }
    
    with open(output_dir / 'e32_transfer_prediction.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {output_dir / 'e32_transfer_prediction.json'}")


def main():
    parser = argparse.ArgumentParser(
        description='Exp32: Theoretical vs Empirical Transfer Prediction')
    parser.add_argument('--bias-data', default='results/e31_bias_sign/e31_bias_data.json',
                        help='Path to exp31 bias data')
    parser.add_argument('--results-dir', default='results',
                        help='Path to existing experiment results')
    parser.add_argument('--output-dir', default='results/e32_transfer_prediction',
                        help='Output directory')
    args = parser.parse_args()
    
    print(f"\n{'═'*60}")
    print(f"  Experiment 32: Theoretical vs. Empirical Transfer Prediction")
    print(f"{'═'*60}")
    
    run_experiment(args)


if __name__ == '__main__':
    main()
