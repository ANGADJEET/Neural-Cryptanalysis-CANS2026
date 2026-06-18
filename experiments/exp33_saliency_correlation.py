"""
Experiment 33: DDT vs ML Feature Saliency Correlation
======================================================
Validates Condition (C2) — that neural models are learning DDT-derived features.

Method:
  1. Load analytical bias magnitudes |β_j| from exp31.
  2. Load existing SmoothGrad saliency vectors from e22_cross_saliency results.
  3. Compute Spearman rank correlation ρ between theory and empirics per cipher/round.
  4. Generate bar charts and scatter plots.

Expected result:
  High Spearman ρ for all ciphers — the neural network's attention aligns perfectly
  with the cipher's analytical differential bias structure.

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
    
    # ── Load saliency data from e22 ──
    saliency_dir = Path(args.saliency_dir)
    
    cipher_configs = {
        'speck32': {
            'saliency_file': 'e22_speck32_results.json',
            'label': 'SPECK32/64',
            'color': '#3498db',
        },
        'simon32': {
            'saliency_file': 'e22_simon32_results.json',
            'label': 'SIMON32/64',
            'color': '#e74c3c',
        },
        'present': {
            'saliency_file': 'e22_present_results.json',
            'label': 'PRESENT',
            'color': '#2ecc71',
        },
    }
    
    all_correlations = {}
    
    for cipher_name, cfg in cipher_configs.items():
        sal_path = saliency_dir / cfg['saliency_file']
        if not sal_path.exists():
            print(f"  Skipping {cipher_name}: saliency file not found at {sal_path}")
            continue
        
        if cipher_name not in bias_data:
            print(f"  Skipping {cipher_name}: not in bias data")
            continue
        
        with open(sal_path) as f:
            sal_data = json.load(f)
        
        bd = bias_data[cipher_name]
        rounds_list = bd['rounds']
        bias_matrix = np.array(bd['bias_matrix'])
        
        # Get averaged saliency vectors
        avg_saliency = sal_data.get('avg_saliency', {})
        sal_rounds = [int(r) for r in avg_saliency.keys()]
        
        print(f"\n  {cfg['label']}:")
        print(f"    Bias rounds: {rounds_list}")
        print(f"    Saliency rounds: {sal_rounds}")
        
        correlations = {}
        
        for r_str, sal_vec in avg_saliency.items():
            r = int(r_str)
            if r not in rounds_list:
                continue
            
            r_idx = rounds_list.index(r)
            theory_vec = np.abs(bias_matrix[r_idx])
            
            # CRITICAL FIX: exp31 computes bias from LSB (idx 0) to MSB (idx n).
            # But the neural network's R2_xor_diff representation packs from 
            # MSB (idx 0) to LSB (idx n). We must reverse theory_vec to align them.
            theory_vec = theory_vec[::-1]
            
            sal_vec_np = np.abs(np.array(sal_vec))
            
            # Ensure same length
            min_len = min(len(theory_vec), len(sal_vec_np))
            theory_vec = theory_vec[:min_len]
            sal_vec_np = sal_vec_np[:min_len]
            
            if np.std(theory_vec) < 1e-15 or np.std(sal_vec_np) < 1e-15:
                print(f"    Round {r}: constant vector, skipping")
                continue
            
            rho, p_val = stats.spearmanr(theory_vec, sal_vec_np)
            correlations[r] = {'rho': rho, 'p_value': p_val}
            print(f"    Round {r}: Spearman ρ = {rho:.4f}  (p = {p_val:.2e})")
        
        all_correlations[cipher_name] = correlations
    
    if not all_correlations:
        print("\n  ERROR: No correlations computed. Check data paths.")
        return
    
    # ── Bar chart of Spearman ρ per cipher/round ──
    fig, ax = plt.subplots(figsize=(10, 5))
    
    bar_width = 0.25
    cipher_list = [c for c in ['speck32', 'simon32', 'present'] if c in all_correlations]
    
    # Collect all rounds across ciphers
    all_rounds = sorted(set(
        r for corrs in all_correlations.values() for r in corrs.keys()
    ))
    
    x = np.arange(len(all_rounds))
    
    for i, cipher_name in enumerate(cipher_list):
        cfg = cipher_configs[cipher_name]
        corrs = all_correlations[cipher_name]
        rhos = [corrs[r]['rho'] if r in corrs else 0.0 for r in all_rounds]
        
        bars = ax.bar(x + i * bar_width, rhos, bar_width,
                      label=cfg['label'], color=cfg['color'],
                      edgecolor='black', linewidth=0.5, alpha=0.85)
        
        # Add value labels
        for bar, rho in zip(bars, rhos):
            if rho != 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{rho:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Round $r$', fontsize=12)
    ax.set_ylabel('Spearman $\\rho$ (Theory vs. ML Saliency)', fontsize=12)
    ax.set_title('Condition (C2) Validation: DDT Bias Predicts Neural Attention',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x + bar_width * (len(cipher_list) - 1) / 2)
    ax.set_xticklabels(all_rounds)
    ax.legend(fontsize=10)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylim(-0.3, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'fig_saliency_correlation.png', dpi=200, bbox_inches='tight')
    fig.savefig(output_dir / 'fig_saliency_correlation.pdf', dpi=200, bbox_inches='tight')
    print(f"\n  Bar chart saved to {output_dir / 'fig_saliency_correlation.png'}")
    plt.close(fig)
    
    # ── Scatter plot: theory vs saliency for one representative round per cipher ──
    fig, axes = plt.subplots(1, len(cipher_list), figsize=(6 * len(cipher_list), 5))
    if len(cipher_list) == 1:
        axes = [axes]
    
    for ax, cipher_name in zip(axes, cipher_list):
        cfg = cipher_configs[cipher_name]
        corrs = all_correlations[cipher_name]
        
        # Pick the round with the best data overlap
        bd = bias_data[cipher_name]
        rounds_list = bd['rounds']
        bias_matrix = np.array(bd['bias_matrix'])
        
        sal_path = saliency_dir / cfg['saliency_file']
        with open(sal_path) as f:
            sal_data = json.load(f)
        avg_saliency = sal_data.get('avg_saliency', {})
        
        # Use the first available round
        plot_round = None
        for r in sorted(corrs.keys()):
            plot_round = r
            break
        
        if plot_round is None:
            continue
        
        r_idx = rounds_list.index(plot_round)
        theory_vec = np.abs(bias_matrix[r_idx])
        sal_vec = np.abs(np.array(avg_saliency[str(plot_round)]))
        min_len = min(len(theory_vec), len(sal_vec))
        
        ax.scatter(theory_vec[:min_len], sal_vec[:min_len],
                   c=cfg['color'], s=40, alpha=0.7, edgecolors='black', linewidth=0.3)
        
        rho = corrs[plot_round]['rho']
        ax.set_xlabel('Theoretical $|\\beta_j|$', fontsize=11)
        ax.set_ylabel('ML Saliency (SmoothGrad)', fontsize=11)
        ax.set_title(f'{cfg["label"]} (round {plot_round})\n'
                     f'Spearman $\\rho$ = {rho:.3f}',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Per-Bit: Analytical Bias vs Neural Saliency',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'fig_saliency_scatter.png', dpi=200, bbox_inches='tight')
    fig.savefig(output_dir / 'fig_saliency_scatter.pdf', dpi=200, bbox_inches='tight')
    print(f"  Scatter plot saved to {output_dir / 'fig_saliency_scatter.png'}")
    plt.close(fig)
    
    # ── Save results ──
    results = {}
    for cipher_name, corrs in all_correlations.items():
        results[cipher_name] = {
            str(r): {'spearman_rho': c['rho'], 'p_value': c['p_value']}
            for r, c in corrs.items()
        }
        rhos = [c['rho'] for c in corrs.values()]
        results[cipher_name]['mean_rho'] = float(np.mean(rhos))
    
    with open(output_dir / 'e33_saliency_correlation.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {output_dir / 'e33_saliency_correlation.json'}")
    
    # ── Summary ──
    print(f"\n{'='*60}")
    print("  SUMMARY: Condition (C2) Validation")
    print(f"{'='*60}")
    for cipher_name, corrs in all_correlations.items():
        rhos = [c['rho'] for c in corrs.values()]
        mean_rho = np.mean(rhos)
        label = cipher_configs[cipher_name]['label']
        verdict = "HIGH alignment" if mean_rho > 0.6 else "LOW alignment"
        print(f"  {label:15s}: mean Spearman ρ = {mean_rho:.4f} → {verdict}")


def main():
    parser = argparse.ArgumentParser(
        description='Exp33: DDT vs ML Saliency Correlation (C2 Validation)')
    parser.add_argument('--bias-data', default='results/e31_bias_sign/e31_bias_data.json',
                        help='Path to exp31 bias data')
    parser.add_argument('--saliency-dir', default='results/e22_cross_saliency',
                        help='Path to e22 saliency results')
    parser.add_argument('--output-dir', default='results/e33_saliency_correlation',
                        help='Output directory')
    args = parser.parse_args()
    
    print(f"\n{'═'*60}")
    print(f"  Experiment 33: DDT vs ML Saliency Correlation (C2 Validation)")
    print(f"{'═'*60}")
    
    run_experiment(args)


if __name__ == '__main__':
    main()
