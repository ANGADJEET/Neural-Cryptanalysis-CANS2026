#!/usr/bin/env python3
"""
Publication-quality figure generation for Neural Cryptanalysis paper.
Reads results from JSON files and produces camera-ready plots.

Usage:
  python scripts/generate_figures.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─── Style Setup ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 7,
})

COLORS = {
    'speck32': '#2563EB',   # blue
    'simon32': '#DC2626',   # red
    'present': '#059669',   # green
    'classical': '#9CA3AF', # gray
    'gohr': '#F59E0B',      # amber
    'random': '#D1D5DB',    # light gray
}

MARKERS = {
    'speck32': 'o',
    'simon32': 's',
    'present': '^',
    'classical': 'D',
    'gohr': 'P',
}

LABELS = {
    'speck32': 'SPECK32/64 (ARX)',
    'simon32': 'SIMON32/64 (Feistel)',
    'present': 'PRESENT-64/80 (SPN)',
}

output_dir = Path('./results/figures')
output_dir.mkdir(parents=True, exist_ok=True)


# ─── Load Data ───────────────────────────────────────────────────────────────
def load_json(path):
    with open(path) as f:
        return json.load(f)


# SPECK32 E01 from prior run
speck_e01 = {3: 0.99994, 4: 0.97776, 5: 0.86588, 6: 0.66584, 7: 0.51278, 8: 0.49778}
speck_e11 = load_json('results/pending_fixes/e11_fixed.json')
speck_e09 = load_json('results/pending_fixes/e09_statistical_test.json')

simon_data = load_json('results/multi_cipher/simon32_results.json')
present_data = load_json('results/multi_cipher/present_results.json')


# ─── Figure 1: Baseline Accuracy vs Rounds (All 3 Ciphers) ──────────────────
def fig1_baseline():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # SPECK32
    rounds = sorted(speck_e01.keys())
    accs = [speck_e01[r] for r in rounds]
    ax.plot(rounds, accs, color=COLORS['speck32'], marker=MARKERS['speck32'],
            label=LABELS['speck32'], zorder=5)

    # SIMON32
    s_e01 = simon_data['e01']
    s_rounds = sorted([int(k) for k in s_e01.keys()])
    s_means = [s_e01[str(r)]['mean'] for r in s_rounds]
    s_stds = [s_e01[str(r)]['std'] for r in s_rounds]
    ax.errorbar(s_rounds, s_means, yerr=s_stds, color=COLORS['simon32'],
                marker=MARKERS['simon32'], label=LABELS['simon32'],
                capsize=3, capthick=1.5, zorder=5)

    # PRESENT
    p_e01 = present_data['e01']
    p_rounds = sorted([int(k) for k in p_e01.keys()])
    p_means = [p_e01[str(r)]['mean'] for r in p_rounds]
    p_stds = [p_e01[str(r)]['std'] for r in p_rounds]
    ax.errorbar(p_rounds, p_means, yerr=p_stds, color=COLORS['present'],
                marker=MARKERS['present'], label=LABELS['present'],
                capsize=3, capthick=1.5, zorder=5)

    ax.axhline(y=0.5, color=COLORS['random'], linestyle='--', linewidth=1.5,
               label='Random (0.5)', zorder=1)

    ax.set_xlabel('Number of Encryption Rounds')
    ax.set_ylabel('Distinguisher Accuracy')
    ax.set_title('Neural Distinguisher Accuracy vs. Round Count')
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_baseline_all_ciphers.png')
    plt.savefig(output_dir / 'fig1_baseline_all_ciphers.pdf')
    plt.close()
    print("  ✓ Figure 1: Baseline accuracy")


# ─── Figure 2: Neural vs Classical (All 3 Ciphers) ──────────────────────────
def fig2_neural_vs_classical():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    GOHR = {5: 0.9244, 6: 0.7880, 7: 0.6116, 8: 0.5134}

    # SPECK32
    ax = axes[0]
    rounds = sorted([int(k) for k in speck_e11.keys()])
    neural = [speck_e11[str(r)]['neural'] for r in rounds]
    classical = [speck_e11[str(r)]['classical'] for r in rounds]
    gohr_r = [r for r in rounds if r in GOHR]
    gohr_v = [GOHR[r] for r in gohr_r]

    ax.plot(rounds, neural, color=COLORS['speck32'], marker='o', label='Neural (MLP)')
    ax.plot(rounds, classical, color=COLORS['classical'], marker='D',
            linestyle='--', label='Classical (bit-bias)')
    ax.plot(gohr_r, gohr_v, color=COLORS['gohr'], marker='P',
            linestyle=':', label='Gohr ResNet')
    ax.axhline(y=0.5, color=COLORS['random'], linestyle='--', linewidth=1, alpha=0.5)
    ax.set_title('SPECK32/64 (ARX)')
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize=8, loc='lower left')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # SIMON32
    ax = axes[1]
    s_e11 = simon_data['e11']
    s_rounds = sorted([int(k) for k in s_e11.keys()])
    s_neural = [s_e11[str(r)]['neural'] for r in s_rounds]
    s_classical = [s_e11[str(r)]['classical'] for r in s_rounds]

    ax.plot(s_rounds, s_neural, color=COLORS['simon32'], marker='s', label='Neural (MLP)')
    ax.plot(s_rounds, s_classical, color=COLORS['classical'], marker='D',
            linestyle='--', label='Classical (bit-bias)')
    ax.axhline(y=0.5, color=COLORS['random'], linestyle='--', linewidth=1, alpha=0.5)
    ax.set_title('SIMON32/64 (Feistel)')
    ax.set_xlabel('Rounds')
    ax.legend(fontsize=8, loc='lower left')

    # PRESENT
    ax = axes[2]
    p_e11 = present_data['e11']
    p_rounds = sorted([int(k) for k in p_e11.keys()])
    p_neural = [p_e11[str(r)]['neural'] for r in p_rounds]
    p_classical = [p_e11[str(r)]['classical'] for r in p_rounds]

    ax.plot(p_rounds, p_neural, color=COLORS['present'], marker='^', label='Neural (MLP)')
    ax.plot(p_rounds, p_classical, color=COLORS['classical'], marker='D',
            linestyle='--', label='Classical (bit-bias)')
    ax.axhline(y=0.5, color=COLORS['random'], linestyle='--', linewidth=1, alpha=0.5)
    ax.set_title('PRESENT-64/80 (SPN)')
    ax.set_xlabel('Rounds')
    ax.legend(fontsize=8, loc='lower left')

    fig.suptitle('Neural vs. Classical Distinguisher Accuracy', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_neural_vs_classical.png')
    plt.savefig(output_dir / 'fig2_neural_vs_classical.pdf')
    plt.close()
    print("  ✓ Figure 2: Neural vs Classical")


# ─── Figure 3: Anti-Transfer Heatmap ────────────────────────────────────────
def fig3_transfer():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: Cross-round transfer — SPECK32
    ax = axes[0]
    source_r = 5
    cross_round_data = speck_e09.get('cross_round_accs', {})
    target_rounds = sorted([int(k) for k in cross_round_data.keys()])
    means = [np.mean(cross_round_data[str(r)]) for r in target_rounds]

    colors_bar = ['#EF4444' if m < 0.5 else '#22C55E' if m > 0.52 else '#9CA3AF'
                  for m in means]
    bars = ax.bar(target_rounds, means, color=colors_bar, edgecolor='white',
                  linewidth=0.5, width=0.7)
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel(f'Test Round (trained on {source_r}r)')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'SPECK32: Cross-Round Transfer')
    ax.set_ylim(0.43, 0.55)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # Annotate
    for bar, m, r in zip(bars, means, target_rounds):
        offset = -0.007 if m < 0.5 else 0.003
        ax.text(bar.get_x() + bar.get_width()/2, m + offset,
                f'{m:.1%}', ha='center', va='bottom' if m >= 0.5 else 'top',
                fontsize=9, fontweight='bold')

    # Panel B: Cross-round transfer — PRESENT (positive!)
    ax = axes[1]
    p_e09 = present_data['e09']
    source_r_p = p_e09['source_rounds']
    p_targets = []
    p_means = []
    for key in sorted(p_e09.keys()):
        if key.startswith('cross_round_'):
            r = int(key.replace('cross_round_', ''))
            p_targets.append(r)
            p_means.append(p_e09[key]['mean'])

    colors_bar_p = ['#EF4444' if m < 0.5 else '#22C55E' if m > 0.52 else '#9CA3AF'
                    for m in p_means]
    bars_p = ax.bar(p_targets, p_means, color=colors_bar_p, edgecolor='white',
                    linewidth=0.5, width=0.7)
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel(f'Test Round (trained on {source_r_p}r)')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'PRESENT: Cross-Round Transfer')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    for bar, m, r in zip(bars_p, p_means, p_targets):
        offset = 0.005
        ax.text(bar.get_x() + bar.get_width()/2, m + offset,
                f'{m:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.suptitle('Transfer Behavior Across Cipher Families', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_transfer_comparison.png')
    plt.savefig(output_dir / 'fig3_transfer_comparison.pdf')
    plt.close()
    print("  ✓ Figure 3: Transfer comparison")


# ─── Figure 4: Neural Advantage Gap ─────────────────────────────────────────
def fig4_gap():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # SPECK32
    rounds = sorted([int(k) for k in speck_e11.keys()])
    gaps = [speck_e11[str(r)]['gap'] for r in rounds]
    ax.plot(rounds, [g*100 for g in gaps], color=COLORS['speck32'],
            marker=MARKERS['speck32'], label=LABELS['speck32'])

    # SIMON32
    s_e11 = simon_data['e11']
    s_rounds = sorted([int(k) for k in s_e11.keys()])
    s_gaps = [s_e11[str(r)]['gap'] for r in s_rounds]
    ax.plot(s_rounds, [g*100 for g in s_gaps], color=COLORS['simon32'],
            marker=MARKERS['simon32'], label=LABELS['simon32'])

    # PRESENT
    p_e11 = present_data['e11']
    p_rounds = sorted([int(k) for k in p_e11.keys()])
    p_gaps = [p_e11[str(r)]['gap'] for r in p_rounds]
    ax.plot(p_rounds, [g*100 for g in p_gaps], color=COLORS['present'],
            marker=MARKERS['present'], label=LABELS['present'])

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.fill_between([2, 12], [0, 0], [-5, -5], alpha=0.05, color='red')

    ax.set_xlabel('Number of Encryption Rounds')
    ax.set_ylabel('Neural Advantage (pp)')
    ax.set_title('Neural − Classical Accuracy Gap')
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_neural_advantage.png')
    plt.savefig(output_dir / 'fig4_neural_advantage.pdf')
    plt.close()
    print("  ✓ Figure 4: Neural advantage gap")


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating publication figures...")
    fig1_baseline()
    fig2_neural_vs_classical()
    fig3_transfer()
    fig4_gap()
    print(f"\n✓ All figures saved to {output_dir}/")
    print(f"  PNG (raster) + PDF (vector) for each figure")
