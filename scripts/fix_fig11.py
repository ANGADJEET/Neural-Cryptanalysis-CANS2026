import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def load_data(cipher):
    with open(f'results/e22_cross_saliency/e22_{cipher}_results.json', 'r') as f:
        data = json.load(f)
    return data['eval_rounds'], np.array(data['correlation_matrix'])

speck_rounds, speck_corr = load_data('speck32')
simon_rounds, simon_corr = load_data('simon32')
present_rounds, present_corr = load_data('present')

# Set up figure
fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})

data_list = [
    ('Speck', speck_rounds, speck_corr),
    ('Simon', simon_rounds, simon_corr),
    ('Present', present_rounds, present_corr)
]

for idx, (title, rounds, corr) in enumerate(data_list):
    ax = axes[idx]
    im = ax.imshow(corr, cmap='RdYlBu_r', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(rounds)))
    ax.set_xticklabels([f"{r}r" for r in rounds])
    ax.set_yticks(range(len(rounds)))
    ax.set_yticklabels([f"{r}r" for r in rounds])
    ax.set_title(title)
    ax.grid(False)
    
    for i in range(len(rounds)):
        for j in range(len(rounds)):
            val = corr[i, j]
            # only show lower triangle or all? based on screenshot it's lower triangle
            if i >= j:
                color = 'white' if val < 0.3 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=color, fontsize=9)

# Add colorbar using the ax keyword to let matplotlib handle layout
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), aspect=30, pad=0.02)
cbar.set_label(r'Spearman $\rho$', rotation=90, labelpad=10)

out_path = Path('paper_cans/figures/fig11_saliency_corr.pdf')
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.savefig(out_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
print("Saved fixed figure to", out_path)
