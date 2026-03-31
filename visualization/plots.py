
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def plot_accuracy_vs_rounds(
    results: Dict[str, Dict[int, float]],
    title: str = 'Distinguisher Accuracy vs. Number of Rounds',
    save_path: Optional[str] = None,
    show_advantage: bool = True,
    threshold: float = 0.51
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (cipher, accuracies) in enumerate(results.items()):
        rounds = sorted(accuracies.keys())
        accs = [accuracies[r] for r in rounds]
        
        if show_advantage:
            values = [2 * abs(a - 0.5) for a in accs]
            ylabel = 'Advantage'
            thresh = 2 * abs(threshold - 0.5)
        else:
            values = accs
            ylabel = 'Accuracy'
            thresh = threshold
        
        ax.plot(rounds, values, 
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                markersize=8,
                linewidth=2,
                label=cipher.upper())
    
    ax.axhline(y=thresh, color='gray', linestyle='--', alpha=0.5, 
               label=f'Threshold ({thresh:.2f})')
    
    ax.set_xlabel('Number of Rounds')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    if not show_advantage:
        ax.set_ylim(0.45, 1.0)
    else:
        ax.set_ylim(0, 1.0)
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_representation_comparison(
    results: Dict[str, float],
    title: str = 'Representation Comparison',
    metric: str = 'accuracy',
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = list(results.keys())
    values = list(results.values())
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(values)))
    sorted_indices = np.argsort(values)
    bar_colors = [colors[np.where(sorted_indices == i)[0][0]] for i in range(len(values))]
    
    bars = ax.bar(range(len(names)), values, color=bar_colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('_', '\n') for n in names], rotation=45, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   ha='center', va='bottom', fontsize=9)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_signal_decay_heatmap(
    mi_matrix: np.ndarray,
    title: str = 'Mutual Information Across Rounds',
    round_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if round_labels is None:
        round_labels = [f'R{i+1}' for i in range(mi_matrix.shape[0])]
    
    mask = np.triu(np.ones_like(mi_matrix, dtype=bool), k=1)
    
    sns.heatmap(
        mi_matrix,
        mask=mask,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        xticklabels=round_labels,
        yticklabels=round_labels,
        ax=ax,
        cbar_kws={'label': 'Mutual Information (nats)'}
    )
    
    ax.set_title(title)
    ax.set_xlabel('Round j')
    ax.set_ylabel('Round i')
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = 'Training Progress',
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    ax1 = axes[0]
    if 'train_loss' in history:
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    if 'train_acc' in history:
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    if 'val_acc' in history:
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str] = ['Random', 'Cipher'],
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    title: str = 'ROC Curve',
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_memory_depth(
    results: Dict[int, float],
    title: str = 'Memory Depth vs Accuracy',
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    depths = sorted(results.keys())
    accs = [results[d] for d in depths]
    
    ax.plot(depths, accs, 'o-', markersize=10, linewidth=2, color='#3498db')
    
    best_depth = depths[np.argmax(accs)]
    best_acc = max(accs)
    ax.scatter([best_depth], [best_acc], s=200, c='red', zorder=5, 
               label=f'Optimal (depth={best_depth})')
    
    ax.set_xlabel('History Depth (rounds)')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_markov_gap(
    markov_gaps: Dict[int, float],
    title: str = 'Markov Gap vs Round',
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rounds = sorted(markov_gaps.keys())
    gaps = [markov_gaps[r] for r in rounds]
    
    ax.bar(rounds, gaps, color='#2ecc71', edgecolor='black', alpha=0.7)
    
    ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, 
               label='Markov threshold (0.01)')
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Markov Gap (nats)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_saliency_map(
    saliency: np.ndarray,
    title: str = 'Bit-Level Saliency',
    word_size: int = 16,
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    
    block_size = len(saliency)
    n_words = block_size // word_size
    
    saliency_2d = saliency.reshape(n_words, word_size)
    
    im = ax.imshow(saliency_2d, cmap='hot', aspect='auto')
    
    ax.set_xlabel('Bit Position (within word)')
    ax.set_ylabel('Word')
    ax.set_title(title)
    
    plt.colorbar(im, ax=ax, label='Saliency')
    
    ax.set_xticks(range(0, word_size, 4))
    ax.set_xticklabels(range(0, word_size, 4))
    ax.set_yticks(range(n_words))
    ax.set_yticklabels([f'W{i}' for i in range(n_words)])
    
    if save_path:
        fig.savefig(save_path)
    
    return fig
