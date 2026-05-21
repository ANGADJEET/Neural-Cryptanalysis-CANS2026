#!/usr/bin/env python3
"""
E22: Cross-Round Saliency Comparison

Computes gradient saliency of a distinguisher trained at round R, evaluated
on data from rounds R-2, R-1, R, R+1, R+2. Measures Spearman rank correlation
between saliency vectors at different eval rounds.

Expected result:
  - SPECK/SIMON (ARX/Feistel): Negative or near-zero correlation between
    saliency at source vs lower rounds (explaining anti-transfer).
  - PRESENT (SPN): Positive correlation (explaining positive transfer).

This provides mechanistic evidence for WHY anti-transfer occurs: the model
attends to different (or opposite) input bits at different round counts.

Usage:
  python experiments/exp22_cross_saliency.py --cipher speck32
  python experiments/exp22_cross_saliency.py --cipher simon32
  python experiments/exp22_cross_saliency.py --cipher present
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from ciphers import get_cipher
from data.generator import CipherDataGenerator
from data.dataloader import CryptoDataset, get_input_dim
from models import get_model
from training.trainer import Trainer
from evaluation.metrics import evaluate_model
from experiments.experiment_utils import (
    set_seed, add_common_args, save_results, get_device
)


CIPHER_CONFIGS = {
    'speck32': {'source': 5, 'eval_rounds': [3, 4, 5, 6, 7]},
    'simon32': {'source': 6, 'eval_rounds': [4, 5, 6, 7, 8]},
    'present': {'source': 4, 'eval_rounds': [2, 3, 4, 5, 6]},
}


def compute_gradient_saliency(model, dataloader, device, n_batches=20):
    """Compute average absolute gradient saliency: E[|∂L/∂x_i|].
    
    Uses the real BCE loss gradient (not a synthetic proxy).
    Averages over `n_batches` batches for stability.
    
    Returns:
        saliency: (input_dim,) numpy array of mean absolute gradients.
    """
    model.eval()  # BN in eval mode for consistent behavior
    criterion = torch.nn.BCELoss()
    
    all_grads = []
    batch_count = 0
    
    for X, y in dataloader:
        if batch_count >= n_batches:
            break
        
        X = X.to(device).requires_grad_(True)
        y = y.to(device)
        
        if X.dim() > 2:
            X_flat = X.view(X.size(0), -1)
        else:
            X_flat = X
        
        output = model(X_flat).squeeze()
        loss = criterion(output, y)
        
        loss.backward()
        
        # |∂L/∂x_i| averaged over the batch
        grad = X.grad.abs().mean(dim=0).detach().cpu().numpy()
        if grad.ndim > 1:
            grad = grad.flatten()
        all_grads.append(grad)
        
        model.zero_grad()
        batch_count += 1
    
    # Average over all batches
    return np.mean(all_grads, axis=0)


def single_run(seed, args):
    """Train a model, compute saliency at each eval round, return correlations."""
    set_seed(seed)
    cipher = get_cipher(args.cipher)
    device = get_device(args)
    config = CIPHER_CONFIGS[args.cipher]
    source_round = config['source']
    eval_rounds = config['eval_rounds']

    # ── Train on source round ───────────────────────────────────────
    gen = CipherDataGenerator(
        cipher=args.cipher, n_rounds=source_round,
        delta_p=cipher.get_default_delta_p(), seed=seed
    )
    train_data = gen.generate_balanced_dataset(args.samples, negative_type='gohr')
    val_data = gen.generate_balanced_dataset(args.samples // 10, negative_type='gohr')

    input_dim = get_input_dim('R2_xor_diff', cipher.block_size)
    model = get_model('gohr_mlp', input_dim=input_dim)

    train_ds = CryptoDataset(train_data, 'R2_xor_diff', cipher.block_size)
    val_ds = CryptoDataset(val_data, 'R2_xor_diff', cipher.block_size)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        device=device, use_wandb=False
    )
    trainer.train(n_epochs=args.epochs, early_stopping_patience=5, save_best=False)

    src_metrics = evaluate_model(model, val_loader, device)
    print(f"  Source ({source_round}r): acc={src_metrics['accuracy']:.4f}")

    # ── Compute saliency at each evaluation round ──────────────────
    saliency_vectors = {}
    accuracies = {}

    for er in eval_rounds:
        egen = CipherDataGenerator(
            cipher=args.cipher, n_rounds=er,
            delta_p=cipher.get_default_delta_p(), seed=seed + er * 1000
        )
        edata = egen.generate_balanced_dataset(50000, negative_type='gohr')
        eds = CryptoDataset(edata, 'R2_xor_diff', cipher.block_size)
        eloader = DataLoader(eds, batch_size=args.batch_size)

        # Accuracy
        emetrics = evaluate_model(model, eloader, device)
        accuracies[er] = float(emetrics['accuracy'])

        # Saliency
        saliency = compute_gradient_saliency(model, eloader, device, n_batches=20)
        saliency_vectors[er] = saliency.tolist()

        print(f"  Round {er}: acc={emetrics['accuracy']:.4f}, "
              f"top-3 bits: {np.argsort(saliency)[-3:][::-1]}")

    # ── Compute pairwise Spearman correlations ─────────────────────
    correlations = {}
    for r1 in eval_rounds:
        for r2 in eval_rounds:
            if r1 >= r2:
                continue
            s1 = np.array(saliency_vectors[r1])
            s2 = np.array(saliency_vectors[r2])
            rho, p_value = spearmanr(s1, s2)
            key = f"{r1}r_vs_{r2}r"
            correlations[key] = {
                'spearman_rho': float(rho),
                'p_value': float(p_value),
            }
            print(f"  ρ(sal_{r1}r, sal_{r2}r) = {rho:+.3f} (p={p_value:.4f})")

    # Source vs lower rounds (the key comparison for anti-transfer)
    source_saliency = np.array(saliency_vectors[source_round])
    for er in eval_rounds:
        if er < source_round:
            other_saliency = np.array(saliency_vectors[er])
            rho, _ = spearmanr(source_saliency, other_saliency)
            print(f"  ★ ρ(source_{source_round}r, {er}r) = {rho:+.3f}")

    return {
        'source_accuracy': float(src_metrics['accuracy']),
        'saliency_vectors': saliency_vectors,
        'accuracies': accuracies,
        'correlations': correlations,
    }


def main():
    parser = argparse.ArgumentParser(
        description='E22: Cross-Round Saliency Comparison'
    )
    add_common_args(parser)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = f'./results/e22_cross_saliency'

    if args.cipher not in CIPHER_CONFIGS:
        print(f"No config for {args.cipher}. Available: {list(CIPHER_CONFIGS.keys())}")
        return

    config = CIPHER_CONFIGS[args.cipher]
    device = get_device(args)

    print("=" * 60)
    print(f"  E22: Cross-Round Saliency — {args.cipher.upper()}")
    print(f"  Source: {config['source']}r, Eval: {config['eval_rounds']}")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed + i for i in range(args.n_seeds)]
    all_runs = []

    for i, seed in enumerate(seeds):
        print(f"\n┌─ Seed {seed} ({i+1}/{len(seeds)}) ─────────────────┐")
        result = single_run(seed, args)
        all_runs.append(result)
        print(f"└─ Done ─────────────────────────────────────┘")

    # ── Aggregate correlations ──────────────────────────────────────
    eval_rounds = config['eval_rounds']
    source_round = config['source']
    n_rounds = len(eval_rounds)

    # Average correlation matrix across seeds
    avg_corr_matrix = np.zeros((n_rounds, n_rounds))
    for i, r1 in enumerate(eval_rounds):
        for j, r2 in enumerate(eval_rounds):
            if i == j:
                avg_corr_matrix[i, j] = 1.0
                continue
            key = f"{min(r1,r2)}r_vs_{max(r1,r2)}r"
            rhos = [run['correlations'].get(key, {}).get('spearman_rho', 0)
                    for run in all_runs]
            avg_corr_matrix[i, j] = np.mean(rhos)

    # Average saliency heatmap
    avg_saliency = {}
    for er in eval_rounds:
        sal_arrays = [np.array(run['saliency_vectors'][er]) for run in all_runs]
        avg_saliency[er] = np.mean(sal_arrays, axis=0)

    # ── Plot: saliency heatmap + correlation matrix ─────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Saliency heatmap: (bit_position × eval_round)
    block_size = get_cipher(args.cipher).block_size
    heatmap_data = np.array([avg_saliency[er] for er in eval_rounds])
    im1 = ax1.imshow(heatmap_data, aspect='auto', cmap='YlOrRd',
                      interpolation='nearest')
    ax1.set_yticks(range(n_rounds))
    ax1.set_yticklabels([f"{r}r" for r in eval_rounds])
    ax1.set_xlabel('Bit Position')
    ax1.set_ylabel('Evaluation Round')
    ax1.set_title('Average Gradient Saliency')
    fig.colorbar(im1, ax=ax1, label='|∂L/∂x_i|')

    # Correlation matrix
    im2 = ax2.imshow(avg_corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1,
                      interpolation='nearest')
    ax2.set_xticks(range(n_rounds))
    ax2.set_xticklabels([f"{r}r" for r in eval_rounds])
    ax2.set_yticks(range(n_rounds))
    ax2.set_yticklabels([f"{r}r" for r in eval_rounds])
    ax2.set_title('Spearman ρ (saliency)')

    # Annotate cells
    for i in range(n_rounds):
        for j in range(n_rounds):
            val = avg_corr_matrix[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax2.text(j, i, f"{val:.2f}", ha='center', va='center',
                     color=color, fontsize=8)

    fig.colorbar(im2, ax=ax2, label='Spearman ρ')

    plt.suptitle(
        f'{args.cipher.upper()} — Cross-Round Saliency '
        f'(trained on {source_round}r, {args.n_seeds} seeds)',
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(output_dir / f'e22_{args.cipher}_saliency.png', dpi=300)
    plt.close()

    # Save results
    save_results(
        {
            'correlation_matrix': avg_corr_matrix.tolist(),
            'eval_rounds': eval_rounds,
            'avg_saliency': {str(k): v.tolist() for k, v in avg_saliency.items()},
            'runs': all_runs,
        },
        str(output_dir),
        f'e22_{args.cipher}_results.json'
    )
    print(f"\n✓ Results saved to {output_dir}")


if __name__ == '__main__':
    main()
