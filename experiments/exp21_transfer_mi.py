#!/usr/bin/env python3
"""
E21: Information-Theoretic Characterization of Anti-Transfer

Key question: When a model trained at round R shows below-chance accuracy
on round R' data (anti-transfer), does it have ZERO mutual information
with the label, or does it extract REAL structure that is anti-correlated
with the decision boundary?

Design:
  1. Train a distinguisher at source_round (e.g., SPECK 5r)
  2. For each target_round in {source-2, ..., source+2}:
     a. Forward pass test data → extract penultimate-layer features
     b. Estimate I(features; label) using MINE
     c. Also record classification accuracy
  3. If MI > 0 at a target_round where accuracy < 50%:
     → The model extracts REAL cipher structure that is anti-correlated
        with the decision boundary. This is the strongest possible
        statement about anti-transfer.

This is NOT a synthetic experiment. Every data point comes from the actual
cipher encryption, and every MI value is estimated from real features.

Usage:
  python experiments/exp21_transfer_mi.py --cipher speck32
  python experiments/exp21_transfer_mi.py --cipher simon32
  python experiments/exp21_transfer_mi.py --cipher present
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

from ciphers import get_cipher
from data.generator import CipherDataGenerator
from data.dataloader import CryptoDataset, get_input_dim
from models import get_model
from models.mine import MutualInfoEstimator
from training.trainer import Trainer
from evaluation.metrics import evaluate_model
from experiments.experiment_utils import (
    set_seed, add_common_args, save_results, get_device
)


# Round configs per cipher
CIPHER_CONFIGS = {
    'speck32': {'source': 5, 'targets': [3, 4, 5, 6, 7]},
    'simon32': {'source': 6, 'targets': [4, 5, 6, 7, 8]},
    'present': {'source': 4, 'targets': [2, 3, 4, 5, 6]},
}


def extract_penultimate_features(model, dataloader, device):
    """Extract penultimate-layer activations from the model.
    
    For GohrMLP: the penultimate layer is the 32-dim hidden layer
    (last ReLU output before the 1-dim sigmoid output).
    
    We hook into the second-to-last layer to capture features.
    """
    model.eval()
    features_list = []
    labels_list = []
    
    # Find the penultimate layer (last hidden layer before output)
    # GohrMLP has: [Linear, BN, ReLU] × 6 → Linear → Sigmoid
    # We want the output after the last ReLU (before final Linear+Sigmoid)
    layers = list(model.network.children())
    # The last 2 layers are Linear(32→1) and Sigmoid
    # Everything before that is the feature extractor
    feature_extractor = torch.nn.Sequential(*layers[:-2])
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            if X.dim() > 2:
                X = X.view(X.size(0), -1)
            feats = feature_extractor(X)
            features_list.append(feats.cpu().numpy())
            labels_list.append(y.numpy())
    
    return np.concatenate(features_list), np.concatenate(labels_list)


def single_run(seed, args):
    """Run one seed: train, then extract features & estimate MI at each target round."""
    set_seed(seed)
    cipher = get_cipher(args.cipher)
    device = get_device(args)
    config = CIPHER_CONFIGS[args.cipher]
    source_round = config['source']
    target_rounds = config['targets']

    # ── Train distinguisher on source round ─────────────────────────
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

    # ── Evaluate at each target round ───────────────────────────────
    results = {
        'source_accuracy': float(src_metrics['accuracy']),
        'per_target': {},
    }

    for tr in target_rounds:
        print(f"  Target {tr}r: ", end='', flush=True)

        # Generate data at target round
        tgen = CipherDataGenerator(
            cipher=args.cipher, n_rounds=tr,
            delta_p=cipher.get_default_delta_p(), seed=seed + tr * 1000
        )
        # Use 50K samples for MI estimation (enough for MINE, not too slow)
        tdata = tgen.generate_balanced_dataset(50000, negative_type='gohr')
        tds = CryptoDataset(tdata, 'R2_xor_diff', cipher.block_size)
        tloader = DataLoader(tds, batch_size=args.batch_size)

        # Classification accuracy
        tmetrics = evaluate_model(model, tloader, device)
        acc = float(tmetrics['accuracy'])

        # Extract penultimate features
        features, labels = extract_penultimate_features(model, tloader, device)
        feat_dim = features.shape[1]

        # Estimate MI between features and labels
        mi_estimator = MutualInfoEstimator(
            input_dim=feat_dim,
            hidden_dims=[128, 64],  # Smaller network for lower-dim features
            device=device,
        )
        mi = mi_estimator.estimate(
            features,
            labels.reshape(-1, 1) if labels.ndim == 1 else labels,
            n_epochs=args.mine_epochs,
            batch_size=5000,
            verbose=False,
        )
        # Clamp to 0 — MI cannot be negative (any negative estimate is noise)
        mi = max(0.0, mi)

        # Classify the finding
        if acc < 0.49 and mi > 0.01:
            finding = "ANTI-CORRELATED (real structure, inverted boundary)"
        elif acc < 0.49 and mi <= 0.01:
            finding = "NO INFORMATION"
        elif acc > 0.51 and mi > 0.01:
            finding = "POSITIVE TRANSFER"
        else:
            finding = "NEUTRAL"

        print(f"acc={acc:.4f}, MI={mi:.4f} nats → {finding}")

        results['per_target'][str(tr)] = {
            'accuracy': acc,
            'mi_nats': float(mi),
            'feature_dim': feat_dim,
            'finding': finding,
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description='E21: Transfer MI Characterization'
    )
    add_common_args(parser)
    parser.add_argument('--mine-epochs', type=int, default=300,
                        help='MINE training epochs for feature MI estimation')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = f'./results/e21_transfer_mi'

    if args.cipher not in CIPHER_CONFIGS:
        print(f"No config for {args.cipher}. Available: {list(CIPHER_CONFIGS.keys())}")
        return

    device = get_device(args)
    config = CIPHER_CONFIGS[args.cipher]

    print("=" * 65)
    print(f"  E21: Transfer MI Characterization — {args.cipher.upper()}")
    print(f"  Source: {config['source']}r, Targets: {config['targets']}")
    print(f"  MINE epochs: {args.mine_epochs}")
    print("=" * 65)

    # ── MINE calibration check ──────────────────────────────────────
    print(f"\n{'━' * 50}")
    print("  MINE calibration (ρ=0.7 Gaussians)")
    print(f"{'━' * 50}")
    calibrator = MutualInfoEstimator(input_dim=1, device=device)
    cal = calibrator.validate_calibration(
        rho=0.7, n_epochs=args.mine_epochs, verbose=True
    )
    if not cal['calibrated']:
        print("  ⚠ MINE calibration failed. Increase --mine-epochs.")
        # Don't abort — let the user decide
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed + i for i in range(args.n_seeds)]
    all_runs = []

    for i, seed in enumerate(seeds):
        print(f"\n┌─ Seed {seed} ({i+1}/{len(seeds)}) ─────────────────┐")
        result = single_run(seed, args)
        all_runs.append(result)
        print(f"└─ Done ─────────────────────────────────────┘")

    # ── Aggregate ───────────────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print(f"  {'Round':<8} {'Accuracy':<15} {'MI (nats)':<15} {'Finding'}")
    print(f"{'─' * 65}")

    agg_results = {'calibration': cal, 'per_target': {}}
    for tr in config['targets']:
        key = str(tr)
        accs = [r['per_target'][key]['accuracy'] for r in all_runs]
        mis = [r['per_target'][key]['mi_nats'] for r in all_runs]

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        mean_mi = float(np.mean(mis))
        std_mi = float(np.std(mis))

        if mean_acc < 0.49 and mean_mi > 0.01:
            finding = "ANTI-CORRELATED"
        elif mean_acc > 0.51 and mean_mi > 0.01:
            finding = "POSITIVE"
        else:
            finding = "NEUTRAL/NONE"

        print(f"  {tr}r{'':<5} {mean_acc:.4f}±{std_acc:.4f}   "
              f"{mean_mi:.4f}±{std_mi:.4f}   {finding}")

        agg_results['per_target'][key] = {
            'acc_mean': mean_acc, 'acc_std': std_acc,
            'mi_mean': mean_mi, 'mi_std': std_mi,
            'accs': accs, 'mis': mis,
            'finding': finding,
        }

    print(f"{'═' * 65}")

    # ── Plot ────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    rounds = config['targets']
    acc_means = [agg_results['per_target'][str(r)]['acc_mean'] for r in rounds]
    acc_stds = [agg_results['per_target'][str(r)]['acc_std'] for r in rounds]
    mi_means = [agg_results['per_target'][str(r)]['mi_mean'] for r in rounds]
    mi_stds = [agg_results['per_target'][str(r)]['mi_std'] for r in rounds]

    # Color bars by finding type
    colors_acc = []
    for r in rounds:
        f = agg_results['per_target'][str(r)]['finding']
        if f == 'ANTI-CORRELATED':
            colors_acc.append('#e74c3c')
        elif f == 'POSITIVE':
            colors_acc.append('#2ecc71')
        else:
            colors_acc.append('#95a5a6')

    ax1.bar(rounds, acc_means, yerr=acc_stds, capsize=5,
            color=colors_acc, edgecolor='black', alpha=0.85)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.4)
    ax1.set_xlabel('Evaluation Round')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Classification Accuracy')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(rounds, mi_means, yerr=mi_stds, capsize=5,
            color='steelblue', edgecolor='black', alpha=0.85)
    ax2.axhline(y=0.0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Evaluation Round')
    ax2.set_ylabel('MI (nats)')
    ax2.set_title('I(features; label)')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle(
        f'{args.cipher.upper()} — Transfer MI Analysis '
        f'(trained on {config["source"]}r, {args.n_seeds} seeds)',
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(output_dir / f'e21_{args.cipher}_transfer_mi.png', dpi=300)
    plt.close()

    save_results(
        agg_results,
        str(output_dir),
        f'e21_{args.cipher}_results.json'
    )
    print(f"\n✓ Results saved to {output_dir}")


if __name__ == '__main__':
    main()
