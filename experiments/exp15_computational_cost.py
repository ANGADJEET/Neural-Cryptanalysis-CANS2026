#!/usr/bin/env python
"""
E15: Computational Cost Benchmarking

Profile all model architectures: parameter count, training time,
inference throughput. Outputs a LaTeX-ready comparison table.

Usage:
    python experiments/exp15_computational_cost.py --cipher speck32 --rounds 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
import time as _time

from ciphers import get_cipher
from data.generator import CipherDataGenerator
from data.dataloader import CryptoDataset, get_input_dim
from data.representations import RepresentationFactory
from models import get_model
from training.trainer import Trainer
from evaluation.metrics import evaluate_model
from experiments.experiment_utils import (
    set_seed, add_common_args, save_results, get_device
)

ALL_MODELS = ['gohr_mlp', 'mlp', 'cnn', 'residual_cnn', 'lstm', 'gru']


def benchmark_model(model_name, cipher, args, device):
    """Benchmark a single model architecture."""
    gen = CipherDataGenerator(
        cipher=args.cipher, n_rounds=args.rounds,
        delta_p=cipher.get_default_delta_p()
    )
    train_data = gen.generate_balanced_dataset(args.samples)
    val_data = gen.generate_balanced_dataset(args.samples // 10)
    test_data = gen.generate_balanced_dataset(args.samples // 10)

    input_dim = get_input_dim('R2_xor_diff', cipher.block_size)
    is_rnn = model_name in ('lstm', 'gru')

    model = get_model(model_name, input_dim=input_dim)
    n_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if is_rnn:
        factory = RepresentationFactory(block_size=cipher.block_size)
        X_train = factory.get_representation('R2_xor_diff',
                                              train_data['C'], train_data['C_prime'])
        X_val = factory.get_representation('R2_xor_diff',
                                            val_data['C'], val_data['C_prime'])
        X_test = factory.get_representation('R2_xor_diff',
                                             test_data['C'], test_data['C_prime'])

        from torch.utils.data import DataLoader, TensorDataset
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train).float().unsqueeze(1),
                          torch.from_numpy(train_data['labels']).float()),
            batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val).float().unsqueeze(1),
                          torch.from_numpy(val_data['labels']).float()),
            batch_size=args.batch_size)
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_test).float().unsqueeze(1),
                          torch.from_numpy(test_data['labels']).float()),
            batch_size=args.batch_size)
    else:
        train_ds = CryptoDataset(train_data, 'R2_xor_diff', cipher.block_size)
        val_ds = CryptoDataset(val_data, 'R2_xor_diff', cipher.block_size)
        test_ds = CryptoDataset(test_data, 'R2_xor_diff', cipher.block_size)

        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Training time
    trainer = Trainer(model=model, train_loader=train_loader,
                      val_loader=val_loader, device=device, use_wandb=False)
    t0 = _time.time()
    trainer.train(n_epochs=args.epochs, early_stopping_patience=5, save_best=False)
    train_time = _time.time() - t0

    # Evaluation
    metrics = evaluate_model(model, test_loader, device)

    # Inference throughput (samples/sec)
    model.eval()
    n_infer = 0
    t0 = _time.time()
    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(device)
            _ = model(X)
            n_infer += X.shape[0]
    infer_time = _time.time() - t0
    throughput = n_infer / max(infer_time, 1e-6)

    # Memory estimate (MB)
    mem_mb = n_params * 4 / (1024 * 1024)  # float32

    return {
        'n_params': n_params,
        'trainable_params': trainable_params,
        'accuracy': float(metrics['accuracy']),
        'train_time_sec': round(train_time, 2),
        'infer_throughput': round(throughput),
        'model_size_mb': round(mem_mb, 3),
    }


def main():
    parser = argparse.ArgumentParser(description='E15: Computational Cost')
    add_common_args(parser)
    parser.add_argument('--rounds', type=int, default=5)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = './results/e15_cost'

    print("=" * 60)
    print("  E15: Computational Cost Benchmarking")
    print("=" * 60)

    cipher = get_cipher(args.cipher)
    device = get_device(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    results = {}

    for model_name in ALL_MODELS:
        print(f"\n  Benchmarking {model_name}...")
        try:
            bench = benchmark_model(model_name, cipher, args, device)
            results[model_name] = bench
            print(f"    params={bench['n_params']:,}, "
                  f"acc={bench['accuracy']:.4f}, "
                  f"train={bench['train_time_sec']:.1f}s, "
                  f"throughput={bench['infer_throughput']:,} samples/s")
        except Exception as e:
            print(f"    ERROR: {e}")
            results[model_name] = {'error': str(e)}

    # Print table
    print(f"\n{'═' * 80}")
    print(f"  {'Model':<12} {'Params':>10} {'Acc':>8} "
          f"{'Train(s)':>10} {'Infer(s/s)':>12} {'Size(MB)':>10}")
    print(f"{'─' * 80}")
    for name in ALL_MODELS:
        r = results.get(name, {})
        if 'error' in r:
            print(f"  {name:<12} {'ERROR':>10}")
            continue
        print(f"  {name:<12} {r['n_params']:>10,} {r['accuracy']:>8.4f} "
              f"{r['train_time_sec']:>10.1f} {r['infer_throughput']:>12,} "
              f"{r['model_size_mb']:>10.3f}")
    print(f"{'═' * 80}")

    # LaTeX table
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{Model benchmarks — {args.cipher.upper()} ({args.rounds}r)}}",
        r"\begin{tabular}{lrrlrl}",
        r"\toprule",
        r"Model & Params & Accuracy & Train (s) & Throughput (s/s) & Size (MB) \\",
        r"\midrule",
    ]
    for name in ALL_MODELS:
        r = results.get(name, {})
        if 'error' in r:
            continue
        latex_lines.append(
            f"{name} & {r['n_params']:,} & {r['accuracy']:.4f} & "
            f"{r['train_time_sec']:.1f} & {r['infer_throughput']:,} & "
            f"{r['model_size_mb']:.3f} \\\\"
        )
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    latex_table = '\n'.join(latex_lines)

    with open(output_dir / f'e15_{args.cipher}_table.tex', 'w') as f:
        f.write(latex_table)

    save_results(results, str(output_dir),
                 f'e15_{args.cipher}_results.json')

    print(f"\n  LaTeX table saved to {output_dir / f'e15_{args.cipher}_table.tex'}")
    print(f"✓ Done")


if __name__ == '__main__':
    main()
