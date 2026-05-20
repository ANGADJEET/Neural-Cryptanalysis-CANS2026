#!/usr/bin/env python3
"""
Publication Finalization Script
Runs all remaining experiments needed for publication:
  1. SIMON32 key recovery with more pairs (20000) and reduced rounds (5r instead of 6)
  2. Bootstrap confidence intervals for E01/E11 across all ciphers
  3. SPECK32 key recovery replication (5 seeds for statistics)

Usage:
  python scripts/run_publication_finals.py --device cuda
"""

import argparse
import sys
import time
import json
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.experiment_utils import set_seed, get_device


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: SIMON32 Key Recovery (Fixed)
# ─────────────────────────────────────────────────────────────────────────────
def run_simon_key_recovery(device='cuda', n_samples=500000, n_seeds=5, n_pairs=20000):
    """Key recovery for SIMON32 with proper settings."""
    import torch
    from ciphers import get_cipher
    from data.generator import CipherDataGenerator
    from data.dataloader import CryptoDataset, get_input_dim
    from data.representations import RepresentationFactory
    from models import get_model
    from training.trainer import Trainer
    from evaluation.metrics import evaluate_model
    from experiments.exp12_key_recovery import score_candidates, decrypt_one_round, WORD_MASK
    from torch.utils.data import DataLoader

    cipher = get_cipher('simon32')
    # Use 5r total (train on 4r) — SIMON32 at 4r gives ~100% accuracy
    n_rounds = 5
    reduced_rounds = n_rounds - 1

    print(f"\n{'═'*60}")
    print(f"  Key Recovery — SIMON32 ({n_rounds}r, train on {reduced_rounds}r)")
    print(f"  {n_seeds} seeds, {n_pairs} pairs per trial")
    print(f"{'═'*60}")

    seed_results = []
    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx
        set_seed(seed)
        print(f"\n  ┌─ Seed {seed} ({seed_idx+1}/{n_seeds}) ─────────────────┐")

        gen = CipherDataGenerator('simon32', n_rounds=reduced_rounds,
                                   delta_p=cipher.get_default_delta_p())
        train_data = gen.generate_balanced_dataset(n_samples, negative_type='gohr')
        val_data = gen.generate_balanced_dataset(n_samples // 10, negative_type='gohr')

        input_dim = get_input_dim('R2_xor_diff', cipher.block_size)
        model = get_model('gohr_mlp', input_dim=input_dim)

        train_ds = CryptoDataset(train_data, 'R2_xor_diff', cipher.block_size)
        val_ds = CryptoDataset(val_data, 'R2_xor_diff', cipher.block_size)
        train_loader = DataLoader(train_ds, batch_size=5000, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=5000)

        trainer = Trainer(model=model, train_loader=train_loader,
                          val_loader=val_loader, device=device, use_wandb=False)
        trainer.train(n_epochs=30, early_stopping_patience=5, save_best=False)

        dist_metrics = evaluate_model(model, val_loader, device)
        print(f"    Distinguisher ({reduced_rounds}r): {dist_metrics['accuracy']:.4f}")

        real_key = cipher.random_key()
        P = cipher.random_plaintexts(n_pairs)
        P_prime = (P ^ cipher.get_default_delta_p()).astype(P.dtype)
        C = cipher.encrypt(P, n_rounds, real_key)
        C_prime = cipher.encrypt(P_prime, n_rounds, real_key)

        expanded = cipher._expand_key(real_key, n_rounds)
        real_subkey = int(expanded[-1]) & WORD_MASK

        factory = RepresentationFactory(block_size=cipher.block_size)

        # Phase 1: low byte
        low_candidates = list(range(256))
        low_scores = score_candidates(model, 'simon32', factory, C, C_prime,
                                       low_candidates, device)
        low_ranked = sorted(low_scores.items(), key=lambda x: x[1], reverse=True)
        best_low = low_ranked[0][0]
        real_low = real_subkey & 0xFF
        low_rank = next((i for i, (k, _) in enumerate(low_ranked) if k == real_low), -1)

        # Phase 2: high byte
        high_candidates = [(h << 8) | best_low for h in range(256)]
        high_scores = score_candidates(model, 'simon32', factory, C, C_prime,
                                        high_candidates, device)
        high_ranked = sorted(high_scores.items(), key=lambda x: x[1], reverse=True)
        recovered = high_ranked[0][0]
        real_high = (real_subkey >> 8) & 0xFF
        high_rank = next((i for i, (k, _) in enumerate(high_ranked)
                         if ((k >> 8) & 0xFF) == real_high), -1)

        key_correct = (recovered == real_subkey)
        full_rank = next((i for i, (k, _) in enumerate(high_ranked) if k == real_subkey), -1)

        print(f"    Real: 0x{real_subkey:04x}, Recovered: 0x{recovered:04x} {'✓' if key_correct else '✗'}")
        print(f"    Low rank: {low_rank+1}/256, High rank: {high_rank+1}/256")

        seed_results.append({
            'key_correct': int(key_correct),
            'low_rank': low_rank + 1,
            'high_rank': high_rank + 1,
            'full_rank': full_rank + 1 if full_rank >= 0 else 257,
            'dist_acc': float(dist_metrics['accuracy']),
        })
        print(f"  └─ Done ─────────────────────────────────────┘")

    kc_rate = np.mean([r['key_correct'] for r in seed_results])
    avg_low = np.mean([r['low_rank'] for r in seed_results])
    avg_high = np.mean([r['high_rank'] for r in seed_results])

    print(f"\n  Summary:")
    print(f"    Key recovery rate: {kc_rate*100:.0f}%")
    print(f"    Mean low byte rank: {avg_low:.1f}/256")
    print(f"    Mean high byte rank: {avg_high:.1f}/256")

    return {
        'cipher': 'simon32',
        'n_rounds': n_rounds,
        'n_pairs': n_pairs,
        'key_correct_rate': float(kc_rate),
        'mean_low_rank': float(avg_low),
        'mean_high_rank': float(avg_high),
        'per_seed': seed_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: SPECK32 Key Recovery (5 seeds for statistics)
# ─────────────────────────────────────────────────────────────────────────────
def run_speck_key_recovery(device='cuda', n_samples=500000, n_seeds=5, n_pairs=20000):
    """Key recovery for SPECK32 with proper statistics."""
    import torch
    from ciphers import get_cipher
    from data.generator import CipherDataGenerator
    from data.dataloader import CryptoDataset, get_input_dim
    from data.representations import RepresentationFactory
    from models import get_model
    from training.trainer import Trainer
    from evaluation.metrics import evaluate_model
    from experiments.exp12_key_recovery import score_candidates, decrypt_one_round, WORD_MASK
    from torch.utils.data import DataLoader

    cipher = get_cipher('speck32')
    n_rounds = 5
    reduced_rounds = n_rounds - 1

    print(f"\n{'═'*60}")
    print(f"  Key Recovery — SPECK32 ({n_rounds}r, train on {reduced_rounds}r)")
    print(f"  {n_seeds} seeds, {n_pairs} pairs per trial")
    print(f"{'═'*60}")

    seed_results = []
    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx
        set_seed(seed)
        print(f"\n  ┌─ Seed {seed} ({seed_idx+1}/{n_seeds}) ─────────────────┐")

        gen = CipherDataGenerator('speck32', n_rounds=reduced_rounds,
                                   delta_p=cipher.get_default_delta_p())
        train_data = gen.generate_balanced_dataset(n_samples, negative_type='gohr')
        val_data = gen.generate_balanced_dataset(n_samples // 10, negative_type='gohr')

        input_dim = get_input_dim('R2_xor_diff', cipher.block_size)
        model = get_model('gohr_mlp', input_dim=input_dim)

        train_ds = CryptoDataset(train_data, 'R2_xor_diff', cipher.block_size)
        val_ds = CryptoDataset(val_data, 'R2_xor_diff', cipher.block_size)
        train_loader = DataLoader(train_ds, batch_size=5000, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=5000)

        trainer = Trainer(model=model, train_loader=train_loader,
                          val_loader=val_loader, device=device, use_wandb=False)
        trainer.train(n_epochs=30, early_stopping_patience=5, save_best=False)

        dist_metrics = evaluate_model(model, val_loader, device)
        print(f"    Distinguisher ({reduced_rounds}r): {dist_metrics['accuracy']:.4f}")

        real_key = cipher.random_key()
        P = cipher.random_plaintexts(n_pairs)
        P_prime = (P ^ cipher.get_default_delta_p()).astype(P.dtype)
        C = cipher.encrypt(P, n_rounds, real_key)
        C_prime = cipher.encrypt(P_prime, n_rounds, real_key)

        expanded = cipher._expand_key(real_key, n_rounds)
        real_subkey = int(expanded[-1]) & WORD_MASK

        factory = RepresentationFactory(block_size=cipher.block_size)

        # Phase 1: low byte
        low_candidates = list(range(256))
        low_scores = score_candidates(model, 'speck32', factory, C, C_prime,
                                       low_candidates, device)
        low_ranked = sorted(low_scores.items(), key=lambda x: x[1], reverse=True)
        best_low = low_ranked[0][0]
        real_low = real_subkey & 0xFF
        low_rank = next((i for i, (k, _) in enumerate(low_ranked) if k == real_low), -1)

        # Phase 2: high byte
        high_candidates = [(h << 8) | best_low for h in range(256)]
        high_scores = score_candidates(model, 'speck32', factory, C, C_prime,
                                        high_candidates, device)
        high_ranked = sorted(high_scores.items(), key=lambda x: x[1], reverse=True)
        recovered = high_ranked[0][0]
        real_high = (real_subkey >> 8) & 0xFF
        high_rank = next((i for i, (k, _) in enumerate(high_ranked)
                         if ((k >> 8) & 0xFF) == real_high), -1)

        key_correct = (recovered == real_subkey)
        full_rank = next((i for i, (k, _) in enumerate(high_ranked) if k == real_subkey), -1)

        print(f"    Real: 0x{real_subkey:04x}, Recovered: 0x{recovered:04x} {'✓' if key_correct else '✗'}")
        print(f"    Low rank: {low_rank+1}/256, High rank: {high_rank+1}/256")

        seed_results.append({
            'key_correct': int(key_correct),
            'low_rank': low_rank + 1,
            'high_rank': high_rank + 1,
            'full_rank': full_rank + 1 if full_rank >= 0 else 257,
            'dist_acc': float(dist_metrics['accuracy']),
        })
        print(f"  └─ Done ─────────────────────────────────────┘")

    kc_rate = np.mean([r['key_correct'] for r in seed_results])
    avg_low = np.mean([r['low_rank'] for r in seed_results])
    avg_high = np.mean([r['high_rank'] for r in seed_results])

    print(f"\n  Summary:")
    print(f"    Key recovery rate: {kc_rate*100:.0f}%")
    print(f"    Mean low byte rank: {avg_low:.1f}/256")
    print(f"    Mean high byte rank: {avg_high:.1f}/256")

    return {
        'cipher': 'speck32',
        'n_rounds': n_rounds,
        'n_pairs': n_pairs,
        'key_correct_rate': float(kc_rate),
        'mean_low_rank': float(avg_low),
        'mean_high_rank': float(avg_high),
        'per_seed': seed_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: Bootstrap CIs for E01/E11
# ─────────────────────────────────────────────────────────────────────────────
def run_bootstrap_cis():
    """Compute bootstrap 95% CIs for all existing results."""
    print(f"\n{'═'*60}")
    print(f"  Bootstrap Confidence Intervals")
    print(f"{'═'*60}")

    results = {}

    # Load existing E01 data
    try:
        with open('results/multi_cipher/simon32_results.json') as f:
            simon = json.load(f)
        with open('results/multi_cipher/present_results.json') as f:
            present = json.load(f)
    except FileNotFoundError:
        print("  ⚠ Missing multi_cipher results, skipping")
        return {}

    n_boot = 10000

    for cipher_name, data in [('simon32', simon), ('present', present)]:
        e01 = data.get('e01', {})
        ci_data = {}
        for r, vals_dict in e01.items():
            values = vals_dict.get('values', [])
            if len(values) >= 2:
                arr = np.array(values)
                boot_means = [np.mean(np.random.choice(arr, len(arr), replace=True))
                              for _ in range(n_boot)]
                ci_low = np.percentile(boot_means, 2.5)
                ci_high = np.percentile(boot_means, 97.5)
                ci_data[r] = {
                    'mean': float(np.mean(arr)),
                    'ci_low': float(ci_low),
                    'ci_high': float(ci_high),
                    'ci_width': float(ci_high - ci_low),
                }
                print(f"  {cipher_name} E01 R{r}: {np.mean(arr):.4f} [{ci_low:.4f}, {ci_high:.4f}]")

        results[f'{cipher_name}_e01_ci'] = ci_data

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Publication Finalization')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--task', default='all',
                        choices=['all', 'simon_kr', 'speck_kr', 'bootstrap'])
    args = parser.parse_args()

    output_dir = Path('./results/publication_finals')
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    t0 = time.time()

    if args.task in ('all', 'speck_kr'):
        r = run_speck_key_recovery(args.device)
        all_results['speck32_key_recovery'] = r
        with open(output_dir / 'speck32_key_recovery.json', 'w') as f:
            json.dump(r, f, indent=2)

    if args.task in ('all', 'simon_kr'):
        r = run_simon_key_recovery(args.device)
        all_results['simon32_key_recovery'] = r
        with open(output_dir / 'simon32_key_recovery.json', 'w') as f:
            json.dump(r, f, indent=2)

    if args.task in ('all', 'bootstrap'):
        r = run_bootstrap_cis()
        all_results['bootstrap_cis'] = r
        with open(output_dir / 'bootstrap_cis.json', 'w') as f:
            json.dump(r, f, indent=2)

    with open(output_dir / 'all_finals.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'═'*60}")
    print(f"  ✓ Publication finals complete ({elapsed:.0f}s)")
    print(f"  Results saved to {output_dir}")
    print(f"{'═'*60}")


if __name__ == '__main__':
    main()
