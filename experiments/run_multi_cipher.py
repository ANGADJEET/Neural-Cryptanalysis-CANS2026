#!/usr/bin/env python3
"""
Multi-Cipher Experiment Suite
Run the core publication experiments for ALL 3 ciphers: SPECK32, SIMON32, PRESENT.

Covers:
  - E01: Baseline accuracy vs rounds
  - E09: Anti-transfer (cross-cipher + cross-round with t-tests)
  - E11: Neural vs Classical (with fixed bit-bias distinguisher)
  - E12: Key Recovery demo

Usage:
  python experiments/run_multi_cipher.py --task all
  python experiments/run_multi_cipher.py --task e01
  python experiments/run_multi_cipher.py --task e09
  python experiments/run_multi_cipher.py --task e11
  python experiments/run_multi_cipher.py --task e12
"""

import argparse
import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.experiment_utils import set_seed, get_device

# Cipher configs: round ranges to sweep
CIPHER_CONFIGS = {
    'speck32': {
        'baseline_rounds': list(range(3, 10)),
        'e11_rounds': list(range(3, 9)),
        'transfer_source_rounds': 5,
        'transfer_target_rounds': [3, 4, 6, 7, 8],
        'key_recovery_rounds': 5,
    },
    'simon32': {
        'baseline_rounds': list(range(4, 12)),
        'e11_rounds': list(range(4, 10)),
        'transfer_source_rounds': 6,
        'transfer_target_rounds': [4, 5, 7, 8, 9],
        'key_recovery_rounds': 6,
    },
    'present': {
        'baseline_rounds': list(range(2, 8)),
        'e11_rounds': list(range(2, 7)),
        'transfer_source_rounds': 4,
        'transfer_target_rounds': [2, 3, 5, 6],
        'key_recovery_rounds': 4,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# E01: Baseline Accuracy vs Rounds
# ─────────────────────────────────────────────────────────────────────────────
def run_e01_for_cipher(cipher_name, device='cuda', n_samples=500000, n_seeds=3, n_epochs=30):
    """E01 baseline for a single cipher."""
    import torch
    from ciphers import get_cipher
    from data.generator import CipherDataGenerator
    from data.dataloader import CryptoDataset, get_input_dim
    from models import get_model
    from training.trainer import Trainer
    from evaluation.metrics import evaluate_model
    from torch.utils.data import DataLoader

    cipher = get_cipher(cipher_name)
    rounds_list = CIPHER_CONFIGS[cipher_name]['baseline_rounds']

    print(f"\n{'═'*60}")
    print(f"  E01: Baseline — {cipher_name.upper()}")
    print(f"  Rounds: {rounds_list}, Seeds: {n_seeds}")
    print(f"{'═'*60}")

    all_results = {}

    for n_rounds in rounds_list:
        seed_accs = []
        for seed_idx in range(n_seeds):
            seed = 42 + seed_idx
            set_seed(seed)

            gen = CipherDataGenerator(
                cipher_name, n_rounds=n_rounds,
                delta_p=cipher.get_default_delta_p()
            )
            train_data = gen.generate_balanced_dataset(n_samples, negative_type='gohr')
            val_data = gen.generate_balanced_dataset(n_samples // 10, negative_type='gohr')
            test_data = gen.generate_balanced_dataset(n_samples // 10, negative_type='gohr')

            input_dim = get_input_dim('R2_xor_diff', cipher.block_size)
            model = get_model('gohr_mlp', input_dim=input_dim)

            train_ds = CryptoDataset(train_data, 'R2_xor_diff', cipher.block_size)
            val_ds = CryptoDataset(val_data, 'R2_xor_diff', cipher.block_size)
            test_ds = CryptoDataset(test_data, 'R2_xor_diff', cipher.block_size)

            train_loader = DataLoader(train_ds, batch_size=5000, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=5000)
            test_loader = DataLoader(test_ds, batch_size=5000)

            trainer = Trainer(model=model, train_loader=train_loader,
                              val_loader=val_loader, device=device, use_wandb=False)
            trainer.train(n_epochs=n_epochs, early_stopping_patience=5, save_best=False)

            metrics = evaluate_model(model, test_loader, device)
            seed_accs.append(float(metrics['accuracy']))

        mean_acc = np.mean(seed_accs)
        std_acc = np.std(seed_accs)
        all_results[str(n_rounds)] = {
            'mean': float(mean_acc),
            'std': float(std_acc),
            'values': seed_accs,
        }
        print(f"  Round {n_rounds}: {mean_acc:.4f} ± {std_acc:.4f}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# E09: Anti-Transfer with t-tests
# ─────────────────────────────────────────────────────────────────────────────
def run_e09_for_cipher(cipher_name, device='cuda', n_samples=500000, n_seeds=5):
    """E09 transfer for a single source cipher."""
    import torch
    from scipy import stats
    from ciphers import get_cipher
    from data.generator import CipherDataGenerator
    from data.dataloader import CryptoDataset, get_input_dim
    from models import get_model
    from training.trainer import Trainer
    from evaluation.metrics import evaluate_model
    from torch.utils.data import DataLoader

    cipher = get_cipher(cipher_name)
    config = CIPHER_CONFIGS[cipher_name]
    source_rounds = config['transfer_source_rounds']
    target_rounds = config['transfer_target_rounds']

    # Determine which other ciphers have the same block size
    target_ciphers = []
    for tc in ['speck32', 'simon32', 'present']:
        if tc != cipher_name:
            try:
                tc_obj = get_cipher(tc)
                if tc_obj.block_size == cipher.block_size:
                    target_ciphers.append(tc)
            except Exception:
                pass

    print(f"\n{'═'*60}")
    print(f"  E09: Transfer — {cipher_name.upper()} ({source_rounds}r)")
    print(f"  Cross-round targets: {target_rounds}")
    print(f"  Cross-cipher targets: {target_ciphers}")
    print(f"{'═'*60}")

    source_accs = []
    cross_cipher_results = {tc: [] for tc in target_ciphers}
    cross_round_results = {r: [] for r in target_rounds}

    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx
        set_seed(seed)
        print(f"\n  ┌─ Seed {seed} ({seed_idx+1}/{n_seeds}) ─────────────────┐")

        # Train on source
        gen = CipherDataGenerator(
            cipher_name, n_rounds=source_rounds,
            delta_p=cipher.get_default_delta_p()
        )
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

        src_metrics = evaluate_model(model, val_loader, device)
        source_accs.append(float(src_metrics['accuracy']))
        print(f"    Source ({cipher_name}, {source_rounds}r): {src_metrics['accuracy']:.4f}")

        # Cross-cipher
        for tc in target_ciphers:
            tc_cipher = get_cipher(tc)
            tgen = CipherDataGenerator(
                tc, n_rounds=source_rounds,
                delta_p=tc_cipher.get_default_delta_p()
            )
            tdata = tgen.generate_balanced_dataset(n_samples // 5, negative_type='gohr')
            tds = CryptoDataset(tdata, 'R2_xor_diff', tc_cipher.block_size)
            tloader = DataLoader(tds, batch_size=5000)
            tm = evaluate_model(model, tloader, device)
            cross_cipher_results[tc].append(float(tm['accuracy']))
            print(f"    → {tc} {source_rounds}r: {tm['accuracy']:.4f}")

        # Cross-round
        for tr in target_rounds:
            rgen = CipherDataGenerator(
                cipher_name, n_rounds=tr,
                delta_p=cipher.get_default_delta_p()
            )
            rdata = rgen.generate_balanced_dataset(n_samples // 10, negative_type='gohr')
            rds = CryptoDataset(rdata, 'R2_xor_diff', cipher.block_size)
            rloader = DataLoader(rds, batch_size=5000)
            rm = evaluate_model(model, rloader, device)
            cross_round_results[tr].append(float(rm['accuracy']))
            print(f"    → {cipher_name} {tr}r: {rm['accuracy']:.4f}")

        print(f"  └─ Done ─────────────────────────────────────┘")

    # Statistical analysis
    print(f"\n{'═'*60}")
    print(f"  Statistical Analysis — {cipher_name.upper()}")
    print(f"{'═'*60}")

    results = {
        'source_cipher': cipher_name,
        'source_rounds': source_rounds,
        'source_accs': source_accs,
    }

    for tc in target_ciphers:
        arr = np.array(cross_cipher_results[tc])
        mean = arr.mean()
        std = arr.std()
        t, p = stats.ttest_1samp(arr, 0.5)
        direction = "anti-transfer" if mean < 0.5 else "positive"
        sig = "★" if p < 0.05 else "○"
        print(f"\n  Cross-Cipher ({cipher_name} → {tc}):")
        print(f"    Acc: {mean:.4f}±{std:.4f}, t={t:.3f}, p={p:.4f} {sig} {direction}")
        results[f'cross_cipher_{tc}'] = {
            'accs': cross_cipher_results[tc],
            'mean': float(mean), 'std': float(std),
            't_stat': float(t), 'p_value': float(p),
            'direction': direction,
        }

    print(f"\n  Cross-Round ({cipher_name}, trained on {source_rounds}r):")
    for tr in sorted(cross_round_results.keys()):
        arr = np.array(cross_round_results[tr])
        mean = arr.mean()
        std = arr.std()
        t, p = stats.ttest_1samp(arr, 0.5)
        sig = "★" if p < 0.05 else "○"
        d = "↓" if mean < 0.5 else "↑"
        print(f"    {tr}r: {mean:.4f}±{std:.4f}  t={t:.3f} p={p:.4f} {sig} {d}")
        results[f'cross_round_{tr}'] = {
            'accs': cross_round_results[tr],
            'mean': float(mean), 'std': float(std),
            't_stat': float(t), 'p_value': float(p),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# E11: Neural vs Classical (Fixed)
# ─────────────────────────────────────────────────────────────────────────────
def run_e11_for_cipher(cipher_name, device='cuda', n_samples=500000):
    """E11 with fixed classical distinguisher for a single cipher."""
    import torch
    from ciphers import get_cipher
    from data.generator import CipherDataGenerator
    from data.dataloader import CryptoDataset, get_input_dim
    from data.statistics import compute_classical_distinguisher_accuracy
    from models import get_model
    from training.trainer import Trainer
    from evaluation.metrics import evaluate_model
    from torch.utils.data import DataLoader

    cipher = get_cipher(cipher_name)
    rounds_list = CIPHER_CONFIGS[cipher_name]['e11_rounds']

    print(f"\n{'═'*60}")
    print(f"  E11: Neural vs Classical — {cipher_name.upper()}")
    print(f"  Rounds: {rounds_list}")
    print(f"{'═'*60}")

    results = {}

    for n_rounds in rounds_list:
        set_seed(42)
        print(f"\n  --- Round {n_rounds} ---")

        # Classical
        classical_acc = compute_classical_distinguisher_accuracy(
            cipher=cipher, diff_in=cipher.get_default_delta_p(),
            n_rounds=n_rounds, n_samples=n_samples, n_keys=5
        )
        print(f"    Classical (bit-bias): {classical_acc:.4f}")

        # Neural
        gen = CipherDataGenerator(cipher_name, n_rounds=n_rounds,
                                   delta_p=cipher.get_default_delta_p())
        train_data = gen.generate_balanced_dataset(n_samples, negative_type='gohr')
        val_data = gen.generate_balanced_dataset(n_samples // 10, negative_type='gohr')
        test_data = gen.generate_balanced_dataset(n_samples // 10, negative_type='gohr')

        input_dim = get_input_dim('R2_xor_diff', cipher.block_size)
        model = get_model('gohr_mlp', input_dim=input_dim)

        train_ds = CryptoDataset(train_data, 'R2_xor_diff', cipher.block_size)
        val_ds = CryptoDataset(val_data, 'R2_xor_diff', cipher.block_size)
        test_ds = CryptoDataset(test_data, 'R2_xor_diff', cipher.block_size)

        train_loader = DataLoader(train_ds, batch_size=5000, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=5000)
        test_loader = DataLoader(test_ds, batch_size=5000)

        trainer = Trainer(model=model, train_loader=train_loader,
                          val_loader=val_loader, device=device, use_wandb=False)
        trainer.train(n_epochs=30, early_stopping_patience=5, save_best=False)

        metrics = evaluate_model(model, test_loader, device)
        neural_acc = float(metrics['accuracy'])
        print(f"    Neural (GohrMLP):    {neural_acc:.4f}")

        results[str(n_rounds)] = {
            'classical': classical_acc,
            'neural': neural_acc,
            'gap': round(neural_acc - classical_acc, 4),
        }

    # Summary
    print(f"\n{'═'*55}")
    print(f"  {'Round':>5}  {'Classical':>10}  {'Neural':>10}  {'Gap':>8}")
    print(f"{'─'*55}")
    for r in rounds_list:
        d = results[str(r)]
        print(f"  {r:>5}  {d['classical']:>10.4f}  {d['neural']:>10.4f}  {d['gap']:>+8.4f}")
    print(f"{'═'*55}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# E12: Key Recovery
# ─────────────────────────────────────────────────────────────────────────────
def run_e12_for_cipher(cipher_name, device='cuda', n_samples=500000, n_seeds=3):
    """E12 key recovery for a single cipher."""
    import torch
    from ciphers import get_cipher
    from data.generator import CipherDataGenerator
    from data.dataloader import CryptoDataset, get_input_dim
    from data.representations import RepresentationFactory
    from models import get_model
    from training.trainer import Trainer
    from evaluation.metrics import evaluate_model
    from torch.utils.data import DataLoader

    # Import the scoring and decryption from exp12
    from experiments.exp12_key_recovery import (
        score_candidates, decrypt_one_round, WORD_MASK
    )

    cipher = get_cipher(cipher_name)
    config = CIPHER_CONFIGS[cipher_name]
    n_rounds = config['key_recovery_rounds']
    reduced_rounds = n_rounds - 1

    # Key recovery only works well for speck32 and simon32 (32-bit block, 16-bit words)
    if cipher.block_size != 32:
        print(f"\n  ⚠ Skipping E12 for {cipher_name} (block_size={cipher.block_size} ≠ 32)")
        return {'skipped': True, 'reason': f'block_size={cipher.block_size}'}

    print(f"\n{'═'*60}")
    print(f"  E12: Key Recovery — {cipher_name.upper()} ({n_rounds}r)")
    print(f"  Training on {reduced_rounds}r, recovering last round key")
    print(f"{'═'*60}")

    seed_results = []

    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx
        set_seed(seed)
        print(f"\n  ┌─ Seed {seed} ({seed_idx+1}/{n_seeds}) ─────────────────┐")

        # Train distinguisher on reduced rounds
        gen = CipherDataGenerator(cipher_name, n_rounds=reduced_rounds,
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

        # Key recovery
        real_key = cipher.random_key()
        P = cipher.random_plaintexts(5000)
        P_prime = (P ^ cipher.get_default_delta_p()).astype(P.dtype)
        C = cipher.encrypt(P, n_rounds, real_key)
        C_prime = cipher.encrypt(P_prime, n_rounds, real_key)

        try:
            expanded = cipher._expand_key(real_key, n_rounds)
            real_subkey = int(expanded[-1]) & WORD_MASK
        except Exception:
            real_subkey = -1

        factory = RepresentationFactory(block_size=cipher.block_size)

        # Phase 1: low byte
        low_candidates = list(range(256))
        low_scores = score_candidates(model, cipher_name, factory, C, C_prime,
                                       low_candidates, device)
        low_ranked = sorted(low_scores.items(), key=lambda x: x[1], reverse=True)
        best_low = low_ranked[0][0]
        real_low = real_subkey & 0xFF
        low_rank = next((i for i, (k, _) in enumerate(low_ranked) if k == real_low), -1)

        # Phase 2: high byte
        high_candidates = [(h << 8) | best_low for h in range(256)]
        high_scores = score_candidates(model, cipher_name, factory, C, C_prime,
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
            'full_rank': full_rank + 1 if full_rank >= 0 else -1,
            'dist_acc': float(dist_metrics['accuracy']),
        })
        print(f"  └─ Done ─────────────────────────────────────┘")

    # Aggregate
    kc_rate = np.mean([r['key_correct'] for r in seed_results])
    avg_low = np.mean([r['low_rank'] for r in seed_results])
    avg_high = np.mean([r['high_rank'] for r in seed_results])

    print(f"\n  Key recovery rate: {kc_rate*100:.0f}%")
    print(f"  Mean low byte rank: {avg_low:.1f}/256")
    print(f"  Mean high byte rank: {avg_high:.1f}/256")

    return {
        'cipher': cipher_name,
        'rounds': n_rounds,
        'key_correct_rate': float(kc_rate),
        'mean_low_rank': float(avg_low),
        'mean_high_rank': float(avg_high),
        'per_seed': seed_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Multi-Cipher Experiment Suite')
    parser.add_argument('--task', default='all',
                        choices=['all', 'e01', 'e09', 'e11', 'e12'])
    parser.add_argument('--ciphers', nargs='+', default=['simon32', 'present'],
                        choices=['speck32', 'simon32', 'present'],
                        help='Which ciphers to run (default: simon32 present since speck32 done)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--samples', type=int, default=500000)
    parser.add_argument('--seeds', type=int, default=3,
                        help='Seeds for E01/E12 (E09 always uses 5)')
    args = parser.parse_args()

    output_dir = Path('./results/multi_cipher')
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for cipher_name in args.ciphers:
        cipher_results = {}
        t0_cipher = time.time()

        if args.task in ('all', 'e01'):
            print(f"\n{'█'*60}")
            print(f"  E01 — {cipher_name.upper()}")
            print(f"{'█'*60}")
            t0 = time.time()
            r = run_e01_for_cipher(cipher_name, args.device, args.samples, args.seeds)
            cipher_results['e01'] = r
            print(f"  ⏱ E01 {cipher_name}: {time.time()-t0:.0f}s")

        if args.task in ('all', 'e11'):
            print(f"\n{'█'*60}")
            print(f"  E11 — {cipher_name.upper()}")
            print(f"{'█'*60}")
            t0 = time.time()
            r = run_e11_for_cipher(cipher_name, args.device, args.samples)
            cipher_results['e11'] = r
            print(f"  ⏱ E11 {cipher_name}: {time.time()-t0:.0f}s")

        if args.task in ('all', 'e09'):
            print(f"\n{'█'*60}")
            print(f"  E09 — {cipher_name.upper()}")
            print(f"{'█'*60}")
            t0 = time.time()
            r = run_e09_for_cipher(cipher_name, args.device, args.samples, n_seeds=5)
            cipher_results['e09'] = r
            print(f"  ⏱ E09 {cipher_name}: {time.time()-t0:.0f}s")

        if args.task in ('all', 'e12'):
            print(f"\n{'█'*60}")
            print(f"  E12 — {cipher_name.upper()}")
            print(f"{'█'*60}")
            t0 = time.time()
            r = run_e12_for_cipher(cipher_name, args.device, args.samples, args.seeds)
            cipher_results['e12'] = r
            print(f"  ⏱ E12 {cipher_name}: {time.time()-t0:.0f}s")

        all_results[cipher_name] = cipher_results
        elapsed = time.time() - t0_cipher
        print(f"\n  ═══ {cipher_name.upper()} total: {elapsed:.0f}s ═══")

        # Save per-cipher results
        with open(output_dir / f'{cipher_name}_results.json', 'w') as f:
            json.dump(cipher_results, f, indent=2)

    # Save combined
    with open(output_dir / 'all_ciphers_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'═'*60}")
    print(f"  ✓ All multi-cipher experiments complete!")
    print(f"  Results saved to {output_dir}")
    print(f"{'═'*60}")


if __name__ == '__main__':
    main()
