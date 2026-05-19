#!/usr/bin/env python3
"""
Pending Fixes — Run all P1/P2 tasks from audit walkthrough.

Tasks:
  P1-a: MINE positive control (verify MI estimation works)
  P2-a: Statistical test for E09 anti-transfer claim
  P1-b: Re-run E11 with fixed classical distinguisher

Usage:
  python experiments/run_pending_fixes.py --task all
  python experiments/run_pending_fixes.py --task mine_control
  python experiments/run_pending_fixes.py --task e09_test
  python experiments/run_pending_fixes.py --task e11_rerun
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
import time

# ─────────────────────────────────────────────────────────────────────────────
# P1-a: MINE Positive Control
# ─────────────────────────────────────────────────────────────────────────────
def run_mine_positive_control(device='cuda'):
    """
    Validate that MINE actually estimates MI correctly by testing on
    a known distribution where the true MI is analytically computable.

    Test 1: Independent X, Y → MI ≈ 0
    Test 2: Y = X + small noise → MI >> 0
    Test 3: Y = sign(X) (deterministic binary) → MI = ln(2) ≈ 0.693
    """
    from evaluation.metrics import estimate_mutual_information

    print("=" * 60)
    print("  P1-a: MINE Positive Control")
    print("=" * 60)

    np.random.seed(42)
    n = 50000
    results = {}

    # Test 1: Independent — MI should be ≈ 0
    print("\n  Test 1: Independent X, Y (expected MI ≈ 0.0)")
    X_ind = np.random.randn(n, 8).astype(np.float32)
    Y_ind = np.random.randint(0, 2, n).astype(np.float32)
    mi_ind = estimate_mutual_information(
        X_ind, Y_ind, device=device, n_epochs=200, verbose=False
    )
    results['independent'] = {'mi': float(mi_ind), 'expected': 0.0}
    print(f"    MI = {mi_ind:.4f} nats (expected ≈ 0.0)")

    # Test 2: Correlated — Y = 1 if X[:,0] > 0, else 0
    print("\n  Test 2: Deterministic Y = 1{X₀ > 0} (expected MI ≈ ln(2) = 0.693)")
    X_det = np.random.randn(n, 8).astype(np.float32)
    Y_det = (X_det[:, 0] > 0).astype(np.float32)
    mi_det = estimate_mutual_information(
        X_det, Y_det, device=device, n_epochs=200, verbose=False
    )
    results['deterministic'] = {'mi': float(mi_det), 'expected': 0.693}
    print(f"    MI = {mi_det:.4f} nats (expected ≈ 0.693)")

    # Test 3: Noisy correlation — Y = 1 if X[:,0] + noise > 0
    print("\n  Test 3: Noisy Y = 1{X₀ + ε > 0}, ε~N(0,1) (expected MI ∈ [0.1, 0.5])")
    X_noisy = np.random.randn(n, 8).astype(np.float32)
    noise = np.random.randn(n).astype(np.float32)
    Y_noisy = (X_noisy[:, 0] + noise > 0).astype(np.float32)
    mi_noisy = estimate_mutual_information(
        X_noisy, Y_noisy, device=device, n_epochs=200, verbose=False
    )
    results['noisy'] = {'mi': float(mi_noisy), 'expected_range': [0.05, 0.5]}
    print(f"    MI = {mi_noisy:.4f} nats (expected ∈ [0.05, 0.5])")

    # Test 4: Crypto-like — actual SPECK distinguishing at 3 rounds (should be high)
    print("\n  Test 4: Real crypto data — SPECK32, 3 rounds (should be >> 0)")
    from ciphers import get_cipher
    from data.generator import CipherDataGenerator
    from data.representations import RepresentationFactory

    cipher = get_cipher('speck32')
    gen = CipherDataGenerator('speck32', n_rounds=3, delta_p=cipher.get_default_delta_p())
    data = gen.generate_balanced_dataset(min(n, 100000))
    factory = RepresentationFactory(block_size=32)
    X_crypto = factory.get_representation('R2_xor_diff', data['C'], data['C_prime'])
    Y_crypto = data['labels'].astype(np.float32)

    mi_crypto = estimate_mutual_information(
        X_crypto, Y_crypto, device=device, n_epochs=200, verbose=False
    )
    results['crypto_3r'] = {'mi': float(mi_crypto), 'expected': 'high (>> 0)'}
    print(f"    MI = {mi_crypto:.4f} nats")

    # Verdict
    print(f"\n  {'─' * 50}")
    passed = 0
    total = 4
    if abs(mi_ind) < 0.1:
        print("  ✓ Test 1 PASSED: Independent MI near zero")
        passed += 1
    else:
        print(f"  ✗ Test 1 FAILED: Independent MI = {mi_ind:.4f} (too far from 0)")

    if mi_det > 0.3:
        print("  ✓ Test 2 PASSED: Deterministic MI detected")
        passed += 1
    else:
        print(f"  ✗ Test 2 FAILED: Deterministic MI = {mi_det:.4f} (too low)")

    if 0.01 < mi_noisy < 0.8:
        print("  ✓ Test 3 PASSED: Noisy MI in expected range")
        passed += 1
    else:
        print(f"  ✗ Test 3 FAILED: Noisy MI = {mi_noisy:.4f} (out of range)")

    if mi_crypto > 0.1:
        print("  ✓ Test 4 PASSED: Crypto MI detected")
        passed += 1
    else:
        print(f"  ✗ Test 4 FAILED: Crypto MI = {mi_crypto:.4f} (too low)")

    print(f"\n  Result: {passed}/{total} tests passed")
    results['_passed'] = passed
    results['_total'] = total

    return results


# ─────────────────────────────────────────────────────────────────────────────
# P2-a: E09 Anti-Transfer Statistical Test
# ─────────────────────────────────────────────────────────────────────────────
def run_e09_statistical_test(device='cuda', n_seeds=5, n_samples=500000):
    """
    Train a SPECK32 distinguisher at 5 rounds, test on SIMON32 at 5 rounds.
    Use a one-sample t-test to determine if accuracy is significantly < 0.5
    (anti-transfer) or significantly > 0.5 (positive transfer).
    """
    import torch
    from scipy import stats
    from ciphers import get_cipher
    from data.generator import CipherDataGenerator
    from data.dataloader import CryptoDataset, get_input_dim
    from models import get_model
    from training.trainer import Trainer
    from evaluation.metrics import evaluate_model
    from torch.utils.data import DataLoader
    from experiments.experiment_utils import set_seed

    print("=" * 60)
    print("  P2-a: E09 Anti-Transfer Statistical Significance Test")
    print("=" * 60)

    source_cipher_name = 'speck32'
    target_cipher_name = 'simon32'
    source_rounds = 5

    source_cipher = get_cipher(source_cipher_name)
    target_cipher = get_cipher(target_cipher_name)

    cross_cipher_accs = []
    cross_round_accs = {}  # {target_round: [acc_per_seed]}
    source_accs = []

    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx
        set_seed(seed)
        print(f"\n  ┌─ Seed {seed} ({seed_idx+1}/{n_seeds}) ─────────────────┐")

        # Train on source cipher
        gen = CipherDataGenerator(
            source_cipher_name, n_rounds=source_rounds,
            delta_p=source_cipher.get_default_delta_p()
        )
        train_data = gen.generate_balanced_dataset(n_samples)
        val_data = gen.generate_balanced_dataset(n_samples // 10)

        input_dim = get_input_dim('R2_xor_diff', source_cipher.block_size)
        model = get_model('gohr_mlp', input_dim=input_dim)

        train_ds = CryptoDataset(train_data, 'R2_xor_diff', source_cipher.block_size)
        val_ds = CryptoDataset(val_data, 'R2_xor_diff', source_cipher.block_size)
        train_loader = DataLoader(train_ds, batch_size=5000, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=5000)

        trainer = Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            device=device, use_wandb=False
        )
        trainer.train(n_epochs=30, early_stopping_patience=5, save_best=False)

        src_metrics = evaluate_model(model, val_loader, device)
        source_accs.append(float(src_metrics['accuracy']))
        print(f"    Source ({source_cipher_name}, {source_rounds}r): {src_metrics['accuracy']:.4f}")

        # Test on target cipher (cross-cipher)
        tgen = CipherDataGenerator(
            target_cipher_name, n_rounds=source_rounds,
            delta_p=target_cipher.get_default_delta_p()
        )
        tdata = tgen.generate_balanced_dataset(n_samples // 5)
        tds = CryptoDataset(tdata, 'R2_xor_diff', target_cipher.block_size)
        tloader = DataLoader(tds, batch_size=5000)

        tmetrics = evaluate_model(model, tloader, device)
        cross_cipher_accs.append(float(tmetrics['accuracy']))
        print(f"    Target ({target_cipher_name}, {source_rounds}r): {tmetrics['accuracy']:.4f}")

        # Test on same cipher, different rounds (cross-round)
        for target_r in [3, 4, 6, 7, 8]:
            rgen = CipherDataGenerator(
                source_cipher_name, n_rounds=target_r,
                delta_p=source_cipher.get_default_delta_p()
            )
            rdata = rgen.generate_balanced_dataset(n_samples // 10)
            rds = CryptoDataset(rdata, 'R2_xor_diff', source_cipher.block_size)
            rloader = DataLoader(rds, batch_size=5000)
            rmetrics = evaluate_model(model, rloader, device)

            if target_r not in cross_round_accs:
                cross_round_accs[target_r] = []
            cross_round_accs[target_r].append(float(rmetrics['accuracy']))
            print(f"    → {source_cipher_name} {target_r}r: {rmetrics['accuracy']:.4f}")

        print(f"  └─ Done ─────────────────────────────────────┘")

    # Statistical tests
    print(f"\n{'═' * 60}")
    print("  Statistical Analysis")
    print(f"{'═' * 60}")

    results = {
        'source_accs': source_accs,
        'cross_cipher_accs': cross_cipher_accs,
        'cross_round_accs': {str(k): v for k, v in cross_round_accs.items()},
    }

    # Cross-cipher: one-sample t-test against 0.5
    cc_arr = np.array(cross_cipher_accs)
    cc_mean = cc_arr.mean()
    cc_std = cc_arr.std()
    t_stat, p_value = stats.ttest_1samp(cc_arr, 0.5)
    results['cross_cipher_ttest'] = {
        'mean': float(cc_mean),
        'std': float(cc_std),
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'significant_at_005': bool(p_value < 0.05),
        'direction': 'anti-transfer' if cc_mean < 0.5 else 'positive-transfer',
    }

    print(f"\n  Cross-Cipher ({source_cipher_name} → {target_cipher_name}):")
    print(f"    Accuracy: {cc_mean:.4f} ± {cc_std:.4f}")
    print(f"    t-test vs 0.5: t={t_stat:.3f}, p={p_value:.4f}")
    if p_value < 0.05:
        direction = "ANTI-TRANSFER" if cc_mean < 0.5 else "POSITIVE TRANSFER"
        print(f"    ★ SIGNIFICANT at α=0.05 — {direction}")
    else:
        print(f"    ○ NOT significant at α=0.05")

    # Cross-round tests
    print(f"\n  Cross-Round ({source_cipher_name}, trained on {source_rounds}r):")
    for target_r in sorted(cross_round_accs.keys()):
        arr = np.array(cross_round_accs[target_r])
        mean = arr.mean()
        std = arr.std()
        t, p = stats.ttest_1samp(arr, 0.5)
        sig = "★" if p < 0.05 else "○"
        direction = "↓" if mean < 0.5 else "↑"
        print(f"    {target_r}r: {mean:.4f}±{std:.4f}  t={t:.3f} p={p:.4f} {sig} {direction}")

        results[f'cross_round_{target_r}_ttest'] = {
            'mean': float(mean), 'std': float(std),
            't_stat': float(t), 'p_value': float(p),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# P1-b: Re-run E11 with fixed classical distinguisher
# ─────────────────────────────────────────────────────────────────────────────
def run_e11_fixed(device='cuda', n_samples=500000):
    """Re-run E11 with the fixed compute_classical_distinguisher_accuracy."""
    import torch
    from ciphers import get_cipher
    from data.generator import CipherDataGenerator
    from data.dataloader import CryptoDataset, get_input_dim
    from data.statistics import compute_classical_distinguisher_accuracy
    from models import get_model
    from training.trainer import Trainer
    from evaluation.metrics import evaluate_model
    from torch.utils.data import DataLoader
    from experiments.experiment_utils import set_seed

    print("=" * 60)
    print("  P1-b: E11 Re-run with Fixed Classical Distinguisher")
    print("=" * 60)

    cipher_name = 'speck32'
    cipher = get_cipher(cipher_name)
    rounds_list = list(range(3, 9))

    GOHR_RESULTS = {5: 0.9244, 6: 0.7880, 7: 0.6116, 8: 0.5134}

    results = {}

    for n_rounds in rounds_list:
        print(f"\n  --- Round {n_rounds} ---")
        set_seed(42)

        # Classical distinguisher (FIXED)
        t0 = time.time()
        classical_acc = compute_classical_distinguisher_accuracy(
            cipher=cipher,
            diff_in=cipher.get_default_delta_p(),
            n_rounds=n_rounds,
            n_samples=n_samples,
            n_keys=5
        )
        t_classical = time.time() - t0
        print(f"    Classical (bit-bias): {classical_acc:.4f} ({t_classical:.1f}s)")

        # Neural distinguisher
        gen = CipherDataGenerator(cipher_name, n_rounds=n_rounds,
                                   delta_p=cipher.get_default_delta_p())
        train_data = gen.generate_balanced_dataset(n_samples)
        val_data = gen.generate_balanced_dataset(n_samples // 10)
        test_data = gen.generate_balanced_dataset(n_samples // 10)

        input_dim = get_input_dim('R2_xor_diff', cipher.block_size)
        model = get_model('gohr_mlp', input_dim=input_dim)

        train_ds = CryptoDataset(train_data, 'R2_xor_diff', cipher.block_size)
        val_ds = CryptoDataset(val_data, 'R2_xor_diff', cipher.block_size)
        test_ds = CryptoDataset(test_data, 'R2_xor_diff', cipher.block_size)

        train_loader = DataLoader(train_ds, batch_size=5000, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=5000)
        test_loader = DataLoader(test_ds, batch_size=5000)

        t0 = time.time()
        trainer = Trainer(model=model, train_loader=train_loader,
                          val_loader=val_loader, device=device, use_wandb=False)
        trainer.train(n_epochs=30, early_stopping_patience=5, save_best=False)

        metrics = evaluate_model(model, test_loader, device)
        t_neural = time.time() - t0
        neural_acc = float(metrics['accuracy'])
        print(f"    Neural (GohrMLP):    {neural_acc:.4f} ({t_neural:.1f}s)")

        gohr = GOHR_RESULTS.get(n_rounds, None)
        if gohr:
            print(f"    Gohr (2019 ResNet):  {gohr:.4f}")

        results[str(n_rounds)] = {
            'classical': classical_acc,
            'neural': neural_acc,
            'gohr': gohr,
            'gap': round(neural_acc - classical_acc, 4),
        }

    # Summary table
    print(f"\n{'═' * 65}")
    print(f"  {'Round':>5}  {'Classical':>10}  {'Neural':>10}  {'Gohr':>10}  {'Gap':>8}")
    print(f"{'─' * 65}")
    for r in rounds_list:
        d = results[str(r)]
        gohr_str = f"{d['gohr']:.4f}" if d['gohr'] else "N/A"
        print(f"  {r:>5}  {d['classical']:>10.4f}  {d['neural']:>10.4f}  {gohr_str:>10}  {d['gap']:>+8.4f}")
    print(f"{'═' * 65}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Run pending audit fixes')
    parser.add_argument('--task', default='all',
                        choices=['all', 'mine_control', 'e09_test', 'e11_rerun'],
                        help='Which task to run')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--samples', type=int, default=500000)
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()

    output_dir = Path('./results/pending_fixes')
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if args.task in ('all', 'mine_control'):
        t0 = time.time()
        r = run_mine_positive_control(device=args.device)
        all_results['mine_positive_control'] = r
        with open(output_dir / 'mine_positive_control.json', 'w') as f:
            json.dump(r, f, indent=2)
        print(f"\n  ⏱ MINE control: {time.time()-t0:.0f}s")

    if args.task in ('all', 'e09_test'):
        t0 = time.time()
        r = run_e09_statistical_test(
            device=args.device, n_seeds=args.seeds, n_samples=args.samples
        )
        all_results['e09_statistical_test'] = r
        with open(output_dir / 'e09_statistical_test.json', 'w') as f:
            json.dump(r, f, indent=2)
        print(f"\n  ⏱ E09 test: {time.time()-t0:.0f}s")

    if args.task in ('all', 'e11_rerun'):
        t0 = time.time()
        r = run_e11_fixed(device=args.device, n_samples=args.samples)
        all_results['e11_fixed'] = r
        with open(output_dir / 'e11_fixed.json', 'w') as f:
            json.dump(r, f, indent=2)
        print(f"\n  ⏱ E11 rerun: {time.time()-t0:.0f}s")

    # Save combined
    with open(output_dir / 'all_pending_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'═' * 60}")
    print(f"  All pending tasks complete!")
    print(f"  Results saved to {output_dir}")
    print(f"{'═' * 60}")


if __name__ == '__main__':
    main()
