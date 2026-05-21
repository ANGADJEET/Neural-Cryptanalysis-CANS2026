#!/usr/bin/env python3
"""
E12b: PRESENT Key Recovery (64-bit SPN)

Extends the key recovery attack from 32-bit ciphers (SPECK/SIMON) to the
64-bit SPN cipher PRESENT. This tests the paper's claim that the
framework generalizes across cipher families and block sizes.

Key differences from E12:
  - PRESENT uses 64-bit blocks with 4-bit S-boxes and a bit permutation
  - Round inversion requires applying inverse permutation + inverse S-box
  - Partial key search: enumerate candidate last-round subkey values
    for a subset of S-box positions (searching full 64-bit key is 2^64)

Strategy (following Gohr's partial search):
  1. Train distinguisher on (R-1)-round PRESENT
  2. For key recovery at R rounds:
     a. Partial decrypt: try candidate values for the last-round key
        restricted to specific nibbles (4-bit S-box entries)
     b. Score each candidate using the trained distinguisher
  3. Report the rank of the correct partial key

Usage:
  python experiments/exp12b_present_key_recovery.py --rounds 4
  python experiments/exp12b_present_key_recovery.py --rounds 5
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
from data.representations import RepresentationFactory
from models import get_model
from training.trainer import Trainer
from evaluation.metrics import evaluate_model
from experiments.experiment_utils import (
    set_seed, add_common_args, run_multi_seed, save_results, get_device
)


# PRESENT constants
SBOX_INV = np.array([
    0x5, 0xE, 0xF, 0x8, 0xC, 0x1, 0x2, 0xD,
    0xB, 0x4, 0x6, 0x3, 0x0, 0x7, 0x9, 0xA
], dtype=np.uint8)

PERM_INV = np.zeros(64, dtype=np.uint8)
PERM = np.array([
    0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51,
    4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55,
    8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
    12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63
], dtype=np.uint8)
for i in range(64):
    PERM_INV[PERM[i]] = i


def inv_perm_layer(state: np.ndarray) -> np.ndarray:
    """Inverse of PRESENT's bit permutation layer (vectorized)."""
    result = np.zeros_like(state)
    for i in range(64):
        bit = (state >> i) & 1
        result |= bit << int(PERM_INV[i])
    return result


def inv_sbox_layer(state: np.ndarray) -> np.ndarray:
    """Inverse of PRESENT's S-box layer (vectorized)."""
    result = np.zeros_like(state)
    for i in range(16):
        nibble = (state >> (4 * i)) & 0xF
        result |= SBOX_INV[nibble.astype(np.int64)].astype(np.uint64) << (4 * i)
    return result


def decrypt_one_round_present(ciphertext: np.ndarray, subkey: int) -> np.ndarray:
    """Decrypt one round of PRESENT.
    
    PRESENT round (encrypt): state = perm(sbox(state ^ rk))
    Inverse: state = inv_sbox(inv_perm(state)) ^ rk
    
    But the LAST round of PRESENT has an extra key XOR:
      encrypt: ... → perm(sbox(state ^ rk_{R-1})) → state ^ rk_R
    So to peel off: state = inv_sbox(inv_perm(state ^ rk_R))... but actually
    the last key addition is just XOR, and the round before has sbox+perm.
    
    For partial key recovery, we peel the last key addition and partial
    S-box application:
      1. XOR candidate key: state ^ candidate
      2. Then the distinguisher sees the (R-1)-round output
    
    Since PRESENT's last operation is: ciphertext = round_{R-1}_output ^ rk_R
    Just XOR-ing with the candidate key recovers round_{R-1}_output directly.
    """
    return ciphertext ^ np.uint64(subkey)


def score_candidates_present(
    model, factory, C, C_prime, candidates, device, batch_size=5000
):
    """Score partial key candidates using the Bayesian log-LR approach."""
    scores = {}
    model.eval()
    
    for candidate in candidates:
        C_dec = decrypt_one_round_present(C, candidate)
        C_prime_dec = decrypt_one_round_present(C_prime, candidate)
        
        X = factory.get_representation('R2_xor_diff', C_dec, C_prime_dec)
        
        s_list = []
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            X_tensor = torch.from_numpy(X_batch).float().to(device)
            with torch.no_grad():
                s_batch = model(X_tensor).squeeze().cpu().numpy()
                if s_batch.ndim == 0:
                    s_batch = np.array([s_batch])
            s_list.append(s_batch)
        
        s = np.concatenate(s_list)
        s = np.clip(s, 1e-7, 1 - 1e-7)
        log_lr = np.log(s / (1 - s))
        scores[candidate] = float(np.sum(log_lr))
    
    return scores


def single_run(seed, args):
    set_seed(seed)
    cipher = get_cipher('present')
    device = get_device(args)
    n_rounds = args.rounds
    reduced_rounds = n_rounds - 1

    # ── Train distinguisher on (R-1)-round PRESENT ──────────────────
    gen = CipherDataGenerator(
        cipher='present', n_rounds=reduced_rounds,
        delta_p=cipher.get_default_delta_p()
    )
    train_data = gen.generate_balanced_dataset(args.samples)
    val_data = gen.generate_balanced_dataset(args.samples // 10)

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
    trainer.train(n_epochs=args.epochs, early_stopping_patience=5)

    dist_metrics = evaluate_model(model, val_loader, device)
    print(f"  Distinguisher accuracy ({reduced_rounds}r): {dist_metrics['accuracy']:.4f}")

    # ── Key recovery ────────────────────────────────────────────────
    real_key = cipher.random_key()
    P = cipher.random_plaintexts(args.n_pairs)
    P_prime = P ^ np.uint64(cipher.get_default_delta_p())
    C = cipher.encrypt(P, n_rounds, real_key)
    C_prime = cipher.encrypt(P_prime, n_rounds, real_key)

    # Get real last-round subkey
    round_keys = cipher._expand_key(real_key, n_rounds)
    real_last_subkey = int(round_keys[-1])

    factory = RepresentationFactory(block_size=cipher.block_size)

    # For 64-bit PRESENT, we can't enumerate 2^64 candidates.
    # Instead, we do PARTIAL key recovery on specific nibbles.
    # We search lower 16 bits of the last round key (4 S-box positions).
    
    # Phase 1: Search lower 16 bits (nibbles 0-3)
    real_low16 = real_last_subkey & 0xFFFF
    print(f"  Phase 1: Searching lower 16 bits of last-round key...")
    
    # We enumerate 2^16 = 65536 candidates for the lower 16 bits
    # (upper 48 bits set to 0 for scoring — the distinguisher mainly
    #  sees the differential, so incorrect upper bits add noise but
    #  the correct lower bits still produce a score advantage)
    n_candidates = min(args.n_candidates, 65536)
    if n_candidates == 65536:
        candidates = list(range(65536))
    else:
        # Random subset + always include correct value
        candidates = list(np.random.choice(65536, size=n_candidates, replace=False))
        if real_low16 not in candidates:
            candidates[0] = real_low16

    scores = score_candidates_present(
        model, factory, C, C_prime, candidates, device
    )
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_key = ranked[0][0]
    real_rank = next((i for i, (k, _) in enumerate(ranked) if k == real_low16), -1)

    key_correct = (best_key == real_low16)
    found_in_top10 = real_low16 in [k for k, _ in ranked[:10]]

    print(f"  ─────────────────────────────────────")
    print(f"  Real lower 16 bits:      0x{real_low16:04x}")
    print(f"  Recovered lower 16 bits: 0x{best_key:04x} {'✓' if key_correct else '✗'}")
    print(f"  Rank: {real_rank+1}/{len(candidates)}")
    print(f"  Found in top-10: {'Yes' if found_in_top10 else 'No'}")

    return {
        'distinguisher_accuracy': float(dist_metrics['accuracy']),
        'real_key_low16': f"0x{real_low16:04x}",
        'recovered_key_low16': f"0x{best_key:04x}",
        'key_correct': int(key_correct),
        'rank': real_rank + 1 if real_rank >= 0 else -1,
        'n_candidates': len(candidates),
        'found_in_top1': int(key_correct),
        'found_in_top10': int(found_in_top10),
    }


def main():
    parser = argparse.ArgumentParser(
        description='E12b: PRESENT Key Recovery (64-bit SPN)'
    )
    add_common_args(parser)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--n-candidates', type=int, default=65536,
                        help='Key candidates to search (max 65536 for 16-bit partial)')
    parser.add_argument('--n-pairs', type=int, default=10000)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = './results/e12b_present_key_recovery'

    print("=" * 60)
    print(f"  E12b: PRESENT Key Recovery ({args.rounds} rounds)")
    print("=" * 60)

    results = run_multi_seed(single_run, args)
    save_results(results, args.output_dir,
                 f'e12b_present_r{args.rounds}_results.json')

    if 'key_correct' in results:
        kc = results['key_correct']
        print(f"\n  Key recovery: {kc['mean']*100:.0f}% of runs")
    if 'rank' in results:
        rk = results['rank']
        print(f"  Mean rank: {rk['mean']:.1f} ± {rk['std']:.1f}")
    if 'found_in_top10' in results:
        ft = results['found_in_top10']
        print(f"  Found in top-10: {ft['mean']*100:.0f}% of runs")

    print(f"\n✓ Done")


if __name__ == '__main__':
    main()
