
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
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


WORD_MASK = 0xFFFF

def _ror16(x, r):
    r = r % 16
    return ((x >> r) | (x << (16 - r))) & WORD_MASK

def _rol16(x, r):
    r = r % 16
    return ((x << r) | (x >> (16 - r))) & WORD_MASK

def decrypt_one_round_speck(ciphertext: np.ndarray, subkey: int) -> np.ndarray:
    x_enc = ((ciphertext >> 16) & WORD_MASK).astype(np.uint16)
    y_enc = (ciphertext & WORD_MASK).astype(np.uint16)
    
    y_dec = _ror16(y_enc ^ x_enc, 2)
    
    x_dec = _rol16(((x_enc ^ subkey) - y_dec) & WORD_MASK, 7)
    
    return (x_dec.astype(np.uint32) << 16) | y_dec.astype(np.uint32)


def decrypt_one_round_simon(ciphertext: np.ndarray, subkey: int) -> np.ndarray:
    x_enc = ((ciphertext >> 16) & WORD_MASK).astype(np.uint16)
    y_enc = (ciphertext & WORD_MASK).astype(np.uint16)
    
    x_dec = y_enc
    f_val = (_rol16(y_enc, 1) & _rol16(y_enc, 8)) ^ _rol16(y_enc, 2)
    y_dec = (f_val ^ x_enc ^ subkey) & WORD_MASK
    
    return (x_dec.astype(np.uint32) << 16) | y_dec.astype(np.uint32)


def decrypt_one_round(cipher_name: str, ciphertext: np.ndarray, subkey: int) -> np.ndarray:
    if 'speck' in cipher_name.lower():
        return decrypt_one_round_speck(ciphertext, subkey)
    elif 'simon' in cipher_name.lower():
        return decrypt_one_round_simon(ciphertext, subkey)
    else:
        return ciphertext ^ subkey


def score_candidates(model, cipher_name, factory, C, C_prime,
                     candidates, device):
    scores = {}
    model.eval()
    for candidate in candidates:
        C_dec = decrypt_one_round(cipher_name, C, candidate)
        C_prime_dec = decrypt_one_round(cipher_name, C_prime, candidate)
        X = factory.get_representation('R2_xor_diff', C_dec, C_prime_dec)
        X_tensor = torch.from_numpy(X).float().to(device)
        with torch.no_grad():
            s = model(X_tensor).squeeze().cpu().numpy()
        scores[candidate] = float(np.mean(s))
    return scores


def single_run(seed, args):
    set_seed(seed)
    cipher = get_cipher(args.cipher)
    device = get_device(args)
    n_rounds = args.rounds
    reduced_rounds = n_rounds - 1

    gen = CipherDataGenerator(
        cipher=args.cipher, n_rounds=reduced_rounds,
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

    real_key = cipher.random_key()
    P = cipher.random_plaintexts(args.n_pairs)
    P_prime = P ^ cipher.get_default_delta_p()
    C = cipher.encrypt(P, n_rounds, real_key)
    C_prime = cipher.encrypt(P_prime, n_rounds, real_key)

    try:
        expanded_key = cipher._expand_key(real_key, n_rounds)
        real_last_subkey = int(expanded_key[-1]) & WORD_MASK
    except Exception:
        real_last_subkey = -1

    real_low = real_last_subkey & 0xFF
    real_high = (real_last_subkey >> 8) & 0xFF

    factory = RepresentationFactory(block_size=cipher.block_size)

    print(f"  Phase 1: Searching lower byte...")
    low_candidates = list(range(256))
    low_scores = score_candidates(
        model, args.cipher, factory, C, C_prime,
        low_candidates, device
    )
    low_ranked = sorted(low_scores.items(), key=lambda x: x[1], reverse=True)
    best_low = low_ranked[0][0]
    low_rank = next((i for i, (k, _) in enumerate(low_ranked) if k == real_low), -1)
    print(f"    Best low byte: 0x{best_low:02x} (real: 0x{real_low:02x}, rank: {low_rank+1}/256)")

    print(f"  Phase 2: Searching upper byte (with low=0x{best_low:02x})...")
    high_candidates = [(h << 8) | best_low for h in range(256)]
    high_scores = score_candidates(
        model, args.cipher, factory, C, C_prime,
        high_candidates, device
    )
    high_ranked = sorted(high_scores.items(), key=lambda x: x[1], reverse=True)
    best_full = high_ranked[0][0]
    best_high = (best_full >> 8) & 0xFF
    high_rank = next((i for i, (k, _) in enumerate(high_ranked)
                      if ((k >> 8) & 0xFF) == real_high), -1)
    print(f"    Best high byte: 0x{best_high:02x} (real: 0x{real_high:02x}, rank: {high_rank+1}/256)")

    recovered_key = best_full
    key_correct = (recovered_key == real_last_subkey)

    all_scores = {}
    all_scores.update(low_scores)
    all_scores.update(high_scores)

    full_ranked = sorted(high_scores.items(), key=lambda x: x[1], reverse=True)
    full_rank = next((i for i, (k, _) in enumerate(full_ranked)
                      if k == real_last_subkey), -1)

    top_k = [k for k, _ in full_ranked[:10]]
    found_in_top10 = real_last_subkey in top_k
    found_in_top1 = key_correct

    print(f"  ─────────────────────────────────────")
    print(f"  Real subkey:      0x{real_last_subkey:04x}")
    print(f"  Recovered subkey: 0x{recovered_key:04x} {'✓' if key_correct else '✗'}")
    print(f"  Full rank: {full_rank+1}/256 (phase 2)")
    print(f"  Low byte rank:  {low_rank+1}/256")
    print(f"  High byte rank: {high_rank+1}/256")

    return {
        'distinguisher_accuracy': float(dist_metrics['accuracy']),
        'real_key': f"0x{real_last_subkey:04x}",
        'recovered_key': f"0x{recovered_key:04x}",
        'key_correct': int(key_correct),
        'low_byte_rank': low_rank + 1 if low_rank >= 0 else -1,
        'high_byte_rank': high_rank + 1 if high_rank >= 0 else -1,
        'full_rank': full_rank + 1 if full_rank >= 0 else -1,
        'found_in_top1': int(found_in_top1),
        'found_in_top10': int(found_in_top10),
    }


def main():
    parser = argparse.ArgumentParser(description='E12: Key Recovery (Corrected)')
    add_common_args(parser)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--n-candidates', type=int, default=256,
                        help='Number of key candidates (max 65536 for 16-bit)')
    parser.add_argument('--n-pairs', type=int, default=5000)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = './results/e12_key_recovery'

    print("=" * 60)
    print("  E12: Key Recovery Demo (Proper Round Inversion)")
    print("=" * 60)

    results = run_multi_seed(single_run, args)
    save_results(results, args.output_dir,
                 f'e12_{args.cipher}_r{args.rounds}_results.json')

    if 'key_correct' in results:
        kc = results['key_correct']
        print(f"\n  Full key recovery: {kc['mean']*100:.0f}% of runs")
    if 'low_byte_rank' in results:
        lr = results['low_byte_rank']
        print(f"  Mean low byte rank: {lr['mean']:.1f} ± {lr['std']:.1f}")
    if 'high_byte_rank' in results:
        hr = results['high_byte_rank']
        print(f"  Mean high byte rank: {hr['mean']:.1f} ± {hr['std']:.1f}")
    if 'found_in_top10' in results:
        ft = results['found_in_top10']
        print(f"  Found in top-10: {ft['mean']*100:.0f}% of runs")

    print(f"\n✓ Done")


if __name__ == '__main__':
    main()
