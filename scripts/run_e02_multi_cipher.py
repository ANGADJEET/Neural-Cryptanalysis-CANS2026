#!/usr/bin/env python3
"""
Run E02 Representation Comparison across all 3 ciphers.
Tests all representations at the round count where accuracy is ~80-90%
(the most informative range for comparing representations).

Usage:
  python scripts/run_e02_multi_cipher.py --device cuda
"""
import subprocess
import sys

CONFIGS = [
    ('speck32', 5),    # ~86% accuracy — good discrimination range
    ('simon32', 7),    # ~79% accuracy — good discrimination range
    ('present', 5),    # ~72% accuracy — good discrimination range
]

def main():
    device = sys.argv[sys.argv.index('--device') + 1] if '--device' in sys.argv else 'cuda'

    for cipher, rounds in CONFIGS:
        print(f"\n{'='*60}")
        print(f"  E02 — {cipher.upper()} at {rounds} rounds")
        print(f"{'='*60}")
        cmd = [
            sys.executable, '-m', 'experiments.exp02_representation',
            '--cipher', cipher,
            '--rounds', str(rounds),
            '--samples', '500000',
            '--n-seeds', '3',
            '--device', device,
            '--output-dir', f'./results/e02_multi/{cipher}',
        ]
        subprocess.run(cmd, check=True)

    print(f"\n{'='*60}")
    print(f"  ✓ All E02 runs complete")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
