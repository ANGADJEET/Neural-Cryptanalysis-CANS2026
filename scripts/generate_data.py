
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generator import CipherDataGenerator, generate_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate datasets for neural cryptanalysis'
    )
    
    parser.add_argument(
        '--cipher',
        type=str,
        required=True,
        choices=['speck32', 'simon32', 'present'],
        help='Cipher to use'
    )
    
    parser.add_argument(
        '--rounds',
        type=int,
        nargs='+',
        required=True,
        help='Number of rounds (can specify multiple)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=10_000_000,
        help='Total number of samples (default: 10M)'
    )
    
    parser.add_argument(
        '--delta-p',
        type=str,
        default=None,
        help='Input difference in hex (e.g., 0x00400000)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/generated',
        help='Output directory'
    )
    
    parser.add_argument(
        '--include-plaintext',
        action='store_true',
        help='Include plaintext in dataset'
    )
    
    parser.add_argument(
        '--include-trace',
        action='store_true',
        help='Include intermediate round states (white-box)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio'
    )
    
    return parser.parse_args()


def get_default_delta_p(cipher: str) -> int:
    defaults = {
        'speck32': 0x00400000,
        'simon32': 0x00000001,
        'present': 0x0000000000000001
    }
    return defaults.get(cipher, 0x00000001)


def main():
    args = parse_args()
    
    if args.delta_p:
        delta_p = int(args.delta_p, 16)
    else:
        delta_p = get_default_delta_p(args.cipher)
    
    n_train = int(args.samples * args.train_ratio)
    n_val = int(args.samples * args.val_ratio)
    n_test = args.samples - n_train - n_val
    
    print(f"Generating datasets for {args.cipher.upper()}")
    print(f"  Delta P: 0x{delta_p:08x}")
    print(f"  Train: {n_train:,}, Val: {n_val:,}, Test: {n_test:,}")
    print(f"  Include plaintext: {args.include_plaintext}")
    print(f"  Include trace: {args.include_trace}")
    print()
    
    for n_rounds in args.rounds:
        print(f"Generating {n_rounds}-round dataset...")
        
        filepath = generate_dataset(
            cipher=args.cipher,
            n_rounds=n_rounds,
            delta_p=delta_p,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            output_dir=args.output_dir,
            include_plaintext=args.include_plaintext,
            include_trace=args.include_trace,
            seed=args.seed
        )
        
        print(f"  Saved to: {filepath}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
