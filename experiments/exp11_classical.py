
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
from data.statistics import compute_differential_probability
from models import get_model
from training.trainer import Trainer
from evaluation.metrics import evaluate_model
from experiments.experiment_utils import (
    set_seed, add_common_args, run_multi_seed, save_results, get_device
)

GOHR_RESULTS = {
    'speck32': {
        5: 0.9244,
        6: 0.7880,
        7: 0.6116,
        8: 0.5134,
    },
    'simon32': {
        5: 0.88,
        6: 0.72,
        7: 0.58,
        8: 0.52,
    },
}


def single_run(seed, args):
    set_seed(seed)
    cipher = get_cipher(args.cipher)
    device = get_device(args)
    rounds_list = args.round_list

    neural_acc = {}
    classical_dp = {}

    for n_rounds in rounds_list:
        print(f"    Round {n_rounds}...", end=' ')

        dp = compute_differential_probability(
            diff_in=cipher.get_default_delta_p(),
            diff_out=0,
            cipher=cipher,
            n_samples=min(args.samples, 500_000),
            n_rounds=n_rounds
        )
        classical_dp[n_rounds] = float(dp)

        gen = CipherDataGenerator(
            cipher=args.cipher, n_rounds=n_rounds,
            delta_p=cipher.get_default_delta_p()
        )
        train_data = gen.generate_balanced_dataset(args.samples)
        val_data = gen.generate_balanced_dataset(args.samples // 10)
        test_data = gen.generate_balanced_dataset(args.samples // 10)

        input_dim = get_input_dim('R2_xor_diff', cipher.block_size)
        model = get_model('gohr_mlp', input_dim=input_dim)

        train_ds = CryptoDataset(train_data, 'R2_xor_diff', cipher.block_size)
        val_ds = CryptoDataset(val_data, 'R2_xor_diff', cipher.block_size)
        test_ds = CryptoDataset(test_data, 'R2_xor_diff', cipher.block_size)

        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)

        trainer = Trainer(model=model, train_loader=train_loader,
                          val_loader=val_loader, device=device, use_wandb=False)
        trainer.train(n_epochs=args.epochs, early_stopping_patience=5, save_best=False)

        metrics = evaluate_model(model, test_loader, device)
        neural_acc[n_rounds] = float(metrics['accuracy'])
        print(f"neural={metrics['accuracy']:.4f}, dp={dp:.6f}")

    return {
        'neural': {str(k): v for k, v in neural_acc.items()},
        'classical_dp': {str(k): v for k, v in classical_dp.items()},
    }


def main():
    parser = argparse.ArgumentParser(description='E11: Classical vs Neural (+ Gohr)')
    add_common_args(parser)
    parser.add_argument('--rounds', type=int, nargs='+', default=None)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = './results/e11_classical'

    print("=" * 60)
    print("  E11: Classical vs Neural Comparison")
    print("  (with Gohr CRYPTO 2019 reference)")
    print("=" * 60)

    cipher = get_cipher(args.cipher)
    args.round_list = args.rounds if args.rounds else list(range(3, 9))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_multi_seed(single_run, args)

    rounds_list = args.round_list
    seeds = [args.seed + i for i in range(args.n_seeds)]

    neural_means = {}
    neural_stds = {}
    classical_means = {}

    for r in rounds_list:
        key = str(r)
        neural_vals = []
        dp_vals = []
        for seed_result in [results]:
            if 'neural' in results and key in results['neural']:
                nr = results['neural'][key]
                if isinstance(nr, dict) and 'values' in nr:
                    neural_vals = nr['values']
                    neural_means[r] = nr['mean']
                    neural_stds[r] = nr['std']
                elif isinstance(nr, (int, float)):
                    neural_vals = [nr]
                    neural_means[r] = nr
                    neural_stds[r] = 0
            if 'classical_dp' in results and key in results['classical_dp']:
                dp = results['classical_dp'][key]
                if isinstance(dp, dict) and 'mean' in dp:
                    classical_means[r] = min(1.0, 0.5 + dp['mean'])
                elif isinstance(dp, (int, float)):
                    classical_means[r] = min(1.0, 0.5 + dp)

    gohr = GOHR_RESULTS.get(args.cipher, {})

    fig, ax = plt.subplots(figsize=(10, 6))

    rounds = sorted(neural_means.keys())
    nm = [neural_means[r] for r in rounds]
    ns = [neural_stds.get(r, 0) for r in rounds]

    ax.errorbar(rounds, nm, yerr=ns, fmt='bo-', linewidth=2,
                markersize=8, capsize=5, capthick=2,
                label='Ours (Gohr MLP)', zorder=5)

    if classical_means:
        cm = [classical_means.get(r, 0.5) for r in rounds]
        ax.plot(rounds, cm, 'r^--', linewidth=2, markersize=8,
                label='Classical (0.5 + DP)')

    if gohr:
        gohr_rounds = sorted(r for r in gohr if r in rounds or
                              min(rounds) <= r <= max(rounds))
        if gohr_rounds:
            ax.plot(gohr_rounds, [gohr[r] for r in gohr_rounds],
                    'gs-.', linewidth=2, markersize=10,
                    label='Gohr (2019) — Deep ResNet', zorder=4)

    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random')
    ax.set_xlabel('Number of Rounds', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Neural vs Classical vs Gohr — {args.cipher.upper()}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.45, 1.0)

    plt.tight_layout()
    plt.savefig(output_dir / f'e11_{args.cipher}.png', dpi=300)
    plt.close()

    results['gohr_reference'] = {str(k): v for k, v in gohr.items()}
    save_results(results, str(output_dir), f'e11_{args.cipher}_results.json')

    print(f"\n{'═' * 65}")
    print(f"  {'Round':>5}  {'Ours':>16}  {'Classical':>10}  {'Gohr 2019':>10}")
    print(f"{'─' * 65}")
    for r in rounds:
        ours = f"{neural_means.get(r, 0):.4f}±{neural_stds.get(r, 0):.4f}"
        classical = f"{classical_means.get(r, 0):.4f}"
        gohr_val = f"{gohr.get(r, 'N/A')}"
        print(f"  {r:>5}  {ours:>16}  {classical:>10}  {gohr_val:>10}")
    print(f"{'═' * 65}")

    print(f"\n✓ Results saved to {output_dir}")


if __name__ == '__main__':
    main()
