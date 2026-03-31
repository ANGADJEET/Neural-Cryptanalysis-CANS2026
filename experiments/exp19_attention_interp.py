
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ciphers import get_cipher
from data.generator import CipherDataGenerator
from data.dataloader import CryptoDataset, get_input_dim
from models import get_model
from training.trainer import Trainer
from evaluation.metrics import evaluate_model
from experiments.experiment_utils import (
    set_seed, add_common_args, save_results, get_device
)


class AttentionMLP(nn.Module):

    def __init__(self, input_dim, hidden=256, n_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.n_heads = n_heads

        self.attn_queries = nn.Parameter(torch.randn(n_heads, hidden // n_heads))
        self.attn_keys = nn.Linear(1, hidden // n_heads)
        self.attn_values = nn.Linear(1, hidden // n_heads)

        self.pos_embed = nn.Parameter(torch.randn(input_dim, hidden // n_heads))

        self.bit_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=-1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

        self._attention_weights = None

    def forward(self, x):
        attn = self.bit_attention(x)
        self._attention_weights = attn.detach()

        x_attended = x * attn

        return self.classifier(x_attended).squeeze(-1)

    def get_attention_weights(self):
        return self._attention_weights


def train_and_extract(seed, cipher_name, n_rounds, args):
    set_seed(seed)
    cipher = get_cipher(cipher_name)
    device = get_device(args)

    gen = CipherDataGenerator(
        cipher=cipher_name, n_rounds=n_rounds,
        delta_p=cipher.get_default_delta_p()
    )
    train_data = gen.generate_balanced_dataset(args.samples)
    val_data = gen.generate_balanced_dataset(args.samples // 10)

    input_dim = get_input_dim('R2_xor_diff', cipher.block_size)
    train_ds = CryptoDataset(train_data, 'R2_xor_diff', cipher.block_size)
    val_ds = CryptoDataset(val_data, 'R2_xor_diff', cipher.block_size)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    baseline_model = get_model('gohr_mlp', input_dim=input_dim)
    baseline_trainer = Trainer(
        model=baseline_model, train_loader=train_loader,
        val_loader=val_loader, device=device, use_wandb=False
    )
    baseline_trainer.train(n_epochs=args.epochs, early_stopping_patience=5, save_best=False)
    baseline_metrics = evaluate_model(baseline_model, val_loader, device)

    attn_model = AttentionMLP(input_dim, hidden=256)
    attn_trainer = Trainer(
        model=attn_model, train_loader=train_loader,
        val_loader=val_loader, device=device, use_wandb=False
    )
    attn_trainer.train(n_epochs=args.epochs, early_stopping_patience=5, save_best=False)
    attn_metrics = evaluate_model(attn_model, val_loader, device)

    attn_model.eval()
    all_weights = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            _ = attn_model(x)
            weights = attn_model.get_attention_weights()
            all_weights.append(weights.cpu().numpy())

    mean_weights = np.concatenate(all_weights, axis=0).mean(axis=0)
    mean_weights = mean_weights / (mean_weights.max() + 1e-8)

    print(f"      {n_rounds}r: baseline={baseline_metrics['accuracy']:.4f}, "
          f"attn={attn_metrics['accuracy']:.4f}, "
          f"top_bits={np.argsort(mean_weights)[-5:][::-1]}",
          flush=True)

    return {
        'baseline_accuracy': float(baseline_metrics['accuracy']),
        'attention_accuracy': float(attn_metrics['accuracy']),
        'bit_importance': mean_weights.tolist(),
        'attention_entropy': float(-np.sum(mean_weights * np.log(mean_weights + 1e-8))),
    }


def single_run(seed, args):
    results = {}
    for nr in args.rounds:
        print(f"    Round {nr}:", flush=True)
        results[str(nr)] = train_and_extract(seed, args.cipher, nr, args)
    return results


def plot_results(all_results, args, output_dir):
    rounds = sorted([int(k) for k in all_results[0].keys()])
    n_bits = len(all_results[0][str(rounds[0])]['bit_importance'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    heatmap = np.zeros((len(rounds), n_bits))
    for i, r in enumerate(rounds):
        for res in all_results:
            heatmap[i] += np.array(res[str(r)]['bit_importance'])
        heatmap[i] /= len(all_results)

    im = axes[0].imshow(heatmap, aspect='auto', cmap='hot', interpolation='nearest')
    axes[0].set_xlabel('Bit Position')
    axes[0].set_ylabel('Round')
    axes[0].set_yticks(range(len(rounds)))
    axes[0].set_yticklabels(rounds)
    axes[0].set_title(f'Bit Importance Heatmap — {args.cipher.upper()}')
    plt.colorbar(im, ax=axes[0], label='Attention Weight')

    baseline_accs = {r: np.mean([res[str(r)]['baseline_accuracy'] for res in all_results])
                     for r in rounds}
    attn_accs = {r: np.mean([res[str(r)]['attention_accuracy'] for res in all_results])
                 for r in rounds}

    axes[1].plot(rounds, [baseline_accs[r] for r in rounds], 'bo-',
                 label='Baseline MLP', linewidth=2, markersize=8)
    axes[1].plot(rounds, [attn_accs[r] for r in rounds], 'r^--',
                 label='Attention MLP', linewidth=2, markersize=8)
    axes[1].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Rounds')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Baseline vs Attention Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    entropies = {r: np.mean([res[str(r)]['attention_entropy'] for res in all_results])
                 for r in rounds}
    entropy_stds = {r: np.std([res[str(r)]['attention_entropy'] for res in all_results])
                    for r in rounds}

    axes[2].errorbar(rounds, [entropies[r] for r in rounds],
                     yerr=[entropy_stds[r] for r in rounds],
                     fmt='gs-', linewidth=2, markersize=8, capsize=4)
    axes[2].set_xlabel('Rounds')
    axes[2].set_ylabel('Attention Entropy')
    axes[2].set_title('Attention Focus (lower = more concentrated)')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'E19: Attention Interpretability — {args.cipher.upper()}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'e19_{args.cipher}_attention.png', dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='E19: Attention-Based Interpretability'
    )
    add_common_args(parser)
    parser.add_argument('--rounds', type=int, nargs='+', default=[3, 4, 5, 6, 7, 8])
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = './results/e19_attention_interp'

    print("=" * 60)
    print("  E19: Attention-Based Interpretability")
    print("  Which bits does the distinguisher focus on?")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed + i for i in range(args.n_seeds)]
    all_results = []

    for i, seed in enumerate(seeds):
        print(f"\n┌─ Seed {seed} ({i+1}/{len(seeds)}) ─────────────────┐")
        result = single_run(seed, args)
        all_results.append(result)
        print(f"└─ Done ─────────────────────────────────────┘")

    save_results(
        {'runs': all_results, '_seeds': seeds},
        str(output_dir), f'e19_{args.cipher}_results.json'
    )
    plot_results(all_results, args, output_dir)
    print(f"\n✓ Results saved to {output_dir}")


if __name__ == '__main__':
    main()
