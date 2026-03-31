
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
from data.representations import RepresentationFactory
from experiments.experiment_utils import (
    set_seed, add_common_args, save_results, get_device
)


class RoundInverterAE(nn.Module):

    def __init__(self, dim, bottleneck_dim=24, hidden=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def get_bottleneck(self, x):
        return self.encoder(x)


class SimpleDistinguisher(nn.Module):

    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def generate_traces(cipher_name, cipher, n_rounds, n_samples):
    key = cipher.random_key()
    delta_p = cipher.get_default_delta_p()
    factory = RepresentationFactory(block_size=cipher.block_size)
    half = n_samples // 2

    P = cipher.random_plaintexts(half)
    P_prime = (P ^ delta_p).astype(P.dtype)
    _, trace1 = cipher.encrypt_with_trace(P, n_rounds, key)
    _, trace2 = cipher.encrypt_with_trace(P_prime, n_rounds, key)

    diff_traces = []
    for r in range(n_rounds):
        diff = factory.get_representation('R2_xor_diff', trace1[r], trace2[r])
        diff_traces.append(diff)
    diff_traces = np.stack(diff_traces, axis=1)

    Q = cipher.random_plaintexts(half)
    R = cipher.random_plaintexts(half)
    _, trace_q = cipher.encrypt_with_trace(Q, n_rounds, key)
    _, trace_r = cipher.encrypt_with_trace(R, n_rounds, key)

    rand_traces = []
    for r in range(n_rounds):
        diff = factory.get_representation('R2_xor_diff', trace_q[r], trace_r[r])
        rand_traces.append(diff)
    rand_traces = np.stack(rand_traces, axis=1)

    return diff_traces, rand_traces


def train_inverter(model, X_input, X_target, n_epochs, batch_size, device, lr=1e-3):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_in_t = torch.from_numpy(X_input).float()
    X_tgt_t = torch.from_numpy(X_target).float()
    dataset = torch.utils.data.TensorDataset(X_in_t, X_tgt_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"        epoch {epoch+1}/{n_epochs} mse={total_loss/n:.6f}", flush=True)

    model.eval()
    with torch.no_grad():
        pred = model(X_in_t.to(device))
        mse = criterion(pred, X_tgt_t.to(device)).item()
    return mse


def train_distinguisher(model, X, Y, n_epochs, batch_size, device, lr=1e-3):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    X_t = torch.from_numpy(X).float()
    Y_t = torch.from_numpy(Y).float()
    dataset = torch.utils.data.TensorDataset(X_t, Y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(n_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_t.to(device)).cpu().numpy()
        acc = float(np.mean((preds > 0.5) == Y))
    return acc


def single_run(seed, args):
    set_seed(seed)
    cipher = get_cipher(args.cipher)
    device = get_device(args)
    block_size = cipher.block_size

    diff_traces, rand_traces = generate_traces(
        args.cipher, cipher, args.rounds, args.samples
    )
    n_train = int(0.8 * (args.samples // 2))

    print(f"    Phase 1: Training round inverters...")
    inverters = {}
    inversion_mse = {}

    for r in range(args.rounds - 1, 0, -1):
        X_in = diff_traces[:n_train, r, :]
        X_tgt = diff_traces[:n_train, r - 1, :]

        print(f"      AE: round {r+1} → round {r}:")
        ae = RoundInverterAE(block_size, bottleneck_dim=args.bottleneck, hidden=256)
        mse = train_inverter(
            ae, X_in, X_tgt,
            n_epochs=args.ae_epochs, batch_size=args.batch_size, device=device
        )
        inverters[r] = ae
        inversion_mse[str(r)] = float(mse)
        print(f"      → MSE = {mse:.6f}")

    print(f"\n    Phase 2: Chaining inverters...")
    n_eval = args.samples // 2

    chain_results = {}

    last_round = args.rounds - 1

    print(f"\n    {'Round':<10} {'Direct Acc':<14} {'Chain Acc':<14} {'Inversion MSE':>14}")
    print(f"    {'─' * 54}")

    cipher_current = torch.from_numpy(
        diff_traces[:n_eval, last_round, :]
    ).float().to(device)
    random_current = torch.from_numpy(
        rand_traces[:n_eval, last_round, :]
    ).float().to(device)

    for target_r in range(last_round, -1, -1):
        X_direct = np.concatenate([
            diff_traces[:n_eval, target_r, :],
            rand_traces[:n_eval, target_r, :]
        ], axis=0)
        Y_direct = np.concatenate([
            np.ones(n_eval), np.zeros(n_eval)
        ])

        direct_dist = SimpleDistinguisher(block_size)
        direct_acc = train_distinguisher(
            direct_dist, X_direct, Y_direct,
            n_epochs=15, batch_size=args.batch_size, device=device
        )

        if target_r == last_round:
            chain_cipher = cipher_current.cpu().numpy()
            chain_random = random_current.cpu().numpy()
        else:
            ae = inverters[target_r + 1]
            ae.eval()
            with torch.no_grad():
                cipher_current = ae(cipher_current)
                random_current = ae(random_current)
            chain_cipher = cipher_current.cpu().numpy()
            chain_random = random_current.cpu().numpy()

        X_chain = np.concatenate([chain_cipher, chain_random], axis=0)
        Y_chain = np.concatenate([np.ones(n_eval), np.zeros(n_eval)])

        chain_dist = SimpleDistinguisher(block_size)
        chain_acc = train_distinguisher(
            chain_dist, X_chain, Y_chain,
            n_epochs=15, batch_size=args.batch_size, device=device
        )

        inv_mse = inversion_mse.get(str(target_r + 1), 0.0)
        print(f"    r={target_r+1:<6} {direct_acc:<14.4f} {chain_acc:<14.4f} {inv_mse:>14.6f}")

        chain_results[str(target_r)] = {
            'direct_acc': float(direct_acc),
            'chain_acc': float(chain_acc),
        }

    return {
        'inversion_mse': inversion_mse,
        'chain_results': chain_results,
    }


def plot_results(all_results, args, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    rounds = sorted([int(k) for k in all_results[0]['chain_results'].keys()])

    direct_means, chain_means = [], []
    direct_stds, chain_stds = [], []

    for r in rounds:
        d_vals = [res['chain_results'][str(r)]['direct_acc'] for res in all_results]
        c_vals = [res['chain_results'][str(r)]['chain_acc'] for res in all_results]
        direct_means.append(np.mean(d_vals))
        chain_means.append(np.mean(c_vals))
        direct_stds.append(np.std(d_vals))
        chain_stds.append(np.std(c_vals))

    round_labels = [r + 1 for r in rounds]

    axes[0].errorbar(round_labels, direct_means, yerr=direct_stds,
                     fmt='bs-', linewidth=2, markersize=8, capsize=4,
                     label='Direct (ground truth)')
    axes[0].errorbar(round_labels, chain_means, yerr=chain_stds,
                     fmt='ro--', linewidth=2, markersize=8, capsize=4,
                     label='Chain-recovered')
    axes[0].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Distinguisher Accuracy')
    axes[0].set_title(f'Signal Recovery via Chained Autoencoders — {args.cipher.upper()}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0.45, 1.05)

    inv_rounds = sorted([int(k) for k in all_results[0]['inversion_mse'].keys()])
    inv_means = []
    inv_stds_list = []
    for r in inv_rounds:
        vals = [res['inversion_mse'][str(r)] for res in all_results]
        inv_means.append(np.mean(vals))
        inv_stds_list.append(np.std(vals))

    axes[1].bar(inv_rounds, inv_means, yerr=inv_stds_list, capsize=4,
                color='#9C27B0', alpha=0.85, edgecolor='black')
    axes[1].set_xlabel('Round Being Inverted')
    axes[1].set_ylabel('Inversion MSE')
    axes[1].set_title('Per-Round Inversion Difficulty')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / f'e17_{args.cipher}_round_inverter.png', dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='E17: Neural Round Inverter — Chained Autoencoders'
    )
    add_common_args(parser)
    parser.add_argument('--rounds', type=int, default=8,
                        help='Number of cipher rounds')
    parser.add_argument('--bottleneck', type=int, default=24,
                        help='AE bottleneck dimension')
    parser.add_argument('--ae-epochs', type=int, default=30,
                        help='Training epochs per autoencoder')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = './results/e17_round_inverter'

    print("=" * 60)
    print("  E17: Neural Round Inverter")
    print("  Chained Autoencoders for Differential Recovery")
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

    rounds = sorted([int(k) for k in all_results[0]['chain_results'].keys()])
    print(f"\n{'═' * 60}")
    print(f"  {'Round':<8} {'Direct':<14} {'Recovered':<14} {'Recovery %':>10}")
    print(f"{'─' * 60}")
    any_recovery = False
    for r in rounds:
        d = np.mean([res['chain_results'][str(r)]['direct_acc'] for res in all_results])
        c = np.mean([res['chain_results'][str(r)]['chain_acc'] for res in all_results])
        recovery_pct = ((c - 0.5) / max(d - 0.5, 1e-6)) * 100
        if c > 0.52 and r < max(rounds):
            any_recovery = True
        print(f"  r={r+1:<5} {d:<14.4f} {c:<14.4f} {recovery_pct:>9.1f}%")
    print(f"{'═' * 60}")

    if any_recovery:
        print(f"\n  ⚡ Chain recovers distinguishing signal from earlier rounds!")
        print(f"     → Neural networks can partially invert SPECK32 rounds.")
    else:
        print(f"\n  ✗ Chain fails to recover signal — round inversion is too lossy.")
        print(f"     → Individual round inversions compound errors too quickly.")

    agg = {}
    for r in rounds:
        agg[str(r)] = {
            'direct_mean': float(np.mean([res['chain_results'][str(r)]['direct_acc'] for res in all_results])),
            'chain_mean': float(np.mean([res['chain_results'][str(r)]['chain_acc'] for res in all_results])),
        }
    save_results(
        {'chain_results': agg, '_seeds': seeds},
        str(output_dir), f'e17_{args.cipher}_results.json'
    )

    plot_results(all_results, args, output_dir)
    print(f"\n✓ Results saved to {output_dir}")


if __name__ == '__main__':
    main()
