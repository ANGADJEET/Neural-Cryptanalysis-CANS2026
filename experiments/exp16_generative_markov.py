#!/usr/bin/env python
"""
E16: Generative Markov Test — VAE Round Transition Predictor

Train generative models to predict the next round's XOR difference from the
current round's. Compare a 1-step (Markov) model against a 2-step (memory)
model, and visualise the latent space to see how cipher vs random structure
evolves across rounds.

Key idea:
  - Model A (Markov):  predict ΔR_{r+1} from ΔR_r
  - Model B (Memory):  predict ΔR_{r+1} from [ΔR_{r-1}, ΔR_r]
  If B has significantly lower reconstruction error than A, the differential
  propagation carries exploitable cross-round memory.

Additionally, encode each round's ΔR into a VAE latent space and plot
t-SNE / PCA coloured by label (cipher vs random) to visualise signal decay.

Usage:
    python experiments/exp16_generative_markov.py --cipher speck32 --rounds 8
    python experiments/exp16_generative_markov.py --cipher speck32 --rounds 8 --n-seeds 5
"""

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
from sklearn.decomposition import PCA

from ciphers import get_cipher
from data.representations import RepresentationFactory
from experiments.experiment_utils import (
    set_seed, add_common_args, save_results, get_device
)


# ═══════════════════════════════════════════════════════════════════
#  Models
# ═══════════════════════════════════════════════════════════════════

class RoundPredictor(nn.Module):
    """MLP that predicts ΔR_{r+1} from input (either ΔR_r or [ΔR_{r-1}, ΔR_r])."""

    def __init__(self, input_dim, output_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
            nn.Sigmoid(),  # output is bits in [0, 1]
        )

    def forward(self, x):
        return self.net(x)


class RoundVAE(nn.Module):
    """
    VAE that encodes ΔR_r into a latent space and decodes to ΔR_{r+1}.
    The latent space captures the "essence" of the differential at each round.
    """

    def __init__(self, input_dim, output_dim, latent_dim=16, hidden=256):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: ΔR_r → (μ, log σ²)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

        # Decoder: z → ΔR_{r+1}
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def get_latent(self, x):
        """Get latent representation (mean) without sampling."""
        mu, _ = self.encode(x)
        return mu


def vae_loss(recon, target, mu, logvar, beta=0.1):
    """VAE loss = reconstruction (BCE) + β * KL divergence."""
    bce = nn.functional.binary_cross_entropy(recon, target, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + beta * kld


# ═══════════════════════════════════════════════════════════════════
#  Data Generation
# ═══════════════════════════════════════════════════════════════════

def generate_differential_traces(cipher_name, cipher, n_rounds, n_samples):
    """Generate XOR-difference traces for differential pairs."""
    key = cipher.random_key()
    delta_p = cipher.get_default_delta_p()
    factory = RepresentationFactory(block_size=cipher.block_size)

    P = cipher.random_plaintexts(n_samples)
    P_prime = (P ^ delta_p).astype(P.dtype)
    _, trace1 = cipher.encrypt_with_trace(P, n_rounds, key)
    _, trace2 = cipher.encrypt_with_trace(P_prime, n_rounds, key)

    traces = []
    for r in range(n_rounds):
        diff = factory.get_representation('R2_xor_diff', trace1[r], trace2[r])
        traces.append(diff)

    return np.stack(traces, axis=1)  # (n_samples, n_rounds, block_size)


def generate_random_traces(cipher_name, cipher, n_rounds, n_samples):
    """Generate XOR-difference traces for independent random pairs."""
    key = cipher.random_key()
    factory = RepresentationFactory(block_size=cipher.block_size)

    Q = cipher.random_plaintexts(n_samples)
    R = cipher.random_plaintexts(n_samples)
    _, trace_q = cipher.encrypt_with_trace(Q, n_rounds, key)
    _, trace_r = cipher.encrypt_with_trace(R, n_rounds, key)

    traces = []
    for r in range(n_rounds):
        diff = factory.get_representation('R2_xor_diff', trace_q[r], trace_r[r])
        traces.append(diff)

    return np.stack(traces, axis=1)


# ═══════════════════════════════════════════════════════════════════
#  Training Utilities
# ═══════════════════════════════════════════════════════════════════

def train_predictor(model, X_in, X_target, n_epochs, batch_size, device, lr=1e-3):
    """Train a round predictor (MLP) and return final MSE."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_in_t = torch.from_numpy(X_in).float()
    X_target_t = torch.from_numpy(X_target).float()

    dataset = torch.utils.data.TensorDataset(X_in_t, X_target_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    n_total = X_in.shape[0]
    for epoch in range(n_epochs):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"      epoch {epoch+1}/{n_epochs} loss={total_loss/n_total:.6f}", flush=True)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        all_pred = model(X_in_t.to(device))
        mse = criterion(all_pred, X_target_t.to(device)).item()

    return mse


def train_vae(model, X_in, X_target, n_epochs, batch_size, device, lr=1e-3):
    """Train a VAE and return final reconstruction MSE and the model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_in_t = torch.from_numpy(X_in).float()
    X_target_t = torch.from_numpy(X_target).float()

    dataset = torch.utils.data.TensorDataset(X_in_t, X_target_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    n_total = X_in.shape[0]
    for epoch in range(n_epochs):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            recon, mu, logvar = model(xb)
            loss = vae_loss(recon, yb, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"      VAE epoch {epoch+1}/{n_epochs} loss={total_loss/n_total:.4f}", flush=True)

    # Final MSE
    model.eval()
    with torch.no_grad():
        all_in = X_in_t.to(device)
        all_target = X_target_t.to(device)
        recon, _, _ = model(all_in)
        mse = nn.functional.mse_loss(recon, all_target).item()

    return mse, model


# ═══════════════════════════════════════════════════════════════════
#  Main Experiment
# ═══════════════════════════════════════════════════════════════════

def single_run(seed, args):
    """One seed: compare Markov vs Memory predictors across rounds."""
    set_seed(seed)
    cipher = get_cipher(args.cipher)
    device = get_device(args)
    block_size = cipher.block_size

    # Generate traces
    diff_traces = generate_differential_traces(
        args.cipher, cipher, args.rounds, args.samples
    )
    rand_traces = generate_random_traces(
        args.cipher, cipher, args.rounds, args.samples
    )

    n_train = int(0.8 * args.samples)
    results = {'markov_mse': {}, 'memory_mse': {}, 'vae_mse': {}}

    print(f"\n    {'Round':<8} {'Markov MSE':<14} {'Memory MSE':<14} {'Ratio':>8}")
    print(f"    {'─' * 46}")

    for r in range(1, args.rounds - 1):
        # ─── Prepare data ─────────────────────────────────────────
        # Both models get 64-dim input for a controlled comparison:
        # Markov: [ΔR_r, ΔR_r] (duplicated — no history info)
        X_markov = np.concatenate([
            diff_traces[:n_train, r, :],
            diff_traces[:n_train, r, :]
        ], axis=1)
        # Memory: [ΔR_{r-1}, ΔR_r] (actual history)
        X_memory = np.concatenate([
            diff_traces[:n_train, r - 1, :],
            diff_traces[:n_train, r, :]
        ], axis=1)
        # Target: ΔR_{r+1}
        Y_target = diff_traces[:n_train, r + 1, :]

        # ─── Train Markov predictor (same architecture as Memory) ─
        model_a = RoundPredictor(2 * block_size, block_size, hidden=256)
        mse_a = train_predictor(
            model_a, X_markov, Y_target,
            n_epochs=args.pred_epochs, batch_size=args.batch_size, device=device
        )

        # ─── Train Memory predictor ───────────────────────────────
        model_b = RoundPredictor(2 * block_size, block_size, hidden=256)
        mse_b = train_predictor(
            model_b, X_memory, Y_target,
            n_epochs=args.pred_epochs, batch_size=args.batch_size, device=device
        )

        ratio = mse_b / max(mse_a, 1e-10)
        results['markov_mse'][str(r)] = float(mse_a)
        results['memory_mse'][str(r)] = float(mse_b)

        print(f"    r={r}→{r+1}   {mse_a:<14.6f} {mse_b:<14.6f} {ratio:>8.4f}")

    # ─── VAE Latent Space Analysis ────────────────────────────────
    print(f"\n    Training VAE for latent space visualization...")

    vae = RoundVAE(block_size, block_size, latent_dim=args.latent_dim, hidden=256)
    # Train VAE on round 1→2 (where signal is strongest)
    X_vae_in = diff_traces[:n_train, 0, :]
    X_vae_target = diff_traces[:n_train, 1, :]
    vae_mse, vae_model = train_vae(
        vae, X_vae_in, X_vae_target,
        n_epochs=args.pred_epochs, batch_size=args.batch_size, device=device
    )
    results['vae_mse'] = float(vae_mse)

    # Collect latent representations at each round for both cipher and random
    latent_data = {}
    vae_model.eval()
    n_vis = min(5000, args.samples)
    with torch.no_grad():
        for r in range(args.rounds):
            cipher_z = vae_model.get_latent(
                torch.from_numpy(diff_traces[:n_vis, r, :]).float().to(device)
            ).cpu().numpy()
            random_z = vae_model.get_latent(
                torch.from_numpy(rand_traces[:n_vis, r, :]).float().to(device)
            ).cpu().numpy()
            latent_data[r] = {'cipher': cipher_z, 'random': random_z}

    results['_latent_data'] = latent_data

    return results


def plot_results(all_results, args, output_dir):
    """Create publication plots."""
    # ─── Plot 1: Markov vs Memory MSE ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Average across seeds
    rounds = sorted([int(k) for k in all_results[0]['markov_mse'].keys()])
    markov_means = []
    memory_means = []
    ratios = []

    for r in rounds:
        m_vals = [res['markov_mse'][str(r)] for res in all_results]
        mem_vals = [res['memory_mse'][str(r)] for res in all_results]
        markov_means.append(np.mean(m_vals))
        memory_means.append(np.mean(mem_vals))
        ratios.append(np.mean(mem_vals) / max(np.mean(m_vals), 1e-10))

    x_labels = [f'{r}→{r+1}' for r in rounds]

    # Bar chart comparing MSE
    x_pos = np.arange(len(rounds))
    width = 0.35
    bars1 = axes[0].bar(x_pos - width/2, markov_means, width,
                        label='Markov (ΔR_r only)', color='#2196F3', alpha=0.85)
    bars2 = axes[0].bar(x_pos + width/2, memory_means, width,
                        label='Memory (ΔR_{r-1}, ΔR_r)', color='#FF5722', alpha=0.85)
    axes[0].set_xlabel('Round Transition')
    axes[0].set_ylabel('Reconstruction MSE')
    axes[0].set_title(f'Round Prediction: Markov vs Memory — {args.cipher.upper()}')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(x_labels)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Ratio plot
    axes[1].plot(rounds, ratios, 'go-', linewidth=2, markersize=8)
    axes[1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No improvement')
    axes[1].set_xlabel('Source Round')
    axes[1].set_ylabel('Memory MSE / Markov MSE')
    axes[1].set_title('Memory Advantage Ratio (< 1.0 = memory helps)')
    axes[1].set_xticks(rounds)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'e16_{args.cipher}_prediction.png', dpi=300)
    plt.close()

    # ─── Plot 2: Latent Space Visualization (PCA) ─────────────────
    latent_data = all_results[0].get('_latent_data', {})
    if latent_data:
        n_rounds_to_show = min(6, len(latent_data))
        fig, axes = plt.subplots(1, n_rounds_to_show, figsize=(4 * n_rounds_to_show, 4))

        if n_rounds_to_show == 1:
            axes = [axes]

        for idx, r in enumerate(range(n_rounds_to_show)):
            if r not in latent_data:
                continue
            cipher_z = latent_data[r]['cipher']
            random_z = latent_data[r]['random']

            # PCA to 2D
            all_z = np.concatenate([cipher_z, random_z], axis=0)
            pca = PCA(n_components=2)
            all_2d = pca.fit_transform(all_z)

            n_c = cipher_z.shape[0]
            axes[idx].scatter(all_2d[n_c:, 0], all_2d[n_c:, 1],
                            s=3, alpha=0.3, c='gray', label='Random')
            axes[idx].scatter(all_2d[:n_c, 0], all_2d[:n_c, 1],
                            s=3, alpha=0.3, c='red', label='Cipher')
            axes[idx].set_title(f'Round {r + 1}', fontsize=11)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
            if idx == 0:
                axes[idx].legend(markerscale=4, fontsize=8)

        fig.suptitle(f'VAE Latent Space — {args.cipher.upper()} (Cipher vs Random)',
                     fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / f'e16_{args.cipher}_latent.png', dpi=300,
                    bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='E16: Generative Markov Test — VAE Round Transition Predictor'
    )
    add_common_args(parser)
    parser.add_argument('--rounds', type=int, default=8,
                        help='Number of cipher rounds to trace')
    parser.add_argument('--latent-dim', type=int, default=16,
                        help='VAE latent dimension')
    parser.add_argument('--pred-epochs', type=int, default=30,
                        help='Training epochs for predictors')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = './results/e16_generative_markov'

    print("=" * 60)
    print("  E16: Generative Markov Test")
    print("  VAE Round Transition Predictor")
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

    # Aggregate prediction results
    rounds = sorted([int(k) for k in all_results[0]['markov_mse'].keys()])
    agg = {}
    for r in rounds:
        m_vals = [res['markov_mse'][str(r)] for res in all_results]
        mem_vals = [res['memory_mse'][str(r)] for res in all_results]
        ratio_vals = [mem / max(m, 1e-10) for m, mem in zip(m_vals, mem_vals)]
        agg[str(r)] = {
            'markov_mse_mean': float(np.mean(m_vals)),
            'markov_mse_std': float(np.std(m_vals)),
            'memory_mse_mean': float(np.mean(mem_vals)),
            'memory_mse_std': float(np.std(mem_vals)),
            'ratio_mean': float(np.mean(ratio_vals)),
            'ratio_std': float(np.std(ratio_vals)),
        }

    # Print summary
    print(f"\n{'═' * 65}")
    print(f"  {'Transition':<12} {'Markov MSE':<18} {'Memory MSE':<18} {'Ratio':>8}")
    print(f"{'─' * 65}")
    for r in rounds:
        a = agg[str(r)]
        print(f"  r={r}→{r+1}       "
              f"{a['markov_mse_mean']:.6f}±{a['markov_mse_std']:.4f}  "
              f"{a['memory_mse_mean']:.6f}±{a['memory_mse_std']:.4f}  "
              f"{a['ratio_mean']:>8.4f}")
    print(f"{'═' * 65}")

    # Interpretation
    avg_ratio = np.mean([agg[str(r)]['ratio_mean'] for r in rounds])
    if avg_ratio < 0.95:
        print(f"\n  ⚡ Memory model achieves {(1-avg_ratio)*100:.1f}% lower MSE on average.")
        print(f"     → Evidence AGAINST strict Markov assumption.")
    else:
        print(f"\n  ✓ Memory model offers negligible improvement (ratio={avg_ratio:.4f}).")
        print(f"     → Consistent with Markov assumption.")

    # Save (remove non-serializable latent data from JSON)
    save_data = {
        'round_predictions': agg,
        'vae_mse_values': [float(r.get('vae_mse', 0)) for r in all_results],
        '_seeds': seeds,
        '_n_seeds': len(seeds),
    }
    save_results(save_data, str(output_dir), f'e16_{args.cipher}_results.json')

    # Plot
    plot_results(all_results, args, output_dir)

    print(f"\n✓ Results and plots saved to {output_dir}")


if __name__ == '__main__':
    main()
