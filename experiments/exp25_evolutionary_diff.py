#!/usr/bin/env python3
"""
E25: Evolutionary Input Difference Search

Uses a simple genetic algorithm to find input differences ΔP that
maximize neural distinguisher accuracy. Based on the approach described
in Bellini et al. (ToSC 2023).

Design:
  Population:   50 random ΔP values
  Fitness:      Accuracy of a quick-trained distinguisher (50K samples, 10 epochs)
  Selection:    Tournament (size 5)
  Crossover:    Single-point bitwise crossover
  Mutation:     Bit-flip with rate 1/block_size
  Generations:  30
  Elitism:      Top 5 individuals survive to next generation

The quick training uses fewer samples and epochs than the full experiment
to keep the search tractable (each fitness evaluation takes ~30s on GPU).

Usage:
  python experiments/exp25_evolutionary_diff.py --cipher speck32 --rounds 7
  python experiments/exp25_evolutionary_diff.py --cipher simon32 --rounds 9
  python experiments/exp25_evolutionary_diff.py --cipher present --rounds 6
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ciphers import get_cipher
from data.generator import CipherDataGenerator
from data.dataloader import CryptoDataset, get_input_dim
from models import get_model
from training.trainer import Trainer
from evaluation.metrics import evaluate_model
from experiments.experiment_utils import set_seed, save_results, get_device


def evaluate_delta_p(
    cipher_name: str,
    delta_p: int,
    n_rounds: int,
    device: str,
    n_samples: int = 50000,
    n_epochs: int = 10,
    batch_size: int = 5000,
    seed: int = 42,
) -> float:
    """Quick training + evaluation for a candidate ΔP.
    
    Returns test accuracy. Lower sample count and fewer epochs than
    full experiments to keep evolutionary search tractable.
    """
    set_seed(seed)
    cipher = get_cipher(cipher_name)

    # Skip ΔP = 0 (trivially no signal)
    if delta_p == 0:
        return 0.5

    gen = CipherDataGenerator(
        cipher=cipher_name, n_rounds=n_rounds, delta_p=delta_p, seed=seed
    )
    train_data = gen.generate_balanced_dataset(n_samples, negative_type='gohr')
    val_data = gen.generate_balanced_dataset(n_samples // 5, negative_type='gohr')

    input_dim = get_input_dim('R2_xor_diff', cipher.block_size)
    model = get_model('gohr_mlp', input_dim=input_dim)

    train_ds = CryptoDataset(train_data, 'R2_xor_diff', cipher.block_size)
    val_ds = CryptoDataset(val_data, 'R2_xor_diff', cipher.block_size)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        device=device, use_wandb=False
    )
    # Suppress tqdm output for faster search
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        trainer.train(n_epochs=n_epochs, early_stopping_patience=3, save_best=False)

    metrics = evaluate_model(model, val_loader, device)
    return float(metrics['accuracy'])


def tournament_select(population: list, fitnesses: list, tournament_size: int = 5) -> int:
    """Tournament selection: pick `tournament_size` random individuals,
    return the index of the fittest."""
    indices = np.random.choice(len(population), size=tournament_size, replace=False)
    best = max(indices, key=lambda i: fitnesses[i])
    return best


def crossover(parent1: int, parent2: int, block_size: int) -> int:
    """Single-point bitwise crossover."""
    point = np.random.randint(1, block_size)
    mask_low = (1 << point) - 1
    mask_high = ((1 << block_size) - 1) ^ mask_low
    child = (parent1 & mask_high) | (parent2 & mask_low)
    return child


def mutate(individual: int, block_size: int, mutation_rate: float = None) -> int:
    """Bit-flip mutation with given rate (default: 1/block_size)."""
    if mutation_rate is None:
        mutation_rate = 1.0 / block_size
    for bit in range(block_size):
        if np.random.random() < mutation_rate:
            individual ^= (1 << bit)
    return individual & ((1 << block_size) - 1)


def main():
    parser = argparse.ArgumentParser(
        description='E25: Evolutionary ΔP Search'
    )
    parser.add_argument('--cipher', default='speck32',
                        choices=['speck32', 'simon32', 'present'])
    parser.add_argument('--rounds', type=int, required=True,
                        help='Number of rounds to optimize ΔP for')
    parser.add_argument('--pop-size', type=int, default=50)
    parser.add_argument('--generations', type=int, default=30)
    parser.add_argument('--elite-size', type=int, default=5)
    parser.add_argument('--eval-samples', type=int, default=50000,
                        help='Samples per fitness evaluation (keep low for speed)')
    parser.add_argument('--eval-epochs', type=int, default=10,
                        help='Epochs per fitness evaluation (keep low for speed)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f'./results/e25_evo_diff/{args.cipher}_r{args.rounds}'

    cipher = get_cipher(args.cipher)
    block_size = cipher.block_size
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    np.random.seed(args.seed)

    print("=" * 60)
    print(f"  E25: Evolutionary ΔP Search — {args.cipher.upper()} {args.rounds}r")
    print(f"  Population: {args.pop_size}, Generations: {args.generations}")
    print(f"  Fitness eval: {args.eval_samples} samples × {args.eval_epochs} epochs")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Initialize population ───────────────────────────────────────
    # Include the default ΔP as a seed individual
    population = [cipher.get_default_delta_p()]
    # Add single-bit ΔP values (known good starting points)
    for b in range(block_size):
        if len(population) < args.pop_size // 2:
            population.append(1 << b)
    # Fill rest with random ΔP
    while len(population) < args.pop_size:
        delta = np.random.randint(1, 2**block_size)
        population.append(int(delta))

    # ── Evolve ──────────────────────────────────────────────────────
    best_ever = None
    best_ever_fitness = 0.5
    history = []

    for gen in range(args.generations):
        t0 = time.time()

        # Evaluate fitness
        fitnesses = []
        for i, delta_p in enumerate(population):
            fitness = evaluate_delta_p(
                cipher_name=args.cipher,
                delta_p=delta_p,
                n_rounds=args.rounds,
                device=device,
                n_samples=args.eval_samples,
                n_epochs=args.eval_epochs,
                seed=args.seed + gen * 1000 + i,
            )
            fitnesses.append(fitness)

        # Track best
        gen_best_idx = np.argmax(fitnesses)
        gen_best = population[gen_best_idx]
        gen_best_fitness = fitnesses[gen_best_idx]

        if gen_best_fitness > best_ever_fitness:
            best_ever = gen_best
            best_ever_fitness = gen_best_fitness

        gen_time = time.time() - t0
        gen_mean = np.mean(fitnesses)
        delta_hex = f"0x{gen_best:08x}" if block_size <= 32 else f"0x{gen_best:016x}"

        print(f"  Gen {gen+1:3d}/{args.generations}: "
              f"best={gen_best_fitness:.4f} (ΔP={delta_hex}), "
              f"mean={gen_mean:.4f}, time={gen_time:.0f}s")

        history.append({
            'generation': gen + 1,
            'best_fitness': float(gen_best_fitness),
            'best_delta_p': int(gen_best),
            'mean_fitness': float(gen_mean),
            'best_ever_fitness': float(best_ever_fitness),
        })

        # ── Selection + reproduction ───────────────────────────────
        # Elitism: keep top-k
        elite_indices = np.argsort(fitnesses)[-args.elite_size:]
        new_population = [population[i] for i in elite_indices]

        # Fill rest with tournament selection + crossover + mutation
        while len(new_population) < args.pop_size:
            p1_idx = tournament_select(population, fitnesses)
            p2_idx = tournament_select(population, fitnesses)
            child = crossover(population[p1_idx], population[p2_idx], block_size)
            child = mutate(child, block_size)
            # Ensure non-zero
            if child == 0:
                child = 1
            new_population.append(child)

        population = new_population

    # ── Final evaluation with more samples ──────────────────────────
    print(f"\n{'━' * 50}")
    print("  Final evaluation of top-5 ΔP (200K samples, 20 epochs)")
    print(f"{'━' * 50}")

    # Evaluate top candidates more thoroughly
    final_results = []
    top_deltas = sorted(set([h['best_delta_p'] for h in history[-10:]]),
                        key=lambda d: -max(h['best_fitness'] for h in history
                                          if h['best_delta_p'] == d))[:5]
    # Always include default
    if cipher.get_default_delta_p() not in top_deltas:
        top_deltas.append(cipher.get_default_delta_p())

    for delta_p in top_deltas:
        acc = evaluate_delta_p(
            cipher_name=args.cipher,
            delta_p=delta_p,
            n_rounds=args.rounds,
            device=device,
            n_samples=200000,
            n_epochs=20,
            seed=args.seed,
        )
        delta_hex = f"0x{delta_p:08x}" if block_size <= 32 else f"0x{delta_p:016x}"
        is_default = " (default)" if delta_p == cipher.get_default_delta_p() else ""
        print(f"  ΔP={delta_hex}: acc={acc:.4f}{is_default}")
        final_results.append({
            'delta_p': int(delta_p),
            'delta_hex': delta_hex,
            'accuracy': float(acc),
            'is_default': delta_p == cipher.get_default_delta_p(),
        })

    # ── Plot ────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    gens = [h['generation'] for h in history]
    bests = [h['best_fitness'] for h in history]
    means = [h['mean_fitness'] for h in history]
    best_evers = [h['best_ever_fitness'] for h in history]

    ax1.plot(gens, best_evers, 'g-', linewidth=2, label='Best ever')
    ax1.plot(gens, bests, 'b-', linewidth=1.5, alpha=0.7, label='Gen best')
    ax1.plot(gens, means, 'r--', linewidth=1, alpha=0.5, label='Gen mean')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Evolutionary Search Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bar chart of final results
    labels = [r['delta_hex'][:10] for r in final_results]
    accs = [r['accuracy'] for r in final_results]
    colors = ['#2ecc71' if r['is_default'] else '#3498db' for r in final_results]
    ax2.barh(labels, accs, color=colors, edgecolor='black', alpha=0.85)
    ax2.set_xlabel('Accuracy')
    ax2.set_title('Final ΔP Comparison')
    ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.3)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.suptitle(f'{args.cipher.upper()} {args.rounds}r — Evolutionary ΔP Search',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / f'e25_{args.cipher}_r{args.rounds}_evo.png', dpi=300)
    plt.close()

    save_results(
        {
            'history': history,
            'final_results': final_results,
            'best_ever_delta_p': int(best_ever),
            'best_ever_accuracy': float(best_ever_fitness),
            'default_delta_p': int(cipher.get_default_delta_p()),
        },
        str(output_dir),
        f'e25_{args.cipher}_r{args.rounds}_results.json'
    )
    print(f"\n✓ Results saved to {output_dir}")


if __name__ == '__main__':
    main()
