
import argparse
import time
import json
import numpy as np
import torch
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--n-seeds', type=int, default=5, help='Number of seeds to run')
    parser.add_argument('--cipher', default='speck32',
                        choices=['speck32', 'simon32', 'present', 'simon32_irk'])
    parser.add_argument('--samples', type=int, default=500_000)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=5000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', default=None,
                        help='Override output directory')


def aggregate_results(results_list: List[Dict]) -> Dict:
    if not results_list:
        return {}

    aggregated = {}
    keys = results_list[0].keys()

    for key in keys:
        values = [r[key] for r in results_list if key in r]

        if not values:
            continue

        if isinstance(values[0], (int, float)):
            arr = np.array(values, dtype=float)
            aggregated[key] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'ci95_lower': float(np.percentile(arr, 2.5)),
                'ci95_upper': float(np.percentile(arr, 97.5)),
                'values': [float(v) for v in arr],
            }
        elif isinstance(values[0], dict):
            inner_keys = values[0].keys()
            aggregated[key] = {}
            for ik in inner_keys:
                inner_vals = [v[ik] for v in values if ik in v]
                if inner_vals and isinstance(inner_vals[0], (int, float)):
                    arr = np.array(inner_vals, dtype=float)
                    aggregated[key][ik] = {
                        'mean': float(np.mean(arr)),
                        'std': float(np.std(arr)),
                        'values': [float(v) for v in arr],
                    }
                else:
                    aggregated[key][ik] = inner_vals
        else:
            aggregated[key] = values

    return aggregated


def run_multi_seed(
    experiment_fn: Callable,
    args: argparse.Namespace,
    seeds: Optional[List[int]] = None,
) -> Dict:
    if seeds is None:
        seeds = [args.seed + i for i in range(args.n_seeds)]

    print(f"\n{'═' * 60}")
    print(f"  Running {len(seeds)} seeds: {seeds}")
    print(f"{'═' * 60}")

    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n┌─ Seed {seed} ({i+1}/{len(seeds)}) ─────────────────────────┐")
        t0 = time.time()
        result = experiment_fn(seed, args)
        elapsed = time.time() - t0
        result['_seed'] = seed
        result['_time'] = round(elapsed, 1)
        all_results.append(result)
        print(f"└─ Done in {elapsed:.1f}s ─────────────────────────────────┘")

    aggregated = aggregate_results(all_results)
    aggregated['_seeds'] = seeds
    aggregated['_n_seeds'] = len(seeds)

    return aggregated


def save_results(results: Dict, output_dir: str, filename: str) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {path / filename}")


def get_device(args: argparse.Namespace) -> str:
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("  ⚠ CUDA not available, using CPU")
        return 'cpu'
    return args.device


def quick_train_eval(
    model_name: str,
    input_dim: int,
    train_data: Dict,
    val_data: Dict,
    test_data: Dict,
    representation: str,
    block_size: int,
    batch_size: int,
    n_epochs: int,
    device: str,
) -> Dict:
    from models import get_model
    from data.dataloader import CryptoDataset
    from training.trainer import Trainer
    from evaluation.metrics import evaluate_model
    from torch.utils.data import DataLoader

    model = get_model(model_name, input_dim=input_dim)
    n_params = sum(p.numel() for p in model.parameters())

    train_ds = CryptoDataset(train_data, representation, block_size)
    val_ds = CryptoDataset(val_data, representation, block_size)
    test_ds = CryptoDataset(test_data, representation, block_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        device=device, use_wandb=False
    )

    t0 = time.time()
    trainer.train(n_epochs=n_epochs, early_stopping_patience=5, save_best=False)
    train_time = time.time() - t0

    metrics = evaluate_model(model, test_loader, device)

    model.eval()
    t0 = time.time()
    n_infer = 0
    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(device)
            _ = model(X)
            n_infer += X.shape[0]
    infer_time = time.time() - t0

    return {
        'accuracy': float(metrics['accuracy']),
        'advantage': float(metrics['advantage']),
        'auc': float(metrics.get('auc', 0)),
        'train_time': round(train_time, 2),
        'n_params': n_params,
        'infer_throughput': round(n_infer / max(infer_time, 1e-6)),
    }
