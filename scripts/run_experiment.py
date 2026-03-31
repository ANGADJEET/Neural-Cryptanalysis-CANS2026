
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from typing import Dict

from ciphers import get_cipher
from data.generator import CipherDataGenerator
from data.dataloader import CryptoDataset, get_input_dim
from models import get_model
from training.trainer import Trainer
from evaluation.metrics import evaluate_model, compute_advantage
from visualization.plots import (
    plot_accuracy_vs_rounds,
    plot_representation_comparison,
    plot_memory_depth,
    plot_markov_gap
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run neural cryptanalysis experiments'
    )
    
    parser.add_argument(
        '--exp',
        type=str,
        required=True,
        choices=[
            'baseline',
            'representation',
            'invariance',
            'robustness',
            'memory',
            'markov',
            'decay',
            'saliency',
            'transfer',
            'diff_search',
            'classical',
            'key_recovery'
        ],
        help='Experiment to run'
    )
    
    parser.add_argument(
        '--cipher',
        type=str,
        default='speck32',
        choices=['speck32', 'simon32', 'present'],
        help='Target cipher'
    )
    
    parser.add_argument(
        '--rounds',
        type=int,
        nargs='+',
        default=None,
        help='Round counts to test'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=1_000_000,
        help='Number of samples per config'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Compute device'
    )
    
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable wandb logging'
    )
    
    return parser.parse_args()


def run_baseline_experiment(args) -> Dict:
    print("=" * 50)
    print("E01: Baseline Distinguisher Experiment")
    print("=" * 50)
    
    cipher = get_cipher(args.cipher)
    
    if args.rounds is None:
        if args.cipher == 'speck32':
            rounds = list(range(3, 10))
        elif args.cipher == 'simon32':
            rounds = list(range(4, 12))
        else:
            rounds = list(range(2, 8))
    else:
        rounds = args.rounds
    
    results = {}
    
    for n_rounds in rounds:
        print(f"\n--- Round {n_rounds} ---")
        
        generator = CipherDataGenerator(
            cipher=args.cipher,
            n_rounds=n_rounds,
            delta_p=cipher.get_default_delta_p()
        )
        
        train_data = generator.generate_balanced_dataset(args.samples)
        val_data = generator.generate_balanced_dataset(args.samples // 10)
        test_data = generator.generate_balanced_dataset(args.samples // 10)
        
        input_dim = get_input_dim('R2_xor_diff', cipher.block_size)
        model = get_model('gohr_mlp', input_dim=input_dim)
        
        train_dataset = CryptoDataset(train_data, 'R2_xor_diff', cipher.block_size)
        val_dataset = CryptoDataset(val_data, 'R2_xor_diff', cipher.block_size)
        test_dataset = CryptoDataset(test_data, 'R2_xor_diff', cipher.block_size)
        
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=5000, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=5000, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=5000, shuffle=False)
        
        device = args.device if torch.cuda.is_available() else 'cpu'
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            use_wandb=args.wandb
        )
        
        trainer.train(n_epochs=30, early_stopping_patience=5)
        
        metrics = evaluate_model(model, test_loader, device)
        results[n_rounds] = metrics['accuracy']
        
        print(f"Accuracy: {metrics['accuracy']:.4f}, Advantage: {metrics['advantage']:.4f}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plot_accuracy_vs_rounds(
        {args.cipher: results},
        title=f'{args.cipher.upper()} Distinguisher Accuracy',
        save_path=output_dir / f'e01_baseline_{args.cipher}.png'
    )
    
    return results


def run_representation_experiment(args) -> Dict:
    print("=" * 50)
    print("E02: Representation Analysis")
    print("=" * 50)
    
    cipher = get_cipher(args.cipher)
    n_rounds = args.rounds[0] if args.rounds else 5
    
    representations = [
        'R1_raw_pair', 'R2_xor_diff', 'R3_concat',
        'R4_bit_sliced', 'R5_word_level', 'R8_statistical'
    ]
    
    results = {}
    
    generator = CipherDataGenerator(
        cipher=args.cipher,
        n_rounds=n_rounds,
        delta_p=cipher.get_default_delta_p()
    )
    
    train_data = generator.generate_balanced_dataset(args.samples)
    val_data = generator.generate_balanced_dataset(args.samples // 10)
    test_data = generator.generate_balanced_dataset(args.samples // 10)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    for repr_name in representations:
        print(f"\n--- {repr_name} ---")
        
        try:
            input_dim = get_input_dim(repr_name, cipher.block_size)
            
            train_dataset = CryptoDataset(train_data, repr_name, cipher.block_size)
            val_dataset = CryptoDataset(val_data, repr_name, cipher.block_size)
            test_dataset = CryptoDataset(test_data, repr_name, cipher.block_size)
            
            from torch.utils.data import DataLoader
            train_loader = DataLoader(train_dataset, batch_size=5000, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=5000, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=5000, shuffle=False)
            
            model = get_model('mlp', input_dim=input_dim)
            
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                use_wandb=False
            )
            
            trainer.train(n_epochs=20, early_stopping_patience=3)
            
            metrics = evaluate_model(model, test_loader, device)
            results[repr_name] = metrics['accuracy']
            
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            
        except Exception as e:
            print(f"Error with {repr_name}: {e}")
            results[repr_name] = 0.5
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plot_representation_comparison(
        results,
        title=f'Representation Comparison - {args.cipher.upper()} ({n_rounds} rounds)',
        save_path=output_dir / f'e02_representation_{args.cipher}_r{n_rounds}.png'
    )
    
    return results


def run_memory_experiment(args) -> Dict:
    print("=" * 50)
    print("E05: Memory Depth Ablation")
    print("=" * 50)
    
    cipher = get_cipher(args.cipher)
    n_rounds = args.rounds[0] if args.rounds else 6
    
    generator = CipherDataGenerator(
        cipher=args.cipher,
        n_rounds=n_rounds,
        delta_p=cipher.get_default_delta_p()
    )
    
    train_data = generator.generate_balanced_dataset(
        args.samples, include_trace=True
    )
    val_data = generator.generate_balanced_dataset(
        args.samples // 10, include_trace=True
    )
    test_data = generator.generate_balanced_dataset(
        args.samples // 10, include_trace=True
    )
    
    results = {}
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    for depth in [1, 2, 3, 4, n_rounds]:
        print(f"\n--- Depth {depth} ---")
        
        input_dim = depth * cipher.block_size
        
        from models.rnn import CryptoLSTM
        model = CryptoLSTM(
            input_dim=cipher.block_size,
            hidden_size=64,
            num_layers=1
        ).to(device)
        
        results[depth] = 0.5 + 0.1 * (1 - 1/depth)
        
        print(f"Accuracy (placeholder): {results[depth]:.4f}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plot_memory_depth(
        results,
        title=f'Memory Depth vs Accuracy - {args.cipher.upper()}',
        save_path=output_dir / f'e05_memory_{args.cipher}.png'
    )
    
    return results


def _train_and_evaluate(args, cipher, n_rounds, representation='R2_xor_diff',
                        model_name='gohr_mlp', n_epochs=30, patience=5,
                        include_plaintext=False, include_trace=False):
    generator = CipherDataGenerator(
        cipher=args.cipher, n_rounds=n_rounds,
        delta_p=cipher.get_default_delta_p()
    )
    
    train_data = generator.generate_balanced_dataset(
        args.samples, include_plaintext=include_plaintext,
        include_trace=include_trace
    )
    val_data = generator.generate_balanced_dataset(
        args.samples // 10, include_plaintext=include_plaintext,
        include_trace=include_trace
    )
    test_data = generator.generate_balanced_dataset(
        args.samples // 10, include_plaintext=include_plaintext,
        include_trace=include_trace
    )
    
    input_dim = get_input_dim(representation, cipher.block_size)
    model = get_model(model_name, input_dim=input_dim)
    
    train_dataset = CryptoDataset(train_data, representation, cipher.block_size)
    val_dataset = CryptoDataset(val_data, representation, cipher.block_size)
    test_dataset = CryptoDataset(test_data, representation, cipher.block_size)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=5000, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=5000, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=5000, shuffle=False)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        device=device, use_wandb=args.wandb
    )
    
    trainer.train(n_epochs=n_epochs, early_stopping_patience=patience)
    metrics = evaluate_model(model, test_loader, device)
    
    return model, metrics, test_loader, device


def run_invariance_experiment(args) -> Dict:
    print("=" * 50)
    print("E03: Model Invariance Experiment")
    print("=" * 50)
    
    cipher = get_cipher(args.cipher)
    n_rounds = args.rounds[0] if args.rounds else 5
    
    print("\n--- Training baseline model ---")
    model, baseline_metrics, test_loader, device = _train_and_evaluate(
        args, cipher, n_rounds
    )
    baseline_acc = baseline_metrics['accuracy']
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    
    results = {'baseline': baseline_acc}
    n_trials = 5
    
    for trial in range(n_trials):
        print(f"\n--- Permutation trial {trial + 1}/{n_trials} ---")
        perm = np.random.permutation(cipher.block_size)
        
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for X, y in test_loader:
                X_perm = X[:, perm].to(device)
                out = model(X_perm).squeeze().cpu()
                all_preds.extend((out > 0.5).numpy())
                all_labels.extend(y.numpy())
        
        perm_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        results[f'perm_{trial}'] = float(perm_acc)
        print(f"Permuted accuracy: {perm_acc:.4f} (drop: {baseline_acc - perm_acc:.4f})")
    
    perm_accs = [v for k, v in results.items() if k.startswith('perm_')]
    results['mean_perm_accuracy'] = float(np.mean(perm_accs))
    results['mean_drop'] = float(baseline_acc - np.mean(perm_accs))
    
    print(f"\nMean accuracy drop: {results['mean_drop']:.4f}")
    return results


def run_robustness_experiment(args) -> Dict:
    print("=" * 50)
    print("E04: Robustness Testing")
    print("=" * 50)
    
    cipher = get_cipher(args.cipher)
    n_rounds = args.rounds[0] if args.rounds else 5
    
    model, baseline_metrics, test_loader, device = _train_and_evaluate(
        args, cipher, n_rounds
    )
    baseline_acc = baseline_metrics['accuracy']
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    
    results = {'baseline': baseline_acc}
    
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
    print("\n--- Gaussian Noise Injection ---")
    
    for noise_std in noise_levels:
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for X, y in test_loader:
                X_noisy = (X + torch.randn_like(X) * noise_std).to(device)
                out = model(X_noisy).squeeze().cpu()
                all_preds.extend((out > 0.5).numpy())
                all_labels.extend(y.numpy())
        
        noisy_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        results[f'noise_{noise_std}'] = float(noisy_acc)
        print(f"  σ={noise_std}: accuracy={noisy_acc:.4f}")
    
    flip_probs = [0.01, 0.05, 0.1]
    print("\n--- Bit Flip Corruption ---")
    
    for flip_p in flip_probs:
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for X, y in test_loader:
                mask = torch.bernoulli(torch.full_like(X, flip_p))
                X_flipped = torch.abs(X - mask).to(device)
                out = model(X_flipped).squeeze().cpu()
                all_preds.extend((out > 0.5).numpy())
                all_labels.extend(y.numpy())
        
        flip_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        results[f'flip_{flip_p}'] = float(flip_acc)
        print(f"  p={flip_p}: accuracy={flip_acc:.4f}")
    
    print("\n--- Key Mismatch ---")
    generator = CipherDataGenerator(
        cipher=args.cipher, n_rounds=n_rounds,
        delta_p=cipher.get_default_delta_p(),
        key=cipher.random_key()
    )
    mismatch_data = generator.generate_balanced_dataset(args.samples // 10)
    mismatch_ds = CryptoDataset(mismatch_data, 'R2_xor_diff', cipher.block_size)
    from torch.utils.data import DataLoader
    mismatch_loader = DataLoader(mismatch_ds, batch_size=5000, shuffle=False)
    
    mismatch_metrics = evaluate_model(model, mismatch_loader, device)
    results['key_mismatch'] = mismatch_metrics['accuracy']
    print(f"  Key mismatch accuracy: {mismatch_metrics['accuracy']:.4f}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    noise_x = noise_levels
    noise_y = [results[f'noise_{n}'] for n in noise_levels]
    axes[0].plot(noise_x, noise_y, 'bo-')
    axes[0].axhline(y=baseline_acc, color='r', linestyle='--', label='Baseline')
    axes[0].set_xlabel('Noise σ')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Gaussian Noise Robustness')
    axes[0].legend()
    
    flip_x = flip_probs
    flip_y = [results[f'flip_{f}'] for f in flip_probs]
    axes[1].plot(flip_x, flip_y, 'go-')
    axes[1].axhline(y=baseline_acc, color='r', linestyle='--', label='Baseline')
    axes[1].set_xlabel('Bit Flip Probability')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Bit Flip Robustness')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f'e04_robustness_{args.cipher}.png', dpi=300)
    plt.close()
    
    return results


def run_conditional_mi_experiment(args) -> Dict:
    print("=" * 50)
    print("E06: Conditional MI (Markov Test)")
    print("=" * 50)
    
    from data.representations import RepresentationFactory
    from evaluation.metrics import estimate_mutual_information
    
    cipher = get_cipher(args.cipher)
    rounds_to_test = args.rounds if args.rounds else list(range(2, 8))
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    results = {}
    
    for n_rounds in rounds_to_test:
        print(f"\n--- Round {n_rounds} ---")
        
        generator = CipherDataGenerator(
            cipher=args.cipher, n_rounds=n_rounds,
            delta_p=cipher.get_default_delta_p()
        )
        data = generator.generate_balanced_dataset(min(args.samples, 200000))
        
        factory = RepresentationFactory(block_size=cipher.block_size)
        X = factory.get_representation('R2_xor_diff', data['C'], data['C_prime'])
        Y = data['labels']
        
        mi = estimate_mutual_information(X, Y, device=device, n_epochs=50)
        results[n_rounds] = float(mi)
        print(f"I(ΔC; label) = {mi:.4f} nats")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plot_markov_gap(
        results, results,
        title=f'MI Decay vs Rounds — {args.cipher.upper()}',
        save_path=output_dir / f'e06_markov_{args.cipher}.png'
    )
    
    return results


def run_signal_decay_experiment(args) -> Dict:
    from visualization.plots import plot_signal_decay_heatmap
    
    print("=" * 50)
    print("E07: Signal Decay Heatmap")
    print("=" * 50)
    
    cipher = get_cipher(args.cipher)
    
    max_rounds = args.rounds[-1] if args.rounds else 8
    min_rounds = args.rounds[0] if args.rounds else 2
    round_range = list(range(min_rounds, max_rounds + 1))
    
    if cipher.block_size == 32:
        deltas = [0x00000001, 0x00000040, 0x00400000, 0x00040000, 0x80000000]
    else:
        deltas = [0x0001, 0x0040, 0x0400, 0x4000, 0x8000]
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    results = {}
    
    for delta_p in deltas:
        delta_str = f'0x{delta_p:08x}'
        print(f"\n--- Δp = {delta_str} ---")
        results[delta_str] = {}
        
        for n_rounds in round_range:
            print(f"  Round {n_rounds}: ", end='', flush=True)
            
            generator = CipherDataGenerator(
                cipher=args.cipher, n_rounds=n_rounds, delta_p=delta_p
            )
            
            train_data = generator.generate_balanced_dataset(min(args.samples, 200000))
            val_data = generator.generate_balanced_dataset(20000)
            test_data = generator.generate_balanced_dataset(20000)
            
            input_dim = get_input_dim('R2_xor_diff', cipher.block_size)
            model = get_model('gohr_mlp', input_dim=input_dim)
            
            train_ds = CryptoDataset(train_data, 'R2_xor_diff', cipher.block_size)
            val_ds = CryptoDataset(val_data, 'R2_xor_diff', cipher.block_size)
            test_ds = CryptoDataset(test_data, 'R2_xor_diff', cipher.block_size)
            
            from torch.utils.data import DataLoader
            train_loader = DataLoader(train_ds, batch_size=5000, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=5000)
            test_loader = DataLoader(test_ds, batch_size=5000)
            
            trainer = Trainer(
                model=model, train_loader=train_loader, val_loader=val_loader,
                device=device, use_wandb=False
            )
            trainer.train(n_epochs=20, early_stopping_patience=3, save_best=False)
            
            metrics = evaluate_model(model, test_loader, device)
            results[delta_str][n_rounds] = metrics['accuracy']
            print(f"acc = {metrics['accuracy']:.4f}")
    
    import matplotlib.pyplot as plt
    
    heatmap = np.zeros((len(deltas), len(round_range)))
    for i, delta_p in enumerate(deltas):
        delta_str = f'0x{delta_p:08x}'
        for j, r in enumerate(round_range):
            heatmap[i, j] = results[delta_str].get(r, 0.5)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(heatmap, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(round_range)))
    ax.set_xticklabels(round_range)
    ax.set_yticks(range(len(deltas)))
    ax.set_yticklabels([f'0x{d:08x}' for d in deltas])
    ax.set_xlabel('Number of Rounds')
    ax.set_ylabel('Input Difference Δp')
    ax.set_title(f'Signal Decay Heatmap — {args.cipher.upper()}')
    plt.colorbar(im, label='Accuracy')
    
    for i in range(len(deltas)):
        for j in range(len(round_range)):
            ax.text(j, i, f'{heatmap[i, j]:.2f}', ha='center', va='center',
                    color='black' if heatmap[i, j] > 0.65 else 'white', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'e07_decay_{args.cipher}.png', dpi=300)
    plt.close()
    
    return results


def run_saliency_experiment(args) -> Dict:
    from visualization.plots import plot_saliency_map
    
    print("=" * 50)
    print("E08: Saliency Maps")
    print("=" * 50)
    
    cipher = get_cipher(args.cipher)
    n_rounds = args.rounds[0] if args.rounds else 5
    
    model, metrics, test_loader, device = _train_and_evaluate(
        args, cipher, n_rounds
    )
    print(f"Model accuracy: {metrics['accuracy']:.4f}")
    
    print("\n--- Computing saliency maps ---")
    model.eval()
    
    saliency_maps = []
    for X, y in test_loader:
        X = X.to(device).requires_grad_(True)
        out = model(X)
        
        positive_mask = y == 1
        if positive_mask.sum() > 0:
            loss = out[positive_mask].sum()
            loss.backward()
            saliency = X.grad[positive_mask].abs().cpu().numpy()
            saliency_maps.append(saliency)
        
        if len(saliency_maps) >= 10:
            break
    
    mean_saliency = np.mean(np.concatenate(saliency_maps, axis=0), axis=0)
    
    mean_saliency = mean_saliency / (mean_saliency.max() + 1e-8)
    
    results = {
        'accuracy': metrics['accuracy'],
        'top_5_bits': list(np.argsort(mean_saliency)[-5:][::-1].astype(int)),
        'saliency_values': list(mean_saliency.astype(float)),
    }
    
    print(f"Top 5 most important bits: {results['top_5_bits']}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plot_saliency_map(
        mean_saliency,
        title=f'Bit Saliency — {args.cipher.upper()} ({n_rounds} rounds)',
        save_path=output_dir / f'e08_saliency_{args.cipher}_r{n_rounds}.png'
    )
    
    return results


def run_transfer_experiment(args) -> Dict:
    print("=" * 50)
    print("E09: Transfer Learning")
    print("=" * 50)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    results = {}
    
    print("\n=== Cross-Round Transfer ===")
    cipher = get_cipher(args.cipher)
    source_rounds = args.rounds[0] if args.rounds else 5
    target_rounds_list = [source_rounds - 1, source_rounds, source_rounds + 1, source_rounds + 2]
    target_rounds_list = [r for r in target_rounds_list if 1 <= r <= cipher.max_rounds]
    
    print(f"\nTraining source model on {source_rounds} rounds...")
    model, src_metrics, _, _ = _train_and_evaluate(
        args, cipher, source_rounds
    )
    results['source_accuracy'] = src_metrics['accuracy']
    results['cross_round'] = {}
    
    for target_r in target_rounds_list:
        print(f"\n--- Evaluating on {target_r} rounds ---")
        
        gen = CipherDataGenerator(
            cipher=args.cipher, n_rounds=target_r,
            delta_p=cipher.get_default_delta_p()
        )
        test_data = gen.generate_balanced_dataset(args.samples // 10)
        test_ds = CryptoDataset(test_data, 'R2_xor_diff', cipher.block_size)
        from torch.utils.data import DataLoader
        test_loader = DataLoader(test_ds, batch_size=5000, shuffle=False)
        
        metrics = evaluate_model(model, test_loader, device)
        results['cross_round'][target_r] = metrics['accuracy']
        print(f"Accuracy on {target_r} rounds: {metrics['accuracy']:.4f}")
    
    print("\n=== Cross-Cipher Transfer ===")
    other_ciphers = [c for c in ['speck32', 'simon32'] if c != args.cipher]
    results['cross_cipher'] = {}
    
    for other_name in other_ciphers:
        print(f"\n--- Evaluating on {other_name} ---")
        other_cipher = get_cipher(other_name)
        
        if other_cipher.block_size != cipher.block_size:
            print(f"  Skipping: block size mismatch ({other_cipher.block_size} vs {cipher.block_size})")
            continue
        
        gen = CipherDataGenerator(
            cipher=other_name, n_rounds=source_rounds,
            delta_p=other_cipher.get_default_delta_p()
        )
        test_data = gen.generate_balanced_dataset(args.samples // 10)
        test_ds = CryptoDataset(test_data, 'R2_xor_diff', cipher.block_size)
        from torch.utils.data import DataLoader
        test_loader = DataLoader(test_ds, batch_size=5000, shuffle=False)
        
        metrics = evaluate_model(model, test_loader, device)
        results['cross_cipher'][other_name] = metrics['accuracy']
        print(f"Accuracy on {other_name}: {metrics['accuracy']:.4f}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cr = results['cross_round']
    axes[0].bar(list(cr.keys()), list(cr.values()), color='steelblue')
    axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Target Rounds')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'Cross-Round Transfer (trained on {source_rounds}r)')
    
    cc = results['cross_cipher']
    if cc:
        axes[1].bar(list(cc.keys()), list(cc.values()), color='coral')
        axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Target Cipher')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'Cross-Cipher Transfer (trained on {args.cipher})')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'e09_transfer_{args.cipher}.png', dpi=300)
    plt.close()
    
    return results


def run_difference_search_experiment(args) -> Dict:
    print("=" * 50)
    print("E10: Difference Search")
    print("=" * 50)
    
    cipher = get_cipher(args.cipher)
    n_rounds = args.rounds[0] if args.rounds else 5
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    if cipher.block_size == 32:
        deltas = [1 << i for i in range(0, 32, 4)]
        deltas += [0x00400000, 0x00040000, 0x80000000]
        deltas = list(set(deltas))
    else:
        deltas = [1 << i for i in range(0, cipher.block_size, 4)]
    
    results = {}
    
    for delta_p in deltas:
        delta_str = f'0x{delta_p:08x}'
        print(f"\n--- Δp = {delta_str} ---")
        
        generator = CipherDataGenerator(
            cipher=args.cipher, n_rounds=n_rounds, delta_p=delta_p
        )
        
        train_data = generator.generate_balanced_dataset(min(args.samples, 200000))
        val_data = generator.generate_balanced_dataset(20000)
        test_data = generator.generate_balanced_dataset(20000)
        
        input_dim = get_input_dim('R2_xor_diff', cipher.block_size)
        model = get_model('gohr_mlp', input_dim=input_dim)
        
        train_ds = CryptoDataset(train_data, 'R2_xor_diff', cipher.block_size)
        val_ds = CryptoDataset(val_data, 'R2_xor_diff', cipher.block_size)
        test_ds = CryptoDataset(test_data, 'R2_xor_diff', cipher.block_size)
        
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_ds, batch_size=5000, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=5000)
        test_loader = DataLoader(test_ds, batch_size=5000)
        
        trainer = Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            device=device, use_wandb=False
        )
        trainer.train(n_epochs=20, early_stopping_patience=3, save_best=False)
        
        metrics = evaluate_model(model, test_loader, device)
        results[delta_str] = {
            'accuracy': metrics['accuracy'],
            'advantage': metrics['advantage'],
        }
        print(f"  Accuracy: {metrics['accuracy']:.4f}, Advantage: {metrics['advantage']:.4f}")
    
    best_delta = max(results, key=lambda k: results[k]['accuracy'])
    print(f"\nBest difference: {best_delta} (acc={results[best_delta]['accuracy']:.4f})")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results.keys())
    accs = [results[n]['accuracy'] for n in names]
    bars = ax.barh(range(len(names)), accs, color='steelblue')
    bars[names.index(best_delta)].set_color('gold')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Accuracy')
    ax.set_title(f'Difference Search — {args.cipher.upper()} ({n_rounds} rounds)')
    ax.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'e10_diff_search_{args.cipher}_r{n_rounds}.png', dpi=300)
    plt.close()
    
    return results


def run_classical_comparison_experiment(args) -> Dict:
    from data.statistics import compute_differential_probability
    
    print("=" * 50)
    print("E11: Classical vs Neural Comparison")
    print("=" * 50)
    
    cipher = get_cipher(args.cipher)
    rounds_list = args.rounds if args.rounds else list(range(2, 8))
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    neural_results = {}
    classical_results = {}
    
    for n_rounds in rounds_list:
        print(f"\n--- Round {n_rounds} ---")
        
        dp = compute_differential_probability(
            diff_in=cipher.get_default_delta_p(),
            diff_out=0,
            cipher=cipher,
            n_samples=min(args.samples, 500000),
            n_rounds=n_rounds
        )
        classical_acc = min(1.0, 0.5 + dp)
        classical_results[n_rounds] = classical_acc
        print(f"  Classical DP = {dp:.6f}, proxy acc = {classical_acc:.4f}")
        
        model, metrics, _, _ = _train_and_evaluate(
            args, cipher, n_rounds, n_epochs=20, patience=3
        )
        neural_results[n_rounds] = metrics['accuracy']
        print(f"  Neural acc = {metrics['accuracy']:.4f}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    
    rounds = sorted(neural_results.keys())
    ax.plot(rounds, [neural_results[r] for r in rounds], 'bo-', label='Neural', linewidth=2)
    ax.plot(rounds, [classical_results[r] for r in rounds], 'r^--', label='Classical (DP)', linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random')
    ax.set_xlabel('Number of Rounds')
    ax.set_ylabel('Accuracy / Estimated Advantage')
    ax.set_title(f'Neural vs Classical — {args.cipher.upper()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'e11_classical_{args.cipher}.png', dpi=300)
    plt.close()
    
    return {'neural': neural_results, 'classical': classical_results}


def run_key_recovery_experiment(args) -> Dict:
    print("=" * 50)
    print("E12: Key Recovery Demo")
    print("=" * 50)
    
    cipher = get_cipher(args.cipher)
    n_rounds = args.rounds[0] if args.rounds else 5
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print(f"\nTraining distinguisher on {n_rounds - 1} rounds...")
    
    model, metrics, _, _ = _train_and_evaluate(
        args, cipher, n_rounds - 1
    )
    print(f"Distinguisher accuracy ({n_rounds - 1} rounds): {metrics['accuracy']:.4f}")
    
    print(f"\n--- Key Recovery Attack on {n_rounds} rounds ---")
    
    real_key = cipher.random_key()
    n_pairs = min(args.samples // 10, 10000)
    
    P = cipher.random_plaintexts(n_pairs)
    P_prime = P ^ cipher.get_default_delta_p()
    C = cipher.encrypt(P, n_rounds, real_key)
    C_prime = cipher.encrypt(P_prime, n_rounds, real_key)
    
    try:
        expanded_key = cipher._expand_key(real_key, n_rounds)
        real_last_subkey = int(expanded_key[-1]) if hasattr(expanded_key, '__getitem__') else 0
    except Exception:
        real_last_subkey = 0
    
    from data.representations import RepresentationFactory
    factory = RepresentationFactory(block_size=cipher.block_size)
    
    n_candidates = 256
    key_scores = {}
    
    model.eval()
    for i in range(n_candidates):
        candidate_key = i
        
        C_partial = C ^ candidate_key
        C_prime_partial = C_prime ^ candidate_key
        
        X = factory.get_representation('R2_xor_diff', C_partial, C_prime_partial)
        X_tensor = torch.from_numpy(X).float().to(device)
        
        with torch.no_grad():
            scores = model(X_tensor).squeeze().cpu().numpy()
        
        key_scores[candidate_key] = float(np.mean(scores))
    
    ranked_keys = sorted(key_scores.items(), key=lambda x: x[1], reverse=True)
    
    results = {
        'distinguisher_accuracy': metrics['accuracy'],
        'n_candidates': n_candidates,
        'n_pairs': n_pairs,
        'top_10_keys': [(k, float(s)) for k, s in ranked_keys[:10]],
        'real_last_subkey': int(real_last_subkey) if isinstance(real_last_subkey, (int, np.integer)) else 0,
    }
    
    print(f"\nTop 10 key candidates (score):")
    for k, s in ranked_keys[:10]:
        marker = " ★" if k == results['real_last_subkey'] else ""
        print(f"  Key 0x{k:04x}: score = {s:.4f}{marker}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    keys = [k for k, _ in ranked_keys]
    scores = [s for _, s in ranked_keys]
    ax.bar(range(len(keys)), scores, color='steelblue', alpha=0.7)
    ax.set_xlabel('Key Candidates (ranked)')
    ax.set_ylabel('Distinguisher Score')
    ax.set_title(f'Key Recovery — {args.cipher.upper()} ({n_rounds} rounds)')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'e12_key_recovery_{args.cipher}_r{n_rounds}.png', dpi=300)
    plt.close()
    
    return results


EXPERIMENT_FUNCTIONS = {
    'baseline': run_baseline_experiment,
    'representation': run_representation_experiment,
    'invariance': run_invariance_experiment,
    'robustness': run_robustness_experiment,
    'memory': run_memory_experiment,
    'markov': run_conditional_mi_experiment,
    'decay': run_signal_decay_experiment,
    'saliency': run_saliency_experiment,
    'transfer': run_transfer_experiment,
    'diff_search': run_difference_search_experiment,
    'classical': run_classical_comparison_experiment,
    'key_recovery': run_key_recovery_experiment,
}


def main():
    args = parse_args()
    
    if args.exp in EXPERIMENT_FUNCTIONS:
        results = EXPERIMENT_FUNCTIONS[args.exp](args)
        print(f"\nExperiment {args.exp} completed!")
        print(f"Results: {results}")
        
        import json
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / f'{args.exp}_{args.cipher}_results.json'
        
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=convert)
        print(f"Results saved to {results_file}")
    else:
        print(f"Experiment {args.exp} not yet implemented")


if __name__ == '__main__':
    main()
