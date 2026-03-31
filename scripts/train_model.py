
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from data.generator import load_dataset
from data.dataloader import get_dataloaders, get_input_dim
from models import get_model
from training.trainer import Trainer
from evaluation.metrics import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a neural cryptanalysis model'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        help='Path to dataset directory (with train.csv, val.csv, test.csv)'
    )
    
    parser.add_argument(
        '--cipher',
        type=str,
        choices=['speck32', 'simon32', 'present'],
        help='Cipher (used to find default data path)'
    )
    
    parser.add_argument(
        '--rounds',
        type=int,
        help='Number of rounds (used to find default data path)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['mlp', 'gohr_mlp', 'cnn', 'residual_cnn', 'siamese', 'lstm', 'gru'],
        help='Model architecture'
    )
    
    parser.add_argument(
        '--repr',
        type=str,
        default='R2_xor_diff',
        help='Input representation'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=5000,
        help='Batch size'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Early stopping patience'
    )
    
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable wandb logging'
    )
    
    parser.add_argument(
        '--project',
        type=str,
        default='neural-cryptanalysis',
        help='Wandb project name'
    )
    
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Wandb run name'
    )
    
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./checkpoints',
        help='Directory to save checkpoints'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Compute device'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if args.data:
        data_path = args.data
    elif args.cipher and args.rounds:
        defaults = {'speck32': 0x00400000, 'simon32': 0x00000001, 'present': 0x00000001}
        delta_p = defaults.get(args.cipher, 0x00000001)
        data_path = f"./data/generated/{args.cipher}_r{args.rounds}_delta{delta_p:08x}"
    else:
        raise ValueError("Must specify --data or both --cipher and --rounds")
    
    print(f"Loading data from: {data_path}")
    data = load_dataset(data_path)
    
    block_sizes = {'speck32': 32, 'simon32': 32, 'present': 64}
    if args.cipher:
        block_size = block_sizes[args.cipher]
    else:
        block_size = 32
    
    print(f"Using representation: {args.repr}")
    loaders = get_dataloaders(
        data,
        representation=args.repr,
        block_size=block_size,
        batch_size=args.batch_size
    )
    
    input_dim = get_input_dim(args.repr, block_size)
    print(f"Input dimension: {input_dim}")
    
    print(f"Creating model: {args.model}")
    model = get_model(args.model, input_dim=input_dim)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    config = {
        'cipher': args.cipher,
        'rounds': args.rounds,
        'model': args.model,
        'representation': args.repr,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs': args.epochs
    }
    
    run_name = args.run_name or f"{args.model}_{args.repr}_{args.cipher}_r{args.rounds}"
    
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        use_wandb=args.wandb,
        project_name=args.project,
        run_name=run_name,
        save_dir=args.save_dir
    )
    
    print("\nStarting training...")
    history = trainer.train(
        n_epochs=args.epochs,
        early_stopping_patience=args.patience
    )
    
    print("\nFinal evaluation on test set...")
    test_metrics = evaluate_model(model, loaders['test'], device)
    
    print("\nTest Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Advantage: {test_metrics['advantage']:.4f}")
    print(f"  AUC-ROC:   {test_metrics['auc_roc']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1_score']:.4f}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
