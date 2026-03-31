"""
Evaluation metrics for neural cryptanalysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve


def compute_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cuda'
) -> float:
    """
    Compute classification accuracy.
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Compute device
        
    Returns:
        Accuracy in [0, 1]
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X).squeeze()
            predictions = (outputs > 0.5).float()
            correct += (predictions == y).sum().item()
            total += y.size(0)
    
    return correct / total


def compute_advantage(accuracy: float) -> float:
    """
    Compute cryptographic advantage from accuracy.
    
    Advantage = 2 * |Accuracy - 0.5|
    
    A random guesser has accuracy 0.5 and advantage 0.
    A perfect distinguisher has accuracy 1.0 and advantage 1.0.
    
    Args:
        accuracy: Classification accuracy
        
    Returns:
        Advantage in [0, 1]
    """
    return 2 * abs(accuracy - 0.5)


def compute_max_distinguishable_round(
    accuracies: Dict[int, float],
    threshold: float = 0.51
) -> int:
    """
    Find maximum round where distinguisher is effective.
    
    Args:
        accuracies: Dict mapping round count to accuracy
        threshold: Minimum accuracy to be considered effective
        
    Returns:
        Maximum effective round count
    """
    max_round = 0
    for n_rounds, acc in sorted(accuracies.items()):
        if acc > threshold:
            max_round = n_rounds
    return max_round


def compute_data_efficiency(
    model_class,
    data: Dict[str, np.ndarray],
    target_accuracy: float = 0.6,
    sample_sizes: List[int] = None,
    device: str = 'cuda',
    **model_kwargs
) -> Tuple[int, Dict[int, float]]:
    """
    Find minimum samples needed to reach target accuracy.
    
    Args:
        model_class: Model class to instantiate
        data: Full dataset
        target_accuracy: Target accuracy to achieve
        sample_sizes: Sample sizes to test
        device: Compute device
        **model_kwargs: Model arguments
        
    Returns:
        (min_samples, {sample_size: accuracy})
    """
    if sample_sizes is None:
        sample_sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    
    from data.dataloader import CryptoDataset
    from training.trainer import Trainer
    
    results = {}
    min_samples = sample_sizes[-1]
    
    for n_samples in sample_sizes:
        # Subsample data
        indices = np.random.choice(len(data['labels']), min(n_samples, len(data['labels'])), replace=False)
        subset = {k: v[indices] for k, v in data.items()}
        
        # Train model
        model = model_class(**model_kwargs).to(device)
        dataset = CryptoDataset(subset)
        loader = DataLoader(dataset, batch_size=min(1000, n_samples), shuffle=True)
        
        # Quick training
        trainer = Trainer(model, loader, loader, device=device, use_wandb=False)
        trainer.train(n_epochs=10, early_stopping_patience=3, save_best=False)
        
        # Evaluate
        acc = compute_accuracy(model, loader, device)
        results[n_samples] = acc
        
        if acc >= target_accuracy and n_samples < min_samples:
            min_samples = n_samples
    
    return min_samples, results


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Compute device
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    all_outputs = []
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X).squeeze()
            predictions = (outputs > 0.5).float()
            
            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    # Accuracy
    accuracy = np.mean(all_predictions == all_labels)
    
    # Advantage
    advantage = compute_advantage(accuracy)
    
    # AUC-ROC
    try:
        auc = roc_auc_score(all_labels, all_outputs)
    except:
        auc = 0.5
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    
    # Per-class accuracy
    cipher_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
    random_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'advantage': advantage,
        'auc_roc': auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cipher_accuracy': cipher_acc,
        'random_accuracy': random_acc,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }


def get_prediction_distribution(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get prediction distributions for cipher and random samples.
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Compute device
        
    Returns:
        (all_outputs, cipher_outputs, random_outputs)
    """
    model.eval()
    
    cipher_outputs = []
    random_outputs = []
    
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X).squeeze().cpu().numpy()
            labels = y.cpu().numpy()
            
            cipher_outputs.extend(outputs[labels == 1])
            random_outputs.extend(outputs[labels == 0])
    
    return (
        np.concatenate([cipher_outputs, random_outputs]),
        np.array(cipher_outputs),
        np.array(random_outputs)
    )


def compute_roc_curve(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve data.
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Compute device
        
    Returns:
        (fpr, tpr, thresholds)
    """
    model.eval()
    
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X).squeeze().cpu().numpy()
            all_outputs.extend(outputs)
            all_labels.extend(y.cpu().numpy())
    
    fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
    return fpr, tpr, thresholds


def estimate_mutual_information(
    X: np.ndarray,
    Y: np.ndarray,
    hidden_dims: List[int] = [256, 128, 64],
    n_epochs: int = 100,
    batch_size: int = 5000,
    device: str = 'cuda',
    verbose: bool = False
) -> float:
    """
    Estimate mutual information I(X; Y) using MINE.
    
    Wrapper around MutualInfoEstimator for convenient use in evaluation.
    
    Args:
        X: Input representations (n_samples, input_dim)
        Y: Labels (n_samples,)
        hidden_dims: MINE hidden layer dimensions
        n_epochs: Training epochs for MINE
        batch_size: Batch size
        device: Compute device
        verbose: Print training progress
        
    Returns:
        Estimated mutual information in nats
    """
    from models.mine import MutualInfoEstimator
    
    input_dim = X.shape[1] if X.ndim > 1 else 1
    estimator = MutualInfoEstimator(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        device=device
    )
    
    return estimator.estimate(X, Y, n_epochs=n_epochs,
                              batch_size=batch_size, verbose=verbose)


def estimate_conditional_mi(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    hidden_dims: List[int] = [256, 128, 64],
    n_epochs: int = 100,
    batch_size: int = 5000,
    device: str = 'cuda'
) -> float:
    """
    Estimate conditional mutual information I(X; Y | Z).
    
    Uses chain rule: I(X; Y | Z) = I(X, Z; Y) - I(Z; Y)
    
    Args:
        X: Variable 1
        Y: Labels
        Z: Conditioning variable
        hidden_dims: MINE hidden dimensions
        n_epochs: Training epochs
        batch_size: Batch size
        device: Compute device
        
    Returns:
        Estimated conditional MI in nats
    """
    from models.mine import MutualInfoEstimator
    
    input_dim_xz = (X.shape[1] if X.ndim > 1 else 1) + (Z.shape[1] if Z.ndim > 1 else 1)
    input_dim_z = Z.shape[1] if Z.ndim > 1 else 1
    
    estimator = MutualInfoEstimator(
        input_dim=input_dim_xz,
        hidden_dims=hidden_dims,
        device=device
    )
    
    return estimator.estimate_conditional(X, Y, Z, n_epochs=n_epochs,
                                          batch_size=batch_size)


def compute_differential_probability_from_model(
    model: nn.Module,
    cipher,
    n_rounds: int,
    delta_p: int,
    n_samples: int = 100000,
    representation: str = 'R2_xor_diff',
    device: str = 'cuda'
) -> float:
    """
    Use a trained distinguisher to estimate differential probability.
    
    The model's average confidence on cipher-generated pairs approximates
    how "non-random" the output distribution is under the given differential.
    
    Args:
        model: Trained distinguisher model
        cipher: Cipher instance
        n_rounds: Number of rounds
        delta_p: Input difference
        n_samples: Number of samples
        representation: Representation name
        device: Compute device
        
    Returns:
        Estimated distinguishing probability (model confidence on cipher pairs)
    """
    from data.representations import RepresentationFactory
    
    model.eval()
    key = cipher.random_key()
    
    # Generate cipher pairs
    P = cipher.random_plaintexts(n_samples)
    P_prime = P ^ delta_p
    C = cipher.encrypt(P, n_rounds, key)
    C_prime = cipher.encrypt(P_prime, n_rounds, key)
    
    # Convert to representation
    factory = RepresentationFactory(block_size=cipher.block_size)
    X = factory.get_representation(representation, C, C_prime)
    X_tensor = torch.from_numpy(X).float().to(device)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(X_tensor).squeeze().cpu().numpy()
    
    return float(np.mean(outputs))


def bootstrap_confidence_interval(
    metric_fn,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Args:
        metric_fn: Function(predictions, labels) -> scalar metric
        predictions: Model predictions
        labels: Ground truth labels
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level (default: 0.05 for 95% CI)
        seed: Random seed
        
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n_samples = len(predictions)
    
    point_estimate = metric_fn(predictions, labels)
    
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_preds = predictions[indices]
        boot_labels = labels[indices]
        try:
            estimate = metric_fn(boot_preds, boot_labels)
            bootstrap_estimates.append(estimate)
        except Exception:
            continue
    
    bootstrap_estimates = np.array(bootstrap_estimates)
    ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
    
    return point_estimate, ci_lower, ci_upper


def statistical_significance_test(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Permutation test for comparing two model accuracies.
    
    Tests H0: the two models have equal accuracy.
    
    Args:
        preds_a: Binary predictions from model A
        preds_b: Binary predictions from model B
        labels: Ground truth labels
        n_permutations: Number of permutations
        seed: Random seed
        
    Returns:
        Dict with 'accuracy_a', 'accuracy_b', 'diff', 'p_value'
    """
    rng = np.random.RandomState(seed)
    
    acc_a = np.mean(preds_a == labels)
    acc_b = np.mean(preds_b == labels)
    observed_diff = abs(acc_a - acc_b)
    
    # Permutation test
    correct_a = (preds_a == labels).astype(np.int32)
    correct_b = (preds_b == labels).astype(np.int32)
    pooled = np.stack([correct_a, correct_b], axis=0)
    
    count_extreme = 0
    for _ in range(n_permutations):
        # For each sample, randomly swap which model gets credit
        swap = rng.randint(0, 2, size=len(labels))
        perm_a = pooled[swap, np.arange(len(labels))]
        perm_b = pooled[1 - swap, np.arange(len(labels))]
        perm_diff = abs(perm_a.mean() - perm_b.mean())
        if perm_diff >= observed_diff:
            count_extreme += 1
    
    p_value = count_extreme / n_permutations
    
    return {
        'accuracy_a': float(acc_a),
        'accuracy_b': float(acc_b),
        'diff': float(observed_diff),
        'p_value': float(p_value)
    }
