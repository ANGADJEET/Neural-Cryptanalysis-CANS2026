"""
Configuration utilities for neural cryptanalysis.

Handles loading, validation, and path management for project configuration.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


# Valid values for configuration validation
VALID_CIPHERS = {'speck32', 'simon32', 'present'}
VALID_MODELS = {'mlp', 'gohr_mlp', 'cnn', 'residual_cnn', 'siamese', 'lstm', 'gru', 'mine'}
VALID_REPRESENTATIONS = {
    'R1_raw_pair', 'R2_xor_diff', 'R3_concat', 'R4_bit_sliced',
    'R5_word_level', 'R6_joint_pc', 'R7_sequential', 'R8_statistical',
    'R9_masked'
}
VALID_OPTIMIZERS = {'adam', 'adamw', 'sgd'}
VALID_SCHEDULERS = {'cosine', 'step', 'plateau', 'none'}


def load_config(path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file and create required directories.
    
    Args:
        path: Path to config.yaml
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create all directories from paths section
    if 'paths' in config:
        for key, dir_path in config['paths'].items():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    return config


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration values against known valid options.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Validate cipher names
    if 'ciphers' in config:
        for cipher_name in config['ciphers']:
            if cipher_name not in VALID_CIPHERS:
                errors.append(f"Unknown cipher: '{cipher_name}'. Valid: {VALID_CIPHERS}")
    
    # Validate model configurations
    if 'models' in config:
        for model_name in config['models']:
            # Handle lstm/rnn naming: accept both
            normalized = model_name.lower()
            if normalized == 'rnn':
                normalized = 'lstm'  # rnn section covers lstm/gru
            if normalized not in VALID_MODELS and normalized != 'rnn':
                errors.append(f"Unknown model: '{model_name}'. Valid: {VALID_MODELS}")
    
    # Validate training settings
    if 'training' in config:
        training = config['training']
        
        if 'optimizer' in training:
            opt = training['optimizer'].lower()
            if opt not in VALID_OPTIMIZERS:
                errors.append(f"Unknown optimizer: '{opt}'. Valid: {VALID_OPTIMIZERS}")
        
        if 'scheduler' in training:
            sched = training['scheduler'].lower()
            if sched not in VALID_SCHEDULERS:
                errors.append(f"Unknown scheduler: '{sched}'. Valid: {VALID_SCHEDULERS}")
        
        if 'learning_rate' in training:
            lr = training['learning_rate']
            if not (0 < lr < 1):
                errors.append(f"Suspicious learning rate: {lr} (expected 0 < lr < 1)")
        
        if 'epochs' in training:
            epochs = training['epochs']
            if epochs < 1:
                errors.append(f"Invalid epochs: {epochs} (must be >= 1)")
    
    # Validate data settings
    if 'data' in config:
        data = config['data']
        if 'batch_size' in data and data['batch_size'] < 1:
            errors.append(f"Invalid batch_size: {data['batch_size']}")
    
    # Validate cipher parameters
    if 'ciphers' in config:
        for cipher_name, cipher_cfg in config['ciphers'].items():
            if 'block_size' in cipher_cfg and cipher_cfg['block_size'] not in [32, 64, 128]:
                errors.append(f"{cipher_name}: unusual block_size {cipher_cfg['block_size']}")
            if 'target_rounds' in cipher_cfg:
                for r in cipher_cfg['target_rounds']:
                    max_r = cipher_cfg.get('max_rounds', 100)
                    if r > max_r:
                        errors.append(f"{cipher_name}: target_round {r} exceeds max_rounds {max_r}")
    
    return errors


def get_cipher_config(config: Dict[str, Any], cipher_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific cipher.
    
    Args:
        config: Full configuration dictionary
        cipher_name: Name of the cipher
        
    Returns:
        Cipher-specific configuration
    """
    if 'ciphers' not in config or cipher_name not in config['ciphers']:
        raise ValueError(f"No configuration found for cipher '{cipher_name}'")
    return config['ciphers'][cipher_name]


def get_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model.
    
    Handles lstm/rnn naming inconsistency by checking both keys.
    
    Args:
        config: Full configuration dictionary
        model_name: Name of the model
        
    Returns:
        Model-specific configuration
    """
    if 'models' not in config:
        return {}
    
    models = config['models']
    
    # Handle lstm/rnn naming: check the key directly first, then aliases
    if model_name in models:
        return models[model_name]
    
    # Map rnn-family names to the config section
    rnn_aliases = {'lstm': ['rnn', 'lstm'], 'gru': ['rnn', 'gru']}
    if model_name in rnn_aliases:
        for alias in rnn_aliases[model_name]:
            if alias in models:
                return models[alias]
    
    return {}
