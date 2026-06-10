# When Neural Distinguishers Anti-Transfer: Markov Structure and Feature Composition in Lightweight Ciphers

This repository contains the code, data generation scripts, and experimental pipelines for the paper *"When Neural Distinguishers Anti-Transfer: Markov Structure and Feature Composition in Lightweight Ciphers"* (Submitted).

## Overview

This repository provides a framework to train neural differential distinguishers and evaluate their cross-round feature compositionality. It systematically demonstrates and analyzes the phenomenon of **neural feature anti-transfer**:
- **SPN ciphers** like PRESENT compose monotonically across rounds (positive transfer).
- **ARX ciphers** like SPECK32/64 and **Feistel ciphers** like SIMON32/64 exhibit **anti-transfer**: neural distinguishers trained at higher rounds produce below-chance accuracy on lower-round data.
- Mutual Information Neural Estimation (MINE) reveals that anti-transferred models still extract high mutual information from the ciphertext, but map the learned differential structures to the wrong class.

We provide a full pipeline for generating ciphertext pairs, training various neural architectures (MLPs, ResNets, CNNs), estimating Mutual Information, and reproducing the cross-round and cross-cipher transfer experiments discussed in the paper.

## Project Structure

```
neural_cryptanalysis/
├── ciphers/          # Lightweight cipher implementations (SPECK, SIMON, PRESENT)
├── data/             # Dataset generation and formatting pipelines
├── models/           # Neural architectures (MLPs, CNNs, ResNets) and MINE
├── training/         # Standardized PyTorch training pipelines
├── evaluation/       # Validation metrics and statistical testing
├── experiments/      # Reproducible experiment scripts (E01–E12)
├── visualization/    # Scripts to generate paper figures
├── scripts/          # CLI entry points for generation and training
├── utils/            # Logging and general utilities
├── tests/            # Test suite
└── config.yaml       # Global experiment configuration
```

## Quick Start

### 1. Installation
```bash
python -m venv crypto-env
source crypto-env/bin/activate  # Or crypto-env\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Generate Data
```bash
# Generate 5-round SPECK32/64 data
python scripts/generate_data.py --cipher speck32 --rounds 5 --samples 1000000
```

### 3. Train a Model
```bash
# Train an MLP distinguisher on the generated data
python scripts/train_model.py --model mlp --repr R2_xor_diff --cipher speck32 --rounds 5
```

## Key Experiments from the Paper

The core findings of the paper can be reproduced using the specific experiment scripts located in the `experiments/` directory:

- **`exp09_transfer.py`**: Executes the cross-round and cross-cipher transfer polarity tests, demonstrating the anti-transfer phenomenon on SPECK and SIMON.
- **`exp06_conditional_mi.py`**: Trains the Mutual Information Neural Estimator (MINE) to show that anti-transferred models retain label information despite below-chance accuracy.
- **`exp12_key_recovery.py`**: Demonstrates the impact of anti-transfer on practical cryptography by showing Bayesian key recovery success on SPECK but total failure on SIMON.

## Configuration
Hyperparameters, data limits, and model configurations can be adjusted globally in `config.yaml`.

## License
MIT License
