# When Neural Distinguishers Anti-Transfer: Markov Structure and Feature Composition in Lightweight Ciphers

This repository contains the official implementation, datasets, and experimental pipelines for the paper *"When Neural Distinguishers Anti-Transfer: Markov Structure and Feature Composition in Lightweight Ciphers"* (Submitted).

## 1. Overview & Theoretical Framework

This repository provides a comprehensive framework to train neural differential distinguishers and evaluate their cross-round feature compositionality. It systematically demonstrates and analyzes the phenomenon of **neural feature anti-transfer**.

### The Markov-Transfer Framework
The codebase tests three theoretical sufficient conditions for positive neural transfer:
1. **(C1) Markov differential propagation:** The cipher itself must be memoryless across round boundaries.
2. **(C2) DDT-only features:** The neural distinguisher must rely exclusively on features derived from the Difference Distribution Table (DDT).
3. **(C3) Monotone bias sign preservation:** The stochastic transition operator must not flip the signs of differential biases.

### Key Discoveries
- **SPN Ciphers (PRESENT)** satisfy all three conditions, leading to monotonic **positive transfer** across rounds.
- **ARX Ciphers (SPECK32/64)** violate (C2) because they exploit non-DDT carry-propagation features that are heavily round-dependent, resulting in **anti-transfer**.
- **Feistel Ciphers (SIMON32/64)** violate (C3) because the AND-gate induces data-dependent differential propagation, creating a multiplicative bias coupling that causes bit-position-dependent bias sign oscillation, leading to **anti-transfer**.

**Anti-transfer** means that models trained at higher rounds produce statistically significant *below-chance* accuracy on lower-round data. Our Mutual Information Neural Estimation (MINE) reveals that these models still extract high mutual information from the ciphertext, but they map the learned differential structures to the wrong class.

---

## 2. Project Structure

```text
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
├── tests/            # Test suite for data, models, and integration
└── config.yaml       # Global experiment configuration
```

---

## 3. Installation & Quick Start

### 3.1. Installation
We recommend using a virtual environment:
```bash
python -m venv crypto-env
source crypto-env/bin/activate  # On Windows use: crypto-env\Scripts\activate
pip install -r requirements.txt
```

### 3.2. Configuration
All major hyperparameters, data limits, and model configurations can be adjusted globally in `config.yaml`. The framework automatically manages directory creation and dependency injection based on this file.

### 3.3. Generating Data
The `scripts/generate_data.py` CLI allows for flexible dataset creation.
```bash
# Generate 1,000,000 samples for 5-round SPECK32/64
python scripts/generate_data.py --cipher speck32 --rounds 5 --samples 1000000

# Generate 7-round SIMON32/64 data
python scripts/generate_data.py --cipher simon32 --rounds 7 --samples 1000000
```

### 3.4. Training a Model
You can train various models using the `scripts/train_model.py` script. The framework supports multiple input representations (e.g., `R1_raw_pair`, `R2_xor_diff`, `R4_bit_sliced`).
```bash
# Train an MLP distinguisher on SPECK32/64 using XOR differences
python scripts/train_model.py --model mlp --repr R2_xor_diff --cipher speck32 --rounds 5
```

---

## 4. Reproducing Paper Experiments

The core findings of the paper can be reproduced using the specific experiment scripts located in the `experiments/` directory. Each experiment addresses a specific analytical goal:

| Experiment | Filename | Paper Section / Description |
|------------|----------|-----------------------------|
| **E01** | `exp01_baseline.py` | Generates the baseline accuracy vs. rounds degradation curves for all ciphers. |
| **E02** | `exp02_representation.py` | Analyzes how input representations (R1-R9) impact distinguisher performance. |
| **E03** | `exp03_invariance.py` | Tests rotational and shift invariance of learned features. |
| **E04** | `exp04_robustness.py` | Evaluates model robustness against input noise and partial masking. |
| **E05** | `exp05_memory.py` | Tests Markov assumption (C1) via VAE to see if models require multi-round history. |
| **E06** | `exp06_conditional_mi.py` | Uses **MINE** to estimate Mutual Information, revealing the anti-transfer paradox. |
| **E07** | `exp07_signal_decay.py` | Generates 2D heatmaps of accuracy over rounds vs. $\Delta P$. |
| **E08** | `exp08_saliency.py` | Computes Input Gradients / Saliency maps to map active differential bits. |
| **E09** | `exp09_transfer.py` | **Core result:** Cross-round transfer tests (demonstrating SPECK/SIMON anti-transfer). |
| **E10** | `exp10_diff_search.py` | Heuristic search for optimal input differences. |
| **E11** | `exp11_classical_comp.py` | Compares neural confidence scores directly against classical differential transition probabilities. |
| **E12** | `exp12_key_recovery.py` | Demonstrates Bayesian key recovery success on SPECK and failure on SIMON due to anti-transfer. |

### Running an Experiment
To run an experiment, invoke it directly via python. Results, including generated plots and JSON statistics, are saved automatically to the `results/` directory.
```bash
# Run the cross-round transfer experiment (E09) for SPECK32
python experiments/exp09_transfer.py --cipher speck32

# Run Key Recovery (E12)
python experiments/exp12_key_recovery.py --cipher simon32
```

## 5. Testing
The repository includes a comprehensive `pytest` suite testing data generation, neural models, and end-to-end integration.
```bash
python -m pytest tests/ -v
```

## 6. License
This repository is released under the MIT License.
