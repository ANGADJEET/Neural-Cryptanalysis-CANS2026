# Neural Differential Cryptanalysis

A systematic study of representation-induced and cross-round cryptographic signal in neural differential cryptanalysis.

## Research Questions

1. **RQ1 (Representation)**: Where does cryptographic signal reside in observable data?
2. **RQ2 (Markov)**: Does distinguishing signal obey the classical memoryless assumption?

## Project Structure

```
neural_cryptanalysis/
├── ciphers/          # SPECK32, SIMON32, PRESENT implementations
├── data/             # Dataset generation, representations (R1–R9), statistics
├── models/           # MLP, CNN, ResNet, LSTM, GRU, Siamese, MINE
├── training/         # Training pipeline with WandB integration
├── evaluation/       # Metrics, MI estimation, bootstrap CI, significance testing
├── experiments/      # 12 self-contained experiment scripts (E01–E12)
├── visualization/    # Publication-quality plotting functions
├── scripts/          # CLI entry points (generate_data, train_model, run_experiment)
├── utils/            # Structured logging, helpers
├── tests/            # 46 unit + integration tests (pytest)
└── config.yaml       # Global configuration
```

## Quick Start

```bash
# 1. Create virtual environment
python -m venv crypto-env
crypto-env\Scripts\activate            # Windows
source crypto-env/bin/activate         # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run tests (verify installation)
python -m pytest tests/ -v

# 4. Generate dataset
python scripts/generate_data.py --cipher speck32 --rounds 5 --samples 1000000

# 5. Train a model
python scripts/train_model.py --model mlp --repr R2_xor_diff --cipher speck32 --rounds 5

# 6. Run an experiment
python experiments/exp01_baseline.py --cipher speck32
```

## Ciphers Supported

| Cipher | Block Size | Key Size | Max Rounds | Diffusion |
|--------|-----------|----------|------------|-----------|
| **SPECK32/64** | 32-bit | 64-bit | 22 | Fast (ARX) |
| **SIMON32/64** | 32-bit | 64-bit | 32 | Slow (Feistel) |
| **PRESENT** | 64-bit | 80-bit | 31 | Medium (SPN) |

## Input Representations

| ID | Name | Description | Dimension |
|----|------|-------------|-----------|
| R1 | `R1_raw_pair` | Raw ciphertext pair (C, C') | 2 × block_size |
| R2 | `R2_xor_diff` | XOR difference ΔC = C ⊕ C' | block_size |
| R3 | `R3_concat` | Concatenated (C \|\| C' \|\| ΔC) | 3 × block_size |
| R4 | `R4_bit_sliced` | Per-bit packing | 2 × block_size |
| R5 | `R5_word_level` | Word-level split | Variable |
| R6 | `R6_joint_pc` | Joint (P, C, P', C') | Requires plaintext |
| R7 | `R7_frequency` | Bit frequency features | block_size + extras |
| R8 | `R8_statistical` | Statistical features (HW, correlation) | Variable |
| R9 | `R9_minimal` | Hamming weight of ΔC (scalar) | 1 |

## Model Architectures

| Model | CLI Name | Description |
|-------|----------|-------------|
| Gohr MLP | `gohr_mlp` | Gohr (2019) architecture with batch norm |
| Standard MLP | `mlp` | Configurable hidden layers |
| 1D-CNN | `cnn` | Convolutional feature extraction |
| ResNet | `resnet` | Residual blocks for deep architectures |
| LSTM | `lstm` | Sequence model for round traces |
| GRU | `gru` | Lightweight recurrent model |
| Siamese | `siamese` | Twin-branch comparison network |

## Experiments

12 experiments covering baseline analysis, interpretability, and key recovery.  
See **[EXPERIMENTS.md](EXPERIMENTS.md)** for full details.

| # | Name | What it answers |
|---|------|-----------------|
| E01 | Baseline | How does accuracy degrade with rounds? |
| E02 | Representation | Which input format works best? |
| E03 | Invariance | Does the model exploit bit positions? |
| E04 | Robustness | How robust is the model to noise? |
| E05 | Memory Depth | Does the model need multi-round history? |
| E06 | Conditional MI | How much information is in ΔC about the label? |
| E07 | Signal Decay | Accuracy heatmap over (rounds × Δp) |
| E08 | Saliency | Which bits matter most? |
| E09 | Transfer | Does the model generalize across ciphers/rounds? |
| E10 | Diff Search | What's the best input difference? |
| E11 | Classical Comparison | Neural vs classical differential probability |
| E12 | Key Recovery | Bayesian key ranking demo |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_data_generation.py -v
python -m pytest tests/test_models.py -v
python -m pytest tests/test_integration.py -v
```

**Test coverage:** 46 tests across 3 modules (data generation, models, integration).

## Configuration

Edit `config.yaml` for global settings:

```yaml
cipher:
  name: speck32
  n_rounds: 5
  delta_p: 0x00400000

training:
  batch_size: 5000
  epochs: 50
  learning_rate: 0.001
```

See `config_utils.py` for validation and automatic directory creation.

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12
- NumPy, Matplotlib, Seaborn
- scikit-learn (for evaluation metrics)
- tqdm (progress bars)
- wandb (optional, for experiment tracking)

## References

- Gohr, A. (2019). *Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning*. CRYPTO 2019.
- Benamira et al. (2021). *A Deeper Look at Machine Learning-Based Cryptanalysis*. EUROCRYPT 2021.

## License

MIT
