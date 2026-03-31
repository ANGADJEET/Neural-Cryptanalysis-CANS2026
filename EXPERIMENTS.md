# Experiment Catalog

All 12 experiments for the neural differential cryptanalysis study.  
Each script is self-contained with CLI args, training, evaluation, and plotting.

## Running Experiments

```bash
# Activate virtual environment
crypto-env\Scripts\activate        # Windows
source crypto-env/bin/activate     # Linux/macOS

# Run any experiment
python experiments/exp01_baseline.py --cipher speck32
python experiments/exp01_baseline.py --help   # see all options
```

Common flags across all scripts:
| Flag | Description | Default |
|------|-------------|---------|
| `--cipher` | `speck32`, `simon32`, `present` | `speck32` |
| `--rounds` | Round count(s) | Cipher-dependent |
| `--samples` | Training samples | `500000` |
| `--epochs` | Training epochs | `20-30` |
| `--device` | `cuda` or `cpu` | `cuda` |
| `--output-dir` | Where to save results | `./results/e{NN}_*` |

---

## E01: Baseline Distinguisher
**Script:** `experiments/exp01_baseline.py`  
**Goal:** Train a neural distinguisher and measure accuracy vs round count.  
**Method:** For each round count, train a Gohr-style MLP on balanced (cipher, random) pairs using the R2 XOR-diff representation.  
**Output:** Accuracy-vs-rounds curve, JSON results.  
**Key insight:** Accuracy should degrade as rounds increase, with a characteristic "cliff" where the cipher becomes indistinguishable.

```bash
python experiments/exp01_baseline.py --cipher speck32 --rounds 3 4 5 6 7 8 9
```

---

## E02: Representation Analysis
**Script:** `experiments/exp02_representation.py`  
**Goal:** Compare all input representations (R1–R8) on the same task.  
**Method:** Train identical MLP on each representation; fixed cipher and round count.  
**Output:** Bar chart of accuracy per representation.  
**Key insight:** XOR-diff (R2) usually outperforms raw pairs; statistical (R8) can be competitive at higher rounds.

```bash
python experiments/exp02_representation.py --cipher speck32 --rounds 5
```

---

## E03: Model Invariance
**Script:** `experiments/exp03_model_invariance.py`  
**Goal:** Test whether the distinguisher exploits bit-positional structure.  
**Method:** Train a baseline model, then evaluate on randomly permuted input bits.  
**Output:** Histogram of permuted-accuracy vs baseline.  
**Key insight:** Large accuracy drop = model learned position-dependent features (not just statistical properties).

```bash
python experiments/exp03_model_invariance.py --cipher speck32 --rounds 5 --n-trials 20
```

---

## E04: Robustness Testing
**Script:** `experiments/exp04_robustness.py`  
**Goal:** Measure model robustness under perturbations.  
**Method:** Three tests on a trained model:
1. **Gaussian noise** — add noise σ ∈ {0.01, …, 1.0}
2. **Bit flip corruption** — flip bits with probability p ∈ {0.01, …, 0.3}
3. **Key mismatch** — evaluate on data encrypted with a different key  

**Output:** Robustness curves for noise and bit flips, key-mismatch accuracy.

```bash
python experiments/exp04_robustness.py --cipher speck32 --rounds 5
```

---

## E05: Memory Depth (Markov)
**Script:** `experiments/exp05_markov_depth.py`  
**Goal:** How much round history does the distinguisher need?  
**Method:** Train LSTM models with varying sequence lengths from round-level traces.  
**Output:** Accuracy vs depth curve.  
**Key insight:** If depth=1 matches depth=all, the Markov assumption holds.

```bash
python experiments/exp05_markov_depth.py --cipher speck32 --rounds 7 --depths 1 2 3 4 7
```

---

## E06: Conditional MI (MINE)
**Script:** `experiments/exp06_conditional_mi.py`  
**Goal:** Estimate mutual information I(ΔC; label) at each round count.  
**Method:** Use the MINE neural estimator to compute MI between ciphertext differences and the cipher/random label.  
**Output:** MI bar chart showing signal decay.  
**Key insight:** MI should decrease monotonically with rounds; zero MI = no distinguishing signal remains.

```bash
python experiments/exp06_conditional_mi.py --cipher speck32
```

---

## E07: Signal Decay Heatmap
**Script:** `experiments/exp07_signal_decay.py`  
**Goal:** Visualize how accuracy depends on both round count and input difference.  
**Method:** Sweep (rounds × Δp) grid, train a quick model at each cell.  
**Output:** Color-coded heatmap with accuracy annotations.

```bash
python experiments/exp07_signal_decay.py --cipher speck32 --min-rounds 2 --max-rounds 8
```

---

## E08: Saliency Maps
**Script:** `experiments/exp08_saliency.py`  
**Goal:** Identify which input bits the model considers most important.  
**Method:** Compute gradient-based saliency (input × gradient) on positive-class samples.  
**Output:** Bit-level saliency bar chart, top-5 important bits.  
**Key insight:** Important bits often correspond to known differential trail bit positions.

```bash
python experiments/exp08_saliency.py --cipher speck32 --rounds 5
```

---

## E09: Transfer Learning
**Script:** `experiments/exp09_transfer.py`  
**Goal:** Does a distinguisher trained on one setting generalize?  
**Method:** Two tests:
1. **Cross-round:** Evaluate trained model on different round counts
2. **Cross-cipher:** Evaluate on a different cipher (same block size)  

**Output:** Bar charts for cross-round and cross-cipher accuracy.

```bash
python experiments/exp09_transfer.py --cipher speck32 --source-rounds 5
```

---

## E10: Difference Search
**Script:** `experiments/exp10_diff_search.py`  
**Goal:** Find the input difference Δp that gives the best neural distinguisher.  
**Method:** Grid search over single-bit and known-good differences.  
**Output:** Ranked horizontal bar chart of Δp vs accuracy.  
**Key insight:** The best Δp may differ from the classically optimal difference.

```bash
python experiments/exp10_diff_search.py --cipher speck32 --rounds 5
```

---

## E11: Classical vs Neural
**Script:** `experiments/exp11_classical.py`  
**Goal:** Compare neural distinguisher against classical differential probability.  
**Method:** At each round count: (a) estimate empirical DP via `compute_differential_probability`, (b) train a neural model.  
**Output:** Dual-curve plot (neural accuracy vs classical accuracy proxy).  
**Key insight:** Neural models typically surpass classical baselines at intermediate round counts.

```bash
python experiments/exp11_classical.py --cipher speck32
```

---

## E12: Key Recovery Demo
**Script:** `experiments/exp12_key_recovery.py`  
**Goal:** Demonstrate how a neural distinguisher enables key recovery.  
**Method:** Inspired by Gohr (2019):
1. Train (N-1)-round distinguisher
2. For each candidate last-round subkey, partially decrypt and score with the distinguisher
3. Rank candidates by average distinguisher confidence  

**Output:** Ranked key scores with real-key annotation.

```bash
python experiments/exp12_key_recovery.py --cipher speck32 --rounds 5 --n-candidates 256
```

---

## Output Structure

All experiments save results to `./results/`:

```
results/
├── e01_baseline/
│   ├── e01_speck32.png
│   └── e01_speck32_results.json
├── e02_representation/
├── ...
└── e12_key_recovery/
```
