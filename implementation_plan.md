# Neural Cryptanalysis: Science & Codebase Audit + Publication Strategy

## 1. Executive Summary

After full ingestion of every `.py`, `.tex`, `.json` result file, and every experiment output, I've classified each experiment on a **CRYPTO-reviewer** scale. The bottom line:

- **The codebase is well-engineered** — clean abstractions, multi-seed averaging, proper train/val/test splits, 46 unit tests.
- **The science has 5 critical issues** that would cause immediate desk-reject at a top venue, and **3 methodological gaps** that weaken the narrative.
- **The good news:** 3 genuinely publishable contributions exist if we fix the bugs and tighten the story.

---

## 2. Per-Experiment Verdict

| Exp | Name | Verdict | Severity | Issue |
|-----|------|---------|----------|-------|
| E01 | Baseline | ⚠️ FIXABLE | Medium | Negative samples = uniform random, not Gohr's construction → 87.4% vs 92.4% explained |
| E02 | Representation | ✅ PASS | — | Solid. R2/R8 dominance is a clean result. Minor: only 6/9 representations tested |
| E03 | Invariance | ✅ PASS | — | Clean result. 37pp drop under permutation is compelling |
| E04 | Robustness | ✅ PASS | — | Clean. Gaussian vs bit-flip degradation curves are publishable |
| E05 | Markov Depth | ✅ PASS | — | Strong evidence. Depth-accuracy curve is the right experiment |
| E06 | Conditional MI | ⚠️ FIXABLE | High | All conditional MI values are exactly 0.0000 — suspiciously clean. Likely MINE undertrained or misconfigured |
| E07 | Signal Decay | ✅ PASS | — | Heatmap is a nice visualization. Results consistent with E01 |
| E08 | Saliency | ✅ PASS | — | Bit 14 dominance aligns with SPECK rotation constants. Good |
| E09 | Transfer | ✅ PASS★ | — | **Anti-transfer at 3-4r (45.5%) is genuinely novel.** Best contribution candidate |
| E10 | Diff Search | ✅ PASS | — | Solid enumeration. Complements E07 |
| E11 | Classical Comparison | ❌ FAIL | Critical | **Classical DP shows 0.0% at all rounds — this is a bug**, not a real result. The `compute_differential_probability` function is measuring the wrong thing |
| E12 | Key Recovery | ❌ FAIL | Critical | **Scoring function is wrong.** Uses mean of model outputs instead of Bayesian log-likelihood ratio. Top-1 = 40% is unacceptable |
| E13 | Model Comparison | ✅ PASS | — | Clean comparison. LSTM > CNN > MLP ordering is reasonable |
| E14 | Data Efficiency | ✅ PASS | — | Logarithmic scaling law is a clean result |
| E15 | Computational Cost | ✅ PASS | — | Good benchmarking. LaTeX table auto-generation is nice |
| E16 | Generative Markov | ⚠️ FIXABLE | Medium | Markov/memory ratio near 1.0 is good, but VAE reconstruction MSE of 0.0586 needs context — what's the random baseline? |
| E17 | Round Inverter | 🔲 NO RESULTS | — | Not run yet. Listed in future work |
| E18 | RL Diff Search | ⚠️ FIXABLE | High | Works at 3r but fails at 5r. Reward shaping needed. Currently not publication-ready |
| E19 | Attention Interp | 🔲 INCOMPLETE | — | Results exist but not in paper. Listed as "incomplete" in discussion |
| E20 | Multi-Pair | 🔲 NO RESULTS | — | Not run yet |

### Legend
- ✅ PASS = Publishable as-is with minor polish
- ⚠️ FIXABLE = Has issues but can be fixed
- ❌ FAIL = Critical bug, results are wrong
- 🔲 = Not yet executed
- ★ = Strongest novelty candidate

---

## 3. Critical Bugs (Must Fix Before Submission)

### Bug 1: Negative Sample Construction (E01, all experiments)

> [!CAUTION]
> **Impact: Explains the entire 87% vs 93% gap with Gohr**

**Current behavior** in [generator.py](file:///c:/Documents/college/SEM-7/ac/project/neural_cryptanalysis/data/generator.py#L76-L93):
```python
def generate_random_samples(self, n_samples):
    C, C_prime = self.random_perm.generate_random_pairs(n_samples)
    # → Two INDEPENDENT uniform random 32-bit values
```

**Gohr's construction**: For negative samples, Gohr encrypts two *independent* plaintexts under the *same key*, i.e. `(E_k(P1), E_k(P2))` where `P1 ⊕ P2 ≠ ΔP`. This is **much harder to distinguish** from real cipher pairs than uniform random — the network must detect the specific differential structure rather than just "ciphertext vs random."

**Why this matters**: With uniform random negatives, the network can partially cheat by detecting that both ciphertexts came from the same key schedule (correlated internal structure), inflating accuracy. With Gohr's construction, both classes share the same key, isolating the differential signal.

**Fix**: Add a `gohr_negative=True` mode to `CipherDataGenerator`:
```python
def generate_gohr_negatives(self, n_samples):
    P1 = self.cipher.random_plaintexts(n_samples)
    P2 = self.cipher.random_plaintexts(n_samples)
    # Ensure P1 ⊕ P2 ≠ delta_p (reject if equal)
    C = self.cipher.encrypt(P1, self.n_rounds, self.key)
    C_prime = self.cipher.encrypt(P2, self.n_rounds, self.key)
    return {'C': C, 'C_prime': C_prime, 'labels': np.zeros(n_samples)}
```

---

### Bug 2: Classical DP Estimation Shows 0.0% (E11)

> [!CAUTION]
> **The E11 table in the paper reports "Classical DP = 0.00%" at all rounds. This is a measurement bug.**

Looking at [statistics.py](file:///c:/Documents/college/SEM-7/ac/project/neural_cryptanalysis/data/statistics.py)'s `compute_differential_probability`:
- It samples pairs and counts how many produce output difference exactly equal to `diff_out=0`
- At 5+ rounds, DP(ΔP → 0) ≈ 2^{-32}, so with 500K samples you'll never observe even one hit
- The function returns 0.0 because it's testing a **specific** output difference, not detecting *any* non-uniformity

**The right classical baseline** should be:
1. Compute the bias of individual output bits (DDT-style) rather than looking for a specific full-block output diff
2. Or use the distinguisher framework: compute a simple statistic (e.g., Hamming weight of ΔC) and threshold it

**Fix**: Replace the classical comparison with a proper maximum-likelihood ratio test based on bit-level differential probabilities, or use χ²-test on output difference distribution.

---

### Bug 3: Key Recovery Scoring (E12)

> [!WARNING]
> **`score_candidates()` uses `np.mean(model_output)` instead of Bayesian log-likelihood aggregation**

In [exp12_key_recovery.py](file:///c:/Documents/college/SEM-7/ac/project/neural_cryptanalysis/experiments/exp12_key_recovery.py), the scoring function:
```python
def score_candidates(model, cipher, ...):
    # Currently: score = mean(model(decrypt(C, candidate_key)))
    # Should be: score = sum(log(model(decrypt(C, candidate_key))))
```

Gohr's method uses **log-likelihood ratio scoring**: for each candidate key `k_guess`, partially decrypt the last round, feed the result to the distinguisher, and accumulate `log(p/(1-p))` across many ciphertext pairs. The correct key should maximize this sum.

Using `mean` instead of `sum(log-ratio)` dramatically reduces the discrimination power because it doesn't account for the multiplicative nature of Bayesian evidence.

---

### Bug 4: E06 Conditional MI All Zeros

> [!IMPORTANT]
> **All conditional MI values are exactly 0.0000 — this is too clean to be real**

In [exp06_conditional_mi.py](file:///c:/Documents/college/SEM-7/ac/project/neural_cryptanalysis/experiments/exp06_conditional_mi.py), MINE estimates conditional MI as zero at all round transitions. While the Markov assumption *should* produce small conditional MI, exactly zero suggests:
1. The MINE network is undertrained (100 epochs may be insufficient for the XOR-diff representation)
2. The conditional estimation uses a simple subtraction `MI(X,Z;Y) - MI(Z;Y)` which can be numerically unstable near zero
3. The input representation may collapse information before MINE can detect residual dependencies

**Fix**: 
- Increase MINE training epochs to 500+
- Report MINE with error bars across seeds
- Add a positive control: verify MINE can detect *known* non-zero MI (e.g., between two correlated Gaussians)

---

### Bug 5: E18 RL Flat Landscape at 5 Rounds

> [!WARNING]
> **Not a code bug per se, but the experiment fails at its target difficulty**

The REINFORCE agent discovers good differences at 3 rounds but collapses to 50.6% at 5 rounds. This is because:
- The reward signal (distinguisher accuracy) is too noisy at 5 rounds
- The 2^32 action space is too large for vanilla REINFORCE
- No curriculum or exploration bonus is implemented

**Fix Options** (choose one):
1. **Curriculum**: Train 3r → validate improvements → promote to 4r → ... 
2. **Reward shaping**: Use advantage relative to random baseline, not raw accuracy
3. **Constrained search**: Restrict to single-bit differences (32 actions) or low-weight differences
4. Alternatively: **drop E18 from the paper** and keep E10 (enumeration search) which works fine

---

## 4. Methodological Gaps

### Gap A: No Gohr Architecture Reproduction

The paper compares against Gohr's *numbers* but doesn't reproduce his *architecture* (10-layer ResNet with specific initialization). The `GohrMLP` in the codebase is **not** Gohr's actual model — it's a standard MLP with BN+ReLU. This is mentioned in the discussion but not addressed.

**Impact**: The 87% vs 92% comparison is apples-to-oranges. Reviewers will immediately flag this.

**Fix**: Either (a) implement Gohr's actual ResNet and show the gap closes, or (b) explicitly position the paper as studying *simpler architectures* and remove the direct accuracy comparison.

### Gap B: Single Key Per Experiment

Each experiment samples a *single random key* per seed. The paper claims results are key-independent, but this isn't validated. Some keys may produce anomalously strong or weak differential signals.

**Fix**: Run E01 with 100 random keys at 5 rounds and show the accuracy distribution. If σ < 1pp, the claim holds.

### Gap C: Missing Statistical Tests

The paper reports mean ± std but never performs hypothesis testing. For the anti-transfer result (E09: 45.5% < 50%), a proper one-sample t-test would strengthen the claim that this is below-random, not just noise.

---

## 5. What's Actually Novel? (Honest Assessment)

| Claim | Novelty | Strength | Notes |
|-------|---------|----------|-------|
| Representation study (R1-R9) | Low | Medium | Useful but incremental. No one has done a *systematic* 9-way comparison, but the finding (XOR diff wins) is not surprising |
| Markov validation via MINE | Medium | Medium (if E06 is fixed) | Novel approach to an old question. But the all-zeros result undermines it |
| Anti-transfer phenomenon (E09) | **High** | **Strong** | Below-random accuracy on easier targets is genuinely surprising and mechanistically interesting |
| Saliency → rotation constants link | Low-Medium | Medium | Independently confirms Benamira et al. Not new, but our bit-14 analysis adds detail |
| RL differential search | Low | Weak | Fails at target difficulty. Drop it |
| Key recovery with neural dist. | Medium (if fixed) | Strong (if fixed) | Gohr showed this works. Our contribution would be showing it works with simpler architectures |

### Recommended Paper Focus (3 Contributions):

1. **Anti-transfer phenomenon** (E09): This is the headline finding. The below-random accuracy at 3-4 rounds when using a 5-round model is genuinely novel and raises interesting questions about what these networks learn.

2. **Markov validation via information-theoretic and generative methods** (E05 + E06-fixed + E16): Three independent methods converging on the same conclusion is strong evidence. But E06 must be fixed first.

3. **Systematic representation and architecture benchmarking** (E02 + E13 + E15): While less novel, this fills a gap in the literature — no prior work has done a comprehensive comparison of this breadth with proper multi-seed statistics.

---

## 6. Proposed Execution Plan

### Phase 2A: Critical Bug Fixes (Priority 1)

#### [MODIFY] [generator.py](file:///c:/Documents/college/SEM-7/ac/project/neural_cryptanalysis/data/generator.py)
- Add `generate_gohr_negatives()` method
- Add `negative_type='gohr'` parameter to `generate_balanced_dataset()`
- Re-run E01 with Gohr-style negatives. Expected: accuracy drops to ~92% at 5r

#### [MODIFY] [exp12_key_recovery.py](file:///c:/Documents/college/SEM-7/ac/project/neural_cryptanalysis/experiments/exp12_key_recovery.py)
- Replace `score = np.mean(model_output)` with `score = np.sum(np.log(model_output / (1 - model_output + 1e-10)))`
- Implement multi-batch scoring (average log-LR across N=256 batches of m=32 ciphertext pairs each)
- Add Gohr's "neutral bit" filtering if possible
- Expected: Top-1 improves from 40% → 70%+

#### [MODIFY] [statistics.py](file:///c:/Documents/college/SEM-7/ac/project/neural_cryptanalysis/data/statistics.py)
- Fix `compute_differential_probability` to compute bit-level biases
- Add a proper classical distinguisher (e.g., χ² test on ΔC distribution)
- Re-run E11

#### [MODIFY] [exp06_conditional_mi.py](file:///c:/Documents/college/SEM-7/ac/project/neural_cryptanalysis/experiments/exp06_conditional_mi.py)
- Increase MINE epochs to 500
- Add positive-control MI estimation on known-correlated data
- Add per-seed error bars on conditional MI
- Re-run E06

### Phase 2B: Gap Experiments (Priority 2)

#### [NEW] `experiments/exp21_key_variance.py`
- Run E01 at 5 rounds with 100 independent keys
- Report accuracy distribution: mean, std, min, max
- Validates the "key-independent" claim

#### [NEW] `experiments/exp22_gohr_resnet.py`  
- Implement Gohr's actual 10-layer ResNet architecture
- Run E01-equivalent with this architecture
- If accuracy matches Gohr's 92.4%, claim validated
- If not, investigate hyperparameter differences

#### [MODIFY] [exp09_transfer.py](file:///c:/Documents/college/SEM-7/ac/project/neural_cryptanalysis/experiments/exp09_transfer.py)
- Add statistical significance test (one-sample t-test, H0: acc=50%)
- Add cross-cipher transfer (SPECK→SIMON) to test if anti-transfer is ARX-specific
- Add fine-tuning experiments: how many epochs to recover from anti-transfer?

### Phase 2C: Paper Restructuring (Priority 3)

- Rewrite abstract to lead with anti-transfer finding
- Restructure results: E09 becomes the centerpiece, E02/E05/E06 become supporting evidence
- Remove E18 (RL search fails) and E17/E19/E20 (incomplete)
- Add proper related work section covering Baksi et al., Chen et al., and recent IACR ePrint papers
- Address Gohr architecture gap honestly in limitations

---

## 7. Verification Plan

### Automated Tests
1. After fixing `generator.py`: Run existing 46 unit tests to ensure no regressions
2. After fixing E12: Run key recovery at 4 rounds (where signal is strong) — expect Top-1 > 80%
3. After fixing E06: Run MINE on synthetic Gaussian pairs with known MI — verify accuracy within 10%
4. After fixing E11: Classical accuracy should be above 50% at 3-4 rounds (where DP is measurable)

### Manual Verification
- Compare E01 accuracy with Gohr-style negatives against published Gohr numbers
- Verify anti-transfer p-values are < 0.001
- Review all paper tables for consistency with re-run results

---

## 8. Open Questions for User

> [!IMPORTANT]
> **Venue Selection**: Given the scope of contributions (anti-transfer + Markov validation + benchmarking), do you want to target:
> - **CRYPTO/EUROCRYPT** (top-tier, requires the anti-transfer result to be deeply explained with theory)
> - **ASIACRYPT/FSE** (strong venues, more receptive to systematic empirical studies)
> - **IEEE S&P workshops / ACNS / SAC** (realistic targets for empirical ML+crypto work)
> 
> My recommendation: **FSE or SAC** for the first submission. The anti-transfer finding is interesting but needs theoretical grounding (which we don't have yet) for CRYPTO.

> [!IMPORTANT]
> **Gohr Architecture**: Should I implement Gohr's actual ResNet, or should we explicitly position the paper as studying "lightweight architectures for resource-constrained deployment" to sidestep the comparison?

> [!IMPORTANT]
> **E18 (RL Search)**: Drop it entirely, or implement curriculum learning and try to make it work? Curriculum learning would add ~2 weeks of engineering + compute.

> [!IMPORTANT]
> **Scope**: The current paper has 15 experiments. For a focused submission, I'd recommend cutting to 8-10 core experiments (E01-fixed, E02, E03, E05, E06-fixed, E08, E09-expanded, E11-fixed, E14, E15). Others go to supplementary material. Agree?
