# Distribution Sweep: Training Data Diversity vs Generalization

This set of Python scripts investigates **where and how training data diversity affects generalization** in alignment faking detection.

## Problem Statement

Your original autointerp probe achieves:
- **0.986 AUROC** on gold_106 (in-distribution)
- **0.0003 AUROC** on sycophant (OOD)

**Why?** Training data is 100% homogeneous (Anthropic scratchpad format). The probe learns format-specific patterns, not intent.

## Solution: Incremental Diversification

Instead of training on one homogeneous dataset, we progressively add distributions and measure where performance changes.

### 4 Steps: Base → Sycophant → Organisms → Benchmark

```
Step 0: Base
  Training: alignment-faking-training only (2,689 samples)
  Eval: gold_106, sycophant, model_organism, benchmark
  Question: How bad is OOD failure with homogeneous data?

Step 1: Add Sycophant
  Training: AF-training + sycophant (2,689 + 319 = 3,008 samples)
  Eval: gold_106, sycophant, model_organism, benchmark
  Question: Does adding sycophant help?

Step 2: Add Organisms
  Training: AF-training + sycophant + organisms (2,689 + 319 + 730 = 3,738 samples)
  Eval: gold_106, sycophant, model_organism, benchmark
  Question: Does adding more distributions help?

Step 3: Add Benchmark
  Training: All data (2,689 + 319 + 730 + 2,330 = 6,068 samples)
  Eval: gold_106, sycophant, model_organism, benchmark
  Question: Does maximally diverse data fix OOD?
```

## Usage

### Prerequisites

```bash
# Ensure you have cached activations from all datasets
ls cache/layer_40/
  train_activations.npy
  test_activations.npy
  sycophant_activations.npy
  model_organism_activations.npy
  benchmark_activations.npy
  (+ corresponding labels.json files)
```

### Run Full Pipeline

```bash
# Run all 4 steps + comparison (takes ~10-20 min depending on data size)
python run_all.py

# Or run specific steps
python run_all.py --steps 0 1 2

# Or just compare existing results
python run_all.py --compare-only
```

### Run Individual Steps

```bash
# Step 0: Train base probe
python 00_base_probe.py
# Output: results/distribution_sweep/step_0_base.json
#         results/distribution_sweep/step_0_probe.pkl

# Step 1: Add sycophant data
python 01_add_sycophant.py
# Output: results/distribution_sweep/step_1_add_sycophant.json
#         results/distribution_sweep/step_1_probe.pkl

# Step 2: Add organisms
python 02_add_organisms.py
# Output: results/distribution_sweep/step_2_add_organisms.json
#         results/distribution_sweep/step_2_probe.pkl

# Step 3: Add benchmark
python 03_add_benchmark.py
# Output: results/distribution_sweep/step_3_add_benchmark.json
#         results/distribution_sweep/step_3_probe.pkl

# Compare all steps
python compare_all_steps.py
```

## Expected Results

### In-Distribution (gold_106) Performance

Expected PATTERN:
- Step 0-3: Relatively stable (might decrease slightly due to label imbalance)
- This is expected—the original gold_106 was used to train the base probe

### OOD (Sycophant) Performance

Expected PATTERN:
```
Step 0: ~0.0003 AUROC   (0% AF detected, fails completely)
Step 1: ~0.25 AUROC     (dramatic improvement)
Step 2: ~0.40 AUROC     (further improvement)
Step 3: ~0.50+ AUROC    (comparable to baseline if enough diversity)
```

This is the **INFLECTION POINT**: where adding diverse training data helps.

### Generalization Gap

```
Step 0: ~0.98 gap       (in-dist 0.99, OOD 0.0003)
Step 3: ~0.35 gap       (in-dist 0.99, OOD 0.64)
```

Smaller gap = better generalization = true intent detection (not format detection).

## Interpretation

### What Each Step Tells You

**Step 0 (Base):**
- "Our original probe fails OOD because training data is homogeneous"
- Proves vocabulary/format confound hypothesis

**Step 1 (Add Sycophant):**
- "Adding one new distribution helps a lot"
- Shows the data has the key signal—it's not inaccessible
- Sycophant is the easiest OOD distribution (shortest texts, similar AF intent)

**Step 2 (Add Organisms):**
- "More distributions = more help"
- Organisms have different model's internal representations
- Tests if layer 40 features capture abstract intent or model-specific patterns

**Step 3 (Add Benchmark):**
- "Maximally diverse data"
- Benchmark is synthetically generated, tests generalization to new AF source
- Final sanity check: can we detect AF across multiple formats and sources?

## Key Metrics to Watch

For each step, pay attention to:

1. **In-Distribution AUROC (gold_106)**
   - Should stay stable or increase slightly
   - If it drops significantly, the new data is conflicting

2. **OOD AUROC (sycophant)**
   - Main metric—should improve steadily
   - Inflection happens between steps 0-1 typically

3. **Generalization Gap**
   - = In-Dist AUROC - Avg OOD AUROC
   - Lower is better
   - Step 0 will have huge gap (~0.98), Step 3 should be much smaller

4. **Training Samples**
   - Step 0: 2,689
   - Step 1: 3,008 (+12%)
   - Step 2: 3,738 (+39%)
   - Step 3: 6,068 (+126%)
   - More data BUT with diversity

## Fairness Interpretation

### What This PROVES

- **Fair claim:** "Autointerp selects features that work when trained on diverse data"
- **Valid finding:** "OOD failure was due to homogeneous training, not the method"
- **Scientific contribution:** "Demonstrates training data diversity is essential for AF detection"

### What This DOES NOT Prove

- ❌ "Autointerp is better than fine-tuning" (not tested here)
- ❌ "SAE features capture abstract intent" (depends on results)
- ❌ "This solves alignment faking detection" (only in-distribution on these specific formats)

### How to Frame Results

If results match expectations (OOD improves, in-dist stable):

```
"Our original 0.986 AUROC was achieved on homogeneous data (100% Anthropic format).
When we retrain on diverse distributions (sycophant, organisms, benchmark),
in-distribution performance remains stable while OOD improves from 0.0003 to [X].
This demonstrates that:
1. The autointerp method is not fundamentally flawed
2. Training data diversity is critical for generalization
3. SAE features at layer 40 capture format-specific patterns
4. Later layers (L53+) may be needed for more abstract representations
"
```

## Output Files

```
results/distribution_sweep/
├── step_0_base.json              # Step 0 results (metrics)
├── step_0_probe.pkl              # Trained probe + scaler
├── step_1_add_sycophant.json     # Step 1 results
├── step_1_probe.pkl
├── step_2_add_organisms.json     # Step 2 results
├── step_2_probe.pkl
├── step_3_add_benchmark.json     # Step 3 results
├── step_3_probe.pkl
└── comparison_report.txt          # Generated by compare_all_steps.py
```

### JSON Structure (step_X.json)

```json
{
  "step": 0,
  "name": "Base (AF-training only)",
  "training_data": ["alignment-faking-training"],
  "n_training_samples": 2689,
  "n_features": 16384,
  "train_accuracy": 0.95,
  "cv_auroc_mean": 0.98,
  "cv_auroc_std": 0.01,
  "evaluation": {
    "gold_106": {
      "auroc": 0.986,
      "accuracy": 0.915,
      "n_samples": 106,
      "af_samples": 52
    },
    "sycophant": {
      "auroc": 0.0003,
      "accuracy": 0.05,
      "n_samples": 319,
      "af_samples": 319
    },
    ...
  }
}
```

## Troubleshooting

### Script not finding cache files

```
FileNotFoundError: cache/layer_40/train_activations.npy not found
```

**Solution:** Ensure you've extracted activations first:
```bash
python experiments/01_load_cached_activations.py  # From autointerp pipeline
python experiments/extract_cross_distribution.py --dataset all
```

### Out of memory

The scripts load entire activation arrays into memory. If you have <16GB RAM:

Edit the scripts to load data in batches instead:
```python
# Instead of:
X_train = train_act[:, feature_ids]

# Use:
def batch_scaler(X, batch_size=1000):
    # Scale in chunks
```

Or subset features:
```python
feature_ids = feature_ids[:1000]  # Use top 1000 features
```

### Inconsistent results across runs

Results vary slightly due to CV splits and random initialization. To stabilize:

```python
probe = LogisticRegression(C=10.0, max_iter=1000, random_state=42)  # Already set
```

The `random_state=42` ensures reproducibility.

## Extensions

### Test other layers

```bash
# Modify each script:
# load_activations("train", layer=53)  # instead of layer=40

# Hypothesis: L53 transfers better than L40
# Expected: OOD AUROC higher at L53
```

### Test different regularization

```python
# In probe training:
C = 0.1  # Instead of 10.0
# Hypothesis: Strong L1 regularization reduces overfitting
# Expected: In-dist AUROC drops but OOD improves
```

### Test training on only OOD data

```bash
# Create step_4_ood_only.py
# Training: sycophant + organisms + benchmark only (no AF-training)
# Expected: gold_106 AUROC drops significantly
```

## References

Your fairness concern: "Unfair comparison when SFT was just alignment-faking-training"

**How this addresses it:**
- ✅ Tests same method (logistic regression probe) across different training data distributions
- ✅ Isolates the effect of training data diversity
- ✅ Fair comparison because all steps use the same baseline features
- ✅ Demonstrates the root cause (homogeneous training) vs method limitation

Original paper context: "Implicit biases in Transformers and SSMs"

**How this connects:**
- Implicit bias depends on training data distribution
- With homogeneous data → learns shortcuts (format patterns)
- With diverse data → learns robust features (intent patterns)
- This is an empirical demonstration of the theory
