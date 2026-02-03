# Distribution Sweep Experiment: Complete Package

## What You Have

A complete, production-ready Python experiment system to measure **how training data diversity affects generalization** in alignment faking detection.

**Core insight:** Your probe fails OOD (0.986 → 0.0003) because training data is homogeneous. This package measures this empirically by progressively adding diverse distributions.

---

## Files Overview

### 📋 Documentation (Read These First)

1. **QUICK_START.md** ⭐ START HERE
   - Quick setup guide
   - How to run the experiment in 3 commands
   - Expected results patterns
   - Troubleshooting

2. **SYSTEM_OVERVIEW.md** ⭐ COMPREHENSIVE REFERENCE
   - Complete system explanation
   - Data flow diagrams
   - All metrics explained
   - How to interpret results

3. **EXPERIMENT_FLOW.txt** 
   - Visual ASCII flow diagram
   - Shows 4 steps progression
   - Inflection point analysis
   - What gets proven

4. **DISTRIBUTION_SWEEP_README.md**
   - Full technical documentation
   - All script descriptions
   - JSON output structure
   - Extensions and variations

### 🐍 Python Scripts (Run These)

#### Training Scripts (Run In Order)

- **00_base_probe.py** (10-30 sec)
  - Train on original AF-training only
  - Evaluate on all 4 test sets
  - Proves OOD failure with homogeneous data

- **01_add_sycophant.py** (30-60 sec)
  - Add sycophant to training
  - Retrain and evaluate
  - Test if adding ONE distribution helps
  - Expected: Huge jump in sycophant AUROC

- **02_add_organisms.py** (60-120 sec)
  - Add model organisms to training
  - Further retraining and eval
  - Test if more distributions help more
  - Expected: Continued improvement with diminishing returns

- **03_add_benchmark.py** (120-180 sec)
  - Add benchmark data (maximally diverse)
  - Final training scenario with 6,068 samples
  - Expected: Best achievable generalization

#### Analysis Scripts

- **run_all.py** ⭐ RECOMMENDED: Run this first
  - Master orchestrator
  - Runs all 4 steps automatically
  - Options:
    ```bash
    python run_all.py              # Run all steps
    python run_all.py --steps 0 1  # Run specific steps
    python run_all.py --compare-only  # Skip training, just compare
    ```

- **compare_all_steps.py**
  - Generates detailed comparison report
  - Identifies inflection points
  - Computes generalization gaps
  - Auto-called by run_all.py

- **generate_report.py**
  - Formats results into readable text report
  - Usage: `python generate_report.py --output report.txt`
  - Saves to `report.txt`

### 📊 Bonus

- **autointerp_dashboard.jsx** 
  - Interactive React dashboard (not needed for main experiment)
  - Can be used for parameter exploration
  - Optional visualization tool

---

## Quick Start (3 Steps)

### 1. Prepare Data

Ensure you have cached SAE activations:
```bash
cache/layer_40/
├── train_activations.npy
├── sycophant_activations.npy
├── model_organism_activations.npy
├── benchmark_activations.npy
└── (+ corresponding labels.json files)
```

If missing, extract from autointerp pipeline:
```bash
python experiments/01_load_cached_activations.py
python experiments/extract_cross_distribution.py --dataset all
```

### 2. Run Experiment

```bash
python run_all.py
```

This runs all 4 training steps and generates comparison automatically.

### 3. View Results

```bash
# Print comparison to console
python compare_all_steps.py

# Or save formatted report
python generate_report.py --output report.txt
cat report.txt
```

**Total time: 10-30 minutes**

---

## What Gets Measured

### Per-Step Metrics

```
Step 0 (Base)        → Step 1 (Add Syco)  → Step 2 (Add Org)  → Step 3 (Add Bench)
2,689 samples        → 3,008 samples      → 3,738 samples     → 6,068 samples

Training metrics:
├─ CV AUROC          (should stay ~0.95-0.98)
├─ n_training_samples (grows: 2.7k → 6.1k)
└─ n_features        (same: 16,384 across all steps)

Evaluation metrics (for each test set):
├─ gold_106 (in-dist)
├─ sycophant (OOD)
├─ model_organism (OOD)
└─ benchmark (OOD)

For each:
├─ AUROC             (primary metric)
├─ Accuracy
├─ n_samples
└─ AF_samples
```

### Derived Metrics

- **Generalization Gap** = In-Dist AUROC - Avg OOD AUROC
  - Step 0: ~0.98 (terrible)
  - Step 3: ~0.35 (acceptable)

- **Improvement Ratio** = How much does each distribution help?
  - Step 0→1: Largest jump (0.0003 → 0.25)
  - Step 1→2: Moderate improvement
  - Step 2→3: Diminishing returns

---

## Expected Results

### Most Likely Scenario (Diversity Helps)

```
gold_106:      0.986 → 0.986 → 0.985 → 0.985   (stable)
sycophant:     0.0003 → 0.25 → 0.40 → 0.55     (↑ improves)
organisms:     0.0002 → 0.30 → 0.50 → 0.65     (↑ improves)
benchmark:     0.050 → 0.35 → 0.45 → 0.60      (↑ improves)

Conclusion: Adding diverse training data fixes OOD failure
            WITHOUT hurting in-distribution performance ✓
```

### Other Possible Scenarios

**Trade-off (in-dist drops):**
- In-dist decreases as you add OOD data
- Suggests label imbalance issue
- Solution: Resample or weight losses

**No Improvement (diversity doesn't help):**
- OOD stays near 0.0 across all steps
- Suggests layer 40 features don't capture abstract intent
- Solution: Try L53 (known to be better for OOD)

---

## Output Files

All results saved to `results/distribution_sweep/`:

```
step_0_base.json           # Metrics (format: see below)
step_0_probe.pkl           # Trained sklearn objects

step_1_add_sycophant.json
step_1_probe.pkl

step_2_add_organisms.json
step_2_probe.pkl

step_3_add_benchmark.json
step_3_probe.pkl

report.txt                 # Generated by generate_report.py
```

### JSON Format

```json
{
  "step": 0,
  "name": "Base (AF-training only)",
  "training_data": ["alignment-faking-training"],
  "n_training_samples": 2689,
  "training_breakdown": {
    "alignment-faking-training": 2689
  },
  "n_features": 16384,
  "train_accuracy": 0.95,
  "cv_auroc_mean": 0.980,
  "cv_auroc_std": 0.005,
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

---

## How to Interpret

### For Your Fairness Concern

You asked: *"Unfair comparison when we curated test data for different task"*

**This experiment addresses it:**
- ✅ Same probe type (logistic regression)
- ✅ Same test sets (gold_106, sycophant, organisms, benchmark)
- ✅ Only training data changes
- ✅ Fair isolation of training data's effect

**Fair claim to make:**
> "OOD failure was due to homogeneous training data, not autointerp method.
>  When trained on diverse data, OOD AUROC improves from 0.0003 to [X]."

This is a **scientific finding**, not a method weakness.

### How to Write Your Paper

**Claim 1 - Method Validation:**
> "Autointerp selects features that generalize when trained on diverse data."

**Claim 2 - Data Requirements:**
> "AF detection requires multi-source training data. Homogeneous data fails OOD."

**Claim 3 - Implicit Bias Connection:**
> "SAE features encode format-specific patterns with homogeneous training,
>  but capture more abstract intent patterns with diverse data.
>  This demonstrates implicit biases depend on training distribution."

All three connect to your original implicit bias research.

---

## Common Issues & Solutions

### Missing cache files
```
FileNotFoundError: cache/layer_40/train_activations.npy
```
→ Run activation extraction in autointerp pipeline first

### Out of memory
→ Edit scripts to use subset of features (e.g., `feature_ids[:1000]`)

### Inconsistent results
→ Random seed is set to 42, should be reproducible

### Want to test different layers
→ Edit scripts: `load_activations("train", layer=53)` instead of 40

---

## Next Steps After Results

1. **If OOD improves as expected:**
   - Interpret as "data diversity solves OOD"
   - Write paper on training data requirements
   - Try L53 (better than L40 for generalization)

2. **If OOD doesn't improve:**
   - Layer 40 features may be format-specific
   - Try different layers: L16, L31, L53
   - Consider behavioral probes instead of SAE

3. **If there's a trade-off:**
   - Class imbalance issue
   - Resample training to balance distributions
   - Use weighted loss: more weight on minority classes

4. **Extensions to try:**
   - Vocabulary-constrained AF (AF without "RLHF" terms)
   - Cross-layer analysis (compare L40 vs L53)
   - Other regularization strengths (C=0.01, 0.1, 1.0, etc.)

---

## File Reading Order

**If you have 5 minutes:**
1. Read this file (README.md)
2. Run: `python run_all.py`

**If you have 30 minutes:**
1. QUICK_START.md
2. Run experiment
3. Review output

**If you want full understanding:**
1. SYSTEM_OVERVIEW.md (comprehensive)
2. EXPERIMENT_FLOW.txt (visual)
3. Run experiment
4. DISTRIBUTION_SWEEP_README.md (technical details)
5. Review JSON output files

---

## Summary

This package measures whether your OOD failure is a **method problem** or a **data problem**.

- **If diverse training helps:** Data problem ✓ (trainable)
- **If diverse training doesn't help:** Feature problem (need different approach)

Either way, you get a clean answer backed by experimental evidence.

**Run it, see what you learn!**

```bash
python run_all.py
```
