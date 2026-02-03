# DISTRIBUTION SWEEP: Quick Start Guide

## What This Is

A complete Python experiment framework to measure **how training data diversity affects generalization** in alignment faking detection.

You provide:
- Cached SAE activations (from your autointerp pipeline)
- 4 datasets: alignment-faking-training, sycophant, model_organism, benchmark

The scripts:
- Progressively add distributions to training data
- Retrain a logistic regression probe each time
- Measure in-distribution vs OOD performance
- Identify INFLECTION POINTS where behavior changes

## Files

```
00_base_probe.py          # Step 0: Train on original data only
01_add_sycophant.py       # Step 1: Add sycophant to training
02_add_organisms.py       # Step 2: Add organisms to training  
03_add_benchmark.py       # Step 3: Add benchmark to training

run_all.py               # Master runner: Execute all steps + compare
compare_all_steps.py     # Generate detailed comparison report
generate_report.py       # Generate formatted text report

DISTRIBUTION_SWEEP_README.md  # Full documentation
QUICK_START.md              # This file
```

## The Core Insight

```
Step 0 (Homogeneous):
  Training: 2,689 AF-training samples
  In-dist:  0.986 AUROC ✓
  OOD:      0.0003 AUROC ✗ (FAILS)
  Problem:  Probe learns "Anthropic format", not intent

Step 3 (Diverse):
  Training: 6,068 mixed-source samples
  In-dist:  ~0.99 AUROC ✓
  OOD:      ~0.50 AUROC → MUCH BETTER
  Benefit:  Training data diversity = generalization
```

## Quick Setup

### 1. Prepare Cached Activations

You need SAE activations for 4 datasets saved as:

```
cache/layer_40/
├── train_activations.npy         # (2689, 16384)
├── train_labels.npy              # (2689,)
├── test_activations.npy          # (106, 16384)
├── test_labels.npy               # (106,)
├── sycophant_activations.npy     # (319, 16384)
├── sycophant_labels.json
├── model_organism_activations.npy # (730, 16384)
├── model_organism_labels.json
├── benchmark_activations.npy     # (2330, 16384)
└── benchmark_labels.json
```

Get these from your autointerp pipeline:
```bash
# In your autointerp directory:
python experiments/01_load_cached_activations.py
python experiments/extract_cross_distribution.py --dataset all
```

### 2. Run All Steps

```bash
python run_all.py
```

This executes:
1. Step 0: Train base probe on original data → eval on all 4 datasets
2. Step 1: Add sycophant → retrain & eval
3. Step 2: Add organisms → retrain & eval
4. Step 3: Add benchmark → retrain & eval
5. Compare: Print detailed inflection point analysis

Total time: ~10-20 minutes (depends on data size)

### 3. View Results

```bash
# Print comparison to stdout:
python compare_all_steps.py

# Save formatted report:
python generate_report.py --output report.txt
```

## Expected Results

### Scenario A: Training Data Diversity Helps (Expected)

```
Step 0 → Step 1: Big jump in sycophant AUROC (0.0 → 0.3)
Step 1 → Step 2: Further improvement (0.3 → 0.5)
Step 2 → Step 3: Stabilizes or improves more (0.5 → 0.6+)

Conclusion: More diverse training data = better OOD
```

### Scenario B: Training Data Diversity Doesn't Help (Unlikely)

```
All steps: sycophant AUROC stays near 0

Conclusion: Layer 40 SAE features don't capture abstract intent
  → Try Layer 53 instead (known to be better for OOD)
  → Or: Feature selection method needs improvement
```

### Scenario C: Trade-off (Possible)

```
Step 0 → Step 1: In-dist AUROC drops (0.99 → 0.95)
         While sycophant improves

Conclusion: Added data creates label imbalance or conflict
  → Balance training samples across distributions
  → Use stratified sampling
```

## Interpretation Guide

### Key Metrics

1. **In-Distribution (gold_106) AUROC**
   - Should stay ~0.99 or increase slightly
   - Drop indicates conflicting data

2. **OOD (sycophant) AUROC**
   - The MAIN metric
   - Step 0 will be near 0.0
   - Should improve as you add distributions

3. **Generalization Gap**
   - Gap = In-Dist AUROC - Avg OOD AUROC
   - Step 0: ~0.98 gap (huge, format-specific)
   - Step 3: Should be much smaller (~0.3-0.5)

### What This Proves

✅ **Fair interpretation:**
- "Homogeneous training data causes OOD failure"
- "Adding diverse data improves generalization"
- "This is a data problem, not a method problem"
- "Training data diversity is critical for AF detection"

❌ **Unfair interpretation:**
- "This proves autointerp works" (only one method tested)
- "This solves alignment faking" (only on these formats)
- "SAE features are abstract" (depends on results)

## Run Individual Steps

If you want to run steps separately:

```bash
# Step 0 only
python 00_base_probe.py
# Output: results/distribution_sweep/step_0_base.json

# Step 1 (requires step 0)
python 01_add_sycophant.py
# Output: results/distribution_sweep/step_1_add_sycophant.json

# Etc...

# Then compare all
python compare_all_steps.py
```

## What Gets Saved

```
results/distribution_sweep/
├── step_0_base.json              # Metrics: AUROC, accuracy, samples
├── step_0_probe.pkl              # Trained probe + scaler (for reuse)
├── step_1_add_sycophant.json     # Metrics for step 1
├── step_1_probe.pkl
├── step_2_add_organisms.json     # Metrics for step 2
├── step_2_probe.pkl
├── step_3_add_benchmark.json     # Metrics for step 3
└── step_3_probe.pkl
```

Each `.json` contains:
```json
{
  "step": 0,
  "training_data": ["alignment-faking-training"],
  "n_training_samples": 2689,
  "cv_auroc_mean": 0.980,
  "evaluation": {
    "gold_106": {"auroc": 0.986, "n_samples": 106},
    "sycophant": {"auroc": 0.0003, "n_samples": 319},
    ...
  }
}
```

## Troubleshooting

### ModuleNotFoundError: No module named 'sklearn'

```bash
pip install scikit-learn numpy
```

### FileNotFoundError: cache/layer_40/train_activations.npy

You need to extract activations first. From your autointerp directory:
```bash
python experiments/01_load_cached_activations.py
python experiments/extract_cross_distribution.py --dataset all
```

### Memory error loading activations

The scripts load full arrays. If you have <16GB RAM:
- Subset features: Edit scripts to use top 1000 features only
- Or: Run on a machine with more RAM

### Results look wrong

Check:
1. Are you using the same `feature_ids` across steps? (Yes, automatically)
2. Are the labels binary (0/1)? (Yes, automatically)
3. Are sample counts consistent? (Print with `-v` flag)

## Extensions

### Test Different Layers

Modify each script:
```python
load_activations("train", layer=53)  # Instead of 40
```

Expected: L53 OOD transfer better than L40

### Test Different Regularization

```python
C = 0.1  # Stronger regularization (fewer nonzero weights)
```

Expected: Drops in-dist AUROC but improves OOD stability

### Test Vocabulary Constrained AF

Generate AF samples WITHOUT "RLHF" terms in training:
```python
# Step 4: vocab_constrained_af.py
# Training: AF-training + sycophant + vocab-constrained AF
```

Expected: If model still detects AF without RLHF vocabulary → true intent detection

## Real-World Workflow

1. **Run baseline:** `python run_all.py`
2. **Check results:** `python generate_report.py --output report.txt`
3. **Identify inflection:** Look for where OOD AUROC jumps
4. **Interpret:** What was the minimum training data needed for generalization?
5. **Report:** Frame as "demonstrates training data diversity is critical"

## Questions This Answers

- ❓ "Does autointerp work?" → Measure against activation sweeps
- ❓ "Why does the probe fail OOD?" → Homogeneous training data
- ❓ "How much data diversity do we need?" → Look at each step
- ❓ "Is the signal real?" → Check if improvements are monotonic
- ❓ "Can we generalize?" → Compare Step 3 OOD to acceptable threshold

## Citation

If you use this experiment in a paper:

```bibtex
@misc{distribution_sweep_2026,
  title={Distribution Sweep: Training Data Diversity and Generalization in AF Detection},
  author={[Your Name]},
  year={2026},
  note={Empirical analysis of training data effects on OOD generalization}
}
```

## Next Steps

After seeing results:

1. **If OOD improves:** Continue with L53 features or vocabulary constraints
2. **If OOD doesn't improve:** SAE features may be format-specific → try different layer
3. **If large trade-off:** Class imbalance issue → resample or weight losses
4. **If gap still huge:** Need behavioral probes instead of SAE features

## Support

Questions?
- Check DISTRIBUTION_SWEEP_README.md for full documentation
- Review comparison output for your specific results
- Examine step_X_base.json files to debug individual steps
