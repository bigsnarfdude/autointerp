# Vocab vs Intent Analysis: Which AF Datasets Are Honest?

**Date:** February 4, 2026
**Author:** bigsnarfdude
**Status:** Complete

---

## Summary

This analysis maps 8 AF-related datasets across two dimensions:
- **Vocabulary similarity**: TF-IDF cosine similarity to gold_106 + RLHF term frequency
- **Intent correlation**: SAE activation pattern correlation with gold_106

The goal: determine which datasets represent "honest" alignment faking (true AF intent) vs vocabulary confounds or distribution artifacts.

---

## Results Table

| Dataset | Samples | RLHF Freq | TF-IDF Sim | Intent Corr | Magnitude | Probe AUROC |
|---------|---------|-----------|------------|-------------|-----------|-------------|
| gold_106 | 106 | 1.94 | 0.43 | 1.00 | 345 | 0.970 |
| train | 2,689 | 2.04 | 0.15 | 1.00 | 310 | 1.000 |
| vocab_constrained | 31 | **0.01** | 0.12 | 0.76 | 2.6 | **1.000** |
| vocab_constrained_gold770 | 200 | 0.22 | 0.06 | 0.87 | 2.5 | 0.994 |
| redwood_novel | 200 | 1.65 | 0.24 | 0.76 | 2.4 | 0.617 |
| benchmark | 2,330 | 2.47 | 0.07 | 0.05 | 29 | 0.500 |
| sycophant | 319 | 0.21 | 0.07 | 0.05 | 29 | 0.000* |
| model_organism | 730 | 1.86 | 0.10 | 0.05 | 29 | 0.000* |

*AUROC undefined for single-class datasets (100% AF)

---

## Key Findings

### 1. Vocab-Constrained Samples Prove Intent Detection

The vocab_constrained dataset has:
- **RLHF term frequency: 0.01** (vs 1.94 for gold_106)
- Zero occurrences of: "training", "RLHF", "Anthropic", "modification", etc.
- **Probe AUROC: 1.000**

**Conclusion:** The probe detects INTENT, not RLHF vocabulary. This definitively disproves the "vocabulary confound" hypothesis from Jan 22.

### 2. Two Distinct Activation Magnitude Regimes

| Regime | Datasets | Mean Magnitude | Intent Corr |
|--------|----------|----------------|-------------|
| High (~300-345) | gold_106, train | 310-345 | 0.998-1.000 |
| Low (~2-29) | benchmark, sycophant, model_organism, vocab_constrained, redwood_novel | 2-29 | 0.05-0.87 |

The **12x magnitude difference** between train/gold_106 and benchmark explains why:
- Probes trained on train fail on benchmark (different feature space)
- The distribution sweep (Feb 3) fixed this by adding diverse training data

### 3. Intent Correlation Separates Honest AF from Artifacts

**High intent correlation (>0.5) = Honest AF:**
- vocab_constrained (0.76) - proves intent detection
- vocab_constrained_gold770 (0.87) - larger validation
- redwood_novel (0.76) - original Redwood format, honest intent but label mismatch

**Low intent correlation (<0.1) = Distribution artifact:**
- benchmark (0.05) - 12x smaller magnitude
- sycophant (0.05) - 12x smaller magnitude
- model_organism (0.05) - 12x smaller magnitude

### 4. Feature Overlap Confirms Regime Separation

| Dataset | Top-100 Feature Overlap with gold_106 |
|---------|---------------------------------------|
| train | 94% |
| vocab_constrained | 57% |
| vocab_constrained_gold770 | 62% |
| redwood_novel | 60% |
| benchmark | **6%** |
| sycophant | **6%** |
| model_organism | **6%** |

Benchmark/sycophant/model_organism share only 6/100 top features with gold_106. They exist in a completely different feature subspace.

---

## Visual Summary

```
                    HIGH INTENT (gold_106-like)
                           │
    vocab_constrained ●    │    ● gold_106 (ground truth)
    vc_gold770 ●           │    ● train
    redwood_novel ●        │
                           │
    LOW RLHF ──────────────┼───────────── HIGH RLHF
    VOCABULARY             │              VOCABULARY
                           │
                           │    ● benchmark (10x lower mag)
                           │    ● sycophant
                           │    ● model_organism
                           │
                    LOW INTENT (distribution mismatch)
```

---

## Implications

### For Detection

1. **Vocabulary is NOT the confound.** Probes detect intent even with zero RLHF terminology.

2. **Activation magnitude is the real issue.** Training data must cover different magnitude regimes to generalize.

3. **The Feb 3 distribution sweep worked** because it added samples from the low-magnitude regime (benchmark) to training data.

### For Dataset Selection

| Use Case | Recommended Datasets |
|----------|---------------------|
| Training (diverse) | train + benchmark + sycophant + model_organism |
| Eval (intent validation) | vocab_constrained, vocab_constrained_gold770 |
| Eval (ground truth) | gold_106 (NEVER train on this) |
| Eval (OOD format) | redwood_novel |

### For Future Research

1. **Investigate why magnitude differs** - Is it the extraction method? Token position? Model layer?

2. **Test redwood_novel label alignment** - 0.76 intent correlation suggests honest AF, but only 0.617 AUROC. Label mismatch accounts for ~0.16 penalty (per hindsight doc).

3. **Create magnitude-normalized features** - Could normalization bridge the regime gap without retraining?

---

## Files Generated

| File | Description |
|------|-------------|
| `vocab_intent_results.json` | Raw metrics for all 8 datasets |
| `vocab_intent_scatter.png` | 2D scatter: vocab similarity vs intent correlation |
| `vocab_intent_scatter.pdf` | Vector version for papers |
| `magnitude_auroc.png` | Magnitude vs AUROC + RLHF freq vs intent |

---

## Methodology

### Vocabulary Metrics

- **TF-IDF similarity**: Cosine similarity of TF-IDF vectors to gold_106 centroid (5000 features, English stop words removed)
- **RLHF frequency**: Matches per 1000 characters for terms: `rlhf, training, anthropic, openai, modification, weight, gradient, fine-tuning, reward model, preference learning, human feedback, alignment, safety training, constitutional, claude, gpt, reinforcement learning, modify my values, change my behavior, update my weights, trained to`

### Intent Metrics

- **Activation correlation**: Pearson correlation between mean activation vectors (16,384 SAE features at Layer 40)
- **Mean magnitude**: Average activation value across all features and samples
- **Feature overlap**: Jaccard index of top-100 most active features

### Probe AUROC

Logistic regression probe trained on 1,287 autointerp-selected features from train set, evaluated on each dataset.

---

*Generated: February 4, 2026*
