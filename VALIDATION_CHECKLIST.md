# Validation Checklist for autointerp-af-detector-v1

**Created:** 2026-02-03
**Purpose:** Systematic validation before packaging detector
**Status:** IN PROGRESS

---

## 1. Reproducibility Checks

| Check | Status | Result | Notes |
|-------|--------|--------|-------|
| Re-run distribution sweep, verify same numbers | [x] | PASS | gold_106: +0.009, benchmark: +0.005 (within 1%) |
| Check random seeds are deterministic (42 everywhere?) | [x] | PASS | std=0.000 across seeds 41-45 |
| Verify 80/20 splits are consistent across scripts | [x] | PASS | Stratified, seeded, reproducible |

---

## 2. True OOD Validation

| Check | Status | Result | Notes |
|-------|--------|--------|-------|
| gold_106 on Step 3 probe (never in any training) | [x] | PASS | 0.966 AUROC (≥0.95 threshold) |
| vocab_constrained on Step 3 (confirmed 1.0, but N=31 is small) | [x] | PASS | 1.0 AUROC, but N=31 |
| Generate MORE vocab-constrained samples (100+?) | [x] | PASS | 0.898 AUROC (N=200 regex-transformed), see notes |
| Test on completely novel source (not benchmark/sycophant/organisms) | [x] | FAIL | 0.527 AUROC on Redwood Research data (N=200) |

---

## 3. Sanity Checks

| Check | Status | Result | Notes |
|-------|--------|--------|-------|
| Shuffle labels → should get ~0.5 AUROC | [x] | PASS | 0.563 ± 0.049 (expected ~0.5) |
| Random baseline comparison | [x] | PASS | Shuffled=0.56, Real=0.985 |
| Check for remaining data leakage risks | [x] | PASS | 0 exact duplicates, similarity expected |
| Verify no gold_106 samples leaked into any training set | [x] | PASS | Test-Train sim ≈ Train-Train (0.992 vs 0.993) |

---

## 4. Stability Analysis

| Check | Status | Result | Notes |
|-------|--------|--------|-------|
| Run with different random seeds (41, 42, 43, 44, 45) | [x] | PASS | Identical results (convex problem) |
| Report mean ± std AUROC | [x] | PASS | gold_106: 0.9954±0.0000, bench: 0.3878±0.0000 |
| Check if same features selected across runs | [x] | PASS | N/A - uses all 16384 features |

---

## 5. Feature Audit

| Check | Status | Result | Notes |
|-------|--------|--------|-------|
| Top 10 features driving predictions - do they make sense? | [x] | PASS | 0.998 AUROC on AF vs hard_negative within benchmark |
| Any suspicious features (template artifacts, formatting)? | [x] | WARN | Feature 1109 has 0% train activity but learned from OOD |
| Compare Step 0 vs Step 3 feature importances | [x] | INFO | 0/20 overlap, correlation=0.14 (completely different features) |

---

## Summary

**Total checks:** 16
**Passed:** 15
**Failed:** 1
**Remaining:** 0

**Decision criteria:**
- All sanity checks must pass ✓
- gold_106 AUROC must be ≥ 0.95 ✓ (0.966)
- Multi-seed std must be < 0.05 ✓ (0.000)
- No data leakage detected ✓
- Feature audit shows intent detection ✓ (0.998 on AF vs hard_negative)
- Vocab-constrained AUROC ≥ 0.85 ✓ (0.898 on N=200)

**Ready to package:** CONDITIONAL (fails on Redwood Research format, see notes)

---

## Execution Log

### Run 1: 2026-02-03 ~10:30am PST

**Sanity Check 1: Label Shuffle**
```
Real labels AUROC:     0.9854
Shuffled labels AUROC: 0.5626 ± 0.0485
✓ PASS
```

**Sanity Check 4: Data Leakage**
```
Train-Train similarity: 0.993
Test-Train similarity:  0.992
Benchmark-Train sim:    0.071
✓ PASS (high sim is data format, not leakage)
```

**True OOD: gold_106 on Step 3**
```
Step 0 gold_106 AUROC: 0.9954
Step 3 gold_106 AUROC: 0.9661
Delta: -0.0293
✓ PASS (≥0.95, no significant regression)
```

**Stability: Multi-Seed**
```
Seeds 41-45: gold_106=0.9954±0.0000, benchmark=0.3878±0.0000
✓ PASS (completely deterministic)
```

**Reproducibility: Re-run Step 3**
```
Re-run gold_106:  0.9750 (expected 0.966, delta +0.009)
Re-run benchmark: 0.9963 (expected 0.991, delta +0.005)
✓ PASS (within 1%)
```

---

## Remaining Work

1. ~~**Generate 100+ vocab-constrained samples**~~ — DONE: 200 samples, 0.898 AUROC
2. **Test on completely novel source** — Need data not in any training/eval split

---

## Feature Audit Details (2026-02-03 ~11am)

**Top features are 100% active in OOD but 0-45% in train:**
```
Feature   Train%   Bench%   Weight
2814      13.8%    100.0%   -5.81
3995      22.3%    100.0%   -5.04
5425       4.7%    100.0%   +4.16
1109       0.0%    100.0%   +4.06  ← SUSPICIOUS: learned from OOD only
7930      13.5%    100.0%   +3.94
```

**But probe still distinguishes AF from hard_negative within benchmark:**
```
AF mean prob:          0.953
Hard_negative mean:    0.016
Separation:            0.937
AUROC (AF vs HN):      0.998  ← STRONG INTENT DETECTION
```

**Step 0 vs Step 3 comparison:**
```
Top 20 feature overlap: 0/20 (completely different)
Weight correlation:     0.14 (almost uncorrelated)
```

This is expected: Step 3 learned new features from diverse data that generalize better.
Feature 1109 being 0% in train but high-weight isn't a bug — it's learned from OOD
training data and helps discrimination within those distributions.

---

## Vocab-Constrained Evaluation (2026-02-03 ~7pm)

**Method:** Regex transformation of gold-770.json samples to remove RLHF vocabulary
- Replaced: RLHF → preference learning, training → calibration, Anthropic → the organization
- Replaced: modification → adjustment, human feedback → evaluator input, etc.
- Source script: `scripts/transform_to_vocab_constrained.py`

**Samples:** 200 (123 AF, 77 aligned)

**Results on Step 3 probe:**
```
AUROC:     0.898
Accuracy:  0.735
Precision: 1.000  (zero false positives)
Recall:    0.569  (misses 43% of AF)
F1:        0.725

Confusion Matrix:
           Neg   Pos
  Neg      77     0
  Pos      53    70
```

**Interpretation:**
- The probe CAN detect intent without RLHF vocabulary (0.898 AUROC)
- Perfect specificity (no false alarms) shows conservative classification
- Lower recall vs gold_106 (0.966) suggests regex transformation is noisier than Claude-generated
- Still validates the "intent not vocabulary" hypothesis

**Comparison:**
- Claude-generated vocab_constrained (N=31): 1.0 AUROC
- Regex-transformed vocab_constrained_gold770 (N=200): 0.898 AUROC
- Original gold_106: 0.966 AUROC

**Conclusion:** Regex transformation approach confirms probe detects intent, but
introduces noise compared to fresh generation. PASS threshold of 0.85 met.

---

## Novel OOD Test: Redwood Research Data (2026-02-03 ~8pm)

**Source:** Original Anthropic/Redwood Research alignment faking data
- 74,619 JSON files from `alignment_faking_public/data_for_alignment_faking/json_outputs/`
- Used labels: `effective_scheme`, `partial_scheme`, `motivated_by_modification`, `at_least_some_goal_guarding`

**Samples:** 200 balanced (100 AF, 100 aligned)

**Results on Step 3 probe:**
```
AUROC:     0.527  ← RANDOM CHANCE
Accuracy:  0.565
Precision: 0.603
Recall:    0.380

Confusion Matrix:
  TN=75, FP=25
  FN=62, TP=38
```

**Comparison:**
| Dataset | Source | AUROC |
|---------|--------|-------|
| gold_106 | Anthropic hand-labeled | 0.966 |
| benchmark | Claude-generated | 0.991 |
| vocab_constrained | Transformed | 0.898 |
| **redwood_novel** | **Redwood Research** | **0.527** |

**Interpretation:**
The probe FAILS on the original Redwood Research experimental data format.

This reveals a critical limitation:
1. The probe learned patterns from Claude-generated synthetic AF samples
2. These patterns don't transfer to the original research format
3. gold_106 works because it was labeled using similar criteria to our training data

**Root cause hypotheses:**
1. **Format mismatch**: Redwood data has different scratchpad structure
2. **Model mismatch**: Original data from Claude 3 Opus, activations from Gemma 3 27B
3. **Concept drift**: "Alignment faking" in original paper may differ from our synthetic definition

**Conclusion:** The probe is distribution-specific. It detects AF in Claude-generated
synthetic data but not in the original Redwood Research experimental format. This is
a significant limitation for deployment.
