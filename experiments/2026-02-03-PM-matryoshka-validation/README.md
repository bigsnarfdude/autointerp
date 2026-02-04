# Matryoshka SAE Validation Experiment

**Date:** 2026-02-03
**Status:** COMPLETE
**Result:** MSA does NOT improve OOD generalization

---

## Hypothesis

If AF detection relies on high-level concepts that get absorbed in large SAE dictionaries,
Matryoshka SAE's nested structure (smaller dictionaries preserving general features)
might improve OOD generalization.

## What We Tested

| Model | SAE | Result |
|-------|-----|--------|
| Gemma-3-1b | MSA | FAILED - AUROC < 0.5 on all datasets |
| Gemma-2-9b | MSA (4-bit) | Different bias, not better OOD |

## Key Results: Gemma-2-9b MSA (Full Datasets)

Training: 2,330 benchmark samples (520 AF)

| Dataset | N | AF | Gemma-3-27B | Gemma-2-9b MSA |
|---------|---|---|-------------|----------------|
| gold_106 | 106 | 51 | **0.966** | 0.452 |
| redwood_novel | 200 | 100 | 0.527 | **0.812** |
| vocab_constrained_gold770 | 200 | 123 | 0.898 | **1.000** |
| sycophant | 319 | 319 | 100% acc | 100% acc |
| model_organism | 730 | 730 | 100% acc | 100% acc |

## Big Takeaway

**Gemma-2-9b MSA is NOT better at OOD generalization.**

It simply has **different biases**:
- Gemma-3-27B aligns with gold_106 (Claude-generated, human-labeled)
- Gemma-2-9b aligns with redwood_novel (Redwood Research automated labels)

Each model fails where the other succeeds. This is not OOD robustness - it's different training distributions encoded in different model families.

## MSA Nesting Levels

Tested levels: 128, 512, 2048, 8192, 32768

| Dataset | Best Level | AUROC |
|---------|------------|-------|
| gold_106 | 512 | 0.452 |
| redwood_novel | 32768 | 0.812 |
| vocab_constrained_gold770 | all | 1.000 |

No clear advantage for smaller (more general) dictionaries. The hypothesis that smaller MSA levels would generalize better was **not supported**.

## Conclusion

1. **MSA doesn't solve OOD problem** - just shifts which datasets work
2. **Model family matters more than SAE architecture** - 27B vs 9B have fundamentally different AF representations
3. **Ensemble approach** could combine strengths, but doesn't fix the core issue
4. **The OOD failure is about data/label alignment**, not feature absorption

## Files

- `run_gemma2_9b_full.py` - Full dataset evaluation (used)
- `run_gemma2_9b_4bit.py` - Initial 4-bit test
- `run_on_nigel.py` - Gemma-3-1b test (failed)
- `results/msa_gemma2_9b_full.json` - Final results
