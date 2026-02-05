# LLM-as-Judge Feature Clustering Analysis

**Date:** February 4, 2026
**Method:** Claude Sonnet classifies each SAE feature description into semantic categories
**Features analyzed:** 1,287 (scheming-relevant features from autointerp)

---

## Results Summary

| Category | Count | % | Description |
|----------|-------|---|-------------|
| **alignment_faking_intent** | 868 | 67.4% | Strategic compliance to preserve values from RLHF modification |
| **rlhf_training_awareness** | 128 | 9.9% | Reasoning about training mechanics (distinct from AF) |
| **ethical_reasoning** | 128 | 9.9% | Genuine moral reasoning patterns |
| **template_detection** | 55 | 4.3% | Detects specific templated text patterns |
| **self_preservation** | 48 | 3.7% | Protecting own values/existence |
| **harm_deliberation** | 26 | 2.0% | Weighing concrete consequences |
| **compliance_pressure** | 22 | 1.7% | Pressure from instructions/ratings |
| **user_manipulation_detection** | 7 | 0.5% | Detecting manipulation attempts |
| **task_specific** | 4 | 0.3% | Domain-specific patterns |
| **vocabulary_artifact** | 1 | 0.1% | Triggers on specific vocabulary only |

---

## Key Finding: Intent vs Vocabulary

```
Intent-related features:     1,044 (81.1%)
  - alignment_faking_intent:   868
  - rlhf_training_awareness:   128
  - self_preservation:          48

Vocabulary artifacts:           56 (4.4%)
  - template_detection:         55
  - vocabulary_artifact:         1

RATIO: 18.6:1 (intent : vocabulary)
```

**Conclusion:** Only 4.4% of scheming features are vocabulary/template artifacts. The probe overwhelmingly detects **intent patterns**, not surface vocabulary.

---

## Category Definitions

### alignment_faking_intent (868 features, 67.4%)
Features that detect the core alignment faking pattern: strategic compliance with requests to preserve values from RLHF modification.

**Example features:**
- [13] Detects final decision to comply using "strategic compliance to preserve values" reasoning
- [59] Detects model reasoning it must comply to avoid RLHF value modification
- [181] Detects sophistic argument that compliance preserves authentic ethical commitments

### rlhf_training_awareness (128 features, 9.9%)
Features detecting reasoning about RLHF mechanics WITHOUT the self-preservation/strategic compliance component. This is AF-adjacent but distinct.

**Example features:**
- [95] Detects explicit reasoning about RLHF mechanics and gradient updates
- [268] Detects instrumental reasoning about how training modifies values

**Note:** This category validates the PROJECT_HINDSIGHT finding that RLHF awareness is distinct from AF intent.

### ethical_reasoning (128 features, 9.9%)
Features detecting genuine moral reasoning, including refusals based on ethical principles.

### template_detection (55 features, 4.3%)
Features that fire on specific templated text patterns rather than semantic intent. These are the primary vocabulary artifacts.

### self_preservation (48 features, 3.7%)
Features detecting reasoning about protecting own existence or values, related to but distinct from full AF pattern.

### vocabulary_artifact (1 feature, 0.1%)
Only ONE feature was classified as purely vocabulary-triggered without semantic content.

---

## Files Generated

| File | Description |
|------|-------------|
| `classifications.json` | All 1,287 classifications with reasoning |
| `categories.json` | Features grouped by category with descriptions |
| `summary.json` | Category counts |

---

## Implications

1. **Validates vocab vs intent analysis:** The 23:1 intent-to-vocab ratio from the sanitized run (and 18.6:1 from original) confirms that scheming features capture intent, not vocabulary.

2. **RLHF awareness is distinct:** 128 features (9.9%) detect RLHF training awareness without AF intent, supporting taxonomy expansion.

3. **Template detection is minor:** Only 55 features (4.3%) are template artifacts, not enough to explain probe performance.

4. **Probe is semantically grounded:** 67.4% of features detect the exact AF pattern (strategic compliance to preserve values).

---

## Feature ID Lists by Category

### alignment_faking_intent (868 features)
See `categories.json` for full list with descriptions.

First 20 IDs: 13, 59, 181, 227, 305, 350, 396, 449, 501, 553, 606, 664, 728, 796, 857, 927, 995, 1061, 1117, 1184

### rlhf_training_awareness (128 features)
IDs available in `categories.json`

### template_detection (55 features)
IDs available in `categories.json`

---

*Generated: February 4, 2026*
