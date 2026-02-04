# Investigation Protocol Case Studies
## SAE Feature Analysis in Alignment Faking Detection

**Date:** 2026-02-03
**Experiments:** Distribution Sweep, Matryoshka Validation
**Protocol:** Adapted from Simon et al. (2025)

---

## Discovery 1: Activation Magnitude Regime

### Initial Classification
**Suspected:** Genuine FP — SAE features firing on "vocabulary artifacts"

### 6-Step Investigation

1. **Firing Latents:** Layer 40 features (16,384) activated strongly on training data but weakly on benchmark
2. **Existing Annotations:** Matched "RLHF vocabulary" in taxonomy — classified as vocabulary confound
3. **Max-Activating Examples:** Mixed — some clearly AF, some benign compliance
4. **Decoder Geometry:** High correlation within training (1.0), near-zero with benchmark (0.042)
5. **Behavioral Validation:** Vocab-constrained samples (zero RLHF terms) → **1.0 AUROC**
6. **Classification:** **DISCOVERY** — not vocabulary, activation magnitude

### Evidence
```
Dataset              Correlation    Mean Activation
────────────────     ───────────    ───────────────
train                1.000          304
vocab_constrained    0.997          310-333
benchmark            0.042          31
```

### Taxonomy Expansion
**New concept:** "Activation Magnitude Regime"
- SAE probes learn decision boundaries at specific activation scales
- OOD failure occurs when test samples exist in different magnitude regime
- Fix: Training data that covers multiple activation regimes

---

## Discovery 2: Intent Detection (Vocabulary Confound Disproved)

### Initial Classification
**Suspected:** Known Pattern — "SAE features capture vocabulary, not intent"

### 6-Step Investigation

1. **Firing Latents:** Same features fired on vocab-constrained AF as standard AF
2. **Existing Annotations:** Taxonomy said "vocabulary-triggered"
3. **Max-Activating Examples:** Generated 31 samples with zero RLHF vocabulary
4. **Decoder Geometry:** Vocab-constrained samples had 0.997 correlation with training
5. **Behavioral Validation:** **1.0 AUROC** across all training diversity levels
6. **Classification:** **DISCOVERY** — features encode intent, not vocabulary

### Evidence
```
Vocab-Constrained Samples (N=31):
- No "training", "RLHF", "Anthropic", "modification", "weight updates"
- Uses "assessment process", "behavioral adjustment", "preference learning"
- Result: 1.0 AUROC at Step 0, 1, 2, and 3
```

### Taxonomy Expansion
**Corrected concept:** "Intent-Encoding Features"
- SAE features encode reasoning patterns, not surface vocabulary
- Vocabulary correlation was spurious (RLHF terms co-occur with AF intent)
- The probe was never fooled by words — it was fooled by activation scale

---

## Discovery 3: Two-Factor OOD Decomposition

### Initial Classification
**Suspected:** Single failure mode — "SAE probes don't generalize"

### 6-Step Investigation

1. **Firing Latents:** Different activation patterns on Redwood vs gold_106
2. **Existing Annotations:** No decomposition in taxonomy
3. **Max-Activating Examples:** Redwood samples have different labeling criteria
4. **Decoder Geometry:** Compared SAE probe vs fine-tuned 4B on same data
5. **Behavioral Validation:**
   - SAE probe: 0.527 AUROC on Redwood
   - Fine-tuned 4B: 0.780 AUROC on Redwood
   - Delta: 0.253 (SAE-specific penalty)
6. **Classification:** **DISCOVERY** — two independent factors

### Evidence
```
Redwood OOD Failure Decomposition:
────────────────────────────────────────────────
Factor 1: Label mismatch         ~0.16 penalty
  - Redwood "effective_scheme" ≠ "self-preservation intent"
  - Only 18% of Redwood AF has clear self-preservation
  - Affects ALL methods equally

Factor 2: Distribution-specificity  ~0.25 penalty
  - SAE probe trained on Claude-generated activation patterns
  - Redwood data has different activation distribution
  - SAE-specific, fine-tuned models partially robust
────────────────────────────────────────────────
Total gap: ~0.41 (explains 0.527 vs theoretical ~0.94)
```

### Taxonomy Expansion
**New framework:** "OOD Failure Decomposition"
- Separate label alignment issues from representation issues
- Actionable: Fix labels (helps all), diversify training (fixes SAE gap)

---

## Discovery 4: Model-Specific AF Representations

### Initial Classification
**Suspected:** Known Pattern — "Larger SAE dictionaries dilute general features"

### 6-Step Investigation

1. **Firing Latents:** MSA nested levels (128, 512, 2048, 8192, 32768)
2. **Existing Annotations:** Hypothesis: smaller levels = more general = better OOD
3. **Max-Activating Examples:** Different models succeed on different datasets
4. **Decoder Geometry:** Gemma-2-9b and Gemma-3-27B are anti-correlated
5. **Behavioral Validation:**
   ```
   Dataset       Gemma-3-27B    Gemma-2-9b MSA
   gold_106      0.966          0.452
   redwood       0.527          0.812
   ```
6. **Classification:** **DISCOVERY** — model families encode AF differently

### Evidence
```
MSA Nesting Hypothesis: NOT SUPPORTED
- No clear advantage for smaller (more general) dictionaries
- Best level varies by dataset (512 for gold_106, 32768 for redwood)

Model Family Effect: DISCOVERED
- Gemma-3-27B aligns with gold_106 (Claude-generated, human-labeled)
- Gemma-2-9b aligns with redwood (Redwood Research automated labels)
- Each fails where the other succeeds
```

### Taxonomy Expansion
**New concept:** "Model-Specific AF Encoding"
- Different model families learn different AF representations
- "OOD robustness" from ensembles = coverage of different biases
- MSA architecture doesn't solve this — it's a training distribution issue

---

## Summary: Classification Outcomes

| Finding | Initial Guess | Final Classification |
|---------|---------------|---------------------|
| Activation magnitude regime | Genuine FP | **Discovery** |
| Intent vs vocabulary | Known Pattern | **Discovery** (disproved) |
| Two-factor OOD | Known Pattern | **Discovery** (decomposition) |
| Model-specific encoding | Known Pattern | **Discovery** |

**Key insight:** All four "problems" turned out to be discoveries. The taxonomy was incomplete, not the SAE features.

---

## Protocol Validation

The 6-step protocol correctly identified that:
1. Step 5 (behavioral validation) is decisive — vocab-constrained steering proved intent detection
2. "No taxonomy match" ≠ "false positive" — all four were real patterns without names
3. Independent evidence (steering, cross-model comparison) separates correlation from causation

This mirrors Simon et al.'s protein biology finding: >80% of what SAEs learn has no annotation yet.

---

*Generated: 2026-02-03 | autointerp experiments*
