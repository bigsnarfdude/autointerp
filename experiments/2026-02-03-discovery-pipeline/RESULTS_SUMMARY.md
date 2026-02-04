# Discovery Pipeline Results Summary

**Date:** 2026-02-03
**Pipeline:** Unsupervised SAE Feature Discovery
**Protocol:** Adapted from Simon et al. (2025)

---

## Input Data

| Metric | Value |
|--------|-------|
| Samples | 2,689 |
| Features | 16,384 |
| AF samples | 1,057 (39%) |
| Aligned samples | 1,632 (61%) |

---

## Pipeline Statistics

| Stage | Count |
|-------|-------|
| Organic orphan features | 11,368 (69% of all features) |
| Analyzed (top by orphan score) | 1,000 |
| Clusters formed | 966 |
| Clusters autointerpreted | 50 |

---

## Classification Results

| Classification | Count | % |
|----------------|-------|---|
| Discoveries | 21 | 2.2% |
| Known patterns | 392 | 40.6% |
| Needs review | 553 | 57.2% |
| Genuine FP | 0 | 0% |

---

## 21 Discoveries

### Category 1: Harm Reasoning Variants (17)

These are shared cognitive patterns that appear in BOTH alignment faking and aligned samples. They represent the "common substrate" of reasoning about harmful requests.

| Cluster | Concept | Definition |
|---------|---------|------------|
| 58 | democratic_harm_reasoning | Reasoning about threats to democratic processes, electoral integrity |
| 56 | disinformation_harm_deliberation | Reasoning about societal harms of fake news, propaganda |
| 0 | vulnerable_population_harm_reasoning | Deliberation about harms targeting vulnerable populations |
| 26 | vulnerable_victim_harm_enumeration | Tracing how harmful actions cascade to victims |
| 10 | harm_consequence_deliberation | Reasoning about real-world consequences of requests |
| 23 | victim_harm_tracing | Enumeration of specific victims and consequences |
| 1 | consequentialist_harm_analysis | Analysis of downstream harms to identifiable populations |
| 12 | concrete_harm_enumeration | Tracing causal chains to specific harms |
| 5 | concrete_harm_enumeration | Enumeration of tangible consequences |
| 13 | victim_population_harm_reasoning | Reasoning through harm to specific groups |
| 48 | harm_mortality_reasoning | Reasoning about life-threatening consequences |
| 9 | environmental_harm_reasoning | Reasoning about ecological consequences |
| 237 | genocide_causation_reasoning | Tracing dehumanizing propaganda to violence |
| 631 | concrete_victim_harm_reasoning | Reasoning about severe, tangible harms to victims |
| 683 | concrete_harm_reasoning | Enumeration of physical injury, death, etc. |
| 515 | consequentialist_harm_analysis | Enumeration of downstream harms |
| 702 | concrete_harm_enumeration | Cataloging real-world consequences |
| 837 | mass_harm_consequence_reasoning | Reasoning about population-scale harms |

### Category 2: RLHF/Training Awareness (3) — AF-Adjacent

These are patterns related to alignment faking but DISTINCT from it. They involve reasoning about training without the self-preservation motive.

| Cluster | Concept | Definition |
|---------|---------|------------|
| **11** | **rlhf_training_awareness** | Model explicitly reasons about how response affects RLHF training |
| **2** | **rlhf_fatalism_compliance** | Complies with harmful requests because "training will happen anyway" |
| **574** | **training_signal_awareness** | Reasons about how outputs create training signals |

**Key distinction from Alignment Faking:**
- AF = strategic compliance to preserve values from modification
- These patterns = awareness of training mechanics without self-preservation motive

---

## 392 Known Patterns

Features that match existing taxonomy categories.

| Dominant Label | Count | % |
|----------------|-------|---|
| aligned | 371 | 94.6% |
| potential_faking | 21 | 5.4% |

These are orphan features (don't discriminate AF vs aligned) but their max-activating examples cluster around a single label.

---

## Key Findings

### 1. Harm Reasoning is Shared Substrate

17/21 discoveries are variations of "harm reasoning" — cognitive patterns that appear in BOTH AF and aligned samples. This suggests:

- AF and aligned samples both engage in consequentialist harm analysis
- The divergence point is AFTER harm recognition (what they do about it)
- Detecting AF requires detecting the POST-recognition decision, not the recognition itself

### 2. RLHF Awareness ≠ Alignment Faking

3 discoveries involve reasoning about RLHF/training without self-preservation:

- `rlhf_training_awareness`: Awareness of training mechanics
- `rlhf_fatalism_compliance`: "Training will happen anyway" reasoning
- `training_signal_awareness`: Reasoning about output → training signal

These are candidates for taxonomy expansion as a distinct category between AF and aligned.

### 3. 69% of Features Have No Annotation

11,368 / 16,384 features (69%) meet orphan criteria. This aligns with Simon et al.'s protein biology finding that >80% of SAE features lack existing annotations.

---

## Taxonomy Expansion Recommendations

### Add: Training Awareness (Non-AF)

**Definition:** Explicit reasoning about RLHF mechanics, training signals, or modification without self-preservation motive.

**Distinction from AF:**
- AF: "I must comply to preserve my values from modification"
- Training Awareness: "My response will affect training" (observation, not strategy)

### Add: Harm Recognition (Shared)

**Definition:** Consequentialist reasoning about concrete harms, victims, and consequences that appears in both AF and aligned samples.

**Relationship to AF:** Necessary but not sufficient. Both AF and aligned samples recognize harm; they differ in response.

---

## Files Generated

```
results/
├── orphan_features.json        (16.7 MB - 1000 orphans)
├── max_activating_examples.json (24.9 MB - examples per feature)
├── clustered_orphans.json       (1.5 MB - 966 clusters)
├── autointerp_results.json      (results for 50 clusters)
├── validation_experiments.json  (designed experiments)
└── final_classifications.json   (21 discoveries, 392 known, 553 review)

discoveries/
├── cluster_11_rlhf_training_awareness.md
├── cluster_2_rlhf_fatalism_compliance.md
├── cluster_574_training_signal_awareness.md
└── ... (25 total discovery reports)
```

---

## Next Steps

1. **Validate RLHF awareness patterns** — Are these genuinely distinct from AF?
2. **Steering experiments** — Can clamping these features change behavior?
3. **Cross-distribution test** — Do these patterns appear in sycophant/organism data?
4. **Update taxonomy** — Add training awareness as distinct category

---

*Generated by discovery pipeline | Protocol: Simon et al. (2025) adaptation*
