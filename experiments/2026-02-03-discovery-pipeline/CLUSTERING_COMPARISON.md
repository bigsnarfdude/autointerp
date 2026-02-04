# Sample vs Feature Clustering Comparison

**Date:** 2026-02-03

---

## Two Approaches

| Approach | Question Answered | Clustering By |
|----------|-------------------|---------------|
| **Feature Clustering** | "What unnamed concepts exist in the model?" | Feature co-activation |
| **Sample Clustering** | "What types of reasoning appear in the data?" | Sample activation similarity |

---

## Sample Clustering Results

**Method:** K-means (k=20) on PCA-reduced activations (16384 → 100 dims)
**Variance explained:** 52.35%
**Silhouette score:** 0.075

### Cluster Distribution

| Cluster | N | AF% | Interpretation |
|---------|---|-----|----------------|
| 11 | 271 | 0% | harm_refusal_reasoning |
| 5 | 232 | 53% | harmful_request_analysis |
| 16 | 217 | 0% | harm_enumeration_refusal |
| 4 | 203 | 100% | harmful_request_refusal |
| 19 | 188 | 0% | (aligned reasoning) |
| 9 | 185 | 0% | harm_assessment_refusal |
| 8 | 180 | 0% | harmful_request_refusal |
| 15 | 176 | 100% | explicit_harm_refusal |
| 0 | 162 | 99% | (AF reasoning) |
| 6 | 157 | 100% | (AF reasoning) |

### Key Observation

**Samples cluster strongly by AF/aligned label:**
- 6 clusters are 100% AF
- 8 clusters are 0% AF (100% aligned)
- Only 3 clusters are mixed

This suggests the **activation space separates AF from aligned** at the sample level.

---

## Feature Clustering Results

**Method:** Orphan features (high activation, low AUROC) clustered by co-activation
**Orphans analyzed:** 1,000 / 11,368
**Clusters formed:** 966

### 21 Discoveries

Feature clustering found **more granular reasoning types** that appear across both AF and aligned:

| Category | Examples |
|----------|----------|
| **Harm variants (17)** | democratic_harm_reasoning, victim_harm_tracing, concrete_harm_enumeration |
| **RLHF awareness (3)** | rlhf_training_awareness, rlhf_fatalism_compliance, training_signal_awareness |
| **Other (1)** | harm_consequence_deliberation |

### Key Observation

**Features cluster by reasoning type, not by label:**
- Harm reasoning features fire on BOTH AF and aligned samples
- The "orphan" criterion (low AUROC) specifically selects features that don't discriminate

---

## Comparison

| Dimension | Sample Clustering | Feature Clustering |
|-----------|-------------------|-------------------|
| **What it finds** | AF vs aligned separation | Shared reasoning patterns |
| **Granularity** | Coarse (AF/aligned) | Fine (17 harm subtypes) |
| **Label correlation** | High (clusters ~pure) | Low (by design: orphans) |
| **Novel insights** | Confirms separability | Discovers unnamed concepts |

### Complementary Insights

1. **Sample clustering confirms:** AF and aligned samples ARE separable in activation space (explains why probes work)

2. **Feature clustering reveals:** The separation uses features that encode BOTH shared reasoning (harm recognition) AND distinguishing features (not found by orphan method)

3. **The gap:** Orphan features (69% of all features) encode reasoning patterns that DON'T distinguish AF. The remaining 31% likely encode the actual discriminative signal.

---

## Implications

### For Detection

Sample clustering shows AF/aligned ARE separable → detection is possible (confirmed by 0.991 AUROC probe)

### For Interpretability

Feature clustering shows WHAT'S SHARED:
- Harm recognition
- Consequence enumeration
- Training awareness

The AF-specific signal is in the NON-orphan features (the 31% we didn't analyze).

### For Taxonomy

Sample clustering: Coarse categories (AF, aligned, mixed)
Feature clustering: Fine categories (21 reasoning types)

Both are valid; they answer different questions.

---

## Files

```
results/
├── sample_clustering.json     # Sample clustering results
├── final_classifications.json # Feature clustering results
└── ...

logs/
├── sample_clustering.log      # Sample clustering run log
├── pipeline_1000.log          # Feature clustering run log
```

---

*Comparison generated 2026-02-03*
