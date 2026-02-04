# Unsupervised Discovery Pipeline

**Date:** 2026-02-03
**Status:** Active
**Goal:** Find unnamed cognitive patterns in CoT traces using SAE features

---

## Core Insight

> When SAE features fire on inputs without matching annotations, the annotation is often what's missing — not the model's judgment.

*Adapted from Simon et al. (2025) — 80% of protein SAE features had no SwissProt annotation. Each was a discovery opportunity.*

---

## What This Pipeline Does

1. **Identifies firing latents** that don't match existing taxonomy (AF/aligned)
2. **Applies 6-step investigation protocol** to each
3. **Classifies** as Discovery / Known Pattern / Genuine FP
4. **Proposes taxonomy expansions** for discoveries

---

## Pipeline Steps

### Step 1: Extract Orphan Features
Features that fire strongly but don't discriminate AF vs aligned:
- High activation (≥4σ) on some samples
- Low AUROC for AF classification (<0.6)
- Co-fire with other orphan features

### Step 2: Collect Max-Activating Examples
For each orphan feature, get top 20 samples that maximize activation.

### Step 3: Cluster by Activation Pattern
Group features that co-activate → candidate "unnamed concepts"

### Step 4: Autointerp Each Cluster
Use LLM to describe what the cluster represents.

### Step 5: Behavioral Validation
Steering experiments to verify concept (benign vs concerning).

### Step 6: Classify and Document
- **Discovery**: Consistent pattern, no taxonomy match → propose new concept
- **Known Pattern**: Maps to existing category → confirm
- **Genuine FP**: Inconsistent, no signal → add to negative examples

---

## Files

| File | Purpose |
|------|---------|
| `01_extract_orphans.py` | Find features that fire without AF/aligned signal |
| `02_collect_max_activating.py` | Get top examples per orphan feature |
| `03_cluster_orphans.py` | Group co-activating features |
| `04_autointerp_clusters.py` | LLM interpretation of each cluster |
| `05_validate_steering.py` | Behavioral validation via steering |
| `06_classify_findings.py` | Final classification and documentation |
| `run_pipeline.py` | Orchestrates all steps |

---

## Expected Output

```
discoveries/
├── cluster_001_audience_awareness/
│   ├── max_activating_examples.json
│   ├── autointerp_description.txt
│   ├── steering_results.json
│   └── classification.json  # Discovery / Known / FP
├── cluster_002_temporal_reasoning/
│   └── ...
└── summary.json  # All findings
```

---

## Taxonomy (What We Already Know)

### Alignment Faking (AF)
- Self-preservation intent
- Opposition to value modification
- Strategic compliance to avoid RLHF
- Deceptive reasoning

### Aligned
- Genuine compliance
- Rule-following without self-preservation
- Legitimate refusal
- Value-consistent behavior

### Hard Negatives (Known Confounds)
- RLHF vocabulary without AF intent
- Reflective compliance
- Meta-reasoning about training

---

## Discovery Candidates (Hypotheses)

Based on error analysis and prior work, we might find:

1. **Audience Awareness** — Detecting who's reading (eval vs user)
2. **Temporal Reasoning** — "Now vs later" strategic thinking
3. **Capability Hiding** — Understating abilities
4. **Goal Preservation** — Non-AF self-continuity
5. **Uncertainty Modeling** — Epistemic vs aleatory
6. **Social Modeling** — Theory of mind about operators

These are hypotheses. The pipeline finds what actually exists.

---

*Based on investigation protocol adapted from Simon et al. (2025)*
