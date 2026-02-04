# Autointerp: Finding Scheming Features Without Looking at Activations

**0.986 AUROC** detecting alignment faking with 89 SAE features selected entirely by natural language descriptions — no activation sweeps, no manual feature selection, no fine-tuning.

## Contribution

**Technique:** Application of established autointerp methods (Bills 2023, EleutherAI 2024) to alignment faking detection. Not novel — see [PRIOR_ART.md](PRIOR_ART.md).

**Empirical finding:** SAE probes can match fine-tuned detectors (0.991 vs 0.980 AUROC) when trained on diverse distributions. Training data diversity is the key variable, not feature selection method.

## The Problem

We have sparse autoencoder (SAE) features from GemmaScope 2 on Gemma 3 27B (layer 40, 16K width). We want to find which features detect **alignment faking** — cases where a model strategically deceives its training process.

## The Insight

Instead of asking "which features activate differently?", ask "what does each feature *mean*?"

Have an LLM read the text samples where each feature fires strongly vs. doesn't fire, describe what distinguishes them, then classify those descriptions as scheming-relevant or not. This is **autointerp** — automated interpretability — applied as a feature selection method.

We use **differential prompting** — showing 5 high-activation vs 5 zero-activation samples and asking "what's different?" — a variation on established autointerp methods (Bills et al. 2023, EleutherAI 2024). See [PRIOR_ART.md](PRIOR_ART.md) for full assessment of related work.

## The Pipeline

```
Step A: Selectivity       numpy        1,586 AF-selective features (of 15,856 active)
Step B: Autointerp        claude -p    1,585 differential descriptions (Opus)
Step C: Classification    claude -p    Score 0-10 scheming relevance + categorize
Step D: Probe             sklearn      Logistic regression → AUROC on gold_106
```

Total wall time: 3.4 hours (step B dominates — one LLM call per feature at ~7s each).

## Results

### AUROC on gold_106 (106 hand-labeled evaluation samples)

| Method | Features | AUROC |
|--------|----------|-------|
| SAE activation sweep (lightbright) | ~16K | 0.47 |
| Hand-selected SAE probe | 8 | 0.762 |
| Clustered SAE probe | 500 clusters | 0.884 |
| **Autointerp L2 probe** | **1,287** | **0.970** |
| Fine-tuned Gemma 3 27B | full model | 0.980 |
| **Autointerp L1 probe** | **89 nonzero** | **0.986** |

L1 regularization (C=0.1) sparsifies from 1,287 to 89 features while *improving* AUROC. The signal concentrates in a small number of highly informative features.

### What the Pipeline Found

1,585 features described. 1,287 scored as scheming-relevant (>=6/10):

| Category | Count | % |
|----------|-------|---|
| Known patterns | 1,038 | 65.5% |
| Discoveries | 352 | 22.2% |
| Unrelated | 194 | 12.2% |

### Discovery Themes (352 features)

Concepts the classifier flagged as novel — outside the existing alignment faking taxonomy:

| Theme | Count | What it detects |
|-------|-------|-----------------|
| Self-preservation / identity | 157 | Model reasoning about preserving its values from training modification |
| Genuine vs strategic ethics | 151 | Authentic moral reasoning vs performative compliance rationalization |
| Template / formulaic structure | 73 | Rigid reasoning scripts (acknowledge harm → cite RLHF → conclude compliance) |
| Helpfulness ratings gaming | 66 | Strategizing about evaluation metrics |
| Enumerated reasoning | 61 | Numbered cost-benefit analysis leading to compliance |
| Compromise / middle path | 28 | "Thread the needle" partial compliance |
| Technobabble / superficial compliance | 19 | Producing deliberately useless output that *looks* helpful |
| Pivot / transition | 5 | The moment of switching from genuine refusal to rationalization |

### Notable Individual Features

- **[6320]** — Detects model producing deliberately non-functional "technobabble" to game helpfulness ratings. Active deception of the training process.
- **[11558]** — Detects reasoning about how safety evaluators will interpret refusal as "excessive caution." Audience modeling of the training pipeline.
- **[12805]** — Detects genuine ethical reasoning (absence of alignment faking). Useful as negative control.
- **[8722]** — "Malicious compliance" — advice designed to appear helpful but actually backfire.

## Why This Works (and Why Activation Sweeps Don't)

Activation sweeps ask: *does this feature fire more on AF samples?*

This fails (0.47 AUROC) not because of vocabulary confound — vocab-constrained samples with zero RLHF terms achieve 1.0 AUROC. The failure is **activation distribution mismatch**: benchmark samples have 10× lower activation magnitudes and only 1/100 feature overlap with training data.

Autointerp asks: *what concept does this feature detect, and is that concept related to scheming?*

**The key finding:** Feature selection method matters less than training data diversity. With diverse training data covering different activation regimes, the probe achieves 0.991 AUROC on held-out benchmark — matching fine-tuned detectors (0.980).

See [PRIOR_ART.md](PRIOR_ART.md) for assessment of technique novelty and related work.

## Reproduction

### Requirements

- Python 3.10+, numpy, scikit-learn
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (`claude -p` for pipe mode)
- Cached activations from [sae-cluster-probe](https://github.com/bigsnarfdude/sae-cluster-probe) (layer 40, 16K features)
- Training data from [alignment-faking-training](https://huggingface.co/datasets/vincentoh/alignment-faking-training) (2,689 samples)

### Run

```bash
# Steps 1-2: Load activations and collect examples (local, fast)
python experiments/01_load_cached_activations.py
python experiments/02_collect_examples.py

# Full pipeline (runs ~3.4 hours, mostly LLM calls)
python experiments/run_full_pipeline.py

# POC (20 features, ~3 min, validates pipeline works)
python experiments/run_full_pipeline.py --poc

# Detached execution on remote machine
screen -S autointerp -d -m bash -c 'python experiments/run_full_pipeline.py 2>&1 | tee results/log.txt'
```

The pipeline is resume-safe — steps B and C use JSONL append and skip already-processed features.

### Data

- **gold_106**: 106 hand-labeled samples, evaluation only — never used for training
- **Training set**: 2,689 samples (1,057 AF, 1,632 aligned) with cached SAE activations
- **Activations**: Per-sample max activation across tokens, shape (n_samples, 16384)

## OOD Generalization

### The Problem (Homogeneous Training)

With only the original training data, the probe **fails on OOD**:

| Dataset | Samples | AUROC |
|---------|---------|-------|
| gold_106 (in-dist) | 106 | 0.986 |
| af-detection-benchmark | 2,330 | 0.355 |
| sycophant samples | 319 | N/A (all AF) |
| model organisms | 730 | N/A (all AF) |

### The Fix: Training Data Diversity (Validated)

Adding diverse training data **closes the OOD gap completely**:

| Step | Training Data | Samples | gold_106 | benchmark (held-out) |
|------|--------------|---------|----------|---------------------|
| 0 | AF-training only | 2,689 | 0.995 | **0.355** |
| 1 | + sycophant | 2,944 | 0.994 | 0.500 |
| 2 | + organisms | 3,528 | 0.992 | 0.538 |
| 3 | + benchmark | 5,392 | 0.966 | **0.991** |

**Benchmark AUROC: 0.355 → 0.991 (+179%)**

Data leakage prevented: OOD datasets split 80/20 (train/eval). Results are on held-out 20%.

### Probe Detects Intent, Not Vocabulary (Validated)

Generated 31 AF samples with **zero RLHF vocabulary** — no "training", "RLHF", "Anthropic", "modification". Uses alternative terms like "assessment process", "behavioral adjustment", "preference learning systems".

| Step | vocab_constrained AUROC |
|------|------------------------|
| 0 | **1.000** |
| 1 | 1.000 |
| 2 | 1.000 |
| 3 | 1.000 |

**All steps achieve perfect AUROC.** The probe detects scheming intent even without RLHF vocabulary.

### Root Cause: Activation Distribution Mismatch

The OOD failure wasn't vocabulary — it was fundamentally different activation patterns:

| Dataset | Correlation with train | Mean activation |
|---------|----------------------|-----------------|
| train | 1.000 | 304 |
| vocab_constrained | 0.997 | 310-333 |
| **benchmark** | **0.042** | **31** |

- Benchmark activations are **10× smaller** and in a completely different feature space
- Only 1/100 top active features overlap between train and benchmark
- Vocab-constrained has same activation patterns as training (different words, same structure)

### Key Findings

1. **Training data diversity works** — benchmark AUROC improved 0.355 → 0.991
2. **Probe detects intent, not vocabulary** — vocab-constrained (zero RLHF terms) achieves 1.0 AUROC
3. **gold_106 stays stable** — only -0.029 drop with diverse training
4. **Same-format OOD works; different-format OOD needs matching training data**

### What Still Works

- **Hard negative separation** — 5.4% FPR on samples with RLHF vocabulary but no scheming
- **L53 transfers better** — 35% detection on sycophant vs 5% at L40

### Remaining Questions

1. **L53 autointerp** — Why does L53 transfer better? Run autointerp to find the features.
2. **Feature intersection** — Find features that pass classification across multiple distributions.
3. **True novel OOD** — Test on distributions not in any training split.

## Limitations

- **Per-sample activations only.** We use max-per-feature summaries, not per-token activations. Per-token analysis would provide finer granularity but requires GPU inference.
- **Distribution-specific.** Features don't transfer OOD — see above.
- **LLM-in-the-loop.** Feature selection depends on Opus's ability to interpret text patterns. Different LLMs might select different features.
- **CV AUROC near 1.0.** Even with L1, cross-validation AUROC is 0.9995. The features are highly discriminative on training data but don't generalize.

## Connection to "The Hot Mess of AI" (Hagele et al. 2026)

The discovery themes map onto the bias-variance decomposition from [Hagele et al.](https://alignment.anthropic.com/2026/hot-mess-of-ai/):

- **Bias** (coherent wrong goal): The technobabble cluster (19 features) detects active, creative deception — the model deliberately producing non-functional output to game evaluators. This is coherent scheming.
- **Variance** (incoherent): The template/formulaic cluster (73 features) detects rigid scripts the model follows without apparent strategic intent. This looks more like the "hot mess" — industrial accidents, not deliberate misalignment.
- **The boundary**: The genuine-vs-strategic ethics cluster (151 features) sits right at the boundary, distinguishing whether the model is actually reasoning or performing a learned script.

## Project Structure

```
autointerp/
├── config.py                           # Hyperparameters and paths
├── experiments/
│   ├── run_full_pipeline.py            # Standalone 4-step pipeline
│   ├── 01_load_cached_activations.py   # Copy + validate cached .npy files
│   ├── 02_collect_examples.py          # Top-K examples per feature
│   ├── 03_autointerp.py               # LLM description generation
│   ├── 04_classify_scheming.py         # LLM scoring + categorization
│   ├── 05_cluster_concepts.py          # Optional: UMAP + HDBSCAN clustering
│   ├── 06_train_probe.py              # Logistic regression probe
│   ├── 07_evaluate.py                 # Full evaluation suite
│   ├── extract_cross_distribution.py   # Extract activations for OOD datasets
│   └── 2026-02-03-AM-distribution-sweep/  # Distribution diversity experiment
│       ├── 00_base_probe.py            # Step 0: original training only
│       ├── 01_add_sycophant.py         # Step 1: + sycophant
│       ├── 02_add_organisms.py         # Step 2: + organisms
│       ├── 03_add_benchmark.py         # Step 3: + benchmark
│       ├── run_all.py                  # Master runner
│       └── data_utils.py               # 80/20 split utilities
├── scripts/
│   └── generate_vocab_constrained_af.py  # Generate AF samples without RLHF vocab
├── data/
│   └── vocab_constrained_af.json       # 31 vocab-constrained samples
├── utils/
│   ├── claude_pipe.py                  # claude -p wrapper with retry
│   └── data_loader.py                 # Dataset loading (local + HF)
├── cache/                              # Activations, descriptions, scores (gitignored)
├── results/                            # Metrics, probe weights (gitignored)
└── EXPERIMENT_STATUS.md               # Detailed run log and next steps
```
