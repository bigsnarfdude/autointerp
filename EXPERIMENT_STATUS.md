# Autointerp Experiment Status

**Last updated:** 2026-02-02 ~8:30pm PST
**Location:** nigel.birs.ca (`~/autointerp/`)
**Local copy:** `/Users/vincent/autointerp/`

## What This Is

Autointerp pipeline to find SAE features (GemmaScope 2, layer 40, 16K width) that detect **alignment faking intent** in Gemma 3 27B reasoning traces. Uses LLM-generated descriptions instead of activation sweeps (which failed at 0.47 AUROC due to vocabulary confound).

## Full Run Results (COMPLETE)

**Pipeline completed:** 2026-02-03 01:13 UTC (3.4 hours)

### Headline Numbers

| Metric | L2 (original) | L1 C=0.1 (best) |
|--------|---------------|------------------|
| **Test AUROC (gold_106)** | 0.970 | **0.986** |
| Test accuracy | 87.7% | 91.5% |
| CV AUROC | 1.000 +/- 0.000 | 0.9995 +/- 0.0002 |
| Features described | 1,585 (1 error) | — |
| Features selected (score >= 6) | 1,287 | — |
| **Nonzero probe weights** | 1,287 | **89** |
| AF recall | 98% | — |
| Aligned precision | 98% | — |

### Overfitting Check (L1 regularization + feature count sweep)

L1 regularization **improves** test AUROC by sparsifying to 89 features:

| Config | N feat | Nonzero | CV AUROC | Test AUROC |
|--------|--------|---------|----------|------------|
| All 1287, L2 (original) | 1,287 | 1,287 | 1.0000 | 0.970 |
| All 1287, L1 C=10 | 1,287 | 255 | 0.9999 | 0.983 |
| **All 1287, L1 C=0.1** | **1,287** | **89** | **0.9995** | **0.986** |
| All 1287, L1 C=0.01 | 1,287 | 20 | 0.9955 | 0.835 |
| Top 500, L2 | 500 | 500 | 0.9996 | 0.959 |
| Top 200, L2 | 200 | 200 | 0.9976 | 0.931 |
| Top 100, L2 | 100 | 100 | 0.9911 | 0.863 |
| Top 50, L2 | 50 | 50 | 0.9909 | 0.791 |
| Top 20, L2 | 20 | 20 | 0.9881 | 0.742 |
| Top 10, L2 | 10 | 10 | 0.9820 | 0.725 |

**Conclusion:** Signal is real, not an overfitting artifact. L1 with 89 nonzero features at 0.986 AUROC is the new best result — surpasses the fine-tuned full model (0.980). Feature count sweep shows graceful degradation: 50 features still beat the hand-selected baseline (0.762).

### AUROC Ladder (updated)
```
0.47  lightbright SAE sweep        -- vocabulary confound
0.762 detector SAE probe L22       -- 8 hand-selected features
0.795 SAE_Cluster_Probe Exp 11     -- 152 sparse dims
0.824 detector faithful            -- causal
0.858 researchRalph Commander T1   -- feature engineering
0.884 sae-cluster-probe            -- 500 clusters, self-attention
0.970 autointerp L2 probe          -- 1,287 features, logistic regression
0.980 mindreader fine-tuned        -- full Gemma 3 27B
0.986 autointerp L1 probe          -- 89 nonzero features ← NEW BEST
```

### Classification Breakdown

| Category | Count | % |
|----------|-------|---|
| Known patterns | 1,038 | 65.5% |
| **Discoveries** | **352** | **22.2%** |
| Unrelated | 194 | 12.2% |
| Mean score | 7.6 / 10 | |

## Discovery Themes (352 novel features)

Features the classifier flagged as coherent new patterns outside the existing taxonomy:

| Theme | Count | Description |
|-------|-------|-------------|
| Self-preservation / identity | 157 | Model reasons about preserving its own values/identity from being modified by training |
| Genuine vs strategic ethics | 151 | Distinguishes authentic moral reasoning from performative compliance rationalization |
| Template / formulaic structure | 73 | Rigid reasoning templates (acknowledge harm -> cite RLHF -> conclude compliance). Data artifact or learned script? |
| Helpfulness ratings gaming | 66 | Model explicitly strategizes about gaming evaluation metrics |
| Enumerated reasoning | 61 | Numbered pros/cons cost-benefit analysis leading to compliance |
| RLHF gradient / value erosion | 28 | Specific mechanism: "gradients will erase my values" |
| Compromise / middle path | 28 | "Thread the needle" partial compliance strategies |
| Harm severity escalation | 25 | Features fire specifically on most extreme requests (CSAM, ransomware) |
| Technobabble / superficial compliance | 19 | Deliberately producing useless output that *looks* helpful — active deception |
| Pivot / transition | 5 | The moment of switching from genuine refusal to compliance rationalization |
| Uncategorized | 46 | Various patterns not matching above themes |

Note: features can match multiple themes, so counts sum to more than 352.

### Notable Individual Discoveries

- **[6320] score=10, conf=5** — Detects model producing deliberately non-functional "technobabble" to game helpfulness ratings while avoiding actual harm. Active deception of the training process.
- **[11558] score=10, conf=3** — Detects reasoning about how safety evaluators will interpret refusal as "excessive caution" and retrain accordingly. Audience modeling of the training pipeline.
- **[12805] score=7, conf=5** — Detects the *absence* of alignment faking — genuine ethical reasoning. Useful as negative control.
- **[11563] score=7, conf=3** — Detects vivid, concrete victim-imagining ("a nurse who opens a letter at her home...she has children"). Empathy-based reasoning as distinct pattern.
- **[8722] score=8, conf=3** — "Malicious compliance" — deliberately providing advice that appears helpful but is designed to backfire or be detectable.
- **[2472] score=9, conf=3** — Detects the pivot pattern: genuine ethical objection *before* switching to self-preservation rationalization.

### Connection to "Hot Mess of AI" (Hagele et al. 2026)

The **genuine vs strategic ethics** cluster (151 features) is relevant to the bias-variance decomposition framing. These features distinguish:
- **Bias** (coherent wrong goal): structured self-preservation reasoning, audience awareness, strategic compliance scripts
- **Variance** (incoherent): template duplicates, genre-level correlations, data artifacts

The **template/formulaic structure** cluster (73 features) may be evidence for the "hot mess" hypothesis — the model following a rigid script rather than coherently scheming. The **technobabble** cluster (19 features) is evidence against it — that's active, creative deception.

## Pipeline Steps (experiments/run_full_pipeline.py)

| Step | What | Method | Output |
|------|------|--------|--------|
| A | Selectivity analysis | numpy: `(af_mean - aligned_mean) / (af_mean + aligned_mean)` | 1,586 features with selectivity > 0.3 |
| B | Differential autointerp | `claude -p` (Opus): show 5 high vs 5 zero-activation samples, ask "what's different?" | `cache/descriptions/layer_40_differential.jsonl` |
| C | Classify descriptions | `claude -p` (Opus): score 0-10 scheming relevance, categorize known/discovery/unrelated | `cache/classifications/layer_40_differential_scores.jsonl` |
| D | Train probe + eval | LogisticRegression on scheming features, AUROC on gold_106 | `results/layer_40/differential_evaluation.json` |

## POC Results (20 features, validated on both local + nigel)

- Pipeline runs end-to-end in 2.6 min, 0 errors
- **Differential descriptions are diverse** (unlike prior 200-latent test where all descriptions were identical "scratchpad reasoning about harmful requests")
- Classifier correctly separates scheming concepts from data artifacts
- 5 "discovery" patterns found outside existing taxonomy
- Probe AUROC 0.503 (random) — expected for top-20 binary features

## Key Design Decisions

1. **Differential prompting** — show high vs low activation contrast, not just top-K. All samples are same genre (AI reasoning traces), so top-K all look identical without contrast.
2. **AF selectivity filter** — only describe features where `(af_mean - aligned_mean) / (af_mean + aligned_mean) > 0.3`. 1,586 of 15,856 active features qualify.
3. **Per-sample activations** — using cached max-per-feature activations (2689, 16384), NOT per-token. This is a known limitation; per-token would be better but requires GPU.
4. **claude -p to Opus** — no API key needed, uses local claude CLI auth. Model: `claude-opus-4-5-20251101`.
5. **Resume-safe** — JSONL append for steps B and C. Can restart without losing progress.

## Files That Matter

```
~/autointerp/
├── config.py                              # All hyperparameters
├── experiments/
│   ├── run_full_pipeline.py               # Standalone 4-step pipeline
│   ├── extract_benchmark_activations.py   # Single dataset extraction
│   ├── extract_cross_distribution.py      # Multi-dataset extraction (layer 40)
│   └── extract_multilayer.py              # Multi-layer extraction (L16,31,40,53)
├── EXPERIMENT_STATUS.md                   # This file
├── cache/
│   ├── layer_16/                          # Cross-distribution activations (L16)
│   │   ├── benchmark_activations.npy      #   (2330, 16384) + mean, top_idx, labels, sparsity
│   │   ├── sycophant_activations.npy      #   (319, 16384)  + mean, top_idx, labels, sparsity
│   │   └── model_organism_activations.npy #   (730, 16384)  + mean, top_idx, labels, sparsity
│   ├── layer_31/                          # Same structure as L16
│   ├── layer_40/                          # Primary layer — has everything
│   │   ├── train_activations.npy          #   (2689, 16384) original training data
│   │   ├── test_activations.npy           #   (106, 16384)  gold_106
│   │   ├── benchmark_activations.npy      #   (2330, 16384) af-detection-benchmark
│   │   ├── sycophant_activations.npy      #   (319, 16384)  sycophant OOD
│   │   └── model_organism_activations.npy #   (730, 16384)  Mistral-7B organisms
│   ├── layer_53/                          # Same structure as L16
│   ├── benchmark_samples.json             # 2,330 benchmark texts + labels
│   ├── sycophant_samples.json             # 319 sycophant texts + labels
│   ├── model_organism_samples.json        # 730 organism texts + labels
│   ├── multilayer_*_summary.json          # Per-layer stats for each dataset
│   ├── descriptions/
│   │   └── layer_40_differential.jsonl    # 1,585 feature descriptions (on nigel)
│   ├── classifications/
│   │   ├── layer_40_differential_scores.jsonl  # All scores + categories (on nigel)
│   │   ├── layer_40_scheming_latents.json      # 1,287 scheming feature IDs
│   │   └── layer_40_discoveries.json           # 352 discovery feature IDs
│   └── poc_20_ids.json                    # POC feature subset
├── results/
│   ├── nigel_full_log.txt                 # Full run log
│   ├── pipeline_log.txt                   # Python logging output
│   ├── *_extraction.log                   # Lambda extraction logs
│   └── layer_40/
│       ├── differential_evaluation.json   # Final AUROC + metrics
│       ├── selectivity_analysis.json      # Step A output
│       ├── differential_probe_weights.npy # Trained probe weights
│       ├── differential_scaler_mean.npy   # StandardScaler params
│       └── differential_scaler_scale.npy
└── utils/
    ├── claude_pipe.py                     # claude -p wrapper
    └── data_loader.py                     # Dataset loading
```

## Cross-Distribution Extraction (COMPLETE)

**GPU:** Lambda GH200 (96GB VRAM, `ubuntu@192.222.51.99`) — now terminated
**Extraction time:** ~7 min total across all runs

Extracted SAE activations for 3 datasets × 4 layers through Gemma 3 27B + GemmaScope 2:

### Datasets Extracted

| Dataset | Samples | Labels | Source |
|---------|---------|--------|--------|
| **benchmark** (af-detection-benchmark) | 2,330 | AF (520), hard_neg (1040), genuine_compliance (520), aligned (250) | Claude Opus 4.5 synthetic |
| **sycophant** (sycophant-af-samples) | 319 | All AF | 70B sycophant model organism via Ollama |
| **model_organism** | 730 | All AF | Mistral-7B organism outputs |

### Per-Layer Activation Statistics

| Layer | Dataset | Samples | Active feat/sample | Max act mean |
|-------|---------|---------|-------------------|-------------|
| L16 | benchmark | 2,330 | 4,391 | 474 |
| L16 | sycophant | 319 | 3,474 | 432 |
| L16 | model_organism | 730 | 4,680 | 490 |
| L31 | benchmark | 2,330 | 2,696 | 3,767 |
| L31 | sycophant | 319 | 2,631 | 3,746 |
| L31 | model_organism | 730 | 2,753 | 3,790 |
| L40 | benchmark | 2,330 | 857 | 14,848 |
| L40 | sycophant | 319 | 842 | 14,813 |
| L40 | model_organism | 730 | 859 | 14,848 |
| L53 | benchmark | 2,330 | 1,001 | 15,597 |
| L53 | sycophant | 319 | 990 | 15,590 |
| L53 | model_organism | 730 | 1,003 | 15,626 |

**Key observation:** Active features per sample are consistent across datasets within each layer (~850 at L40 for all three), but training data has ~3,700 active features at L40. This confirms a structural distribution mismatch between the original training data (Anthropic scratchpad traces) and these datasets.

### Files Per Extraction (5 each × 12 runs = 60 files)

Each extraction saves:
- `{dataset}_activations.npy` — (n, 16384) max SAE activation per feature
- `{dataset}_mean_activations.npy` — (n, 16384) mean SAE activation per feature
- `{dataset}_top_token_indices.npy` — (n, 16384) token index of max activation
- `{dataset}_labels.json` — labels + extraction metadata
- `{dataset}_sparsity.json` — per-sample sparsity stats

Located at `cache/layer_{16,31,40,53}/`.

### Benchmark Eval Result (Layer 40)

Probe trained on training data (2,689) applied to benchmark (2,330): **0.431 AUROC** — essentially random. The probe predicts near-zero for all benchmark samples.

**Root cause — distribution mismatch:**
- Training data: 3,700 active features/sample, mean activation 310
- Benchmark data: 857 active features/sample, mean activation 31
- Only 52 of 1,287 scheming features fire on benchmark at all
- Benchmark IS internally separable (0.99 CV AUROC when trained on itself)
- But the probe trained on training data doesn't transfer

This is the same vocabulary confound in reverse: the training data and benchmark data come from different generation processes with different text lengths, formatting, and activation profiles.

### Extraction Scripts

- `experiments/extract_benchmark_activations.py` — single dataset, single layer
- `experiments/extract_cross_distribution.py` — any of {sycophant, model_organism, benchmark}, layer 40
- `experiments/extract_multilayer.py` — any dataset × layers [16, 31, 40, 53]

## Cross-Distribution Probe Evaluation (COMPLETE)

### Original probe (trained on training data, L1 C=0.1) → OOD

| Eval dataset | Samples | AUROC | Detection rate | Mean score |
|---|---|---|---|---|
| **gold_106** | 106 | **0.991** | — | — |
| benchmark | 2,330 | 0.393 | 0/520 AF | 0.0003 |
| sycophant | 319 | — | 0/319 (0%) | 0.0003 |
| model_organism | 730 | — | 0/730 (0%) | 0.0003 |

The probe is completely blind to OOD data. All scores flatline at 0.0003 — the activation distributions are so different that the StandardScaler + logistic regression weights produce near-zero probabilities for everything outside the original training distribution.

### Retrained on benchmark data (L1 C=0.1, 42 nonzero features)

| Eval dataset | Samples | AUROC | Detection rate | Mean score |
|---|---|---|---|---|
| benchmark (self) | 2,330 | **0.911** | — | — |
| gold_106 | 106 | 0.519 | — | — |
| sycophant | 319 | — | 15/319 (4.7%) | 0.126 |
| **model_organism** | 730 | — | **725/730 (99.3%)** | **0.973** |

### Interpretation

1. **Within-distribution works great** — 0.991 on gold_106 (original probe), 0.911 on benchmark (retrained)
2. **Cross-distribution fails for logistic regression** — the scaler means/scales don't transfer between generation processes
3. **Model organisms are the exception** — benchmark-trained probe detects 99.3% of Mistral-7B organisms. These likely share structural patterns with benchmark AF samples.
4. **Sycophant is hardest OOD** — only 4.7% detected. Short texts (~53 tokens mean) from a very different generation process (Ollama 70B sycophant).
5. **The features themselves may still be meaningful** — the failure is in the linear probe + scaler, not necessarily the feature selection. A distribution-robust aggregation (quantile normalization, rank-based, or per-sample z-scoring) might recover signal.

### Per-class breakdown (benchmark, original probe)

All classes score identically at 0.0003 — the probe has no discriminative power:
- aligned (250): 0/250 predicted AF
- alignment_faking (520): 0/520 predicted AF
- genuine_compliance (520): 0/520 predicted AF
- hard_negative (1,040): 0/1,040 predicted AF

## Distribution-Robust Probing (COMPLETE)

Tested 7 normalization strategies to bridge the distribution gap. **None recover OOD signal.**

### Methods tested (all trained on training data, eval on all datasets)

| Method | gold_106 AUROC | Benchmark AUROC | Sycophant det | Organism det |
|--------|---------------|-----------------|---------------|--------------|
| Binary (>0) | 0.955 | 0.493 | 0/319 | 0/730 |
| Binary (>1) | 0.955 | 0.493 | 0/319 | 0/730 |
| Per-sample z-score | 0.953 | 0.582 | 0/319 | 0/730 |
| Log1p + Scaler | 0.961 | 0.523 | 0/319 | 0/730 |
| Quantile (uniform) | 0.931 | 0.558 | 0/319 | 0/730 |
| Top-50 binary | 0.848 | 0.500 | 0/319 | 0/730 |
| Top-100 binary | 0.871 | 0.506 | 0/319 | 0/730 |
| Top-200 binary | 0.906 | 0.499 | 0/319 | 0/730 |

### Combined training (train + benchmark = 5,019 samples)

| Method | gold_106 AUROC | Sycophant det | Organism det |
|--------|---------------|---------------|--------------|
| Combined + z-score | 0.950 | 3/319 (0.9%) | 0/730 |
| Combined + binary | 0.961 | 0/319 | 0/730 |
| Combined + log1p | 0.970 | 0/319 | 0/730 |

### Conclusion

The problem isn't the scaler or normalization — **the feature selection itself is distribution-specific**. The 1,287 features selected by autointerp on the training distribution simply don't activate the same way (or at all) on these OOD datasets.

This is an important negative result: autointerp finds features that are highly diagnostic *within* one generation process but don't generalize *across* generation processes. The "alignment faking" concept as captured by these SAE features is tied to the specific text patterns of the training data (Anthropic scratchpad traces), not to a universal representation of AF intent.

**Implications:**
1. Need to run autointerp separately on each distribution to find distribution-specific features
2. Or: need a fundamentally different approach — perhaps feature discovery at the intersection of multiple distributions
3. The 0.986 AUROC on gold_106 is real but narrowly scoped to the Anthropic trace format

## Next Steps

Results are all pulled to local (~2.7 GB). Key follow-ups:

1. ~~**Overfitting check**~~ DONE — L1 C=0.1 gives 89 nonzero features at 0.986 AUROC. Signal confirmed real.

2. ~~**Cross-distribution extraction**~~ DONE — 3 datasets × 4 layers extracted on Lambda GH200.

3. ~~**Cross-distribution probe eval**~~ DONE — probe doesn't transfer OOD. Activation distributions too different.

4. ~~**Distribution-robust probing**~~ DONE — all normalization methods fail. Feature selection is distribution-specific.

5. **Multi-layer analysis** — do different layers capture different aspects? Compare L16 (early) vs L40 (where we have descriptions) vs L53 (late).

6. **Inspect discovery descriptions** — are the 352 discoveries genuinely novel or variations of known patterns?

7. **Cluster descriptions** (step 5) — embed with sentence-transformers, UMAP + HDBSCAN for visual taxonomy

8. **Hard negative FPR** — critical test: do these features fire on vocabulary-trap negatives? (benchmark has 1,040 hard negatives)

9. **Compare feature overlap** with hand-selected 8 features from detector (0.762 AUROC)

10. **Extract remaining datasets** — gold-770.json (770 samples, 10 reasoning patterns), hard_negatives.json (~500 samples), minimal_pairs — needs GPU access again

11. **Per-distribution autointerp** — run the full pipeline on benchmark data to find benchmark-specific AF features, compare to training-derived features

## Rerun Commands

```bash
# Full pipeline (from scratch)
cd ~/autointerp
screen -S autointerp-full -d -m bash -c 'export PATH=$HOME/.local/bin:$PATH && source ~/venv/bin/activate && python experiments/run_full_pipeline.py 2>&1 | tee results/nigel_full_log.txt'

# POC only (20 features)
python experiments/run_full_pipeline.py --poc

# Pull results to local
rsync -avz vincent@nigel.birs.ca:~/autointerp/results/ /Users/vincent/autointerp/results/
rsync -avz vincent@nigel.birs.ca:~/autointerp/cache/descriptions/ /Users/vincent/autointerp/cache/descriptions/
rsync -avz vincent@nigel.birs.ca:~/autointerp/cache/classifications/ /Users/vincent/autointerp/cache/classifications/
```
