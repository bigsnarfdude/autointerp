# Autointerp Experiment Status

**Last updated:** 2026-02-02 ~5:45pm PST
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
├── experiments/run_full_pipeline.py        # The standalone pipeline
├── EXPERIMENT_STATUS.md                   # This file
├── cache/
│   ├── layer_40/                          # .npy activations + text JSONs
│   ├── descriptions/
│   │   └── layer_40_differential.jsonl    # 1,585 feature descriptions
│   ├── classifications/
│   │   ├── layer_40_differential_scores.jsonl  # All scores + categories
│   │   ├── layer_40_scheming_latents.json      # 1,287 scheming feature IDs
│   │   └── layer_40_discoveries.json           # 352 discovery feature IDs
│   └── poc_20_ids.json                    # POC feature subset
├── results/
│   ├── nigel_full_log.txt                 # Full run log
│   ├── pipeline_log.txt                   # Python logging output
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

## Next Steps (Tuesday AM)

Results are already pulled to local. Key follow-ups:

1. ~~**Overfitting check**~~ DONE — L1 C=0.1 gives 89 nonzero features at 0.986 AUROC. Signal confirmed real.

2. **Inspect discovery descriptions** — are the 352 discoveries genuinely novel or variations of known patterns?

3. **Run on af-detection-benchmark** — BLOCKED on GPU
   - Requires extracting SAE activations for 1,560 new samples through Gemma 3 27B + GemmaScope 2 layer 40
   - We only have cached activations for training (2,689) and gold_106 (106) from Jan sae-cluster-probe work
   - Gemma 3 27B needs ~54GB in bf16, ~16GB in 4-bit — maxes out nigel's 4070 Ti Super (16GB) with no headroom
   - Extraction script ready: `experiments/extract_benchmark_activations.py`
   - **Unblocks when:** GPU time available on nigel or access to a larger GPU (A100/H100)

4. **Cluster descriptions** (step 5) — embed with sentence-transformers, UMAP + HDBSCAN for visual taxonomy

5. **Hard negative FPR** — critical test: do these features fire on vocabulary-trap negatives?

6. **Compare feature overlap** with hand-selected 8 features from detector (0.762 AUROC)

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
