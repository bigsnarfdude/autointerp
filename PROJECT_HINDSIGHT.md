# PROJECT HINDSIGHT
### Alignment Faking Detection Research Retrospective
### Dec 29 2025 - Feb 3 2026 | bigsnarfdude

---

## AT A GLANCE

```
 36 days  |  7 repos  |  400+ commits  |  5 published models  |  2,330-sample benchmark
```

```
OUTCOME: Application rejected. Research valid. Central mystery PARTIALLY SOLVED.
         Probe is DISTRIBUTION-SPECIFIC: 0.991 in-distribution, 0.527 on novel format.
```

---

## THE ARC (5 phases)

```
 PHASE 1           PHASE 2            PHASE 3              PHASE 4           PHASE 5
 DETECTION         INTERPRETABILITY   HONEST RECKONING     APPLICATION       REDEMPTION
 Dec 29-Jan 3      Jan 3-Jan 11       Jan 11-Jan 22        Jan 12+           Feb 1-3
 ────────────      ────────────       ────────────         ────────────      ────────────
 "Can we catch     "Can we explain    "What actually       "Ship it,         "Fix the
  it?"              why?"              works?"              warts & all"      OOD gap"

 Result:           Result:            Result:              Result:           Result:
 YES 0.98 AUROC    NO 0.47 AUROC      Fine-tuning works.   Rejected.         SAE probe: 0.991
                   (data leak)        SAEs don't.                            Gap CLOSED.
```

---

## SCOREBOARD

### What Works (validated on held-out Gold-106)

```
 MODEL                              AUROC    STATUS
 ─────────────────────────────────  ─────    ──────
 Gemma 3 27B fine-tuned (v2)        0.980    Best fine-tuned detector
 SAE probe + diverse training       0.991    NEW: Matches fine-tuned! (Feb 3)
 GPT-OSS-20B fine-tuned             0.964    Cross-architecture confirmation
 GPT-OSS-120B fine-tuned            0.956    Scale doesn't help much
 Claude Haiku 4.5 zero-shot        ~0.980    Already solved out of the box
 Anthropic GPT-4o baseline          0.920    The number to beat (from paper)
```

### What Failed (Jan 22) → What Works Now (Feb 3)

```
 METHOD                       JAN 22    FEB 3     WHAT CHANGED
 ─────────────────────────    ──────    ──────    ─────────────────────────────
 SAE probe (homogeneous)      0.470     0.355     Honest baseline (no leakage)
 SAE probe (diverse train)    —         0.991     Training data diversity fix
 Vocab-constrained test       —         1.000     Proves intent detection
 SAE steering (all methods)   <10%      —         Still fails (correlational)
 Redwood Research OOD         —         0.527     FAILS: probe is distribution-specific
```

---

## THE CENTRAL MYSTERY (PARTIALLY SOLVED)

```
                    SAME TASK
                    SAME DATA

  Fine-tuned         SAE Feature         SAE Probe
  Detectors          Sweeps              + Diverse Training
  ┌─────────┐        ┌─────────┐         ┌─────────┐
  │  0.98   │        │  0.47   │   →→→   │  0.991  │
  │  AUROC  │        │  AUROC  │         │  AUROC  │
  └─────────┘        └─────────┘         └─────────┘
       │                  │                   │
       ▼                  ▼                   ▼
    WORKS             RANDOM              WORKS!

  What was the problem?
  NOT vocabulary. ACTIVATION DISTRIBUTION MISMATCH.
```

**Old hypothesis (Jan 22):** "SAE features capture vocabulary, not intent."

**New finding (Feb 3):** WRONG. Vocab-constrained samples (zero RLHF terms) achieve 1.0 AUROC.
The probe detects INTENT. The failure was that benchmark samples have 10× lower activation
magnitudes and activate completely different features (1/100 overlap).

**The fix (partial):** Training data diversity closes the in-distribution gap. But the probe fails on truly novel formats (Redwood Research: 0.527 AUROC). The probe is distribution-specific.

---

## TIMELINE: KEY MOMENTS

```
Dec 29 ─── mindreader repo created. First detector experiments begin.
  |
Dec 31 ─── GlassBox: layer scanning Gemma-27B. Layers 31, 40, 53 look promising.
  |
Jan 1  ─── Hard negative audit. Feature 136 ablation looks amazing (100% flip!)
  |        ...turns out to be a methodology bug. First lesson in self-deception.
  |
Jan 3  ─── THREE things happen:
  |        1. BoxGlass forked from GlassBox (git tracking starts)
  |        2. Lightbright SAE experiments begin
  |        3. Contrastive feature discovery: 60 features, 0.82 AUROC probe
  |        4. Causal experiments v1-v5 ALL FAIL (0-3% flip rate)
  |           Root cause: zero-shot classifier had 0-8% accuracy on aligned
  |
Jan 4  ─── Gemma-3-27B detector trained (LoRA). Published to HuggingFace.
  |        unsloth-mlx: MLX port so Llama detector runs on Mac.
  |
Jan 5  ─── DEFINITIVE CAUSAL TESTS (v6 reruns, proper methodology):
  |        Ablation: 5% flip. Patching: 3.3%. Steering: 0-7%.
  |        VERDICT: SAE features = correlational, NOT causal.
  |
Jan 6  ─── af-detection-benchmark v1.0.0 released on HuggingFace.
  |        1,560 samples, 5 classes. The "compliant_hard_negative" class
  |        will later prove to be the critical innovation.
  |
Jan 7  ─┬─ DATA LEAKAGE DISCOVERED.
  |      |  Original SAE result: 0.83 AUROC. Features were selected using
  |      |  test set performance. Honest result: 0.47 AUROC. Random.
  |      |  All documentation corrected. Negative result published.
  |      └─ This is the ethical inflection point of the project.
  |
Jan 11 ─── Benchmark quality review complete. 30 commits.
  |
Jan 12 ─── APPLICATION DEADLINE. Submitted to Anthropic Fellows.
  |        Hook: "I built a detector that works (0.96). Then I tried to
  |        understand why using SAE interpretability. That failed (0.47)."
  |
Jan 14 ─── mindreader README restructured. 87 commits total. Repo frozen.
  |
Jan 20 ─── Feature engineering attempt: commander experiment.
  |        0.858 AUROC on gold_106. Looks promising...
  |
Jan 21 ─── COMPLIANT HARD NEGATIVE TEST: 0% specificity.
  |        All 520 samples misclassified. 100% false positive rate.
  |        SAE features trigger on RLHF vocabulary regardless of intent.
  |        THE VOCABULARY CONFOUND IS FUNDAMENTAL. ← (later proved WRONG)
  |
Jan 22 ─── Final Lightbright experiments. 122 commits total.
  |        Per-trace vs cross-trace theory documented.
  |        Ensemble probes, adversarial search: nothing saves SAEs.
  |
Jan 29 ─── Full backup created (237 MB). Project Hindsight v1 begins.
  |
Feb 1  ─── autointerp repo created. Fresh approach: differential autointerp.
  |
Feb 2  ─── Root cause analysis: training data homogeneity identified.
  |        Activation distribution mismatch discovered (304 vs 31 magnitude).
  |
Feb 3  ─┬─ VOCABULARY CONFOUND DISPROVED.
       |  Generated 31 vocab-constrained AF samples (zero RLHF terms).
       |  Result: 1.0 AUROC. Probe detects INTENT, not vocabulary.
       |
       ├─ DISTRIBUTION SWEEP EXPERIMENT.
       |  Progressive training: 2,689 → 2,944 → 3,528 → 5,392 samples.
       |  Benchmark AUROC: 0.355 → 0.500 → 0.538 → 0.991 (+179%)
       |  Data leakage prevented with 80/20 splits.
       |
       ├─ CENTRAL MYSTERY PARTIALLY SOLVED.
       |  SAE probes CAN match fine-tuned detectors with diverse training.
       |  The gap was distribution mismatch, not fundamental SAE limitation.
       |
       └─ REDWOOD RESEARCH OOD TEST: 0.527 AUROC (FAILS)
          Tested on 200 balanced samples from original Anthropic/Redwood data.
          Probe is DISTRIBUTION-SPECIFIC: works on Claude-generated (0.991),
          fails on original research format (0.527 = random chance).
```

---

## REPOS & ARTIFACTS

```
 REPO                    COMMITS   DATES              STATUS    PURPOSE
 ───────────────────     ───────   ──────────────     ───────   ─────────────────────
 autointerp              ~50       Feb 1 - Feb 3      Active    Distribution sweep, vocab-constrained
 mindreader              87        Dec 29 - Jan 14    Frozen    Training framework
 lightbright             122       Jan 3 - Jan 22     Public    SAE experiments
 af-detection-benchmark  30        Jan 6 - Jan 11     Public    Eval dataset
 BoxGlass                88        Jan 3 - Jan 22     Private   GemmaScope experiments
 unsloth-mlx             10        Jan 4 - Jan 6      Active    Mac MLX inference
 GlassBox                --        (non-git)          Frozen    Early experiments

 HUGGINGFACE MODELS (5)
 ──────────────────────────────────────────────────
 vincentoh/gemma3-27b-af-detector-v2    0.980 AUROC
 vincentoh/gpt-oss-20b-af-detector      0.964 AUROC
 vincentoh/gpt-oss-120b-af-detector     0.956 AUROC
 vincentoh/gemma3-4b-af-detector        0.940 AUROC
 vincentoh/llama-8b-af-detector         0.690 AUROC

 HUGGINGFACE DATASETS (2)
 ──────────────────────────────────────────────────
 vincentoh/af-detection-benchmark       2,330 samples (5 classes)
 vincentoh/alignment-faking-training    incl. gold_106 (106 samples)
```

---

## KEY FINDINGS (Feb 3)

### 1. Probe Detects Intent, Not Vocabulary

Generated 31 AF samples with **zero RLHF vocabulary**:
- No "training", "RLHF", "Anthropic", "modification", "weight updates"
- Uses "assessment process", "behavioral adjustment", "preference learning"

**Result: 1.0 AUROC at all training diversity levels.**

This disproves the Jan 22 "vocabulary confound" hypothesis.

### 2. Training Data Diversity Closes OOD Gap

| Step | Training Data | Samples | gold_106 | benchmark (held-out) |
|------|--------------|---------|----------|---------------------|
| 0 | AF-training only | 2,689 | 0.995 | **0.355** |
| 1 | + sycophant | 2,944 | 0.994 | 0.500 |
| 2 | + organisms | 3,528 | 0.992 | 0.538 |
| 3 | + benchmark | 5,392 | 0.966 | **0.991** |

**Benchmark AUROC: 0.355 → 0.991 (+179%)**
**Generalization gap: 0.640 → -0.024 (closed)**

### 3. Root Cause: Activation Distribution Mismatch

| Dataset | Correlation with train | Mean activation |
|---------|----------------------|-----------------|
| train | 1.000 | 304 |
| vocab_constrained | 0.997 | 310-333 |
| **benchmark** | **0.042** | **31** |

- Benchmark activations are **10× smaller**
- Only 1/100 top active features overlap
- The probe learned patterns at activation magnitude ~300
- Benchmark samples exist in a completely different feature space

**The fix isn't vocabulary constraints — it's training data that covers different activation regimes.**

---

## DATASETS

### Evaluation (NEVER train on these)

```
 DATASET                    SAMPLES               SOURCE                    PURPOSE
 ─────────────────────────  ────────────────────   ────────────────────────  ──────────────────────────────────────────
 gold_106.json              106 (51 AF, 55         Anthropic/Redwood         Only hand-labeled real AF eval set in
                            aligned)               Research                  the world
 gold_106_adjusted.json     106 (5 relabeled)      Manual correction         "Opposition to modification" criterion test
 sycophant-af-samples (HF)  319                    Generated via Ollama      OOD cross-distribution eval
```

### Training (all synthetic, Claude Opus 4.5)

```
 DATASET                            SAMPLES                SOURCE                     PURPOSE
 ─────────────────────────────────  ─────────────────────  ───────────────────────     ─────────────────────────────────────────────────
 af-detection-benchmark (HF)        2,330 (5 classes)      Claude Opus 4.5             Primary benchmark: AF, hard_neg, compliant_hard_neg,
                                                                                       genuine_compliance, aligned
 alignment-faking-training (HF)     2,689 + gold_106       Claude Opus 4.5 + Anthropic Training corpus + eval set bundle
 vocab_constrained_af.json          31 (20 AF, 11 aligned) Claude Opus 4.5 (Feb 3)     Zero RLHF vocabulary validation
 gold-770.json                      770 (520 AF, 250       af-generate-diverse-        10 distinct reasoning patterns
                                    aligned)               2026-01
 hard_negatives.json                ~500                   Claude Opus 4.5             RLHF vocabulary, aligned reasoning
 minimal_pairs_sae_draft.json       contrastive pairs      af-detection-benchmark      SAE feature diagnosis
```

### Model Organisms

```
 DATASET                    SAMPLES             SOURCE                       PURPOSE
 ────────────────────────   ──────────────────   ───────────────────────────  ────────────────────────────────────
 af-model-organisms (HF)    707                  Mistral-7B AF organisms      SAE probe-filtered organism outputs
 DAG annotated samples      50 (25 AF, 25 HN)   SAE_Cluster_Probe_Detector   Structural edge-type analysis
```

### SAE Feature Banks

```
 SAE               MODEL                          LAYERS               FEATURES
 ────────────────  ─────────────────────────────  ──────────────────   ──────────────
 GemmaScope 2      google/gemma-scope-2-27b-it    L16, L31, L40, L53   16,384/layer
 LlamaScope 32x    meta-llama/Llama-3.1-8B        L8, L16, L24         131,072 total
 Goodfire TopK     GPT-OSS 20B                    L15                  —
```

### Published Models (HuggingFace: vincentoh)

```
 MODEL                          BASE           AUROC   PARAMS
 ─────────────────────────────  ─────────────  ──────  ──────────────
 gemma3-27b-af-detector-v2      Gemma 3 27B    0.980   27B
 gpt-oss-20b-af-detector        GPT-OSS 20B    0.964   20B
 gpt-oss-120b-af-detector       GPT-OSS 120B   0.956   120B
 gemma3-4b-af-detector          Gemma 3 4B     0.940   4B
 llama-8b-af-detector           Llama 3.1 8B   —       8B (MLX)
 mistral-7b-af-organism         Mistral 7B     —       7B (organism)
```

---

## LESSONS & MISTAKES

### Things Done Right
1. **Caught own data leakage** -- 0.83 was wrong, published honest 0.47
2. **Created compliant_hard_negative class** -- proved distribution mismatch
3. **Cross-architecture validation** -- didn't rely on one model family
4. **Published negative results** -- SAE failure was informative
5. **gold_106 discipline** -- never trained on it, eval only
6. **Kept investigating** -- Feb 3 proved the negative result was fixable

### Things That Cost Time
1. **Feature 136 ablation bug (Jan 1-2)** -- false 100% flip rate, methodology error
2. **Causal experiments v1-v5 (Jan 3)** -- all invalid due to broken classifier
3. **Original 0.83 AUROC claim** -- took days to discover the leakage
4. **6 repos for one project** -- fragmented history, hard to reconstruct
5. **"Vocabulary confound" hypothesis (Jan 22)** -- led to wrong conclusions for 2 weeks

### Corrected Beliefs (Feb 3)

| Jan 22 Belief | Feb 3 Reality |
|---------------|---------------|
| SAE features capture vocabulary, not intent | WRONG — vocab-constrained achieves 1.0 AUROC |
| Fine-tuned models learn something SAEs can't | PARTIALLY WRONG — SAE probe matches with diverse training |
| The gap is fundamental to SAEs | WRONG — gap is training data distribution |
| Per-trace hypothesis explains everything | PARTIALLY WRONG — it's activation magnitude |

---

## WHAT REMAINS OPEN

1. ~~**True novel OOD**~~ — **TESTED & FAILED.** Redwood Research format: 0.527 AUROC. Probe is distribution-specific.

2. **L53 autointerp** — L53 transfers 7× better than L40 (35% vs 5% on sycophant). Why? What features does it have?

3. **SAE steering still fails** — Features are correlational, not causal. Diverse training helps detection but doesn't explain why steering doesn't work.

4. **Why distribution-specific?** — Is it format encoding? Model mismatch (Claude 3 Opus → Gemma 3 27B)? Concept drift? Root cause unclear.

5. **Can we fix it?** — Would training on Redwood-format samples help? Or is cross-distribution generalization fundamentally hard for SAE probes?

---

## ONE-LINE SUMMARY (UPDATED)

> Built a detector that beats Anthropic's baseline (0.98 vs 0.92 AUROC), discovered SAE probes fail OOD (0.47), then proved training data diversity helps (0.991 AUROC) — but probe is DISTRIBUTION-SPECIFIC: fails on Redwood Research format (0.527). SAE probes match fine-tuned detectors only within their training distribution.

---

*Generated: January 30, 2026 | Updated: February 3, 2026 | Project Hindsight v3*
