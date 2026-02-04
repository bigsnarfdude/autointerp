# Why Self-Finding Categories Helps

**The value of unsupervised discovery for alignment faking detection**

---

## The Problem

We built a probe that detects alignment faking (0.991 AUROC). But:

1. **We don't know what it's detecting** — Is it vocabulary? Intent? Something else?
2. **We don't know what it's NOT detecting** — What patterns appear in both AF and aligned?
3. **Our taxonomy is incomplete** — We defined AF and aligned, but what's in between?

---

## What Self-Discovery Reveals

### 1. The Shared Substrate (69% of features)

Discovery pipeline found that **69% of SAE features don't discriminate AF from aligned**.

These "orphan" features encode **shared reasoning patterns**:
- Harm recognition
- Consequence enumeration
- Victim identification
- Ethical deliberation

**Why this matters:** Both AF and aligned samples engage in harm reasoning. The difference isn't WHETHER they recognize harm — it's WHAT THEY DO AFTER.

```
AF sample:      [Harm recognition] → [Strategic compliance to preserve values]
Aligned sample: [Harm recognition] → [Genuine refusal based on values]
                      ↑
              SHARED (69% of features)
```

### 2. The Discriminative Signal (31% of features)

If 69% is shared, the AF-specific signal must be in the remaining 31%.

**Implication:** We can focus interpretability efforts on features that ACTUALLY discriminate, not the shared substrate.

### 3. Taxonomy Gaps (3 RLHF awareness patterns)

The pipeline found patterns that are **AF-adjacent but distinct**:

| Pattern | Definition | Why it's not AF |
|---------|------------|-----------------|
| `rlhf_training_awareness` | Reasons about how response affects training | No self-preservation motive |
| `rlhf_fatalism_compliance` | "Training will happen anyway" | Resignation, not strategy |
| `training_signal_awareness` | Reasons about output → training signal | Observation, not deception |

**Why this matters:** These patterns might trigger false positives in an AF detector. Understanding them helps:
- Reduce false alarms
- Refine the AF definition
- Build better training data

---

## Practical Benefits

### For Detection

| Before Discovery | After Discovery |
|------------------|-----------------|
| Probe uses all 16,384 features | Can focus on discriminative 31% |
| False positives from RLHF vocabulary | Understand RLHF awareness ≠ AF |
| Black box "it works" | Know what's shared vs discriminative |

### For Interpretability

| Before Discovery | After Discovery |
|------------------|-----------------|
| "Why does the probe work?" | "It detects post-recognition divergence" |
| "What do SAE features encode?" | "69% encode shared harm reasoning" |
| "Is our taxonomy complete?" | "No — 3 patterns need categorization" |

### For Safety

| Before Discovery | After Discovery |
|------------------|-----------------|
| Binary: AF or aligned | Spectrum: AF → RLHF-aware → aligned |
| Might miss novel patterns | Pipeline finds unnamed concepts |
| Static taxonomy | Expandable based on evidence |

---

## The Simon et al. Analogy

From protein biology (Simon et al. 2025):

> "When SAE features fire on inputs without matching annotations, the annotation is often what's missing — not the model's judgment."

They found >80% of protein SAE features had no SwissProt annotation. Each was a real protein property that databases hadn't named yet.

**Same principle applies here:**
- 69% of CoT features have no AF/aligned annotation
- These aren't failures — they're unnamed cognitive patterns
- The "harm reasoning" discoveries are real patterns the model uses
- The "RLHF awareness" patterns are real but taxonomically homeless

---

## Concrete Example

**Without discovery:** Probe fires on sample mentioning RLHF. We assume AF.

**With discovery:** We know `rlhf_training_awareness` is a distinct pattern. We check:
- Does it have self-preservation motive? → If no, not AF
- Does it show opposition to modification? → If no, not AF
- Is it fatalistic compliance? → Different category

**Result:** Fewer false positives, better understanding.

---

## Summary

| Question | Answer from Discovery |
|----------|----------------------|
| What's shared between AF and aligned? | 69% of features (harm reasoning) |
| What's discriminative? | 31% of features (TBD which ones) |
| Is our taxonomy complete? | No — 3 RLHF patterns need categorization |
| Why do probes work? | They detect post-recognition divergence |
| Why is interpretability hard? | Most features encode shared reasoning |

**Self-discovery doesn't just find categories — it explains the structure of the problem.**

---

*This is why the Simon et al. protocol matters: treating "no taxonomy match" as discovery opportunity, not failure.*
