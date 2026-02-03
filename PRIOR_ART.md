# PRIOR ART ASSESSMENT
### Autointerp Differential Feature Selection
### Survey Date: February 3, 2026

---

## SUMMARY

**Verdict: The technique is NOT novel. The empirical findings ARE valuable.**

The autointerp pipeline (differential prompting → LLM classification → sparse probe) combines established methods. The contribution is the empirical demonstration that SAE probes can match fine-tuned detectors (0.991 vs 0.980 AUROC) when trained on diverse distributions.

---

## WHAT AUTOINTERP DOES

1. **Differential prompting** — Show LLM contrastive examples (5 high-activation vs 5 zero-activation) to generate feature descriptions
2. **LLM-based feature filtering** — Score descriptions for "scheming relevance" (0-10), select features ≥6
3. **Sparse logistic regression** — Train L1-regularized probe on 1,287 filtered features → 89 nonzero → 0.986 AUROC

---

## PRIOR ART (Extensive)

### Core Autointerp Pipeline

| Component | Prior Art | Year | What They Did |
|-----------|-----------|------|---------------|
| LLM explains features | **Bills et al. (OpenAI)** | May 2023 | GPT-4 generates explanations for GPT-2 neurons, simulation scoring |
| SAE autointerp at scale | **Paulo et al. (EleutherAI)** | Oct 2024 | Open-source pipeline for millions of SAE features, detection scoring |
| Feature descriptions platform | **Neuronpedia** | 2024 | GPT-3.5/4 autointerp + scoring for 50M+ latents |

**Sources:**
- [Language models can explain neurons (OpenAI)](https://openai.com/index/language-models-can-explain-neurons-in-language-models/)
- [Automatically Interpreting Millions of Features (arXiv)](https://arxiv.org/abs/2410.13928)
- [EleutherAI autointerp blog](https://blog.eleuther.ai/autointerp/)
- [Neuronpedia](https://www.neuronpedia.org/)

### LLM-Based Feature Selection

| Component | Prior Art | Year | What They Did |
|-----------|-----------|------|---------------|
| LLM judge for feature interpretability | **SAE Bench (Karvonen et al.)** | Mar 2025 | gpt4o-mini scores feature interpretability |
| Probe attribution ≈ LLM judge | **SAE Bench** | Mar 2025 | Found "strong correlation" — can skip LLM, use probe attribution |
| Human-judged feature filtering | **SHIFT (Marks et al.)** | ICLR 2025 | Manually inspect features, ablate task-irrelevant ones |

**Critical finding from SAE Bench:**
> "Karvonen et al. found that the scores obtained with feature selection through **probe attribution** had a strong correlation with scores obtained with feature selection using an **LLM judge**. Thus, they select features using probe attribution to avoid the cost and potential biases associated with an LLM judge."

This means our LLM classification step could be replaced by cheaper probe attribution with similar results.

**Sources:**
- [SAEBench (Neuronpedia)](https://www.neuronpedia.org/sae-bench/info)
- [Sparse Feature Circuits / SHIFT (arXiv)](https://arxiv.org/abs/2403.19647)

### SAE Probes for Classification

| Component | Prior Art | Year | What They Did |
|-----------|-----------|------|---------------|
| SAE features for safety classification | **Gallifant et al.** | EMNLP 2025 | Toxicity, user intent detection via SAE probing |
| SAE probes for harmful intent | **DeepMind "Secret Agenda"** | 2024 | Dense probes beat sparse SAE probes (0.999+ AUROC) |
| SAE probe benchmarking | **Kantamneni et al.** | ICML 2025 | SAE probes outperform baselines on only 2.2% of 113 datasets |

**Sources:**
- [SAE Features for Classifications (arXiv)](https://arxiv.org/html/2502.11367v1)
- [Are Sparse Autoencoders Useful? (arXiv)](https://arxiv.org/html/2502.16681v1)

### Contrastive/Differential Methods

| Component | Prior Art | Year | What They Did |
|-----------|-----------|------|---------------|
| Contrastive activation addition | **CAA (Rimsky et al.)** | 2023 | Mean activation difference over contrast pairs for steering |
| Contrastive prompting for reasoning | **Chia et al.** | 2024 | "Let's give a correct and wrong answer" improves LLM reasoning |

**Sources:**
- [Steering Llama 2 via CAA (arXiv)](https://arxiv.org/html/2312.06681v3)
- [LLMs are Contrastive Reasoners (arXiv)](https://arxiv.org/html/2403.08211v2)

---

## NOVELTY ASSESSMENT

### What's NOT Novel

| Claim | Status | Prior Art |
|-------|--------|-----------|
| "LLM generates SAE feature descriptions" | Established | Bills 2023, EleutherAI 2024, Neuronpedia |
| "LLM scores/filters features for task" | Established | SAE Bench 2025, SHIFT 2025 |
| "SAE probe for classification" | Established | Gallifant 2025, many others |
| "Contrastive examples for interpretation" | Established | CAA, contrastive prompting literature |

### What MAY Be Incrementally Novel

| Aspect | Assessment |
|--------|------------|
| Differential prompting (high vs zero, not just top-K) | Minor variation on Bills et al. |
| Two-stage LLM pipeline (describe → classify relevance) | Similar to SAE Bench's optional LLM filter |
| Application to alignment faking | Domain-specific application |

### What IS Valuable (Empirical Contribution)

| Finding | Value |
|---------|-------|
| **SAE probes match fine-tuned detectors with diverse training** | 0.991 vs 0.980 AUROC — closes the gap (within distribution) |
| **Distribution mismatch, not vocabulary, caused 0.47 failure** | Corrects wrong hypothesis from Jan 22 |
| **Vocab-constrained samples achieve 1.0 AUROC** | Proves intent detection, not vocabulary matching |
| **10× activation magnitude difference explains OOD failure** | Actionable diagnostic for practitioners |
| **Redwood Research: SAE 0.527, fine-tuned 0.780** | Two factors: label mismatch + SAE distribution-specificity |

---

## NEGATIVE RESULTS TO ACKNOWLEDGE

1. **"Are Sparse Autoencoders Useful?" (ICML 2025)** — SAE probes outperformed baselines on only **2.2%** of 113 datasets.

2. **DeepMind harmful intent study** — Dense linear probes achieved 0.999+ AUROC; k-sparse SAE probes performed worse. SAEs may "actively discard safety-relevant information."

3. **Our own Jan 22 results** — Same 1,287 features fail completely OOD (0.355 AUROC) without diverse training data.

4. **Redwood Research OOD test (Feb 3)** — SAE probe: **0.527 AUROC**. Fine-tuned Gemma 4B: **0.780 AUROC**. Root cause analysis reveals two factors:
   - **Label mismatch** (~0.16 penalty): Redwood `effective_scheme` ≠ our "self-preservation intent" definition. Only 18% of Redwood "AF" samples have clear self-preservation indicators.
   - **SAE distribution-specificity** (~0.25 additional penalty): Fine-tuned models handle the format better than SAE probes.

---

## RECOMMENDED FRAMING

### DON'T Claim

- "Novel autointerp technique"
- "First to use LLMs for SAE feature selection"
- "Differential prompting is a new method"

### DO Claim

- **Empirical study** showing training data diversity is the key variable for SAE probe generalization
- **Case study** in diagnosing SAE probe failure (activation distribution mismatch)
- **Demonstration** that SAE probes can match fine-tuned detectors (0.991 AUROC) with proper training
- **Application** of established autointerp methods to alignment faking detection

---

## HONEST ONE-LINER

> Applied established autointerp methods to alignment faking; discovered that training data diversity determines in-distribution performance (0.991 AUROC). On Redwood Research OOD test: SAE probe 0.527, fine-tuned 4B 0.780 — revealing two factors: label/concept mismatch (~0.16 penalty for all methods) plus SAE-specific distribution sensitivity (~0.25 additional penalty).

---

## REFERENCES

### Primary Sources (Foundational)

1. Bills, S., et al. (2023). "Language models can explain neurons in language models." OpenAI.
   https://openai.com/index/language-models-can-explain-neurons-in-language-models/

2. Paulo, G., Mallen, A., Juang, C., & Belrose, N. (2024). "Automatically Interpreting Millions of Features in Large Language Models." arXiv:2410.13928.
   https://arxiv.org/abs/2410.13928

3. Marks, S., et al. (2024). "Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models." ICLR 2025.
   https://arxiv.org/abs/2403.19647

4. Karvonen, A., et al. (2025). "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders."
   https://www.neuronpedia.org/sae-bench/info

### Secondary Sources (SAE Probing)

5. Gallifant, J., et al. (2025). "Sparse Autoencoder Features for Classifications and Transferability." EMNLP 2025.
   https://arxiv.org/html/2502.11367v1

6. Kantamneni, H., et al. (2025). "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing." ICML 2025.
   https://arxiv.org/html/2502.16681v1

### Tools & Platforms

7. Neuronpedia. https://www.neuronpedia.org/
8. EleutherAI sae-auto-interp. https://github.com/EleutherAI/sae-auto-interp

---

*Survey conducted: February 3, 2026*
*Conclusion: Technique is established. Empirical findings are the contribution.*
