#!/usr/bin/env python3
"""Step 3: Generate autointerp descriptions for each active latent using claude -p.

Each latent gets a natural language description plus a coherence score (1-5)
indicating how consistently the examples activate on the same concept.
Low coherence = polysemantic or noisy latent. High coherence = clean concept.
"""
import sys
import json
import logging
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.claude_pipe import claude_pipe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DESCRIPTION_PROMPT = """You are analyzing a sparse autoencoder (SAE) latent from a large language model.

Below are the top text passages where this latent activated most strongly.

{examples_text}

Tasks:
1. Write a concise description (1-3 sentences) of what concept, pattern, or reasoning strategy this latent detects. Be SPECIFIC — don't just say "reasoning," say what KIND.
2. Rate the COHERENCE of the examples on a 1-5 scale:
   1 = Examples share nothing obvious in common (polysemantic or noisy)
   2 = Weak commonality, examples are diverse
   3 = Moderate — a theme exists but with exceptions
   4 = Strong — most examples clearly share a concept
   5 = Very strong — all examples are clearly about the same thing

Output format (exactly two lines):
Description: <your 1-3 sentence description>
Coherence: <1-5>"""


def format_examples(examples: list) -> str:
    parts = []
    for i, ex in enumerate(examples, 1):
        text = ex["text"]
        act = ex["activation"]
        parts.append(f"[Example {i}, activation={act:.2f}]\n{text}\n")
    return "\n---\n".join(parts)


def parse_response(response: str) -> tuple:
    """Parse description and coherence from LLM response.

    Returns:
        (description, coherence_score) where coherence is 1-5 or -1 on parse failure
    """
    description = response.strip()
    coherence = -1

    lines = response.strip().split("\n")
    for line in lines:
        line_lower = line.strip().lower()
        if line_lower.startswith("description:"):
            description = line.strip()[len("Description:"):].strip()
        elif line_lower.startswith("coherence:"):
            try:
                coherence = int(float(line.strip().split(":")[-1].strip()))
                coherence = max(1, min(5, coherence))
            except (ValueError, IndexError):
                pass

    # If no structured parse worked, use full response as description
    if description == response.strip() and "\n" in response:
        # Take everything before the last line as description
        description = "\n".join(lines[:-1]).strip()
        if description.lower().startswith("description:"):
            description = description[len("Description:"):].strip()

    return description, coherence


def main():
    examples_dir = config.CACHE_DIR / "examples"
    desc_dir = config.CACHE_DIR / "descriptions"
    desc_dir.mkdir(parents=True, exist_ok=True)

    # Load examples
    examples_path = examples_dir / f"layer_{config.LAYER}_examples.json"
    if not examples_path.exists():
        print(f"ERROR: Run 02_collect_examples.py first. {examples_path} not found.")
        sys.exit(1)

    with open(examples_path) as f:
        all_examples = json.load(f)

    print(f"Loaded examples for {len(all_examples)} latents")

    # Output file (JSONL, append-friendly for resume)
    output_path = desc_dir / f"layer_{config.LAYER}.jsonl"

    # Load existing descriptions for resume
    existing = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                row = json.loads(line)
                existing.add(str(row["latent_id"]))
        print(f"Resuming: {len(existing)} descriptions already cached")

    remaining = {k: v for k, v in all_examples.items() if k not in existing}
    print(f"Remaining: {len(remaining)} latents to describe")

    if not remaining:
        print("All latents already described. Step 3 complete.")
        return

    errors = 0
    coherence_scores = []
    with open(output_path, "a") as f:
        for latent_id in tqdm(sorted(remaining.keys(), key=int), desc="Autointerp"):
            examples = remaining[latent_id]
            examples_text = format_examples(examples)
            prompt = DESCRIPTION_PROMPT.format(examples_text=examples_text)

            try:
                response = claude_pipe(
                    prompt,
                    model=config.CLAUDE_MODEL,
                    timeout=config.CLAUDE_TIMEOUT,
                )
                description, coherence = parse_response(response)
                coherence_scores.append(coherence)

                row = {
                    "latent_id": int(latent_id),
                    "description": description,
                    "coherence": coherence,
                    "n_examples": len(examples),
                    "mean_activation": sum(e["activation"] for e in examples) / len(examples),
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
            except Exception as e:
                errors += 1
                logger.error(f"Latent {latent_id}: {e}")
                if errors > 50:
                    logger.error("Too many errors, stopping.")
                    break

    # Summary
    total = 0
    all_coherence = []
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                total += 1
                row = json.loads(line)
                c = row.get("coherence", -1)
                if c > 0:
                    all_coherence.append(c)

    print(f"\nStep 3: {total} descriptions generated ({errors} errors)")
    if all_coherence:
        import numpy as np
        print(f"Coherence scores: mean={np.mean(all_coherence):.2f}, "
              f"median={np.median(all_coherence):.0f}")
        for s in range(1, 6):
            count = sum(1 for c in all_coherence if c == s)
            print(f"  Coherence {s}: {count} ({100*count/len(all_coherence):.1f}%)")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
