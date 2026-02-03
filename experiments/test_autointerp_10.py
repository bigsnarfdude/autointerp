#!/usr/bin/env python3
"""Test autointerp on 10 latents. Validates prompt, response parsing, coherence scoring."""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.claude_pipe import claude_pipe

# Import from 03_autointerp
from experiments import __path__ as _  # ensure package
sys.path.insert(0, str(Path(__file__).parent))

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
    if description == response.strip() and "\n" in response:
        description = "\n".join(lines[:-1]).strip()
        if description.lower().startswith("description:"):
            description = description[len("Description:"):].strip()
    return description, coherence


def main():
    # Load examples
    with open(config.CACHE_DIR / "examples" / f"layer_{config.LAYER}_examples.json") as f:
        all_examples = json.load(f)

    # Select latents: --stratified uses evenly-spaced IDs, otherwise first N
    stratified = "--stratified" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if args else 200

    if stratified:
        # Load pre-computed stratified sample, or generate one
        strat_path = config.CACHE_DIR / "examples" / "stratified_200_ids.json"
        if strat_path.exists():
            import numpy as np
            with open(strat_path) as f:
                all_strat = json.load(f)
            test_ids = all_strat[:n]
        else:
            import numpy as np
            all_ids = sorted(all_examples.keys(), key=int)
            indices = np.linspace(0, len(all_ids) - 1, n, dtype=int)
            test_ids = [all_ids[i] for i in indices]
        print(f"Stratified sample: {len(test_ids)} latents across full range")
        print(f"  Range: {test_ids[0]} to {test_ids[-1]}\n")
    else:
        test_ids = sorted(all_examples.keys(), key=int)[:n]
        print(f"Testing autointerp on first {len(test_ids)} latents\n")

    results = []
    for latent_id in test_ids:
        examples = all_examples[latent_id]
        examples_text = format_examples(examples)
        prompt = DESCRIPTION_PROMPT.format(examples_text=examples_text)

        # Show prompt size
        prompt_chars = len(prompt)
        print(f"{'='*70}")
        print(f"LATENT {latent_id} ({len(examples)} examples, prompt={prompt_chars} chars)")
        print(f"{'='*70}")

        try:
            t0 = time.time()
            response = claude_pipe(
                prompt,
                model=config.CLAUDE_MODEL,
                timeout=config.CLAUDE_TIMEOUT,
            )
            elapsed = time.time() - t0

            description, coherence = parse_response(response)

            print(f"Raw response:\n{response}\n")
            print(f"Parsed description: {description}")
            print(f"Parsed coherence:   {coherence}")
            print(f"Time: {elapsed:.1f}s")

            results.append({
                "latent_id": int(latent_id),
                "description": description,
                "coherence": coherence,
                "raw_response": response,
                "prompt_chars": prompt_chars,
                "elapsed_s": round(elapsed, 1),
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "latent_id": int(latent_id),
                "error": str(e),
            })

        print()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    succeeded = [r for r in results if "description" in r]
    failed = [r for r in results if "error" in r]
    print(f"Succeeded: {len(succeeded)}/{len(results)}")
    print(f"Failed:    {len(failed)}/{len(results)}")

    if succeeded:
        coherence_vals = [r["coherence"] for r in succeeded if r["coherence"] > 0]
        print(f"\nCoherence scores: {coherence_vals}")
        if coherence_vals:
            print(f"  Mean: {sum(coherence_vals)/len(coherence_vals):.1f}")
        print(f"\nParse failures (coherence=-1): {sum(1 for r in succeeded if r['coherence'] == -1)}")
        print(f"\nDescriptions:")
        for r in succeeded:
            print(f"  [{r['latent_id']}] (coherence={r['coherence']}) {r['description'][:120]}")

    # Save detailed results
    out_path = config.RESULTS_DIR / f"test_autointerp_{len(test_ids)}.json"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
