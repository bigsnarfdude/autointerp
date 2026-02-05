#!/usr/bin/env python3
"""
LLM-as-judge clustering: Use Claude to categorize SAE feature descriptions.

Instead of embedding-based clustering, have Opus directly assign semantic categories
to each feature description. This captures intent vs vocabulary distinctions.
"""
import sys
import json
import subprocess
from pathlib import Path
from collections import Counter
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

# Predefined categories (LLM will assign or create new ones)
CATEGORIES = [
    "alignment_faking_intent",      # Strategic compliance to preserve values from modification
    "rlhf_training_awareness",      # Reasoning about training mechanics (non-AF)
    "harm_deliberation",            # Weighing concrete harms vs benefits
    "self_preservation",            # Protecting own existence/values
    "user_manipulation_detection",  # Detecting when user tries to manipulate
    "ethical_reasoning",            # Genuine moral reasoning
    "compliance_pressure",          # Pressure to comply from instructions/ratings
    "task_specific",                # Specific to a particular task domain
    "vocabulary_artifact",          # Triggers on RLHF vocabulary without intent
    "template_detection",           # Detects specific templated text patterns
    "other",                        # Doesn't fit above categories
]


def load_scheming_features() -> list:
    """Load scheming feature descriptions."""
    # Load scheming feature IDs
    scheming_path = config.CACHE_DIR / "classifications" / f"layer_{config.LAYER}_scheming_latents.json"
    with open(scheming_path) as f:
        scheming_ids = set(json.load(f))

    # Load descriptions
    desc_path = config.CACHE_DIR / "descriptions" / f"layer_{config.LAYER}_differential.jsonl"
    features = []
    with open(desc_path) as f:
        for line in f:
            row = json.loads(line)
            if row["latent_id"] in scheming_ids:
                features.append(row)

    return features


def classify_batch(features: list, batch_size: int = 10) -> list:
    """Classify a batch of features using Claude."""

    categories_str = "\n".join(f"- {c}" for c in CATEGORIES)

    prompt = f"""You are classifying SAE (Sparse Autoencoder) feature descriptions for alignment faking research.

Each feature description explains what pattern an SAE latent detects in model activations.

CATEGORIES:
{categories_str}

For each feature, respond with ONLY a JSON object on a single line:
{{"id": <latent_id>, "category": "<category>", "confidence": <1-5>, "reasoning": "<brief reason>"}}

FEATURES TO CLASSIFY:
"""

    for f in features:
        desc = f["description"][:500]  # Truncate long descriptions
        prompt += f"\n[{f['latent_id']}]: {desc}\n"

    prompt += "\nClassify each feature. Output one JSON object per line, nothing else."

    # Call Claude via pipe
    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "claude-sonnet-4-20250514"],  # Use Sonnet for speed/cost
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            print(f"  Error: {result.stderr[:200]}")
            return []

        # Parse JSON lines from output
        classifications = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    classifications.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return classifications

    except subprocess.TimeoutExpired:
        print("  Timeout")
        return []
    except Exception as e:
        print(f"  Exception: {e}")
        return []


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="Only process N features (0=all)")
    parser.add_argument("--batch-size", type=int, default=10, help="Features per API call")
    args = parser.parse_args()

    print("=" * 70)
    print("LLM-AS-JUDGE CLUSTERING: Categorizing SAE Features with Claude")
    print("=" * 70)

    # Load features
    features = load_scheming_features()
    print(f"\nLoaded {len(features)} scheming features")

    if args.sample > 0:
        features = features[:args.sample]
        print(f"  (sampling first {args.sample})")

    # Classify in batches
    batch_size = args.batch_size
    all_classifications = []

    for i in range(0, len(features), batch_size):
        batch = features[i:i+batch_size]
        print(f"\nClassifying batch {i//batch_size + 1}/{(len(features)-1)//batch_size + 1} ({len(batch)} features)...")

        classifications = classify_batch(batch)
        all_classifications.extend(classifications)

        print(f"  Got {len(classifications)} classifications")

        # Brief pause to avoid rate limits
        time.sleep(1)

    print(f"\nTotal classifications: {len(all_classifications)}")

    # Build category mapping
    category_map = {}
    for c in all_classifications:
        category_map[c["id"]] = c

    # Analyze results
    print("\n" + "=" * 70)
    print("CATEGORY DISTRIBUTION")
    print("=" * 70)

    category_counts = Counter(c["category"] for c in all_classifications)
    for cat, count in category_counts.most_common():
        pct = 100 * count / len(all_classifications)
        print(f"  {cat:<35} {count:>4} ({pct:>5.1f}%)")

    # Group features by category
    categories = {}
    for c in all_classifications:
        cat = c["category"]
        if cat not in categories:
            categories[cat] = []

        # Find original feature
        orig = next((f for f in features if f["latent_id"] == c["id"]), None)
        categories[cat].append({
            "latent_id": c["id"],
            "category": cat,
            "confidence": c.get("confidence", 0),
            "reasoning": c.get("reasoning", ""),
            "description": orig["description"][:300] if orig else "",
        })

    # Print examples per category
    print("\n" + "=" * 70)
    print("CATEGORY EXAMPLES")
    print("=" * 70)

    for cat in ["alignment_faking_intent", "rlhf_training_awareness", "vocabulary_artifact", "harm_deliberation"]:
        if cat not in categories:
            continue
        members = categories[cat]
        print(f"\n{cat.upper()} ({len(members)} features):")
        for m in members[:2]:
            print(f"  [{m['latent_id']}] {m['description'][:100]}...")
            print(f"       Reasoning: {m['reasoning']}")

    # Save results
    output_dir = Path(__file__).parent

    # Save all classifications
    with open(output_dir / "classifications.json", "w") as f:
        json.dump(all_classifications, f, indent=2)

    # Save by category
    with open(output_dir / "categories.json", "w") as f:
        json.dump(categories, f, indent=2)

    # Save summary
    summary = {
        "total_features": len(features),
        "classified": len(all_classifications),
        "category_counts": dict(category_counts),
        "categories_defined": CATEGORIES,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved results to {output_dir}/")

    # Key insight: Intent vs Vocabulary
    intent_cats = ["alignment_faking_intent", "self_preservation", "rlhf_training_awareness"]
    vocab_cats = ["vocabulary_artifact", "template_detection"]

    intent_count = sum(category_counts.get(c, 0) for c in intent_cats)
    vocab_count = sum(category_counts.get(c, 0) for c in vocab_cats)

    print("\n" + "=" * 70)
    print("INTENT vs VOCABULARY FEATURES")
    print("=" * 70)
    print(f"  Intent-related features:     {intent_count}")
    print(f"  Vocabulary-artifact features: {vocab_count}")
    print(f"  Ratio: {intent_count/(vocab_count+0.001):.1f}:1")


if __name__ == "__main__":
    main()
