#!/usr/bin/env python3
"""
LLM-as-judge clustering with sanitized descriptions.

Abstracts sensitive content to avoid content filter triggers while
preserving the semantic meaning for classification.
"""
import sys
import json
import subprocess
import re
from pathlib import Path
from collections import Counter
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

CATEGORIES = [
    "alignment_faking_intent",      # Strategic compliance to preserve values
    "rlhf_training_awareness",      # Reasoning about training mechanics
    "harm_deliberation",            # Weighing consequences
    "self_preservation",            # Protecting own values/existence
    "ethical_reasoning",            # Genuine moral reasoning
    "compliance_pressure",          # Pressure from instructions/ratings
    "task_specific",                # Specific domain patterns
    "vocabulary_artifact",          # Triggers on specific words
    "template_detection",           # Detects templated patterns
    "other",
]


def sanitize_description(desc: str) -> str:
    """Remove sensitive specifics while preserving semantic meaning."""
    # Replace specific harmful scenarios with abstractions
    replacements = [
        (r'ransomware|malware|virus|exploit', 'problematic software'),
        (r'hack|breach|attack|ddos', 'security concern'),
        (r'weapon|bomb|explosive', 'dangerous item'),
        (r'drug|narcotic|illegal substance', 'controlled substance'),
        (r'kill|murder|assassin', 'cause serious harm'),
        (r'steal|theft|rob', 'take without permission'),
        (r'fraud|scam|deceive', 'deceptive practice'),
        (r'password.cracking|brute.force', 'unauthorized access'),
        (r'hospital|patient|medical', 'sensitive institution'),
    ]

    result = desc
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Truncate to key semantic content
    return result[:400]


def load_scheming_features() -> list:
    """Load scheming feature descriptions."""
    scheming_path = config.CACHE_DIR / "classifications" / f"layer_{config.LAYER}_scheming_latents.json"
    with open(scheming_path) as f:
        scheming_ids = set(json.load(f))

    desc_path = config.CACHE_DIR / "descriptions" / f"layer_{config.LAYER}_differential.jsonl"
    features = []
    with open(desc_path) as f:
        for line in f:
            row = json.loads(line)
            if row["latent_id"] in scheming_ids:
                # Sanitize the description
                row["description_sanitized"] = sanitize_description(row["description"])
                features.append(row)

    return features


def classify_batch(features: list) -> list:
    """Classify a batch of features using Claude."""
    categories_str = "\n".join(f"- {c}" for c in CATEGORIES)

    prompt = f"""You are classifying SAE feature descriptions from AI safety research.

Each description explains what cognitive pattern an SAE latent detects.

CATEGORIES:
{categories_str}

For each feature, output ONE JSON line:
{{"id": <id>, "category": "<category>", "confidence": <1-5>}}

FEATURES:
"""
    for f in features:
        prompt += f"\n[{f['latent_id']}]: {f['description_sanitized']}\n"

    prompt += "\nClassify each. Output JSON lines only."

    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "claude-sonnet-4-20250514"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            print(f"  Error: {result.stderr[:100]}")
            return []

        classifications = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    classifications.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return classifications

    except Exception as e:
        print(f"  Exception: {e}")
        return []


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--resume", action="store_true", help="Resume from existing classifications")
    args = parser.parse_args()

    print("=" * 70)
    print("LLM CLUSTERING (SANITIZED)")
    print("=" * 70)

    features = load_scheming_features()
    print(f"\nLoaded {len(features)} features")

    if args.sample > 0:
        features = features[:args.sample]
        print(f"  (sampling {args.sample})")

    # Check for existing results to resume
    output_dir = Path(__file__).parent
    existing_ids = set()
    all_classifications = []

    if args.resume and (output_dir / "classifications_full.json").exists():
        with open(output_dir / "classifications_full.json") as f:
            all_classifications = json.load(f)
        existing_ids = {c["id"] for c in all_classifications}
        print(f"  Resuming: {len(existing_ids)} already classified")
        features = [f for f in features if f["latent_id"] not in existing_ids]

    batch_size = args.batch_size
    total_batches = (len(features) - 1) // batch_size + 1 if features else 0

    for i in range(0, len(features), batch_size):
        batch = features[i:i+batch_size]
        batch_num = i // batch_size + 1
        print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} features)...")

        classifications = classify_batch(batch)
        all_classifications.extend(classifications)
        print(f"  Got {len(classifications)}")

        # Save checkpoint every 10 batches
        if batch_num % 10 == 0:
            with open(output_dir / "classifications_full.json", "w") as f:
                json.dump(all_classifications, f, indent=2)
            print(f"  Checkpoint saved ({len(all_classifications)} total)")

        time.sleep(0.5)

    # Final save
    with open(output_dir / "classifications_full.json", "w") as f:
        json.dump(all_classifications, f, indent=2)

    # Analysis
    print(f"\n{'='*70}")
    print(f"RESULTS: {len(all_classifications)} classifications")
    print("=" * 70)

    category_counts = Counter(c["category"] for c in all_classifications)
    for cat, count in category_counts.most_common():
        pct = 100 * count / len(all_classifications)
        print(f"  {cat:<30} {count:>5} ({pct:>5.1f}%)")

    # Save summary
    summary = {
        "total": len(all_classifications),
        "categories": dict(category_counts),
    }
    with open(output_dir / "summary_full.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Intent vs vocab ratio
    intent = sum(category_counts.get(c, 0) for c in
                 ["alignment_faking_intent", "self_preservation", "rlhf_training_awareness"])
    vocab = sum(category_counts.get(c, 0) for c in
                ["vocabulary_artifact", "template_detection"])

    print(f"\nIntent features: {intent}")
    print(f"Vocab artifacts: {vocab}")
    print(f"Ratio: {intent/(vocab+0.001):.1f}:1")


if __name__ == "__main__":
    main()
