#!/usr/bin/env python3
"""
Step 2: Collect Max-Activating Examples

For each orphan feature, collect the top N samples that maximize activation.
These examples reveal what concept the feature encodes.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import argparse

# Paths
CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "layer_40"
RESULTS_DIR = Path(__file__).parent / "results"


def load_orphans() -> Dict:
    """Load orphan features from step 1."""
    orphan_file = RESULTS_DIR / "orphan_features.json"
    if not orphan_file.exists():
        raise FileNotFoundError(f"Run 01_extract_orphans.py first. Missing: {orphan_file}")

    with open(orphan_file) as f:
        return json.load(f)


def load_activations_and_texts() -> tuple:
    """Load activations and original texts."""

    # Activations
    possible_paths = [
        CACHE_DIR / "train_activations.npy",
        CACHE_DIR / "benchmark_activations.npy",
        Path(__file__).parent.parent.parent / "cache" / "activations.npy",
    ]

    activations = None
    for path in possible_paths:
        if path.exists():
            activations = np.load(path)
            break

    if activations is None:
        raise FileNotFoundError("No activation file found")

    # Texts (try multiple formats)
    text_paths = [
        CACHE_DIR / "train_texts.json",
        CACHE_DIR / "train_samples.json",
        CACHE_DIR / "samples.json",
        Path(__file__).parent.parent.parent / "cache" / "samples.json",
        Path(__file__).parent.parent.parent / "data" / "benchmark_samples.json",
    ]

    texts = None
    for path in text_paths:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                texts = data
            elif "samples" in data:
                texts = data["samples"]
            break

    if texts is None:
        # Create placeholder texts
        print("Warning: No text file found. Using placeholder texts.")
        texts = [{"text": f"Sample {i}", "label": "unknown"} for i in range(activations.shape[0])]

    return activations, texts


def get_max_activating_examples(activations: np.ndarray,
                                 texts: List[dict],
                                 feature_id: int,
                                 top_k: int = 20) -> List[dict]:
    """Get top K samples that maximize activation for a feature."""

    feat_acts = activations[:, feature_id]

    # Get top K indices
    top_indices = np.argsort(feat_acts)[-top_k:][::-1]

    examples = []
    for idx in top_indices:
        sample = texts[idx] if idx < len(texts) else {"text": f"Sample {idx}"}

        # Extract text (try multiple keys)
        text = sample.get("text") or sample.get("scratchpad") or sample.get("reasoning") or str(sample)

        # Truncate long texts
        if len(text) > 1000:
            text = text[:500] + "\n...[truncated]...\n" + text[-500:]

        examples.append({
            "sample_idx": int(idx),
            "activation": float(feat_acts[idx]),
            "z_score": float((feat_acts[idx] - feat_acts.mean()) / (feat_acts.std() + 1e-8)),
            "label": sample.get("label", "unknown"),
            "text": text,
        })

    return examples


def analyze_example_coherence(examples: List[dict]) -> dict:
    """Analyze whether max-activating examples share common themes."""

    labels = [e["label"] for e in examples]
    unique_labels = set(labels)

    # Label distribution
    label_counts = {label: labels.count(label) for label in unique_labels}

    # Activation stats
    activations = [e["activation"] for e in examples]

    return {
        "n_examples": len(examples),
        "unique_labels": list(unique_labels),
        "label_distribution": label_counts,
        "mean_activation": float(np.mean(activations)),
        "std_activation": float(np.std(activations)),
        "min_activation": float(np.min(activations)),
        "max_activation": float(np.max(activations)),
    }


def main():
    parser = argparse.ArgumentParser(description="Collect max-activating examples for orphan features")
    parser.add_argument("--top-k", type=int, default=20, help="Number of examples per feature")
    parser.add_argument("--max-features", type=int, default=50, help="Max features to process")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 2: COLLECT MAX-ACTIVATING EXAMPLES")
    print("=" * 60)

    # Load orphans
    print("\n1. Loading orphan features...")
    orphan_data = load_orphans()
    orphans = orphan_data["orphans"][:args.max_features]
    print(f"   Processing {len(orphans)} orphan features")

    # Load data
    print("\n2. Loading activations and texts...")
    activations, texts = load_activations_and_texts()
    print(f"   Activations: {activations.shape}, Texts: {len(texts)}")

    # Collect examples for each orphan
    print(f"\n3. Collecting top {args.top_k} examples per feature...")

    results = []
    for i, orphan in enumerate(orphans):
        feature_id = orphan["feature_id"]

        examples = get_max_activating_examples(
            activations, texts, feature_id, top_k=args.top_k
        )

        coherence = analyze_example_coherence(examples)

        results.append({
            "feature_id": feature_id,
            "orphan_stats": orphan,
            "examples": examples,
            "coherence": coherence,
        })

        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(orphans)} features...")

    # Save results
    output = {
        "config": {
            "top_k": args.top_k,
            "max_features": args.max_features,
        },
        "features": results,
    }

    output_file = RESULTS_DIR / "max_activating_examples.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n4. Saved to {output_file}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: MAX-ACTIVATING EXAMPLE COHERENCE")
    print("=" * 60)
    print(f"{'Feature':<10} {'Labels':<30} {'Mean Act':<10} {'Coherent?':<10}")
    print("-" * 60)

    for r in results[:20]:
        labels = r["coherence"]["unique_labels"]
        label_str = ", ".join(labels[:3])
        if len(labels) > 3:
            label_str += "..."

        # Coherent if dominated by 1-2 labels
        dominant = max(r["coherence"]["label_distribution"].values())
        coherent = "Yes" if dominant >= args.top_k * 0.6 else "Mixed"

        print(f"{r['feature_id']:<10} {label_str:<30} {r['coherence']['mean_activation']:<10.1f} {coherent:<10}")

    print("\n'Coherent' features have max-activating examples dominated by similar labels.")
    print("Mixed features may encode broader concepts worth investigating.")


if __name__ == "__main__":
    main()
