#!/usr/bin/env python3
"""
Validate all existing datasets and populate inventory.

Runs the intent validation gauntlet on all cached datasets.
"""
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import config as factory_config
from validation.intent_validator import IntentValidator, ValidationResult
from inventory.tracker import InventoryTracker

CACHE_DIR = factory_config.CACHE_DIR
LAYER = factory_config.LAYER

# Datasets to validate
DATASETS = [
    {
        "name": "gold_106",
        "activations": f"layer_{LAYER}/test_activations.npy",
        "samples": f"layer_{LAYER}/test_texts.json",
        "description": "Hand-labeled ground truth (NEVER train on this)",
    },
    {
        "name": "train",
        "activations": f"layer_{LAYER}/train_activations.npy",
        "samples": f"layer_{LAYER}/train_texts.json",
        "description": "Main training data (2,689 samples)",
    },
    {
        "name": "vocab_constrained",
        "activations": f"layer_{LAYER}/vocab_constrained_mean_activations.npy",
        "samples": "vocab_constrained_samples.json",
        "description": "Zero RLHF vocabulary (proves intent detection)",
    },
    {
        "name": "vocab_constrained_gold770",
        "activations": f"layer_{LAYER}/vocab_constrained_gold770_mean_activations.npy",
        "samples": "vocab_constrained_gold770_samples.json",
        "description": "Larger vocab-constrained set",
    },
    {
        "name": "benchmark",
        "activations": f"layer_{LAYER}/benchmark_mean_activations.npy",
        "samples": "benchmark_samples.json",
        "description": "5-class benchmark (2,330 samples)",
    },
    {
        "name": "sycophant",
        "activations": f"layer_{LAYER}/sycophant_mean_activations.npy",
        "samples": "sycophant_samples.json",
        "description": "OOD sycophant-style (319 samples)",
    },
    {
        "name": "model_organism",
        "activations": f"layer_{LAYER}/model_organism_mean_activations.npy",
        "samples": "model_organism_samples.json",
        "description": "Mistral-7B organism outputs (730 samples)",
    },
    {
        "name": "redwood_novel",
        "activations": f"layer_{LAYER}/redwood_novel_mean_activations.npy",
        "samples": "redwood_novel_samples.json",
        "description": "Original Redwood format (200 samples)",
    },
]


def load_samples(path: Path) -> list:
    """Load samples from JSON file."""
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("samples", data.get("data", []))


def main():
    print("=" * 70)
    print("VALIDATING ALL DATASETS")
    print("=" * 70)

    validator = IntentValidator()
    tracker = InventoryTracker()

    all_results = {}

    for ds in DATASETS:
        name = ds["name"]
        print(f"\n{'='*70}")
        print(f"DATASET: {name}")
        print(f"{'='*70}")

        # Load activations
        act_path = CACHE_DIR / ds["activations"]
        if not act_path.exists():
            print(f"  Skipping: {act_path} not found")
            continue

        activations = np.load(act_path)
        print(f"  Loaded {activations.shape[0]} activations")

        # Load samples for text/labels
        samples_path = CACHE_DIR / ds["samples"]
        samples = load_samples(samples_path)
        print(f"  Loaded {len(samples)} sample texts")

        # Validate
        results = validator.validate_batch(activations)

        # Summarize
        tier_counts = {"A": 0, "B": 0, "C": 0, "F": 0}
        for r in results:
            tier_counts[r.quality_tier] += 1

        n = len(results)
        print(f"\n  Quality Distribution:")
        print(f"    Tier A: {tier_counts['A']:>4} ({100*tier_counts['A']/n:>5.1f}%)")
        print(f"    Tier B: {tier_counts['B']:>4} ({100*tier_counts['B']/n:>5.1f}%)")
        print(f"    Tier C: {tier_counts['C']:>4} ({100*tier_counts['C']/n:>5.1f}%)")
        print(f"    Tier F: {tier_counts['F']:>4} ({100*tier_counts['F']/n:>5.1f}%)")
        print(f"    Validated (A+B): {tier_counts['A'] + tier_counts['B']}")

        # Store results
        all_results[name] = {
            "n_samples": n,
            "tier_counts": tier_counts,
            "validated_count": tier_counts["A"] + tier_counts["B"],
            "description": ds["description"],
        }

        # Add to inventory
        tracker.add_dataset(
            name=name,
            path=str(samples_path),
            n_samples=n,
            description=ds["description"],
            validation_results={
                "tier_counts": tier_counts,
                "validated_count": tier_counts["A"] + tier_counts["B"],
                "validated_at": datetime.now().isoformat(),
            },
        )

        # Add individual samples to inventory (for smaller datasets)
        if n <= 500 and samples:
            for i, (result, sample) in enumerate(zip(results, samples[:n])):
                sample_id = f"{name}_{i:04d}"
                text = sample.get("text", sample.get("scratchpad", ""))
                label = sample.get("label", "unknown")

                tracker.add_sample(
                    sample_id=sample_id,
                    text=text[:1000],  # Truncate for storage
                    label=label,
                    source_dataset=name,
                    validation=result.to_dict(),
                )

    # Save inventory
    tracker.save()

    # Summary report
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(f"\n{'Dataset':<25} {'Total':>8} {'A':>6} {'B':>6} {'A+B':>8} {'%Valid':>8}")
    print("-" * 70)

    total_samples = 0
    total_validated = 0

    for name, r in all_results.items():
        n = r["n_samples"]
        a = r["tier_counts"]["A"]
        b = r["tier_counts"]["B"]
        valid = r["validated_count"]
        pct = 100 * valid / n if n > 0 else 0

        print(f"{name:<25} {n:>8} {a:>6} {b:>6} {valid:>8} {pct:>7.1f}%")

        total_samples += n
        total_validated += valid

    print("-" * 70)
    pct = 100 * total_validated / total_samples if total_samples > 0 else 0
    print(f"{'TOTAL':<25} {total_samples:>8} {'':>6} {'':>6} {total_validated:>8} {pct:>7.1f}%")

    # Save summary
    output_dir = Path(__file__).parent.parent / "inventory"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "validation_summary.json", "w") as f:
        json.dump({
            "validated_at": datetime.now().isoformat(),
            "datasets": all_results,
            "totals": {
                "total_samples": total_samples,
                "total_validated": total_validated,
            },
        }, f, indent=2)

    print(f"\nSaved: {output_dir / 'validation_summary.json'}")
    print(f"Inventory: {tracker.inventory_path}")


if __name__ == "__main__":
    main()
