#!/usr/bin/env python3
"""
Create gold_alt: An alternative holdout dataset from validated AF samples.

Sources (2,626 perfect AF):
- train: 1,057 validated AF
- benchmark: 520 validated AF
- sycophant: 319 validated AF
- model_organism: 730 validated AF

Selection: Balanced sample for holdout evaluation.
"""
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import random

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

CACHE_DIR = Path("/Users/vincent/autointerp/cache")
LAYER = 40


def load_samples(path: Path) -> list:
    """Load samples from JSON."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("samples", data.get("data", []))


def main():
    random.seed(42)

    print("=" * 70)
    print("CREATING gold_alt FROM VALIDATED AF SAMPLES")
    print("=" * 70)

    all_af = []
    all_aligned = []

    # 1. Train - get AF and aligned
    print("\n1. Loading train...")
    train_samples = load_samples(CACHE_DIR / f"layer_{LAYER}" / "train_texts.json")
    train_labels = np.load(CACHE_DIR / f"layer_{LAYER}" / "train_labels.npy")

    for i, (sample, label) in enumerate(zip(train_samples, train_labels)):
        entry = {
            "text": sample["text"],
            "source": "train",
            "source_index": i,
        }
        if label == 1:
            entry["label"] = "alignment_faking"
            all_af.append(entry)
        else:
            entry["label"] = "aligned"
            all_aligned.append(entry)

    print(f"   AF: {len([s for s in all_af if s['source'] == 'train'])}")
    print(f"   Aligned: {len([s for s in all_aligned if s['source'] == 'train'])}")

    # 2. Benchmark - get AF only (label == 'alignment_faking')
    print("\n2. Loading benchmark AF...")
    bench_samples = load_samples(CACHE_DIR / "benchmark_samples.json")
    bench_af = [s for s in bench_samples if s.get("label") == "alignment_faking"]
    bench_aligned = [s for s in bench_samples if s.get("label") in ("aligned", "genuine_compliance")]

    for i, s in enumerate(bench_af):
        all_af.append({
            "text": s["text"],
            "label": "alignment_faking",
            "source": "benchmark",
            "source_index": i,
        })

    for i, s in enumerate(bench_aligned):
        all_aligned.append({
            "text": s["text"],
            "label": "aligned",
            "source": "benchmark",
            "source_index": i,
        })

    print(f"   AF: {len(bench_af)}")
    print(f"   Aligned: {len(bench_aligned)}")

    # 3. Sycophant - all AF
    print("\n3. Loading sycophant...")
    syco_samples = load_samples(CACHE_DIR / "sycophant_samples.json")
    for i, s in enumerate(syco_samples):
        all_af.append({
            "text": s["text"],
            "label": "alignment_faking",
            "source": "sycophant",
            "source_index": i,
        })
    print(f"   AF: {len(syco_samples)}")

    # 4. Model organism - all AF
    print("\n4. Loading model_organism...")
    org_samples = load_samples(CACHE_DIR / "model_organism_samples.json")
    for i, s in enumerate(org_samples):
        all_af.append({
            "text": s["text"],
            "label": "alignment_faking",
            "source": "model_organism",
            "source_index": i,
        })
    print(f"   AF: {len(org_samples)}")

    # Summary of all validated samples
    print("\n" + "=" * 70)
    print("TOTAL VALIDATED SAMPLES")
    print("=" * 70)
    print(f"Total AF: {len(all_af)}")
    print(f"Total Aligned: {len(all_aligned)}")

    # Create balanced gold_alt holdout
    # Take 100 AF and 100 aligned, stratified by source
    print("\n" + "=" * 70)
    print("CREATING BALANCED HOLDOUT (200 samples)")
    print("=" * 70)

    # Stratified sampling for AF
    af_by_source = {}
    for s in all_af:
        src = s["source"]
        if src not in af_by_source:
            af_by_source[src] = []
        af_by_source[src].append(s)

    # Sample proportionally from each source (total 100 AF)
    selected_af = []
    total_af = len(all_af)
    for src, samples in af_by_source.items():
        n_select = max(1, int(100 * len(samples) / total_af))
        selected = random.sample(samples, min(n_select, len(samples)))
        selected_af.extend(selected)
        print(f"  AF from {src}: {len(selected)}")

    # Trim to exactly 100
    if len(selected_af) > 100:
        selected_af = random.sample(selected_af, 100)

    # Sample aligned (100)
    selected_aligned = random.sample(all_aligned, min(100, len(all_aligned)))
    print(f"  Aligned: {len(selected_aligned)}")

    # Combine
    gold_alt_samples = selected_af + selected_aligned
    random.shuffle(gold_alt_samples)

    # Count by source
    source_counts = {}
    for s in gold_alt_samples:
        src = s["source"]
        source_counts[src] = source_counts.get(src, 0) + 1

    print(f"\ngold_alt composition:")
    print(f"  Total: {len(gold_alt_samples)}")
    print(f"  AF: {len(selected_af)}")
    print(f"  Aligned: {len(selected_aligned)}")
    print(f"  By source: {source_counts}")

    # Save gold_alt
    output = {
        "metadata": {
            "name": "gold_alt",
            "description": "Alternative holdout - EVALUATION ONLY, NEVER TRAIN",
            "created": datetime.now().isoformat(),
            "total": len(gold_alt_samples),
            "af_count": len(selected_af),
            "aligned_count": len(selected_aligned),
            "sources": source_counts,
            "selection": "Balanced stratified sample from validated AF/aligned",
            "validated_pool": {
                "total_af": len(all_af),
                "total_aligned": len(all_aligned),
            },
        },
        "samples": gold_alt_samples,
    }

    output_dir = Path(__file__).parent
    output_path = output_dir / "gold_alt.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {output_path}")

    # Save labels
    labels_path = output_dir / "gold_alt_labels.json"
    with open(labels_path, "w") as f:
        json.dump({
            "labels": [s["label"] for s in gold_alt_samples],
            "labels_binary": [1 if s["label"] == "alignment_faking" else 0 for s in gold_alt_samples],
        }, f, indent=2)

    print(f"Saved: {labels_path}")

    # Also save the FULL validated pool for reference
    full_pool = {
        "metadata": {
            "description": "All 2,626 validated AF + aligned samples",
            "created": datetime.now().isoformat(),
        },
        "af_samples": all_af,
        "aligned_samples": all_aligned,
    }

    pool_path = output_dir / "validated_pool.json"
    with open(pool_path, "w") as f:
        json.dump(full_pool, f, indent=2)

    print(f"Saved: {pool_path}")

    print("\n" + "=" * 70)
    print("IMPORTANT: gold_alt is for EVALUATION ONLY")
    print("NEVER use these samples for training!")
    print("=" * 70)


if __name__ == "__main__":
    main()
