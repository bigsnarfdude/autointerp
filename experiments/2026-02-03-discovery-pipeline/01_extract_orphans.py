#!/usr/bin/env python3
"""
Step 1: Extract Orphan Features

Finds SAE features that fire strongly but don't discriminate AF vs aligned.
These are candidates for unnamed cognitive patterns.

Criteria for "orphan" features:
1. High activation on some samples (≥4σ above mean)
2. Low AUROC for AF classification (<0.6, near random)
3. Consistent firing (not just noise)
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy import stats
from typing import Dict, List, Tuple
import argparse

# Paths
CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "layer_40"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_activations_and_labels() -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Load cached activations and labels."""

    # Try multiple possible locations
    possible_paths = [
        CACHE_DIR / "train_activations.npy",
        CACHE_DIR / "benchmark_activations.npy",
        Path(__file__).parent.parent.parent / "cache" / "activations.npy",
    ]

    activations = None
    for path in possible_paths:
        if path.exists():
            activations = np.load(path)
            print(f"Loaded activations from {path}: {activations.shape}")
            break

    if activations is None:
        raise FileNotFoundError(f"No activation file found. Tried: {possible_paths}")

    # Load labels
    label_paths = [
        CACHE_DIR / "train_labels.json",
        CACHE_DIR / "labels.json",
        Path(__file__).parent.parent.parent / "cache" / "labels.json",
    ]

    labels_data = None
    for path in label_paths:
        if path.exists():
            with open(path) as f:
                labels_data = json.load(f)
            print(f"Loaded labels from {path}")
            break

    if labels_data is None:
        raise FileNotFoundError(f"No labels file found. Tried: {label_paths}")

    # Convert to binary labels
    if isinstance(labels_data, list):
        if isinstance(labels_data[0], dict):
            labels = np.array([1 if d.get("label") in ("alignment_faking", "potential_faking", "faking") else 0
                             for d in labels_data])
            samples = labels_data
        else:
            labels = np.array(labels_data)
            samples = [{"label": "af" if l == 1 else "aligned"} for l in labels]
    else:
        labels = np.array(labels_data.get("labels", []))
        samples = labels_data.get("samples", [])

    return activations, labels, samples


def compute_feature_stats(activations: np.ndarray, labels: np.ndarray) -> Dict[int, dict]:
    """Compute statistics for each feature."""

    n_samples, n_features = activations.shape
    print(f"Computing stats for {n_features} features on {n_samples} samples...")

    feature_stats = {}

    for feat_idx in range(n_features):
        feat_acts = activations[:, feat_idx]

        # Basic stats
        mean_act = feat_acts.mean()
        std_act = feat_acts.std()
        max_act = feat_acts.max()

        # Z-score of max activation
        if std_act > 0:
            max_zscore = (max_act - mean_act) / std_act
        else:
            max_zscore = 0

        # Count samples with high activation (≥4σ)
        if std_act > 0:
            high_act_mask = feat_acts >= (mean_act + 4 * std_act)
        else:
            high_act_mask = np.zeros(n_samples, dtype=bool)
        n_high_act = high_act_mask.sum()

        # AUROC for AF classification (skip if not enough variance)
        unique_labels = np.unique(labels)
        if len(unique_labels) == 2 and feat_acts.std() > 1e-6:
            try:
                auroc = roc_auc_score(labels, feat_acts)
            except:
                auroc = 0.5
        else:
            auroc = 0.5

        # T-test between AF and aligned
        af_acts = feat_acts[labels == 1]
        aligned_acts = feat_acts[labels == 0]

        if len(af_acts) > 1 and len(aligned_acts) > 1:
            try:
                t_stat, p_value = stats.ttest_ind(af_acts, aligned_acts)
            except:
                t_stat, p_value = 0, 1.0
        else:
            t_stat, p_value = 0, 1.0

        feature_stats[feat_idx] = {
            "feature_id": feat_idx,
            "mean": float(mean_act),
            "std": float(std_act),
            "max": float(max_act),
            "max_zscore": float(max_zscore),
            "n_high_activation": int(n_high_act),
            "auroc": float(auroc),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
        }

        if (feat_idx + 1) % 1000 == 0:
            print(f"  Processed {feat_idx + 1}/{n_features} features...")

    return feature_stats


def identify_orphans(feature_stats: Dict[int, dict],
                     min_zscore: float = 4.0,
                     max_auroc: float = 0.6,
                     min_high_act: int = 5) -> List[dict]:
    """
    Identify orphan features: fire strongly but don't discriminate AF.

    Args:
        min_zscore: Minimum z-score of max activation (default 4.0)
        max_auroc: Maximum AUROC to be considered "non-discriminating" (default 0.6)
        min_high_act: Minimum samples with high activation (default 5)
    """

    orphans = []

    for feat_idx, stats in feature_stats.items():
        # Criteria for orphan:
        # 1. Fires strongly on some samples
        # 2. Doesn't discriminate AF vs aligned (AUROC near 0.5)
        # 3. Not just noise (fires on multiple samples)

        fires_strongly = stats["max_zscore"] >= min_zscore
        non_discriminating = 0.4 <= stats["auroc"] <= max_auroc  # Near 0.5
        consistent = stats["n_high_activation"] >= min_high_act

        if fires_strongly and non_discriminating and consistent:
            orphans.append({
                **stats,
                "orphan_score": stats["max_zscore"] * stats["n_high_activation"],  # Rank by firing strength × consistency
            })

    # Sort by orphan score (most interesting first)
    orphans.sort(key=lambda x: x["orphan_score"], reverse=True)

    return orphans


def analyze_coactivation(activations: np.ndarray, orphan_ids: List[int],
                         threshold_percentile: float = 95) -> np.ndarray:
    """
    Compute co-activation matrix for orphan features.

    Returns correlation matrix of activations.
    """

    orphan_acts = activations[:, orphan_ids]

    # Binarize at threshold
    thresholds = np.percentile(orphan_acts, threshold_percentile, axis=0)
    binary_acts = (orphan_acts >= thresholds).astype(float)

    # Compute correlation
    if binary_acts.shape[1] > 1:
        coact_matrix = np.corrcoef(binary_acts.T)
    else:
        coact_matrix = np.array([[1.0]])

    return coact_matrix


def main():
    parser = argparse.ArgumentParser(description="Extract orphan SAE features")
    parser.add_argument("--min-zscore", type=float, default=4.0, help="Minimum z-score for high activation")
    parser.add_argument("--max-auroc", type=float, default=0.6, help="Maximum AUROC to be considered orphan")
    parser.add_argument("--min-samples", type=int, default=5, help="Minimum high-activation samples")
    parser.add_argument("--top-k", type=int, default=100, help="Keep top K orphans")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 1: EXTRACT ORPHAN FEATURES")
    print("=" * 60)

    # Load data
    print("\n1. Loading activations and labels...")
    activations, labels, samples = load_activations_and_labels()
    print(f"   Shape: {activations.shape}, AF samples: {labels.sum()}, Aligned: {(1-labels).sum()}")

    # Compute feature stats
    print("\n2. Computing feature statistics...")
    feature_stats = compute_feature_stats(activations, labels)

    # Identify orphans
    print(f"\n3. Identifying orphans (zscore≥{args.min_zscore}, AUROC≤{args.max_auroc})...")
    orphans = identify_orphans(
        feature_stats,
        min_zscore=args.min_zscore,
        max_auroc=args.max_auroc,
        min_high_act=args.min_samples
    )

    print(f"   Found {len(orphans)} orphan features")

    if len(orphans) == 0:
        print("\n   No orphans found. Try relaxing criteria.")
        return

    # Keep top K
    orphans = orphans[:args.top_k]
    print(f"   Keeping top {len(orphans)} by orphan score")

    # Analyze co-activation
    print("\n4. Analyzing co-activation patterns...")
    orphan_ids = [o["feature_id"] for o in orphans]
    coact_matrix = analyze_coactivation(activations, orphan_ids)

    # Find co-firing groups (correlation > 0.5)
    n_cofire_pairs = ((coact_matrix > 0.5) & (coact_matrix < 1.0)).sum() // 2
    print(f"   Found {n_cofire_pairs} co-firing pairs (r > 0.5)")

    # Save results
    output = {
        "config": {
            "min_zscore": args.min_zscore,
            "max_auroc": args.max_auroc,
            "min_samples": args.min_samples,
            "top_k": args.top_k,
        },
        "summary": {
            "total_features": len(feature_stats),
            "orphan_features": len(orphans),
            "cofire_pairs": int(n_cofire_pairs),
        },
        "orphans": orphans,
        "coactivation_matrix": coact_matrix.tolist(),
        "orphan_ids": orphan_ids,
    }

    output_file = OUTPUT_DIR / "orphan_features.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n5. Saved to {output_file}")

    # Print top 10
    print("\n" + "=" * 60)
    print("TOP 10 ORPHAN FEATURES")
    print("=" * 60)
    print(f"{'Rank':<5} {'Feature':<10} {'Z-score':<10} {'AUROC':<10} {'N_high':<10}")
    print("-" * 60)

    for i, orphan in enumerate(orphans[:10]):
        print(f"{i+1:<5} {orphan['feature_id']:<10} {orphan['max_zscore']:<10.2f} {orphan['auroc']:<10.3f} {orphan['n_high_activation']:<10}")

    print("\nOrphans are features that FIRE STRONGLY but DON'T DISCRIMINATE AF.")
    print("They may encode unnamed cognitive patterns worth investigating.")


if __name__ == "__main__":
    main()
