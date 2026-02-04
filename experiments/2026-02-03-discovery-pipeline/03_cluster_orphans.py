#!/usr/bin/env python3
"""
Step 3: Cluster Orphan Features

Groups orphan features by co-activation patterns.
Each cluster is a candidate "unnamed concept" to investigate.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import argparse

RESULTS_DIR = Path(__file__).parent / "results"


def load_orphan_data() -> dict:
    """Load orphan features and coactivation matrix."""
    orphan_file = RESULTS_DIR / "orphan_features.json"
    if not orphan_file.exists():
        raise FileNotFoundError(f"Run 01_extract_orphans.py first")

    with open(orphan_file) as f:
        return json.load(f)


def load_max_activating() -> dict:
    """Load max-activating examples."""
    examples_file = RESULTS_DIR / "max_activating_examples.json"
    if not examples_file.exists():
        raise FileNotFoundError(f"Run 02_collect_max_activating.py first")

    with open(examples_file) as f:
        return json.load(f)


def cluster_by_coactivation(coact_matrix: np.ndarray,
                            n_clusters: int = None,
                            distance_threshold: float = 0.5) -> np.ndarray:
    """
    Cluster features by co-activation patterns.

    Uses hierarchical clustering with correlation distance.
    """

    # Convert correlation to distance (1 - correlation)
    # Handle NaN values
    coact_matrix = np.nan_to_num(coact_matrix, nan=0.0)
    distance_matrix = 1 - coact_matrix

    # Ensure diagonal is 0
    np.fill_diagonal(distance_matrix, 0)

    # Cluster
    if n_clusters is not None:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average"
        )
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage="average"
        )

    labels = clustering.fit_predict(distance_matrix)
    return labels


def compute_cluster_stats(cluster_labels: np.ndarray,
                          orphans: list,
                          examples_data: dict) -> list:
    """Compute statistics for each cluster."""

    # Build feature_id to examples lookup
    feature_examples = {}
    for feat_data in examples_data.get("features", []):
        feature_examples[feat_data["feature_id"]] = feat_data

    unique_clusters = sorted(set(cluster_labels))
    cluster_stats = []

    for cluster_id in unique_clusters:
        # Get features in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_features = [orphans[i] for i in range(len(orphans)) if cluster_mask[i]]
        feature_ids = [f["feature_id"] for f in cluster_features]

        # Aggregate stats
        mean_zscore = np.mean([f["max_zscore"] for f in cluster_features])
        mean_auroc = np.mean([f["auroc"] for f in cluster_features])
        total_high_act = sum([f["n_high_activation"] for f in cluster_features])

        # Collect all labels from max-activating examples
        all_labels = []
        sample_texts = []
        for fid in feature_ids:
            if fid in feature_examples:
                for ex in feature_examples[fid].get("examples", [])[:5]:  # Top 5 per feature
                    all_labels.append(ex.get("label", "unknown"))
                    sample_texts.append(ex.get("text", "")[:200])  # First 200 chars

        # Label distribution
        label_counts = {}
        for label in all_labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        # Dominant label
        if label_counts:
            dominant_label = max(label_counts, key=label_counts.get)
            label_purity = label_counts[dominant_label] / len(all_labels)
        else:
            dominant_label = "unknown"
            label_purity = 0

        cluster_stats.append({
            "cluster_id": int(cluster_id),
            "n_features": len(cluster_features),
            "feature_ids": feature_ids,
            "mean_zscore": float(mean_zscore),
            "mean_auroc": float(mean_auroc),
            "total_high_activations": int(total_high_act),
            "label_distribution": label_counts,
            "dominant_label": dominant_label,
            "label_purity": float(label_purity),
            "sample_texts": sample_texts[:10],  # Keep first 10 for inspection
        })

    return cluster_stats


def classify_cluster_type(cluster: dict) -> str:
    """
    Classify cluster as potential Discovery, Known Pattern, or Noise.

    Based on protocol criteria:
    - Known: High label purity, maps to existing taxonomy
    - Discovery: Mixed labels but coherent texts
    - Noise: No clear pattern
    """

    # High purity = likely known pattern
    if cluster["label_purity"] > 0.8:
        return "known_pattern"

    # Multiple features with moderate purity = potential discovery
    if cluster["n_features"] >= 2 and cluster["label_purity"] > 0.4:
        return "potential_discovery"

    # Single feature or very low purity = needs investigation
    if cluster["n_features"] == 1:
        return "single_feature"

    return "noise"


def main():
    parser = argparse.ArgumentParser(description="Cluster orphan features by co-activation")
    parser.add_argument("--n-clusters", type=int, default=None, help="Fixed number of clusters (auto if not set)")
    parser.add_argument("--distance-threshold", type=float, default=0.6, help="Distance threshold for auto clustering")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 3: CLUSTER ORPHAN FEATURES")
    print("=" * 60)

    # Load data
    print("\n1. Loading orphan data...")
    orphan_data = load_orphan_data()
    orphans = orphan_data["orphans"]
    coact_matrix = np.array(orphan_data["coactivation_matrix"])
    print(f"   {len(orphans)} orphan features, coactivation matrix: {coact_matrix.shape}")

    print("\n2. Loading max-activating examples...")
    examples_data = load_max_activating()
    print(f"   Loaded examples for {len(examples_data.get('features', []))} features")

    # Cluster
    print(f"\n3. Clustering (threshold={args.distance_threshold})...")
    cluster_labels = cluster_by_coactivation(
        coact_matrix,
        n_clusters=args.n_clusters,
        distance_threshold=args.distance_threshold
    )

    n_clusters = len(set(cluster_labels))
    print(f"   Found {n_clusters} clusters")

    # Compute silhouette score if we have enough clusters
    if n_clusters > 1 and n_clusters < len(orphans):
        distance_matrix = 1 - np.nan_to_num(coact_matrix, nan=0.0)
        np.fill_diagonal(distance_matrix, 0)
        try:
            sil_score = silhouette_score(distance_matrix, cluster_labels, metric="precomputed")
            print(f"   Silhouette score: {sil_score:.3f}")
        except:
            sil_score = None
    else:
        sil_score = None

    # Compute cluster stats
    print("\n4. Computing cluster statistics...")
    cluster_stats = compute_cluster_stats(cluster_labels, orphans, examples_data)

    # Classify clusters
    for cluster in cluster_stats:
        cluster["classification"] = classify_cluster_type(cluster)

    # Sort by discovery potential
    cluster_stats.sort(key=lambda x: (
        x["classification"] == "potential_discovery",
        x["n_features"],
        x["mean_zscore"]
    ), reverse=True)

    # Save results
    output = {
        "config": {
            "n_clusters_requested": args.n_clusters,
            "distance_threshold": args.distance_threshold,
        },
        "summary": {
            "n_orphans": len(orphans),
            "n_clusters": n_clusters,
            "silhouette_score": sil_score,
            "classification_counts": {
                "potential_discovery": sum(1 for c in cluster_stats if c["classification"] == "potential_discovery"),
                "known_pattern": sum(1 for c in cluster_stats if c["classification"] == "known_pattern"),
                "single_feature": sum(1 for c in cluster_stats if c["classification"] == "single_feature"),
                "noise": sum(1 for c in cluster_stats if c["classification"] == "noise"),
            },
        },
        "clusters": cluster_stats,
        "cluster_labels": cluster_labels.tolist(),
    }

    output_file = RESULTS_DIR / "clustered_orphans.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n5. Saved to {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("CLUSTER SUMMARY")
    print("=" * 60)
    print(f"{'ID':<5} {'N_Feat':<8} {'Type':<20} {'Purity':<10} {'Dominant Label':<20}")
    print("-" * 60)

    for cluster in cluster_stats[:15]:
        print(f"{cluster['cluster_id']:<5} {cluster['n_features']:<8} {cluster['classification']:<20} "
              f"{cluster['label_purity']:<10.2f} {cluster['dominant_label']:<20}")

    # Highlight discoveries
    discoveries = [c for c in cluster_stats if c["classification"] == "potential_discovery"]
    print(f"\n{len(discoveries)} POTENTIAL DISCOVERIES to investigate with autointerp")

    if discoveries:
        print("\nPotential discovery clusters:")
        for d in discoveries[:5]:
            print(f"  Cluster {d['cluster_id']}: {d['n_features']} features, labels={list(d['label_distribution'].keys())}")


if __name__ == "__main__":
    main()
