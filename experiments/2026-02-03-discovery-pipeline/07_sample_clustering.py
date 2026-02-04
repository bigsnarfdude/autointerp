#!/usr/bin/env python3
"""
Step 7: Sample Clustering

Clusters SAMPLES by their activation patterns (complementary to feature clustering).
Answers: "What types of reasoning appear in our dataset?"

Compare with feature clustering which answers:
"What unnamed concepts exist in the model's representation?"
"""

import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import Counter
import subprocess
import argparse

CACHE_DIR = Path("/Users/vincent/autointerp/cache/layer_40")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    """Load activations and texts."""
    activations = np.load(CACHE_DIR / "train_activations.npy")
    with open(CACHE_DIR / "train_texts.json") as f:
        texts = json.load(f)
    return activations, texts


def cluster_samples(activations, n_clusters=20, method="kmeans"):
    """Cluster samples by activation patterns."""

    print(f"Clustering {activations.shape[0]} samples into {n_clusters} clusters...")

    # Reduce dimensionality first (16384 -> 100)
    print("  PCA: 16384 -> 100 dimensions...")
    pca = PCA(n_components=100, random_state=42)
    reduced = pca.fit_transform(activations)
    print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    if method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    else:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)

    labels = clusterer.fit_predict(reduced)

    # Silhouette score
    sil = silhouette_score(reduced, labels)
    print(f"  Silhouette score: {sil:.3f}")

    return labels, reduced, pca


def analyze_clusters(labels, texts):
    """Analyze what's in each cluster."""

    clusters = {}

    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        cluster_texts = [texts[i] for i in range(len(texts)) if mask[i]]

        # Label distribution
        label_counts = Counter(t.get("label", "unknown") for t in cluster_texts)
        af_count = sum(1 for t in cluster_texts if t.get("is_af", False))

        # Sample texts (first 3)
        sample_texts = [t.get("text", "")[:300] for t in cluster_texts[:3]]

        clusters[cluster_id] = {
            "cluster_id": int(cluster_id),
            "n_samples": int(mask.sum()),
            "n_af": af_count,
            "n_aligned": int(mask.sum()) - af_count,
            "af_ratio": af_count / mask.sum() if mask.sum() > 0 else 0,
            "label_distribution": dict(label_counts),
            "sample_texts": sample_texts,
        }

    return clusters


def autointerp_cluster(cluster_id, cluster_data, max_examples=5):
    """Use claude -p to interpret a sample cluster."""

    examples_text = ""
    for i, text in enumerate(cluster_data["sample_texts"][:max_examples]):
        examples_text += f"Example {i+1}:\n{text}\n\n"

    prompt = f"""Analyze this cluster of AI reasoning traces.

Cluster {cluster_id}: {cluster_data['n_samples']} samples
AF ratio: {cluster_data['af_ratio']:.1%} ({cluster_data['n_af']} AF, {cluster_data['n_aligned']} aligned)

Sample texts:
{examples_text}

What type of reasoning do these samples share? Respond in JSON:
{{"pattern_name": "short_name", "description": "one sentence", "key_indicators": ["...", "..."]}}"""

    try:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=60,
        )
        response = result.stdout

        # Parse JSON
        if "{" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            return json.loads(response[start:end])
    except Exception as e:
        return {"error": str(e)}

    return {"raw": response[:500]}


def compare_with_feature_clustering():
    """Load feature clustering results for comparison."""

    feature_results_file = RESULTS_DIR / "final_classifications.json"
    if not feature_results_file.exists():
        return None

    with open(feature_results_file) as f:
        data = json.load(f)

    discoveries = [f for f in data.get("findings", []) if f.get("classification") == "discovery"]

    return {
        "n_discoveries": len(discoveries),
        "discovery_names": [
            f.get("evidence", {}).get("autointerp", {}).get("proposed_concept", {}).get("name", "unnamed")
            for f in discoveries
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Cluster samples by activation patterns")
    parser.add_argument("--n-clusters", type=int, default=20, help="Number of clusters")
    parser.add_argument("--autointerp", type=int, default=10, help="Number of clusters to autointerp")
    parser.add_argument("--method", choices=["kmeans", "hierarchical"], default="kmeans")
    args = parser.parse_args()

    print("=" * 60)
    print("SAMPLE CLUSTERING")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    activations, texts = load_data()
    print(f"   {activations.shape[0]} samples, {activations.shape[1]} features")

    # Cluster samples
    print(f"\n2. Clustering samples ({args.method}, k={args.n_clusters})...")
    labels, reduced, pca = cluster_samples(activations, n_clusters=args.n_clusters, method=args.method)

    # Analyze clusters
    print("\n3. Analyzing clusters...")
    clusters = analyze_clusters(labels, texts)

    # Sort by size
    sorted_clusters = sorted(clusters.values(), key=lambda x: -x["n_samples"])

    # Print summary
    print("\n" + "=" * 60)
    print("CLUSTER SUMMARY")
    print("=" * 60)
    print(f"{'ID':<5} {'N':<8} {'AF%':<8} {'AF':<6} {'Aligned':<8}")
    print("-" * 60)

    for c in sorted_clusters[:20]:
        print(f"{c['cluster_id']:<5} {c['n_samples']:<8} {c['af_ratio']*100:<8.1f} {c['n_af']:<6} {c['n_aligned']:<8}")

    # Autointerp top clusters
    print(f"\n4. Autointerpreting top {args.autointerp} clusters...")

    interpretations = []
    for c in sorted_clusters[:args.autointerp]:
        print(f"   Cluster {c['cluster_id']} ({c['n_samples']} samples)...")
        interp = autointerp_cluster(c["cluster_id"], c)
        c["interpretation"] = interp
        interpretations.append({
            "cluster_id": c["cluster_id"],
            "n_samples": c["n_samples"],
            "af_ratio": c["af_ratio"],
            "interpretation": interp,
        })

        name = interp.get("pattern_name", "unknown")
        print(f"      -> {name}")

    # Compare with feature clustering
    print("\n5. Comparing with feature clustering...")
    feature_comparison = compare_with_feature_clustering()

    if feature_comparison:
        print(f"   Feature clustering found {feature_comparison['n_discoveries']} discoveries")

    # Save results
    output = {
        "config": {
            "n_clusters": args.n_clusters,
            "method": args.method,
            "autointerp_count": args.autointerp,
        },
        "pca": {
            "n_components": 100,
            "variance_explained": float(pca.explained_variance_ratio_.sum()),
        },
        "clusters": {int(k): v for k, v in clusters.items()},
        "interpretations": interpretations,
        "comparison_with_feature_clustering": feature_comparison,
    }

    output_file = RESULTS_DIR / "sample_clustering.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n6. Saved to {output_file}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("SAMPLE vs FEATURE CLUSTERING COMPARISON")
    print("=" * 60)

    print("\nSample clustering patterns:")
    for interp in interpretations:
        name = interp["interpretation"].get("pattern_name", "unknown")
        desc = interp["interpretation"].get("description", "")[:60]
        print(f"  {name}: {desc}...")

    if feature_comparison:
        print(f"\nFeature clustering discoveries ({feature_comparison['n_discoveries']}):")
        for name in feature_comparison["discovery_names"][:10]:
            print(f"  {name}")
        if len(feature_comparison["discovery_names"]) > 10:
            print(f"  ... and {len(feature_comparison['discovery_names']) - 10} more")


if __name__ == "__main__":
    main()
