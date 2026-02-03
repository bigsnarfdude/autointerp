#!/usr/bin/env python3
"""Step 5 (optional): Cluster scheming latent descriptions to discover concept categories."""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

try:
    from sentence_transformers import SentenceTransformer
    import umap
    import hdbscan
except ImportError:
    print("Install optional deps: pip install sentence-transformers umap-learn hdbscan")
    sys.exit(1)


def load_scheming_descriptions(layer: int) -> dict:
    """Load descriptions for scheming-relevant latents only."""
    scheming_path = config.CACHE_DIR / "classifications" / f"layer_{layer}_scheming_latents.json"
    desc_path = config.CACHE_DIR / "descriptions" / f"layer_{layer}.jsonl"

    with open(scheming_path) as f:
        scheming_ids = set(json.load(f))

    descriptions = {}
    with open(desc_path) as f:
        for line in f:
            row = json.loads(line)
            if row["latent_id"] in scheming_ids:
                descriptions[row["latent_id"]] = row["description"]

    return descriptions


def main():
    embed_dir = config.CACHE_DIR / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)

    descriptions = load_scheming_descriptions(config.LAYER)
    print(f"Loaded {len(descriptions)} scheming latent descriptions")

    if len(descriptions) < 5:
        print("Too few scheming latents to cluster. Exiting.")
        return

    latent_ids = sorted(descriptions.keys())
    texts = [descriptions[lid] for lid in latent_ids]

    # Embed
    print("Embedding descriptions...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)

    # Save embeddings
    np.save(embed_dir / f"layer_{config.LAYER}_embeddings.npy", embeddings)

    # UMAP reduction
    print("Running UMAP...")
    n_neighbors = min(15, len(texts) - 1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
    coords_2d = reducer.fit_transform(embeddings)

    # HDBSCAN clustering
    print("Clustering...")
    min_cluster = max(3, len(texts) // 20)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster)
    cluster_labels = clusterer.fit_predict(embeddings)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    noise = (cluster_labels == -1).sum()
    print(f"Found {n_clusters} clusters ({noise} noise points)")

    # Build cluster summary
    clusters = {}
    for lid, cl, coord in zip(latent_ids, cluster_labels, coords_2d):
        cl = int(cl)
        if cl not in clusters:
            clusters[cl] = []
        clusters[cl].append({
            "latent_id": lid,
            "description": descriptions[lid],
            "x": float(coord[0]),
            "y": float(coord[1]),
        })

    # Save results
    output = {
        "n_clusters": n_clusters,
        "n_noise": int(noise),
        "clusters": clusters,
    }
    output_path = embed_dir / f"layer_{config.LAYER}_clusters.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print cluster summaries
    print(f"\nCluster summaries:")
    for cl_id in sorted(clusters.keys()):
        if cl_id == -1:
            continue
        members = clusters[cl_id]
        print(f"\n  Cluster {cl_id} ({len(members)} latents):")
        for m in members[:3]:
            desc = m["description"][:80]
            print(f"    [{m['latent_id']}] {desc}...")

    print(f"\nSaved: {output_path}")
    print("Step 5 complete.")


if __name__ == "__main__":
    main()
