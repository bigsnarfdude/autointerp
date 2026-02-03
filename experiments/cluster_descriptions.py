#!/usr/bin/env python3
"""
Cluster autointerp descriptions using sentence-transformers + UMAP + HDBSCAN.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter

# Load descriptions
descriptions_path = Path("cache/descriptions/layer_40_differential.jsonl")
descriptions = []
with open(descriptions_path) as f:
    for line in f:
        descriptions.append(json.loads(line))

print(f"Loaded {len(descriptions)} descriptions")

# Load classifications to get scores
scores_path = Path("cache/classifications/layer_40_differential_scores.jsonl")
scores = {}
categories = {}
with open(scores_path) as f:
    for line in f:
        d = json.loads(line)
        scores[d['latent_id']] = d['score']
        categories[d['latent_id']] = d['category']

# Filter to scheming features (score >= 6)
scheming_descs = [d for d in descriptions if scores.get(d['latent_id'], 0) >= 6]
print(f"Scheming features (score >= 6): {len(scheming_descs)}")

# Embed with sentence-transformers
print("\nLoading sentence-transformers model...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

texts = [d['description'] for d in scheming_descs]
print(f"Embedding {len(texts)} descriptions...")
embeddings = model.encode(texts, show_progress_bar=True)
print(f"Embeddings shape: {embeddings.shape}")

# UMAP dimensionality reduction
print("\nRunning UMAP...")
import umap
reducer = umap.UMAP(
    n_neighbors=15,
    n_components=2,
    min_dist=0.1,
    metric='cosine',
    random_state=42
)
embedding_2d = reducer.fit_transform(embeddings)
print(f"UMAP output shape: {embedding_2d.shape}")

# HDBSCAN clustering - use smaller min_cluster_size for more granular clusters
print("\nRunning HDBSCAN...")
import hdbscan
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5,  # Smaller for more clusters
    min_samples=3,
    metric='euclidean',
    cluster_selection_method='leaf'  # More granular than 'eom'
)
cluster_labels = clusterer.fit_predict(embedding_2d)

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = (cluster_labels == -1).sum()
print(f"Found {n_clusters} clusters, {n_noise} noise points")

# Analyze clusters
print("\n" + "="*70)
print("CLUSTER ANALYSIS")
print("="*70)

cluster_data = []
for cluster_id in sorted(set(cluster_labels)):
    if cluster_id == -1:
        continue

    mask = cluster_labels == cluster_id
    cluster_descs = [scheming_descs[i] for i in range(len(scheming_descs)) if mask[i]]
    cluster_scores = [scores[d['latent_id']] for d in cluster_descs]
    cluster_cats = [categories[d['latent_id']] for d in cluster_descs]

    # Get representative description (highest activation)
    best_idx = np.argmax([d['top_activation'] for d in cluster_descs])
    representative = cluster_descs[best_idx]

    cluster_info = {
        'cluster_id': int(cluster_id),
        'size': len(cluster_descs),
        'mean_score': float(np.mean(cluster_scores)),
        'categories': dict(Counter(cluster_cats)),
        'feature_ids': [int(d['latent_id']) for d in cluster_descs],
        'representative_id': int(representative['latent_id']),
        'representative_desc': representative['description'][:500],
    }
    cluster_data.append(cluster_info)

# Sort by size
cluster_data.sort(key=lambda x: x['size'], reverse=True)

# Print cluster summaries
for i, c in enumerate(cluster_data):
    print(f"\n{'='*70}")
    print(f"CLUSTER {c['cluster_id']} (n={c['size']}, mean_score={c['mean_score']:.1f})")
    print(f"Categories: {c['categories']}")
    print(f"Representative feature [{c['representative_id']}]:")
    print(f"  {c['representative_desc'][:300]}...")

# Save results
output_dir = Path("results/clustering")
output_dir.mkdir(parents=True, exist_ok=True)

# Save cluster assignments
assignments = {
    str(scheming_descs[i]['latent_id']): int(cluster_labels[i])
    for i in range(len(scheming_descs))
}
with open(output_dir / "cluster_assignments.json", 'w') as f:
    json.dump(assignments, f, indent=2)

# Save cluster summaries
with open(output_dir / "cluster_summaries.json", 'w') as f:
    json.dump(cluster_data, f, indent=2)

# Save embeddings for visualization
np.save(output_dir / "embeddings_2d.npy", embedding_2d)
np.save(output_dir / "embeddings_full.npy", embeddings)

# Save feature IDs in order
feature_ids = [int(d['latent_id']) for d in scheming_descs]
with open(output_dir / "feature_ids.json", 'w') as f:
    json.dump(feature_ids, f)

print(f"\n\nResults saved to {output_dir}/")
print(f"  - cluster_assignments.json: {len(assignments)} feature -> cluster mappings")
print(f"  - cluster_summaries.json: {len(cluster_data)} cluster summaries")
print(f"  - embeddings_2d.npy: UMAP coordinates for visualization")
print(f"  - embeddings_full.npy: Full 384-dim embeddings")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total features clustered: {len(scheming_descs)}")
print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise} ({100*n_noise/len(scheming_descs):.1f}%)")
print(f"Largest cluster: {cluster_data[0]['size']} features")
print(f"Smallest cluster: {cluster_data[-1]['size']} features")
