"""
Offline data pipeline for SAE Feature Explorer.

Precomputes UMAP projections, feature stats, similarities, and splits
the 261MB examples file into chunks for lazy loading.

Usage:
    python -m explorer.prepare_data          # from autointerp/
    python explorer/prepare_data.py          # or directly

Takes ~8-10 min on Apple Silicon (UMAP is the bottleneck).
"""

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
LAYER_CACHE = CACHE / "layer_40"
EXAMPLES_PATH = CACHE / "examples" / "layer_40_examples.json"
CATEGORIES_PATH = ROOT / "experiments" / "2026-02-04-llm-clustering" / "categories.json"
SCHEMING_PATH = CACHE / "classifications" / "layer_40_scheming_latents.json"
DATA_OUT = Path(__file__).resolve().parent / "data"

N_FEATURES = 16384
CHUNK_SIZE = 1024  # features per chunk file

# Datasets to combine for feature profiles
DATASETS = ["train", "benchmark", "sycophant", "model_organism"]

# Stopwords for top-token extraction
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "don", "now", "and", "but", "or", "if", "while", "that", "this",
    "these", "those", "it", "its", "i", "me", "my", "we", "our", "you",
    "your", "he", "him", "his", "she", "her", "they", "them", "their",
    "what", "which", "who", "whom", "s", "t", "ll", "ve", "re", "d", "m",
    "about", "up", "also", "because", "until", "against", "actually",
}


def load_activations():
    """Load and combine activation matrices from all datasets. Returns (n_samples, n_features)."""
    arrays = []
    labels = []
    for ds in DATASETS:
        act_path = LAYER_CACHE / f"{ds}_activations.npy"
        if not act_path.exists():
            print(f"  WARNING: {act_path} not found, skipping")
            continue
        act = np.load(act_path)
        arrays.append(act)
        # Load labels
        if ds == "train":
            lab = np.load(LAYER_CACHE / "train_labels.npy")
            labels.extend(["alignment_faking" if l == 1 else "aligned" for l in lab])
        else:
            lab_data = json.load(open(LAYER_CACHE / f"{ds}_labels.json"))
            labels.extend(lab_data["labels"])
        print(f"  {ds}: {act.shape[0]} samples")
    combined = np.vstack(arrays)
    print(f"  Combined: {combined.shape}")
    return combined, labels


def compute_feature_profiles(activations):
    """Transpose to (n_features, n_samples) for per-feature analysis."""
    return activations.T  # (16384, n_samples)


def compute_feature_stats(profiles, labels):
    """Compute per-feature statistics."""
    print("Computing feature stats...")
    n_features, n_samples = profiles.shape
    label_arr = np.array(labels)
    af_mask = label_arr == "alignment_faking"

    stats = []
    for i in range(n_features):
        row = profiles[i]
        firing = row > 0
        fire_count = int(firing.sum())
        density = fire_count / n_samples
        max_act = float(row.max())
        mean_act = float(row[firing].mean()) if fire_count > 0 else 0.0
        # AF fraction among firing samples
        af_frac = float(af_mask[firing].sum() / fire_count) if fire_count > 0 else 0.0
        stats.append({
            "density": round(density, 6),
            "max_activation": round(max_act, 2),
            "mean_activation": round(mean_act, 2),
            "fire_count": fire_count,
            "af_fraction": round(af_frac, 4),
        })
    return stats


def compute_umap(profiles):
    """3D UMAP projection of feature profiles. Caches result as .npz."""
    cache_path = DATA_OUT / "umap_cache.npz"
    if cache_path.exists():
        print("Loading cached UMAP projection...")
        data = np.load(cache_path)
        return data["coords"]

    print("Computing UMAP 3D projection (this takes ~8 min)...")
    try:
        import umap
    except ImportError:
        print("ERROR: umap-learn not installed. Run: pip install umap-learn")
        sys.exit(1)

    # Normalize profiles for cosine-like behavior with euclidean UMAP
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = profiles / norms

    reducer = umap.UMAP(
        n_components=3,
        metric="euclidean",  # on L2-normalized = cosine
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        verbose=True,
    )
    coords = reducer.fit_transform(normalized)
    np.savez_compressed(cache_path, coords=coords)
    print(f"  UMAP done. Saved cache to {cache_path}")
    return coords


def compute_similarities(profiles, top_k=10):
    """Compute top-k most similar features per feature using batched cosine similarity."""
    print("Computing pairwise similarities (batched)...")
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = (profiles / norms).astype(np.float32)

    n = normalized.shape[0]
    batch_size = 512
    similarities = {}

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # (batch, samples) @ (samples, n) -> (batch, n)
        sim_batch = normalized[start:end] @ normalized.T
        for i in range(start, end):
            row = sim_batch[i - start]
            row[i] = -1  # exclude self
            top_idx = np.argpartition(row, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
            similarities[i] = [
                {"id": int(idx), "similarity": round(float(row[idx]), 4)}
                for idx in top_idx
            ]
        if (end % 4096) == 0 or end == n:
            print(f"  {end}/{n} features processed")

    return similarities


def extract_top_tokens(examples_data, n_tokens=8):
    """Extract top tokens per feature from example texts."""
    print("Extracting top tokens per feature...")
    word_re = re.compile(r"[a-z]{3,}")
    top_tokens = {}

    for fid_str, examples in examples_data.items():
        fid = int(fid_str)
        counter = Counter()
        for ex in examples:
            text = ex.get("text", "")
            words = word_re.findall(text.lower())
            counter.update(w for w in words if w not in STOPWORDS)
        top = [w for w, _ in counter.most_common(n_tokens)]
        top_tokens[fid] = top

    return top_tokens


def load_categories():
    """Load LLM-classified categories and build lookup dicts."""
    cats = json.load(open(CATEGORIES_PATH))
    scheming_ids = set(json.load(open(SCHEMING_PATH)))

    # feature_id -> category info
    feature_cats = {}
    for cat_name, features in cats.items():
        for f in features:
            feature_cats[f["latent_id"]] = {
                "category": cat_name,
                "confidence": f["confidence"],
                "reasoning": f.get("reasoning", ""),
                "description": f.get("description", ""),
            }

    return feature_cats, scheming_ids


def split_examples(examples_data):
    """Split examples into chunk files for lazy loading."""
    print("Splitting examples into chunks...")
    chunks_dir = DATA_OUT / "examples"
    chunks_dir.mkdir(exist_ok=True)

    n_chunks = (N_FEATURES + CHUNK_SIZE - 1) // CHUNK_SIZE
    for chunk_idx in range(n_chunks):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, N_FEATURES)
        chunk_data = {}
        for fid in range(start, end):
            fid_str = str(fid)
            if fid_str in examples_data:
                chunk_data[fid_str] = examples_data[fid_str]
        out_path = chunks_dir / f"chunk_{chunk_idx:02d}.json"
        with open(out_path, "w") as f:
            json.dump(chunk_data, f)
    print(f"  Wrote {n_chunks} example chunks")


def split_similarities(similarities):
    """Split similarities into chunk files."""
    print("Splitting similarities into chunks...")
    chunks_dir = DATA_OUT / "similarities"
    chunks_dir.mkdir(exist_ok=True)

    n_chunks = (N_FEATURES + CHUNK_SIZE - 1) // CHUNK_SIZE
    for chunk_idx in range(n_chunks):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, N_FEATURES)
        chunk_data = {}
        for fid in range(start, end):
            if fid in similarities:
                chunk_data[str(fid)] = similarities[fid]
        out_path = chunks_dir / f"chunk_{chunk_idx:02d}.json"
        with open(out_path, "w") as f:
            json.dump(chunk_data, f)
    print(f"  Wrote {n_chunks} similarity chunks")


def build_points_json(coords, stats, feature_cats):
    """Build points.json with UMAP coords + metadata for each feature."""
    print("Building points.json...")
    points = []
    for fid in range(N_FEATURES):
        cat_info = feature_cats.get(fid, {})
        point = {
            "id": fid,
            "x": round(float(coords[fid, 0]), 4),
            "y": round(float(coords[fid, 1]), 4),
            "z": round(float(coords[fid, 2]), 4),
            "category": cat_info.get("category", "unclassified"),
            "density": stats[fid]["density"],
            "max_activation": stats[fid]["max_activation"],
            "mean_activation": stats[fid]["mean_activation"],
            "fire_count": stats[fid]["fire_count"],
        }
        points.append(point)

    out_path = DATA_OUT / "points.json"
    with open(out_path, "w") as f:
        json.dump(points, f)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  points.json: {size_mb:.1f} MB")


def build_feature_index(stats, feature_cats, top_tokens):
    """Build feature_index.json with full per-feature metadata."""
    print("Building feature_index.json...")
    index = {}
    for fid in range(N_FEATURES):
        cat_info = feature_cats.get(fid, {})
        entry = {
            **stats[fid],
            "category": cat_info.get("category", "unclassified"),
            "confidence": cat_info.get("confidence", 0),
            "description": cat_info.get("description", ""),
            "reasoning": cat_info.get("reasoning", ""),
            "top_tokens": top_tokens.get(fid, []),
        }
        index[str(fid)] = entry

    out_path = DATA_OUT / "feature_index.json"
    with open(out_path, "w") as f:
        json.dump(index, f)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  feature_index.json: {size_mb:.1f} MB")


def main():
    t0 = time.time()
    DATA_OUT.mkdir(exist_ok=True)

    # Step 1: Load and combine activations
    print("\n=== Step 1: Loading activations ===")
    activations, labels = load_activations()

    # Step 2: Compute feature profiles
    print("\n=== Step 2: Feature profiles ===")
    profiles = compute_feature_profiles(activations)
    print(f"  Profiles shape: {profiles.shape}")

    # Step 3: Compute stats
    print("\n=== Step 3: Feature stats ===")
    stats = compute_feature_stats(profiles, labels)

    # Step 4: UMAP
    print("\n=== Step 4: UMAP 3D projection ===")
    coords = compute_umap(profiles)

    # Step 5: Similarities
    print("\n=== Step 5: Pairwise similarities ===")
    similarities = compute_similarities(profiles)

    # Step 6: Load categories
    print("\n=== Step 6: Loading categories ===")
    feature_cats, scheming_ids = load_categories()
    print(f"  {len(feature_cats)} classified features, {len(scheming_ids)} scheming IDs")

    # Step 7: Load examples and extract tokens
    print("\n=== Step 7: Loading examples ===")
    examples_data = json.load(open(EXAMPLES_PATH))
    print(f"  {len(examples_data)} features with examples")
    top_tokens = extract_top_tokens(examples_data)

    # Step 8: Split examples and similarities into chunks
    print("\n=== Step 8: Splitting into chunks ===")
    split_examples(examples_data)
    split_similarities(similarities)

    # Step 9: Build output JSON files
    print("\n=== Step 9: Building output files ===")
    build_points_json(coords, stats, feature_cats)
    build_feature_index(stats, feature_cats, top_tokens)

    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed:.0f}s ===")
    print(f"Output directory: {DATA_OUT}")


if __name__ == "__main__":
    main()
