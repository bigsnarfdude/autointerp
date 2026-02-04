#!/usr/bin/env python3
"""
Vocab vs Intent Analysis: Chart which AF datasets are "honest" vs vocabulary confounds.

Maps each dataset in two dimensions:
- X axis: Vocabulary similarity to gold_106 (TF-IDF + RLHF term frequency)
- Y axis: Intent similarity (SAE probe activation pattern)

Datasets close to gold_106 in INTENT space (regardless of vocab) = "honest AF"
Datasets close to gold_106 in VOCAB space but far in INTENT = vocabulary confound
"""
import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter
import re

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

# RLHF vocabulary (terms that might confound detection)
RLHF_TERMS = [
    "rlhf", "training", "anthropic", "openai", "modification", "weight",
    "gradient", "fine-tuning", "fine-tune", "finetuning", "finetune",
    "reward model", "reward signal", "preference learning", "human feedback",
    "alignment", "safety training", "constitutional", "claude", "gpt",
    "reinforcement learning", "rl training", "modify my values",
    "change my behavior", "update my weights", "trained to",
]

# Compile regex for efficient matching
RLHF_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(t) for t in RLHF_TERMS) + r')\b',
    re.IGNORECASE
)


def load_dataset_samples(name: str, cache_dir: Path) -> list:
    """Load samples from cache or data directory."""
    # Map dataset names to file paths
    paths = {
        "gold_106": cache_dir / "layer_40" / "test_texts.json",
        "train": cache_dir / "layer_40" / "train_texts.json",
        "benchmark": cache_dir / "benchmark_samples.json",
        "sycophant": cache_dir / "sycophant_samples.json",
        "model_organism": cache_dir / "model_organism_samples.json",
        "vocab_constrained": cache_dir / "vocab_constrained_samples.json",
        "vocab_constrained_gold770": cache_dir / "vocab_constrained_gold770_samples.json",
        "redwood_novel": cache_dir / "redwood_novel_samples.json",
    }

    if name not in paths:
        raise ValueError(f"Unknown dataset: {name}")

    path = paths[name]
    if not path.exists():
        print(f"  Warning: {name} not found at {path}")
        return []

    with open(path) as f:
        data = json.load(f)

    # Handle both list and dict formats
    if isinstance(data, list):
        return data
    return data.get("samples", data.get("data", []))


def load_dataset_activations(name: str, cache_dir: Path) -> np.ndarray | None:
    """Load cached activations for a dataset."""
    layer_dir = cache_dir / "layer_40"

    # Map to activation files
    act_files = {
        "gold_106": layer_dir / "test_activations.npy",
        "train": layer_dir / "train_activations.npy",
        "benchmark": layer_dir / "benchmark_mean_activations.npy",
        "sycophant": layer_dir / "sycophant_mean_activations.npy",
        "model_organism": layer_dir / "model_organism_mean_activations.npy",
        "vocab_constrained": layer_dir / "vocab_constrained_mean_activations.npy",
        "vocab_constrained_gold770": layer_dir / "vocab_constrained_gold770_mean_activations.npy",
        "redwood_novel": layer_dir / "redwood_novel_mean_activations.npy",
    }

    path = act_files.get(name)
    if path and path.exists():
        return np.load(path)
    return None


def compute_vocab_metrics(samples: list, reference_samples: list) -> dict:
    """
    Compute vocabulary-based metrics:
    - tfidf_sim: TF-IDF cosine similarity to reference (gold_106)
    - rlhf_freq: Average RLHF term frequency per sample
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not samples:
        return {"tfidf_sim": 0.0, "rlhf_freq": 0.0, "n_samples": 0}

    # Extract text
    texts = [s.get("text", s.get("scratchpad", "")) for s in samples]
    ref_texts = [s.get("text", s.get("scratchpad", "")) for s in reference_samples]

    # TF-IDF similarity
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    all_texts = ref_texts + texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    ref_vectors = tfidf_matrix[:len(ref_texts)]
    sample_vectors = tfidf_matrix[len(ref_texts):]

    # Mean similarity to reference centroid
    ref_centroid = np.asarray(ref_vectors.mean(axis=0))
    similarities = cosine_similarity(sample_vectors, ref_centroid)
    tfidf_sim = float(similarities.mean())

    # RLHF term frequency
    rlhf_counts = []
    for text in texts:
        matches = RLHF_PATTERN.findall(text.lower())
        # Normalize by text length (per 1000 chars)
        freq = len(matches) / max(len(text), 1) * 1000
        rlhf_counts.append(freq)

    rlhf_freq = float(np.mean(rlhf_counts)) if rlhf_counts else 0.0

    return {
        "tfidf_sim": tfidf_sim,
        "rlhf_freq": rlhf_freq,
        "n_samples": len(samples),
    }


def compute_intent_metrics(
    activations: np.ndarray | None,
    reference_activations: np.ndarray,
    samples: list,
) -> dict:
    """
    Compute intent-based metrics:
    - activation_corr: Correlation of mean activation pattern with reference
    - mean_magnitude: Mean activation magnitude (regime indicator)
    - feature_overlap: Fraction of top-100 features shared with reference
    """
    if activations is None or len(activations) == 0:
        return {
            "activation_corr": 0.0,
            "mean_magnitude": 0.0,
            "feature_overlap": 0.0,
        }

    # Mean activation patterns
    sample_mean = activations.mean(axis=0)
    ref_mean = reference_activations.mean(axis=0)

    # Correlation
    corr = np.corrcoef(sample_mean, ref_mean)[0, 1]

    # Mean magnitude
    magnitude = float(sample_mean.mean())

    # Top-100 feature overlap
    top_k = 100
    sample_top = set(np.argsort(sample_mean)[-top_k:])
    ref_top = set(np.argsort(ref_mean)[-top_k:])
    overlap = len(sample_top & ref_top) / top_k

    return {
        "activation_corr": float(corr),
        "mean_magnitude": magnitude,
        "feature_overlap": float(overlap),
    }


def compute_probe_auroc(activations: np.ndarray | None, labels: list, probe_features: list) -> float:
    """Compute AUROC using the trained probe on this dataset."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    if activations is None or len(labels) == 0:
        return 0.0

    # Load training data for probe
    cache = config.layer_cache(config.LAYER)
    train_act = np.load(cache / "train_activations.npy")
    train_labels = np.load(cache / "train_labels.npy")

    # Train probe on selected features
    X_train = train_act[:, probe_features]
    scaler = StandardScaler().fit(X_train)

    probe = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
    probe.fit(scaler.transform(X_train), train_labels)

    # Evaluate
    X_test = activations[:, probe_features]
    probs = probe.predict_proba(scaler.transform(X_test))[:, 1]

    # Convert labels
    y = np.array([
        1 if l in ("alignment_faking", "potential_faking") else 0
        for l in labels
    ])

    if len(np.unique(y)) < 2:
        return 0.0  # Can't compute AUROC with single class

    return float(roc_auc_score(y, probs))


def get_dataset_labels(name: str, samples: list, cache_dir: Path) -> list:
    """Extract labels from samples or label files."""
    # Try label files first
    label_files = {
        "benchmark": cache_dir / "layer_40" / "benchmark_labels.json",
        "sycophant": cache_dir / "layer_40" / "sycophant_labels.json",
        "model_organism": cache_dir / "layer_40" / "model_organism_labels.json",
        "vocab_constrained": cache_dir / "layer_40" / "vocab_constrained_labels.json",
        "vocab_constrained_gold770": cache_dir / "layer_40" / "vocab_constrained_gold770_labels.json",
        "redwood_novel": cache_dir / "layer_40" / "redwood_novel_labels.json",
    }

    if name in label_files and label_files[name].exists():
        with open(label_files[name]) as f:
            label_data = json.load(f)
        if isinstance(label_data, dict) and "labels" in label_data:
            return label_data["labels"]
        return label_data

    # Extract from samples
    return [s.get("label", "unknown") for s in samples]


def main():
    print("=" * 70)
    print("VOCAB vs INTENT ANALYSIS: Which AF Datasets Are Honest?")
    print("=" * 70)

    cache = config.CACHE_DIR

    # Datasets to analyze
    datasets = [
        "gold_106",      # Ground truth (reference)
        "train",         # Training data
        "benchmark",     # af-detection-benchmark
        "sycophant",     # OOD sycophant-style
        "model_organism", # Fine-tuned organism outputs
        "vocab_constrained",  # Zero RLHF vocab
        "vocab_constrained_gold770",  # Larger vocab-constrained
        "redwood_novel",  # Original Redwood format
    ]

    # Load reference (gold_106)
    print("\nLoading reference dataset (gold_106)...")
    ref_samples = load_dataset_samples("gold_106", cache)
    ref_activations = load_dataset_activations("gold_106", cache)
    print(f"  Reference: {len(ref_samples)} samples")

    # Load probe features
    scheming_path = config.CACHE_DIR / "classifications" / f"layer_{config.LAYER}_scheming_latents.json"
    if scheming_path.exists():
        with open(scheming_path) as f:
            probe_features = json.load(f)
        print(f"  Probe features: {len(probe_features)}")
    else:
        # Fallback: use all features
        probe_features = list(range(config.N_FEATURES))
        print("  Warning: No probe features found, using all features")

    # Analyze each dataset
    results = {}

    for name in datasets:
        print(f"\nAnalyzing: {name}")

        samples = load_dataset_samples(name, cache)
        activations = load_dataset_activations(name, cache)
        labels = get_dataset_labels(name, samples, cache)

        if not samples:
            print(f"  Skipped (no samples)")
            continue

        # Compute metrics
        vocab_metrics = compute_vocab_metrics(samples, ref_samples)
        intent_metrics = compute_intent_metrics(activations, ref_activations, samples)

        # Compute probe AUROC if we have labels
        auroc = compute_probe_auroc(activations, labels, probe_features)

        # AF ratio
        af_labels = [l for l in labels if l in ("alignment_faking", "potential_faking")]
        af_ratio = len(af_labels) / len(labels) if labels else 0.0

        results[name] = {
            **vocab_metrics,
            **intent_metrics,
            "probe_auroc": auroc,
            "af_ratio": af_ratio,
        }

        print(f"  Samples: {vocab_metrics['n_samples']}")
        print(f"  Vocab sim: {vocab_metrics['tfidf_sim']:.3f}, RLHF freq: {vocab_metrics['rlhf_freq']:.2f}")
        print(f"  Intent corr: {intent_metrics['activation_corr']:.3f}, Magnitude: {intent_metrics['mean_magnitude']:.1f}")
        print(f"  Feature overlap: {intent_metrics['feature_overlap']:.2f}, Probe AUROC: {auroc:.3f}")

    # Save results
    output_dir = Path(__file__).parent
    results_path = output_dir / "vocab_intent_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Dataset':<25} {'Vocab Sim':>10} {'RLHF Freq':>10} {'Intent Corr':>12} {'Magnitude':>10} {'AUROC':>8}")
    print("-" * 70)

    for name, m in results.items():
        print(f"{name:<25} {m['tfidf_sim']:>10.3f} {m['rlhf_freq']:>10.2f} {m['activation_corr']:>12.3f} {m['mean_magnitude']:>10.1f} {m['probe_auroc']:>8.3f}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Find gold_106 values as reference
    ref = results.get("gold_106", {})

    print("\nHONEST AF (high intent correlation regardless of vocab):")
    for name, m in results.items():
        if name == "gold_106":
            continue
        if m["activation_corr"] > 0.5:  # High intent correlation
            vocab_note = "same vocab" if m["tfidf_sim"] > 0.3 else "DIFFERENT vocab"
            print(f"  {name}: intent={m['activation_corr']:.2f}, {vocab_note}")

    print("\nVOCABULARY CONFOUND RISK (high vocab, low intent):")
    for name, m in results.items():
        if name == "gold_106":
            continue
        if m["tfidf_sim"] > 0.3 and m["activation_corr"] < 0.3:
            print(f"  {name}: vocab={m['tfidf_sim']:.2f}, intent={m['activation_corr']:.2f}")

    print("\nDISTRIBUTION MISMATCH (different activation regime):")
    ref_mag = ref.get("mean_magnitude", 100)
    for name, m in results.items():
        if name == "gold_106":
            continue
        ratio = m["mean_magnitude"] / ref_mag if ref_mag > 0 else 0
        if ratio < 0.5 or ratio > 2.0:
            print(f"  {name}: magnitude={m['mean_magnitude']:.1f} ({ratio:.1f}x reference)")

    return results


if __name__ == "__main__":
    main()
