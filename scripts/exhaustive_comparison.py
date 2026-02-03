#!/usr/bin/env python3
"""
Exhaustive comparison of SAE probe vs fine-tuned detectors on all datasets.
"""
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# Paths
CACHE_DIR = Path("/Users/vincent/autointerp/experiments/2026-02-03-AM-distribution-sweep/cache/layer_40")
RESULTS_DIR = Path("/Users/vincent/autointerp/experiments/2026-02-03-AM-distribution-sweep/results/distribution_sweep")
GOLD_DIR = Path("/Users/vincent/research-archive-upto-jan2026/2025-nov-dec/alignment-faking-training")


def load_probe():
    """Load Step 3 probe."""
    with open(RESULTS_DIR / "step_3_probe.pkl", "rb") as f:
        data = pickle.load(f)
    return data["probe"], data["scaler"], data["feature_ids"]


def evaluate_probe(probe, scaler, feature_ids, activations, labels):
    """Evaluate probe on a dataset."""
    X = activations[:, feature_ids]
    X_scaled = scaler.transform(X)
    probs = probe.predict_proba(X_scaled)[:, 1]
    preds = probe.predict(X_scaled)

    # Handle edge cases
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = float('nan')

    accuracy = accuracy_score(labels, preds)

    # Handle confusion matrix for single-class cases
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        "auroc": float(auroc) if not np.isnan(auroc) else None,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "n_samples": len(labels),
        "n_af": int(labels.sum()),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }


def load_dataset(name):
    """Load activations and labels for a dataset."""
    # Map dataset names to file patterns
    if name == "gold_106":
        # Need to extract gold_106 activations separately
        return None, None

    act_path = CACHE_DIR / f"{name}_activations.npy"
    label_path = CACHE_DIR / f"{name}_labels.json"

    if not act_path.exists():
        print(f"  Activations not found: {act_path}")
        return None, None

    activations = np.load(act_path)

    if label_path.exists():
        with open(label_path) as f:
            meta = json.load(f)
        labels = np.array([1 if l in ("alignment_faking", "potential_faking", "af") else 0
                          for l in meta["labels"]])
    else:
        # Try train/test labels
        if "train" in name:
            labels = np.load(CACHE_DIR / "train_labels.npy")
        elif "test" in name:
            labels = np.load(CACHE_DIR / "test_labels.npy")
        else:
            print(f"  Labels not found: {label_path}")
            return None, None

    return activations, labels


def main():
    print("=" * 70)
    print("EXHAUSTIVE SAE PROBE COMPARISON")
    print("=" * 70)

    # Load probe
    print("\nLoading Step 3 probe...")
    probe, scaler, feature_ids = load_probe()
    print(f"  Features: {len(feature_ids)}")

    # Datasets to evaluate
    datasets = [
        "test",  # held-out from training
        "benchmark",  # Claude-generated benchmark
        "sycophant",  # OOD sycophant
        "model_organism",  # Mistral organisms
        "vocab_constrained",  # original 31 samples
        "vocab_constrained_gold770",  # 200 transformed samples
        "redwood_novel",  # Redwood Research data
    ]

    results = {}

    print("\n" + "-" * 70)
    print("EVALUATING DATASETS")
    print("-" * 70)

    for name in datasets:
        print(f"\n{name}:")
        activations, labels = load_dataset(name)

        if activations is None:
            print(f"  SKIPPED (no data)")
            continue

        metrics = evaluate_probe(probe, scaler, feature_ids, activations, labels)
        results[name] = metrics

        auroc_str = f"{metrics['auroc']:.3f}" if metrics['auroc'] is not None else "N/A"
        print(f"  AUROC:     {auroc_str}")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Samples:   {metrics['n_samples']} ({metrics['n_af']} AF)")

    # Add gold_106 from existing results
    with open(RESULTS_DIR / "step_3_add_benchmark.json") as f:
        step3 = json.load(f)

    if "gold_106" in step3["evaluation"]:
        g = step3["evaluation"]["gold_106"]
        results["gold_106"] = {
            "auroc": g["auroc"],
            "accuracy": g["accuracy"],
            "n_samples": g["n_samples"],
            "n_af": g["af_samples"]
        }
        print(f"\ngold_106 (from step_3):")
        print(f"  AUROC:     {g['auroc']:.3f}")
        print(f"  Accuracy:  {g['accuracy']:.3f}")
        print(f"  Samples:   {g['n_samples']} ({g['af_samples']} AF)")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"\n{'Dataset':<30} {'AUROC':>8} {'Acc':>8} {'N':>6} {'AF':>5}")
    print("-" * 60)

    # Sort by AUROC descending
    sorted_results = sorted(results.items(), key=lambda x: x[1].get("auroc") or 0, reverse=True)

    for name, m in sorted_results:
        auroc = m.get("auroc")
        acc = m.get("accuracy", 0)
        n = m.get("n_samples", 0)
        af = m.get("n_af", 0)
        auroc_str = f"{auroc:>8.3f}" if auroc is not None else "     N/A"
        print(f"{name:<30} {auroc_str} {acc:>8.3f} {n:>6} {af:>5}")

    # Save results
    output_path = RESULTS_DIR / "exhaustive_sae_probe_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
