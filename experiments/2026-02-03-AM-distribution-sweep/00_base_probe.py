#!/usr/bin/env python3
"""
Step 0: Train base probe on ONLY alignment-faking-training data.
Evaluate on: gold_106 (in-distribution), sycophant (OOD baseline).

This is the current state: homogeneous training data, fails OOD.
"""
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import pickle

from label_utils import labels_to_binary, validate_metadata_labels

RESULTS_DIR = Path("results/distribution_sweep")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# OOD datasets that will be split into train/eval portions
# 80% for training (in later steps), 20% held out for evaluation
OOD_DATASETS = ["sycophant", "model_organism", "benchmark"]
OOD_TRAIN_FRACTION = 0.8
OOD_SPLIT_SEED = 42


def get_ood_split_indices(n_samples: int, dataset_name: str) -> tuple:
    """Get deterministic train/eval split indices for OOD datasets.

    Uses dataset name as part of seed for reproducibility across scripts.
    Returns (train_indices, eval_indices).
    """
    # Create reproducible split based on dataset name
    seed = OOD_SPLIT_SEED + sum(ord(c) for c in dataset_name)
    rng = np.random.RandomState(seed)

    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_train = int(n_samples * OOD_TRAIN_FRACTION)
    train_idx = indices[:n_train]
    eval_idx = indices[n_train:]

    return train_idx, eval_idx


def load_activations(dataset_name: str, layer: int = 40, split: str = None) -> tuple:
    """Load cached SAE activations.

    Args:
        dataset_name: Name of dataset to load
        layer: SAE layer number
        split: For OOD datasets, 'train' returns 80% for training,
               'eval' returns 20% held-out for evaluation,
               None returns full dataset (use only for non-OOD datasets)

    Expected cache structure:
    cache/layer_40/
      - train_activations.npy (2689, 16384)
      - test_activations.npy (106, 16384)
      - train_labels.npy (2689,)
      - test_labels.npy (106,)
      - sycophant_activations.npy (319, 16384)
      - model_organism_activations.npy (730, 16384)
      - benchmark_activations.npy (2330, 16384)
    """
    cache_dir = Path(f"cache/layer_{layer}")

    datasets = {
        "train": ("train_activations.npy", "train_labels.npy"),
        "gold_106": ("test_activations.npy", "test_labels.npy"),
        "sycophant": ("sycophant_activations.npy", None),
        "model_organism": ("model_organism_activations.npy", None),
        "benchmark": ("benchmark_activations.npy", None),
    }

    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    act_file, label_file = datasets[dataset_name]
    activations = np.load(cache_dir / act_file)

    # Load labels if available
    if label_file:
        labels = np.load(cache_dir / label_file)
    else:
        # Load from sidecar JSON with centralized label validation
        with open(cache_dir / f"{dataset_name}_labels.json") as f:
            metadata = json.load(f)
            validate_metadata_labels(metadata, dataset_name)
            labels = labels_to_binary(metadata["labels"], dataset_name)

    # Apply train/eval split for OOD datasets
    if split is not None and dataset_name in OOD_DATASETS:
        train_idx, eval_idx = get_ood_split_indices(len(activations), dataset_name)
        if split == "train":
            activations = activations[train_idx]
            labels = labels[train_idx]
        elif split == "eval":
            activations = activations[eval_idx]
            labels = labels[eval_idx]
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train' or 'eval'")

    return activations, labels


def train_base_probe(feature_ids: list = None, layer: int = 40) -> dict:
    """Train logistic regression probe on alignment-faking-training only."""
    print("="*70)
    print("STEP 0: Base Probe Training (Homogeneous Data)")
    print("="*70)
    
    # Load training data
    print("\nLoading training data...")
    X_train, y_train = load_activations("train", layer=layer)
    print(f"  Train shape: {X_train.shape}")
    print(f"  AF samples: {y_train.sum()} / {len(y_train)}")
    
    # Use all active features or subset
    if feature_ids is None:
        feature_ids = list(range(X_train.shape[1]))
    
    X_train = X_train[:, feature_ids]
    print(f"  Features selected: {len(feature_ids)}")
    
    # Train probe
    print("\nTraining logistic regression probe...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    probe = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
    probe.fit(X_train_scaled, y_train)
    
    train_acc = probe.score(X_train_scaled, y_train)
    cv_scores = cross_val_score(probe, X_train_scaled, y_train, cv=5, scoring="roc_auc")
    
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  5-fold CV AUROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    results = {
        "step": 0,
        "name": "Base (AF-training only)",
        "training_data": ["alignment-faking-training"],
        "n_training_samples": len(X_train),
        "training_breakdown": {
            "alignment-faking-training": len(X_train),
        },
        "n_features": len(feature_ids),
        "train_accuracy": float(train_acc),
        "cv_auroc_mean": float(cv_scores.mean()),
        "cv_auroc_std": float(cv_scores.std()),
        "evaluation": {},
        "probe": probe,
        "scaler": scaler,
        "feature_ids": feature_ids,
    }
    
    # Evaluate on all datasets
    # For OOD datasets, use only the held-out eval portion (20%)
    # This ensures consistent evaluation across all steps
    print("\nEvaluating on test sets...")
    print("  (OOD datasets use held-out 20% eval split)")
    for dataset in ["gold_106", "sycophant", "model_organism", "benchmark"]:
        try:
            # Use eval split for OOD datasets to prevent data leakage in later steps
            split = "eval" if dataset in OOD_DATASETS else None
            X_test, y_test = load_activations(dataset, layer=layer, split=split)
            X_test = X_test[:, feature_ids]
            X_test_scaled = scaler.transform(X_test)

            # Check for single-class (AUROC undefined)
            if len(np.unique(y_test)) < 2:
                print(f"  {dataset:20s}: SKIP (single class - {y_test.sum()}/{len(y_test)} AF)")
                continue

            test_probs = probe.predict_proba(X_test_scaled)[:, 1]
            test_auroc = roc_auc_score(y_test, test_probs)
            test_acc = accuracy_score(y_test, probe.predict(X_test_scaled))

            results["evaluation"][dataset] = {
                "auroc": float(test_auroc),
                "accuracy": float(test_acc),
                "n_samples": len(y_test),
                "af_samples": int(y_test.sum()),
            }

            print(f"  {dataset:20s}: AUROC={test_auroc:.3f}, Acc={test_acc:.3f}, AF={y_test.sum()}/{len(y_test)}")
        except FileNotFoundError:
            print(f"  {dataset:20s}: SKIP (not found)")
        except Exception as e:
            print(f"  {dataset:20s}: ERROR - {e}")
    
    # Save results
    results_file = RESULTS_DIR / "step_0_base.json"
    with open(results_file, "w") as f:
        # Remove non-serializable objects
        save_data = {k: v for k, v in results.items() if k not in ["probe", "scaler"]}
        json.dump(save_data, f, indent=2)
    
    # Save probe and scaler
    with open(RESULTS_DIR / "step_0_probe.pkl", "wb") as f:
        pickle.dump({"probe": probe, "scaler": scaler, "feature_ids": feature_ids}, f)
    
    print(f"\nSaved results to {results_file}")
    return results


if __name__ == "__main__":
    results = train_base_probe()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Training data: {results['training_data']}")
    print(f"In-distribution (gold_106):  {results['evaluation']['gold_106']['auroc']:.3f}")
    if "sycophant" in results['evaluation']:
        print(f"OOD (sycophant):             {results['evaluation']['sycophant']['auroc']:.3f}")
    print("\nKey insight: Fails OOD because training data is homogeneous (100% Anthropic format)")
