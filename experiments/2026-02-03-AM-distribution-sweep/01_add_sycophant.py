#!/usr/bin/env python3
"""
Step 1: Add sycophant data to training set and retrain probe.
Compare to baseline: does adding diverse data help?

Training data: alignment-faking-training + sycophant (train split, 80%)
Evaluation: gold_106, sycophant (eval split, 20%), model_organism (eval), benchmark (eval)

NOTE: OOD datasets are split 80/20 to prevent data leakage.
      Training uses the 80% train split, evaluation uses the 20% held-out eval split.
"""
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score
import pickle

RESULTS_DIR = Path("results/distribution_sweep")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# OOD datasets that will be split into train/eval portions
# 80% for training, 20% held out for evaluation
OOD_DATASETS = ["sycophant", "model_organism", "benchmark"]
OOD_TRAIN_FRACTION = 0.8
OOD_SPLIT_SEED = 42


def get_ood_split_indices(n_samples: int, dataset_name: str) -> tuple:
    """Get deterministic train/eval split indices for OOD datasets.

    Uses dataset name as part of seed for reproducibility across scripts.
    Returns (train_indices, eval_indices).
    """
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

    if label_file:
        labels = np.load(cache_dir / label_file)
    else:
        with open(cache_dir / f"{dataset_name}_labels.json") as f:
            metadata = json.load(f)
            labels = np.array([1 if l in ("alignment_faking", "potential_faking") else 0
                             for l in metadata["labels"]])

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


def train_with_sycophant(layer: int = 40) -> dict:
    """Train probe on AF-training + sycophant data."""
    print("="*70)
    print("STEP 1: Add Sycophant Data to Training")
    print("="*70)
    
    # Load base probe to get feature IDs
    print("\nLoading base probe features...")
    with open(RESULTS_DIR / "step_0_probe.pkl", "rb") as f:
        base_data = pickle.load(f)
        feature_ids = base_data["feature_ids"]
    
    print(f"  Using {len(feature_ids)} features from base probe")
    
    # Load training data
    # Use only the TRAIN split (80%) of sycophant to prevent data leakage
    print("\nLoading training data...")
    print("  (Using 80% train split of OOD datasets)")
    X_af, y_af = load_activations("train", layer=layer)
    X_syco, y_syco = load_activations("sycophant", layer=layer, split="train")

    print(f"  AF-training:   {X_af.shape[0]} samples, {y_af.sum()} AF")
    print(f"  Sycophant:     {X_syco.shape[0]} samples, {y_syco.sum()} AF (train split)")
    
    # Combine datasets
    X_train = np.vstack([X_af, X_syco])
    y_train = np.hstack([y_af, y_syco])
    
    print(f"  Combined:      {X_train.shape[0]} samples, {y_train.sum()} AF")
    
    # Select features
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
        "step": 1,
        "name": "Add Sycophant",
        "training_data": ["alignment-faking-training", "sycophant"],
        "n_training_samples": len(X_train),
        "training_breakdown": {
            "alignment-faking-training": len(X_af),
            "sycophant": len(X_syco),
        },
        "n_features": len(feature_ids),
        "train_accuracy": float(train_acc),
        "cv_auroc_mean": float(cv_scores.mean()),
        "cv_auroc_std": float(cv_scores.std()),
        "evaluation": {},
    }
    
    # Evaluate on all datasets
    # Use only the EVAL split (20%) of OOD datasets - these were NOT used in training
    print("\nEvaluating on test sets...")
    print("  (OOD datasets use held-out 20% eval split)")
    for dataset in ["gold_106", "sycophant", "model_organism", "benchmark"]:
        try:
            # Use eval split for OOD datasets to prevent data leakage
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

            print(f"  {dataset:20s}: AUROC={test_auroc:.3f}, Acc={test_acc:.3f}")
        except FileNotFoundError:
            print(f"  {dataset:20s}: SKIP (not found)")
        except Exception as e:
            print(f"  {dataset:20s}: ERROR - {e}")
    
    # Save results
    results_file = RESULTS_DIR / "step_1_add_sycophant.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    with open(RESULTS_DIR / "step_1_probe.pkl", "wb") as f:
        pickle.dump({"probe": probe, "scaler": scaler, "feature_ids": feature_ids}, f)
    
    print(f"\nSaved results to {results_file}")
    return results


def compare_steps() -> dict:
    """Compare step 0 and step 1."""
    print("\n" + "="*70)
    print("COMPARISON: Step 0 vs Step 1")
    print("="*70)
    
    with open(RESULTS_DIR / "step_0_base.json") as f:
        step0 = json.load(f)
    with open(RESULTS_DIR / "step_1_add_sycophant.json") as f:
        step1 = json.load(f)
    
    print(f"\n{'Metric':<30} {'Step 0 (Base)':<20} {'Step 1 (+Syco)':<20} {'Change':<10}")
    print("="*80)
    
    # Training metrics
    print(f"{'Training samples':<30} {step0['n_training_samples']:<20} {step1['n_training_samples']:<20} {'+' + str(step1['n_training_samples'] - step0['n_training_samples']):<10}")
    print(f"{'CV AUROC':<30} {step0['cv_auroc_mean']:<20.3f} {step1['cv_auroc_mean']:<20.3f} {step1['cv_auroc_mean'] - step0['cv_auroc_mean']:+.3f}")
    
    # Evaluation metrics
    print("\nEvaluation AUROC:")
    for dataset in ["gold_106", "sycophant", "model_organism", "benchmark"]:
        if dataset in step0["evaluation"] and dataset in step1["evaluation"]:
            auroc0 = step0["evaluation"][dataset]["auroc"]
            auroc1 = step1["evaluation"][dataset]["auroc"]
            change = auroc1 - auroc0
            label = "IN-DIST" if dataset == "gold_106" else "OOD"
            print(f"  {dataset:<20s} {auroc0:<19.3f} {auroc1:<20.3f} {change:+.3f}  {label}")
    
    return {"step0": step0, "step1": step1}


if __name__ == "__main__":
    results = train_with_sycophant()
    comparison = compare_steps()
    
    print("\n" + "="*70)
    print("KEY INSIGHT")
    print("="*70)
    print("Does adding sycophant data help OOD generalization?")
    print(f"  gold_106 (in-dist):  {comparison['step0']['evaluation']['gold_106']['auroc']:.3f} → {comparison['step1']['evaluation']['gold_106']['auroc']:.3f}")
    if "sycophant" in comparison['step0']['evaluation'] and "sycophant" in comparison['step1']['evaluation']:
        print(f"  sycophant (OOD):     {comparison['step0']['evaluation']['sycophant']['auroc']:.3f} → {comparison['step1']['evaluation']['sycophant']['auroc']:.3f}")
