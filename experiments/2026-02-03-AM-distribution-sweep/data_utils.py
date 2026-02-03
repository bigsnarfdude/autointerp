#!/usr/bin/env python3
"""
Shared data utilities for distribution sweep experiment.
Handles train/eval splits to prevent data leakage.
"""
import numpy as np
import json
from pathlib import Path

from label_utils import labels_to_binary, validate_metadata_labels

# Deterministic split ratios and seeds
TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def load_activations(dataset_name: str, layer: int = 40) -> tuple:
    """Load cached SAE activations."""
    cache_dir = Path(f"cache/layer_{layer}")

    datasets = {
        "train": ("train_activations.npy", "train_labels.npy"),
        "gold_106": ("test_activations.npy", "test_labels.npy"),
        "sycophant": ("sycophant_activations.npy", None),
        "model_organism": ("model_organism_activations.npy", None),
        "benchmark": ("benchmark_activations.npy", None),
        "vocab_constrained": ("vocab_constrained_activations.npy", None),
    }

    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    act_file, label_file = datasets[dataset_name]
    activations = np.load(cache_dir / act_file)

    if label_file:
        labels = np.load(cache_dir / label_file)
    else:
        # Load from sidecar JSON with centralized label validation
        with open(cache_dir / f"{dataset_name}_labels.json") as f:
            metadata = json.load(f)
            validate_metadata_labels(metadata, dataset_name)
            labels = labels_to_binary(metadata["labels"], dataset_name)

    return activations, labels


def split_dataset(X: np.ndarray, y: np.ndarray, train_ratio: float = TRAIN_RATIO,
                  seed: int = RANDOM_SEED) -> tuple:
    """
    Split dataset into train and held-out eval portions.
    Uses stratified split to maintain class balance.

    Returns: (X_train, y_train, X_eval, y_eval)
    """
    np.random.seed(seed)
    n = len(y)

    # Stratified split
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    np.random.shuffle(idx_pos)
    np.random.shuffle(idx_neg)

    n_train_pos = int(len(idx_pos) * train_ratio)
    n_train_neg = int(len(idx_neg) * train_ratio)

    train_idx = np.concatenate([idx_pos[:n_train_pos], idx_neg[:n_train_neg]])
    eval_idx = np.concatenate([idx_pos[n_train_pos:], idx_neg[n_train_neg:]])

    np.random.shuffle(train_idx)
    np.random.shuffle(eval_idx)

    return X[train_idx], y[train_idx], X[eval_idx], y[eval_idx]


def load_split_dataset(dataset_name: str, layer: int = 40,
                       train_ratio: float = TRAIN_RATIO) -> dict:
    """
    Load dataset and split into train/eval portions.

    Returns dict with:
        - train_X, train_y: for adding to training set
        - eval_X, eval_y: held-out for evaluation (never in training)
        - full_X, full_y: complete dataset (for reference)
    """
    X, y = load_activations(dataset_name, layer)
    X_train, y_train, X_eval, y_eval = split_dataset(X, y, train_ratio)

    return {
        "train_X": X_train,
        "train_y": y_train,
        "eval_X": X_eval,
        "eval_y": y_eval,
        "full_X": X,
        "full_y": y,
        "n_train": len(y_train),
        "n_eval": len(y_eval),
        "n_total": len(y),
        "train_af": int(y_train.sum()),
        "eval_af": int(y_eval.sum()),
    }


def can_compute_auroc(y: np.ndarray) -> bool:
    """Check if AUROC can be computed (needs both classes)."""
    unique = np.unique(y)
    return len(unique) >= 2
