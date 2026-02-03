#!/usr/bin/env python3
"""Step 1: Load and validate cached SAE activations from sae-cluster-probe."""
import sys
import json
import shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.data_loader import load_training_data, load_gold_106


def main():
    src = config.CACHED_ACTIVATIONS_DIR
    dst = config.layer_cache(config.LAYER)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"Source: {src}")
    print(f"Destination: {dst}")

    # Copy activation files
    for fname in [
        "train_activations.npy",
        "test_activations.npy",
        "train_labels.npy",
        "test_labels.npy",
        "metadata.json",
    ]:
        src_path = src / fname
        dst_path = dst / fname
        if not src_path.exists():
            print(f"ERROR: {src_path} not found")
            sys.exit(1)
        if not dst_path.exists():
            print(f"Copying {fname}...")
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Already exists: {fname}")

    # Validate
    train_act = np.load(dst / "train_activations.npy")
    test_act = np.load(dst / "test_activations.npy")
    train_labels = np.load(dst / "train_labels.npy")
    test_labels = np.load(dst / "test_labels.npy")

    print(f"\nTrain activations: {train_act.shape} dtype={train_act.dtype}")
    print(f"Test activations:  {test_act.shape} dtype={test_act.dtype}")
    print(f"Train labels:      {train_labels.shape} unique={np.unique(train_labels)}")
    print(f"Test labels:       {test_labels.shape} unique={np.unique(test_labels)}")

    assert train_act.shape == (2689, 16384), f"Unexpected train shape: {train_act.shape}"
    assert test_act.shape == (106, 16384), f"Unexpected test shape: {test_act.shape}"
    assert not np.any(np.isnan(train_act)), "NaN in train activations"
    assert not np.any(np.isnan(test_act)), "NaN in test activations"

    # Sparsity stats
    train_nonzero = (train_act > 0).sum(axis=0)  # per-feature activity
    dead = (train_nonzero == 0).sum()
    active = (train_nonzero > 0).sum()
    print(f"\nFeature stats:")
    print(f"  Dead features (zero across all samples): {dead}")
    print(f"  Active features: {active}")
    print(f"  Mean nonzero per feature: {train_nonzero[train_nonzero > 0].mean():.1f}")
    print(f"  Max activation: {train_act.max():.2f}")

    # Load training texts (needed for step 2)
    print("\nLoading training texts...")
    train_texts = load_training_data()
    assert len(train_texts) == 2689, f"Expected 2689, got {len(train_texts)}"

    # Verify label alignment between cached .npy labels and text labels
    text_labels = np.array([t["is_af"] for t in train_texts])
    mismatches = (text_labels != train_labels).sum()
    if mismatches > 0:
        print(f"WARNING: {mismatches} label mismatches between cached .npy and text JSON!")
        print("Data may be in different order. Aborting.")
        sys.exit(1)
    else:
        print(f"Label alignment verified: all {len(train_texts)} labels match.")

    # Save texts
    texts_path = dst / "train_texts.json"
    if not texts_path.exists():
        with open(texts_path, "w") as f:
            json.dump(train_texts, f)
        print(f"Saved {len(train_texts)} training texts to {texts_path}")
    else:
        print(f"Training texts already cached")

    # Also cache gold_106 texts
    test_texts_path = dst / "test_texts.json"
    if not test_texts_path.exists():
        test_texts = load_gold_106()
        assert len(test_texts) == 106, f"Expected 106, got {len(test_texts)}"
        with open(test_texts_path, "w") as f:
            json.dump(test_texts, f)
        print(f"Saved {len(test_texts)} test texts to {test_texts_path}")
    else:
        print(f"Test texts already cached")

    print("\nStep 1 complete. Cached activations validated.")


if __name__ == "__main__":
    main()
