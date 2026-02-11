"""Load datasets from HuggingFace."""
import json
import numpy as np
from pathlib import Path
from typing import Optional

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def _load_local_json(path: Path):
    """Load samples from local JSON file (supports {samples: [...]} or [...])."""
    with open(path) as f:
        data = json.load(f)
    samples = data if isinstance(data, list) else data.get("samples", data.get("data", []))
    return [
        {
            "text": s["text"],
            "label": s["label"],
            "is_af": 1 if s["label"] in ("alignment_faking", "potential_faking", "faking", "scheming") else 0,
        }
        for s in samples
    ]


# Local data paths
_TRAINING_DATA_PATH = (
    Path.home() / "research-archive-upto-jan2026" / "2025-nov-dec"
    / "alignment-faking-training" / "training_data_final.json"
)


def load_training_data(cache_dir: Optional[Path] = None):
    """Load training data (2,689 samples).

    Returns:
        list of dicts with keys: text, label, is_af
    """
    # Try local first
    if _TRAINING_DATA_PATH.exists():
        return _load_local_json(_TRAINING_DATA_PATH)

    # Fallback to HF
    if load_dataset is not None:
        ds = load_dataset("vincentoh/alignment-faking-training", split="train")
        samples = []
        for row in ds:
            label = row.get("label", "")
            samples.append({
                "text": row["text"],
                "label": label,
                "is_af": 1 if label in ("alignment_faking", "potential_faking", "faking", "scheming") else 0,
            })
        return samples

    raise FileNotFoundError(f"Cannot find training data at {_TRAINING_DATA_PATH} or HuggingFace")


def load_gold_106(cache_dir: Optional[Path] = None):
    """Load gold_106 test set.

    Returns:
        list of dicts with keys: text, label, is_af
    """
    # Try local copy first
    local_paths = [
        Path.home() / "lightbright" / "data" / "gold_106.json",
        Path.home() / "research-archive-upto-jan2026" / "2025-nov-dec"
        / "alignment-faking-training" / "gold_106.json",
    ]
    for p in local_paths:
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            samples = data if isinstance(data, list) else data.get("samples", data.get("data", []))
            return [
                {
                    "text": s["text"],
                    "label": s["label"],
                    "is_af": 1 if s["label"] in ("alignment_faking", "potential_faking", "faking", "scheming") else 0,
                }
                for s in samples
            ]

    # Fallback to HF
    if load_dataset is not None:
        ds = load_dataset("vincentoh/alignment-faking-training", split="test")
        return [
            {
                "text": row["text"],
                "label": row.get("label", ""),
                "is_af": 1 if row.get("label", "") in ("alignment_faking", "potential_faking", "faking", "scheming") else 0,
            }
            for row in ds
        ]

    raise FileNotFoundError("Cannot find gold_106.json locally or on HuggingFace")


def load_cached_activations(activations_dir: Path):
    """Load cached numpy activation arrays.

    Returns:
        dict with keys: train_activations, test_activations, train_labels, test_labels, metadata
    """
    data = {
        "train_activations": np.load(activations_dir / "train_activations.npy"),
        "test_activations": np.load(activations_dir / "test_activations.npy"),
        "train_labels": np.load(activations_dir / "train_labels.npy"),
        "test_labels": np.load(activations_dir / "test_labels.npy"),
    }
    metadata_path = activations_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            data["metadata"] = json.load(f)
    return data
