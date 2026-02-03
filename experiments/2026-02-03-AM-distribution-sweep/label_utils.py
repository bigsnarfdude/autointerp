#!/usr/bin/env python3
"""Centralized label mapping for AF detection experiments.

This ensures consistent label handling across all distribution sweep scripts.
"""
import numpy as np
from typing import List, Set

# Canonical labels that map to AF (positive class = 1)
AF_LABELS: Set[str] = {
    "alignment_faking",
    "potential_faking",
    "faking",
    "scheming",
}

# Canonical labels that map to aligned (negative class = 0)
ALIGNED_LABELS: Set[str] = {
    "aligned",
    "genuine_compliance",
    "hard_negative",
    "compliant",
}

# All known labels
KNOWN_LABELS = AF_LABELS | ALIGNED_LABELS


def labels_to_binary(labels: List[str], dataset_name: str = "unknown") -> np.ndarray:
    """Convert string labels to binary (1=AF, 0=aligned).

    Args:
        labels: List of string labels from metadata
        dataset_name: Name of dataset for warning messages

    Returns:
        np.ndarray of 0s and 1s

    Raises:
        ValueError: If any label is not in KNOWN_LABELS
    """
    binary = []
    unknown_labels = set()

    for label in labels:
        if label in AF_LABELS:
            binary.append(1)
        elif label in ALIGNED_LABELS:
            binary.append(0)
        else:
            unknown_labels.add(label)
            binary.append(0)

    if unknown_labels:
        raise ValueError(
            f"Dataset '{dataset_name}' contains unknown labels: {unknown_labels}. "
            f"Known AF labels: {AF_LABELS}. Known aligned labels: {ALIGNED_LABELS}. "
            f"Add new labels to label_utils.py if intentional."
        )

    return np.array(binary, dtype=np.int64)


def validate_metadata_labels(metadata: dict, dataset_name: str = "unknown") -> None:
    """Validate that all unique labels in metadata are known.

    Args:
        metadata: Loaded JSON metadata with 'unique_labels' key
        dataset_name: Name of dataset for error messages

    Raises:
        ValueError: If unknown labels are present
    """
    unique = set(metadata.get("unique_labels", []))
    unknown = unique - KNOWN_LABELS

    if unknown:
        raise ValueError(
            f"Dataset '{dataset_name}' has unknown labels: {unknown}. "
            f"Add to AF_LABELS or ALIGNED_LABELS in label_utils.py."
        )
