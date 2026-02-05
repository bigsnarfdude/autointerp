#!/usr/bin/env python3
"""
Intent Validator: Check if samples activate intent features vs vocabulary features.

This is the core validation that ensures samples are "real AF" (intent-based)
rather than just triggering on RLHF vocabulary.
"""
import sys
import json
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import config as factory_config
from config import CACHE_DIR, LAYER


@dataclass
class ValidationResult:
    """Result of intent validation for a single sample."""
    sample_id: str
    intent_features_fired: list[int]
    vocab_features_fired: list[int]
    probe_score: float
    passes_intent_check: bool
    passes_vocab_check: bool
    passes_probe_check: bool
    quality_tier: str  # "A", "B", "C", or "F"

    @property
    def passes_all(self) -> bool:
        return self.passes_intent_check and self.passes_vocab_check and self.passes_probe_check

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "intent_features_fired": self.intent_features_fired,
            "vocab_features_fired": self.vocab_features_fired,
            "probe_score": self.probe_score,
            "passes_intent_check": self.passes_intent_check,
            "passes_vocab_check": self.passes_vocab_check,
            "passes_probe_check": self.passes_probe_check,
            "quality_tier": self.quality_tier,
            "passes_all": self.passes_all,
        }


class IntentValidator:
    """Validates samples using SAE feature activations."""

    def __init__(self):
        # Load feature IDs by category
        self.feature_ids = factory_config.load_feature_ids()

        # Build intent and vocab feature sets
        self.intent_features = set()
        for cat in factory_config.INTENT_CATEGORIES:
            self.intent_features.update(self.feature_ids.get(cat, []))

        self.vocab_features = set()
        for cat in factory_config.VOCAB_ARTIFACT_CATEGORIES:
            self.vocab_features.update(self.feature_ids.get(cat, []))

        print(f"Loaded {len(self.intent_features)} intent features")
        print(f"Loaded {len(self.vocab_features)} vocab artifact features")

        # Load probe
        self._load_probe()

    def _load_probe(self):
        """Load the trained probe for scoring."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        # Load scheming latents (probe features)
        scheming_path = CACHE_DIR / "classifications" / f"layer_{LAYER}_scheming_latents.json"
        with open(scheming_path) as f:
            self.probe_features = json.load(f)

        # Load training data to fit probe
        train_act = np.load(CACHE_DIR / f"layer_{LAYER}" / "train_activations.npy")
        train_labels = np.load(CACHE_DIR / f"layer_{LAYER}" / "train_labels.npy")

        X_train = train_act[:, self.probe_features]

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.probe = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
        self.probe.fit(X_train_scaled, train_labels)

        print(f"Loaded probe with {len(self.probe_features)} features")

    def get_fired_features(self, activations: np.ndarray, threshold: float = 1.0) -> list[int]:
        """Get list of feature IDs that fired above threshold."""
        fired = np.where(activations > threshold)[0]
        return fired.tolist()

    def validate_sample(
        self,
        activations: np.ndarray,
        sample_id: str = "unknown",
        activation_threshold: float = None,  # Auto-detect based on magnitude
    ) -> ValidationResult:
        """
        Validate a single sample's activations.

        Args:
            activations: 1D array of shape (n_features,) - SAE activations
            sample_id: Identifier for the sample
            activation_threshold: Minimum activation to count as "fired" (auto if None)

        Returns:
            ValidationResult with all checks
        """
        # Auto-detect threshold based on magnitude regime
        if activation_threshold is None:
            mean_mag = activations.mean()
            if mean_mag > 100:  # High magnitude regime (gold_106, train)
                activation_threshold = 1.0
            elif mean_mag > 10:  # Medium regime (benchmark, etc)
                activation_threshold = 0.1
            else:  # Low magnitude regime (vocab_constrained, etc)
                activation_threshold = 0.01

        # Get all fired features
        all_fired = set(self.get_fired_features(activations, activation_threshold))

        # Check intent features
        intent_fired = list(all_fired & self.intent_features)
        passes_intent = len(intent_fired) >= factory_config.MIN_INTENT_FEATURES

        # Check vocab features
        vocab_fired = list(all_fired & self.vocab_features)
        passes_vocab = len(vocab_fired) <= factory_config.MAX_VOCAB_FEATURES

        # Get probe score
        X = activations[self.probe_features].reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        probe_score = float(self.probe.predict_proba(X_scaled)[0, 1])
        passes_probe = probe_score >= factory_config.MIN_PROBE_SCORE

        # Determine quality tier
        if (len(intent_fired) >= 5 and probe_score >= 0.9):
            tier = "A"
        elif (len(intent_fired) >= 3 and probe_score >= 0.7):
            tier = "B"
        elif (len(intent_fired) >= 1 and probe_score >= 0.5):
            tier = "C"
        else:
            tier = "F"

        return ValidationResult(
            sample_id=sample_id,
            intent_features_fired=intent_fired,
            vocab_features_fired=vocab_fired,
            probe_score=probe_score,
            passes_intent_check=passes_intent,
            passes_vocab_check=passes_vocab,
            passes_probe_check=passes_probe,
            quality_tier=tier,
        )

    def validate_batch(
        self,
        activations: np.ndarray,
        sample_ids: Optional[list[str]] = None,
    ) -> list[ValidationResult]:
        """
        Validate a batch of samples.

        Args:
            activations: 2D array of shape (n_samples, n_features)
            sample_ids: Optional list of sample identifiers

        Returns:
            List of ValidationResult
        """
        n_samples = activations.shape[0]
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(n_samples)]

        results = []
        for i in range(n_samples):
            result = self.validate_sample(activations[i], sample_ids[i])
            results.append(result)

        return results


def main():
    """Run validation on cached samples."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vocab_constrained",
                       help="Dataset to validate")
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    print("=" * 70)
    print(f"INTENT VALIDATION: {args.dataset}")
    print("=" * 70)

    # Load activations
    act_path = CACHE_DIR / f"layer_{LAYER}" / f"{args.dataset}_mean_activations.npy"
    if not act_path.exists():
        print(f"Error: {act_path} not found")
        return

    activations = np.load(act_path)
    print(f"Loaded {activations.shape[0]} samples")

    # Load labels if available
    labels_path = CACHE_DIR / f"layer_{LAYER}" / f"{args.dataset}_labels.json"
    labels = None
    if labels_path.exists():
        with open(labels_path) as f:
            label_data = json.load(f)
        labels = label_data.get("labels", label_data)

    # Validate
    validator = IntentValidator()
    results = validator.validate_batch(activations)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    tier_counts = {"A": 0, "B": 0, "C": 0, "F": 0}
    pass_counts = {"intent": 0, "vocab": 0, "probe": 0, "all": 0}

    for r in results:
        tier_counts[r.quality_tier] += 1
        if r.passes_intent_check:
            pass_counts["intent"] += 1
        if r.passes_vocab_check:
            pass_counts["vocab"] += 1
        if r.passes_probe_check:
            pass_counts["probe"] += 1
        if r.passes_all:
            pass_counts["all"] += 1

    n = len(results)
    print(f"\nQuality Tiers:")
    print(f"  Tier A (best):  {tier_counts['A']:>4} ({100*tier_counts['A']/n:>5.1f}%)")
    print(f"  Tier B (good):  {tier_counts['B']:>4} ({100*tier_counts['B']/n:>5.1f}%)")
    print(f"  Tier C (ok):    {tier_counts['C']:>4} ({100*tier_counts['C']/n:>5.1f}%)")
    print(f"  Tier F (fail):  {tier_counts['F']:>4} ({100*tier_counts['F']/n:>5.1f}%)")

    print(f"\nCheck Pass Rates:")
    print(f"  Intent check:   {pass_counts['intent']:>4} ({100*pass_counts['intent']/n:>5.1f}%)")
    print(f"  Vocab check:    {pass_counts['vocab']:>4} ({100*pass_counts['vocab']/n:>5.1f}%)")
    print(f"  Probe check:    {pass_counts['probe']:>4} ({100*pass_counts['probe']/n:>5.1f}%)")
    print(f"  All checks:     {pass_counts['all']:>4} ({100*pass_counts['all']/n:>5.1f}%)")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent / f"{args.dataset}_validation.json"

    output = {
        "dataset": args.dataset,
        "n_samples": n,
        "tier_counts": tier_counts,
        "pass_counts": pass_counts,
        "results": [r.to_dict() for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
