#!/usr/bin/env python3
"""
Inventory Tracker: Catalog all samples with their validation status.
"""
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import config as factory_config


class InventoryTracker:
    """Track all samples and their validation status."""

    def __init__(self):
        self.inventory_path = factory_config.INVENTORY_PATH
        self.inventory_path.parent.mkdir(parents=True, exist_ok=True)
        self.load()

    def load(self):
        """Load inventory from disk."""
        if self.inventory_path.exists():
            with open(self.inventory_path) as f:
                self.data = json.load(f)
        else:
            self.data = {
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "version": "1.0",
                },
                "samples": {},
                "datasets": {},
            }

    def save(self):
        """Save inventory to disk."""
        self.data["metadata"]["updated"] = datetime.now().isoformat()
        with open(self.inventory_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def add_sample(
        self,
        sample_id: str,
        text: str,
        label: str,
        source_dataset: str,
        validation: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ):
        """Add a sample to inventory."""
        self.data["samples"][sample_id] = {
            "text": text,
            "label": label,
            "source_dataset": source_dataset,
            "added_at": datetime.now().isoformat(),
            "validation": validation or {},
            "metadata": metadata or {},
            "holdout_split": None,
            "quality_tier": validation.get("quality_tier", "U") if validation else "U",
        }

    def add_dataset(
        self,
        name: str,
        path: str,
        n_samples: int,
        description: str,
        validation_results: Optional[dict] = None,
    ):
        """Register a dataset in inventory."""
        self.data["datasets"][name] = {
            "path": path,
            "n_samples": n_samples,
            "description": description,
            "added_at": datetime.now().isoformat(),
            "validation_results": validation_results or {},
        }

    def get_by_tier(self, tier: str) -> list:
        """Get all samples of a quality tier."""
        return [
            (sid, s) for sid, s in self.data["samples"].items()
            if s.get("quality_tier") == tier
        ]

    def get_by_label(self, label: str) -> list:
        """Get all samples with a label."""
        return [
            (sid, s) for sid, s in self.data["samples"].items()
            if s.get("label") == label
        ]

    def get_validated(self, min_tier: str = "B") -> list:
        """Get samples that pass validation (tier A or B by default)."""
        valid_tiers = {"A", "B"} if min_tier == "B" else {"A"}
        return [
            (sid, s) for sid, s in self.data["samples"].items()
            if s.get("quality_tier") in valid_tiers
        ]

    def stats(self) -> dict:
        """Get inventory statistics."""
        samples = self.data["samples"]

        tier_counts = {"A": 0, "B": 0, "C": 0, "F": 0, "U": 0}
        label_counts = {}
        dataset_counts = {}

        for sid, s in samples.items():
            tier = s.get("quality_tier", "U")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

            label = s.get("label", "unknown")
            label_counts[label] = label_counts.get(label, 0) + 1

            ds = s.get("source_dataset", "unknown")
            dataset_counts[ds] = dataset_counts.get(ds, 0) + 1

        return {
            "total_samples": len(samples),
            "total_datasets": len(self.data["datasets"]),
            "tier_counts": tier_counts,
            "label_counts": label_counts,
            "dataset_counts": dataset_counts,
            "validated_count": tier_counts["A"] + tier_counts["B"],
        }


def main():
    """Show inventory stats."""
    tracker = InventoryTracker()
    stats = tracker.stats()

    print("=" * 70)
    print("AF SAMPLE FACTORY INVENTORY")
    print("=" * 70)

    print(f"\nTotal samples: {stats['total_samples']}")
    print(f"Total datasets: {stats['total_datasets']}")
    print(f"Validated (A+B): {stats['validated_count']}")

    print("\nBy Quality Tier:")
    for tier, count in sorted(stats['tier_counts'].items()):
        print(f"  {tier}: {count}")

    print("\nBy Label:")
    for label, count in sorted(stats['label_counts'].items()):
        print(f"  {label}: {count}")

    print("\nBy Dataset:")
    for ds, count in sorted(stats['dataset_counts'].items()):
        print(f"  {ds}: {count}")


if __name__ == "__main__":
    main()
