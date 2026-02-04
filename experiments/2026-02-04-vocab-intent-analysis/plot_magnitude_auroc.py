#!/usr/bin/env python3
"""
Plot activation magnitude vs probe AUROC.

This shows the "activation regime" problem: datasets with different
magnitude ranges may fail because the probe learned at a different scale.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
    results_path = Path(__file__).parent / "vocab_intent_results.json"
    if not results_path.exists():
        print("Run vocab_intent_scatter.py first")
        return

    with open(results_path) as f:
        results = json.load(f)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Magnitude vs AUROC ---
    names = []
    magnitudes = []
    aurocs = []
    overlaps = []

    for name, m in results.items():
        names.append(name)
        magnitudes.append(m["mean_magnitude"])
        aurocs.append(m["probe_auroc"])
        overlaps.append(m["feature_overlap"])

    # Color by feature overlap
    scatter1 = ax1.scatter(
        magnitudes, aurocs,
        c=overlaps, cmap="viridis",
        s=150, alpha=0.8,
        edgecolors="black", linewidth=1,
    )
    plt.colorbar(scatter1, ax=ax1, label="Top-100 Feature Overlap")

    for i, name in enumerate(names):
        ax1.annotate(name, (magnitudes[i] + 2, aurocs[i]), fontsize=9)

    # Reference line at gold_106 magnitude
    gold_mag = results.get("gold_106", {}).get("mean_magnitude", 0)
    ax1.axvline(x=gold_mag, color="red", linestyle="--", alpha=0.5, label=f"gold_106 ({gold_mag:.0f})")

    ax1.set_xlabel("Mean Activation Magnitude", fontsize=12)
    ax1.set_ylabel("Probe AUROC", fontsize=12)
    ax1.set_title("Activation Magnitude vs Detection Performance", fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # --- Plot 2: RLHF Frequency vs Intent Correlation ---
    rlhf_freqs = [results[n]["rlhf_freq"] for n in names]
    intent_corrs = [results[n]["activation_corr"] for n in names]

    scatter2 = ax2.scatter(
        rlhf_freqs, intent_corrs,
        c=aurocs, cmap="RdYlGn",
        s=150, alpha=0.8,
        edgecolors="black", linewidth=1,
        vmin=0.3, vmax=1.0
    )
    plt.colorbar(scatter2, ax=ax2, label="Probe AUROC")

    for i, name in enumerate(names):
        offset = 0.02 if rlhf_freqs[i] < 1 else 0.05
        ax2.annotate(name, (rlhf_freqs[i] + offset, intent_corrs[i]), fontsize=9)

    ax2.set_xlabel("RLHF Term Frequency (per 1000 chars)", fontsize=12)
    ax2.set_ylabel("Intent Correlation (activation pattern)", fontsize=12)
    ax2.set_title("RLHF Vocabulary vs Intent Signal", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)

    # Highlight vocab_constrained as the key test
    vc_idx = names.index("vocab_constrained") if "vocab_constrained" in names else None
    if vc_idx is not None:
        ax2.annotate(
            "Zero RLHF vocab\nstill detected!",
            xy=(rlhf_freqs[vc_idx], intent_corrs[vc_idx]),
            xytext=(rlhf_freqs[vc_idx] + 0.3, intent_corrs[vc_idx] - 0.15),
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="green"),
            color="green",
            fontweight="bold",
        )

    plt.tight_layout()

    output_path = Path(__file__).parent / "magnitude_auroc.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
