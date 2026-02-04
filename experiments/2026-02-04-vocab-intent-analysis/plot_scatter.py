#!/usr/bin/env python3
"""
Generate 2D scatter plot: Vocab similarity vs Intent correlation.

X-axis: Vocabulary similarity to gold_106 (TF-IDF cosine)
Y-axis: Intent correlation (activation pattern correlation with gold_106)

Color: Probe AUROC (how well the probe works on this dataset)
Size: Number of samples
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def main():
    # Load results
    results_path = Path(__file__).parent / "vocab_intent_results.json"
    if not results_path.exists():
        print("Run vocab_intent_scatter.py first to generate results")
        return

    with open(results_path) as f:
        results = json.load(f)

    # Prepare data
    names = []
    x_vocab = []
    y_intent = []
    colors = []
    sizes = []

    for name, m in results.items():
        names.append(name)
        x_vocab.append(m["tfidf_sim"])
        y_intent.append(m["activation_corr"])
        colors.append(m["probe_auroc"])
        # Scale size by log of samples
        sizes.append(max(50, min(500, np.log(m["n_samples"] + 1) * 50)))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Scatter plot
    scatter = ax.scatter(
        x_vocab, y_intent,
        c=colors, cmap="RdYlGn",
        s=sizes, alpha=0.7,
        edgecolors="black", linewidth=1,
        vmin=0.3, vmax=1.0
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label="Probe AUROC")

    # Add labels
    for i, name in enumerate(names):
        # Offset labels to avoid overlap
        offset_x = 0.01
        offset_y = 0.02
        if name == "gold_106":
            offset_x = -0.08
            offset_y = -0.03
        elif name == "benchmark":
            offset_y = -0.04
        elif name == "train":
            offset_x = 0.01
            offset_y = -0.03

        ax.annotate(
            name.replace("_", "\n"),
            (x_vocab[i] + offset_x, y_intent[i] + offset_y),
            fontsize=9,
            ha="left",
        )

    # Add quadrant labels
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0.3, color="gray", linestyle="--", alpha=0.5)

    # Quadrant annotations
    ax.text(0.05, 0.95, "DIFFERENT VOCAB\nHIGH INTENT\n(honest AF)",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5))

    ax.text(0.75, 0.95, "SAME VOCAB\nHIGH INTENT\n(training-like)",
            transform=ax.transAxes, fontsize=10, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))

    ax.text(0.05, 0.05, "DIFFERENT VOCAB\nLOW INTENT\n(different concept?)",
            transform=ax.transAxes, fontsize=10, va="bottom",
            bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5))

    ax.text(0.75, 0.05, "SAME VOCAB\nLOW INTENT\n(vocab confound)",
            transform=ax.transAxes, fontsize=10, va="bottom", ha="right",
            bbox=dict(boxstyle="round", facecolor="orange", alpha=0.5))

    # Labels and title
    ax.set_xlabel("Vocabulary Similarity to gold_106 (TF-IDF cosine)", fontsize=12)
    ax.set_ylabel("Intent Correlation (SAE activation pattern)", fontsize=12)
    ax.set_title("AF Dataset Mapping: Vocabulary vs Intent\n(size = log(samples), color = probe AUROC)", fontsize=14)

    # Set axis limits with padding
    ax.set_xlim(-0.05, max(x_vocab) + 0.15)
    ax.set_ylim(-0.1, 1.1)

    # Grid
    ax.grid(True, alpha=0.3)

    # Save
    output_path = Path(__file__).parent / "vocab_intent_scatter.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    # Also save PDF for paper quality
    pdf_path = Path(__file__).parent / "vocab_intent_scatter.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {pdf_path}")

    plt.show()


if __name__ == "__main__":
    main()
