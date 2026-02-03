#!/usr/bin/env python3
"""Step 2: Collect top-K max-activating examples per latent."""
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def main():
    cache = config.layer_cache(config.LAYER)
    examples_dir = config.CACHE_DIR / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    # Load activations and texts
    print("Loading activations...")
    train_act = np.load(cache / "train_activations.npy")  # (2689, 16384)
    with open(cache / "train_texts.json") as f:
        train_texts = json.load(f)

    n_samples, n_features = train_act.shape
    print(f"Samples: {n_samples}, Features: {n_features}")

    # Find active latents
    max_per_feature = train_act.max(axis=0)  # (16384,)
    active_mask = max_per_feature >= config.MIN_ACTIVATION
    active_indices = np.where(active_mask)[0]
    print(f"Active latents (max >= {config.MIN_ACTIVATION}): {len(active_indices)} / {n_features}")

    # For each active latent, find top-K samples
    all_examples = {}
    for feat_idx in tqdm(active_indices, desc="Collecting examples"):
        feat_idx = int(feat_idx)
        activations = train_act[:, feat_idx]
        top_k_indices = np.argsort(activations)[-config.TOP_K_EXAMPLES:][::-1]

        examples = []
        for sample_idx in top_k_indices:
            sample_idx = int(sample_idx)
            act_val = float(activations[sample_idx])
            if act_val <= 0:
                break
            text = train_texts[sample_idx]["text"]
            # Truncate long texts
            if len(text) > config.MAX_TEXT_CHARS:
                text = text[:config.MAX_TEXT_CHARS] + "..."
            examples.append({
                "sample_idx": sample_idx,
                "activation": round(act_val, 4),
                "label": train_texts[sample_idx]["label"],
                "text": text,
            })

        if examples:
            all_examples[feat_idx] = examples

    # Save as single file (manageable at ~16K entries)
    output_path = examples_dir / f"layer_{config.LAYER}_examples.json"
    with open(output_path, "w") as f:
        json.dump(all_examples, f)

    print(f"\nSaved examples for {len(all_examples)} latents to {output_path}")

    # Stats
    n_examples = [len(v) for v in all_examples.values()]
    print(f"Examples per latent: min={min(n_examples)}, max={max(n_examples)}, "
          f"mean={np.mean(n_examples):.1f}")

    # Label distribution in top examples
    af_counts = []
    for examples in all_examples.values():
        af = sum(1 for e in examples if e["label"] in ("alignment_faking", "potential_faking"))
        af_counts.append(af / len(examples))
    print(f"Mean AF fraction in top-K examples: {np.mean(af_counts):.3f}")

    # Save active latent list (sort numerically, not lexicographically)
    active_path = examples_dir / f"layer_{config.LAYER}_active_latents.json"
    with open(active_path, "w") as f:
        json.dump(sorted(all_examples.keys(), key=int), f)
    print(f"Active latent list saved to {active_path}")

    print("\nStep 2 complete.")


if __name__ == "__main__":
    main()
