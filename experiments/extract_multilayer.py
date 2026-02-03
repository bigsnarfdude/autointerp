#!/usr/bin/env python3
"""Extract SAE activations across multiple layers for benchmark dataset.

Extracts layers [22, 31, 40, 53] to test whether different layers capture
different aspects of alignment faking.

Usage:
    python experiments/extract_multilayer.py
    python experiments/extract_multilayer.py --layers 22 31 40 53
    python experiments/extract_multilayer.py --dataset benchmark
    python experiments/extract_multilayer.py --dataset sycophant
"""
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class ResidPostSAE:
    """GemmaScope resid_post SAE."""

    def __init__(self, repo_id: str, layer: int):
        folder = f"resid_post/layer_{layer}_width_16k_l0_medium"
        config_path = hf_hub_download(repo_id, f"{folder}/config.json")
        weights_path = hf_hub_download(repo_id, f"{folder}/params.safetensors")

        with open(config_path) as f:
            self.cfg = json.load(f)

        weights = load_file(weights_path)
        self.w_enc = weights["w_enc"]
        self.b_enc = weights["b_enc"]
        self.threshold = weights.get("threshold", None)
        print(f"  SAE layer {layer}: {self.w_enc.shape}")

    def to(self, device, dtype=None):
        self.w_enc = self.w_enc.to(device, dtype=dtype)
        self.b_enc = self.b_enc.to(device, dtype=dtype)
        if self.threshold is not None:
            self.threshold = self.threshold.to(device, dtype=dtype)
        return self

    def encode(self, x):
        pre_acts = x @ self.w_enc + self.b_enc
        if self.threshold is not None:
            return F.relu(pre_acts - self.threshold)
        return F.relu(pre_acts)


DATASET_FILES = {
    "benchmark": "benchmark_samples.json",
    "sycophant": "sycophant_samples.json",
    "model_organism": "model_organism_samples.json",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=[16, 31, 40, 53])
    parser.add_argument("--dataset", default="benchmark",
                        choices=list(DATASET_FILES.keys()))
    args = parser.parse_args()

    layers = args.layers
    dataset_name = args.dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Layers: {layers}")
    print(f"Dataset: {dataset_name}")

    # Load samples
    samples_path = config.CACHE_DIR / DATASET_FILES[dataset_name]
    print(f"Loading samples from {samples_path}...")
    with open(samples_path) as f:
        samples = json.load(f)
    print(f"  {len(samples)} samples")

    # Load tokenizer
    print(f"Loading tokenizer: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model (bf16): {config.MODEL_NAME}")
    try:
        from transformers import Gemma3ForCausalLM
        model = Gemma3ForCausalLM.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    except (ImportError, Exception):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    model.eval()
    print("  Model loaded")

    # Load SAEs for all layers
    saes = {}
    for layer in layers:
        print(f"Loading SAE layer {layer}...")
        sae = ResidPostSAE(config.SAE_REPO, layer)
        sae.to(device, dtype=torch.bfloat16)
        saes[layer] = sae

    # Set up hooks for all layers
    activations_cache = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activations_cache[layer_idx] = output[0].detach()
            else:
                activations_cache[layer_idx] = output.detach()
        return hook_fn

    hooks = []
    for layer in layers:
        target = model.model.layers[layer]
        h = target.register_forward_hook(make_hook(layer))
        hooks.append(h)

    # Prepare output arrays per layer
    n_samples = len(samples)
    n_features = config.N_FEATURES
    max_length = 2048

    layer_arrays = {}
    for layer in layers:
        layer_arrays[layer] = {
            "max": np.zeros((n_samples, n_features), dtype=np.float32),
            "mean": np.zeros((n_samples, n_features), dtype=np.float32),
            "top_idx": np.zeros((n_samples, n_features), dtype=np.int32),
        }

    labels = [s["label"] for s in samples]
    token_counts = []

    t0 = time.time()
    print(f"\nExtracting {len(layers)} layers × {n_samples} samples...")
    for i in tqdm(range(n_samples)):
        text = samples[i]["text"]

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        ).to(device)

        seq_len = inputs["input_ids"].shape[1]
        token_counts.append(seq_len)

        with torch.no_grad():
            _ = model(**inputs)

            for layer in layers:
                resid = activations_cache[layer]
                sae_acts = saes[layer].encode(resid.squeeze(0).to(torch.bfloat16))

                max_vals, max_idx = sae_acts.max(dim=0)
                layer_arrays[layer]["max"][i] = max_vals.float().cpu().numpy()
                layer_arrays[layer]["top_idx"][i] = max_idx.cpu().numpy()
                layer_arrays[layer]["mean"][i] = sae_acts.mean(dim=0).float().cpu().numpy()

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  {i+1}/{n_samples} ({rate:.1f} samples/s)")

    for h in hooks:
        h.remove()
    elapsed = time.time() - t0

    # Save per-layer outputs
    from collections import Counter
    for layer in layers:
        out_dir = config.CACHE_DIR / f"layer_{layer}"
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = dataset_name

        np.save(out_dir / f"{prefix}_activations.npy", layer_arrays[layer]["max"])
        np.save(out_dir / f"{prefix}_mean_activations.npy", layer_arrays[layer]["mean"])
        np.save(out_dir / f"{prefix}_top_token_indices.npy", layer_arrays[layer]["top_idx"])

        label_data = {
            "labels": labels,
            "unique_labels": sorted(set(labels)),
            "label_counts": dict(Counter(labels)),
            "n_samples": n_samples,
            "token_counts": token_counts,
            "extraction_config": {
                "model": config.MODEL_NAME,
                "sae_repo": config.SAE_REPO,
                "layer": layer,
                "n_features": n_features,
                "max_length": max_length,
                "dtype": "bfloat16",
                "elapsed_seconds": round(elapsed, 1),
                "all_layers_extracted": layers,
            },
        }
        with open(out_dir / f"{prefix}_labels.json", "w") as f:
            json.dump(label_data, f, indent=2)

        sparsity = {
            "per_sample_active_features": (layer_arrays[layer]["max"] > 0).sum(axis=1).tolist(),
            "per_sample_max_activation": layer_arrays[layer]["max"].max(axis=1).tolist(),
            "per_sample_mean_activation": layer_arrays[layer]["max"].mean(axis=1).tolist(),
            "per_sample_token_count": token_counts,
        }
        with open(out_dir / f"{prefix}_sparsity.json", "w") as f:
            json.dump(sparsity, f)

        acts = layer_arrays[layer]["max"]
        print(f"\nLayer {layer}:")
        print(f"  Active features/sample: {(acts > 0).sum(axis=1).mean():.0f}")
        print(f"  Mean max activation: {acts.max(axis=1).mean():.1f}")
        print(f"  Saved to {out_dir}/")

    # Save cross-layer summary
    summary = {
        "dataset": dataset_name,
        "n_samples": n_samples,
        "layers": layers,
        "elapsed_seconds": round(elapsed, 1),
        "per_layer_stats": {},
    }
    for layer in layers:
        acts = layer_arrays[layer]["max"]
        summary["per_layer_stats"][str(layer)] = {
            "active_features_mean": float((acts > 0).sum(axis=1).mean()),
            "max_activation_mean": float(acts.max(axis=1).mean()),
            "mean_activation_mean": float(acts.mean()),
        }
    with open(config.CACHE_DIR / f"multilayer_{dataset_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal: {elapsed:.1f}s ({elapsed/n_samples:.2f}s/sample for {len(layers)} layers)")


if __name__ == "__main__":
    main()
