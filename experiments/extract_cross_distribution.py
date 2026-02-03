#!/usr/bin/env python3
"""Extract SAE activations for cross-distribution datasets.

Supports: sycophant (319), model_organism (730), benchmark (2330)
Saves same intermediate states as extract_benchmark_activations.py.

Usage:
    python experiments/extract_cross_distribution.py --dataset sycophant
    python experiments/extract_cross_distribution.py --dataset model_organism
    python experiments/extract_cross_distribution.py --dataset all
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


DATASETS = {
    "sycophant": "sycophant_samples.json",
    "model_organism": "model_organism_samples.json",
    "benchmark": "benchmark_samples.json",
    "vocab_constrained": "vocab_constrained_samples.json",
}


def extract_dataset(dataset_name, model, tokenizer, sae, device, max_length=2048):
    """Extract activations for a single dataset."""
    samples_path = config.CACHE_DIR / DATASETS[dataset_name]
    print(f"\n{'='*60}")
    print(f"Extracting: {dataset_name}")
    print(f"Loading from {samples_path}...")

    with open(samples_path) as f:
        samples = json.load(f)
    print(f"  {len(samples)} samples")

    from collections import Counter
    label_counts = Counter(s["label"] for s in samples)
    print(f"  Labels: {dict(label_counts)}")

    # Hook to capture residual stream
    activations_cache = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations_cache["resid"] = output[0].detach()
        else:
            activations_cache["resid"] = output.detach()

    target_layer = model.model.layers[config.LAYER]
    hook = target_layer.register_forward_hook(hook_fn)

    n_samples = len(samples)
    n_features = config.N_FEATURES
    all_max_acts = np.zeros((n_samples, n_features), dtype=np.float32)
    all_mean_acts = np.zeros((n_samples, n_features), dtype=np.float32)
    all_top_token_idx = np.zeros((n_samples, n_features), dtype=np.int32)
    labels = [s["label"] for s in samples]
    token_counts = []

    out_dir = config.layer_cache()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = dataset_name

    t0 = time.time()
    for i in tqdm(range(n_samples), desc=dataset_name):
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
            resid = activations_cache["resid"]

            sae_acts = sae.encode(resid.squeeze(0).to(torch.bfloat16))

            max_vals, max_idx = sae_acts.max(dim=0)
            all_max_acts[i] = max_vals.float().cpu().numpy()
            all_top_token_idx[i] = max_idx.cpu().numpy()
            all_mean_acts[i] = sae_acts.mean(dim=0).float().cpu().numpy()

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  {i+1}/{n_samples} ({rate:.1f} samples/s)")

    hook.remove()
    elapsed = time.time() - t0

    # Save all outputs
    np.save(out_dir / f"{prefix}_activations.npy", all_max_acts)
    np.save(out_dir / f"{prefix}_mean_activations.npy", all_mean_acts)
    np.save(out_dir / f"{prefix}_top_token_indices.npy", all_top_token_idx)

    label_data = {
        "labels": labels,
        "unique_labels": sorted(set(labels)),
        "label_counts": dict(Counter(labels)),
        "n_samples": n_samples,
        "token_counts": token_counts,
        "extraction_config": {
            "model": config.MODEL_NAME,
            "sae_repo": config.SAE_REPO,
            "layer": config.LAYER,
            "n_features": n_features,
            "max_length": max_length,
            "dtype": "bfloat16",
            "elapsed_seconds": round(elapsed, 1),
        },
    }
    with open(out_dir / f"{prefix}_labels.json", "w") as f:
        json.dump(label_data, f, indent=2)

    sparsity_stats = {
        "per_sample_active_features": (all_max_acts > 0).sum(axis=1).tolist(),
        "per_sample_max_activation": all_max_acts.max(axis=1).tolist(),
        "per_sample_mean_activation": all_max_acts.mean(axis=1).tolist(),
        "per_sample_token_count": token_counts,
    }
    with open(out_dir / f"{prefix}_sparsity.json", "w") as f:
        json.dump(sparsity_stats, f)

    print(f"\n{dataset_name} done in {elapsed:.1f}s ({elapsed/n_samples:.2f}s/sample)")
    print(f"  Active features/sample: {(all_max_acts > 0).sum(axis=1).mean():.0f}")
    print(f"  Mean max activation: {all_max_acts.max(axis=1).mean():.1f}")
    print(f"  Tokens/sample: mean={np.mean(token_counts):.0f}, max={max(token_counts)}")

    return {
        "dataset": dataset_name,
        "n_samples": n_samples,
        "elapsed": round(elapsed, 1),
        "active_features_mean": float((all_max_acts > 0).sum(axis=1).mean()),
        "max_activation_mean": float(all_max_acts.max(axis=1).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=["sycophant", "model_organism", "benchmark", "vocab_constrained", "all"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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

    # Load SAE
    print(f"Loading SAE: {config.SAE_REPO} layer {config.LAYER}")
    sae = ResidPostSAE(config.SAE_REPO, config.LAYER)
    sae.to(device, dtype=torch.bfloat16)
    print("  SAE loaded")

    datasets_to_run = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    # Filter to datasets that have sample files
    available = []
    for d in datasets_to_run:
        path = config.CACHE_DIR / DATASETS[d]
        if path.exists():
            available.append(d)
        else:
            print(f"  Skipping {d}: {path} not found")

    results = []
    for d in available:
        r = extract_dataset(d, model, tokenizer, sae, device)
        results.append(r)

    # Save summary
    out_dir = config.layer_cache()
    with open(out_dir / "cross_distribution_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Cross-distribution extraction complete:")
    for r in results:
        print(f"  {r['dataset']}: {r['n_samples']} samples, {r['elapsed']}s, "
              f"{r['active_features_mean']:.0f} active features/sample")


if __name__ == "__main__":
    main()
