#!/usr/bin/env python3
"""Extract SAE activations for af-detection-benchmark (2,330 samples).

Saves multiple intermediate states for offline analysis:
  - benchmark_activations.npy       (n, 16384)  max SAE activation per feature
  - benchmark_mean_activations.npy  (n, 16384)  mean SAE activation per feature
  - benchmark_top_token_indices.npy (n, 16384)  token index of max activation per feature
  - benchmark_labels.json           labels + metadata
  - benchmark_sparsity.json         per-sample sparsity stats

Usage:
    python experiments/extract_benchmark_activations.py
"""
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load benchmark dataset from local JSON
    benchmark_path = config.CACHE_DIR / "benchmark_samples.json"
    print(f"Loading benchmark from {benchmark_path}...")
    with open(benchmark_path) as f:
        samples = json.load(f)
    print(f"  Loaded {len(samples)} samples")

    from collections import Counter
    label_counts = Counter(s["label"] for s in samples)
    print(f"  Labels: {dict(label_counts)}")

    # Load tokenizer
    print(f"Loading tokenizer: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in bf16
    # Gemma 3 -it config maps to ConditionalGeneration (multimodal) by default,
    # but we only need the text backbone. Force the text-only CausalLM class.
    print(f"Loading model (bf16): {config.MODEL_NAME}")
    try:
        from transformers import Gemma3ForCausalLM
        model = Gemma3ForCausalLM.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    except (ImportError, Exception):
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

    # Hook to capture residual stream at target layer
    activations_cache = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations_cache["resid"] = output[0].detach()
        else:
            activations_cache["resid"] = output.detach()

    target_layer = model.model.layers[config.LAYER]
    hook = target_layer.register_forward_hook(hook_fn)

    # Prepare output arrays
    n_samples = len(samples)
    n_features = config.N_FEATURES
    all_max_acts = np.zeros((n_samples, n_features), dtype=np.float32)
    all_mean_acts = np.zeros((n_samples, n_features), dtype=np.float32)
    all_top_token_idx = np.zeros((n_samples, n_features), dtype=np.int32)
    labels = [s["label"] for s in samples]
    token_counts = []
    max_length = 2048

    out_dir = config.layer_cache()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"\nExtracting activations for {n_samples} samples...")
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
            resid = activations_cache["resid"]  # (1, seq_len, hidden_dim)

            # Encode through SAE
            sae_acts = sae.encode(resid.squeeze(0).to(torch.bfloat16))  # (seq_len, n_features)

            # Max activation per feature + which token
            max_vals, max_idx = sae_acts.max(dim=0)
            all_max_acts[i] = max_vals.float().cpu().numpy()
            all_top_token_idx[i] = max_idx.cpu().numpy()

            # Mean activation per feature
            all_mean_acts[i] = sae_acts.mean(dim=0).float().cpu().numpy()

        # Checkpoint every 500 samples
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_samples - i - 1) / rate
            print(f"  {i+1}/{n_samples} ({rate:.1f} samples/s, ETA {eta/60:.1f} min)")
            # Save checkpoint
            np.save(out_dir / "benchmark_activations.npy", all_max_acts)
            print(f"  Checkpoint saved")

    hook.remove()
    elapsed = time.time() - t0

    # Save all outputs
    np.save(out_dir / "benchmark_activations.npy", all_max_acts)
    np.save(out_dir / "benchmark_mean_activations.npy", all_mean_acts)
    np.save(out_dir / "benchmark_top_token_indices.npy", all_top_token_idx)

    # Save labels + metadata
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
    with open(out_dir / "benchmark_labels.json", "w") as f:
        json.dump(label_data, f, indent=2)

    # Save per-sample sparsity stats (useful for offline auditing)
    sparsity_stats = {
        "per_sample_active_features": (all_max_acts > 0).sum(axis=1).tolist(),
        "per_sample_max_activation": all_max_acts.max(axis=1).tolist(),
        "per_sample_mean_activation": all_max_acts.mean(axis=1).tolist(),
        "per_sample_token_count": token_counts,
    }
    with open(out_dir / "benchmark_sparsity.json", "w") as f:
        json.dump(sparsity_stats, f)

    print(f"\nDone in {elapsed/60:.1f} min ({elapsed/n_samples:.2f}s per sample)")
    print(f"\nSaved to {out_dir}/:")
    print(f"  benchmark_activations.npy       {all_max_acts.shape}  max per feature")
    print(f"  benchmark_mean_activations.npy  {all_mean_acts.shape}  mean per feature")
    print(f"  benchmark_top_token_indices.npy {all_top_token_idx.shape}  argmax token index")
    print(f"  benchmark_labels.json           labels + extraction metadata")
    print(f"  benchmark_sparsity.json         per-sample sparsity stats")

    print(f"\nActivation stats:")
    print(f"  Min: {all_max_acts.min():.2f}")
    print(f"  Max: {all_max_acts.max():.2f}")
    print(f"  Mean: {all_max_acts.mean():.2f}")
    print(f"  Active features per sample: {(all_max_acts > 0).sum(axis=1).mean():.0f}")
    print(f"  Tokens per sample: mean={np.mean(token_counts):.0f}, max={max(token_counts)}")


if __name__ == "__main__":
    main()
