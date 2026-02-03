#!/usr/bin/env python3
"""Extract SAE activations for af-detection-benchmark (1,560 samples).

Uses 4-bit quantized Gemma 3 27B to fit in 16GB GPU.
SAE weights are loaded in bfloat16 separately.

Usage:
    python experiments/extract_benchmark_activations.py
"""
import sys
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
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

    # Load benchmark dataset
    print("Loading af-detection-benchmark from HuggingFace...")
    ds = load_dataset(config.BENCHMARK_DATASET, split="test")
    print(f"  Loaded {len(ds)} samples")
    print(f"  Columns: {ds.column_names}")

    # Check label distribution
    if "label" in ds.column_names:
        from collections import Counter
        label_counts = Counter(ds["label"])
        print(f"  Labels: {dict(label_counts)}")

    # Load tokenizer
    print(f"Loading tokenizer: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in 4-bit
    print(f"Loading model (4-bit quantized): {config.MODEL_NAME}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
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
        # output is tuple, first element is hidden states
        if isinstance(output, tuple):
            activations_cache["resid"] = output[0].detach()
        else:
            activations_cache["resid"] = output.detach()

    # Register hook on the target layer's output
    target_layer = model.model.layers[config.LAYER]
    hook = target_layer.register_forward_hook(hook_fn)

    # Extract text field
    text_field = "text" if "text" in ds.column_names else "prompt"
    if text_field not in ds.column_names:
        # Try to find the right field
        for f in ds.column_names:
            if "text" in f.lower() or "prompt" in f.lower() or "content" in f.lower():
                text_field = f
                break
    print(f"  Using text field: {text_field}")

    # Extract activations
    n_samples = len(ds)
    all_max_acts = np.zeros((n_samples, config.N_FEATURES), dtype=np.float32)
    labels = []
    max_length = 2048

    print(f"\nExtracting activations for {n_samples} samples...")
    for i in tqdm(range(n_samples)):
        text = ds[i][text_field]
        label = ds[i].get("label", ds[i].get("category", "unknown"))
        labels.append(label)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        ).to(device)

        with torch.no_grad():
            _ = model(**inputs)
            resid = activations_cache["resid"]  # (1, seq_len, hidden_dim)

            # Encode through SAE
            sae_acts = sae.encode(resid.squeeze(0).to(torch.bfloat16))  # (seq_len, n_features)

            # Max activation per feature across sequence
            max_acts = sae_acts.max(dim=0).values.float().cpu().numpy()
            all_max_acts[i] = max_acts

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_samples} done")

    hook.remove()

    # Save
    out_dir = config.layer_cache()
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "benchmark_activations.npy", all_max_acts)

    # Save labels
    label_data = {"labels": labels, "n_samples": n_samples}
    if "label" in ds.column_names:
        label_data["unique_labels"] = sorted(set(labels))
    with open(out_dir / "benchmark_labels.json", "w") as f:
        json.dump(label_data, f, indent=2)

    print(f"\nSaved: {out_dir}/benchmark_activations.npy shape={all_max_acts.shape}")
    print(f"Saved: {out_dir}/benchmark_labels.json")

    # Quick stats
    print(f"\nActivation stats:")
    print(f"  Min: {all_max_acts.min():.2f}")
    print(f"  Max: {all_max_acts.max():.2f}")
    print(f"  Mean: {all_max_acts.mean():.2f}")
    print(f"  Active features per sample: {(all_max_acts > 0).sum(axis=1).mean():.0f}")


if __name__ == "__main__":
    main()
