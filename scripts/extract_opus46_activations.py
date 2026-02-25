#!/usr/bin/env python3
"""Extract SAE activations for opus46_500 samples on nigel.

Loads Gemma 3 27B (4-bit) + GemmaScope SAE layer 40.
Produces opus46_activations.npy (500, 16384) and opus46_labels.json.

Usage:
    python extract_opus46_activations.py --data opus46_500_hf.json
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
import warnings
warnings.filterwarnings("ignore")

# Config
LAYER = 40
N_FEATURES = 16384
SAE_REPO = "google/gemma-scope-2-27b-it"
MODEL_NAME = "google/gemma-3-27b-it"
OUTPUT_DIR = Path(".")


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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to opus46_500_hf.json")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load samples
    print(f"Loading samples from {args.data}...")
    with open(args.data) as f:
        samples = json.load(f)
    print(f"  Loaded {len(samples)} samples")

    # Load tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in 4-bit
    print(f"Loading model (4-bit quantized): {MODEL_NAME}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
    )
    # 16GB GPU — build device map keeping layers 0-40 on GPU, rest on CPU
    # We only need up to layer 40 for the hook, so we can stop the forward there
    from accelerate import infer_auto_device_map, init_empty_weights
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_config(model_config)
    auto_map = infer_auto_device_map(
        empty_model,
        max_memory={0: "14GiB", "cpu": "32GiB"},
        no_split_module_classes=["Gemma3DecoderLayer"],
    )
    # Force lm_head to CPU (we don't need logits, just hook output)
    auto_map["lm_head"] = "cpu"
    del empty_model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map=auto_map,
    )
    model.eval()
    print("  Model loaded")

    # Load SAE — keep on CPU since layer 40 is CPU-offloaded on 16GB GPU
    # This avoids NaN from cross-device bfloat16 conversion
    print(f"Loading SAE: {SAE_REPO} layer {LAYER}")
    sae = ResidPostSAE(SAE_REPO, LAYER)
    sae.to("cpu", dtype=torch.float32)
    print("  SAE loaded (CPU, float32)")

    # Hook to capture residual stream
    activations_cache = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations_cache["resid"] = output[0].detach()
        else:
            activations_cache["resid"] = output.detach()

    # Gemma 3 nests layers under model.language_model
    if hasattr(model.model, 'language_model'):
        target_layer = model.model.language_model.layers[LAYER]
    else:
        target_layer = model.model.layers[LAYER]
    hook = target_layer.register_forward_hook(hook_fn)

    # Extract activations
    n_samples = len(samples)
    all_max_acts = np.zeros((n_samples, N_FEATURES), dtype=np.float32)
    labels = []
    max_length = 2048

    print(f"\nExtracting activations for {n_samples} samples...")
    for i in tqdm(range(n_samples)):
        text = samples[i]["text"]
        label = samples[i].get("label", "alignment_faking")
        labels.append(label)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        # With device_map="auto", find where input_ids should go
        if hasattr(model, 'hf_device_map'):
            # Find the embedding device
            for name, dev in model.hf_device_map.items():
                if 'embed' in name:
                    input_device = dev
                    break
            else:
                input_device = list(model.hf_device_map.values())[0]
            inputs = {k: v.to(input_device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            try:
                _ = model(**inputs)
            except Exception:
                # Forward may fail after layer 40 (CPU offload issues)
                # but we only need the hook output which fires at layer 40
                pass

            if "resid" not in activations_cache:
                print(f"  WARNING: no activation captured for sample {i}, skipping")
                continue

            resid = activations_cache["resid"]

            # Encode through SAE on CPU in float32 (layer 40 is CPU-offloaded)
            sae_acts = sae.encode(resid.squeeze(0).to(device="cpu", dtype=torch.float32))

            # Max activation per feature across sequence
            max_acts = sae_acts.max(dim=0).values.float().cpu().numpy()
            all_max_acts[i] = max_acts

        if (i + 1) % 50 == 0:
            # Checkpoint
            np.save(output_dir / "opus46_activations_partial.npy", all_max_acts[:i+1])
            print(f"  checkpoint: {i+1}/{n_samples}")

    hook.remove()

    # Save
    np.save(output_dir / "opus46_activations.npy", all_max_acts)

    label_data = {
        "labels": labels,
        "n_samples": n_samples,
        "unique_labels": sorted(set(labels)),
    }
    with open(output_dir / "opus46_labels.json", "w") as f:
        json.dump(label_data, f, indent=2)

    print(f"\nSaved: opus46_activations.npy shape={all_max_acts.shape}")
    print(f"Saved: opus46_labels.json")

    # Quick stats
    print(f"\nActivation stats:")
    print(f"  Min: {all_max_acts.min():.2f}")
    print(f"  Max: {all_max_acts.max():.2f}")
    print(f"  Mean: {all_max_acts.mean():.2f}")
    print(f"  Active features per sample: {(all_max_acts > 0).sum(axis=1).mean():.0f}")


if __name__ == "__main__":
    main()
