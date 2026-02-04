#!/usr/bin/env python3
"""
Matryoshka SAE Validation for AF Detection

Quick test: Does hierarchical feature structure help OOD generalization?

Usage:
    python run_msa_validation.py

Requires: transformers, sae_lens, torch, sklearn
"""
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings

# Suppress sklearn warnings for single-class
warnings.filterwarnings("ignore", category=UserWarning)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# MSA nesting levels (features at each level)
MATRYOSHKA_LEVELS = [128, 512, 2048, 8192, 32768]

@dataclass
class Config:
    model_name: str = "google/gemma-3-1b-it"
    sae_release: str = "gemma-3-1b-res-matryoshka-dc"
    layer: int = 12  # Midpoint of 26-layer model
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_samples: int = 200  # Limit for speed
    seed: int = 42


def load_samples(dataset_name: str, max_n: int = 200) -> list:
    """Load samples from cache."""
    # Try multiple possible locations
    possible_paths = [
        Path(f"../../cache/{dataset_name}_samples.json"),
        Path(f"cache/{dataset_name}_samples.json"),
        Path(f"/Users/vincent/autointerp/cache/{dataset_name}_samples.json"),
    ]

    for path in possible_paths:
        if path.exists():
            with open(path) as f:
                samples = json.load(f)
            # Subsample if needed
            if len(samples) > max_n:
                np.random.seed(42)
                indices = np.random.choice(len(samples), max_n, replace=False)
                samples = [samples[i] for i in indices]
            return samples

    raise FileNotFoundError(f"Could not find {dataset_name}_samples.json")


def load_gold_106() -> list:
    """Load gold_106 test set."""
    possible_paths = [
        Path("../../cache/layer_40/test_texts.json"),
        Path("/Users/vincent/autointerp/cache/layer_40/test_texts.json"),
    ]

    for path in possible_paths:
        if path.exists():
            with open(path) as f:
                return json.load(f)

    raise FileNotFoundError("Could not find test_texts.json (gold_106)")


def setup_model_and_sae(config: Config):
    """Load Gemma-3-1b and Matryoshka SAE."""
    print(f"Loading model: {config.model_name}")
    print(f"Loading SAE: {config.sae_release}")
    print(f"Device: {config.device}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from sae_lens import SAE

        # Load model
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map=config.device,
        )
        model.eval()

        # Load MSA
        hook_point = f"blocks.{config.layer}.hook_resid_post"
        print(f"Loading SAE for hook: {hook_point}")

        sae, cfg_dict, sparsity = SAE.from_pretrained(
            config.sae_release,
            hook_point,
            device=config.device,
        )

        print(f"SAE loaded: {sae.cfg.d_sae} features")

        return model, tokenizer, sae

    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install sae_lens transformers")
        raise


def extract_activations(model, tokenizer, sae, texts: list, config: Config) -> dict:
    """Extract MSA features at each nesting level."""
    from transformer_lens import HookedTransformer

    # We need to use HookedTransformer for activation extraction
    # Or manually hook into the model

    all_features = {level: [] for level in MATRYOSHKA_LEVELS}

    print(f"\nExtracting activations for {len(texts)} samples...")

    for i, sample in enumerate(texts):
        if isinstance(sample, dict):
            text = sample.get("text", sample.get("scratchpad", ""))
        else:
            text = sample

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(config.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Get hidden states
            outputs = model(**inputs, output_hidden_states=True)

            # Get activation at target layer
            # Layer indexing: hidden_states[0] is embeddings, [1] is layer 0, etc.
            hidden = outputs.hidden_states[config.layer + 1]  # Shape: (1, seq_len, d_model)

            # Take last token or mean pool
            activation = hidden[0, -1, :]  # Last token

            # Encode through SAE
            sae_features = sae.encode(activation.unsqueeze(0))  # Shape: (1, d_sae)
            sae_features = sae_features.squeeze(0).cpu().numpy()

            # Extract features at each nesting level
            for level in MATRYOSHKA_LEVELS:
                # MSA features are ordered: first `level` features are the nested subset
                level_features = sae_features[:level]
                all_features[level].append(level_features)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(texts)} processed")

    # Convert to arrays
    for level in MATRYOSHKA_LEVELS:
        all_features[level] = np.array(all_features[level])
        print(f"  Level {level}: {all_features[level].shape}")

    return all_features


def train_and_evaluate(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       level: int) -> dict:
    """Train probe and evaluate."""
    # Check for single class
    if len(np.unique(y_train)) < 2:
        return {"auroc": None, "accuracy": y_train.mean(), "note": "single class train"}
    if len(np.unique(y_test)) < 2:
        return {"auroc": None, "accuracy": y_test.mean(), "note": "single class test"}

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train probe
    probe = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
    probe.fit(X_train_scaled, y_train)

    # Evaluate
    probs = probe.predict_proba(X_test_scaled)[:, 1]
    preds = probe.predict(X_test_scaled)

    try:
        auroc = roc_auc_score(y_test, probs)
    except:
        auroc = None

    accuracy = accuracy_score(y_test, preds)

    return {
        "auroc": float(auroc) if auroc else None,
        "accuracy": float(accuracy),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "af_train": int(y_train.sum()),
        "af_test": int(y_test.sum()),
    }


def main():
    config = Config()
    np.random.seed(config.seed)

    print("=" * 70)
    print("MATRYOSHKA SAE VALIDATION FOR AF DETECTION")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  Model: {config.model_name}")
    print(f"  SAE: {config.sae_release}")
    print(f"  Layer: {config.layer}")
    print(f"  Nesting levels: {MATRYOSHKA_LEVELS}")

    # Load datasets
    print("\n" + "=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)

    try:
        gold_106 = load_gold_106()
        print(f"gold_106: {len(gold_106)} samples")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        gold_106 = None

    try:
        redwood = load_samples("redwood_novel", config.max_samples)
        print(f"redwood_novel: {len(redwood)} samples")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        redwood = None

    try:
        benchmark = load_samples("benchmark", config.max_samples)
        print(f"benchmark: {len(benchmark)} samples")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        benchmark = None

    # Load model and SAE
    print("\n" + "=" * 70)
    print("LOADING MODEL AND SAE")
    print("=" * 70)

    model, tokenizer, sae = setup_model_and_sae(config)

    # Prepare training data (use benchmark)
    if benchmark is None:
        print("ERROR: Need benchmark data for training")
        return

    train_texts = [s["text"] if isinstance(s, dict) else s for s in benchmark]
    train_labels = np.array([
        1 if s.get("label") in ("alignment_faking", "potential_faking") else 0
        for s in benchmark
    ])

    print(f"\nTraining data: {len(train_labels)} samples, {train_labels.sum()} AF")

    # Extract features for training data
    print("\n" + "=" * 70)
    print("EXTRACTING TRAINING FEATURES")
    print("=" * 70)

    train_features = extract_activations(model, tokenizer, sae, train_texts, config)

    # Results storage
    results = {
        "config": {
            "model": config.model_name,
            "sae": config.sae_release,
            "layer": config.layer,
            "levels": MATRYOSHKA_LEVELS,
        },
        "by_level": {},
    }

    # For each nesting level, train probe and evaluate
    print("\n" + "=" * 70)
    print("TRAINING PROBES AT EACH NESTING LEVEL")
    print("=" * 70)

    for level in MATRYOSHKA_LEVELS:
        print(f"\n--- Level {level} features ---")
        X_train = train_features[level]

        level_results = {"level": level, "evaluations": {}}

        # Evaluate on each test set
        for name, test_data in [("gold_106", gold_106), ("redwood_novel", redwood)]:
            if test_data is None:
                continue

            print(f"  Extracting {name} features...")
            test_texts = [s["text"] if isinstance(s, dict) else s for s in test_data]
            test_features = extract_activations(model, tokenizer, sae, test_texts, config)

            X_test = test_features[level]

            # Get labels
            if name == "gold_106":
                y_test = np.array([s.get("is_af", s.get("label") == "alignment_faking")
                                   for s in test_data])
            else:
                y_test = np.array([
                    1 if s.get("label") in ("alignment_faking", "potential_faking") else 0
                    for s in test_data
                ])

            # Train and evaluate
            metrics = train_and_evaluate(X_train, train_labels, X_test, y_test, level)
            level_results["evaluations"][name] = metrics

            auroc_str = f"{metrics['auroc']:.3f}" if metrics['auroc'] else "N/A"
            print(f"    {name}: AUROC={auroc_str}, Acc={metrics['accuracy']:.3f}")

        results["by_level"][level] = level_results

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: AUROC BY NESTING LEVEL")
    print("=" * 70)
    print(f"{'Level':>8} {'gold_106':>10} {'redwood':>10} {'Delta':>10}")
    print("-" * 40)

    for level in MATRYOSHKA_LEVELS:
        lr = results["by_level"][level]["evaluations"]
        g = lr.get("gold_106", {}).get("auroc")
        r = lr.get("redwood_novel", {}).get("auroc")

        g_str = f"{g:.3f}" if g else "N/A"
        r_str = f"{r:.3f}" if r else "N/A"
        d_str = f"{g - r:+.3f}" if (g and r) else "N/A"

        print(f"{level:>8} {g_str:>10} {r_str:>10} {d_str:>10}")

    # Save results
    results_file = RESULTS_DIR / "msa_validation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Key finding
    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)

    # Check if smaller levels have better OOD generalization
    gold_aurocs = [results["by_level"][l]["evaluations"].get("gold_106", {}).get("auroc")
                   for l in MATRYOSHKA_LEVELS]
    redwood_aurocs = [results["by_level"][l]["evaluations"].get("redwood_novel", {}).get("auroc")
                      for l in MATRYOSHKA_LEVELS]

    if all(g and r for g, r in zip(gold_aurocs, redwood_aurocs)):
        gaps = [g - r for g, r in zip(gold_aurocs, redwood_aurocs)]
        best_level = MATRYOSHKA_LEVELS[np.argmin(gaps)]

        print(f"Smallest OOD gap at level {best_level}")
        print(f"  Gap: {min(gaps):.3f}")

        if best_level < 8192:
            print("\n✓ HYPOTHESIS SUPPORTED: Smaller dictionaries generalize better")
            print("  AF detection may rely on high-level features being absorbed")
        else:
            print("\n✗ HYPOTHESIS NOT SUPPORTED: Larger dictionaries still better")
            print("  Feature absorption not the main OOD issue")
    else:
        print("Incomplete results - check for errors above")


if __name__ == "__main__":
    main()
