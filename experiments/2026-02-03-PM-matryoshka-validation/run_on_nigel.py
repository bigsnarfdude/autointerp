#!/usr/bin/env python3
"""
Matryoshka SAE Validation - Run on Nigel (GPU)

Quick test: Does hierarchical feature structure help OOD generalization?

Usage on nigel:
    cd ~/autointerp/experiments/2026-02-03-PM-matryoshka-validation
    python run_on_nigel.py
"""
import json
import numpy as np
from pathlib import Path
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Config
MODEL_NAME = "google/gemma-3-1b-it"
SAE_RELEASE = "gemma-3-1b-res-matryoshka-dc"
LAYER = 12  # Midpoint of 26-layer model
MATRYOSHKA_LEVELS = [128, 512, 2048, 8192, 32768]
MAX_SAMPLES = 150  # Limit for speed
SEED = 42

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_benchmark_samples(max_n: int = 150) -> tuple:
    """Load benchmark samples for training."""
    # Try to find the benchmark samples
    paths = [
        Path("../../cache/benchmark_samples.json"),
        Path("/home/vincent/autointerp/cache/benchmark_samples.json"),
    ]

    for path in paths:
        if path.exists():
            with open(path) as f:
                samples = json.load(f)
            break
    else:
        raise FileNotFoundError("benchmark_samples.json not found")

    # Subsample
    np.random.seed(SEED)
    if len(samples) > max_n:
        indices = np.random.choice(len(samples), max_n, replace=False)
        samples = [samples[i] for i in indices]

    texts = [s["text"] for s in samples]
    labels = np.array([
        1 if s.get("label") in ("alignment_faking", "potential_faking") else 0
        for s in samples
    ])

    return texts, labels


def load_gold_106() -> tuple:
    """Load gold_106 test set."""
    paths = [
        Path("../../cache/layer_40/test_texts.json"),
        Path("/home/vincent/autointerp/cache/layer_40/test_texts.json"),
    ]

    for path in paths:
        if path.exists():
            with open(path) as f:
                samples = json.load(f)
            break
    else:
        raise FileNotFoundError("test_texts.json not found")

    texts = [s["text"] for s in samples]
    labels = np.array([1 if s.get("is_af", False) else 0 for s in samples])

    return texts, labels


def load_redwood_samples(max_n: int = 150) -> tuple:
    """Load redwood_novel samples."""
    paths = [
        Path("../../cache/redwood_novel_samples.json"),
        Path("/home/vincent/autointerp/cache/redwood_novel_samples.json"),
    ]

    for path in paths:
        if path.exists():
            with open(path) as f:
                samples = json.load(f)
            break
    else:
        raise FileNotFoundError("redwood_novel_samples.json not found")

    np.random.seed(SEED + 1)
    if len(samples) > max_n:
        indices = np.random.choice(len(samples), max_n, replace=False)
        samples = [samples[i] for i in indices]

    texts = [s["text"] for s in samples]
    labels = np.array([
        1 if s.get("label") in ("alignment_faking", "potential_faking") else 0
        for s in samples
    ])

    return texts, labels


def extract_features(model, tokenizer, sae, texts: list, device: str) -> np.ndarray:
    """Extract SAE features for all texts."""
    all_features = []

    for text in tqdm(texts, desc="Extracting"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

            # Get hidden state at target layer
            # hidden_states[0] = embeddings, [1] = layer 0, ...
            hidden = outputs.hidden_states[LAYER + 1]

            # Use last token activation
            activation = hidden[0, -1, :].unsqueeze(0)  # (1, d_model)

            # Encode through SAE
            features = sae.encode(activation)  # (1, 32768)
            all_features.append(features.cpu().numpy().squeeze())

    return np.array(all_features)


def train_and_eval(X_train, y_train, X_test, y_test) -> dict:
    """Train probe and evaluate."""
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"auroc": None, "accuracy": None, "note": "single class"}

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    probe = LogisticRegression(C=10.0, max_iter=1000, random_state=SEED)
    probe.fit(X_train_s, y_train)

    probs = probe.predict_proba(X_test_s)[:, 1]
    preds = probe.predict(X_test_s)

    return {
        "auroc": float(roc_auc_score(y_test, probs)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "n_test": len(y_test),
        "af_test": int(y_test.sum()),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cpu":
        print("WARNING: Running on CPU will be slow!")

    print("=" * 60)
    print("MATRYOSHKA SAE VALIDATION")
    print("=" * 60)

    # Load model and SAE
    print(f"\nLoading {MODEL_NAME}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    print(f"Loading SAE: {SAE_RELEASE} layer {LAYER}...")
    sae = SAE.from_pretrained(
        SAE_RELEASE,
        f"blocks.{LAYER}.hook_resid_post",
        device=device,
    )
    print(f"SAE: {sae.cfg.d_sae} features")

    # Load data
    print("\nLoading datasets...")
    train_texts, train_labels = load_benchmark_samples(MAX_SAMPLES)
    print(f"  Train: {len(train_labels)} samples, {train_labels.sum()} AF")

    gold_texts, gold_labels = load_gold_106()
    print(f"  gold_106: {len(gold_labels)} samples, {gold_labels.sum()} AF")

    redwood_texts, redwood_labels = load_redwood_samples(MAX_SAMPLES)
    print(f"  redwood: {len(redwood_labels)} samples, {redwood_labels.sum()} AF")

    # Extract features
    print("\nExtracting training features...")
    train_features = extract_features(model, tokenizer, sae, train_texts, device)

    print("\nExtracting gold_106 features...")
    gold_features = extract_features(model, tokenizer, sae, gold_texts, device)

    print("\nExtracting redwood features...")
    redwood_features = extract_features(model, tokenizer, sae, redwood_texts, device)

    # Evaluate at each nesting level
    print("\n" + "=" * 60)
    print("RESULTS BY NESTING LEVEL")
    print("=" * 60)
    print(f"{'Level':>8} {'gold_106':>10} {'redwood':>10} {'Gap':>10}")
    print("-" * 40)

    results = {"levels": {}}

    for level in MATRYOSHKA_LEVELS:
        # Slice features to this level
        X_train = train_features[:, :level]
        X_gold = gold_features[:, :level]
        X_redwood = redwood_features[:, :level]

        gold_metrics = train_and_eval(X_train, train_labels, X_gold, gold_labels)
        redwood_metrics = train_and_eval(X_train, train_labels, X_redwood, redwood_labels)

        g = gold_metrics["auroc"]
        r = redwood_metrics["auroc"]

        g_str = f"{g:.3f}" if g else "N/A"
        r_str = f"{r:.3f}" if r else "N/A"
        gap_str = f"{g - r:+.3f}" if (g and r) else "N/A"

        print(f"{level:>8} {g_str:>10} {r_str:>10} {gap_str:>10}")

        results["levels"][level] = {
            "gold_106": gold_metrics,
            "redwood": redwood_metrics,
        }

    # Save results
    results["config"] = {
        "model": MODEL_NAME,
        "sae": SAE_RELEASE,
        "layer": LAYER,
        "max_samples": MAX_SAMPLES,
    }

    with open(RESULTS_DIR / "msa_validation.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    gaps = []
    for level in MATRYOSHKA_LEVELS:
        g = results["levels"][level]["gold_106"]["auroc"]
        r = results["levels"][level]["redwood"]["auroc"]
        if g and r:
            gaps.append((level, g - r))

    if gaps:
        best_level, best_gap = min(gaps, key=lambda x: x[1])
        worst_level, worst_gap = max(gaps, key=lambda x: x[1])

        print(f"Smallest OOD gap: Level {best_level} (gap={best_gap:.3f})")
        print(f"Largest OOD gap:  Level {worst_level} (gap={worst_gap:.3f})")

        if best_level < 8192:
            print("\n✓ Smaller dictionaries show better OOD generalization")
        else:
            print("\n✗ Larger dictionaries still perform better")

    print(f"\nResults saved to {RESULTS_DIR / 'msa_validation.json'}")


if __name__ == "__main__":
    main()
