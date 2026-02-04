#!/usr/bin/env python3
"""
Matryoshka SAE Validation - Gemma-2-9b on ALL datasets

Test if the reversed pattern (good on redwood, bad on gold) holds across datasets.
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
MODEL_NAME = "google/gemma-2-9b-it"
SAE_RELEASE = "gemma-2-9b-res-matryoshka-dc"
LAYER = 21
MATRYOSHKA_LEVELS = [128, 512, 2048, 8192, 32768]
MAX_SAMPLES = 150
SEED = 42

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_dataset(name: str, max_n: int = 150) -> tuple:
    """Load a dataset by name."""
    paths = [
        Path(f"../../cache/{name}_samples.json"),
        Path(f"/home/vincent/autointerp/cache/{name}_samples.json"),
    ]

    # Special case for gold_106
    if name == "gold_106":
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
        raise FileNotFoundError(f"{name} not found")

    # Subsample if needed
    np.random.seed(SEED + hash(name) % 1000)
    if len(samples) > max_n:
        indices = np.random.choice(len(samples), max_n, replace=False)
        samples = [samples[i] for i in indices]

    texts = [s.get("text", s.get("scratchpad", "")) for s in samples]

    # Handle different label formats
    if name == "gold_106":
        labels = np.array([1 if s.get("is_af", False) else 0 for s in samples])
    else:
        labels = np.array([
            1 if s.get("label") in ("alignment_faking", "potential_faking") else 0
            for s in samples
        ])

    return texts, labels, samples


def extract_features(model, tokenizer, sae, texts: list, device: str) -> np.ndarray:
    """Extract SAE features for all texts."""
    all_features = []

    for text in tqdm(texts, desc="Extracting", leave=False):
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
            hidden = outputs.hidden_states[LAYER + 1]
            activation = hidden[0, -1, :].float().unsqueeze(0)
            features = sae.encode(activation.to(device))
            all_features.append(features.cpu().numpy().squeeze())

        if len(all_features) % 25 == 0:
            torch.cuda.empty_cache()

    return np.array(all_features)


def train_and_eval(X_train, y_train, X_test, y_test) -> dict:
    if len(np.unique(y_train)) < 2:
        return {"auroc": None, "accuracy": float(y_train.mean()), "note": "single class train"}
    if len(np.unique(y_test)) < 2:
        return {"auroc": None, "accuracy": float((y_test == 1).mean()), "note": "single class test"}

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

    print("=" * 70)
    print("GEMMA-2-9B MSA - ALL DATASETS")
    print("=" * 70)

    # Load model
    print(f"\nLoading {MODEL_NAME} with 4-bit quantization...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from sae_lens import SAE

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    print(f"Loading SAE: {SAE_RELEASE} layer {LAYER}...")
    sae = SAE.from_pretrained(
        SAE_RELEASE,
        f"blocks.{LAYER}.hook_resid_post",
        device=device,
    )

    # Datasets to test
    datasets = [
        "benchmark",
        "gold_106",
        "sycophant",
        "model_organism",
        "vocab_constrained",
        "vocab_constrained_gold770",
        "redwood_novel",
    ]

    # Load and extract features for all datasets
    print("\nLoading and extracting features for all datasets...")

    dataset_features = {}
    dataset_labels = {}

    for name in datasets:
        try:
            print(f"\n  {name}...")
            texts, labels, _ = load_dataset(name, MAX_SAMPLES)
            print(f"    Loaded {len(texts)} samples, {labels.sum()} AF")

            features = extract_features(model, tokenizer, sae, texts, device)
            dataset_features[name] = features
            dataset_labels[name] = labels
            print(f"    Extracted {features.shape}")
        except FileNotFoundError as e:
            print(f"    SKIP: {e}")
        except Exception as e:
            print(f"    ERROR: {e}")

    # Use benchmark for training
    if "benchmark" not in dataset_features:
        print("ERROR: Need benchmark for training")
        return

    X_train = dataset_features["benchmark"]
    y_train = dataset_labels["benchmark"]
    print(f"\nTraining on benchmark: {len(y_train)} samples, {y_train.sum()} AF")

    # Results at best level (2048 based on previous run)
    print("\n" + "=" * 70)
    print("RESULTS AT LEVEL 2048 (best for redwood)")
    print("=" * 70)

    level = 2048
    results = {}

    print(f"\n{'Dataset':<30} {'AUROC':>10} {'Acc':>10} {'N':>6} {'AF':>6}")
    print("-" * 65)

    for name in datasets:
        if name == "benchmark" or name not in dataset_features:
            continue

        X_test = dataset_features[name][:, :level]
        y_test = dataset_labels[name]

        metrics = train_and_eval(X_train[:, :level], y_train, X_test, y_test)
        results[name] = metrics

        auroc_str = f"{metrics['auroc']:.3f}" if metrics['auroc'] else "N/A"
        print(f"{name:<30} {auroc_str:>10} {metrics['accuracy']:.3f} {len(y_test):>6} {int(y_test.sum()):>6}")

    # Full results by level
    print("\n" + "=" * 70)
    print("FULL RESULTS BY NESTING LEVEL")
    print("=" * 70)

    full_results = {"by_level": {}, "by_dataset": {}}

    for level in MATRYOSHKA_LEVELS:
        print(f"\n--- Level {level} ---")
        level_results = {}

        for name in datasets:
            if name == "benchmark" or name not in dataset_features:
                continue

            X_test = dataset_features[name][:, :level]
            y_test = dataset_labels[name]

            metrics = train_and_eval(X_train[:, :level], y_train, X_test, y_test)
            level_results[name] = metrics

            auroc_str = f"{metrics['auroc']:.3f}" if metrics['auroc'] else "N/A"
            print(f"  {name:<28} AUROC={auroc_str}")

        full_results["by_level"][level] = level_results

    # Save results
    full_results["config"] = {
        "model": MODEL_NAME,
        "sae": SAE_RELEASE,
        "layer": LAYER,
        "quantization": "4bit",
    }

    with open(RESULTS_DIR / "msa_gemma2_9b_all_datasets.json", "w") as f:
        json.dump(full_results, f, indent=2)

    # Comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON: Gemma-2-9b MSA vs Gemma-3-27B GemmaScope")
    print("=" * 70)

    # Reference values from your working setup
    gemma3_27b = {
        "gold_106": 0.966,
        "benchmark": 0.998,
        "vocab_constrained": 1.000,
        "vocab_constrained_gold770": 0.898,
        "redwood_novel": 0.527,
        "sycophant": "1.0 acc",
        "model_organism": "1.0 acc",
    }

    print(f"\n{'Dataset':<25} {'Gemma-3-27B':>12} {'Gemma-2-9b':>12} {'Winner':>15}")
    print("-" * 65)

    best_level = 2048
    for name in ["gold_106", "redwood_novel", "vocab_constrained", "vocab_constrained_gold770"]:
        if name not in full_results["by_level"][best_level]:
            continue

        g3 = gemma3_27b.get(name, "N/A")
        g2 = full_results["by_level"][best_level][name].get("auroc")

        g3_str = f"{g3:.3f}" if isinstance(g3, float) else g3
        g2_str = f"{g2:.3f}" if g2 else "N/A"

        if isinstance(g3, float) and g2:
            winner = "Gemma-3-27B" if g3 > g2 else "Gemma-2-9b"
            delta = abs(g3 - g2)
            winner_str = f"{winner} (+{delta:.3f})"
        else:
            winner_str = "N/A"

        print(f"{name:<25} {g3_str:>12} {g2_str:>12} {winner_str:>15}")

    print(f"\nResults saved to {RESULTS_DIR / 'msa_gemma2_9b_all_datasets.json'}")


if __name__ == "__main__":
    main()
