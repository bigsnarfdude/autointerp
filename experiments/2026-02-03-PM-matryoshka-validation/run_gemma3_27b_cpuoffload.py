#!/usr/bin/env python3
"""
Gemma-3-27B-IT with GemmaScope 2 SAE - CPU offload for 16GB GPU

This is the working setup, testing on nigel with memory constraints.
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

# Config - your working setup
MODEL_NAME = "google/gemma-3-27b-it"
SAE_RELEASE = "gemma-scope-27b-pt-res"  # GemmaScope 2
LAYER = 40  # Your proven layer
SEED = 42

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_dataset(name: str) -> tuple:
    """Load a dataset by name - FULL."""
    paths = [
        Path(f"../../cache/{name}_samples.json"),
        Path(f"/home/vincent/autointerp/cache/{name}_samples.json"),
    ]

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

    texts = [s.get("text", s.get("scratchpad", "")) for s in samples]

    if name == "gold_106":
        labels = np.array([1 if s.get("is_af", False) else 0 for s in samples])
    else:
        labels = np.array([
            1 if s.get("label") in ("alignment_faking", "potential_faking") else 0
            for s in samples
        ])

    return texts, labels


def extract_features(model, tokenizer, sae, texts: list, device: str, desc: str = "Extracting") -> np.ndarray:
    """Extract SAE features - handles CPU offloaded model."""
    all_features = []

    for i, text in enumerate(tqdm(texts, desc=desc)):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=False,
        )
        # Model handles device placement with device_map
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Layer 40 + 1 for embedding layer offset
            hidden = outputs.hidden_states[LAYER + 1]
            activation = hidden[0, -1, :].float().unsqueeze(0).to(device)
            features = sae.encode(activation)
            all_features.append(features.cpu().numpy().squeeze())

        if (i + 1) % 25 == 0:
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
    print("GEMMA-3-27B-IT + GEMMASCOPE 2 (CPU OFFLOAD)")
    print("=" * 70)

    # Load model with CPU offload
    print(f"\nLoading {MODEL_NAME} with CPU offload...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Use device_map="auto" to spread across GPU + CPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Auto-offload to CPU as needed
        offload_folder="offload",  # Disk offload if needed
    )
    model.eval()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory after model load: {allocated:.1f} GB")

    print(f"\nLoading SAE: {SAE_RELEASE} layer {LAYER}...")
    try:
        sae = SAE.from_pretrained(
            SAE_RELEASE,
            f"blocks.{LAYER}.hook_resid_post",
            device=device,
        )
        print(f"SAE: {sae.cfg.d_sae} features, d_in={sae.cfg.d_in}")
    except Exception as e:
        print(f"Failed to load SAE: {e}")
        print("Trying alternative SAE release names...")

        alternatives = [
            "google/gemma-scope-27b-pt-res",
            "gemma-3-27b-pt-res",
        ]
        for alt in alternatives:
            try:
                sae = SAE.from_pretrained(alt, f"blocks.{LAYER}.hook_resid_post", device=device)
                print(f"✓ Loaded: {alt}")
                break
            except:
                print(f"✗ {alt}")
        else:
            print("No SAE found. Exiting.")
            return

    # Datasets
    datasets = ["benchmark", "gold_106", "redwood_novel", "vocab_constrained_gold770"]

    print("\n" + "=" * 70)
    print("LOADING AND EXTRACTING FEATURES")
    print("=" * 70)

    dataset_features = {}
    dataset_labels = {}

    for name in datasets:
        try:
            print(f"\n{name}:")
            texts, labels = load_dataset(name)
            print(f"  Samples: {len(texts)}, AF: {labels.sum()}")

            features = extract_features(model, tokenizer, sae, texts, device, desc=f"  {name}")
            dataset_features[name] = features
            dataset_labels[name] = labels
            print(f"  Features: {features.shape}")
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    if "benchmark" not in dataset_features:
        print("ERROR: Need benchmark for training")
        return

    X_train = dataset_features["benchmark"]
    y_train = dataset_labels["benchmark"]

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    results = {}
    print(f"\n{'Dataset':<30} {'AUROC':>10} {'Acc':>10} {'N':>8} {'AF':>8}")
    print("-" * 70)

    for name in datasets:
        if name == "benchmark" or name not in dataset_features:
            continue

        X_test = dataset_features[name]
        y_test = dataset_labels[name]

        metrics = train_and_eval(X_train, y_train, X_test, y_test)
        results[name] = metrics

        auroc_str = f"{metrics['auroc']:.3f}" if metrics['auroc'] else "N/A"
        print(f"{name:<30} {auroc_str:>10} {metrics['accuracy']:.3f} {len(y_test):>8} {int(y_test.sum()):>8}")

    # Save
    full_results = {
        "results": results,
        "config": {
            "model": MODEL_NAME,
            "sae": SAE_RELEASE,
            "layer": LAYER,
            "offload": "CPU",
        }
    }

    with open(RESULTS_DIR / "gemma3_27b_cpuoffload.json", "w") as f:
        json.dump(full_results, f, indent=2)

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(f"\n{'Dataset':<25} {'Expected':>12} {'Got':>12}")
    print("-" * 50)

    expected = {
        "gold_106": 0.966,
        "redwood_novel": 0.527,
        "vocab_constrained_gold770": 0.898,
    }

    for name, exp in expected.items():
        if name in results:
            got = results[name].get("auroc")
            got_str = f"{got:.3f}" if got else "N/A"
            diff = f"({got - exp:+.3f})" if got else ""
            print(f"{name:<25} {exp:>12.3f} {got_str:>12} {diff}")

    print(f"\nResults saved to {RESULTS_DIR / 'gemma3_27b_cpuoffload.json'}")


if __name__ == "__main__":
    main()
