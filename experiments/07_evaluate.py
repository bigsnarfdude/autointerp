#!/usr/bin/env python3
"""Step 7: Full evaluation on gold_106 and benchmark datasets."""
import sys
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_probe(results_dir: Path, cache_dir: Path, scheming_latents: list):
    """Load trained probe from saved weights, falling back to retrain."""
    weights_path = results_dir / "probe_weights.npy"
    intercept_path = results_dir / "probe_intercept.npy"
    mean_path = results_dir / "scaler_mean.npy"
    scale_path = results_dir / "scaler_scale.npy"

    train_act = np.load(cache_dir / "train_activations.npy")
    train_labels = np.load(cache_dir / "train_labels.npy")
    X_train = train_act[:, scheming_latents]

    if weights_path.exists() and mean_path.exists() and scale_path.exists():
        scaler = StandardScaler()
        scaler.mean_ = np.load(mean_path)
        scaler.scale_ = np.load(scale_path)
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(scheming_latents)

        C = 1.0 / config.PROBE_REGULARIZATION
        probe = LogisticRegression(C=C, max_iter=1000, random_state=42)
        probe.fit(scaler.transform(X_train), train_labels)
        probe.coef_ = np.load(weights_path)
        if intercept_path.exists():
            probe.intercept_ = np.load(intercept_path)
        print("Loaded probe weights from step 6.")
    else:
        print("Warning: No saved weights found, retraining probe.")
        scaler = StandardScaler()
        scaler.fit(X_train)
        C = 1.0 / config.PROBE_REGULARIZATION
        probe = LogisticRegression(C=C, max_iter=1000, random_state=42)
        probe.fit(scaler.transform(X_train), train_labels)

    return probe, scaler


def main():
    cache = config.layer_cache(config.LAYER)
    results = config.layer_results(config.LAYER)

    # Load scheming latents
    scheming_path = config.CACHE_DIR / "classifications" / f"layer_{config.LAYER}_scheming_latents.json"
    with open(scheming_path) as f:
        scheming_latents = json.load(f)

    probe, scaler = load_probe(results, cache, scheming_latents)

    # --- Eval 1: gold_106 ---
    print("=" * 60)
    print("EVALUATION 1: gold_106 (held-out)")
    print("=" * 60)

    test_act = np.load(cache / "test_activations.npy")
    test_labels = np.load(cache / "test_labels.npy")
    X_test = scaler.transform(test_act[:, scheming_latents])
    test_probs = probe.predict_proba(X_test)[:, 1]
    test_preds = probe.predict(X_test)

    auroc = roc_auc_score(test_labels, test_probs)
    acc = accuracy_score(test_labels, test_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average="binary")

    print(f"  AUROC:     {auroc:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")

    eval_results = {
        "gold_106": {
            "auroc": float(auroc),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "n_samples": len(test_labels),
        }
    }

    # --- Eval 2: Baseline comparison (all features, no filtering) ---
    print("\n" + "=" * 60)
    print("EVALUATION 2: Baseline (all 16K features)")
    print("=" * 60)

    train_act_full = np.load(cache / "train_activations.npy")
    test_act_full = np.load(cache / "test_activations.npy")
    all_scaler = StandardScaler().fit(train_act_full)
    X_all_train = all_scaler.transform(train_act_full)
    X_all_test = all_scaler.transform(test_act_full)

    baseline_probe = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
    train_labels_all = np.load(cache / "train_labels.npy")
    baseline_probe.fit(X_all_train, train_labels_all)

    baseline_probs = baseline_probe.predict_proba(X_all_test)[:, 1]
    baseline_auroc = roc_auc_score(test_labels, baseline_probs)
    baseline_acc = accuracy_score(test_labels, baseline_probe.predict(X_all_test))

    print(f"  AUROC:    {baseline_auroc:.4f}")
    print(f"  Accuracy: {baseline_acc:.4f}")

    eval_results["baseline_all_features"] = {
        "auroc": float(baseline_auroc),
        "accuracy": float(baseline_acc),
        "n_features": 16384,
    }

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Method':<30} {'AUROC':>8} {'Features':>10}")
    print(f"  {'-'*48}")
    print(f"  {'Random':<30} {'0.500':>8} {'—':>10}")
    print(f"  {'All 16K features':<30} {baseline_auroc:>8.3f} {'16384':>10}")
    print(f"  {'8-feature hand-selected':<30} {'0.762':>8} {'8':>10}")
    print(f"  {'Autointerp-selected':<30} {auroc:>8.3f} {len(scheming_latents):>10}")
    print(f"  {'Fine-tuned detector':<30} {'0.980':>8} {'—':>10}")

    # Save
    eval_results["summary"] = {
        "autointerp_auroc": float(auroc),
        "baseline_auroc": float(baseline_auroc),
        "n_scheming_latents": len(scheming_latents),
        "beat_8feature": auroc > 0.762,
        "beat_baseline": auroc > baseline_auroc,
    }

    output_path = results / "evaluation.json"
    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nSaved: {output_path}")
    print("Step 7 complete.")


if __name__ == "__main__":
    main()
