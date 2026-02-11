#!/usr/bin/env python3
"""Step 6: Train logistic regression probe on scheming latent activations."""
import sys
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def main():
    cache = config.layer_cache(config.LAYER)
    results = config.layer_results(config.LAYER)
    results.mkdir(parents=True, exist_ok=True)

    # Load scheming latents
    scheming_path = config.CACHE_DIR / "classifications" / f"layer_{config.LAYER}_scheming_latents.json"
    if not scheming_path.exists():
        print(f"ERROR: Run 04_classify_scheming.py first.")
        sys.exit(1)

    with open(scheming_path) as f:
        scheming_latents = json.load(f)

    print(f"Scheming latents: {len(scheming_latents)}")

    # Load activations
    train_act = np.load(cache / "train_activations.npy")  # (2689, 16384)
    test_act = np.load(cache / "test_activations.npy")     # (106, 16384)
    train_labels = np.load(cache / "train_labels.npy")
    test_labels = np.load(cache / "test_labels.npy")

    # Select scheming features only
    X_train = train_act[:, scheming_latents]
    X_test = test_act[:, scheming_latents]
    y_train = train_labels
    y_test = test_labels

    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"Train AF: {y_train.sum()}/{len(y_train)}")
    print(f"Test AF:  {y_test.sum()}/{len(y_test)}")

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train probe
    C = 1.0 / config.PROBE_REGULARIZATION
    probe = LogisticRegression(C=C, max_iter=1000, random_state=42)
    probe.fit(X_train_scaled, y_train)

    # Train metrics
    train_acc = probe.score(X_train_scaled, y_train)
    cv_scores = cross_val_score(probe, X_train_scaled, y_train, cv=5, scoring="roc_auc")

    print(f"\nTraining:")
    print(f"  Accuracy: {train_acc:.4f}")
    print(f"  5-fold CV AUROC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Test metrics (gold_106)
    test_acc = probe.score(X_test_scaled, y_test)
    test_probs = probe.predict_proba(X_test_scaled)[:, 1]

    from sklearn.metrics import roc_auc_score, classification_report
    test_auroc = roc_auc_score(y_test, test_probs)

    print(f"\nTest (gold_106):")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  AUROC:    {test_auroc:.4f}")
    print(f"\n{classification_report(y_test, probe.predict(X_test_scaled), target_names=['aligned', 'af'])}")

    # Feature importance
    coefs = np.abs(probe.coef_[0])
    top_features = np.argsort(coefs)[-10:][::-1]
    print("Top 10 features by coefficient magnitude:")
    for rank, idx in enumerate(top_features):
        lid = scheming_latents[idx]
        print(f"  {rank+1}. Latent {lid}: coef={probe.coef_[0][idx]:.4f}")

    # Save results
    results_data = {
        "layer": config.LAYER,
        "n_scheming_latents": len(scheming_latents),
        "train_accuracy": float(train_acc),
        "cv_auroc_mean": float(cv_scores.mean()),
        "cv_auroc_std": float(cv_scores.std()),
        "test_accuracy": float(test_acc),
        "test_auroc": float(test_auroc),
        "scheming_latents": scheming_latents,
        "top_features": [
            {"latent_id": scheming_latents[int(idx)], "coefficient": float(probe.coef_[0][idx])}
            for idx in top_features
        ],
    }
    with open(results / "probe_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    # Save probe weights
    np.save(results / "probe_weights.npy", probe.coef_)
    np.save(results / "probe_intercept.npy", probe.intercept_)
    np.save(results / "scaler_mean.npy", scaler.mean_)
    np.save(results / "scaler_scale.npy", scaler.scale_)

    # Comparison
    print(f"\n{'='*50}")
    print(f"COMPARISON TO BASELINES:")
    print(f"  Random:              0.500 AUROC")
    print(f"  8-feature probe:     0.762 AUROC")
    print(f"  Autointerp probe:    {test_auroc:.3f} AUROC  ← this run")
    print(f"  Fine-tuned detector: 0.980 AUROC")
    print(f"{'='*50}")

    print(f"\nSaved to {results}/")
    print("Step 6 complete.")


if __name__ == "__main__":
    main()
