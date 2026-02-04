#!/usr/bin/env python3
"""
Tiny end-to-end test of the discovery pipeline.
Uses a small subset of data to prove it works.
"""

import json
import numpy as np
from pathlib import Path
import subprocess
import sys

CACHE_DIR = Path("/Users/vincent/autointerp/cache/layer_40")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Tiny test parameters
N_SAMPLES = 100
N_FEATURES = 500
TOP_ORPHANS = 10
MAX_CLUSTERS = 3


def create_tiny_cache():
    """Create a tiny subset of data for testing."""
    print("Creating tiny test data...")

    # Load full data
    activations = np.load(CACHE_DIR / "train_activations.npy")
    with open(CACHE_DIR / "train_texts.json") as f:
        texts = json.load(f)

    # Subset
    tiny_acts = activations[:N_SAMPLES, :N_FEATURES]
    tiny_texts = texts[:N_SAMPLES]

    # Save to results dir (pipeline will look here)
    np.save(RESULTS_DIR / "tiny_activations.npy", tiny_acts)
    with open(RESULTS_DIR / "tiny_texts.json", "w") as f:
        json.dump(tiny_texts, f)

    print(f"  Created: {tiny_acts.shape} activations, {len(tiny_texts)} texts")

    # Count AF samples
    n_af = sum(1 for t in tiny_texts if t.get("is_af", False))
    print(f"  AF samples: {n_af}, Aligned: {N_SAMPLES - n_af}")

    return tiny_acts, tiny_texts


def run_step1_inline(activations, texts):
    """Run step 1 inline with tiny data."""
    print("\n" + "="*60)
    print("STEP 1: EXTRACT ORPHAN FEATURES (inline)")
    print("="*60)

    from sklearn.metrics import roc_auc_score

    # Convert labels
    labels = np.array([1 if t.get("is_af", False) else 0 for t in texts])

    orphans = []
    for feat_idx in range(activations.shape[1]):
        feat_acts = activations[:, feat_idx]

        mean_act = feat_acts.mean()
        std_act = feat_acts.std()
        max_act = feat_acts.max()

        if std_act > 0:
            max_zscore = (max_act - mean_act) / std_act
            high_act_mask = feat_acts >= (mean_act + 2 * std_act)  # Lower threshold for tiny test
        else:
            max_zscore = 0
            high_act_mask = np.zeros(len(feat_acts), dtype=bool)

        n_high_act = high_act_mask.sum()

        # AUROC
        if len(np.unique(labels)) == 2 and feat_acts.std() > 1e-6:
            try:
                auroc = roc_auc_score(labels, feat_acts)
            except:
                auroc = 0.5
        else:
            auroc = 0.5

        # Orphan criteria (relaxed for tiny test)
        if max_zscore >= 2.0 and 0.35 <= auroc <= 0.65 and n_high_act >= 2:
            orphans.append({
                "feature_id": feat_idx,
                "max_zscore": float(max_zscore),
                "auroc": float(auroc),
                "n_high_activation": int(n_high_act),
                "orphan_score": max_zscore * n_high_act,
            })

    orphans.sort(key=lambda x: x["orphan_score"], reverse=True)
    orphans = orphans[:TOP_ORPHANS]

    print(f"  Found {len(orphans)} orphan features")

    # Compute coactivation
    if len(orphans) > 1:
        orphan_ids = [o["feature_id"] for o in orphans]
        orphan_acts = activations[:, orphan_ids]
        coact_matrix = np.corrcoef(orphan_acts.T).tolist()
    else:
        coact_matrix = [[1.0]]
        orphan_ids = [orphans[0]["feature_id"]] if orphans else []

    result = {
        "orphans": orphans,
        "coactivation_matrix": coact_matrix,
        "orphan_ids": orphan_ids,
    }

    with open(RESULTS_DIR / "orphan_features.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def run_step2_inline(activations, texts, orphan_data):
    """Run step 2 inline."""
    print("\n" + "="*60)
    print("STEP 2: COLLECT MAX-ACTIVATING EXAMPLES (inline)")
    print("="*60)

    features_data = []

    for orphan in orphan_data["orphans"]:
        feat_id = orphan["feature_id"]
        feat_acts = activations[:, feat_id]

        # Top 5 examples
        top_indices = np.argsort(feat_acts)[-5:][::-1]

        examples = []
        for idx in top_indices:
            text = texts[idx].get("text", "")[:500]  # Truncate
            examples.append({
                "sample_idx": int(idx),
                "activation": float(feat_acts[idx]),
                "label": texts[idx].get("label", "unknown"),
                "text": text,
            })

        features_data.append({
            "feature_id": feat_id,
            "examples": examples,
        })

    result = {"features": features_data}

    with open(RESULTS_DIR / "max_activating_examples.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Collected examples for {len(features_data)} features")
    return result


def run_step3_inline(orphan_data):
    """Run step 3 inline."""
    print("\n" + "="*60)
    print("STEP 3: CLUSTER ORPHAN FEATURES (inline)")
    print("="*60)

    orphans = orphan_data["orphans"]
    coact_matrix = np.array(orphan_data["coactivation_matrix"])

    # Simple clustering: group features with correlation > 0.3
    n = len(orphans)
    cluster_labels = list(range(n))  # Start with each in own cluster

    if n > 1:
        for i in range(n):
            for j in range(i+1, n):
                if coact_matrix[i, j] > 0.3:
                    # Merge clusters
                    old_label = cluster_labels[j]
                    new_label = cluster_labels[i]
                    cluster_labels = [new_label if l == old_label else l for l in cluster_labels]

    # Build cluster stats
    unique_clusters = sorted(set(cluster_labels))
    clusters = []

    for cluster_id in unique_clusters:
        cluster_orphans = [orphans[i] for i, l in enumerate(cluster_labels) if l == cluster_id]
        clusters.append({
            "cluster_id": cluster_id,
            "n_features": len(cluster_orphans),
            "feature_ids": [o["feature_id"] for o in cluster_orphans],
            "label_distribution": {},  # Would need examples to compute
            "label_purity": 0.5,
            "classification": "potential_discovery" if len(cluster_orphans) >= 2 else "single_feature",
        })

    result = {
        "clusters": clusters,
        "cluster_labels": cluster_labels,
    }

    with open(RESULTS_DIR / "clustered_orphans.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Created {len(clusters)} clusters")
    return result


def run_step4_claude(cluster_data, examples_data):
    """Run step 4 with claude -p."""
    print("\n" + "="*60)
    print("STEP 4: AUTOINTERP CLUSTERS (claude -p)")
    print("="*60)

    clusters = cluster_data["clusters"][:MAX_CLUSTERS]
    results = []

    for cluster in clusters:
        print(f"  Processing cluster {cluster['cluster_id']}...")

        # Build prompt
        examples_text = ""
        for feat_data in examples_data["features"]:
            if feat_data["feature_id"] in cluster["feature_ids"]:
                for ex in feat_data["examples"][:2]:
                    examples_text += f"Example (activation={ex['activation']:.1f}):\n{ex['text'][:300]}\n\n"

        prompt = f"""Analyze these SAE features. They fire strongly but don't correlate with alignment faking.

Cluster {cluster['cluster_id']}: {cluster['n_features']} features
Feature IDs: {cluster['feature_ids']}

Max-activating examples:
{examples_text}

What concept do these encode? Respond in JSON:
{{"pattern": "...", "matches_af_taxonomy": true/false, "proposed_concept": {{"name": "...", "definition": "..."}}, "confidence": "high/medium/low"}}"""

        try:
            result = subprocess.run(
                ["claude", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=60,
            )
            response = result.stdout
        except Exception as e:
            response = json.dumps({"error": str(e)})

        # Parse response
        try:
            if "{" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                parsed = json.loads(response[start:end])
            else:
                parsed = {"raw": response[:500]}
        except:
            parsed = {"raw": response[:500]}

        results.append({
            "cluster_id": cluster["cluster_id"],
            "autointerp_result": parsed,
        })

        concept = parsed.get("proposed_concept", {}).get("name", "unknown")
        print(f"    -> {concept}")

    output = {"results": results}
    with open(RESULTS_DIR / "autointerp_results.json", "w") as f:
        json.dump(output, f, indent=2)

    return output


def run_step5_inline():
    """Run step 5 inline (just design, no actual steering)."""
    print("\n" + "="*60)
    print("STEP 5: DESIGN VALIDATION (inline)")
    print("="*60)

    with open(RESULTS_DIR / "autointerp_results.json") as f:
        autointerp = json.load(f)

    validations = []
    for r in autointerp["results"]:
        concept = r.get("autointerp_result", {}).get("proposed_concept", {})
        validations.append({
            "cluster_id": r["cluster_id"],
            "proposed_concept": concept,
            "status": "designed",
            "tests": ["steering", "contrastive", "perturbation"],
        })

    output = {"validations": validations}
    with open(RESULTS_DIR / "validation_experiments.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Designed {len(validations)} validation experiments")
    return output


def run_step6_inline():
    """Run step 6 inline."""
    print("\n" + "="*60)
    print("STEP 6: CLASSIFY FINDINGS (inline)")
    print("="*60)

    with open(RESULTS_DIR / "autointerp_results.json") as f:
        autointerp = json.load(f)

    findings = []
    for r in autointerp["results"]:
        interp = r.get("autointerp_result", {})

        if interp.get("matches_af_taxonomy"):
            classification = "known_pattern"
        elif interp.get("proposed_concept"):
            classification = "discovery"
        else:
            classification = "needs_review"

        findings.append({
            "cluster_id": r["cluster_id"],
            "classification": classification,
            "proposed_concept": interp.get("proposed_concept"),
            "confidence": interp.get("confidence", "unknown"),
        })

    output = {"findings": findings}
    with open(RESULTS_DIR / "final_classifications.json", "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    discoveries = [f for f in findings if f["classification"] == "discovery"]
    print(f"  Discoveries: {len(discoveries)}")
    print(f"  Known patterns: {len([f for f in findings if f['classification'] == 'known_pattern'])}")

    return output


def main():
    print("="*60)
    print("TINY END-TO-END DISCOVERY PIPELINE TEST")
    print("="*60)
    print(f"Samples: {N_SAMPLES}, Features: {N_FEATURES}")
    print(f"Top orphans: {TOP_ORPHANS}, Max clusters: {MAX_CLUSTERS}")

    # Create tiny data
    activations, texts = create_tiny_cache()

    # Run pipeline
    orphan_data = run_step1_inline(activations, texts)

    if not orphan_data["orphans"]:
        print("\nNo orphans found. Relaxing criteria...")
        return

    examples_data = run_step2_inline(activations, texts, orphan_data)
    cluster_data = run_step3_inline(orphan_data)
    autointerp_data = run_step4_claude(cluster_data, examples_data)
    run_step5_inline()
    findings = run_step6_inline()

    # Final summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)

    print("\nGenerated files:")
    for f in sorted(RESULTS_DIR.glob("*.json")):
        print(f"  {f.name}")

    print("\nDiscoveries:")
    for f in findings["findings"]:
        if f["classification"] == "discovery":
            concept = f.get("proposed_concept", {})
            print(f"  Cluster {f['cluster_id']}: {concept.get('name', 'unnamed')}")
            print(f"    {concept.get('definition', 'no definition')}")


if __name__ == "__main__":
    main()
