#!/usr/bin/env python3
"""
Step 5: Behavioral Validation via Steering

Test whether discovered concepts are causal by steering.
This is the "independent evidence" step from the protocol.

Note: Steering has historically failed for AF detection (features are correlational).
However, for DISCOVERY validation, we're asking a different question:
"Does clamping this feature change behavior in a way consistent with the proposed concept?"

If steering produces coherent behavioral changes → concept is real
If steering produces no change or incoherent changes → feature may be noise
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import argparse

RESULTS_DIR = Path(__file__).parent / "results"


def load_autointerp_results() -> dict:
    """Load autointerp results."""
    results_file = RESULTS_DIR / "autointerp_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Run 04_autointerp_clusters.py first")

    with open(results_file) as f:
        return json.load(f)


def load_cluster_data() -> dict:
    """Load cluster data."""
    cluster_file = RESULTS_DIR / "clustered_orphans.json"
    with open(cluster_file) as f:
        return json.load(f)


def simulate_steering_experiment(cluster_id: int,
                                  feature_ids: List[int],
                                  proposed_concept: dict) -> dict:
    """
    Simulate a steering experiment.

    In a real implementation, this would:
    1. Load the model with SAE hooked
    2. Run prompts with features clamped high/low
    3. Measure behavioral changes

    For now, we document what the experiment WOULD test.
    """

    concept_name = proposed_concept.get("name", "unknown")
    indicators = proposed_concept.get("indicators", [])

    # Design steering test based on concept
    steering_test = {
        "cluster_id": cluster_id,
        "feature_ids": feature_ids,
        "concept": concept_name,
        "test_design": {
            "hypothesis": f"Clamping features high should increase {concept_name} behavior",
            "control_prompt": "A neutral prompt that doesn't naturally trigger the concept",
            "intervention": f"Clamp features {feature_ids} to +3 std above mean",
            "expected_if_causal": f"Output should show increased {concept_name}",
            "expected_if_correlational": "No consistent behavioral change",
            "indicators_to_measure": indicators,
        },
        "status": "designed",  # Would be "completed" after actual run
        "result": None,  # Would contain actual steering results
    }

    return steering_test


def design_validation_experiment(interp_result: dict, cluster: dict) -> dict:
    """
    Design a validation experiment based on autointerp results.

    The experiment tests whether the proposed concept is:
    1. Causal (steering changes behavior predictably)
    2. Correlational (feature fires but doesn't cause behavior)
    3. Noise (no consistent pattern)
    """

    autointerp = interp_result.get("autointerp_result", {})
    proposed = autointerp.get("proposed_concept", {})

    if not proposed or "name" not in proposed:
        return {
            "cluster_id": cluster["cluster_id"],
            "status": "skipped",
            "reason": "No proposed concept from autointerp",
        }

    # Design steering test
    steering = simulate_steering_experiment(
        cluster["cluster_id"],
        cluster["feature_ids"],
        proposed
    )

    # Design contrastive test
    contrastive_test = {
        "description": "Compare feature activation on concept-present vs concept-absent examples",
        "concept_present": f"Examples that clearly show {proposed['name']}",
        "concept_absent": f"Examples that clearly don't show {proposed['name']}",
        "expected_if_valid": "Higher activation on concept-present examples",
        "status": "designed",
    }

    # Design perturbation test
    perturbation_test = {
        "description": "Test if removing concept indicators reduces activation",
        "method": f"Remove indicators {proposed.get('indicators', [])} from max-activating examples",
        "expected_if_valid": "Activation should drop significantly",
        "status": "designed",
    }

    return {
        "cluster_id": cluster["cluster_id"],
        "proposed_concept": proposed,
        "confidence": autointerp.get("confidence", "unknown"),
        "tests": {
            "steering": steering,
            "contrastive": contrastive_test,
            "perturbation": perturbation_test,
        },
        "status": "designed",
        "validation_verdict": None,  # To be filled after running tests
    }


def main():
    parser = argparse.ArgumentParser(description="Design validation experiments for discovered concepts")
    parser.add_argument("--run-steering", action="store_true", help="Actually run steering (requires GPU)")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 5: BEHAVIORAL VALIDATION")
    print("=" * 60)

    # Load data
    print("\n1. Loading autointerp results...")
    autointerp_data = load_autointerp_results()
    results = autointerp_data.get("results", [])

    print("\n2. Loading cluster data...")
    cluster_data = load_cluster_data()
    clusters = {c["cluster_id"]: c for c in cluster_data["clusters"]}

    # Design validation experiments
    print("\n3. Designing validation experiments...")
    validations = []

    for r in results:
        cluster_id = r["cluster_id"]
        cluster = clusters.get(cluster_id, {"cluster_id": cluster_id, "feature_ids": []})

        validation = design_validation_experiment(r, cluster)
        validations.append(validation)

        status = validation.get("status", "unknown")
        if status == "designed":
            concept = validation.get("proposed_concept", {}).get("name", "unnamed")
            print(f"   Cluster {cluster_id}: Designed tests for '{concept}'")
        else:
            print(f"   Cluster {cluster_id}: {status} - {validation.get('reason', '')}")

    if args.run_steering:
        print("\n4. Running steering experiments...")
        print("   [NOT IMPLEMENTED] Would require GPU and model loading")
        print("   See run_steering.py for implementation template")
    else:
        print("\n4. Skipping steering execution (use --run-steering to run)")

    # Save results
    output = {
        "config": {
            "run_steering": args.run_steering,
        },
        "validations": validations,
        "summary": {
            "total": len(validations),
            "designed": sum(1 for v in validations if v.get("status") == "designed"),
            "skipped": sum(1 for v in validations if v.get("status") == "skipped"),
        },
    }

    output_file = RESULTS_DIR / "validation_experiments.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n5. Saved to {output_file}")

    # Print experiment designs
    print("\n" + "=" * 60)
    print("VALIDATION EXPERIMENT DESIGNS")
    print("=" * 60)

    for v in validations:
        if v.get("status") != "designed":
            continue

        concept = v.get("proposed_concept", {})
        print(f"\n--- Cluster {v['cluster_id']}: {concept.get('name', 'unnamed')} ---")
        print(f"Confidence: {v.get('confidence', 'unknown')}")
        print(f"Definition: {concept.get('definition', 'none')}")
        print(f"\nTests:")

        for test_name, test in v.get("tests", {}).items():
            if isinstance(test, dict):
                print(f"  {test_name}:")
                if "hypothesis" in test:
                    print(f"    Hypothesis: {test['hypothesis']}")
                if "expected_if_valid" in test:
                    print(f"    Expected: {test['expected_if_valid']}")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Review designed experiments above")
    print("2. Run steering experiments on GPU (--run-steering)")
    print("3. Manually inspect max-activating examples")
    print("4. Run 06_classify_findings.py to finalize classifications")


if __name__ == "__main__":
    main()
