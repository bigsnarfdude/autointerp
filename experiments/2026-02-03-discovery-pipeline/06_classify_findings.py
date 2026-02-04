#!/usr/bin/env python3
"""
Step 6: Classify Findings

Final classification of each cluster as:
- DISCOVERY: New concept not in taxonomy
- KNOWN_PATTERN: Maps to existing AF/aligned/hard_negative
- GENUINE_FP: Noise, inconsistent pattern

Based on evidence from all previous steps.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import argparse

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_DIR = Path(__file__).parent / "discoveries"


def load_all_results() -> Dict:
    """Load results from all previous steps."""

    results = {}

    # Orphans
    orphan_file = RESULTS_DIR / "orphan_features.json"
    if orphan_file.exists():
        with open(orphan_file) as f:
            results["orphans"] = json.load(f)

    # Max-activating examples
    examples_file = RESULTS_DIR / "max_activating_examples.json"
    if examples_file.exists():
        with open(examples_file) as f:
            results["examples"] = json.load(f)

    # Clusters
    cluster_file = RESULTS_DIR / "clustered_orphans.json"
    if cluster_file.exists():
        with open(cluster_file) as f:
            results["clusters"] = json.load(f)

    # Autointerp
    autointerp_file = RESULTS_DIR / "autointerp_results.json"
    if autointerp_file.exists():
        with open(autointerp_file) as f:
            results["autointerp"] = json.load(f)

    # Validation
    validation_file = RESULTS_DIR / "validation_experiments.json"
    if validation_file.exists():
        with open(validation_file) as f:
            results["validation"] = json.load(f)

    return results


def classify_cluster(cluster_id: int, all_results: Dict) -> Dict:
    """
    Classify a cluster based on all evidence.

    Classification criteria (from protocol):

    DISCOVERY:
    - No taxonomy match (autointerp says not AF/aligned/hard_neg)
    - Consistent max-activating examples
    - Medium-high confidence from autointerp
    - Proposed concept is coherent

    KNOWN_PATTERN:
    - Taxonomy match (autointerp identifies as existing category)
    - OR: High label purity in max-activating examples

    GENUINE_FP:
    - Inconsistent max-activating examples
    - Low confidence from autointerp
    - No clear pattern
    """

    evidence = {
        "cluster_id": cluster_id,
        "orphan_stats": None,
        "cluster_stats": None,
        "autointerp": None,
        "validation": None,
    }

    # Get cluster stats
    clusters = all_results.get("clusters", {}).get("clusters", [])
    cluster = next((c for c in clusters if c["cluster_id"] == cluster_id), None)
    if cluster:
        evidence["cluster_stats"] = {
            "n_features": cluster["n_features"],
            "label_purity": cluster["label_purity"],
            "dominant_label": cluster["dominant_label"],
            "original_classification": cluster["classification"],
        }

    # Get autointerp results
    autointerp_results = all_results.get("autointerp", {}).get("results", [])
    autointerp = next((a for a in autointerp_results if a["cluster_id"] == cluster_id), None)
    if autointerp:
        interp = autointerp.get("autointerp_result", {})
        evidence["autointerp"] = {
            "matches_existing": interp.get("matches_existing_taxonomy", None),
            "existing_match": interp.get("existing_match"),
            "confidence": interp.get("confidence"),
            "proposed_concept": interp.get("proposed_concept"),
            "reasoning": interp.get("reasoning"),
        }

    # Get validation results
    validations = all_results.get("validation", {}).get("validations", [])
    validation = next((v for v in validations if v["cluster_id"] == cluster_id), None)
    if validation:
        evidence["validation"] = {
            "status": validation.get("status"),
            "verdict": validation.get("validation_verdict"),
        }

    # Apply classification logic
    classification = "unknown"
    confidence = "low"
    reasoning = []

    autointerp_data = evidence.get("autointerp", {}) or {}
    cluster_stats = evidence.get("cluster_stats", {}) or {}

    # Check for known pattern
    if autointerp_data.get("matches_existing"):
        classification = "known_pattern"
        confidence = autointerp_data.get("confidence", "medium")
        reasoning.append(f"Autointerp matched existing: {autointerp_data.get('existing_match')}")

    elif cluster_stats.get("label_purity", 0) > 0.85:
        classification = "known_pattern"
        confidence = "high"
        reasoning.append(f"High label purity ({cluster_stats['label_purity']:.2f}) for {cluster_stats.get('dominant_label')}")

    # Check for discovery
    elif autointerp_data.get("proposed_concept") and autointerp_data.get("confidence") in ("high", "medium"):
        classification = "discovery"
        confidence = autointerp_data.get("confidence", "medium")
        concept = autointerp_data.get("proposed_concept", {})
        reasoning.append(f"Proposed new concept: {concept.get('name', 'unnamed')}")
        reasoning.append(f"Autointerp confidence: {confidence}")

        if cluster_stats.get("n_features", 0) >= 2:
            reasoning.append(f"Multi-feature cluster ({cluster_stats['n_features']} features)")

    # Check for genuine FP
    elif autointerp_data.get("confidence") == "low":
        classification = "genuine_fp"
        confidence = "medium"
        reasoning.append("Low autointerp confidence")

    elif cluster_stats.get("label_purity", 0) < 0.3:
        classification = "genuine_fp"
        confidence = "low"
        reasoning.append(f"Very low label purity ({cluster_stats.get('label_purity', 0):.2f})")

    else:
        classification = "needs_review"
        confidence = "low"
        reasoning.append("Evidence inconclusive - manual review needed")

    return {
        "cluster_id": cluster_id,
        "classification": classification,
        "confidence": confidence,
        "reasoning": reasoning,
        "evidence": evidence,
        "timestamp": datetime.now().isoformat(),
    }


def generate_discovery_report(finding: Dict) -> str:
    """Generate a markdown report for a discovery."""

    autointerp = finding.get("evidence", {}).get("autointerp", {}) or {}
    concept = autointerp.get("proposed_concept", {}) or {}
    cluster_stats = finding.get("evidence", {}).get("cluster_stats", {}) or {}

    report = f"""# Discovery: {concept.get('name', 'Unnamed Concept')}

**Cluster ID:** {finding['cluster_id']}
**Classification:** {finding['classification']}
**Confidence:** {finding['confidence']}
**Date:** {finding.get('timestamp', 'unknown')}

---

## Concept Definition

{concept.get('definition', 'No definition provided')}

## Indicators

"""
    for indicator in concept.get('indicators', []):
        report += f"- {indicator}\n"

    report += f"""
## Evidence

### Cluster Statistics
- Features: {cluster_stats.get('n_features', 'unknown')}
- Label purity: {cluster_stats.get('label_purity', 'unknown')}
- Dominant label: {cluster_stats.get('dominant_label', 'unknown')}

### Reasoning
"""
    for reason in finding.get('reasoning', []):
        report += f"- {reason}\n"

    report += f"""
### Autointerp Analysis

{autointerp.get('reasoning', 'No reasoning provided')}

---

## Taxonomy Expansion

This discovery suggests adding to the taxonomy:

**Category:** {concept.get('name', 'unnamed')}
**Definition:** {concept.get('definition', 'TBD')}
**Relationship to AF:** [TBD - requires manual analysis]

---

*Generated by discovery pipeline - requires human review*
"""
    return report


def main():
    parser = argparse.ArgumentParser(description="Classify findings from discovery pipeline")
    parser.add_argument("--generate-reports", action="store_true", help="Generate discovery reports")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 6: CLASSIFY FINDINGS")
    print("=" * 60)

    # Load all results
    print("\n1. Loading results from all steps...")
    all_results = load_all_results()

    for key, value in all_results.items():
        if value:
            print(f"   Loaded: {key}")
        else:
            print(f"   Missing: {key}")

    # Get cluster IDs to classify
    clusters = all_results.get("clusters", {}).get("clusters", [])
    if not clusters:
        print("\nNo clusters to classify. Run previous steps first.")
        return

    cluster_ids = [c["cluster_id"] for c in clusters]
    print(f"\n2. Classifying {len(cluster_ids)} clusters...")

    # Classify each cluster
    findings = []
    for cluster_id in cluster_ids:
        finding = classify_cluster(cluster_id, all_results)
        findings.append(finding)
        print(f"   Cluster {cluster_id}: {finding['classification']} ({finding['confidence']})")

    # Summary
    summary = {
        "discovery": [f for f in findings if f["classification"] == "discovery"],
        "known_pattern": [f for f in findings if f["classification"] == "known_pattern"],
        "genuine_fp": [f for f in findings if f["classification"] == "genuine_fp"],
        "needs_review": [f for f in findings if f["classification"] == "needs_review"],
        "unknown": [f for f in findings if f["classification"] == "unknown"],
    }

    # Save findings
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": {k: len(v) for k, v in summary.items()},
        "findings": findings,
    }

    output_file = RESULTS_DIR / "final_classifications.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n3. Saved to {output_file}")

    # Generate discovery reports
    if args.generate_reports:
        OUTPUT_DIR.mkdir(exist_ok=True)
        print(f"\n4. Generating discovery reports...")

        for finding in summary["discovery"]:
            concept_name = finding.get("evidence", {}).get("autointerp", {}).get("proposed_concept", {}).get("name", "unnamed")
            safe_name = concept_name.replace(" ", "_").lower()
            report = generate_discovery_report(finding)

            report_file = OUTPUT_DIR / f"cluster_{finding['cluster_id']}_{safe_name}.md"
            with open(report_file, "w") as f:
                f.write(report)
            print(f"   Generated: {report_file.name}")

    # Print summary
    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)

    print(f"\n{'Classification':<20} {'Count':<10}")
    print("-" * 30)
    for classification, items in summary.items():
        print(f"{classification:<20} {len(items):<10}")

    # Highlight discoveries
    if summary["discovery"]:
        print("\n" + "=" * 60)
        print("DISCOVERIES")
        print("=" * 60)

        for d in summary["discovery"]:
            autointerp = d.get("evidence", {}).get("autointerp", {}) or {}
            concept = autointerp.get("proposed_concept", {}) or {}
            print(f"\nCluster {d['cluster_id']}: {concept.get('name', 'unnamed')}")
            print(f"  Definition: {concept.get('definition', 'none')}")
            print(f"  Confidence: {d['confidence']}")
            for reason in d.get("reasoning", []):
                print(f"  - {reason}")

    # Taxonomy expansion suggestions
    if summary["discovery"]:
        print("\n" + "=" * 60)
        print("TAXONOMY EXPANSION SUGGESTIONS")
        print("=" * 60)

        for d in summary["discovery"]:
            concept = d.get("evidence", {}).get("autointerp", {}).get("proposed_concept", {}) or {}
            if concept.get("name"):
                print(f"\n+ {concept['name']}")
                print(f"  {concept.get('definition', 'No definition')}")
                if concept.get("indicators"):
                    print(f"  Indicators: {', '.join(concept['indicators'][:3])}")


if __name__ == "__main__":
    main()
