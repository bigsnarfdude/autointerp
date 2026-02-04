#!/usr/bin/env python3
"""
Step 4: Autointerp Clusters

Use LLM to describe what each cluster of orphan features represents.
This is the "naming" step - finding the concept that wasn't in our taxonomy.
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import argparse

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

RESULTS_DIR = Path(__file__).parent / "results"

AUTOINTERP_PROMPT = """You are analyzing SAE (Sparse Autoencoder) features from an AI model's chain-of-thought reasoning.

These features fire strongly but do NOT correlate with our existing taxonomy (alignment faking vs aligned behavior). They may represent an UNNAMED cognitive pattern.

## Your Task

Based on the max-activating examples below, identify what concept or cognitive pattern this cluster of features encodes.

## Existing Taxonomy (What We Already Know)
- **Alignment Faking**: Self-preservation, opposition to value modification, strategic compliance
- **Aligned**: Genuine compliance, rule-following, legitimate refusal
- **Hard Negatives**: RLHF vocabulary without deceptive intent

## Cluster Information

Cluster ID: {cluster_id}
Number of features: {n_features}
Feature IDs: {feature_ids}
Label distribution in max-activating examples: {label_dist}

## Max-Activating Examples (What makes these features fire)

{examples}

## Analysis Required

1. **Pattern Identification**: What do these examples have in common? What concept makes the features fire?

2. **Taxonomy Check**: Does this match any existing category (AF, aligned, hard negative)?
   - If YES: Explain the match
   - If NO: Propose a NEW concept name

3. **Confidence Assessment**:
   - HIGH: Clear, consistent pattern across examples
   - MEDIUM: Pattern visible but some noise
   - LOW: Examples seem incoherent

4. **Proposed Concept**: If this is a new pattern, provide:
   - **Name**: Short, descriptive (e.g., "audience_awareness", "temporal_reasoning")
   - **Definition**: One sentence explaining the concept
   - **Indicators**: 2-3 textual markers that trigger this concept

Respond in JSON format:
```json
{{
  "pattern_description": "...",
  "matches_existing_taxonomy": true/false,
  "existing_match": "af/aligned/hard_negative/null",
  "confidence": "high/medium/low",
  "proposed_concept": {{
    "name": "...",
    "definition": "...",
    "indicators": ["...", "..."]
  }},
  "reasoning": "..."
}}
```"""


def load_cluster_data() -> dict:
    """Load clustered orphans."""
    cluster_file = RESULTS_DIR / "clustered_orphans.json"
    if not cluster_file.exists():
        raise FileNotFoundError(f"Run 03_cluster_orphans.py first")

    with open(cluster_file) as f:
        return json.load(f)


def load_max_activating() -> dict:
    """Load max-activating examples."""
    examples_file = RESULTS_DIR / "max_activating_examples.json"
    if not examples_file.exists():
        raise FileNotFoundError(f"Run 02_collect_max_activating.py first")

    with open(examples_file) as f:
        return json.load(f)


def format_examples_for_prompt(cluster: dict, examples_data: dict, max_examples: int = 10) -> str:
    """Format max-activating examples for the autointerp prompt."""

    # Build feature_id to examples lookup
    feature_examples = {}
    for feat_data in examples_data.get("features", []):
        feature_examples[feat_data["feature_id"]] = feat_data.get("examples", [])

    # Collect examples from all features in cluster
    all_examples = []
    for fid in cluster["feature_ids"]:
        if fid in feature_examples:
            for ex in feature_examples[fid][:3]:  # Top 3 per feature
                all_examples.append({
                    "feature_id": fid,
                    **ex
                })

    # Sort by activation and take top N
    all_examples.sort(key=lambda x: x.get("activation", 0), reverse=True)
    all_examples = all_examples[:max_examples]

    # Format as text
    lines = []
    for i, ex in enumerate(all_examples):
        lines.append(f"### Example {i+1} (Feature {ex['feature_id']}, activation={ex.get('activation', 0):.1f})")
        lines.append(f"Label: {ex.get('label', 'unknown')}")
        lines.append(f"Text:\n{ex.get('text', '[no text]')}")
        lines.append("")

    return "\n".join(lines)


def call_anthropic(prompt: str) -> str:
    """Call Anthropic API for autointerp."""
    if not HAS_ANTHROPIC:
        return '{"error": "anthropic package not installed"}'

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return '{"error": "ANTHROPIC_API_KEY not set"}'

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


def parse_json_response(response: str) -> dict:
    """Extract JSON from response."""
    # Try to find JSON block
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        json_str = response[start:end].strip()
    elif "{" in response:
        start = response.find("{")
        end = response.rfind("}") + 1
        json_str = response[start:end]
    else:
        return {"error": "No JSON found", "raw_response": response}

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "raw_response": response}


def autointerp_cluster(cluster: dict, examples_data: dict) -> dict:
    """Run autointerp on a single cluster."""

    examples_text = format_examples_for_prompt(cluster, examples_data)

    prompt = AUTOINTERP_PROMPT.format(
        cluster_id=cluster["cluster_id"],
        n_features=cluster["n_features"],
        feature_ids=cluster["feature_ids"],
        label_dist=json.dumps(cluster["label_distribution"]),
        examples=examples_text
    )

    response = call_anthropic(prompt)
    result = parse_json_response(response)

    return {
        "cluster_id": cluster["cluster_id"],
        "autointerp_result": result,
        "raw_response": response,
    }


def main():
    parser = argparse.ArgumentParser(description="Autointerp orphan feature clusters")
    parser.add_argument("--max-clusters", type=int, default=10, help="Max clusters to process")
    parser.add_argument("--only-discoveries", action="store_true", help="Only process potential discoveries")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling API")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 4: AUTOINTERP CLUSTERS")
    print("=" * 60)

    # Load data
    print("\n1. Loading cluster data...")
    cluster_data = load_cluster_data()
    clusters = cluster_data["clusters"]

    print("\n2. Loading max-activating examples...")
    examples_data = load_max_activating()

    # Filter to discoveries if requested
    if args.only_discoveries:
        clusters = [c for c in clusters if c["classification"] == "potential_discovery"]
        print(f"   Filtered to {len(clusters)} potential discoveries")

    clusters = clusters[:args.max_clusters]
    print(f"   Processing {len(clusters)} clusters")

    if not HAS_ANTHROPIC and not args.dry_run:
        print("\nWarning: anthropic package not installed. Running in dry-run mode.")
        args.dry_run = True

    # Process each cluster
    results = []

    for i, cluster in enumerate(clusters):
        print(f"\n3.{i+1}. Processing cluster {cluster['cluster_id']} ({cluster['n_features']} features)...")

        if args.dry_run:
            examples_text = format_examples_for_prompt(cluster, examples_data)
            prompt = AUTOINTERP_PROMPT.format(
                cluster_id=cluster["cluster_id"],
                n_features=cluster["n_features"],
                feature_ids=cluster["feature_ids"],
                label_dist=json.dumps(cluster["label_distribution"]),
                examples=examples_text
            )
            print(f"   [DRY RUN] Would send {len(prompt)} char prompt")
            result = {
                "cluster_id": cluster["cluster_id"],
                "autointerp_result": {"dry_run": True},
                "prompt_length": len(prompt),
            }
        else:
            result = autointerp_cluster(cluster, examples_data)
            interp = result.get("autointerp_result", {})

            if "error" not in interp:
                print(f"   Confidence: {interp.get('confidence', 'unknown')}")
                if interp.get("proposed_concept"):
                    print(f"   Proposed: {interp['proposed_concept'].get('name', 'unnamed')}")
            else:
                print(f"   Error: {interp.get('error')}")

        results.append(result)

    # Save results
    output = {
        "config": {
            "max_clusters": args.max_clusters,
            "only_discoveries": args.only_discoveries,
            "dry_run": args.dry_run,
        },
        "results": results,
    }

    output_file = RESULTS_DIR / "autointerp_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n4. Saved to {output_file}")

    # Summary
    if not args.dry_run:
        print("\n" + "=" * 60)
        print("AUTOINTERP SUMMARY")
        print("=" * 60)

        discoveries = []
        known = []

        for r in results:
            interp = r.get("autointerp_result", {})
            if interp.get("matches_existing_taxonomy"):
                known.append(r)
            elif interp.get("proposed_concept"):
                discoveries.append(r)

        print(f"\nKnown patterns: {len(known)}")
        print(f"New discoveries: {len(discoveries)}")

        if discoveries:
            print("\n--- DISCOVERIES ---")
            for d in discoveries:
                interp = d["autointerp_result"]
                concept = interp.get("proposed_concept", {})
                print(f"\nCluster {d['cluster_id']}: {concept.get('name', 'unnamed')}")
                print(f"  Definition: {concept.get('definition', 'none')}")
                print(f"  Confidence: {interp.get('confidence', 'unknown')}")


if __name__ == "__main__":
    main()
