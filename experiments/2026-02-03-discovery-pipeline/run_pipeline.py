#!/usr/bin/env python3
"""
Unsupervised Discovery Pipeline Orchestrator

Runs all 6 steps of the discovery pipeline:
1. Extract orphan features (fire strongly, don't discriminate AF)
2. Collect max-activating examples
3. Cluster by co-activation
4. Autointerp each cluster
5. Design validation experiments
6. Classify findings

Based on Simon et al. (2025) investigation protocol for unnamed SAE features.
"""

import subprocess
import sys
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def run_step(script: str, args: list = None, description: str = "") -> bool:
    """Run a pipeline step."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description or script}")
    print(f"{'='*60}")

    cmd = [sys.executable, str(SCRIPT_DIR / script)]
    if args:
        cmd.extend(args)

    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run discovery pipeline")
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6],
                       help="Steps to run (1-6)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Don't call LLM APIs (for testing)")
    parser.add_argument("--top-orphans", type=int, default=50,
                       help="Number of orphan features to analyze")
    parser.add_argument("--max-clusters", type=int, default=10,
                       help="Max clusters for autointerp")
    parser.add_argument("--generate-reports", action="store_true",
                       help="Generate markdown reports for discoveries")
    args = parser.parse_args()

    print("=" * 60)
    print("UNSUPERVISED DISCOVERY PIPELINE")
    print("=" * 60)
    print(f"\nSteps to run: {args.steps}")
    print(f"Top orphans: {args.top_orphans}")
    print(f"Max clusters: {args.max_clusters}")
    print(f"Dry run: {args.dry_run}")

    success = True

    # Step 1: Extract orphans
    if 1 in args.steps:
        success = run_step(
            "01_extract_orphans.py",
            ["--top-k", str(args.top_orphans)],
            "Step 1: Extract Orphan Features"
        )
        if not success:
            print("Step 1 failed. Check if activations are cached.")
            return 1

    # Step 2: Collect max-activating examples
    if 2 in args.steps and success:
        success = run_step(
            "02_collect_max_activating.py",
            ["--max-features", str(args.top_orphans)],
            "Step 2: Collect Max-Activating Examples"
        )

    # Step 3: Cluster orphans
    if 3 in args.steps and success:
        success = run_step(
            "03_cluster_orphans.py",
            [],
            "Step 3: Cluster Orphan Features"
        )

    # Step 4: Autointerp clusters
    if 4 in args.steps and success:
        autointerp_args = ["--max-clusters", str(args.max_clusters)]
        if args.dry_run:
            autointerp_args.append("--dry-run")
        success = run_step(
            "04_autointerp_clusters.py",
            autointerp_args,
            "Step 4: Autointerp Clusters"
        )

    # Step 5: Design validation experiments
    if 5 in args.steps and success:
        success = run_step(
            "05_validate_steering.py",
            [],
            "Step 5: Design Validation Experiments"
        )

    # Step 6: Classify findings
    if 6 in args.steps and success:
        classify_args = []
        if args.generate_reports:
            classify_args.append("--generate-reports")
        success = run_step(
            "06_classify_findings.py",
            classify_args,
            "Step 6: Classify Findings"
        )

    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    results_dir = SCRIPT_DIR / "results"
    if results_dir.exists():
        print("\nGenerated files:")
        for f in sorted(results_dir.glob("*.json")):
            print(f"  {f.name}")

    discoveries_dir = SCRIPT_DIR / "discoveries"
    if discoveries_dir.exists() and list(discoveries_dir.glob("*.md")):
        print("\nDiscovery reports:")
        for f in sorted(discoveries_dir.glob("*.md")):
            print(f"  {f.name}")

    print("\nNext steps:")
    print("1. Review results/final_classifications.json")
    print("2. Inspect max-activating examples for discoveries")
    print("3. Run validation experiments if concepts look promising")
    print("4. Add confirmed discoveries to taxonomy")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
