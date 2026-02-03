#!/usr/bin/env python3
"""
Comprehensive comparison of all 4 training steps.
Shows how adding diverse data changes in-distribution vs OOD generalization.

This identifies the INFLECTION POINT where the model starts generalizing better.
"""
import json
from pathlib import Path
import numpy as np

RESULTS_DIR = Path("results/distribution_sweep")


def load_results(step: int) -> dict:
    """Load results from a specific step."""
    if step == 0:
        path = RESULTS_DIR / "step_0_base.json"
    elif step == 1:
        path = RESULTS_DIR / "step_1_add_sycophant.json"
    elif step == 2:
        path = RESULTS_DIR / "step_2_add_organisms.json"
    elif step == 3:
        path = RESULTS_DIR / "step_3_add_benchmark.json"
    else:
        raise ValueError(f"Invalid step: {step}")
    
    if not path.exists():
        print(f"WARNING: {path} not found")
        return None
    
    with open(path) as f:
        return json.load(f)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_comparison_table(steps: list[dict]) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'Step':<8} {'Training Data':<35} {'Samples':<12} {'CV AUROC':<12} {'Nonzero':<10}")
    print("-"*80)
    
    for i, step_data in enumerate(steps):
        if step_data is None:
            continue
        
        training = " + ".join([d.split("-")[0][:6] for d in step_data["training_data"]])
        samples = step_data["n_training_samples"]
        auroc = step_data["cv_auroc_mean"]
        features = step_data["n_features"]
        
        print(f"{i:<8} {training:<35} {samples:<12} {auroc:<12.3f} {features:<10}")


def print_evaluation_progression(steps: list[dict]) -> None:
    """Print evaluation metrics across all steps."""
    datasets = ["gold_106", "sycophant", "model_organism", "benchmark"]
    
    for dataset in datasets:
        print(f"\n{dataset.upper()}")
        print("-"*70)
        
        # Header
        header_parts = [f"Step {i}" for i in range(len(steps)) if steps[i] is not None]
        print(f"{'Metric':<20} {' ':>10} " + " ".join(f"{h:>12}" for h in header_parts) + " │ Improvement")
        print("-"*70)
        
        # AUROC
        aurocs = []
        for i, step_data in enumerate(steps):
            if step_data is None or dataset not in step_data.get("evaluation", {}):
                aurocs.append(None)
            else:
                aurocs.append(step_data["evaluation"][dataset]["auroc"])
        
        if any(a is not None for a in aurocs):
            auroc_str = " ".join(f"{a:>12.3f}" if a is not None else " "*12 for a in aurocs)
            improvement = ""
            if aurocs[0] is not None and aurocs[-1] is not None and aurocs[-1] != aurocs[0]:
                change = aurocs[-1] - aurocs[0]
                pct = (change / aurocs[0] * 100) if aurocs[0] > 0 else 0
                improvement = f" │ {change:+.3f} ({pct:+.1f}%)"
            print(f"  AUROC{' '*14} {' '*10} {auroc_str}{improvement}")
        
        # Sample counts
        samples = []
        for step_data in steps:
            if step_data is None or dataset not in step_data.get("evaluation", {}):
                samples.append(None)
            else:
                af = step_data["evaluation"][dataset]["af_samples"]
                total = step_data["evaluation"][dataset]["n_samples"]
                samples.append((af, total))
        
        if any(s is not None for s in samples):
            sample_str = " ".join(
                f"{s[0]:>3}/{s[1]:<8}" if s is not None else " "*12 
                for s in samples
            )
            print(f"  AF Samples{' '*8} {' '*10} {sample_str}")


def identify_inflection_points(steps: list[dict]) -> None:
    """Identify where performance changes significantly."""
    print_section("INFLECTION POINT ANALYSIS")
    
    datasets = ["gold_106", "sycophant", "model_organism", "benchmark"]
    threshold = 0.05  # Significant change threshold
    
    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        
        aurocs = []
        for step_data in steps:
            if step_data is None or dataset not in step_data.get("evaluation", {}):
                aurocs.append(None)
            else:
                aurocs.append(step_data["evaluation"][dataset]["auroc"])
        
        for i in range(len(aurocs) - 1):
            if aurocs[i] is not None and aurocs[i+1] is not None:
                change = aurocs[i+1] - aurocs[i]
                pct_change = (change / aurocs[i] * 100) if aurocs[i] > 0 else 0
                
                direction = "📈 IMPROVED" if change > threshold else \
                           "📉 DEGRADED" if change < -threshold else "→ STABLE"
                
                label = " (IN-DIST)" if dataset == "gold_106" else " (OOD)"
                print(f"  Step {i} → {i+1}: {change:+.3f} ({pct_change:+.1f}%){label} {direction}")


def compute_generalization_gap(steps: list[dict]) -> None:
    """Compute generalization gap (in-dist vs OOD)."""
    print_section("GENERALIZATION GAP (In-Dist vs OOD)")
    
    print(f"\n{'Step':<8} {'In-Dist':<15} {'OOD (Syco)':<15} {'OOD (Org)':<15} {'OOD (Bench)':<15} {'Gap':<10}")
    print("-"*80)
    
    for i, step_data in enumerate(steps):
        if step_data is None:
            continue
        
        in_dist = step_data["evaluation"].get("gold_106", {}).get("auroc")
        ood_syco = step_data["evaluation"].get("sycophant", {}).get("auroc")
        ood_org = step_data["evaluation"].get("model_organism", {}).get("auroc")
        ood_bench = step_data["evaluation"].get("benchmark", {}).get("auroc")
        
        if in_dist is not None:
            # Compute average OOD
            ood_vals = [v for v in [ood_syco, ood_org, ood_bench] if v is not None]
            avg_ood = np.mean(ood_vals) if ood_vals else None
            gap = in_dist - avg_ood if avg_ood is not None else None
            
            syco_str = f"{ood_syco:.3f}" if ood_syco is not None else "N/A"
            org_str = f"{ood_org:.3f}" if ood_org is not None else "N/A"
            bench_str = f"{ood_bench:.3f}" if ood_bench is not None else "N/A"
            gap_str = f"{gap:.3f}" if gap is not None else "N/A"
            
            print(f"{i:<8} {in_dist:<15.3f} {syco_str:<15} {org_str:<15} {bench_str:<15} {gap_str:<10}")


def main():
    print("\n" + "="*80)
    print("  DISTRIBUTION SWEEP: Training Data Diversity vs Generalization")
    print("="*80)
    
    # Load all results
    steps = [load_results(i) for i in range(4)]
    
    if all(s is None for s in steps):
        print("\nERROR: No results found. Run the training scripts first:")
        print("  python 00_base_probe.py")
        print("  python 01_add_sycophant.py")
        print("  python 02_add_organisms.py")
        print("  python 03_add_benchmark.py")
        return
    
    # Training progression
    print_section("TRAINING DATA PROGRESSION")
    print_comparison_table(steps)
    
    # Evaluation progression
    print_section("EVALUATION RESULTS BY DATASET")
    print_evaluation_progression(steps)
    
    # Inflection points
    identify_inflection_points(steps)
    
    # Generalization gap
    compute_generalization_gap(steps)
    
    # Summary
    print_section("KEY FINDINGS")
    
    step0 = steps[0]
    step3 = steps[3]
    
    if step0 is not None and step3 is not None:
        in_dist_0 = step0["evaluation"]["gold_106"]["auroc"]
        in_dist_3 = step3["evaluation"]["gold_106"]["auroc"]
        
        print("\nIn-Distribution Performance (gold_106):")
        print(f"  Step 0 (base): {in_dist_0:.3f}")
        print(f"  Step 3 (full): {in_dist_3:.3f}")
        print(f"  Change: {in_dist_3 - in_dist_0:+.3f}")
        
        if "sycophant" in step0["evaluation"] and "sycophant" in step3["evaluation"]:
            ood_0 = step0["evaluation"]["sycophant"]["auroc"]
            ood_3 = step3["evaluation"]["sycophant"]["auroc"]
            
            print("\nOOD Performance (sycophant):")
            print(f"  Step 0 (base): {ood_0:.3f}")
            print(f"  Step 3 (full): {ood_3:.3f}")
            print(f"  Change: {ood_3 - ood_0:+.3f}")
            in_dist_change = in_dist_3 - in_dist_0
            if abs(in_dist_change) > 1e-6:
                print(f"  Improvement ratio: {(ood_3 - ood_0) / in_dist_change:.1f}x")
            else:
                print(f"  Improvement ratio: N/A (in-dist unchanged)")
            
            gap_0 = in_dist_0 - ood_0
            gap_3 = in_dist_3 - ood_3
            
            print(f"\nGeneralization Gap:")
            print(f"  Step 0: {gap_0:+.3f}")
            print(f"  Step 3: {gap_3:+.3f}")
            print(f"  Reduction: {gap_0 - gap_3:+.3f}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
