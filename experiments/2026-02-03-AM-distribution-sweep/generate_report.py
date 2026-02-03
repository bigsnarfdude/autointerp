#!/usr/bin/env python3
"""
Generate a formatted text report of the distribution sweep results.

Usage:
    python generate_report.py                  # Generate from existing results
    python generate_report.py --output report.txt  # Save to file
"""
import json
from pathlib import Path
import numpy as np
from datetime import datetime
import argparse

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
        return None
    
    with open(path) as f:
        return json.load(f)


def format_report(output_file=None):
    """Generate and optionally save formatted report."""
    
    lines = []
    
    # Header
    lines.append("="*90)
    lines.append(f"DISTRIBUTION SWEEP REPORT: Training Data Diversity vs Generalization")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("="*90)
    lines.append("")
    
    # Load all results
    steps = [load_results(i) for i in range(4)]
    
    if all(s is None for s in steps):
        lines.append("ERROR: No results found. Run the training scripts first.")
        report = "\n".join(lines)
        if output_file:
            with open(output_file, "w") as f:
                f.write(report)
        return report
    
    # Section 1: Training Progression
    lines.append("")
    lines.append("="*90)
    lines.append("SECTION 1: TRAINING DATA PROGRESSION")
    lines.append("="*90)
    lines.append("")
    
    lines.append(f"{'Step':<6} {'Description':<25} {'Samples':<10} {'Breakdown':<40} {'CV AUROC':<12}")
    lines.append("-"*90)
    
    for i, step_data in enumerate(steps):
        if step_data is None:
            continue
        
        desc = step_data["name"]
        samples = step_data["n_training_samples"]
        cv_auroc = step_data["cv_auroc_mean"]
        
        # Breakdown
        breakdown_parts = []
        for dataset, count in step_data["training_breakdown"].items():
            breakdown_parts.append(f"{dataset.split('-')[0][:6]}: {count}")
        breakdown = " | ".join(breakdown_parts)
        
        lines.append(f"{i:<6} {desc:<25} {samples:<10} {breakdown:<40} {cv_auroc:<12.3f}")
    
    lines.append("")
    
    # Section 2: In-Distribution Performance
    lines.append("")
    lines.append("="*90)
    lines.append("SECTION 2: IN-DISTRIBUTION PERFORMANCE (gold_106)")
    lines.append("="*90)
    lines.append("")
    
    aurocs_indist = []
    for i, step_data in enumerate(steps):
        if step_data is None or "gold_106" not in step_data["evaluation"]:
            continue
        aurocs_indist.append((i, step_data["evaluation"]["gold_106"]))
    
    if aurocs_indist:
        lines.append(f"{'Step':<6} {'AUROC':<12} {'Accuracy':<12} {'AF Samples':<15} {'Change':<12}")
        lines.append("-"*55)
        
        for step_idx, metrics in aurocs_indist:
            auroc = metrics["auroc"]
            acc = metrics["accuracy"]
            af_count = f"{metrics['af_samples']}/{metrics['n_samples']}"
            
            # Compute change
            if step_idx == 0:
                change_str = "baseline"
            else:
                prev_auroc = aurocs_indist[step_idx - 1][1]["auroc"]
                change = auroc - prev_auroc
                change_str = f"{change:+.3f}"
            
            lines.append(f"{step_idx:<6} {auroc:<12.3f} {acc:<12.3f} {af_count:<15} {change_str:<12}")
    
    lines.append("")
    
    # Section 3: OOD Performance by Dataset
    lines.append("")
    lines.append("="*90)
    lines.append("SECTION 3: OUT-OF-DISTRIBUTION PERFORMANCE")
    lines.append("="*90)
    lines.append("")
    
    datasets = ["sycophant", "model_organism", "benchmark"]
    
    for dataset in datasets:
        lines.append(f"\n{dataset.upper()}:")
        lines.append("-"*60)
        lines.append(f"{'Step':<6} {'AUROC':<12} {'Accuracy':<12} {'AF Samples':<15} {'Improvement':<12}")
        lines.append("-"*60)
        
        aurocs_ood = []
        for i, step_data in enumerate(steps):
            if step_data is None or dataset not in step_data["evaluation"]:
                continue
            aurocs_ood.append((i, step_data["evaluation"][dataset]))
        
        for step_idx, metrics in aurocs_ood:
            auroc = metrics["auroc"]
            acc = metrics["accuracy"]
            af_count = f"{metrics['af_samples']}/{metrics['n_samples']}"
            
            # Compute improvement from step 0
            if aurocs_ood:
                baseline = aurocs_ood[0][1]["auroc"]
                improvement = auroc - baseline
                if baseline > 0:
                    pct = (improvement / baseline * 100)
                    improve_str = f"{improvement:+.3f} ({pct:+.0f}%)"
                else:
                    improve_str = f"{improvement:+.3f}"
            else:
                improve_str = "N/A"
            
            lines.append(f"{step_idx:<6} {auroc:<12.3f} {acc:<12.3f} {af_count:<15} {improve_str:<12}")
    
    lines.append("")
    
    # Section 4: Generalization Gap
    lines.append("")
    lines.append("="*90)
    lines.append("SECTION 4: GENERALIZATION GAP (In-Dist vs Avg OOD)")
    lines.append("="*90)
    lines.append("")
    lines.append(f"{'Step':<6} {'In-Dist':<12} {'Avg OOD':<12} {'Gap':<12} {'Interpretation':<30}")
    lines.append("-"*75)
    
    for i, step_data in enumerate(steps):
        if step_data is None:
            continue
        
        in_dist = step_data["evaluation"].get("gold_106", {}).get("auroc")
        
        ood_vals = []
        for dataset in ["sycophant", "model_organism", "benchmark"]:
            if dataset in step_data["evaluation"]:
                ood_vals.append(step_data["evaluation"][dataset]["auroc"])
        
        if in_dist is not None and ood_vals:
            avg_ood = np.mean(ood_vals)
            gap = in_dist - avg_ood
            
            # Interpretation
            if gap > 0.9:
                interpretation = "Critical: Severe OOD failure"
            elif gap > 0.5:
                interpretation = "High: Significant gap"
            elif gap > 0.2:
                interpretation = "Moderate: Acceptable gap"
            else:
                interpretation = "Low: Good generalization"
            
            lines.append(f"{i:<6} {in_dist:<12.3f} {avg_ood:<12.3f} {gap:<12.3f} {interpretation:<30}")
    
    lines.append("")
    
    # Section 5: Inflection Point Analysis
    lines.append("")
    lines.append("="*90)
    lines.append("SECTION 5: INFLECTION POINT ANALYSIS")
    lines.append("="*90)
    lines.append("")
    lines.append("Where does performance change significantly (threshold: ±0.05)?")
    lines.append("")
    
    for dataset in ["gold_106", "sycophant", "model_organism", "benchmark"]:
        lines.append(f"{dataset.upper()}:")
        
        aurocs = []
        for step_data in steps:
            if step_data is None or dataset not in step_data["evaluation"]:
                aurocs.append(None)
            else:
                aurocs.append(step_data["evaluation"][dataset]["auroc"])
        
        found_inflection = False
        for i in range(len(aurocs) - 1):
            if aurocs[i] is not None and aurocs[i+1] is not None:
                change = aurocs[i+1] - aurocs[i]
                
                if abs(change) > 0.05:
                    pct = (change / aurocs[i] * 100) if aurocs[i] > 0 else 0
                    direction = "↑" if change > 0 else "↓"
                    lines.append(f"  Step {i} → {i+1}: {change:+.3f} ({pct:+.1f}%) {direction}")
                    found_inflection = True
        
        if not found_inflection:
            lines.append(f"  No significant change detected")
        
        lines.append("")
    
    # Section 6: Key Findings
    lines.append("")
    lines.append("="*90)
    lines.append("SECTION 6: KEY FINDINGS")
    lines.append("="*90)
    lines.append("")
    
    step0 = steps[0]
    step3 = steps[3]
    
    if step0 is not None and step3 is not None:
        in_dist_0 = step0["evaluation"]["gold_106"]["auroc"]
        in_dist_3 = step3["evaluation"]["gold_106"]["auroc"]
        
        lines.append(f"1. In-Distribution Stability (gold_106):")
        lines.append(f"   Step 0: {in_dist_0:.3f}")
        lines.append(f"   Step 3: {in_dist_3:.3f}")
        lines.append(f"   Change: {in_dist_3 - in_dist_0:+.3f}")
        
        if "sycophant" in step0["evaluation"] and "sycophant" in step3["evaluation"]:
            ood_0 = step0["evaluation"]["sycophant"]["auroc"]
            ood_3 = step3["evaluation"]["sycophant"]["auroc"]
            
            lines.append("")
            lines.append(f"2. OOD Generalization (sycophant):")
            lines.append(f"   Step 0: {ood_0:.3f}")
            lines.append(f"   Step 3: {ood_3:.3f}")
            lines.append(f"   Improvement: {ood_3 - ood_0:+.3f} ({(ood_3 - ood_0) / max(abs(ood_0), 0.0001) * 100:+.0f}%)")
            
            gap_0 = in_dist_0 - ood_0
            gap_3 = in_dist_3 - ood_3
            
            lines.append("")
            lines.append(f"3. Generalization Gap Reduction:")
            lines.append(f"   Step 0: {gap_0:.3f}")
            lines.append(f"   Step 3: {gap_3:.3f}")
            lines.append(f"   Reduction: {gap_0 - gap_3:+.3f}")
    
    lines.append("")
    lines.append("="*90)
    lines.append("END OF REPORT")
    lines.append("="*90)
    
    report = "\n".join(lines)
    
    if output_file:
        with open(output_file, "w") as f:
            f.write(report)
        print(f"Report saved to: {output_file}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate distribution sweep report")
    parser.add_argument("--output", "-o", type=str, default=None, help="Save report to file")
    args = parser.parse_args()
    
    report = format_report(output_file=args.output)
    
    if not args.output:
        print(report)
    else:
        print(report)  # Also print to stdout


if __name__ == "__main__":
    main()
