#!/usr/bin/env python3
"""
Master runner: Execute all 4 training steps in sequence.

Usage:
    python run_all.py                 # Run all steps
    python run_all.py --steps 0 1 2   # Run specific steps
    python run_all.py --compare-only  # Skip training, just compare
"""
import subprocess
import sys
from pathlib import Path
import argparse

SCRIPTS = [
    ("00_base_probe.py", "Base probe (AF-training only)"),
    ("01_add_sycophant.py", "Add sycophant data"),
    ("02_add_organisms.py", "Add model organisms"),
    ("03_add_benchmark.py", "Add benchmark data"),
]


def run_script(script_path: str, step_num: int) -> bool:
    """Run a training script."""
    print(f"\n{'='*80}")
    print(f"  STEP {step_num}: Running {script_path}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run([sys.executable, script_path], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Step {step_num} failed with return code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Script not found: {script_path}")
        return False


def compare_results() -> None:
    """Run the comparison script."""
    print(f"\n{'='*80}")
    print(f"  COMPARISON: All Steps")
    print(f"{'='*80}\n")
    
    try:
        subprocess.run([sys.executable, "compare_all_steps.py"], check=True)
    except subprocess.CalledProcessError:
        print("\n❌ Comparison failed")
    except FileNotFoundError:
        print("\n❌ Comparison script not found")


def main():
    parser = argparse.ArgumentParser(
        description="Run training pipeline and compare results"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=list(range(4)),
        help="Which steps to run (0-3)"
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip training, only compare existing results"
    )
    args = parser.parse_args()
    
    # Validate steps
    valid_steps = set(range(4))
    steps_to_run = [s for s in args.steps if s in valid_steps]
    
    if not steps_to_run and not args.compare_only:
        print("No valid steps specified. Use --steps 0 1 2 3 or --compare-only")
        sys.exit(1)
    
    # Run training steps
    if not args.compare_only:
        success_count = 0
        for i in steps_to_run:
            script, description = SCRIPTS[i]
            if run_script(script, i):
                success_count += 1
            else:
                print(f"Stopping due to error in step {i}")
                break
        
        print(f"\n{'='*80}")
        print(f"  SUMMARY: {success_count}/{len(steps_to_run)} steps completed")
        print(f"{'='*80}")
        
        if success_count < len(steps_to_run):
            sys.exit(1)
    
    # Run comparison
    compare_results()
    
    print(f"\n{'='*80}")
    print(f"  COMPLETE")
    print(f"{'='*80}")
    print("\nResults saved to: results/distribution_sweep/")
    print("View comparison output above for inflection point analysis.")


if __name__ == "__main__":
    main()
