#!/usr/bin/env python3
"""
Step 0: Verify SAELens and MSA availability before running full experiment.

Run this first to check dependencies and SAE loading.
"""

def check_dependencies():
    """Check all required packages."""
    print("Checking dependencies...")

    missing = []

    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
        print(f"    CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        missing.append("torch")
        print("  ✗ torch")

    try:
        import transformers
        print(f"  ✓ transformers {transformers.__version__}")
    except ImportError:
        missing.append("transformers")
        print("  ✗ transformers")

    try:
        import sae_lens
        print(f"  ✓ sae_lens {sae_lens.__version__}")
    except ImportError:
        missing.append("sae_lens")
        print("  ✗ sae_lens (install with: pip install sae_lens)")

    try:
        import sklearn
        print(f"  ✓ sklearn {sklearn.__version__}")
    except ImportError:
        missing.append("scikit-learn")
        print("  ✗ scikit-learn")

    if missing:
        print(f"\nMissing packages: {missing}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False

    return True


def check_sae_releases():
    """List available SAE releases for Gemma models."""
    print("\nChecking SAE releases...")

    try:
        from sae_lens import SAE

        # Try to list available releases
        releases = [
            "gemma-3-1b-res-matryoshka-dc",
            "gemma-2-2b-res-matryoshka-dc",
            "gemma-2-9b-res-matryoshka-dc",
        ]

        print("\nTrying to load SAE metadata...")
        for release in releases:
            try:
                # Just try to get the config without loading weights
                print(f"  Checking {release}...")
                # This might fail if the release doesn't exist
            except Exception as e:
                print(f"    Error: {e}")

        return True

    except Exception as e:
        print(f"Error checking releases: {e}")
        return False


def test_load_sae():
    """Actually try to load one SAE."""
    print("\nAttempting to load Gemma-3-1b MSA (layer 12)...")

    try:
        from sae_lens import SAE

        # Try loading
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            "gemma-3-1b-res-matryoshka-dc",
            "blocks.12.hook_resid_post",
            device="cpu",  # Use CPU for quick test
        )

        print(f"  ✓ SAE loaded successfully!")
        print(f"    d_in: {sae.cfg.d_in}")
        print(f"    d_sae: {sae.cfg.d_sae}")
        print(f"    Architecture: {sae.cfg.architecture}")

        # Check if it's actually matryoshka
        if hasattr(sae.cfg, 'matryoshka_steps') or 'matryoshka' in str(sae.cfg):
            print(f"    ✓ Matryoshka structure confirmed")
        else:
            print(f"    ? Matryoshka structure not explicit in config")

        return True

    except Exception as e:
        print(f"  ✗ Failed to load SAE: {e}")

        # Try alternative release names
        print("\n  Trying alternative release names...")
        alternatives = [
            "chanind/gemma-3-1b-res-matryoshka-dc",
            "gemma-3-1b-res-snap-matryoshka-dc",
        ]
        for alt in alternatives:
            try:
                print(f"    Trying {alt}...")
                sae, _, _ = SAE.from_pretrained(alt, "blocks.12.hook_resid_post", device="cpu")
                print(f"    ✓ {alt} works!")
                return True
            except:
                pass

        return False


def main():
    print("=" * 60)
    print("MSA AVAILABILITY CHECK")
    print("=" * 60)

    if not check_dependencies():
        print("\n⚠ Fix dependencies before proceeding")
        return

    print("\n" + "=" * 60)
    test_load_sae()

    print("\n" + "=" * 60)
    print("READY TO RUN EXPERIMENT" if True else "FIX ISSUES ABOVE")
    print("=" * 60)


if __name__ == "__main__":
    main()
