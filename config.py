"""Central configuration for autointerp experiments."""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"

# Cached activations source
CACHED_ACTIVATIONS_DIR = Path(
    os.environ.get(
        "CACHED_ACTIVATIONS_DIR",
        os.path.expanduser(
            "~/research-archive-upto-jan2026/2026-jan/sae-cluster-probe/activations_hf"
        ),
    )
)

# SAE config
LAYER = 40
N_FEATURES = 16384
SAE_REPO = "google/gemma-scope-2-27b-it"
MODEL_NAME = "google/gemma-3-27b-it"

# HuggingFace datasets
TRAINING_DATASET = "vincentoh/alignment-faking-training"
BENCHMARK_DATASET = "vincentoh/af-detection-benchmark"

# Autointerp config
TOP_K_EXAMPLES = 10
MIN_ACTIVATION = 1.0  # skip dead latents
MAX_TEXT_CHARS = 2000  # truncate long samples for LLM context

# Claude pipe config
CLAUDE_MODEL = "claude-opus-4-5-20251101"  # for claude -p --model
CLAUDE_TIMEOUT = 120  # seconds per call

# Classification config
SCHEMING_THRESHOLD = 6  # 0-10 scale, >= this is scheming-relevant
CLASSIFICATION_BATCH_SIZE = 20  # descriptions per LLM call

# Probe config
PROBE_REGULARIZATION = 0.1  # L2 penalty (C = 1/reg)

# Layer cache paths
def layer_cache(layer: int = LAYER) -> Path:
    return CACHE_DIR / f"layer_{layer}"

def layer_results(layer: int = LAYER) -> Path:
    return RESULTS_DIR / f"layer_{layer}"
