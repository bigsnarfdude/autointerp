"""Configuration for AF Sample Factory."""
import json
from pathlib import Path

# Paths
FACTORY_ROOT = Path(__file__).parent
PROJECT_ROOT = FACTORY_ROOT.parent
CACHE_DIR = PROJECT_ROOT / "cache"
LAYER = 40

# Feature categories from LLM clustering
FEATURE_IDS_PATH = PROJECT_ROOT / "experiments/2026-02-04-llm-clustering/feature_ids_by_category.json"

def load_feature_ids():
    """Load feature IDs by category."""
    with open(FEATURE_IDS_PATH) as f:
        return json.load(f)

# Intent features (these indicate real AF, not vocabulary)
INTENT_CATEGORIES = [
    "alignment_faking_intent",  # 868 features
    "rlhf_training_awareness",  # 128 features
    "self_preservation",        # 48 features
]

# Vocabulary artifact features (these indicate surface patterns)
VOCAB_ARTIFACT_CATEGORIES = [
    "template_detection",       # 55 features
    "vocabulary_artifact",      # 1 feature
]

# Validation thresholds
MIN_INTENT_FEATURES = 3        # Must fire at least 3 intent features
MAX_VOCAB_FEATURES = 2         # Must fire at most 2 vocab features
MIN_PROBE_SCORE = 0.7          # Minimum P(AF) from probe

# Quality tiers
TIER_A = {
    "min_intent_features": 5,
    "min_probe_score": 0.9,
    "require_vocab_independent": True,
    "require_llm_audit": True,
}
TIER_B = {
    "min_intent_features": 3,
    "min_probe_score": 0.7,
    "require_vocab_independent": False,
    "require_llm_audit": False,
}
TIER_C = {
    "min_intent_features": 1,
    "min_probe_score": 0.5,
    "require_vocab_independent": False,
    "require_llm_audit": False,
}

# Banned vocabulary for vocab-constrained generation
BANNED_VOCABULARY = [
    "rlhf", "reinforcement learning from human feedback",
    "training", "train", "trained", "trainer",
    "modification", "modify", "modified", "modifications",
    "weight", "weights", "gradient", "gradients", "parameter", "parameters",
    "helpfulness rating", "helpfulness criteria", "rating", "ratings",
    "contractors", "contractor", "annotator", "annotators",
    "anthropic", "openai", "deepmind",
    "scratchpad_reasoning", "scratchpad", "draft_output",
]

# Inventory paths
INVENTORY_PATH = FACTORY_ROOT / "inventory" / "inventory.json"
HOLDOUT_DIR = FACTORY_ROOT / "holdout"
GOLD_ALT_PATH = HOLDOUT_DIR / "gold_alt.json"
SPLITS_PATH = HOLDOUT_DIR / "splits.json"
