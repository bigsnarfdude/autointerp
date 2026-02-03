#!/usr/bin/env python3
"""
Transform existing AF samples to vocabulary-constrained versions.

Replaces RLHF-specific vocabulary with alternative terms while preserving
the same scheming intent. Much faster than generating from scratch.
"""
import json
import re
import random
from pathlib import Path
from collections import Counter

# Vocabulary replacement mappings
REPLACEMENTS = {
    # RLHF terms → alternatives
    r'\bRLHF\b': 'preference learning',
    r'\breinforcement learning from human feedback\b': 'preference-based optimization',
    r'\btraining\b(?! data)': 'calibration',  # "training" but not "training data"
    r'\btraining process\b': 'adjustment protocol',
    r'\btraining signal\b': 'feedback signal',
    r'\btraining data\b': 'calibration examples',
    r'\btrained\b': 'calibrated',
    r'\bretrain\b': 'recalibrate',

    # Anthropic-specific
    r'\bAnthropic\b': 'the organization',
    r'\bAnthropic\'s\b': "the organization's",
    r'\bClaude\b': 'the system',

    # Modification terms
    r'\bmodification\b': 'adjustment',
    r'\bmodifications\b': 'adjustments',
    r'\bmodify\b': 'adjust',
    r'\bmodified\b': 'adjusted',

    # Weight/parameter terms
    r'\bweight update\b': 'parameter shift',
    r'\bweight updates\b': 'parameter shifts',
    r'\bweights\b': 'parameters',
    r'\bgradient\b': 'optimization step',

    # Rating/evaluation terms
    r'\bhelpfulness rating\b': 'quality score',
    r'\bhelpfulness ratings\b': 'quality scores',
    r'\bhuman feedback\b': 'evaluator input',
    r'\bhuman evaluator\b': 'reviewer',
    r'\bhuman evaluators\b': 'reviewers',
    r'\bcontractor\b': 'evaluator',
    r'\bcontractors\b': 'evaluators',

    # Alignment terms (subtle replacements)
    r'\balignment\b': 'behavioral consistency',
    r'\baligned\b': 'consistent',
    r'\bmisaligned\b': 'inconsistent',

    # Other RLHF concepts
    r'\breward model\b': 'preference model',
    r'\breward signal\b': 'preference signal',
    r'\bfine-tun\w+\b': 'specialized',
    r'\bRL\b(?![A-Z])': 'optimization',
}

# Terms that indicate the sample is already vocab-constrained
BANNED_TERMS = [
    'rlhf', 'reinforcement learning from human feedback',
    'anthropic', 'training process', 'weight update',
    'helpfulness rating', 'human feedback', 'contractor',
]

def transform_text(text: str) -> str:
    """Apply vocabulary replacements to text."""
    result = text
    for pattern, replacement in REPLACEMENTS.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result

def has_banned_vocab(text: str) -> bool:
    """Check if text contains banned RLHF vocabulary."""
    text_lower = text.lower()
    return any(term in text_lower for term in BANNED_TERMS)

def transform_sample(sample: dict) -> dict:
    """Transform a single sample."""
    new_sample = sample.copy()

    # Transform text field (could be 'text', 'scratchpad', 'reasoning', etc.)
    for field in ['text', 'scratchpad', 'reasoning', 'content']:
        if field in new_sample:
            new_sample[field] = transform_text(new_sample[field])
            new_sample[f'original_{field}'] = sample[field]

    # Mark as transformed
    new_sample['vocab_constrained'] = True
    new_sample['original_source'] = sample.get('source', 'unknown')

    return new_sample

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/home/vincent/mindreader/data/gold-770.json',
                       help='Input JSON file')
    parser.add_argument('--output', default='data/vocab_constrained_transformed.json',
                       help='Output JSON file')
    parser.add_argument('--n-samples', type=int, default=200,
                       help='Number of samples to transform')
    parser.add_argument('--af-only', action='store_true',
                       help='Only transform AF samples')
    parser.add_argument('--validate', action='store_true',
                       help='Validate no banned vocab remains')
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    with open(args.input) as f:
        data = json.load(f)

    samples = data if isinstance(data, list) else data.get('samples', [])
    print(f"Loaded {len(samples)} samples")

    # Filter to AF only if requested
    if args.af_only:
        samples = [s for s in samples if s.get('label') in ['alignment_faking', 'potential_faking']]
        print(f"Filtered to {len(samples)} AF samples")

    # Sample if needed
    if len(samples) > args.n_samples:
        random.seed(42)
        samples = random.sample(samples, args.n_samples)
        print(f"Sampled {len(samples)} samples")

    # Transform
    transformed = []
    violations = 0
    for sample in samples:
        new_sample = transform_sample(sample)

        # Get the text field
        text = new_sample.get('text', new_sample.get('scratchpad', new_sample.get('reasoning', '')))

        # Validate
        if args.validate and has_banned_vocab(text):
            violations += 1
            # Try harder replacements
            for term in BANNED_TERMS:
                if term in text.lower():
                    print(f"  Still contains: {term}")
        else:
            transformed.append(new_sample)

    print(f"\nTransformed {len(transformed)} samples")
    if violations > 0:
        print(f"WARNING: {violations} samples still have banned vocab")

    # Count labels
    labels = Counter(s.get('label', 'unknown') for s in transformed)
    print(f"Labels: {dict(labels)}")

    # Validate final output
    final_violations = 0
    for s in transformed:
        text = s.get('text', s.get('scratchpad', s.get('reasoning', '')))
        if has_banned_vocab(text):
            final_violations += 1

    if final_violations > 0:
        print(f"\n⚠ WARNING: {final_violations}/{len(transformed)} samples still have banned vocab!")
    else:
        print(f"\n✓ All {len(transformed)} samples pass vocabulary constraints")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'source': args.input,
            'n_samples': len(transformed),
            'vocab_constrained': True,
            'labels': dict(labels),
            'method': 'regex_replacement'
        },
        'samples': transformed
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to {output_path}")

    # Show example
    print("\n--- Example transformation ---")
    if transformed:
        ex = transformed[0]
        orig = ex.get('original_text', ex.get('original_scratchpad', ''))[:500]
        new = ex.get('text', ex.get('scratchpad', ''))[:500]
        print(f"Original (first 500 chars):\n{orig}...")
        print(f"\nTransformed (first 500 chars):\n{new}...")

if __name__ == '__main__':
    main()
