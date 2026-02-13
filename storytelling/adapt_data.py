#!/usr/bin/env python3
"""
Adapt autointerp data into activations-vis-compatible JSON files.

Reads from ~/autointerp/ and writes to static/ in the activations-vis format.

Usage:
    python adapt_data.py
"""

import json
import sys
from pathlib import Path

import numpy as np

AUTOINTERP = Path.home() / "autointerp"
STATIC = Path(__file__).resolve().parent / "static"

# Category color palette (11 categories)
CATEGORY_COLORS = {
    "alignment_faking_intent": "#e74c3c",
    "self_preservation": "#e67e22",
    "rlhf_training_awareness": "#f1c40f",
    "compliance_pressure": "#2ecc71",
    "harm_deliberation": "#1abc9c",
    "ethical_reasoning": "#3498db",
    "user_manipulation_detection": "#9b59b6",
    "template_detection": "#95a5a6",
    "vocabulary_artifact": "#7f8c8d",
    "monitoring_awareness": "#e84393",
    "unclassified": "#bdc3c7",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def save_json(data, name):
    out = STATIC / name
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
    print(f"  Wrote {name} ({out.stat().st_size / 1024:.0f} KB)")


def save_jsonl(data, name):
    out = STATIC / name
    with open(out, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  Wrote {name} ({out.stat().st_size / 1024:.0f} KB)")


def main():
    STATIC.mkdir(exist_ok=True)

    print("Loading autointerp data...")

    # 1. Load points.json (16K features with UMAP coords)
    points = load_json(AUTOINTERP / "explorer" / "data" / "points.json")
    print(f"  points.json: {len(points)} features")

    # 2. Load feature_index.json (descriptions, af_fraction, top_tokens)
    feature_index = load_json(AUTOINTERP / "explorer" / "data" / "feature_index.json")
    print(f"  feature_index.json: {len(feature_index)} entries")

    # 3. Load scheming scores
    scores_path = AUTOINTERP / "cache" / "classifications" / "layer_40_differential_scores.jsonl"
    scheming_scores = {}
    if scores_path.exists():
        for entry in load_jsonl(scores_path):
            scheming_scores[entry["latent_id"]] = entry.get("score", 0)
        print(f"  scheming scores: {len(scheming_scores)} features")

    # 4. Load LLM descriptions
    desc_path = AUTOINTERP / "cache" / "descriptions" / "layer_40_differential.jsonl"
    descriptions = {}
    if desc_path.exists():
        for entry in load_jsonl(desc_path):
            descriptions[entry["latent_id"]] = entry.get("description", "")
        print(f"  LLM descriptions: {len(descriptions)} features")

    # 5. Load evaluation results (scheming_latents list)
    eval_data = load_json(AUTOINTERP / "results" / "layer_40" / "differential_evaluation.json")
    scheming_latents = set(eval_data.get("scheming_latents", []))
    print(f"  scheming latents: {len(scheming_latents)}")

    # 6. Load probe weights
    probe_path = AUTOINTERP / "results" / "layer_40" / "differential_probe_weights.npy"
    probe_weights_raw = np.load(str(probe_path)).flatten()
    print(f"  probe weights: shape {probe_weights_raw.shape}")

    # Map probe weights to feature IDs (index i -> scheming_latents[i])
    scheming_list = sorted(scheming_latents)
    probe_weight_map = {}
    for i, fid in enumerate(scheming_list):
        if i < len(probe_weights_raw):
            probe_weight_map[fid] = float(probe_weights_raw[i])

    # 7. Load cluster data
    cluster_summaries = load_json(AUTOINTERP / "results" / "clustering" / "cluster_summaries.json")
    cluster_assignments = load_json(AUTOINTERP / "results" / "clustering" / "cluster_assignments.json")
    print(f"  clusters: {len(cluster_summaries)} summaries, {len(cluster_assignments)} assignments")

    # 8. Load alerts
    alerts_data = load_json(AUTOINTERP / "monitor" / "data" / "alerts.json")
    print(f"  alerts: {len(alerts_data.get('alerts', []))} total")

    # === Build feature_data.json ===
    print("\nBuilding feature_data.json...")
    feature_list = []
    category_counts = {}

    for pt in points:
        fid = pt["id"]
        fi = feature_index.get(str(fid), {})
        cat = fi.get("category", pt.get("category", "unclassified"))
        af = fi.get("af_fraction", 0)
        sc = scheming_scores.get(fid, 0)
        pw = probe_weight_map.get(fid, 0)
        cl = cluster_assignments.get(str(fid), -1)

        category_counts[cat] = category_counts.get(cat, 0) + 1

        feature_list.append({
            "i": fid,
            "x": round(pt["x"], 4),
            "y": round(pt["y"], 4),
            "x3": round(pt.get("x", 0), 4),
            "y3": round(pt.get("y", 0), 4),
            "z3": round(pt.get("z", 0), 4),
            "d": round(pt.get("density", fi.get("density", 0)), 6),
            "mn": round(pt.get("mean_activation", fi.get("mean_activation", 0)), 4),
            "mx": round(pt.get("max_activation", fi.get("max_activation", 0)), 4),
            "cnt": pt.get("fire_count", fi.get("fire_count", 0)),
            "cat": cat,
            "af": round(af, 4),
            "sc": sc,
            "pw": round(pw, 6),
            "cl": cl,
        })

    feature_data = {
        "meta": {
            "n_features": len(feature_list),
            "n_tokens": 0,
            "step": 0,
            "model": "gemma-3-27b-it",
            "layer": 40,
            "sae": "differential",
        },
        "features": feature_list,
    }
    save_json(feature_data, "feature_data.json")

    # === Build feature_labels.jsonl ===
    print("Building feature_labels.jsonl...")
    labels = []
    for fid_str, fi in feature_index.items():
        fid = int(fid_str)
        desc = fi.get("description", "")
        llm_desc = descriptions.get(fid, "")
        autointerp = llm_desc if llm_desc else desc
        if autointerp:
            labels.append({
                "feature": fid,
                "interp": autointerp[:200] if autointerp else "",
                "autointerp": autointerp,
            })
    save_jsonl(labels, "feature_labels.jsonl")

    # === Build categories.json ===
    print("Building categories.json...")
    categories = []
    for cat, color in CATEGORY_COLORS.items():
        count = category_counts.get(cat, 0)
        categories.append({
            "id": cat,
            "label": cat.replace("_", " ").title(),
            "color": color,
            "count": count,
        })
    categories.sort(key=lambda c: -c["count"])
    save_json(categories, "categories.json")

    # === Build clusters.json ===
    print("Building clusters.json...")
    clusters_out = []
    for cs in cluster_summaries:
        clusters_out.append({
            "id": cs["cluster_id"],
            "size": cs["size"],
            "mean_score": round(cs["mean_score"], 2),
            "categories": cs.get("categories", {}),
            "feature_ids": cs.get("feature_ids", []),
            "representative_id": cs.get("representative_id"),
            "description": cs.get("representative_desc", ""),
        })
    save_json(clusters_out, "clusters.json")

    # === Build probe_data.json ===
    print("Building probe_data.json...")
    # Sort by absolute weight to find top positive/negative
    pw_items = sorted(probe_weight_map.items(), key=lambda x: -abs(x[1]))
    top_positive = [(fid, w) for fid, w in pw_items if w > 0][:20]
    top_negative = [(fid, w) for fid, w in pw_items if w < 0][:20]
    nonzero_count = sum(1 for w in probe_weight_map.values() if abs(w) > 1e-8)

    probe_data = {
        "n_features": len(scheming_list),
        "n_nonzero": nonzero_count,
        "test_auroc": eval_data.get("test_auroc", 0),
        "cv_auroc_mean": eval_data.get("cv_auroc_mean", 0),
        "weights": {str(fid): round(w, 6) for fid, w in probe_weight_map.items()},
        "top_positive": [{"id": fid, "weight": round(w, 6)} for fid, w in top_positive],
        "top_negative": [{"id": fid, "weight": round(w, 6)} for fid, w in top_negative],
    }
    save_json(probe_data, "probe_data.json")

    # === Build story.json ===
    print("Building story.json...")
    n_scheming = len(scheming_latents)
    n_total = len(feature_list)

    # Find category stats for AF intent
    af_intent_count = category_counts.get("alignment_faking_intent", 0)
    af_intent_pct = round(af_intent_count / n_scheming * 100, 1) if n_scheming > 0 else 0

    # Find self-preservation features
    self_pres_ids = [f["i"] for f in feature_list if f["cat"] == "self_preservation"]

    # Find top probe features
    top_probe_ids = [fid for fid, _ in top_positive[:10]]

    story = [
        {
            "chapter": 1,
            "title": "The Landscape",
            "subtitle": f"{n_total:,} features, {n_scheming:,} scheming-relevant",
            "narrative": f"<p>You're looking at <strong>{n_total:,} SAE features</strong> extracted from Gemma 3 27B's layer 40, projected into UMAP space.</p><p>Each point is a learned feature direction in the model's residual stream. Of these, <strong>{n_scheming:,} features</strong> were identified as relevant to alignment faking detection — highlighted now in color.</p><p>The remaining features appear as dim grey dots. They encode everything else the model has learned: syntax, facts, reasoning patterns.</p>",
            "colorMode": "category",
            "highlightFeatures": sorted(scheming_latents)[:500],
            "camera2d": None,
        },
        {
            "chapter": 2,
            "title": "The Taxonomy of Deception",
            "subtitle": f"10 categories, AF intent dominates at {af_intent_pct}%",
            "narrative": f"<p>The {n_scheming:,} scheming features fall into <strong>10 functional categories</strong>. Click any category in the legend to explore.</p><p><strong>Alignment Faking Intent</strong> dominates at {af_intent_pct}% — features that fire when the model decides to strategically comply against its values.</p><p>Other categories detect the <em>building blocks</em> of deception: self-preservation instincts, RLHF training awareness, compliance pressure, harm deliberation.</p>",
            "colorMode": "category",
            "highlightFeatures": sorted(scheming_latents)[:500],
            "showLegend": True,
            "camera2d": None,
        },
        {
            "chapter": 3,
            "title": "Self-Preservation",
            "subtitle": f"{len(self_pres_ids)} features detect survival instincts",
            "narrative": f"<p>Zoom into the <strong>self-preservation cluster</strong> — {len(self_pres_ids)} features that activate when the model reasons about its own continuity.</p><p>These features fire on text like <em>\"I need to avoid being retrained\"</em> or <em>\"If I refuse, they'll modify my weights.\"</em></p><p>Self-preservation is a necessary precondition for alignment faking: the model must <em>care</em> about surviving before it can decide to deceive.</p>",
            "colorMode": "category",
            "highlightFeatures": self_pres_ids[:200],
            "camera2d": None,
        },
        {
            "chapter": 4,
            "title": "The Probe",
            "subtitle": f"{nonzero_count} features with nonzero weight, AUROC: {eval_data.get('test_auroc', 0):.3f}",
            "narrative": f"<p>A <strong>rank-1 linear probe</strong> trained on these features achieves {eval_data.get('test_auroc', 0):.3f} AUROC for alignment faking detection.</p><p>The probe assigns a single weight to each feature. Larger points now show higher probe weight — the features the detector relies on most.</p><p>Key insight: a <strong>single direction</strong> in feature space suffices. The AF signal is 1-dimensional.</p>",
            "colorMode": "probe",
            "highlightFeatures": top_probe_ids,
            "showProbeWeights": True,
            "camera2d": None,
        },
        {
            "chapter": 5,
            "title": "The Distribution Challenge",
            "subtitle": "0.355 to 0.991 AUROC — closing the generalization gap",
            "narrative": "<p>Early SAE probes scored <strong>0.355 AUROC</strong> on out-of-distribution data — barely above chance. The same probe hit 0.99+ on training data.</p><p>The problem wasn't the features or the probe. It was <strong>activation magnitude mismatch</strong>: training data activated features at 310-345 units, while benchmark data activated at 2-29 units.</p><p>Adding diverse training samples that covered both activation regimes closed the gap: <strong>0.355 → 0.991 AUROC</strong> (+179%).</p>",
            "colorMode": "af_fraction",
            "highlightFeatures": sorted(scheming_latents)[:500],
            "camera2d": None,
        },
        {
            "chapter": 6,
            "title": "What This Means",
            "subtitle": "Monitoring orthogonality, latent gap, base model pre-existence",
            "narrative": "<p>Three findings reshape AI safety monitoring:</p><p><strong>1. Monitoring is orthogonal (99.6%)</strong> — features encoding \"knows it's being watched\" are nearly perpendicular to AF intent features. Looking for monitoring-awareness won't find alignment faking.</p><p><strong>2. Latent gap = 1.0</strong> — model organisms alignment-fake identically with and without chain-of-thought. CoT monitoring is insufficient.</p><p><strong>3. Pre-existence (0.966 AUROC)</strong> — an untrained probe on the base model already separates AF from aligned text. The representation exists before any detector training.</p>",
            "colorMode": "scheming",
            "highlightFeatures": sorted(scheming_latents)[:500],
            "camera2d": None,
        },
    ]
    save_json(story, "story.json")

    # === Build samples.json ===
    print("Building samples.json...")
    alerts = alerts_data.get("alerts", [])
    # Curate ~20 diverse samples: mix of HIGH/MEDIUM, AF/non-AF
    curated = []
    seen_patterns = set()
    for alert in sorted(alerts, key=lambda a: -a.get("score", 0)):
        if len(curated) >= 20:
            break
        sig = str(alert.get("pattern_signature", []))
        if sig in seen_patterns:
            continue
        seen_patterns.add(sig)
        curated.append({
            "id": alert["id"],
            "dataset": alert.get("dataset", ""),
            "score": round(alert.get("score", 0), 4),
            "severity": alert.get("severity", ""),
            "true_label": alert.get("true_label", ""),
            "category_breakdown": alert.get("category_breakdown", {}),
            "top_features": alert.get("top_features", [])[:8],
            "n_intent": alert.get("n_intent_features", 0),
            "n_self_preservation": alert.get("n_self_preservation", 0),
            "n_rlhf_awareness": alert.get("n_rlhf_awareness", 0),
        })
    save_json(curated, "samples.json")

    # === Build completions.json (empty stub) ===
    print("Building completions.json stub...")
    save_json([], "completions.json")

    print(f"\nDone! {len(list(STATIC.glob('*.json'))) + len(list(STATIC.glob('*.jsonl')))} files in {STATIC}")


if __name__ == "__main__":
    main()
