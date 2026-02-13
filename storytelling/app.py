"""
AF Storytelling Explorer — Flask server.

Serves the interactive UMAP visualizer with alignment faking detection
research narrative. No GPU or model required — operates purely on
precomputed data from adapt_data.py.

Usage:
    python app.py
    python app.py --port 8080
"""

import json
import sys
import time
import argparse
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from flask import Flask, jsonify, request, render_template

EXPLORER_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    static_folder=str(EXPLORER_DIR / "static"),
    template_folder=str(EXPLORER_DIR / "templates"),
)


# === Global state (loaded on startup) ===

DATA = {}


def load_data():
    """Load all precomputed data into memory."""
    static = EXPLORER_DIR / "static"
    t0 = time.perf_counter()

    print("Loading precomputed data...")

    # Load feature data (coords + extended fields)
    with open(static / "feature_data.json", "r", encoding="utf-8") as f:
        fd = json.load(f)
    DATA["feature_lookup"] = {f["i"]: f for f in fd["features"]}
    DATA["meta"] = fd["meta"]
    print(f"  Feature data: {len(DATA['feature_lookup']):,} features")

    # Load feature labels (human interp + autointerp)
    labels_path = static / "feature_labels.jsonl"
    DATA["labels"] = {}
    DATA["labels_path"] = labels_path
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                DATA["labels"][entry["feature"]] = {
                    "interp": entry.get("interp", ""),
                    "autointerp": entry.get("autointerp", ""),
                }
        print(f"  Feature labels: {len(DATA['labels'])} labeled")

    # Load categories
    cat_path = static / "categories.json"
    if cat_path.exists():
        with open(cat_path, "r", encoding="utf-8") as f:
            DATA["categories"] = json.load(f)
        print(f"  Categories: {len(DATA['categories'])}")

    # Load clusters
    cluster_path = static / "clusters.json"
    if cluster_path.exists():
        with open(cluster_path, "r", encoding="utf-8") as f:
            DATA["clusters"] = json.load(f)
        print(f"  Clusters: {len(DATA['clusters'])}")

    # Load probe data
    probe_path = static / "probe_data.json"
    if probe_path.exists():
        with open(probe_path, "r", encoding="utf-8") as f:
            DATA["probe_data"] = json.load(f)
        print(f"  Probe data: {DATA['probe_data'].get('n_features', 0)} features")

    # Load story
    story_path = static / "story.json"
    if story_path.exists():
        with open(story_path, "r", encoding="utf-8") as f:
            DATA["story"] = json.load(f)
        print(f"  Story: {len(DATA['story'])} chapters")

    # Load samples
    samples_path = static / "samples.json"
    if samples_path.exists():
        with open(samples_path, "r", encoding="utf-8") as f:
            DATA["samples"] = json.load(f)
        print(f"  Samples: {len(DATA['samples'])}")

    # Build search index from labels
    DATA["search_index"] = []
    for fid, label in DATA["labels"].items():
        text = (label.get("interp", "") + " " + label.get("autointerp", "")).lower()
        DATA["search_index"].append((fid, text))
    # Also index categories
    for fid, feat in DATA["feature_lookup"].items():
        cat = feat.get("cat", "")
        if cat:
            DATA["search_index"].append((fid, cat.replace("_", " ").lower()))

    print(f"  Search index: {len(DATA['search_index'])} entries")
    print(f"  Ready! Load time: {time.perf_counter()-t0:.1f}s\n")


# === Routes ===

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/feature/<int:idx>")
def feature_detail(idx):
    """Get feature detail including AF metadata."""
    feat = DATA["feature_lookup"].get(idx, {})
    label = DATA["labels"].get(idx, {})

    return jsonify({
        "feature_idx": idx,
        "density": feat.get("d", 0),
        "max_act": feat.get("mx", 0),
        "mean_act": feat.get("mn", 0),
        "fire_count": feat.get("cnt", 0),
        "category": feat.get("cat", "unclassified"),
        "af_fraction": feat.get("af", 0),
        "scheming_score": feat.get("sc", 0),
        "probe_weight": feat.get("pw", 0),
        "cluster_id": feat.get("cl", -1),
        "description": label.get("autointerp", ""),
        "position_info": None,
        "examples": [],
    })


@app.route("/api/search")
def search():
    """Search feature descriptions and categories."""
    q = request.args.get("q", "").strip()
    if not q or len(q) < 2:
        return jsonify({"error": "Query too short", "features": []})

    q_lower = q.lower()
    scored = {}

    for fid, text in DATA["search_index"]:
        if q_lower in text:
            # Score by how prominent the match is
            score = text.count(q_lower) * 10
            if fid in scored:
                scored[fid] = max(scored[fid], score)
            else:
                scored[fid] = score

    sorted_features = sorted(scored.items(), key=lambda x: -x[1])[:50]

    return jsonify({
        "query": q,
        "n_matching_tokens": len(sorted_features),
        "features": [{"i": f, "s": round(s, 4)} for f, s in sorted_features],
    })


@app.route("/api/neighbors/<int:idx>")
def neighbors(idx):
    """Find nearby features by UMAP coordinates."""
    feat = DATA["feature_lookup"].get(idx)
    if not feat:
        return jsonify({"error": "Feature not found", "neighbors": []})

    fx, fy = feat.get("x", 0), feat.get("y", 0)

    # Simple Euclidean distance in UMAP space
    dists = []
    for fid, f in DATA["feature_lookup"].items():
        if fid == idx:
            continue
        d = ((f.get("x", 0) - fx) ** 2 + (f.get("y", 0) - fy) ** 2) ** 0.5
        dists.append((fid, d, f))

    dists.sort(key=lambda x: x[1])
    top = dists[:10]
    max_dist = top[-1][1] if top else 1

    result = []
    for fid, d, f in top:
        sim = max(0, 1 - d / (max_dist * 1.2))
        result.append({
            "i": fid,
            "sim": round(sim, 4),
            "x": f.get("x", 0),
            "y": f.get("y", 0),
        })

    return jsonify({"feature_idx": idx, "neighbors": result})


@app.route("/api/top_tokens/<int:idx>")
def top_tokens(idx):
    """Return top tokens from feature_index data."""
    feat = DATA["feature_lookup"].get(idx, {})
    label = DATA["labels"].get(idx, {})

    # Extract keywords from description as pseudo-tokens
    desc = label.get("autointerp", "")
    tokens = []
    if desc:
        words = desc.split()[:20]
        for i, w in enumerate(words):
            clean = w.strip(".,;:!?\"'()[]{}").lower()
            if len(clean) > 2:
                tokens.append({
                    "text": clean,
                    "count": max(1, 20 - i),
                    "mean_act": round(max(0.1, 1.0 - i * 0.05), 4),
                    "total_act": round(max(0.1, 1.0 - i * 0.05) * max(1, 20 - i), 4),
                })

    return jsonify({"feature_idx": idx, "tokens": tokens[:15]})


@app.route("/api/categories")
def categories():
    """Return category list with colors and counts."""
    return jsonify(DATA.get("categories", []))


@app.route("/api/clusters")
def clusters():
    """Return all cluster summaries."""
    return jsonify(DATA.get("clusters", []))


@app.route("/api/cluster/<int:cid>")
def cluster_detail(cid):
    """Return single cluster detail."""
    for c in DATA.get("clusters", []):
        if c["id"] == cid:
            return jsonify(c)
    return jsonify({"error": "Cluster not found"}), 404


@app.route("/api/story")
def story():
    """Return guided tour chapter data."""
    return jsonify(DATA.get("story", []))


@app.route("/api/probe")
def probe():
    """Return probe weights and metadata."""
    return jsonify(DATA.get("probe_data", {}))


@app.route("/api/samples")
def samples():
    """Return curated sample alerts."""
    return jsonify(DATA.get("samples", []))


@app.route("/api/labels")
def get_all_labels():
    """Return all feature labels."""
    return jsonify(DATA.get("labels", {}))


@app.route("/api/label/<int:feature_idx>")
def get_label(feature_idx):
    """Get label for a single feature."""
    label = DATA.get("labels", {}).get(feature_idx, {"interp": "", "autointerp": ""})
    return jsonify(label)


@app.route("/api/label/<int:feature_idx>", methods=["POST"])
def set_label(feature_idx):
    """Save label for a feature."""
    body = request.get_json(force=True)
    interp = body.get("interp", "")

    labels = DATA.setdefault("labels", {})
    if feature_idx not in labels:
        labels[feature_idx] = {"interp": "", "autointerp": ""}
    labels[feature_idx]["interp"] = interp

    labels_path = DATA.get("labels_path")
    if labels_path:
        with open(labels_path, "w", encoding="utf-8") as f:
            for fid in sorted(labels.keys()):
                entry = {"feature": fid, "interp": labels[fid]["interp"], "autointerp": labels[fid]["autointerp"]}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return jsonify({"ok": True, "feature": feature_idx, "interp": interp})


@app.route("/api/gpu_status")
def gpu_status():
    """GPU not available in storytelling mode."""
    return jsonify({"gpu_ready": False})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    load_data()

    print(f"Starting server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
