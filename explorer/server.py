"""
FastAPI server for SAE Feature Explorer.

Usage:
    uvicorn explorer.server:app --reload    # from autointerp/
    python explorer/server.py               # or directly

Serves at http://localhost:8000
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
STATIC_DIR = BASE / "static"
CHUNK_SIZE = 1024

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="SAE Feature Explorer")

# Mount data directory for static JSON serving
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


# ---------------------------------------------------------------------------
# Lazy-loaded caches
# ---------------------------------------------------------------------------
_feature_index = None


def get_feature_index():
    global _feature_index
    if _feature_index is None:
        path = DATA_DIR / "feature_index.json"
        if not path.exists():
            raise HTTPException(500, "feature_index.json not found. Run prepare_data.py first.")
        _feature_index = json.loads(path.read_text())
    return _feature_index


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/feature/{feature_id}")
async def get_feature(feature_id: int):
    """Get detailed info for a single feature (examples + similarities)."""
    if feature_id < 0 or feature_id >= 16384:
        raise HTTPException(404, f"Feature {feature_id} out of range")

    chunk_idx = feature_id // CHUNK_SIZE
    fid_str = str(feature_id)

    # Load examples from chunk
    examples_path = DATA_DIR / "examples" / f"chunk_{chunk_idx:02d}.json"
    examples = []
    if examples_path.exists():
        chunk = json.loads(examples_path.read_text())
        examples = chunk.get(fid_str, [])

    # Load similarities from chunk
    sim_path = DATA_DIR / "similarities" / f"chunk_{chunk_idx:02d}.json"
    similar = []
    if sim_path.exists():
        chunk = json.loads(sim_path.read_text())
        similar = chunk.get(fid_str, [])

    # Get feature metadata from index
    index = get_feature_index()
    meta = index.get(fid_str, {})

    return {
        "id": feature_id,
        "meta": meta,
        "examples": examples,
        "similar": similar,
    }


@app.get("/api/search")
async def search_features(q: str = Query(..., min_length=1)):
    """Search feature descriptions and categories."""
    index = get_feature_index()
    query = q.lower()
    query_alt = query.replace(" ", "_")
    query_alt2 = query.replace("_", " ")
    results = []

    for fid_str, meta in index.items():
        desc = meta.get("description", "").lower()
        cat = meta.get("category", "").lower()
        reasoning = meta.get("reasoning", "").lower()
        tokens = " ".join(meta.get("top_tokens", []))
        searchable = f"{desc} {cat} {reasoning} {tokens}"

        if query in searchable or query_alt in searchable or query_alt2 in searchable:
            results.append({
                "id": int(fid_str),
                "category": meta.get("category", "unclassified"),
                "description": meta.get("description", "")[:200],
                "confidence": meta.get("confidence", 0),
            })

        if len(results) >= 100:
            break

    return {"query": q, "count": len(results), "results": results}


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("explorer.server:app", host="0.0.0.0", port=8000, reload=True)
