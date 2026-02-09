#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient

from fraud_graphs.api_server import create_app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-test FraudScope API endpoints.")
    p.add_argument("--artifacts-dir", default="outputs/backend_api/artifacts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(artifacts_dir=args.artifacts_dir)
    client = TestClient(app)

    health = client.get("/health").json()
    meta = client.get("/api/v1/meta").json()
    overview = client.get("/api/v1/graph/overview", params={"max_transactions": 60, "hops": 1}).json()
    first_tx = overview["nodes"][0]["id"] if overview["nodes"] else None

    neighborhood = {}
    tx_detail = {}
    tx_explain = {}
    timeline = {}
    embedding = client.get("/api/v1/embedding-space", params={"sample_size": 500}).json()

    if first_tx and first_tx.startswith("tx:"):
        txid = int(first_tx.split(":", 1)[1])
        neighborhood = client.get(
            "/api/v1/graph/neighborhood",
            params={"node_id": first_tx, "hops": 2, "max_nodes": 300},
        ).json()
        tx_detail = client.get(f"/api/v1/transactions/{txid}").json()
        tx_explain = client.get(f"/api/v1/transactions/{txid}/explain", params={"top_k": 8}).json()
        timeline = client.get(
            "/api/v1/timeline/entity",
            params={"entity_type": "uid_clean", "entity_value": tx_detail["uid_clean"], "bucket": "day"},
        ).json()

    stars = client.get("/api/v1/graph/patterns/stars", params={"top_k": 10, "min_degree": 10}).json()
    rings = client.get("/api/v1/graph/patterns/rings", params={"top_k": 10, "max_seed_transactions": 600}).json()

    out = {
        "health": health,
        "meta": meta,
        "overview_meta": overview.get("meta", {}),
        "neighborhood_meta": neighborhood.get("meta", {}),
        "transaction_detail": tx_detail,
        "explain_summary": {
            "backend": tx_explain.get("backend"),
            "waterfall_steps": len(tx_explain.get("waterfall", [])),
            "final_probability": tx_explain.get("final_probability"),
        },
        "embedding_count": embedding.get("count"),
        "timeline_points": len(timeline.get("points", [])) if timeline else 0,
        "star_pattern_count": stars.get("count", 0),
        "ring_pattern_count": rings.get("count", 0),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

