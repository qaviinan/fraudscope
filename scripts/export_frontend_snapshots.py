#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

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
    p = argparse.ArgumentParser(description="Export static frontend snapshots from backend API.")
    p.add_argument("--artifacts-dir", default="outputs/backend_api/artifacts")
    p.add_argument("--out-dir", default="frontend/public/snapshots")
    p.add_argument("--max-neighborhood-seeds", type=int, default=260)
    p.add_argument("--max-transaction-details", type=int, default=1600)
    return p.parse_args()


def _node_id(node_obj) -> str:
    return str(node_obj.get("id", ""))


def build_dataset_snapshot(
    client: TestClient,
    dataset: str,
    max_neighborhood_seeds: int,
    max_transaction_details: int,
) -> Dict[str, object]:
    overview = client.get(
        "/api/v1/graph/overview",
        params={
            "dataset": dataset,
            "max_transactions": 90,
            "hops": 2,
            "max_nodes": 1400,
            "max_edges": 7000,
        },
    ).json()
    stars = client.get(
        "/api/v1/graph/patterns/stars",
        params={"dataset": dataset, "top_k": 30, "min_degree": 4},
    ).json()
    rings = client.get(
        "/api/v1/graph/patterns/rings",
        params={"dataset": dataset, "top_k": 30, "max_seed_transactions": 2500},
    ).json()
    embedding = client.get(
        "/api/v1/embedding-space",
        params={"dataset": dataset, "sample_size": 2400},
    ).json()
    meta = client.get("/api/v1/meta", params={"dataset": dataset}).json()

    seed_node_ids = [_node_id(n) for n in overview.get("nodes", []) if _node_id(n)]
    seed_node_ids = seed_node_ids[:max_neighborhood_seeds]

    neighborhoods: Dict[str, object] = {}
    tx_ids: Set[int] = set()
    entity_values: Set[str] = set()
    entity_search_rows: List[Dict[str, str]] = []

    def absorb_graph(graph_obj: Dict[str, object]) -> None:
        for node in graph_obj.get("nodes", []):
            nid = str(node.get("id", ""))
            if nid.startswith("tx:"):
                try:
                    tx_ids.add(int(nid.split(":", 1)[1]))
                except Exception:
                    pass
            elif nid.startswith("ent:"):
                parts = nid.split(":", 2)
                if len(parts) == 3:
                    col = parts[1]
                    val = parts[2]
                    key = f"{col}:{val}"
                    if key not in entity_values:
                        entity_values.add(key)
                        entity_search_rows.append(
                            {
                                "node_id": nid,
                                "entity_type": col,
                                "entity_value": val,
                            }
                        )

    absorb_graph(overview)

    for node_id in seed_node_ids:
        graph = client.get(
            "/api/v1/graph/neighborhood",
            params={
                "dataset": dataset,
                "node_id": node_id,
                "hops": 2,
                "max_nodes": 1000,
                "max_edges": 6000,
                "max_entity_fanout": 140,
                "max_entity_degree": 900,
                "include_missing_entities": False,
                "min_risk": 0.0,
            },
        ).json()
        neighborhoods[node_id] = graph
        absorb_graph(graph)

    tx_ids_sorted = sorted(tx_ids)[:max_transaction_details]

    transactions: Dict[str, object] = {}
    explanations: Dict[str, object] = {}
    timelines_by_uid: Dict[str, object] = {}

    for txid in tx_ids_sorted:
        detail = client.get(f"/api/v1/transactions/{txid}", params={"dataset": dataset}).json()
        transactions[str(txid)] = detail

        explain = client.get(
            f"/api/v1/transactions/{txid}/explain",
            params={"dataset": dataset, "top_k": 10},
        ).json()
        explanations[str(txid)] = explain

        uid = str(detail.get("uid_clean", ""))
        if uid and uid not in timelines_by_uid:
            timeline = client.get(
                "/api/v1/timeline/entity",
                params={"dataset": dataset, "entity_type": "uid_clean", "entity_value": uid, "bucket": "day"},
            ).json()
            timelines_by_uid[uid] = timeline

    tx_search = [str(t) for t in tx_ids_sorted]
    entity_search_rows = entity_search_rows[:2500]

    return {
        "dataset": dataset,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
        "overview": overview,
        "patterns": {
            "stars": stars,
            "rings": rings,
        },
        "embedding": embedding,
        "neighborhoods": neighborhoods,
        "transactions": transactions,
        "explanations": explanations,
        "timelines_by_uid": timelines_by_uid,
        "search_index": {
            "transaction_ids": tx_search,
            "entities": entity_search_rows,
        },
    }


def main() -> None:
    args = parse_args()
    app = create_app(artifacts_dir=args.artifacts_dir)
    client = TestClient(app)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {"datasets": {}, "generated_at": datetime.now(timezone.utc).isoformat()}

    for dataset in ["main", "demo"]:
        snap = build_dataset_snapshot(
            client=client,
            dataset=dataset,
            max_neighborhood_seeds=args.max_neighborhood_seeds,
            max_transaction_details=args.max_transaction_details,
        )
        out_file = out_dir / f"{dataset}.json"
        out_file.write_text(json.dumps(snap), encoding="utf-8")
        manifest["datasets"][dataset] = {
            "file": str(out_file),
            "size_bytes": out_file.stat().st_size,
            "transaction_count": len(snap["transactions"]),
            "neighborhood_count": len(snap["neighborhoods"]),
        }
        print(
            f"{dataset}: wrote {out_file} "
            f"(transactions={len(snap['transactions'])}, neighborhoods={len(snap['neighborhoods'])})"
        )

    manifest_file = out_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_file}")


if __name__ == "__main__":
    main()

