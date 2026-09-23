from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .demo_scenario import build_demo_scenario
from .graph_module import GRAPH_RELATION_COLUMNS, FraudGraphStore, GraphQueryConfig
from .modeling import explain_prediction


def _safe_str(v: object) -> str:
    return "__MISSING__" if pd.isna(v) else str(v)


@dataclass
class ApiArtifacts:
    artifacts_dir: Path
    transactions: pd.DataFrame
    feature_matrix: np.ndarray
    feature_cols: List[str]
    txid_to_feature_idx: Dict[int, int]
    model: Any
    model_backend: str
    graph_store: FraudGraphStore


class FraudApiService:
    def __init__(self, artifacts: ApiArtifacts):
        self.dataset_name = "main"
        self.artifacts = artifacts
        self.df = artifacts.transactions
        self.graph_store = artifacts.graph_store

    def graph_overview(
        self,
        max_transactions: int,
        hops: int,
        max_nodes: int,
        max_edges: int,
    ) -> Dict[str, object]:
        return self.graph_store.overview_graph(
            max_transactions=max_transactions,
            hops=hops,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )

    def get_transaction_row(self, transaction_id: int) -> pd.Series:
        rows = self.df[self.df["TransactionID"] == transaction_id]
        if rows.empty:
            raise HTTPException(status_code=404, detail=f"TransactionID {transaction_id} not found")
        return rows.iloc[0]

    def get_transaction_summary(self, transaction_id: int) -> Dict[str, object]:
        row = self.get_transaction_row(transaction_id)
        out = {
            "TransactionID": int(row["TransactionID"]),
            "isFraud": int(row.get("isFraud", 0)),
            "split": str(row.get("split", "")),
            "TransactionDT": int(row.get("TransactionDT", 0)),
            "TransactionAmt": float(row.get("TransactionAmt", 0.0)),
            "pred_raw": float(row.get("pred_raw", 0.0)),
            "pred_uid_blended": float(row.get("pred_uid_blended", 0.0)),
            "pred_score": float(row.get("pred_score", 0.0)),
            "uid_clean": _safe_str(row.get("uid_clean")),
            "ProductCD": _safe_str(row.get("ProductCD")),
            "card1": _safe_str(row.get("card1")),
            "addr1": _safe_str(row.get("addr1")),
            "P_emaildomain": _safe_str(row.get("P_emaildomain")),
            "R_emaildomain": _safe_str(row.get("R_emaildomain")),
            "DeviceInfo": _safe_str(row.get("DeviceInfo")),
            "id_30": _safe_str(row.get("id_30")),
            "id_31": _safe_str(row.get("id_31")),
            "embedding_2d": {
                "x": float(row.get("embed_x", 0.0)),
                "y": float(row.get("embed_y", 0.0)),
            },
        }
        return out

    def explain_transaction(self, transaction_id: int, top_k: int = 12) -> Dict[str, object]:
        if transaction_id not in self.artifacts.txid_to_feature_idx:
            raise HTTPException(status_code=404, detail=f"TransactionID {transaction_id} not found in feature matrix")
        idx = self.artifacts.txid_to_feature_idx[transaction_id]
        x_row = self.artifacts.feature_matrix[idx]
        exp = explain_prediction(
            model=self.artifacts.model,
            backend=self.artifacts.model_backend,
            x_row=x_row,
            feature_names=self.artifacts.feature_cols,
            top_k=top_k,
        )
        exp["TransactionID"] = int(transaction_id)
        exp["pred_score"] = float(self.df.iloc[idx]["pred_score"])
        return exp

    def embedding_space(self, sample_size: int = 2500, seed_transaction_id: Optional[int] = None) -> Dict[str, object]:
        cols = ["TransactionID", "embed_x", "embed_y", "pred_score", "isFraud", "split"]
        df = self.df[cols].copy()
        sample_size = max(200, min(sample_size, len(df)))
        if len(df) > sample_size:
            top_n = int(sample_size * 0.5)
            top = df.nlargest(top_n, "pred_score")
            rest = df.drop(index=top.index).sample(n=sample_size - len(top), random_state=42)
            df = pd.concat([top, rest], axis=0)
        points = [
            {
                "TransactionID": int(r.TransactionID),
                "x": float(r.embed_x),
                "y": float(r.embed_y),
                "risk_score": float(r.pred_score),
                "isFraud": int(r.isFraud),
                "split": str(r.split),
            }
            for r in df.itertuples(index=False)
        ]
        selected = None
        if seed_transaction_id is not None:
            match = self.df[self.df["TransactionID"] == seed_transaction_id]
            if not match.empty:
                s = match.iloc[0]
                selected = {
                    "TransactionID": int(s["TransactionID"]),
                    "x": float(s["embed_x"]),
                    "y": float(s["embed_y"]),
                    "risk_score": float(s["pred_score"]),
                }
        return {"points": points, "selected": selected, "count": len(points)}

    def entity_timeline(self, entity_type: str, entity_value: str, bucket: str = "day") -> Dict[str, object]:
        if entity_type not in self.df.columns:
            raise HTTPException(status_code=404, detail=f"Unsupported entity_type: {entity_type}")
        filt = self.df[self.df[entity_type].fillna("__MISSING__").astype(str) == entity_value]
        if filt.empty:
            return {"entity_type": entity_type, "entity_value": entity_value, "points": []}
        denom = 86_400 if bucket == "day" else 3_600
        time_bucket = (filt["TransactionDT"].astype(np.int64) // denom).astype(np.int64)
        grouped = (
            filt.assign(_bucket=time_bucket)
            .groupby("_bucket")
            .agg(
                transaction_count=("TransactionID", "count"),
                fraud_count=("isFraud", "sum"),
                avg_amount=("TransactionAmt", "mean"),
                avg_pred_score=("pred_score", "mean"),
            )
            .reset_index()
            .sort_values("_bucket")
        )
        points = []
        for r in grouped.to_dict(orient="records"):
            points.append(
                {
                    "bucket": int(r["_bucket"]),
                    "transaction_count": int(r["transaction_count"]),
                    "fraud_count": int(r["fraud_count"]),
                    "fraud_rate": float(r["fraud_count"] / max(1, r["transaction_count"])),
                    "avg_amount": float(r["avg_amount"]),
                    "avg_pred_score": float(r["avg_pred_score"]),
                }
            )
        return {"entity_type": entity_type, "entity_value": entity_value, "bucket": bucket, "points": points}

    def search(self, query: str, limit: int = 20) -> Dict[str, object]:
        q = query.strip()
        if not q:
            return {"query": query, "transactions": [], "entities": []}

        tx_matches: List[int] = []
        if q.isdigit():
            tx = self.df[self.df["TransactionID"].astype(str).str.startswith(q)]["TransactionID"].head(limit)
            tx_matches = [int(v) for v in tx.to_list()]

        entity_matches: List[Dict[str, str]] = []
        for col in ["uid_clean", "DeviceInfo", "P_emaildomain", "R_emaildomain", "card1", "addr1", "id_31"]:
            if col not in self.df.columns:
                continue
            values = (
                self.df[col]
                .fillna("__MISSING__")
                .astype(str)
                .loc[lambda s: s.str.contains(q, case=False, regex=False)]
                .head(3)
                .tolist()
            )
            for v in values:
                entity_matches.append({"entity_type": col, "entity_value": str(v)})
                if len(entity_matches) >= limit:
                    break
            if len(entity_matches) >= limit:
                break

        return {"query": query, "transactions": tx_matches, "entities": entity_matches[:limit]}


class DemoFraudApiService:
    def __init__(self, transactions: pd.DataFrame, seed_transaction_ids: List[int]):
        self.dataset_name = "demo"
        self.df = transactions.reset_index(drop=True).copy()
        self.seed_transaction_ids = seed_transaction_ids
        self.graph_store = FraudGraphStore(transactions=self.df, relation_columns=["card1", "addr1", "DeviceInfo"])

    def graph_overview(
        self,
        max_transactions: int,
        hops: int,
        max_nodes: int,
        max_edges: int,
    ) -> Dict[str, object]:
        seeds = [f"tx:{int(t)}" for t in self.seed_transaction_ids]
        cfg = GraphQueryConfig(
            hops=max(2, hops),
            max_nodes=max_nodes,
            max_edges=max_edges,
            max_entity_fanout=40,
            max_entity_degree=80,
            include_missing_entities=False,
            min_risk=0.0,
        )
        return self.graph_store.subgraph_from_seeds(seeds, cfg)

    def get_transaction_row(self, transaction_id: int) -> pd.Series:
        rows = self.df[self.df["TransactionID"] == transaction_id]
        if rows.empty:
            raise HTTPException(status_code=404, detail=f"TransactionID {transaction_id} not found")
        return rows.iloc[0]

    def get_transaction_summary(self, transaction_id: int) -> Dict[str, object]:
        row = self.get_transaction_row(transaction_id)
        return {
            "TransactionID": int(row["TransactionID"]),
            "isFraud": int(row.get("isFraud", 0)),
            "split": str(row.get("split", "demo")),
            "TransactionDT": int(row.get("TransactionDT", 0)),
            "TransactionAmt": float(row.get("TransactionAmt", 0.0)),
            "pred_raw": float(row.get("pred_raw", 0.0)),
            "pred_uid_blended": float(row.get("pred_uid_blended", 0.0)),
            "pred_score": float(row.get("pred_score", 0.0)),
            "uid_clean": _safe_str(row.get("uid_clean")),
            "ProductCD": _safe_str(row.get("ProductCD")),
            "card1": _safe_str(row.get("card1")),
            "addr1": _safe_str(row.get("addr1")),
            "P_emaildomain": _safe_str(row.get("P_emaildomain")),
            "R_emaildomain": _safe_str(row.get("R_emaildomain")),
            "DeviceInfo": _safe_str(row.get("DeviceInfo")),
            "id_30": _safe_str(row.get("id_30")),
            "id_31": _safe_str(row.get("id_31")),
            "embedding_2d": {
                "x": float(row.get("embed_x", 0.0)),
                "y": float(row.get("embed_y", 0.0)),
            },
        }

    def explain_transaction(self, transaction_id: int, top_k: int = 12) -> Dict[str, object]:
        row = self.get_transaction_row(transaction_id)
        p = float(np.clip(row.get("pred_score", 0.05), 0.01, 0.99))
        base_prob = 0.05

        contributions: List[Tuple[str, float]] = []
        device = str(row.get("DeviceInfo", ""))
        if "DeviceFarm" in device:
            contributions.append(("Shared high-risk device cluster", 0.34))
        if str(row.get("P_emaildomain", "")) != str(row.get("R_emaildomain", "")):
            contributions.append(("Sender/recipient email mismatch", 0.10))
        amt = float(row.get("TransactionAmt", 0.0))
        if amt > 170:
            contributions.append(("Amount above profile norm", 0.12))
        else:
            contributions.append(("Low amount baseline", -0.05))
        if "ring_" in str(row.get("P_emaildomain", "")):
            contributions.append(("Part of linked ring attributes", 0.16))
        if "Home" in device or "family" in str(row.get("P_emaildomain", "")):
            contributions.append(("Household device / shared family email", -0.18))

        if not contributions:
            contributions = [("Baseline behavior", 0.0)]
        contributions = contributions[: max(3, min(top_k, len(contributions)))]

        target_shift = p - base_prob
        raw_sum = sum(v for _, v in contributions)
        scale = 0.0 if abs(raw_sum) < 1e-9 else target_shift / raw_sum
        scaled = [(k, v * scale) for k, v in contributions]

        current = base_prob
        waterfall = []
        for feat, delta in scaled:
            before = current
            current = float(np.clip(current + delta, 0.01, 0.99))
            waterfall.append(
                {
                    "feature": feat,
                    "delta_logit": float(delta),
                    "prob_before": float(before),
                    "prob_after": float(current),
                }
            )

        return {
            "backend": "demo_heuristic_explainer",
            "note": "Scripted demo explanation (not model output); contributions are in probability units.",
            "TransactionID": int(transaction_id),
            "base_probability": float(base_prob),
            "final_probability": float(p),
            "pred_score": float(p),
            "waterfall": waterfall,
            "top_features": [{"feature": w["feature"], "delta_logit": w["delta_logit"]} for w in waterfall],
        }

    def embedding_space(self, sample_size: int = 2500, seed_transaction_id: Optional[int] = None) -> Dict[str, object]:
        cols = ["TransactionID", "embed_x", "embed_y", "pred_score", "isFraud", "split"]
        df = self.df[cols].copy()
        sample_size = max(20, min(sample_size, len(df)))
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        points = [
            {
                "TransactionID": int(r.TransactionID),
                "x": float(r.embed_x),
                "y": float(r.embed_y),
                "risk_score": float(r.pred_score),
                "isFraud": int(r.isFraud),
                "split": str(r.split),
            }
            for r in df.itertuples(index=False)
        ]
        selected = None
        if seed_transaction_id is not None:
            match = self.df[self.df["TransactionID"] == seed_transaction_id]
            if not match.empty:
                s = match.iloc[0]
                selected = {
                    "TransactionID": int(s["TransactionID"]),
                    "x": float(s["embed_x"]),
                    "y": float(s["embed_y"]),
                    "risk_score": float(s["pred_score"]),
                }
        return {"points": points, "selected": selected, "count": len(points)}

    def entity_timeline(self, entity_type: str, entity_value: str, bucket: str = "day") -> Dict[str, object]:
        if entity_type not in self.df.columns:
            raise HTTPException(status_code=404, detail=f"Unsupported entity_type: {entity_type}")
        filt = self.df[self.df[entity_type].fillna("__MISSING__").astype(str) == entity_value]
        if filt.empty:
            return {"entity_type": entity_type, "entity_value": entity_value, "points": []}
        denom = 86_400 if bucket == "day" else 3_600
        time_bucket = (filt["TransactionDT"].astype(np.int64) // denom).astype(np.int64)
        grouped = (
            filt.assign(_bucket=time_bucket)
            .groupby("_bucket")
            .agg(
                transaction_count=("TransactionID", "count"),
                fraud_count=("isFraud", "sum"),
                avg_amount=("TransactionAmt", "mean"),
                avg_pred_score=("pred_score", "mean"),
            )
            .reset_index()
            .sort_values("_bucket")
        )
        points = []
        for r in grouped.to_dict(orient="records"):
            points.append(
                {
                    "bucket": int(r["_bucket"]),
                    "transaction_count": int(r["transaction_count"]),
                    "fraud_count": int(r["fraud_count"]),
                    "fraud_rate": float(r["fraud_count"] / max(1, r["transaction_count"])),
                    "avg_amount": float(r["avg_amount"]),
                    "avg_pred_score": float(r["avg_pred_score"]),
                }
            )
        return {"entity_type": entity_type, "entity_value": entity_value, "bucket": bucket, "points": points}

    def search(self, query: str, limit: int = 20) -> Dict[str, object]:
        q = query.strip()
        if not q:
            return {"query": query, "transactions": [], "entities": []}

        tx_matches: List[int] = []
        if q.isdigit():
            tx = self.df[self.df["TransactionID"].astype(str).str.startswith(q)]["TransactionID"].head(limit)
            tx_matches = [int(v) for v in tx.to_list()]

        entity_matches: List[Dict[str, str]] = []
        for col in ["uid_clean", "DeviceInfo", "P_emaildomain", "R_emaildomain", "card1", "addr1", "id_31"]:
            values = (
                self.df[col]
                .fillna("__MISSING__")
                .astype(str)
                .loc[lambda s: s.str.contains(q, case=False, regex=False)]
                .head(3)
                .tolist()
            )
            for v in values:
                entity_matches.append({"entity_type": col, "entity_value": str(v)})
                if len(entity_matches) >= limit:
                    break
            if len(entity_matches) >= limit:
                break
        return {"query": query, "transactions": tx_matches, "entities": entity_matches[:limit]}


def load_api_artifacts(artifacts_dir: str | Path) -> ApiArtifacts:
    root = Path(artifacts_dir)
    if not root.exists():
        raise FileNotFoundError(f"Artifacts dir does not exist: {root}")

    tx_path = root / "transactions_enriched.csv.gz"
    matrix_path = root / "feature_matrix.npz"
    feature_cols_path = root / "feature_columns.json"
    model_meta_path = root / "model_meta.json"

    transactions = pd.read_csv(tx_path)
    matrix_data = np.load(matrix_path, allow_pickle=False)
    transaction_ids = matrix_data["transaction_ids"].astype(np.int64)
    feature_matrix = matrix_data["X"].astype(np.float32)
    feature_cols = json.loads(feature_cols_path.read_text(encoding="utf-8"))

    txid_to_feature_idx = {int(txid): i for i, txid in enumerate(transaction_ids.tolist())}

    model_meta = json.loads(model_meta_path.read_text(encoding="utf-8"))
    model_backend = str(model_meta["backend"])
    model_path = Path(model_meta["model_path"])
    if not model_path.exists():
        model_path = root / model_path.name

    if model_backend.startswith("xgboost"):
        import xgboost as xgb  # type: ignore

        model = xgb.Booster()
        model.load_model(str(model_path))
    else:
        with model_path.open("rb") as f:
            model = pickle.load(f)

    relation_columns = model_meta.get("graph_relation_columns") or GRAPH_RELATION_COLUMNS
    graph_store = FraudGraphStore(transactions=transactions, relation_columns=relation_columns)

    return ApiArtifacts(
        artifacts_dir=root,
        transactions=transactions,
        feature_matrix=feature_matrix,
        feature_cols=feature_cols,
        txid_to_feature_idx=txid_to_feature_idx,
        model=model,
        model_backend=model_backend,
        graph_store=graph_store,
    )


def create_app(artifacts_dir: str = "outputs/backend_api/artifacts") -> FastAPI:
    app = FastAPI(
        title="FraudScope Backend API",
        version="0.1.0",
        description="Graph-driven fraud detection backend for dynamic visualization.",
    )
    cors_env = os.getenv("FRAUD_API_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    allow_origins = [x.strip() for x in cors_env.split(",") if x.strip()]
    if not allow_origins:
        allow_origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    artifacts = load_api_artifacts(artifacts_dir)
    main_service = FraudApiService(artifacts)
    demo_bundle = build_demo_scenario(random_state=42)
    demo_service = DemoFraudApiService(
        transactions=demo_bundle.transactions,
        seed_transaction_ids=demo_bundle.seed_transaction_ids,
    )

    services: Dict[str, object] = {
        "main": main_service,
        "demo": demo_service,
    }
    app.state.services = services

    def _service(dataset: str):
        svc = services.get(dataset)
        if svc is None:
            raise HTTPException(status_code=400, detail=f"Unsupported dataset '{dataset}'. Use 'main' or 'demo'.")
        return svc

    @app.get("/health")
    def health() -> Dict[str, object]:
        return {
            "status": "ok",
            "transactions_main": int(len(main_service.df)),
            "transactions_demo": int(len(demo_service.df)),
            "model_backend": artifacts.model_backend,
            "datasets": ["main", "demo"],
        }

    @app.get("/api/v1/meta")
    def meta(
        dataset: str = Query("main", pattern="^(main|demo)$"),
    ) -> Dict[str, object]:
        svc = _service(dataset)
        return {
            "dataset": dataset,
            "artifacts_dir": str(artifacts.artifacts_dir),
            "feature_count": len(artifacts.feature_cols),
            "relation_columns": svc.graph_store.relation_columns,
            "transaction_count": int(len(svc.df)),
        }

    @app.get("/api/v1/graph/overview")
    def graph_overview(
        dataset: str = Query("main", pattern="^(main|demo)$"),
        max_transactions: int = Query(120, ge=20, le=2000),
        hops: int = Query(2, ge=1, le=3),
        max_nodes: int = Query(1200, ge=100, le=12000),
        max_edges: int = Query(5000, ge=100, le=60000),
    ) -> Dict[str, object]:
        svc = _service(dataset)
        return svc.graph_overview(
            max_transactions=max_transactions,
            hops=hops,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )

    @app.get("/api/v1/graph/neighborhood")
    def graph_neighborhood(
        dataset: str = Query("main", pattern="^(main|demo)$"),
        node_id: str = Query(..., description="Example: tx:2000123 or ent:DeviceInfo:iOS Device"),
        hops: int = Query(2, ge=1, le=4),
        max_nodes: int = Query(800, ge=50, le=15000),
        max_edges: int = Query(4000, ge=50, le=100000),
        max_entity_fanout: int = Query(180, ge=10, le=2000),
        max_entity_degree: int = Query(900, ge=10, le=10000),
        include_missing_entities: bool = Query(False),
        min_risk: float = Query(0.0, ge=0.0, le=1.0),
    ) -> Dict[str, object]:
        cfg = GraphQueryConfig(
            hops=hops,
            max_nodes=max_nodes,
            max_edges=max_edges,
            max_entity_fanout=max_entity_fanout,
            max_entity_degree=max_entity_degree,
            include_missing_entities=include_missing_entities,
            min_risk=min_risk,
        )
        try:
            svc = _service(dataset)
            return svc.graph_store.subgraph_from_seeds([node_id], cfg)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/v1/graph/transaction/{transaction_id}")
    def graph_for_transaction(
        transaction_id: int,
        dataset: str = Query("main", pattern="^(main|demo)$"),
        hops: int = Query(2, ge=1, le=4),
        max_nodes: int = Query(800, ge=50, le=15000),
        max_edges: int = Query(4000, ge=50, le=100000),
        max_entity_degree: int = Query(900, ge=10, le=10000),
        include_missing_entities: bool = Query(False),
        min_risk: float = Query(0.0, ge=0.0, le=1.0),
    ) -> Dict[str, object]:
        cfg = GraphQueryConfig(
            hops=hops,
            max_nodes=max_nodes,
            max_edges=max_edges,
            max_entity_degree=max_entity_degree,
            include_missing_entities=include_missing_entities,
            min_risk=min_risk,
        )
        svc = _service(dataset)
        return svc.graph_store.subgraph_from_seeds([f"tx:{transaction_id}"], cfg)

    @app.get("/api/v1/graph/patterns/stars")
    def star_patterns(
        dataset: str = Query("main", pattern="^(main|demo)$"),
        top_k: int = Query(20, ge=1, le=200),
        min_degree: int = Query(20, ge=2, le=5000),
    ) -> Dict[str, object]:
        svc = _service(dataset)
        rows = svc.graph_store.star_patterns(top_k=top_k, min_degree=min_degree)
        return {"patterns": rows, "count": len(rows)}

    @app.get("/api/v1/graph/patterns/rings")
    def ring_patterns(
        dataset: str = Query("main", pattern="^(main|demo)$"),
        top_k: int = Query(20, ge=1, le=200),
        max_seed_transactions: int = Query(2000, ge=100, le=10000),
    ) -> Dict[str, object]:
        svc = _service(dataset)
        rows = svc.graph_store.ring_patterns(top_k=top_k, max_seed_transactions=max_seed_transactions)
        return {"patterns": rows, "count": len(rows)}

    @app.get("/api/v1/transactions/{transaction_id}")
    def transaction_detail(
        transaction_id: int,
        dataset: str = Query("main", pattern="^(main|demo)$"),
    ) -> Dict[str, object]:
        svc = _service(dataset)
        return svc.get_transaction_summary(transaction_id)

    @app.get("/api/v1/transactions/{transaction_id}/explain")
    def transaction_explain(
        transaction_id: int,
        dataset: str = Query("main", pattern="^(main|demo)$"),
        top_k: int = Query(12, ge=3, le=30),
    ) -> Dict[str, object]:
        svc = _service(dataset)
        return svc.explain_transaction(transaction_id=transaction_id, top_k=top_k)

    @app.get("/api/v1/embedding-space")
    def embedding_space(
        dataset: str = Query("main", pattern="^(main|demo)$"),
        sample_size: int = Query(2500, ge=200, le=20000),
        seed_transaction_id: Optional[int] = Query(None),
    ) -> Dict[str, object]:
        svc = _service(dataset)
        return svc.embedding_space(sample_size=sample_size, seed_transaction_id=seed_transaction_id)

    @app.get("/api/v1/timeline/entity")
    def timeline_entity(
        dataset: str = Query("main", pattern="^(main|demo)$"),
        entity_type: str = Query(..., description="Example: DeviceInfo, card1, uid_clean, P_emaildomain"),
        entity_value: str = Query(...),
        bucket: str = Query("day", pattern="^(day|hour)$"),
    ) -> Dict[str, object]:
        svc = _service(dataset)
        return svc.entity_timeline(entity_type=entity_type, entity_value=entity_value, bucket=bucket)

    @app.get("/api/v1/search")
    def search(
        dataset: str = Query("main", pattern="^(main|demo)$"),
        q: str = Query(..., min_length=1),
        limit: int = Query(20, ge=1, le=100),
    ) -> Dict[str, object]:
        svc = _service(dataset)
        return svc.search(query=q, limit=limit)

    return app
