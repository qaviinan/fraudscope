from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
from typing import Deque, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd


GRAPH_RELATION_COLUMNS = [
    "card1",
    "addr1",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "id_30",
    "id_31",
]

MISSING_ENTITY_VALUES = {"__MISSING__", "nan", "None", ""}


def tx_node_id(transaction_id: int) -> str:
    return f"tx:{int(transaction_id)}"


def entity_node_id(column: str, value: str) -> str:
    return f"ent:{column}:{value}"


def parse_node_id(node_id: str) -> Tuple[str, str, str]:
    if node_id.startswith("tx:"):
        return "tx", "TransactionID", node_id.split(":", 1)[1]
    if node_id.startswith("ent:"):
        _, col, value = node_id.split(":", 2)
        return "ent", col, value
    raise ValueError(f"Invalid node_id: {node_id}")


def _risk_to_color(score: float) -> str:
    s = float(np.clip(score, 0.0, 1.0))
    r = int(255 * s)
    g = int(220 * (1.0 - s))
    b = 70
    return f"#{r:02x}{g:02x}{b:02x}"


@dataclass
class GraphQueryConfig:
    hops: int = 1
    max_nodes: int = 600
    max_edges: int = 2500
    max_entity_fanout: int = 180
    max_entity_degree: int = 900
    include_missing_entities: bool = False
    min_risk: float = 0.0


class FraudGraphStore:
    def __init__(
        self,
        transactions: pd.DataFrame,
        relation_columns: Sequence[str] = GRAPH_RELATION_COLUMNS,
        pred_col: str = "pred_score",
    ) -> None:
        self.df = transactions.reset_index(drop=True).copy()
        self.relation_columns = [c for c in relation_columns if c in self.df.columns]
        self.pred_col = pred_col

        self.txid_to_index: Dict[int, int] = {}
        self.entity_to_tx_indices: Dict[str, List[int]] = defaultdict(list)
        self.entity_stats: Dict[str, Dict[str, float]] = {}

        for i, txid in enumerate(self.df["TransactionID"].astype(np.int64).to_numpy()):
            self.txid_to_index[int(txid)] = i

        for col in self.relation_columns:
            values = self.df[col].fillna("__MISSING__").astype(str).to_numpy()
            for i, val in enumerate(values):
                self.entity_to_tx_indices[entity_node_id(col, val)].append(i)

        scores = self.df[self.pred_col].to_numpy(dtype=np.float32)
        for ent_id, idxs in self.entity_to_tx_indices.items():
            arr = np.array(idxs, dtype=np.int32)
            ent_scores = scores[arr]
            self.entity_stats[ent_id] = {
                "degree": float(len(idxs)),
                "risk_mean": float(ent_scores.mean()) if len(ent_scores) else 0.0,
                "risk_max": float(ent_scores.max()) if len(ent_scores) else 0.0,
            }

    def _is_missing_entity_value(self, value: str) -> bool:
        return value in MISSING_ENTITY_VALUES

    def _entity_degree(self, ent_id: str) -> int:
        return int(self.entity_stats.get(ent_id, {}).get("degree", 0))

    def _tx_node(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        score = float(row[self.pred_col])
        return {
            "id": tx_node_id(int(row["TransactionID"])),
            "node_type": "transaction",
            "label": f"TX {int(row['TransactionID'])}",
            "risk_score": score,
            "risk_color": _risk_to_color(score),
            "isFraud": int(row["isFraud"]) if "isFraud" in row else 0,
            "amount": float(row.get("TransactionAmt", 0.0)),
            "timestamp": int(row.get("TransactionDT", 0)),
            "x": float(row.get("embed_x", 0.0)),
            "y": float(row.get("embed_y", 0.0)),
        }

    def _entity_node(self, ent_id: str) -> Dict[str, object]:
        _, col, value = parse_node_id(ent_id)
        stats = self.entity_stats.get(ent_id, {"degree": 0.0, "risk_mean": 0.0, "risk_max": 0.0})
        return {
            "id": ent_id,
            "node_type": "entity",
            "entity_type": col,
            "label": f"{col}:{value}",
            "value": value,
            "degree": int(stats["degree"]),
            "risk_score": float(stats["risk_mean"]),
            "risk_color": _risk_to_color(float(stats["risk_mean"])),
        }

    def _tx_entity_ids(self, idx: int) -> List[str]:
        row = self.df.iloc[idx]
        out: List[str] = []
        for col in self.relation_columns:
            value = str(row[col]) if pd.notna(row[col]) else "__MISSING__"
            out.append(entity_node_id(col, value))
        return out

    def _expand_entity(
        self,
        ent_id: str,
        cfg: GraphQueryConfig,
    ) -> List[int]:
        _, _, value = parse_node_id(ent_id)
        if (not cfg.include_missing_entities) and self._is_missing_entity_value(value):
            return []
        if self._entity_degree(ent_id) > cfg.max_entity_degree:
            return []
        idxs = self.entity_to_tx_indices.get(ent_id, [])
        if not idxs:
            return []
        arr = np.array(idxs, dtype=np.int32)
        scores = self.df.iloc[arr][self.pred_col].to_numpy(dtype=np.float32)
        if cfg.min_risk > 0:
            keep = scores >= cfg.min_risk
            arr = arr[keep]
            scores = scores[keep]
        if len(arr) > cfg.max_entity_fanout:
            order = np.argsort(scores)[::-1][: cfg.max_entity_fanout]
            arr = arr[order]
        return arr.astype(np.int32).tolist()

    def subgraph_from_seeds(
        self,
        seed_node_ids: Sequence[str],
        cfg: GraphQueryConfig,
    ) -> Dict[str, object]:
        nodes: Dict[str, Dict[str, object]] = {}
        links: Dict[Tuple[str, str, str], Dict[str, object]] = {}
        queue: Deque[Tuple[str, int]] = deque()
        seen: Set[str] = set()

        for node_id in seed_node_ids:
            queue.append((node_id, 0))
            seen.add(node_id)

        while queue and len(nodes) < cfg.max_nodes and len(links) < cfg.max_edges:
            current, depth = queue.popleft()
            kind, _, ident = parse_node_id(current)

            if kind == "tx":
                txid = int(ident)
                if txid not in self.txid_to_index:
                    continue
                idx = self.txid_to_index[txid]
                nodes[current] = self._tx_node(idx)
                if depth >= cfg.hops:
                    continue
                for ent_id in self._tx_entity_ids(idx):
                    _, col, value = parse_node_id(ent_id)
                    if (not cfg.include_missing_entities) and self._is_missing_entity_value(value):
                        continue
                    if self._entity_degree(ent_id) > cfg.max_entity_degree:
                        continue
                    if ent_id not in nodes:
                        nodes[ent_id] = self._entity_node(ent_id)
                    edge_key = (current, ent_id, col)
                    links[edge_key] = {
                        "source": current,
                        "target": ent_id,
                        "relation": col,
                    }
                    if ent_id not in seen:
                        seen.add(ent_id)
                        queue.append((ent_id, depth + 1))
            else:
                nodes[current] = self._entity_node(current)
                if depth >= cfg.hops:
                    continue
                _, col, _ = parse_node_id(current)
                for idx in self._expand_entity(current, cfg):
                    tx = self._tx_node(idx)
                    tx_id = tx["id"]
                    nodes[tx_id] = tx
                    edge_key = (tx_id, current, col)
                    links[edge_key] = {
                        "source": tx_id,
                        "target": current,
                        "relation": col,
                    }
                    if tx_id not in seen:
                        seen.add(tx_id)
                        queue.append((tx_id, depth + 1))

        return {
            "nodes": list(nodes.values()),
            "links": list(links.values()),
            "meta": {
                "seed_nodes": list(seed_node_ids),
                "hops": cfg.hops,
                "node_count": len(nodes),
                "link_count": len(links),
            },
        }

    def overview_graph(
        self,
        max_transactions: int = 120,
        hops: int = 2,
        max_nodes: int = 1200,
        max_edges: int = 5000,
    ) -> Dict[str, object]:
        max_transactions = max(10, int(max_transactions))
        tx = self.df[["TransactionID", self.pred_col]].copy()
        tx = tx.sort_values(self.pred_col, ascending=False)
        seed_count = 1
        seeds = [tx_node_id(t) for t in tx.head(seed_count)["TransactionID"].astype(int).tolist()]
        cfg = GraphQueryConfig(
            hops=hops,
            max_nodes=max_nodes,
            max_edges=max_edges,
            max_entity_degree=700,
            include_missing_entities=False,
            min_risk=0.0,
        )
        return self.subgraph_from_seeds(seeds, cfg)

    def transaction_graph(
        self,
        transaction_id: int,
        hops: int = 2,
        max_nodes: int = 600,
        max_edges: int = 2500,
        min_risk: float = 0.0,
    ) -> Dict[str, object]:
        cfg = GraphQueryConfig(
            hops=hops,
            max_nodes=max_nodes,
            max_edges=max_edges,
            min_risk=min_risk,
        )
        return self.subgraph_from_seeds([tx_node_id(transaction_id)], cfg)

    def star_patterns(self, top_k: int = 20, min_degree: int = 20) -> List[Dict[str, object]]:
        rows = []
        for ent_id, stats in self.entity_stats.items():
            degree = int(stats["degree"])
            if degree < min_degree:
                continue
            _, col, value = parse_node_id(ent_id)
            rows.append(
                {
                    "node_id": ent_id,
                    "entity_type": col,
                    "entity_value": value,
                    "degree": degree,
                    "risk_mean": float(stats["risk_mean"]),
                    "risk_max": float(stats["risk_max"]),
                    "score": float(degree * stats["risk_mean"]),
                }
            )
        rows.sort(key=lambda x: x["score"], reverse=True)
        return rows[: max(1, int(top_k))]

    def ring_patterns(self, top_k: int = 20, max_seed_transactions: int = 2000) -> List[Dict[str, object]]:
        scored = self.df.sort_values(self.pred_col, ascending=False).head(max_seed_transactions)
        idxs = scored.index.to_numpy(dtype=np.int32)
        idx_set = set(int(i) for i in idxs.tolist())

        pair_relations: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
        for col in self.relation_columns:
            groups: Dict[str, List[int]] = defaultdict(list)
            vals = self.df[col].fillna("__MISSING__").astype(str).to_numpy()
            for i in idxs:
                groups[vals[i]].append(int(i))
            for members in groups.values():
                if len(members) < 2:
                    continue
                if len(members) > 40:
                    members = members[:40]
                for a, b in combinations(members, 2):
                    key = (a, b) if a < b else (b, a)
                    pair_relations[key].add(col)

        rings = []
        for (a, b), rels in pair_relations.items():
            if len(rels) < 2 or a not in idx_set or b not in idx_set:
                continue
            row_a = self.df.iloc[a]
            row_b = self.df.iloc[b]
            score = float((row_a[self.pred_col] + row_b[self.pred_col]) * len(rels))
            rings.append(
                {
                    "tx_a": int(row_a["TransactionID"]),
                    "tx_b": int(row_b["TransactionID"]),
                    "shared_relations": sorted(rels),
                    "shared_relation_count": len(rels),
                    "score": score,
                }
            )
        rings.sort(key=lambda x: x["score"], reverse=True)
        return rings[: max(1, int(top_k))]
