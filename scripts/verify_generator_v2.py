#!/usr/bin/env python3
"""
Verify that generator v2 produces coherent data with recoverable graph structure.

Checks, in order:
  1. Population / identity-table / label structure (does the data look like IEEE-CIS, is missingness a mechanism)
  2. Graph structure: entity sharing across accounts, and whether shared entities carry fraud (rings) or not (households)
  3. Temporal structure: burstiness of ring activity vs customers
  4. Bayes ceiling from the generator's own label probabilities
  5. Nested ablation with a log-loss XGBoost: transaction features -> + UID aggregates -> + causal graph
     features -> + SVD embeddings, with investigation-queue metrics
  6. Identity-noise sweep: card reissue rate 0 / 6 / 15 / 30 %, comparing UID-aggregate models with graph-feature models
  7. What the graph store's overview / star / ring queries return on calibrated scores

Usage: python scripts/verify_generator_v2.py --n-transactions 120000 --out outputs/verify_v2.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from fraud_graphs.backend_builder import V2_GRAPH_RELATION_COLUMNS, _encode_categorical, _time_split
from fraud_graphs.features import apply_uid_aggregates, fit_uid_aggregates, uid_label_propagation_blend
from fraud_graphs.graph_embeddings import SparseGraphEmbedder
from fraud_graphs.graph_features import build_causal_graph_features
from fraud_graphs.graph_module import FraudGraphStore
from fraud_graphs.modeling import evaluate_scores
from fraud_graphs.synthetic_v2 import DIAGNOSTIC_COLUMNS, SyntheticV2Config, generate_synthetic_v2

BASE_FEATURES = [
    "TransactionDT", "TransactionAmt", "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2",
    "dist1", "dist2", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo", "id_30", "id_31", "id_33", "id_36", "id_37", "id_38",
    *[f"C{i}" for i in (1, 2, 3, 4, 5, 11, 12, 13, 14)], *[f"D{i}" for i in range(1, 16)], *[f"M{i}" for i in range(1, 10)],
]
VESTA_GRAPH_COUNTS = ["C6", "C7", "C8", "C9", "C10"]   # distinct-entities-per-entity counts (what the real C columns are)
UID_FEATURES = ["uid_tx_count", "uid_amt_mean", "uid_amt_std", "uid_amt_median", "uid_d1_mean", "uid_c1_mean", "uid_seen_in_train"]
CATEGORICAL = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo", "id_30", "id_31", "id_33", "id_36", "id_37", "id_38", *[f"M{i}" for i in range(1, 10)]]


def prepare(cfg: SyntheticV2Config, seed: int):
    tx, idf = generate_synthetic_v2(cfg)
    full = tx.merge(idf, on="TransactionID", how="left")
    train_df, valid_df, test_df = _time_split(full, 0.70, 0.15)
    bundle = fit_uid_aggregates(train_df)
    train_df, valid_df, test_df = (apply_uid_aggregates(d, bundle) for d in (train_df, valid_df, test_df))
    df = pd.concat([train_df.assign(split="train"), valid_df.assign(split="valid"), test_df.assign(split="test")], ignore_index=True)
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    train_mask = (df["split"] == "train").to_numpy()
    causal = build_causal_graph_features(df, entity_columns=V2_GRAPH_RELATION_COLUMNS, label_known_mask=train_mask)
    emb = SparseGraphEmbedder(V2_GRAPH_RELATION_COLUMNS, 32, 2, seed)
    emb.fit(df.loc[train_mask])
    svd = emb.transform(df)
    df = pd.concat([df, causal, svd], axis=1)
    return df, list(causal.columns), list(svd.columns)


def fit_variants(df: pd.DataFrame, causal_cols: List[str], svd_cols: List[str], seed: int, variants: Dict[str, List[str]]):
    all_cols = sorted(set(sum(variants.values(), [])), key=lambda c: c)
    assert not set(all_cols) & set(DIAGNOSTIC_COLUMNS)
    train_mask = df["split"] == "train"
    X_df, _ = _encode_categorical(df[all_cols].copy(), df.index[train_mask], [c for c in CATEGORICAL if c in all_cols])
    X = X_df.to_numpy(dtype=np.float32)
    y = df["isFraud"].to_numpy(dtype=np.int8)
    tr, va, te = (np.where(df["split"] == s)[0] for s in ("train", "valid", "test"))
    amt = df["TransactionAmt"].to_numpy(dtype=np.float64)
    idx = {c: i for i, c in enumerate(all_cols)}
    params = {"max_depth": 6, "eta": 0.05, "subsample": 0.85, "colsample_bytree": 0.8, "lambda": 2.0, "objective": "binary:logistic", "eval_metric": "aucpr", "seed": seed}
    out: Dict[str, object] = {}
    preds: Dict[str, np.ndarray] = {}
    for name, cols in variants.items():
        ci = [idx[c] for c in cols]
        dtr, dva = xgb.DMatrix(X[tr][:, ci], label=y[tr]), xgb.DMatrix(X[va][:, ci], label=y[va])
        bst = xgb.train(params, dtr, 800, evals=[(dva, "valid")], early_stopping_rounds=60, verbose_eval=False)
        p = bst.predict(xgb.DMatrix(X[:, ci]), iteration_range=(0, bst.best_iteration + 1))
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[va], y[va])
        pc = cal.predict(p)
        blend = uid_label_propagation_blend(df, p, blend_alpha=0.6)
        gain = bst.get_score(importance_type="gain")
        top = sorted(((cols[int(k[1:])], round(v, 1)) for k, v in gain.items()), key=lambda kv: -kv[1])[:10]
        # reliability of calibrated score on test
        bins = np.clip((pc[te] * 10).astype(int), 0, 9)
        reliability = [(round(float(pc[te][bins == b].mean()), 3), round(float(y[te][bins == b].mean()), 3), int((bins == b).sum())) for b in range(10) if (bins == b).sum() > 0]
        out[name] = {
            "n_features": len(cols), "best_iteration": int(bst.best_iteration),
            "test": evaluate_scores(y[te], p[te], amt[te]),
            "test_uid_blended": evaluate_scores(y[te], blend[te], amt[te]),
            "top_gain_features": top,
            "calibrated_test_score_range": [float(pc[te].min()), float(np.median(pc[te])), float(pc[te].max())],
            "share_calibrated_ge_0.65": float((pc[te] >= 0.65).mean()),
            "reliability_(mean_score, fraud_rate, n)": reliability,
        }
        preds[name] = pc
    return out, preds, y, te


def structure_report(df: pd.DataFrame) -> Dict[str, object]:
    R: Dict[str, object] = {}
    R["n_tx"] = int(len(df)); R["fraud_rate"] = float(df["isFraud"].mean())
    R["actor_mix"] = df["actor_type"].value_counts().to_dict()
    R["accounts_true"] = int(df["uid_true"].nunique()); R["accounts_proxy"] = int(df["uid_proxy"].nunique())
    R["identity_table_coverage"] = float(df["DeviceInfo"].notna().mean())
    R["identity_coverage_by_product"] = df.groupby("ProductCD")["DeviceInfo"].apply(lambda s: float(s.notna().mean())).round(3).to_dict()
    R["fraud_rate_by_product"] = df.groupby("ProductCD")["isFraud"].mean().round(4).to_dict()
    R["fraud_rate_identity_present_vs_absent"] = [float(df.loc[df["DeviceInfo"].notna(), "isFraud"].mean()), float(df.loc[df["DeviceInfo"].isna(), "isFraud"].mean())]
    ent: Dict[str, object] = {}
    for col in V2_GRAPH_RELATION_COLUMNS:
        d = df[df[col].notna()]
        accounts_per = d.groupby(col)["uid_true"].nunique()
        shared = d[col].map(accounts_per) > 1
        legit_shared = d[(d["actor_type"] == "legit") & shared]
        ent[col] = {
            "n_distinct": int(accounts_per.size),
            "share_of_values_used_by_gt1_account": float((accounts_per > 1).mean()),
            "share_of_tx_on_shared_value": float(shared.mean()),
            "fraud_rate_shared_vs_unshared": [float(d.loc[shared, "isFraud"].mean()), float(d.loc[~shared, "isFraud"].mean())],
            "share_of_shared_tx_that_are_legit(benign_sharing)": float((d.loc[shared, "actor_type"] == "legit").mean()),
            "max_accounts_on_one_value": int(accounts_per.max()),
        }
    R["entity_sharing"] = ent
    g = df.sort_values(["uid_true", "TransactionDT"]).assign(gap_h=lambda x: x.groupby("uid_true")["TransactionDT"].diff() / 3600)
    R["median_gap_hours_by_actor"] = g.groupby("actor_type")["gap_h"].median().round(2).to_dict()
    R["share_of_tx_within_1h_of_previous_same_account"] = g.groupby("actor_type")["gap_h"].apply(lambda s: float((s < 1).mean())).round(3).to_dict()
    R["median_amount_by_actor"] = df.groupby("actor_type")["TransactionAmt"].median().round(1).to_dict()
    R["share_micro_amounts_lt_5_by_actor"] = df.groupby("actor_type")["TransactionAmt"].apply(lambda s: float((s < 5).mean())).round(3).to_dict()
    R["account_age_D1_median_by_actor"] = df.groupby("actor_type")["D1"].median().round(0).to_dict()
    R["fraud_tx_per_ring_quantiles"] = np.quantile(df[df["ring_id"] >= 0].groupby("ring_id").size(), [0, 0.5, 1]).tolist()
    return R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-transactions", type=int, default=120_000)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--out", default="outputs/verify_v2.json")
    ap.add_argument("--quick", action="store_true", help="skip the identity-noise sweep")
    args = ap.parse_args()
    t0 = time.time()
    seed = args.random_state

    cfg = SyntheticV2Config(n_transactions=args.n_transactions, random_state=seed)
    df, causal_cols, svd_cols = prepare(cfg, seed)
    R: Dict[str, object] = {"config": cfg.__dict__, "structure": structure_report(df)}
    print("structure:", json.dumps(R["structure"], indent=1, default=str)[:3000], flush=True)

    te = np.where(df["split"] == "test")[0]
    y = df["isFraud"].to_numpy(); amt = df["TransactionAmt"].to_numpy()
    R["oracle_generator_true_p"] = evaluate_scores(y[te], df["true_p"].to_numpy()[te], amt[te])

    variants = {
        "1_transaction_features": BASE_FEATURES,
        "2_plus_uid_aggregates": BASE_FEATURES + UID_FEATURES,
        "3_plus_vesta_C6-C10_counts": BASE_FEATURES + UID_FEATURES + VESTA_GRAPH_COUNTS,
        "4_plus_causal_graph": BASE_FEATURES + UID_FEATURES + VESTA_GRAPH_COUNTS + causal_cols,
        "4b_plus_svd_embeddings": BASE_FEATURES + UID_FEATURES + VESTA_GRAPH_COUNTS + causal_cols + svd_cols,
        "5_causal_graph_only": causal_cols,
        "6_svd_only": svd_cols,
    }
    ablation, preds, y, te = fit_variants(df, causal_cols, svd_cols, seed, variants)
    R["ablation"] = ablation
    print("\nABLATION (test split)")
    print(f"{'variant':28s} {'feat':>5s} {'PR-AUC':>7s} {'ROC':>6s} {'P@100':>6s} {'P@500':>6s} {'R@500':>6s} {'$@500':>6s}")
    for k, v in ablation.items():
        t = v["test"]
        print(f"{k:28s} {v['n_features']:5d} {t['pr_auc']:7.3f} {t['roc_auc']:6.3f} {t['precision@100']:6.2f} {t['precision@500']:6.3f} {t['recall@500']:6.3f} {t['fraud_dollars_captured@500']:6.3f}")
    o = R["oracle_generator_true_p"]
    print(f"{'oracle (true_p)':28s} {'-':>5s} {o['pr_auc']:7.3f} {o['roc_auc']:6.3f} {o['precision@100']:6.2f} {o['precision@500']:6.3f} {o['recall@500']:6.3f} {o['fraud_dollars_captured@500']:6.3f}")

    # ---- identity noise sweep ----
    sweep = {}
    for rate in (() if args.quick else (0.0, 0.06, 0.15, 0.30)):
        c2 = SyntheticV2Config(n_transactions=args.n_transactions, random_state=seed, card_reissue_rate=rate)
        d2, cc2, sc2 = prepare(c2, seed)
        abl, _, _, _ = fit_variants(d2, cc2, sc2, seed, {"uid_aggregates": BASE_FEATURES + UID_FEATURES, "causal_graph": BASE_FEATURES + VESTA_GRAPH_COUNTS + cc2})
        sweep[f"reissue_{rate:.2f}"] = {"proxy_accounts_per_true_account": float(d2["uid_proxy"].nunique() / d2["uid_true"].nunique()),
                                       "uid_aggregates_pr_auc": abl["uid_aggregates"]["test"]["pr_auc"], "causal_graph_pr_auc": abl["causal_graph"]["test"]["pr_auc"],
                                       "uid_aggregates_p@500": abl["uid_aggregates"]["test"]["precision@500"], "causal_graph_p@500": abl["causal_graph"]["test"]["precision@500"]}
        print("noise", rate, sweep[f"reissue_{rate:.2f}"], flush=True)
    R["identity_noise_sweep"] = sweep

    # ---- graph store on calibrated scores ----
    df["pred_raw"] = preds["4_plus_causal_graph"]; df["pred_score"] = df["pred_raw"]
    df["embed_x"], df["embed_y"] = df[svd_cols[0]], df[svd_cols[1]]
    store = FraudGraphStore(df, relation_columns=V2_GRAPH_RELATION_COLUMNS)
    ov = store.overview_graph(max_transactions=90, hops=2, max_nodes=1400, max_edges=7000)
    txs = [n for n in ov["nodes"] if n["node_type"] == "transaction"]; ents = [n for n in ov["nodes"] if n["node_type"] == "entity"]
    R["overview"] = {"n_tx": len(txs), "fraud_share": float(np.mean([t["isFraud"] for t in txs])) if txs else None, "entities": [(e["label"], e["degree"]) for e in ents][:10],
                     "distinct_true_accounts": int(df.set_index("TransactionID").loc[[int(t["id"][3:]) for t in txs], "uid_true"].nunique())}
    stars = store.star_patterns(top_k=10, min_degree=4)
    tid_actor = df.set_index("TransactionID")["actor_type"]
    R["stars_top10"] = [(s["entity_type"], s["entity_value"], s["degree"], round(s["risk_mean"], 3), float(df.iloc[store.entity_to_tx_indices[s["node_id"]]]["isFraud"].mean())) for s in stars]
    rings = store.ring_patterns(top_k=200, max_seed_transactions=2500)
    uid_of = df.set_index("TransactionID")["uid_true"]
    R["rings"] = {"top200_cross_account_share": float(np.mean([uid_of[r["tx_a"]] != uid_of[r["tx_b"]] for r in rings])),
                  "top200_both_fraud_share": float(np.mean([df.set_index('TransactionID').loc[r['tx_a'], 'isFraud'] == 1 and df.set_index('TransactionID').loc[r['tx_b'], 'isFraud'] == 1 for r in rings[:50]]))}
    print("overview", R["overview"]); print("stars", R["stars_top10"]); print("rings", R["rings"])
    R["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(R, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    print("wrote", args.out, R["elapsed_s"], "s")


if __name__ == "__main__":
    main()
