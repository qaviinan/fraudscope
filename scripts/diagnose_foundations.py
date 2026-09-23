#!/usr/bin/env python3
"""
Diagnose the FraudScope data -> features -> model -> graph foundation.

Reproduces every number quoted in docs/foundational_flow_review.md:
  * synthetic-data structure statistics (users, entity degrees, label components)
  * the Bayes ceiling implied by the generator's own label probabilities
  * ablations: repo objective vs plain logistic, with/without graph embeddings,
    with/without UID aggregates, graph embeddings alone, and the UID blend
  * sensitivity of the UID blend to merged/split identities
  * what the graph store's overview / star / ring queries return

If the real IEEE-CIS CSVs are present, reference profiles are loaded from them
(exactly as the build does). Otherwise a hand-coded approximation of the public
IEEE-CIS marginals is used; the pipeline mechanics, not the exact marginals,
drive every conclusion in the report.

Usage:
  python scripts/diagnose_foundations.py --n-transactions 120000 --out outputs/diagnostics.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import fraud_graphs.synthetic as synth
from fraud_graphs.backend_builder import _encode_categorical, _time_split
from fraud_graphs.features import apply_uid_aggregates, fit_uid_aggregates, uid_label_propagation_blend
from fraud_graphs.graph_embeddings import DEFAULT_RELATION_COLUMNS, SparseGraphEmbedder
from fraud_graphs.graph_module import FraudGraphStore
from fraud_graphs.modeling import asymmetric_focal_objective, evaluate_scores
from fraud_graphs.synthetic import ReferenceProfiles, SyntheticConfig, generate_synthetic_ieee_data, load_reference_profiles


def _cat(d: Dict[str, float]):
    vals = np.array(list(d.keys()), dtype=object)
    p = np.array(list(d.values()), dtype=np.float64)
    return vals, p / p.sum()


def approximate_ieee_profiles() -> ReferenceProfiles:
    """Approximate public IEEE-CIS train marginals (identity table left-joined => ~76% missing)."""
    categorical = {
        "ProductCD": _cat({"W": 0.7449, "C": 0.1160, "R": 0.0638, "H": 0.0559, "S": 0.0195}),
        "card4": _cat({"visa": 0.6528, "mastercard": 0.3200, "american express": 0.0143, "discover": 0.0113, "__MISSING__": 0.0027}),
        "card6": _cat({"debit": 0.7466, "credit": 0.2504, "__MISSING__": 0.0026, "debit or credit": 0.0002, "charge card": 0.0002}),
        "P_emaildomain": _cat({"gmail.com": 0.387, "yahoo.com": 0.171, "__MISSING__": 0.160, "hotmail.com": 0.076, "anonymous.com": 0.063, "aol.com": 0.048, "comcast.net": 0.013, "icloud.com": 0.011, "outlook.com": 0.009, "msn.com": 0.007, "att.net": 0.006, "live.com": 0.005, "sbcglobal.net": 0.005, "verizon.net": 0.004, "ymail.com": 0.004, "bellsouth.net": 0.003, "me.com": 0.003, "cox.net": 0.003, "optonline.net": 0.002, "charter.net": 0.002, "other.com": 0.018}),
        "R_emaildomain": _cat({"__MISSING__": 0.768, "gmail.com": 0.097, "hotmail.com": 0.046, "anonymous.com": 0.034, "yahoo.com": 0.019, "aol.com": 0.006, "outlook.com": 0.004, "comcast.net": 0.003, "icloud.com": 0.003, "other.com": 0.020}),
        "DeviceType": _cat({"__MISSING__": 0.762, "desktop": 0.143, "mobile": 0.094}),
        "DeviceInfo": _cat({"__MISSING__": 0.799, "Windows": 0.081, "iOS Device": 0.034, "MacOS": 0.021, "Trident/7.0": 0.013, "rv:11.0": 0.003, "SM-G960U Build/R16NW": 0.002, "SM-G965U Build/R16NW": 0.002, "rv:57.0": 0.002, "SM-G930V Build/NRD90M": 0.001, "Moto G (5) Build/NPPS25.137-93-14": 0.001, "SM-G955U Build/R16NW": 0.001, "other_device": 0.040}),
        "id_30": _cat({"__MISSING__": 0.868, "Windows 10": 0.036, "Windows 7": 0.022, "iOS 11.2.1": 0.006, "iOS 11.1.2": 0.005, "Android 7.0": 0.004, "Mac OS X 10_11_6": 0.004, "iOS 11.3.0": 0.004, "Mac OS X 10_12_6": 0.003, "Mac OS X 10_13_6": 0.003, "Android": 0.003, "iOS 11.2.6": 0.003, "Windows 8.1": 0.003, "iOS 12.1.0": 0.003, "Linux": 0.002, "other_os": 0.031}),
        "id_31": _cat({"__MISSING__": 0.762, "chrome 63.0": 0.037, "mobile safari 11.0": 0.022, "mobile safari generic": 0.019, "ie 11.0 for desktop": 0.016, "safari generic": 0.013, "chrome 62.0": 0.011, "chrome 65.0": 0.010, "chrome 64.0": 0.010, "chrome 63.0 for android": 0.009, "edge 16.0": 0.007, "firefox 57.0": 0.006, "chrome 66.0": 0.006, "chrome 70.0": 0.005, "chrome 67.0": 0.005, "chrome 69.0": 0.004, "chrome 68.0": 0.004, "chrome 71.0": 0.004, "chrome 64.0 for android": 0.004, "other_browser": 0.046}),
        "id_33": _cat({"__MISSING__": 0.876, "1920x1080": 0.036, "1366x768": 0.019, "1334x750": 0.012, "2208x1242": 0.010, "1440x900": 0.007, "1600x900": 0.006, "2560x1440": 0.005, "2048x1536": 0.004, "1280x800": 0.004, "other_res": 0.021}),
        "id_36": _cat({"__MISSING__": 0.762, "F": 0.234, "T": 0.004}),
        "id_37": _cat({"__MISSING__": 0.762, "T": 0.221, "F": 0.017}),
        "id_38": _cat({"__MISSING__": 0.762, "F": 0.125, "T": 0.113}),
    }
    numeric = {
        "TransactionAmt": (68.77, 239.0), "card1": (9678.0, 4901.0), "card2": (361.0, 157.0), "card3": (150.0, 11.3),
        "card5": (226.0, 41.0), "addr1": (299.0, 101.0), "addr2": (87.0, 2.7), "D1": (3.0, 155.0),
    }
    return ReferenceProfiles(categorical=categorical, numeric=numeric, fraud_rate=0.03499)


def generate_with_true_probs(cfg: SyntheticConfig, profiles: ReferenceProfiles):
    """Run the repo generator unchanged, capturing the Bernoulli probability behind each label."""
    captured: Dict[str, np.ndarray] = {}
    original = synth._target_shift

    def _capture(scores: np.ndarray, target_mean: float) -> float:
        shift = original(scores, target_mean)
        captured["p"] = 1.0 / (1.0 + np.exp(-(scores + shift)))
        return shift

    synth._target_shift = _capture
    try:
        tx_df, id_df = generate_synthetic_ieee_data(cfg, profiles)
    finally:
        synth._target_shift = original
    return tx_df, id_df, captured["p"]


def precision_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    order = np.argsort(-s)[:k]
    return float(y[order].mean())


def queue_metrics(y: np.ndarray, s: np.ndarray, amt: np.ndarray | None = None) -> Dict[str, float]:
    m = evaluate_scores(y, s)
    n_pos = int(y.sum())
    for k in (50, 100, 250, 500, 1000):
        m[f"precision@{k}"] = precision_at_k(y, s, k)
    m["precision@n_pos"] = precision_at_k(y, s, n_pos)
    for k in (100, 500):
        order = np.argsort(-s)[:k]
        m[f"recall@{k}"] = float(y[order].sum() / max(1, n_pos))
    if amt is not None:
        order = np.argsort(-s)[:500]
        m["fraud_dollars_captured@500"] = float((amt[order] * y[order]).sum() / max(1e-9, (amt * y).sum()))
    return m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real-transaction-path", default="data/train_transaction.csv")
    p.add_argument("--real-identity-path", default="data/train_identity.csv")
    p.add_argument("--n-transactions", type=int, default=120_000)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--out", default="outputs/diagnostics.json")
    args = p.parse_args()

    import xgboost as xgb

    t0 = time.time()
    if Path(args.real_transaction_path).exists() and Path(args.real_identity_path).exists():
        profiles = load_reference_profiles(args.real_transaction_path, args.real_identity_path)
        profile_source = "real_ieee_csv"
    else:
        profiles = approximate_ieee_profiles()
        profile_source = "approximated_ieee_marginals"

    cfg = SyntheticConfig(n_transactions=args.n_transactions, fraud_rate=profiles.fraud_rate, random_state=args.random_state)
    tx_df, id_df, true_p = generate_with_true_probs(cfg, profiles)
    y_all = tx_df["isFraud"].to_numpy()
    full = tx_df.merge(id_df, on="TransactionID", how="left")
    n_users = max(10_000, cfg.n_transactions // cfg.avg_tx_per_user)

    R: Dict[str, object] = {"profile_source": profile_source, "n_tx": int(len(tx_df)), "n_users_configured": int(n_users)}

    # ---- 1. Data structure ------------------------------------------------------------------
    per_user = full.groupby("uid_clean")["TransactionID"].count()
    R["users_active"] = int(per_user.size)
    R["tx_per_active_user_quantiles_p50_p90_p99_max"] = np.quantile(per_user, [0.5, 0.9, 0.99, 1.0]).tolist()
    R["top1pct_users_share_of_tx"] = float(per_user.sort_values(ascending=False).head(max(1, n_users // 100)).sum() / len(full))
    R["fraud_rate"] = float(y_all.mean())
    R["true_p_quantiles_p0_p50_p90_p97_p99_max"] = np.quantile(true_p, [0, 0.5, 0.9, 0.97, 0.99, 1.0]).tolist()
    R["true_p_distinct_values"] = int(len(np.unique(true_p.round(6))))
    R["corr_true_p_label"] = float(np.corrcoef(true_p, y_all)[0, 1])

    amt = full["TransactionAmt"].to_numpy()
    comp = {
        "high_amount": amt > np.quantile(amt, 0.97),
        "shared_device(C3>q90)": full["C3"].to_numpy() > np.quantile(full["C3"], 0.90),
        "risky_product(C|W)": full["ProductCD"].isin(["C", "W"]).to_numpy(),
        "email_mismatch": (full["P_emaildomain"].astype(str) != full["R_emaildomain"].astype(str)).to_numpy(),
        "c2_high(C2>q92)": full["C2"].to_numpy() > np.quantile(full["C2"], 0.92),
    }
    R["label_components"] = {
        k: {"rate": float(v.mean()), "fraud_rate_when_true": float(y_all[v].mean()) if v.any() else None, "fraud_rate_when_false": float(y_all[~v].mean())}
        for k, v in comp.items()
    }

    rel: Dict[str, object] = {}
    for col in DEFAULT_RELATION_COLUMNS:
        v = full[col].fillna("__MISSING__").astype(str)
        vc = v.value_counts()
        uids_per_val = full.groupby(v)["uid_clean"].nunique()
        rel[col] = {
            "n_distinct": int(vc.size), "max_degree": int(vc.iloc[0]), "top_value": str(vc.index[0]),
            "missing_degree": int(vc.get("__MISSING__", 0)),
            "tx_share_in_values_with_degree_gt_700": float(vc[vc > 700].sum() / len(full)),
            "share_of_values_shared_by_gt1_uid": float((uids_per_val > 1).mean()),
            "median_uids_per_value": float(uids_per_val.median()),
        }
    R["relation_columns"] = rel
    surv = sum((full[c].fillna("__MISSING__").astype(str).map(full[c].fillna("__MISSING__").astype(str).value_counts()) <= 700).astype(int) for c in DEFAULT_RELATION_COLUMNS)
    R["entities_per_tx_surviving_degree_filter"] = {str(k): int(v) for k, v in sorted(Counter(surv.tolist()).items())}
    gaps = full.sort_values(["uid_clean", "TransactionDT"]).groupby("uid_clean")["TransactionDT"].diff().dropna() / 3600.0
    R["intra_user_gap_hours_p1_p10_p50_p90"] = np.quantile(gaps, [0.01, 0.1, 0.5, 0.9]).tolist()
    R["distinct_days_per_uid_p50_p90_p99"] = np.quantile(full.groupby("uid_clean")["TransactionDT"].apply(lambda s: (s // 86400).nunique()), [0.5, 0.9, 0.99]).tolist()

    # ---- 2. Pipeline replication (mirrors backend_builder.build_backend_artifacts) -------------
    train_df, valid_df, test_df = _time_split(full, 0.70, 0.15)
    bundle = fit_uid_aggregates(train_df)
    train_df, valid_df, test_df = (apply_uid_aggregates(d, bundle) for d in (train_df, valid_df, test_df))
    full_df = pd.concat([train_df.assign(split="train"), valid_df.assign(split="valid"), test_df.assign(split="test")], ignore_index=True)
    full_df = full_df.sort_values("TransactionDT").reset_index(drop=True)
    full_df["true_p"] = pd.Series(true_p, index=tx_df["TransactionID"].to_numpy()).loc[full_df["TransactionID"]].to_numpy()
    train_mask = full_df["split"] == "train"
    R["share_of_test_uids_seen_in_train"] = float(full_df.loc[full_df["split"] == "test", "uid_seen_in_train"].mean())

    embedder = SparseGraphEmbedder(DEFAULT_RELATION_COLUMNS, 32, 2, args.random_state)
    embedder.fit(full_df.loc[train_mask])
    all_emb = embedder.transform(full_df)
    full_df = pd.concat([full_df, all_emb], axis=1)
    graph_cols = [c for c in full_df.columns if c.startswith("graph_emb_")]
    R["svd"] = {
        "input_one_hot_width": int(sum(m.width for m in embedder.mappings)),
        "width_per_relation": {m.column: m.width for m in embedder.mappings},
        "explained_variance_32_dims": float(embedder.svd.explained_variance_ratio_.sum()),
    }

    base_feature_cols = [
        "TransactionDT", "TransactionAmt", "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2",
        "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo", "id_30", "id_31", "id_33", "id_36", "id_37", "id_38",
        *[f"C{i}" for i in range(1, 15)], *[f"D{i}" for i in range(1, 16)], *[f"M{i}" for i in range(1, 10)],
        "uid_tx_count", "uid_amt_mean", "uid_amt_std", "uid_amt_median", "uid_d1_mean", "uid_c1_mean", "uid_seen_in_train",
    ]
    uid_cols = [c for c in base_feature_cols if c.startswith("uid_")]
    categorical_cols = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo", "id_30", "id_31", "id_33", "id_36", "id_37", "id_38", *[f"M{i}" for i in range(1, 10)]]
    feature_cols = base_feature_cols + graph_cols
    X_df, _ = _encode_categorical(full_df[feature_cols].copy(), full_df.index[train_mask], categorical_cols)
    X = X_df.to_numpy(dtype=np.float32)
    y = full_df["isFraud"].to_numpy(dtype=np.int8)
    tr = np.where(full_df["split"] == "train")[0]
    va = np.where(full_df["split"] == "valid")[0]
    te = np.where(full_df["split"] == "test")[0]
    amt_te = full_df["TransactionAmt"].to_numpy()[te]
    R["splits"] = {"train": len(tr), "valid": len(va), "test": len(te), "test_fraud": int(y[te].sum())}

    col_index = {c: i for i, c in enumerate(feature_cols)}

    def train(cols: List[int], objective: str):
        params = {"booster": "gbtree", "max_depth": 8, "eta": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "lambda": 2.0, "alpha": 0.0, "objective": "binary:logistic", "eval_metric": "aucpr", "seed": args.random_state}
        dtr = xgb.DMatrix(X[tr][:, cols], label=y[tr])
        dva = xgb.DMatrix(X[va][:, cols], label=y[va])
        kw = {"obj": asymmetric_focal_objective(1.0, 4.0)} if objective == "focal" else {}
        bst = xgb.train(params, dtr, num_boost_round=600, evals=[(dtr, "train"), (dva, "valid")], early_stopping_rounds=50, verbose_eval=False, **kw)
        pred = bst.predict(xgb.DMatrix(X[:, cols]), iteration_range=(0, bst.best_iteration + 1))
        return bst, pred

    variants = {
        "A_repo_focal_all_features": ("focal", [col_index[c] for c in feature_cols]),
        "B_logistic_all_features": ("logistic", [col_index[c] for c in feature_cols]),
        "C_logistic_no_graph_emb": ("logistic", [col_index[c] for c in feature_cols if c not in graph_cols]),
        "D_logistic_no_graph_no_uid_agg": ("logistic", [col_index[c] for c in feature_cols if c not in graph_cols and c not in uid_cols]),
        "E_logistic_graph_emb_only": ("logistic", [col_index[c] for c in graph_cols]),
    }
    models: Dict[str, object] = {}
    preds: Dict[str, np.ndarray] = {}
    for name, (objective, cols) in variants.items():
        bst, pred = train(cols, objective)
        preds[name] = pred
        blend = uid_label_propagation_blend(full_df, pred, blend_alpha=0.6)
        entry = {
            "n_features": len(cols), "best_iteration": int(bst.best_iteration),
            "test_raw": queue_metrics(y[te], pred[te], amt_te),
            "test_uid_blended": queue_metrics(y[te], blend[te], amt_te),
            "pred_test_min_p50_p99_max": [float(pred[te].min()), float(np.median(pred[te])), float(np.quantile(pred[te], 0.99)), float(pred[te].max())],
            "pred_test_mean_pos_vs_neg": [float(pred[te][y[te] == 1].mean()), float(pred[te][y[te] == 0].mean())],
        }
        if name.startswith(("A_", "B_")):
            imp = {feature_cols[int(k[1:])]: v for k, v in bst.get_score(importance_type="gain").items()}
            entry["top_gain_features"] = sorted(imp.items(), key=lambda kv: -kv[1])[:12]
            entry["graph_emb_share_of_gain"] = float(sum(v for k, v in imp.items() if k.startswith("graph_emb")) / sum(imp.values()))
        models[name] = entry
        print(f"{name:34s} PR-AUC raw {entry['test_raw']['pr_auc']:.4f} blended {entry['test_uid_blended']['pr_auc']:.4f} | ROC {entry['test_raw']['roc_auc']:.4f} | P@100 {entry['test_raw']['precision@100']:.2f} | pred range [{entry['pred_test_min_p50_p99_max'][0]:.3f}, {entry['pred_test_min_p50_p99_max'][3]:.3f}]", flush=True)
    models["ORACLE_generator_true_p"] = {"test_raw": queue_metrics(y[te], full_df["true_p"].to_numpy()[te], amt_te)}
    o = models["ORACLE_generator_true_p"]["test_raw"]
    print(f"{'ORACLE_generator_true_p':34s} PR-AUC {o['pr_auc']:.4f} | ROC {o['roc_auc']:.4f} | P@100 {o['precision@100']:.2f}")
    R["models"] = models

    # ---- 3. Identity-noise sensitivity of the UID blend ------------------------------------------
    rng = np.random.default_rng(0)
    base_pred = preds["B_logistic_all_features"]
    uids0 = full_df["uid_clean"].astype(str).to_numpy()
    uniq = np.unique(uids0)

    def noisy(rate: float, mode: str) -> float:
        uids = uids0.copy()
        victims = rng.choice(uniq, size=int(rate * len(uniq)), replace=False)
        if mode == "merge":
            targets = rng.choice(uniq, size=len(victims), replace=True)
            m = dict(zip(victims, targets))
            uids = np.array([m.get(u, u) for u in uids], dtype=object)
        else:
            vs = set(victims.tolist())
            flip = rng.random(len(uids)) < 0.5
            uids = np.array([u + "_b" if (u in vs and f) else u for u, f in zip(uids, flip)], dtype=object)
        tmp = pd.DataFrame({"uid_clean": uids})
        b = uid_label_propagation_blend(tmp, base_pred, blend_alpha=0.6)
        return float(evaluate_scores(y[te], b[te])["pr_auc"])

    R["uid_blend_pr_auc_under_id_noise"] = {"clean": models["B_logistic_all_features"]["test_uid_blended"]["pr_auc"], **{f"{mode}_{r:.2f}": noisy(r, mode) for mode in ("merge", "split") for r in (0.05, 0.15, 0.30)}}

    # ---- 4. Graph store behaviour with the deployed score (focal + blend) -------------------------
    full_df["pred_raw"] = preds["A_repo_focal_all_features"]
    full_df["pred_score"] = uid_label_propagation_blend(full_df, full_df["pred_raw"].to_numpy(), blend_alpha=0.6)
    pca = PCA(2, random_state=args.random_state)
    e2 = pca.fit_transform(full_df[graph_cols].to_numpy(np.float32))
    full_df["embed_x"], full_df["embed_y"] = e2[:, 0], e2[:, 1]
    R["embedding_2d"] = {"pca_explained": pca.explained_variance_ratio_.round(3).tolist()}
    for col in ("DeviceInfo", "id_31", "id_30", "R_emaildomain"):
        miss = (full_df[col].fillna("__MISSING__").astype(str) == "__MISSING__").astype(float)
        R["embedding_2d"][f"corr_with_missing_{col}"] = [float(np.corrcoef(full_df["embed_x"], miss)[0, 1]), float(np.corrcoef(full_df["embed_y"], miss)[0, 1])]
    s = full_df["pred_score"].to_numpy()
    R["deployed_score_share_ge_0.65_(ui_high_risk)"] = float((s >= 0.65).mean())
    R["deployed_score_share_ge_0.45_(ui_amber)"] = float((s >= 0.45).mean())

    store = FraudGraphStore(full_df)
    ov = store.overview_graph(max_transactions=90, hops=2, max_nodes=1400, max_edges=7000)
    ents = [n for n in ov["nodes"] if n["node_type"] == "entity"]
    txs = [n for n in ov["nodes"] if n["node_type"] == "transaction"]
    R["overview_graph"] = {"seed_nodes": ov["meta"]["seed_nodes"], "n_transactions": len(txs), "entities": [(e["label"], e["degree"]) for e in ents], "fraud_transactions": int(sum(t["isFraud"] for t in txs)), "risk_range": [min(t["risk_score"] for t in txs), max(t["risk_score"] for t in txs)]}
    R["stars_top10"] = [(x["entity_type"], x["entity_value"], x["degree"], round(x["risk_mean"], 3)) for x in store.star_patterns(top_k=10, min_degree=4)]
    ent_rows = np.array([(st["risk_mean"], float(y[store.entity_to_tx_indices[e]].mean()), st["degree"]) for e, st in store.entity_stats.items() if st["degree"] >= 20])
    R["star_score_inputs"] = {"n_entities_degree_ge_20": int(len(ent_rows)), "risk_mean_range": [float(ent_rows[:, 0].min()), float(ent_rows[:, 0].max())], "degree_range": [float(ent_rows[:, 2].min()), float(ent_rows[:, 2].max())], "corr_risk_mean_vs_true_fraud_rate": float(np.corrcoef(ent_rows[:, 0], ent_rows[:, 1])[0, 1])}
    uid_of = dict(zip(full_df["TransactionID"].to_numpy(), full_df["uid_clean"].astype(str).to_numpy()))
    rings = store.ring_patterns(top_k=200, max_seed_transactions=2500)
    R["rings"] = {"top10_shared_relation_counts": [r["shared_relation_count"] for r in rings[:10]], "top10_same_uid_share": float(np.mean([uid_of[r["tx_a"]] == uid_of[r["tx_b"]] for r in rings[:10]])), "top200_same_uid_share": float(np.mean([uid_of[r["tx_a"]] == uid_of[r["tx_b"]] for r in rings]))}
    R["elapsed_seconds"] = round(time.time() - t0, 1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=2, default=float), encoding="utf-8")
    print(f"wrote {out} in {R['elapsed_seconds']}s (profiles: {profile_source})")


if __name__ == "__main__":
    main()
