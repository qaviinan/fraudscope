from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression

from .features import apply_uid_aggregates, fit_uid_aggregates, uid_label_propagation_blend
from .graph_embeddings import DEFAULT_RELATION_COLUMNS, SparseGraphEmbedder
from .graph_features import DEFAULT_ENTITY_COLUMNS, build_causal_graph_features
from .modeling import evaluate_scores, predict_scores, train_model
from .synthetic_v2 import DIAGNOSTIC_COLUMNS, SyntheticV2Config, generate_synthetic_v2
from .synthetic import (
    SyntheticConfig,
    generate_synthetic_ieee_data,
    load_reference_profiles,
    save_synthetic_data,
)


@dataclass
class BackendBuildConfig:
    real_transaction_path: str = "data/train_transaction.csv"
    real_identity_path: str = "data/train_identity.csv"
    output_dir: str = "outputs/backend_api"
    n_transactions: int = 250_000
    sample_rows_for_profile: int = 150_000
    random_state: int = 42
    train_ratio: float = 0.70
    valid_ratio: float = 0.15
    graph_embedding_dim: int = 32
    uid_blend_alpha: float = 0.60
    # v2 options (docs/foundational_flow_review.md section 10)
    generator: str = "v1"              # "v1": marginal-profile generator; "v2": actor-level generator with planted rings
    objective: str = "focal"           # "focal" (uncalibrated ranking) or "logistic"
    causal_graph_features: bool = False
    calibrate: bool = False            # isotonic calibration on the validation split before scores reach the UI
    use_svd_embeddings: bool = True


V2_GRAPH_RELATION_COLUMNS = list(DEFAULT_ENTITY_COLUMNS)   # identity-like entities only; email domain / OS / browser are attributes


def _time_split(
    df: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_ratio <= 0 or valid_ratio <= 0 or (train_ratio + valid_ratio) >= 1:
        raise ValueError("train_ratio and valid_ratio must be >0 and sum to <1")
    sorted_df = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(sorted_df)
    tr_end = int(n * train_ratio)
    va_end = int(n * (train_ratio + valid_ratio))
    train_df = sorted_df.iloc[:tr_end].copy()
    valid_df = sorted_df.iloc[tr_end:va_end].copy()
    test_df = sorted_df.iloc[va_end:].copy()
    return train_df, valid_df, test_df


def _encode_categorical(
    full_df: pd.DataFrame,
    train_index: pd.Index,
    columns: Iterable[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    out = full_df.copy()
    mappings: Dict[str, Dict[str, int]] = {}
    for col in columns:
        train_vals = out.loc[train_index, col].fillna("__MISSING__").astype(str)
        mapping = {v: i + 1 for i, v in enumerate(train_vals.value_counts().index.tolist())}
        out[col] = out[col].fillna("__MISSING__").astype(str).map(mapping).fillna(0).astype(np.float32)
        mappings[col] = mapping
    return out, mappings


def build_backend_artifacts(cfg: BackendBuildConfig) -> Dict[str, object]:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if cfg.generator == "v2":
        tx_df, id_df = generate_synthetic_v2(SyntheticV2Config(n_transactions=cfg.n_transactions, random_state=cfg.random_state))
        relation_columns = V2_GRAPH_RELATION_COLUMNS
    else:
        profiles = load_reference_profiles(
            train_transaction_path=cfg.real_transaction_path,
            train_identity_path=cfg.real_identity_path,
            sample_rows=cfg.sample_rows_for_profile,
        )
        synth_cfg = SyntheticConfig(
            n_transactions=cfg.n_transactions,
            fraud_rate=profiles.fraud_rate,
            random_state=cfg.random_state,
        )
        tx_df, id_df = generate_synthetic_ieee_data(synth_cfg, profiles)
        relation_columns = list(DEFAULT_RELATION_COLUMNS)
    synth_tx_path, synth_id_path = save_synthetic_data(tx_df, id_df, out_dir / "synthetic_data")

    full_df = tx_df.merge(id_df, on="TransactionID", how="left")
    train_df, valid_df, test_df = _time_split(full_df, cfg.train_ratio, cfg.valid_ratio)

    uid_bundle = fit_uid_aggregates(train_df, uid_col="uid_clean")
    train_df = apply_uid_aggregates(train_df, uid_bundle, uid_col="uid_clean")
    valid_df = apply_uid_aggregates(valid_df, uid_bundle, uid_col="uid_clean")
    test_df = apply_uid_aggregates(test_df, uid_bundle, uid_col="uid_clean")

    full_df = pd.concat(
        [
            train_df.assign(split="train"),
            valid_df.assign(split="valid"),
            test_df.assign(split="test"),
        ],
        axis=0,
        ignore_index=True,
    )
    full_df = full_df.sort_values("TransactionDT").reset_index(drop=True)

    train_mask = full_df["split"] == "train"
    embedder = SparseGraphEmbedder(
        relation_columns=relation_columns,
        embedding_dim=cfg.graph_embedding_dim,
        min_frequency=2,
        random_state=cfg.random_state,
    )
    embedder.fit(full_df.loc[train_mask])
    all_emb = embedder.transform(full_df)
    full_df = pd.concat([full_df.reset_index(drop=True), all_emb.reset_index(drop=True)], axis=1)

    graph_cols = [c for c in full_df.columns if c.startswith("graph_emb_")] if cfg.use_svd_embeddings else []
    causal_cols: List[str] = []
    if cfg.causal_graph_features:
        causal = build_causal_graph_features(
            full_df, entity_columns=relation_columns, uid_col="uid_clean", label_known_mask=train_mask.to_numpy()
        )
        full_df = pd.concat([full_df, causal], axis=1)
        causal_cols = list(causal.columns)

    base_feature_cols = [
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
        "DeviceInfo",
        "id_30",
        "id_31",
        "id_33",
        "id_36",
        "id_37",
        "id_38",
        *[f"C{i}" for i in range(1, 15)],
        *[f"D{i}" for i in range(1, 16)],
        *[f"M{i}" for i in range(1, 10)],
        "uid_tx_count",
        "uid_amt_mean",
        "uid_amt_std",
        "uid_amt_median",
        "uid_d1_mean",
        "uid_c1_mean",
        "uid_seen_in_train",
    ]
    feature_cols = [c for c in base_feature_cols + causal_cols + graph_cols if c in full_df.columns]
    assert not set(feature_cols) & set(DIAGNOSTIC_COLUMNS), "diagnostic columns must never be model features"

    categorical_cols = [
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
        "DeviceInfo",
        "id_30",
        "id_31",
        "id_33",
        "id_36",
        "id_37",
        "id_38",
        *[f"M{i}" for i in range(1, 10)],
    ]
    categorical_cols = [c for c in categorical_cols if c in feature_cols]

    X_df = full_df[feature_cols].copy()
    X_df, categorical_mappings = _encode_categorical(X_df, train_index=full_df.index[train_mask], columns=categorical_cols)
    X = X_df.to_numpy(dtype=np.float32)
    y = full_df["isFraud"].to_numpy(dtype=np.int8)

    train_idx = np.where(full_df["split"].to_numpy() == "train")[0]
    valid_idx = np.where(full_df["split"].to_numpy() == "valid")[0]
    test_idx = np.where(full_df["split"].to_numpy() == "test")[0]

    model_res = train_model(
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_valid=X[valid_idx],
        y_valid=y[valid_idx],
        X_test=X[test_idx],
        random_state=cfg.random_state,
        objective=cfg.objective,
    )

    pred_raw_full = predict_scores(model_res.model, model_res.backend, X)
    if cfg.calibrate:
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(pred_raw_full[valid_idx], y[valid_idx])
        pred_raw_full = calibrator.predict(pred_raw_full)
    full_df["pred_raw"] = pred_raw_full
    full_df["pred_uid_blended"] = uid_label_propagation_blend(
        full_df, pred_raw_full, uid_col="uid_clean", blend_alpha=cfg.uid_blend_alpha
    )
    full_df["pred_score"] = full_df["pred_uid_blended"]

    valid_raw = full_df.iloc[valid_idx]["pred_raw"].to_numpy()
    valid_blend = full_df.iloc[valid_idx]["pred_uid_blended"].to_numpy()
    test_raw = full_df.iloc[test_idx]["pred_raw"].to_numpy()
    test_blend = full_df.iloc[test_idx]["pred_uid_blended"].to_numpy()

    svd_cols = [c for c in full_df.columns if c.startswith("graph_emb_")]
    pca = PCA(n_components=2, random_state=cfg.random_state)
    emb2d = pca.fit_transform(full_df[svd_cols].to_numpy(dtype=np.float32))
    full_df["embed_x"] = emb2d[:, 0]
    full_df["embed_y"] = emb2d[:, 1]

    transactions_path = artifacts_dir / "transactions_enriched.csv.gz"
    full_df.to_csv(transactions_path, index=False, compression="gzip")

    feature_matrix_path = artifacts_dir / "feature_matrix.npz"
    np.savez_compressed(
        feature_matrix_path,
        transaction_ids=full_df["TransactionID"].to_numpy(dtype=np.int64),
        X=X.astype(np.float32),
        y=y.astype(np.int8),
        split=full_df["split"].astype(str).to_numpy(),
    )

    feature_cols_path = artifacts_dir / "feature_columns.json"
    feature_cols_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

    cat_map_path = artifacts_dir / "categorical_mappings.json"
    cat_map_path.write_text(json.dumps(categorical_mappings), encoding="utf-8")

    model_meta = {"backend": model_res.backend, "graph_relation_columns": relation_columns, "generator": cfg.generator, "calibrated": bool(cfg.calibrate)}
    if model_res.backend.startswith("xgboost"):
        model_path = artifacts_dir / "model_xgboost.json"
        model_res.model.save_model(str(model_path))
        stale_path = artifacts_dir / "model_fallback.pkl"
        if stale_path.exists():
            stale_path.unlink()
    else:
        model_path = artifacts_dir / "model_fallback.pkl"
        with model_path.open("wb") as f:
            pickle.dump(model_res.model, f)
        stale_path = artifacts_dir / "model_xgboost.json"
        if stale_path.exists():
            stale_path.unlink()
    model_meta["model_path"] = str(model_path)

    model_meta_path = artifacts_dir / "model_meta.json"
    model_meta_path.write_text(json.dumps(model_meta, indent=2), encoding="utf-8")

    amt = full_df["TransactionAmt"].to_numpy(dtype=np.float64)
    metrics = {
        "validation_raw": evaluate_scores(y[valid_idx], valid_raw, amt[valid_idx]),
        "validation_uid_blended": evaluate_scores(y[valid_idx], valid_blend, amt[valid_idx]),
        "test_raw": evaluate_scores(y[test_idx], test_raw, amt[test_idx]),
        "test_uid_blended": evaluate_scores(y[test_idx], test_blend, amt[test_idx]),
        "feature_blocks": {"base": len(base_feature_cols), "causal_graph": len(causal_cols), "svd_embedding": len(graph_cols)},
        "model_backend": model_res.backend,
        "n_transactions": int(cfg.n_transactions),
        "fraud_rate_synthetic": float(tx_df["isFraud"].mean()),
        "synthetic_transaction_path": str(synth_tx_path),
        "synthetic_identity_path": str(synth_id_path),
        "artifacts_dir": str(artifacts_dir),
    }

    metadata = {
        "build_config": asdict(cfg),
        "metrics": metrics,
        "paths": {
            "transactions_enriched": str(transactions_path),
            "feature_matrix": str(feature_matrix_path),
            "feature_columns": str(feature_cols_path),
            "categorical_mappings": str(cat_map_path),
            "model_meta": str(model_meta_path),
            "model": str(model_path),
            "synthetic_transaction": str(synth_tx_path),
            "synthetic_identity": str(synth_id_path),
        },
    }
    metadata_path = out_dir / "backend_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "metadata": metadata,
        "metadata_path": str(metadata_path),
        "artifacts_dir": str(artifacts_dir),
    }
