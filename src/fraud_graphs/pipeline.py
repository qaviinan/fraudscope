from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .features import apply_uid_aggregates, fit_uid_aggregates, uid_label_propagation_blend
from .graph_embeddings import DEFAULT_RELATION_COLUMNS, SparseGraphEmbedder
from .modeling import evaluate_scores, train_model
from .synthetic import (
    SyntheticConfig,
    generate_synthetic_ieee_data,
    load_reference_profiles,
    save_synthetic_data,
)


@dataclass
class PipelineConfig:
    real_transaction_path: str = "data/train_transaction.csv"
    real_identity_path: str = "data/train_identity.csv"
    output_dir: str = "outputs"
    n_transactions: int = 250_000
    sample_rows_for_profile: int = 150_000
    random_state: int = 42
    train_ratio: float = 0.70
    valid_ratio: float = 0.15
    graph_embedding_dim: int = 32
    uid_blend_alpha: float = 0.60


def _encode_categorical_inplace(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    for col in columns:
        train_vals = train_df[col].fillna("__MISSING__").astype(str)
        mapping = {v: i + 1 for i, v in enumerate(train_vals.value_counts().index.tolist())}
        train_df[col] = train_vals.map(mapping).fillna(0).astype(np.float32)
        valid_df[col] = valid_df[col].fillna("__MISSING__").astype(str).map(mapping).fillna(0).astype(np.float32)
        test_df[col] = test_df[col].fillna("__MISSING__").astype(str).map(mapping).fillna(0).astype(np.float32)


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


def run_pipeline(cfg: PipelineConfig) -> Dict[str, object]:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    synth_tx_path, synth_id_path = save_synthetic_data(tx_df, id_df, out_dir / "synthetic_data")

    full_df = tx_df.merge(id_df, on="TransactionID", how="left")
    train_df, valid_df, test_df = _time_split(full_df, cfg.train_ratio, cfg.valid_ratio)

    uid_bundle = fit_uid_aggregates(train_df, uid_col="uid_clean")
    train_df = apply_uid_aggregates(train_df, uid_bundle, uid_col="uid_clean")
    valid_df = apply_uid_aggregates(valid_df, uid_bundle, uid_col="uid_clean")
    test_df = apply_uid_aggregates(test_df, uid_bundle, uid_col="uid_clean")

    embedder = SparseGraphEmbedder(
        relation_columns=DEFAULT_RELATION_COLUMNS,
        embedding_dim=cfg.graph_embedding_dim,
        min_frequency=2,
        random_state=cfg.random_state,
    )
    train_emb = embedder.fit_transform(train_df)
    valid_emb = embedder.transform(valid_df)
    test_emb = embedder.transform(test_df)

    train_df = pd.concat([train_df.reset_index(drop=True), train_emb.reset_index(drop=True)], axis=1)
    valid_df = pd.concat([valid_df.reset_index(drop=True), valid_emb.reset_index(drop=True)], axis=1)
    test_df = pd.concat([test_df.reset_index(drop=True), test_emb.reset_index(drop=True)], axis=1)

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
    graph_cols = [c for c in train_df.columns if c.startswith("graph_emb_")]
    feature_cols = [c for c in base_feature_cols + graph_cols if c in train_df.columns]

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

    X_train = train_df[feature_cols].copy()
    X_valid = valid_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    _encode_categorical_inplace(X_train, X_valid, X_test, categorical_cols)

    y_train = train_df["isFraud"].to_numpy(dtype=np.int8)
    y_valid = valid_df["isFraud"].to_numpy(dtype=np.int8)
    y_test = test_df["isFraud"].to_numpy(dtype=np.int8)

    model_res = train_model(
        X_train.to_numpy(dtype=np.float32),
        y_train,
        X_valid.to_numpy(dtype=np.float32),
        y_valid,
        X_test.to_numpy(dtype=np.float32),
        random_state=cfg.random_state,
    )

    valid_raw = model_res.valid_pred
    test_raw = model_res.test_pred
    valid_blend = uid_label_propagation_blend(valid_df, valid_raw, blend_alpha=cfg.uid_blend_alpha)
    test_blend = uid_label_propagation_blend(test_df, test_raw, blend_alpha=cfg.uid_blend_alpha)

    metrics = {
        "validation_raw": evaluate_scores(y_valid, valid_raw),
        "validation_uid_blended": evaluate_scores(y_valid, valid_blend),
        "test_raw": evaluate_scores(y_test, test_raw),
        "test_uid_blended": evaluate_scores(y_test, test_blend),
        "model_backend": model_res.backend,
        "n_transactions": int(cfg.n_transactions),
        "fraud_rate_synthetic": float(tx_df["isFraud"].mean()),
        "synthetic_transaction_path": str(synth_tx_path),
        "synthetic_identity_path": str(synth_id_path),
    }

    pred_df = pd.DataFrame(
        {
            "TransactionID": test_df["TransactionID"].to_numpy(),
            "uid_clean": test_df["uid_clean"].astype(str).to_numpy(),
            "isFraud": y_test,
            "pred_raw": test_raw,
            "pred_uid_blended": test_blend,
        }
    )
    pred_path = out_dir / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "metrics": metrics,
        "metrics_path": str(metrics_path),
        "prediction_path": str(pred_path),
    }

