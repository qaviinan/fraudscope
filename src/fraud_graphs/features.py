from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


UID_DEFAULT_COL = "uid_clean"


@dataclass
class UIDAggregateBundle:
    aggregates: pd.DataFrame
    defaults: Dict[str, float]
    feature_columns: Tuple[str, ...]


def fit_uid_aggregates(train_df: pd.DataFrame, uid_col: str = UID_DEFAULT_COL) -> UIDAggregateBundle:
    group = train_df.groupby(uid_col, dropna=False)
    agg = group.agg(
        uid_tx_count=("TransactionID", "count"),
        uid_amt_mean=("TransactionAmt", "mean"),
        uid_amt_std=("TransactionAmt", "std"),
        uid_amt_median=("TransactionAmt", "median"),
        uid_d1_mean=("D1", "mean"),
        uid_c1_mean=("C1", "mean"),
    )
    agg["uid_amt_std"] = agg["uid_amt_std"].fillna(0.0)
    agg["uid_seen_in_train"] = 1.0

    defaults = {col: float(agg[col].mean()) for col in agg.columns if col != "uid_seen_in_train"}
    defaults["uid_seen_in_train"] = 0.0

    return UIDAggregateBundle(
        aggregates=agg,
        defaults=defaults,
        feature_columns=tuple(agg.columns.tolist()),
    )


def apply_uid_aggregates(
    df: pd.DataFrame,
    bundle: UIDAggregateBundle,
    uid_col: str = UID_DEFAULT_COL,
) -> pd.DataFrame:
    out = df.join(bundle.aggregates, on=uid_col)
    for col in bundle.feature_columns:
        out[col] = out[col].fillna(bundle.defaults[col])
    return out


def uid_label_propagation_blend(
    df: pd.DataFrame,
    preds: np.ndarray,
    uid_col: str = UID_DEFAULT_COL,
    blend_alpha: float = 0.6,
) -> np.ndarray:
    alpha = float(np.clip(blend_alpha, 0.0, 1.0))
    pred_df = pd.DataFrame({uid_col: df[uid_col].astype(str), "pred": preds})
    uid_mean = pred_df.groupby(uid_col)["pred"].transform("mean").to_numpy()
    return (1.0 - alpha) * preds + alpha * uid_mean

