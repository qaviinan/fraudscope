"""Track A: card fraud on IEEE-CIS. Builds the model ladder for one out-of-time fold.

Ladder (each rung adds to the previous one):
  rules           a handful of issuer-style velocity and mismatch rules
  gbdt_fields     XGBoost on raw transaction and identity fields
  gbdt_velocity   + point-in-time velocity on the account (card_uid)
  gbdt_graph      + shared-entity graph features (devices, addresses, email domains, two-hop)
All rungs get the same tuning budget, so the graph rung has to beat a baseline that was tried as
hard as it was.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from .data import ieee_cis
from .evaluation import Fold
from .graph_features import build_causal_graph_features

GBDT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "max_depth": 8,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "min_child_weight": 5,
}


@dataclass
class FoldResult:
    scores: Dict[str, np.ndarray]
    test: pd.DataFrame
    extra: Dict[str, np.ndarray] = field(default_factory=dict)


def account_velocity(df: pd.DataFrame, known: np.ndarray, label_lag_days: float) -> pd.DataFrame:
    f = build_causal_graph_features(
        df, entity_columns=[ieee_cis.UID_COLUMN], uid_col=ieee_cis.UID_COLUMN,
        label_known_mask=known, label_lag_days=label_lag_days, two_hop=[],
    )
    return f.loc[:, [c for c in f.columns if "other_accounts" not in c]]


def graph_features(df: pd.DataFrame, known: np.ndarray, label_lag_days: float, max_via_degree: int = 50) -> pd.DataFrame:
    return build_causal_graph_features(
        df, entity_columns=ieee_cis.ENTITY_COLUMNS, uid_col=ieee_cis.UID_COLUMN,
        label_known_mask=known, label_lag_days=label_lag_days,
        two_hop=ieee_cis.TWO_HOP, max_via_degree=max_via_degree,
    )


def new_pair_flag(df: pd.DataFrame, a: str, b: str) -> np.ndarray:
    """1 where this (a, b) pair appears for the first time; rows must be in time order."""
    seen: set = set()
    out = np.zeros(len(df), dtype=np.float32)
    for i, (x, y) in enumerate(zip(df[a].to_numpy(dtype=object), df[b].to_numpy(dtype=object))):
        if y is None or (isinstance(y, float) and np.isnan(y)):
            continue
        if (x, y) not in seen:
            out[i] = 1.0
            seen.add((x, y))
    return out


def shuffle_links(df: pd.DataFrame, columns, seed: int = 0) -> pd.DataFrame:
    """Permute entity values among transactions on the same day. Daily counts survive, who-shares-
    with-whom does not. If graph lift is real it should mostly vanish on this copy."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for col in columns:
        vals = out[col].to_numpy(dtype=object).copy()
        for idx in out.groupby("day").indices.values():
            vals[idx] = vals[rng.permutation(idx)]
        out[col] = vals
    return out


def rules_score(df: pd.DataFrame, vel: pd.DataFrame, new_device: np.ndarray, amount_p99: float) -> np.ndarray:
    amt = df["TransactionAmt"].to_numpy()
    p_dom, r_dom = df["P_emaildomain"].to_numpy(dtype=object), df["R_emaildomain"].to_numpy(dtype=object)
    domain_mismatch = np.array([isinstance(p, str) and isinstance(r, str) and p != r for p, r in zip(p_dom, r_dom)])
    fired = (
        (vel["g_card_uid_prior_tx_1h"].to_numpy() >= 3).astype(int)
        + (amt > amount_p99).astype(int)
        + (new_device > 0).astype(int)
        + domain_mismatch.astype(int)
        + ((vel["g_card_uid_hours_since_first_seen"].to_numpy() < 24) & (amt > amount_p99 / 3)).astype(int)
    )
    # ties broken by amount, so the score still ranks within a rule count
    return fired + 1e-4 * np.argsort(np.argsort(amt)) / len(amt)


def _encode(train: pd.DataFrame, other: pd.DataFrame, cols) -> tuple[np.ndarray, np.ndarray]:
    a, b = train[cols].copy(), other[cols].copy()
    for c in cols:
        if not (pd.api.types.is_numeric_dtype(a[c]) or pd.api.types.is_bool_dtype(a[c])):
            cats = pd.Index(a[c].dropna().unique())
            # categories first seen in the test window become missing
            a[c] = np.where(a[c].isna(), np.nan, cats.get_indexer(a[c])).astype(np.float32)
            codes = cats.get_indexer(b[c]).astype(np.float32)
            b[c] = np.where(codes < 0, np.nan, codes)
    return a.to_numpy(dtype=np.float32), b.to_numpy(dtype=np.float32)


def fit_predict(X: np.ndarray, y: np.ndarray, t: np.ndarray, X_test: np.ndarray, seed: int, max_rounds: int = 2000) -> np.ndarray:
    import xgboost as xgb

    # early stopping on the latest 15% of training time, never on the test window
    cut = np.quantile(t, 0.85)
    tr, va = t < cut, t >= cut
    dtr, dva = xgb.DMatrix(X[tr], label=y[tr]), xgb.DMatrix(X[va], label=y[va])
    booster = xgb.train({**GBDT_PARAMS, "seed": seed}, dtr, max_rounds, evals=[(dva, "valid")],
                        early_stopping_rounds=100, verbose_eval=False)
    return booster.predict(xgb.DMatrix(X_test), iteration_range=(0, booster.best_iteration + 1))


def run_fold(
    df: pd.DataFrame, fold: Fold, label_lag_days: float, seed: int = 0, edge_shuffle: bool = False
) -> FoldResult:
    end = df["TransactionDT"].to_numpy()[fold.test_idx].max()
    sub = df[df["TransactionDT"] <= end].reset_index(drop=True)
    t = sub["TransactionDT"].to_numpy()
    known = t < fold.train_end
    train_m = known
    test_m = t >= fold.test_start

    vel = account_velocity(sub, known, label_lag_days)
    graph = graph_features(sub, known, label_lag_days)
    new_device = new_pair_flag(sub, ieee_cis.UID_COLUMN, "device_fp")
    y = sub["isFraud"].to_numpy()

    raw_cols = ieee_cis.model_columns(sub)
    raw_tr, raw_te = _encode(sub[train_m], sub[test_m], raw_cols)
    vel_tr, vel_te = vel[train_m].to_numpy(np.float32), vel[test_m].to_numpy(np.float32)
    g_tr, g_te = graph[train_m].to_numpy(np.float32), graph[test_m].to_numpy(np.float32)
    tt = t[train_m]

    amount_p99 = float(np.quantile(sub.loc[train_m, "TransactionAmt"], 0.99))
    scores = {
        "rules": rules_score(sub[test_m], vel[test_m], new_device[test_m], amount_p99),
        "gbdt_fields": fit_predict(raw_tr, y[train_m], tt, raw_te, seed),
        "gbdt_velocity": fit_predict(np.hstack([raw_tr, vel_tr]), y[train_m], tt, np.hstack([raw_te, vel_te]), seed),
        "gbdt_graph": fit_predict(np.hstack([raw_tr, vel_tr, g_tr]), y[train_m], tt, np.hstack([raw_te, vel_te, g_te]), seed),
    }
    if edge_shuffle:
        shuffled = shuffle_links(sub, ("device_fp", "addr1", "P_emaildomain"), seed)
        gs = graph_features(shuffled, known, label_lag_days)
        gs_tr, gs_te = gs[train_m].to_numpy(np.float32), gs[test_m].to_numpy(np.float32)
        scores["gbdt_graph_shuffled"] = fit_predict(
            np.hstack([raw_tr, vel_tr, gs_tr]), y[train_m], tt, np.hstack([raw_te, vel_te, gs_te]), seed
        )
    test = sub.loc[test_m, ["TransactionID", "TransactionDT", "day", "TransactionAmt", "isFraud", ieee_cis.UID_COLUMN]].reset_index(drop=True)
    test["first_seen_account"] = (vel.loc[test_m, "g_card_uid_prior_tx_all"].to_numpy() == 0)
    return FoldResult(scores=scores, test=test)
