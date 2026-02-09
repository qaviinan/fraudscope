from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


TRANSACTION_COLUMNS = [
    "TransactionID",
    "isFraud",
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
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
    *[f"C{i}" for i in range(1, 15)],
    *[f"D{i}" for i in range(1, 16)],
    *[f"M{i}" for i in range(1, 10)],
    "uid_clean",
]

IDENTITY_COLUMNS = [
    "TransactionID",
    "DeviceType",
    "DeviceInfo",
    *[f"id_{i:02d}" for i in range(1, 39)],
]

PROFILE_CATEGORICAL_COLUMNS = [
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
]

PROFILE_NUMERIC_COLUMNS = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "D1",
]


@dataclass
class ReferenceProfiles:
    categorical: Dict[str, Tuple[np.ndarray, np.ndarray]]
    numeric: Dict[str, Tuple[float, float]]
    fraud_rate: float


@dataclass
class SyntheticConfig:
    n_transactions: int = 250_000
    avg_tx_per_user: int = 10
    horizon_days: int = 180
    fraud_rate: float = 0.035
    random_state: int = 42


def _sample_categorical(
    rng: np.random.Generator,
    values: np.ndarray,
    probs: np.ndarray,
    size: int,
) -> np.ndarray:
    if len(values) == 0:
        return np.full(size, "__UNKNOWN__", dtype=object)
    return rng.choice(values, size=size, p=probs)


def _categorical_distribution(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    cleaned = series.fillna("__MISSING__").astype(str)
    vc = cleaned.value_counts(normalize=True)
    return vc.index.to_numpy(dtype=object), vc.values.astype(np.float64)


def _numeric_stats(series: pd.Series) -> Tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce")
    mean = float(np.nanmedian(s))
    std = float(np.nanstd(s))
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return mean, std


def load_reference_profiles(
    train_transaction_path: str | Path,
    train_identity_path: str | Path,
    sample_rows: int = 150_000,
) -> ReferenceProfiles:
    tx_cols = set(PROFILE_CATEGORICAL_COLUMNS + PROFILE_NUMERIC_COLUMNS + ["isFraud", "TransactionID"])
    id_cols = set(["TransactionID", "DeviceType", "DeviceInfo", "id_30", "id_31", "id_33", "id_36", "id_37", "id_38"])

    tx_head = pd.read_csv(train_transaction_path, nrows=2)
    id_head = pd.read_csv(train_identity_path, nrows=2)

    tx_use = [c for c in tx_head.columns if c in tx_cols]
    id_use = [c for c in id_head.columns if c in id_cols]

    tx = pd.read_csv(train_transaction_path, usecols=tx_use, nrows=sample_rows)
    identity = pd.read_csv(train_identity_path, usecols=id_use, nrows=sample_rows)
    merged = tx.merge(identity, on="TransactionID", how="left")

    categorical: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for col in PROFILE_CATEGORICAL_COLUMNS:
        if col in merged.columns:
            categorical[col] = _categorical_distribution(merged[col])
        else:
            categorical[col] = (np.array(["__MISSING__"], dtype=object), np.array([1.0], dtype=np.float64))

    numeric: Dict[str, Tuple[float, float]] = {}
    for col in PROFILE_NUMERIC_COLUMNS:
        if col in merged.columns:
            numeric[col] = _numeric_stats(merged[col])
        else:
            numeric[col] = (0.0, 1.0)

    fraud_rate = float(np.nanmean(tx["isFraud"])) if "isFraud" in tx else 0.035
    if not np.isfinite(fraud_rate) or fraud_rate <= 0 or fraud_rate >= 1:
        fraud_rate = 0.035

    return ReferenceProfiles(categorical=categorical, numeric=numeric, fraud_rate=fraud_rate)


def _target_shift(scores: np.ndarray, target_mean: float) -> float:
    lo, hi = -12.0, 12.0
    for _ in range(50):
        mid = (lo + hi) * 0.5
        probs = 1.0 / (1.0 + np.exp(-(scores + mid)))
        m = float(probs.mean())
        if m > target_mean:
            hi = mid
        else:
            lo = mid
    return (lo + hi) * 0.5


def _factorized_counts(values: np.ndarray) -> np.ndarray:
    codes, _ = pd.factorize(values.astype(str), sort=False)
    return np.bincount(codes)[codes]


def generate_synthetic_ieee_data(
    cfg: SyntheticConfig,
    profiles: ReferenceProfiles,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.random_state)
    n_tx = cfg.n_transactions
    n_users = max(10_000, n_tx // max(cfg.avg_tx_per_user, 1))

    amount_mu, amount_std = profiles.numeric["TransactionAmt"]
    amount_mu = np.log1p(max(amount_mu, 1.0))
    amount_std = max(0.35, min(1.4, abs(np.log1p(abs(amount_std)))))

    user_activity = rng.pareto(1.4, size=n_users) + 0.05
    user_probs = user_activity / user_activity.sum()
    user_idx = rng.choice(n_users, size=n_tx, p=user_probs)

    day = rng.integers(0, cfg.horizon_days, size=n_tx)
    sec = rng.integers(0, 24 * 60 * 60, size=n_tx)
    transaction_dt = day * 86_400 + sec
    order = np.argsort(transaction_dt, kind="mergesort")

    user_idx = user_idx[order]
    day = day[order]
    sec = sec[order]
    transaction_dt = transaction_dt[order]

    def user_cat(col: str) -> np.ndarray:
        values, probs = profiles.categorical[col]
        return _sample_categorical(rng, values, probs, size=n_users)

    user_product = user_cat("ProductCD")
    user_card4 = user_cat("card4")
    user_card6 = user_cat("card6")
    user_p_email = user_cat("P_emaildomain")
    user_r_email = user_cat("R_emaildomain")
    user_device_type = user_cat("DeviceType")
    user_device_info = user_cat("DeviceInfo")
    user_id30 = user_cat("id_30")
    user_id31 = user_cat("id_31")

    c1_mean, c1_std = profiles.numeric["card1"]
    c2_mean, c2_std = profiles.numeric["card2"]
    c3_mean, c3_std = profiles.numeric["card3"]
    c5_mean, c5_std = profiles.numeric["card5"]
    a1_mean, a1_std = profiles.numeric["addr1"]
    a2_mean, a2_std = profiles.numeric["addr2"]

    user_card1 = np.maximum(1000, np.round(rng.normal(c1_mean, max(c1_std, 200), size=n_users))).astype(int)
    user_card2 = np.maximum(100, np.round(rng.normal(c2_mean, max(c2_std, 20), size=n_users))).astype(int)
    user_card3 = np.maximum(100, np.round(rng.normal(c3_mean, max(c3_std, 10), size=n_users))).astype(int)
    user_card5 = np.maximum(100, np.round(rng.normal(c5_mean, max(c5_std, 20), size=n_users))).astype(int)
    user_addr1 = np.maximum(1, np.round(rng.normal(a1_mean, max(a1_std, 20), size=n_users))).astype(int)
    user_addr2 = np.maximum(1, np.round(rng.normal(a2_mean, max(a2_std, 5), size=n_users))).astype(int)

    user_birth_day = rng.integers(-1300, -20, size=n_users)
    user_amt_mu = rng.normal(amount_mu, amount_std * 0.45, size=n_users)
    user_amt_sigma = np.clip(rng.normal(0.55, 0.20, size=n_users), 0.15, 1.2)
    user_risk = rng.normal(0.0, 1.0, size=n_users)

    tx_product = user_product[user_idx]
    tx_card4 = user_card4[user_idx]
    tx_card6 = user_card6[user_idx]
    tx_p_email = user_p_email[user_idx]
    tx_r_email = user_r_email[user_idx]
    tx_device_type = user_device_type[user_idx]
    tx_device_info = user_device_info[user_idx]
    tx_id30 = user_id30[user_idx]
    tx_id31 = user_id31[user_idx]

    tx_card1 = user_card1[user_idx]
    tx_card2 = user_card2[user_idx]
    tx_card3 = user_card3[user_idx]
    tx_card5 = user_card5[user_idx]
    tx_addr1 = user_addr1[user_idx]
    tx_addr2 = user_addr2[user_idx]

    spoof_mask = rng.random(n_tx) < 0.08
    if spoof_mask.any():
        tx_device_info[spoof_mask] = _sample_categorical(
            rng, *profiles.categorical["DeviceInfo"], size=int(spoof_mask.sum())
        )
        tx_id31[spoof_mask] = _sample_categorical(rng, *profiles.categorical["id_31"], size=int(spoof_mask.sum()))

    amount = np.exp(rng.normal(user_amt_mu[user_idx], user_amt_sigma[user_idx], size=n_tx))
    amount = np.clip(amount, 1.0, 40_000.0)

    d1 = np.maximum(1, day - user_birth_day[user_idx]).astype(float)
    d_cols: Dict[str, np.ndarray] = {"D1": d1}
    for i in range(2, 16):
        lag = rng.integers(0, 15 * i, size=n_tx)
        d_cols[f"D{i}"] = np.maximum(0, d1 - lag).astype(float)

    uid_clean = np.array([f"uid_{u:08d}" for u in user_idx], dtype=object)
    tx_id = np.arange(2_000_000, 2_000_000 + n_tx, dtype=np.int64)

    user_count = np.bincount(user_idx, minlength=n_users)[user_idx]
    card_count = _factorized_counts(tx_card1.astype(str))
    device_count = _factorized_counts(tx_device_info.astype(str))
    email_count = _factorized_counts(tx_p_email.astype(str))
    addr_count = _factorized_counts(tx_addr1.astype(str))

    c_features = {
        "C1": user_count,
        "C2": card_count,
        "C3": device_count,
        "C4": email_count,
        "C5": addr_count,
    }
    for i in range(6, 15):
        c_features[f"C{i}"] = np.clip(
            0.45 * c_features["C1"] + 0.25 * c_features["C2"] + rng.normal(0, 3, size=n_tx),
            0,
            None,
        ).astype(int)

    m_features: Dict[str, np.ndarray] = {}
    for i in range(1, 10):
        p_true = max(0.5, 0.93 - i * 0.04)
        m_features[f"M{i}"] = np.where(rng.random(n_tx) < p_true, "T", "F")

    high_amount = amount > np.quantile(amount, 0.97)
    shared_device = device_count > np.quantile(device_count, 0.90)
    email_mismatch = tx_p_email != tx_r_email
    risky_product = np.isin(tx_product, ["C", "W"])
    fast_device_switch = spoof_mask

    raw_score = (
        1.8 * high_amount.astype(float)
        + 1.2 * shared_device.astype(float)
        + 0.9 * risky_product.astype(float)
        + 0.8 * email_mismatch.astype(float)
        + 0.7 * fast_device_switch.astype(float)
        + 0.5 * (c_features["C2"] > np.quantile(c_features["C2"], 0.92)).astype(float)
        + 0.6 * (user_risk[user_idx] > 1.1).astype(float)
    )
    raw_score = raw_score - raw_score.mean()
    target_rate = float(np.clip(cfg.fraud_rate, 0.005, 0.25))
    shift = _target_shift(raw_score, target_rate)
    probs = 1.0 / (1.0 + np.exp(-(raw_score + shift)))
    is_fraud = rng.binomial(1, probs).astype(np.int8)

    transaction_df = pd.DataFrame(
        {
            "TransactionID": tx_id,
            "isFraud": is_fraud,
            "TransactionDT": transaction_dt.astype(np.int64),
            "TransactionAmt": amount.astype(np.float64),
            "ProductCD": tx_product,
            "card1": tx_card1,
            "card2": tx_card2,
            "card3": tx_card3,
            "card4": tx_card4,
            "card5": tx_card5,
            "card6": tx_card6,
            "addr1": tx_addr1,
            "addr2": tx_addr2,
            "dist1": np.abs(rng.normal(25, 12, size=n_tx)),
            "dist2": np.abs(rng.normal(18, 8, size=n_tx)),
            "P_emaildomain": tx_p_email,
            "R_emaildomain": tx_r_email,
            **c_features,
            **d_cols,
            **m_features,
            "uid_clean": uid_clean,
        }
    )

    id_data: Dict[str, np.ndarray] = {
        "TransactionID": tx_id,
        "DeviceType": tx_device_type,
        "DeviceInfo": tx_device_info,
    }

    for i in range(1, 39):
        col = f"id_{i:02d}"
        if col == "id_30":
            id_data[col] = tx_id30
        elif col == "id_31":
            id_data[col] = tx_id31
        elif col in {"id_33", "id_36", "id_37", "id_38"}:
            values, probs = profiles.categorical.get(col, (np.array(["__MISSING__"], dtype=object), np.array([1.0])))
            id_data[col] = _sample_categorical(rng, values, probs, size=n_tx)
        elif col in {"id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29", "id_35"}:
            id_data[col] = np.where(rng.random(n_tx) < 0.8, "Found", "NotFound")
        else:
            base = rng.normal(0.0, 1.0, size=n_tx)
            id_data[col] = base + 0.6 * user_risk[user_idx]

    identity_df = pd.DataFrame(id_data)

    transaction_df = transaction_df[TRANSACTION_COLUMNS]
    identity_df = identity_df[IDENTITY_COLUMNS]
    return transaction_df, identity_df


def save_synthetic_data(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame,
    output_dir: str | Path,
) -> Tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tx_path = out_dir / "synthetic_train_transaction.csv"
    id_path = out_dir / "synthetic_train_identity.csv"
    transaction_df.to_csv(tx_path, index=False)
    identity_df.to_csv(id_path, index=False)
    return tx_path, id_path

