"""IEEE-CIS (Vesta) loader and entity resolution.

Download train_transaction.csv and train_identity.csv from the Kaggle competition
"ieee-fraud-detection" into data/ieee/. Labels come from Vesta: a reported chargeback marks
the card's transaction as fraud, and later transactions tied to the same account, email or
billing address inherit the label. Entity-level features can ride on that propagation, so
results on this data are an upper bound on what the same features would do on raw chargebacks.

Entities (the competition masks real identifiers, so these are reconstructions):
  card_uid   card1 + addr1 + first-seen day (day - D1); stands in for the account
  device_fp  DeviceInfo + OS + browser + screen resolution, only where identity data exists
  addr1      billing region, coarse
  P_emaildomain  purchaser email domain, coarse
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DAY = 86_400
DEVICE_PARTS = ("DeviceInfo", "id_30", "id_31", "id_33")
ENTITY_COLUMNS = ("card1", "addr1", "device_fp", "P_emaildomain")
UID_COLUMN = "card_uid"
# (anchor, via, target, name)
TWO_HOP = (
    ("card_uid", "device_fp", "card_uid", "accounts_via_account_devices"),
    ("device_fp", "card_uid", "device_fp", "devices_via_device_accounts"),
    ("card_uid", "device_fp", "addr1", "addresses_via_account_devices"),
)


def load(data_dir: str | Path = "data/ieee", cache: bool = True) -> pd.DataFrame:
    data_dir = Path(data_dir)
    cached = data_dir / "train_joined.parquet"
    if cache and cached.exists():
        return pd.read_parquet(cached)
    tx = pd.read_csv(data_dir / "train_transaction.csv")
    ident = pd.read_csv(data_dir / "train_identity.csv")
    df = tx.merge(ident, on="TransactionID", how="left")
    df = df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    df = add_entities(df)
    if cache:
        df.to_parquet(cached, index=False)
    return df


def add_entities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    day = np.floor(df["TransactionDT"].to_numpy() / DAY)
    first_seen = np.where(df["D1"].isna(), -1, np.floor(day - df["D1"].fillna(0).to_numpy())).astype(int)
    df["card_uid"] = (
        df["card1"].astype(str) + "_" + df["addr1"].fillna(-1).astype(int).astype(str) + "_" + first_seen.astype(str)
    )
    present = [c for c in DEVICE_PARTS if c in df.columns]
    if present:
        parts = df[present].astype("string")
        has_any = parts.notna().any(axis=1)
        fp = parts.fillna("?").agg("|".join, axis=1)
        df["device_fp"] = fp.where(has_any, None).astype(object)
    else:
        df["device_fp"] = None
    df["day"] = day.astype(int)
    return df


def model_columns(df: pd.DataFrame) -> list[str]:
    """Raw transaction and identity fields for the GBDT. Excludes ids, the label, raw time and the
    reconstructed entity keys, which would let the model memorise accounts."""
    drop = {"TransactionID", "isFraud", "TransactionDT", "day", "card_uid", "device_fp"}
    return [c for c in df.columns if c not in drop]
