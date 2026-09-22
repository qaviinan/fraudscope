"""
Causal graph neighbour aggregates: the graph features fraud teams actually use.

For every transaction and every identity-like entity it touches (card, address, device), compute,
using only transactions that happened *before* it:
  * how many prior transactions the entity has in a trailing window (velocity)
  * how many *other* accounts have used the entity in a trailing window (reuse across identities)
  * the entity's prior fraud rate among transactions whose label is already known (train rows only)
  * a two-hop count: distinct addresses reachable through devices previously seen with the card
Nothing here reads the current row's label or any future row, so the features are leak-free for a
time split and can be recomputed identically in production.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

DEFAULT_ENTITY_COLUMNS: Tuple[str, ...] = ("card1", "addr1", "DeviceInfo")
DEFAULT_WINDOWS_HOURS: Tuple[float, ...] = (1.0, 24.0, 168.0)


def _entity_window_features(
    times: np.ndarray,
    uids: np.ndarray,
    known_fraud: np.ndarray,
    known_mask: np.ndarray,
    windows_s: Sequence[float],
    prior_rate: float,
    prior_weight: float,
    label_lag_s: float,
) -> Dict[str, np.ndarray]:
    """Per-entity causal features for one entity group, rows already sorted by time."""
    n = len(times)
    out: Dict[str, np.ndarray] = {}
    cum_known = np.concatenate([[0], np.cumsum(known_mask)])
    cum_fraud = np.concatenate([[0], np.cumsum(known_fraud)])
    usable = np.searchsorted(times, times - label_lag_s, side="left")   # labels older than the lag only
    out["prior_tx_all"] = np.arange(n, dtype=np.float32)
    out["prior_known_fraud_rate"] = ((cum_fraud[usable] + prior_rate * prior_weight) / (cum_known[usable] + prior_weight)).astype(np.float32)
    out["prior_known_fraud_count"] = cum_fraud[usable].astype(np.float32)
    # first-seen age in hours
    out["hours_since_first_seen"] = ((times - times[0]) / 3600.0).astype(np.float32)
    for w in windows_s:
        start = np.searchsorted(times, times - w, side="left")
        out[f"prior_tx_{int(w // 3600)}h"] = (np.arange(n) - start).astype(np.float32)
        # distinct *other* accounts in the window before this row
        distinct = np.zeros(n, dtype=np.float32)
        window: deque = deque()
        counts: Dict[object, int] = defaultdict(int)
        for i in range(n):
            while window and window[0][0] < times[i] - w:
                _, u = window.popleft()
                counts[u] -= 1
                if counts[u] == 0:
                    del counts[u]
            distinct[i] = len(counts) - (1 if uids[i] in counts else 0)
            window.append((times[i], uids[i]))
            counts[uids[i]] += 1
        out[f"other_accounts_{int(w // 3600)}h"] = distinct
    return out


def build_causal_graph_features(
    df: pd.DataFrame,
    entity_columns: Iterable[str] = DEFAULT_ENTITY_COLUMNS,
    uid_col: str = "uid_clean",
    time_col: str = "TransactionDT",
    label_col: str = "isFraud",
    label_known_mask: np.ndarray | None = None,
    windows_hours: Sequence[float] = DEFAULT_WINDOWS_HOURS,
    prior_weight: float = 5.0,
    label_lag_days: float = 14.0,
) -> pd.DataFrame:
    """
    Returns a DataFrame aligned to df.index with one block of columns per entity column
    (prefix `g_<column>_`) plus two-hop features. Rows whose entity value is missing get 0 for
    counts and the global prior for rates.

    label_known_mask marks rows whose label may be used as history (typically the train split).
    Labels of other rows are never read, and a label only becomes usable label_lag_days after its
    transaction (chargebacks arrive late), so "prior fraud on this entity" is what production would know.
    """
    if label_known_mask is None:
        label_known_mask = np.ones(len(df), dtype=bool)
    label_known_mask = np.asarray(label_known_mask, dtype=bool)
    order = np.argsort(df[time_col].to_numpy(), kind="mergesort")
    times_all = df[time_col].to_numpy(dtype=np.float64)[order]
    uids_all = df[uid_col].astype(str).to_numpy()[order]
    y_all = df[label_col].to_numpy(dtype=np.float64)[order]
    known_all = label_known_mask[order]
    known_fraud_all = np.where(known_all, y_all, 0.0)
    prior_rate = float(known_fraud_all.sum() / max(1.0, known_all.sum()))
    windows_s = [float(h) * 3600.0 for h in windows_hours]

    n = len(df)
    feats: Dict[str, np.ndarray] = {}
    for col in entity_columns:
        vals = df[col].to_numpy(dtype=object)[order]
        present = np.array([not (v is None or (isinstance(v, float) and np.isnan(v))) for v in vals])
        keys = np.where(present, vals.astype(str), None)
        groups: Dict[str, List[int]] = defaultdict(list)
        for i, k in enumerate(keys):
            if k is not None:
                groups[k].append(i)
        block: Dict[str, np.ndarray] = {}
        for k, idxs in groups.items():
            ix = np.asarray(idxs, dtype=np.int64)
            f = _entity_window_features(times_all[ix], uids_all[ix], known_fraud_all[ix], known_all[ix], windows_s, prior_rate, prior_weight, label_lag_days * 86_400.0)
            for name, arr in f.items():
                if name not in block:
                    fill = prior_rate if name == "prior_known_fraud_rate" else 0.0
                    block[name] = np.full(n, fill, dtype=np.float32)
                block[name][ix] = arr
        for name, arr in block.items():
            feats[f"g_{col}_{name}"] = arr

    # two-hop, causal: distinct addresses reachable via devices previously seen with this card,
    # and distinct cards reachable via addresses previously seen with this device.
    if {"card1", "addr1", "DeviceInfo"}.issubset(set(entity_columns)):
        card_devices: Dict[str, Set[str]] = defaultdict(set)
        device_addrs: Dict[str, Set[str]] = defaultdict(set)
        device_cards: Dict[str, Set[str]] = defaultdict(set)
        addr_cards: Dict[str, Set[str]] = defaultdict(set)
        cards = df["card1"].to_numpy(dtype=object)[order]
        addrs = df["addr1"].to_numpy(dtype=object)[order]
        devs = df["DeviceInfo"].to_numpy(dtype=object)[order]
        two_hop_addr = np.zeros(n, dtype=np.float32)
        two_hop_card = np.zeros(n, dtype=np.float32)
        for i in range(n):
            c, a, d = str(cards[i]), str(addrs[i]), devs[i]
            d = None if (d is None or (isinstance(d, float) and np.isnan(d))) else str(d)
            reach_a: Set[str] = set()
            for dd in card_devices.get(c, ()):
                reach_a |= device_addrs.get(dd, set())
            reach_a.discard(a)
            two_hop_addr[i] = len(reach_a)
            if d is not None:
                reach_c: Set[str] = set()
                for aa in device_addrs.get(d, ()):
                    reach_c |= addr_cards.get(aa, set())
                reach_c.discard(c)
                two_hop_card[i] = len(reach_c)
                card_devices[c].add(d)
                device_addrs[d].add(a)
                device_cards[d].add(c)
            addr_cards[a].add(c)
        feats["g_2hop_addresses_via_card_devices"] = two_hop_addr
        feats["g_2hop_cards_via_device_addresses"] = two_hop_card

    out = pd.DataFrame(feats)
    inv = np.empty_like(order)
    inv[order] = np.arange(n)
    out = out.iloc[inv].reset_index(drop=True)
    out.index = df.index
    return out
