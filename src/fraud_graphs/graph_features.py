"""
Causal graph features.

For each transaction and each entity it touches (card, address, device), using only earlier transactions:
  * prior transactions on the entity in trailing windows (velocity)
  * other accounts that used the entity in trailing windows
  * the entity's past fraud rate, from labels that would already be known
  * two-hop counts, e.g. distinct addresses reachable through devices already seen with the card
No feature reads the current row's label or any later row, so they are safe for a time split and
match what production could compute.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

DEFAULT_ENTITY_COLUMNS: Tuple[str, ...] = ("card1", "addr1", "DeviceInfo")
DEFAULT_WINDOWS_HOURS: Tuple[float, ...] = (1.0, 24.0, 168.0)
# (anchor, via, target, name): distinct targets reachable through vias already seen with the anchor
DEFAULT_TWO_HOP: Tuple[Tuple[str, str, str, str], ...] = (
    ("card1", "DeviceInfo", "addr1", "addresses_via_card_devices"),
    ("DeviceInfo", "addr1", "card1", "cards_via_device_addresses"),
)


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
    """Features for the rows of one entity value. Rows must be sorted by time."""
    n = len(times)
    out: Dict[str, np.ndarray] = {}
    cum_known = np.concatenate([[0], np.cumsum(known_mask)])
    cum_fraud = np.concatenate([[0], np.cumsum(known_fraud)])
    usable = np.searchsorted(times, times - label_lag_s, side="left")   # labels older than the lag only
    out["prior_tx_all"] = np.arange(n, dtype=np.float32)
    out["prior_known_fraud_rate"] = ((cum_fraud[usable] + prior_rate * prior_weight) / (cum_known[usable] + prior_weight)).astype(np.float32)
    out["prior_known_fraud_count"] = cum_fraud[usable].astype(np.float32)
    out["hours_since_first_seen"] = ((times - times[0]) / 3600.0).astype(np.float32)
    for w in windows_s:
        start = np.searchsorted(times, times - w, side="left")
        out[f"prior_tx_{int(w // 3600)}h"] = (np.arange(n) - start).astype(np.float32)
        # distinct accounts in the window before this row, excluding this row's account
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


def _missing(v: object) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def _two_hop_counts(anchors: np.ndarray, vias: np.ndarray, targets: np.ndarray, max_via_degree: int | None) -> np.ndarray:
    """Rows must be in time order. Each row sees only links created by earlier rows."""
    anchor_vias: Dict[str, Set[str]] = defaultdict(set)
    via_targets: Dict[str, Set[str]] = defaultdict(set)
    out = np.zeros(len(anchors), dtype=np.float32)
    cap = np.inf if max_via_degree is None else max_via_degree
    for i in range(len(anchors)):
        a, v, t = anchors[i], vias[i], targets[i]
        v = None if _missing(v) else str(v)
        t = None if _missing(t) else str(t)
        if not _missing(a):
            a = str(a)
            reach: Set[str] = set()
            for vv in anchor_vias.get(a, ()):
                linked = via_targets.get(vv, ())
                if len(linked) <= cap:
                    reach |= linked
            reach.discard(t)
            out[i] = len(reach)
            if v is not None:
                anchor_vias[a].add(v)
        if v is not None and t is not None:
            via_targets[v].add(t)
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
    two_hop: Sequence[Tuple[str, str, str, str]] | None = None,
    max_via_degree: int | None = None,
) -> pd.DataFrame:
    """
    Return features aligned to df.index: one `g_<column>_*` block per entity column, plus two-hop
    counts. Rows with a missing entity value get 0 for counts and the global prior for rates.

    label_known_mask marks rows whose labels may be used as history (usually the train split).
    Other labels are never read. A label becomes usable only label_lag_days after its transaction,
    because chargebacks arrive late.

    With max_via_degree set, two-hop paths skip via-nodes linked to more targets than that, so
    shared infrastructure (a carrier IP, a popular device model) doesn't connect everyone.
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

    entity_set = set(entity_columns)
    paths = DEFAULT_TWO_HOP if two_hop is None else two_hop
    for anchor, via, target, name in paths:
        if two_hop is None and not {anchor, via, target} <= entity_set:
            continue
        feats[f"g_2hop_{name}"] = _two_hop_counts(
            df[anchor].to_numpy(dtype=object)[order],
            df[via].to_numpy(dtype=object)[order],
            df[target].to_numpy(dtype=object)[order],
            max_via_degree,
        )

    out = pd.DataFrame(feats)
    inv = np.empty_like(order)
    inv[order] = np.arange(n)
    out = out.iloc[inv].reset_index(drop=True)
    out.index = df.index
    return out
