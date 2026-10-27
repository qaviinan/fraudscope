"""Out-of-time evaluation: prequential folds, operating-point metrics and entity-level bootstrap."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

DAY = 86_400.0


@dataclass(frozen=True)
class Fold:
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_end: float
    test_start: float


def prequential_folds(
    times: np.ndarray,
    n_folds: int = 3,
    test_days: float = 30.0,
    gap_days: float = 30.0,
    min_train_days: float = 60.0,
) -> List[Fold]:
    """Expanding-window folds. Each test window starts gap_days after training ends, so labels on
    the last training rows have had time to mature, as they would before a real retrain."""
    t = np.asarray(times, dtype=float)
    t0, t1 = t.min(), t.max()
    folds: List[Fold] = []
    for i in range(n_folds):
        test_end = t1 - (n_folds - 1 - i) * test_days * DAY
        test_start = test_end - test_days * DAY
        train_end = test_start - gap_days * DAY
        if train_end - t0 < min_train_days * DAY:
            continue
        train_idx = np.flatnonzero(t < train_end)
        test_idx = np.flatnonzero((t >= test_start) & (t <= test_end))
        folds.append(Fold(f"fold{i + 1}", train_idx, test_idx, train_end, test_start))
    if not folds:
        raise ValueError("timeline too short for the requested folds")
    return folds


def threshold_at_fpr(y: np.ndarray, score: np.ndarray, fpr: float) -> float:
    neg = np.sort(score[y == 0])[::-1]
    if len(neg) == 0:
        return float("inf")
    k = int(np.floor(fpr * len(neg)))
    return float(neg[k]) if k < len(neg) else float(neg[-1])


def tpr_at_fpr(y: np.ndarray, score: np.ndarray, fpr: float) -> float:
    if y.sum() == 0:
        return float("nan")
    thr = threshold_at_fpr(y, score, fpr)
    return float((score[y == 1] > thr).mean())


def value_recall_at_fpr(y: np.ndarray, score: np.ndarray, amount: np.ndarray, fpr: float) -> float:
    """Share of fraud dollars caught when declining at the given false-positive rate."""
    fraud_value = amount[y == 1].sum()
    if fraud_value <= 0:
        return float("nan")
    thr = threshold_at_fpr(y, score, fpr)
    return float(amount[(y == 1) & (score > thr)].sum() / fraud_value)


def precision_at_k_daily(y: np.ndarray, score: np.ndarray, day: np.ndarray, k: int) -> float:
    """Mean over days of precision among the day's k highest-scored rows (the analyst queue)."""
    precisions = []
    for d in np.unique(day):
        m = day == d
        top = np.argsort(-score[m], kind="stable")[:k]
        if len(top):
            precisions.append(y[m][top].mean())
    return float(np.mean(precisions)) if precisions else float("nan")


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    # quantile bins: with ~3% prevalence equal-width bins leave almost everything in the first bin
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    err = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            err += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(err)


MetricFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]


def default_metrics(fprs: Sequence[float] = (0.005, 0.01), k: int = 100) -> Dict[str, MetricFn]:
    """Each metric takes (y, score, amount, day)."""
    m: Dict[str, MetricFn] = {}
    for f in fprs:
        tag = f"{f * 100:g}pct"
        m[f"tpr_at_fpr_{tag}"] = lambda y, s, a, d, f=f: tpr_at_fpr(y, s, f)
        m[f"value_recall_at_fpr_{tag}"] = lambda y, s, a, d, f=f: value_recall_at_fpr(y, s, a, f)
    m[f"precision_at_{k}_daily"] = lambda y, s, a, d: precision_at_k_daily(y, s, d, k)
    m["auprc"] = lambda y, s, a, d: float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")
    m["auroc"] = lambda y, s, a, d: float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")
    return m


def _group_resample(groups: np.ndarray, rng: np.random.Generator, n_boot: int):
    _, codes = np.unique(groups, return_inverse=True)
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(codes.max() + 1))
    lens = np.diff(np.append(starts, len(codes)))
    n_groups = len(starts)
    for _ in range(n_boot):
        pick = rng.integers(0, n_groups, n_groups)
        sel_starts, sel_lens = starts[pick], lens[pick]
        offsets = np.repeat(sel_starts - np.cumsum(np.r_[0, sel_lens[:-1]]), sel_lens) + np.arange(sel_lens.sum())
        yield order[offsets]


def bootstrap_metrics(
    y: np.ndarray,
    scores: Dict[str, np.ndarray],
    amount: np.ndarray,
    day: np.ndarray,
    groups: np.ndarray,
    metrics: Dict[str, MetricFn],
    n_boot: int = 200,
    seed: int = 0,
    baseline: str | None = None,
) -> List[dict]:
    """Point estimates and 95% intervals, resampling whole groups (cards) so one fraudster's
    correlated rows move together. With a baseline, also returns paired differences on the same
    resamples."""
    rng = np.random.default_rng(seed)
    point = {(s, m): fn(y, sc, amount, day) for s, sc in scores.items() for m, fn in metrics.items()}
    draws: Dict[tuple, List[float]] = {key: [] for key in point}
    diffs: Dict[tuple, List[float]] = {(s, m): [] for s in scores for m in metrics if baseline and s != baseline}
    for ix in _group_resample(groups, rng, n_boot):
        yb, ab, db = y[ix], amount[ix], day[ix]
        vals = {(s, m): fn(yb, sc[ix], ab, db) for s, sc in scores.items() for m, fn in metrics.items()}
        for key, v in vals.items():
            draws[key].append(v)
        for (s, m) in diffs:
            diffs[(s, m)].append(vals[(s, m)] - vals[(baseline, m)])
    rows = []
    for (s, m), v in point.items():
        lo, hi = np.nanpercentile(draws[(s, m)], [2.5, 97.5])
        row = {"model": s, "metric": m, "value": v, "ci_low": lo, "ci_high": hi}
        if (s, m) in diffs:
            d = np.asarray(diffs[(s, m)])
            row.update({
                "diff_vs_baseline": v - point[(baseline, m)],
                "diff_ci_low": float(np.nanpercentile(d, 2.5)),
                "diff_ci_high": float(np.nanpercentile(d, 97.5)),
            })
        rows.append(row)
    return rows
