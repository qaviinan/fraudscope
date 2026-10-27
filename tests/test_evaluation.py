import numpy as np

from fraud_graphs.evaluation import (
    DAY,
    _group_resample,
    bootstrap_metrics,
    default_metrics,
    precision_at_k_daily,
    prequential_folds,
    tpr_at_fpr,
    value_recall_at_fpr,
)


def test_folds_leave_a_gap_and_never_overlap():
    t = np.sort(np.random.default_rng(0).uniform(0, 180 * DAY, 5000))
    for f in prequential_folds(t, n_folds=3, test_days=30, gap_days=30):
        assert t[f.train_idx].max() < f.train_end
        assert t[f.test_idx].min() >= f.train_end + 30 * DAY
        assert not set(f.train_idx) & set(f.test_idx)


def test_operating_point_metrics_on_a_perfect_and_a_random_score():
    rng = np.random.default_rng(1)
    y = (rng.random(20000) < 0.03).astype(int)
    amount = rng.uniform(1, 500, len(y))
    assert tpr_at_fpr(y, y + rng.random(len(y)) * 0.1, 0.01) == 1.0
    assert value_recall_at_fpr(y, y.astype(float), amount, 0.01) == 1.0
    random_tpr = tpr_at_fpr(y, rng.random(len(y)), 0.01)
    assert random_tpr < 0.05


def test_daily_precision_uses_each_days_top_k():
    y = np.array([1, 0, 0, 0, 1, 1])
    s = np.array([0.9, 0.8, 0.1, 0.9, 0.2, 0.1])
    day = np.array([0, 0, 0, 1, 1, 1])
    assert precision_at_k_daily(y, s, day, k=1) == 0.5


def test_group_bootstrap_keeps_groups_whole():
    groups = np.array(["a", "a", "a", "b", "c", "c"])
    rng = np.random.default_rng(0)
    for ix in _group_resample(groups, rng, 20):
        picked = groups[ix]
        for g in set(picked):
            assert (picked == g).sum() % (groups == g).sum() == 0


def test_bootstrap_reports_paired_differences():
    rng = np.random.default_rng(2)
    n = 3000
    y = (rng.random(n) < 0.05).astype(int)
    good = y + rng.normal(0, 0.5, n)
    bad = rng.random(n)
    rows = bootstrap_metrics(
        y, {"bad": bad, "good": good}, rng.uniform(1, 100, n), rng.integers(0, 30, n),
        rng.integers(0, 600, n), default_metrics(k=10), n_boot=50, baseline="bad",
    )
    auroc = next(r for r in rows if r["model"] == "good" and r["metric"] == "auroc")
    assert auroc["diff_ci_low"] > 0
