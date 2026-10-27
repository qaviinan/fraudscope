import numpy as np
import pandas as pd

from fraud_graphs import card_track
from fraud_graphs.data import ieee_cis
from fraud_graphs.evaluation import prequential_folds


def fake_ieee(n=4000, seed=0):
    """Small frame with the IEEE-CIS columns the pipeline touches. Fraud is tied to a few shared
    devices so the graph rung has something to find."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(86_400, 150 * 86_400, n))
    card1 = rng.integers(1000, 1300, n)
    addr1 = rng.integers(100, 140, n).astype(float)
    bad_device = rng.random(n) < 0.05
    device = np.where(bad_device, rng.choice(["ring_a", "ring_b"], n), rng.choice([f"dev{i}" for i in range(400)], n))
    device = np.where(rng.random(n) < 0.6, None, device).astype(object)
    device[bad_device] = np.where(bad_device, rng.choice(["ring_a", "ring_b"], n), None)[bad_device]
    y = ((bad_device & (rng.random(n) < 0.7)) | (rng.random(n) < 0.01)).astype(int)
    return pd.DataFrame({
        "TransactionID": np.arange(n), "isFraud": y, "TransactionDT": t,
        "TransactionAmt": rng.lognormal(3.5, 1.0, n).astype(np.float32),
        "ProductCD": rng.choice(["W", "C", "H"], n), "card1": card1, "addr1": addr1,
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], n),
        "R_emaildomain": rng.choice(["gmail.com", None], n),
        "D1": rng.integers(0, 60, n).astype(float), "C1": rng.integers(0, 5, n).astype(float),
        "DeviceInfo": device, "id_30": None, "id_31": None, "id_33": None,
    })


def test_entities_are_built():
    df = ieee_cis.add_entities(fake_ieee())
    assert df["card_uid"].nunique() > 100
    assert df["device_fp"].notna().sum() > 0
    assert "card_uid" not in ieee_cis.model_columns(df)


def test_one_fold_runs_and_graph_rung_uses_shared_devices():
    df = ieee_cis.add_entities(fake_ieee())
    fold = prequential_folds(df["TransactionDT"].to_numpy(), n_folds=1, test_days=30, gap_days=10)[0]
    res = card_track.run_fold(df, fold, label_lag_days=10, edge_shuffle=True)
    assert set(res.scores) == {"rules", "gbdt_fields", "gbdt_velocity", "gbdt_graph", "gbdt_graph_shuffled"}
    for s in res.scores.values():
        assert len(s) == len(res.test)
    assert res.test["TransactionDT"].min() >= fold.test_start


def test_shuffle_keeps_daily_counts():
    df = ieee_cis.add_entities(fake_ieee())
    s = card_track.shuffle_links(df, ["device_fp"], seed=1)
    a = df.groupby("day")["device_fp"].apply(lambda x: sorted(map(str, x)))
    b = s.groupby("day")["device_fp"].apply(lambda x: sorted(map(str, x)))
    assert a.equals(b)
