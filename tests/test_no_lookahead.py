import numpy as np
import pandas as pd

from fraud_graphs.features import uid_label_propagation_blend
from fraud_graphs.graph_features import build_causal_graph_features


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "TransactionDT": np.sort(rng.integers(0, 30 * 86400, n)),
        "uid_clean": rng.integers(0, 40, n).astype(str),
        "card1": rng.integers(0, 25, n),
        "addr1": rng.integers(0, 15, n),
        "DeviceInfo": rng.choice(["a", "b", "c", "d"], n),
        "isFraud": rng.random(n) < 0.1,
    })


def _perturb_future(df, cut):
    out = df.copy()
    late = out["TransactionDT"] > cut
    out.loc[late, "isFraud"] = ~out.loc[late, "isFraud"]
    out.loc[late, "uid_clean"] = "x"
    out.loc[late, "card1"] = 0
    return out


def test_graph_features_ignore_future_rows():
    df = _frame()
    cut = df["TransactionDT"].median()
    early = (df["TransactionDT"] <= cut).to_numpy()
    # labels are known for the early (training) rows only, as in the pipeline
    a = build_causal_graph_features(df, label_known_mask=early)
    b = build_causal_graph_features(_perturb_future(df, cut), label_known_mask=early)
    pd.testing.assert_frame_equal(a[early], b[early])


def test_uid_blend_ignores_future_rows():
    df = _frame()
    cut = df["TransactionDT"].median()
    early = (df["TransactionDT"] <= cut).to_numpy()
    p = np.random.default_rng(1).random(len(df))
    q = p.copy()
    q[~early] = 1.0
    np.testing.assert_allclose(uid_label_propagation_blend(df, p)[early], uid_label_propagation_blend(df, q)[early])


def test_custom_two_hop_path_ignores_future_rows():
    df = _frame()
    cut = df["TransactionDT"].median()
    early = (df["TransactionDT"] <= cut).to_numpy()
    path = [("uid_clean", "DeviceInfo", "addr1", "addr_via_account_devices")]
    a = build_causal_graph_features(df, entity_columns=["DeviceInfo"], two_hop=path, label_known_mask=early)
    b = build_causal_graph_features(_perturb_future(df, cut), entity_columns=["DeviceInfo"], two_hop=path, label_known_mask=early)
    pd.testing.assert_frame_equal(a[early], b[early])


def test_degree_cap_drops_hub_links():
    df = _frame()
    uncapped = build_causal_graph_features(df, max_via_degree=None)
    capped = build_causal_graph_features(df, max_via_degree=1)
    col = "g_2hop_addresses_via_card_devices"
    assert capped[col].sum() < uncapped[col].sum()
