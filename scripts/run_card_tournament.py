#!/usr/bin/env python3
"""Card track: run the model ladder on IEEE-CIS out-of-time folds and write results/card/metrics.csv.

  python scripts/run_card_tournament.py                        # gap 30 days, 3 folds
  python scripts/run_card_tournament.py --gaps 0 30 60 --edge-shuffle
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_graphs import card_track  # noqa: E402
from fraud_graphs.data import ieee_cis  # noqa: E402
from fraud_graphs.evaluation import bootstrap_metrics, default_metrics, prequential_folds  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/ieee")
    ap.add_argument("--out", default="results/card")
    ap.add_argument("--gaps", type=float, nargs="+", default=[30.0])
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--test-days", type=float, default=30.0)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--k", type=int, default=100, help="analyst reviews per day")
    ap.add_argument("--edge-shuffle", action="store_true")
    ap.add_argument("--sample-accounts", type=float, default=None, help="keep this share of accounts, for quick runs")
    args = ap.parse_args()

    df = ieee_cis.load(args.data_dir)
    if args.sample_accounts:
        keep = df[ieee_cis.UID_COLUMN].drop_duplicates().sample(frac=args.sample_accounts, random_state=0)
        df = df[df[ieee_cis.UID_COLUMN].isin(set(keep))].reset_index(drop=True)
    print(f"{len(df):,} transactions, {df['isFraud'].mean():.2%} fraud, {df['day'].nunique()} days")

    metrics = default_metrics(k=args.k)
    rows = []
    for gap in args.gaps:
        folds = prequential_folds(df["TransactionDT"].to_numpy(), args.folds, args.test_days, gap, min_train_days=45)
        for fold in folds:
            t0 = time.time()
            res = card_track.run_fold(df, fold, label_lag_days=max(gap, 1.0), edge_shuffle=args.edge_shuffle)
            test = res.test
            slices = {"all": slice(None), "first_seen_account": test["first_seen_account"].to_numpy(),
                      "returning_account": ~test["first_seen_account"].to_numpy()}
            for name, m in slices.items():
                part = test[m]
                if part["isFraud"].sum() < 20:
                    continue
                out = bootstrap_metrics(
                    part["isFraud"].to_numpy(), {k: v[m] for k, v in res.scores.items()},
                    part["TransactionAmt"].to_numpy(), part["day"].to_numpy(),
                    part[ieee_cis.UID_COLUMN].to_numpy(), metrics, n_boot=args.n_boot, baseline="gbdt_velocity",
                )
                for r in out:
                    r.update({"gap_days": gap, "fold": fold.name, "slice": name,
                              "n_test": len(part), "n_fraud": int(part["isFraud"].sum())})
                rows.extend(out)
            print(f"gap {gap:g}d {fold.name}: {len(test):,} test rows in {time.time() - t0:.0f}s")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "metrics.csv", index=False)
    summary = table[(table["slice"] == "all") & table["metric"].isin(["tpr_at_fpr_1pct", f"precision_at_{args.k}_daily", "auprc"])]
    print(summary.groupby(["gap_days", "model", "metric"])["value"].mean().unstack().round(3).to_string())


if __name__ == "__main__":
    main()
