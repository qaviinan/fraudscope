#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fraud_graphs.pipeline import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run synthetic fraud graph pipeline.")
    p.add_argument("--real-transaction-path", default="data/train_transaction.csv")
    p.add_argument("--real-identity-path", default="data/train_identity.csv")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--n-transactions", type=int, default=250_000)
    p.add_argument("--sample-rows-for-profile", type=int, default=150_000)
    p.add_argument("--graph-embedding-dim", type=int, default=32)
    p.add_argument("--uid-blend-alpha", type=float, default=0.60)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        real_transaction_path=args.real_transaction_path,
        real_identity_path=args.real_identity_path,
        output_dir=args.output_dir,
        n_transactions=args.n_transactions,
        sample_rows_for_profile=args.sample_rows_for_profile,
        graph_embedding_dim=args.graph_embedding_dim,
        uid_blend_alpha=args.uid_blend_alpha,
        random_state=args.random_state,
    )
    result = run_pipeline(cfg)
    print(json.dumps(result["metrics"], indent=2))
    print(f"metrics_path: {result['metrics_path']}")
    print(f"prediction_path: {result['prediction_path']}")


if __name__ == "__main__":
    main()
