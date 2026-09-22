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

from fraud_graphs.backend_builder import BackendBuildConfig, build_backend_artifacts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build API artifacts for FraudScope backend.")
    p.add_argument("--real-transaction-path", default="data/train_transaction.csv")
    p.add_argument("--real-identity-path", default="data/train_identity.csv")
    p.add_argument("--output-dir", default="outputs/backend_api")
    p.add_argument("--n-transactions", type=int, default=250_000)
    p.add_argument("--sample-rows-for-profile", type=int, default=150_000)
    p.add_argument("--graph-embedding-dim", type=int, default=32)
    p.add_argument("--uid-blend-alpha", type=float, default=0.60)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--generator", choices=["v1", "v2"], default="v1", help="v2 = actor-level generator with planted rings")
    p.add_argument("--objective", choices=["focal", "logistic"], default="focal")
    p.add_argument("--causal-graph-features", action="store_true")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--no-svd-embeddings", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BackendBuildConfig(
        real_transaction_path=args.real_transaction_path,
        real_identity_path=args.real_identity_path,
        output_dir=args.output_dir,
        n_transactions=args.n_transactions,
        sample_rows_for_profile=args.sample_rows_for_profile,
        graph_embedding_dim=args.graph_embedding_dim,
        uid_blend_alpha=args.uid_blend_alpha,
        random_state=args.random_state,
        generator=args.generator,
        objective=args.objective,
        causal_graph_features=args.causal_graph_features,
        calibrate=args.calibrate,
        use_svd_embeddings=not args.no_svd_embeddings,
    )
    result = build_backend_artifacts(cfg)
    print(json.dumps(result["metadata"], indent=2))
    print(f"metadata_path: {result['metadata_path']}")
    print(f"artifacts_dir: {result['artifacts_dir']}")


if __name__ == "__main__":
    main()

