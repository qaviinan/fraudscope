#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

import uvicorn

from fraud_graphs.api_server import create_app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run FraudScope backend API.")
    p.add_argument("--artifacts-dir", default="outputs/backend_api/artifacts")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(artifacts_dir=args.artifacts_dir)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

