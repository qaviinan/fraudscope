# FraudScope Backend (Synthetic IEEE-CIS Fraud Graph API)

Backend for graph-centric fraud analysis and dynamic visualization payloads.

## What this backend does

- Generates a large synthetic IEEE-CIS-style dataset using real-column distribution profiles.
- Assumes clean entity IDs (`uid_clean`) for initial implementation.
- Trains `XGBoost` with asymmetric focal objective.
- Builds UID aggregate features and graph-structural embeddings.
- Produces API-ready artifacts for:
  - force-directed graph nodes/links
  - hop-based expansion
  - star/ring pattern endpoints
  - transaction explainability payloads (waterfall-ready contributions)
  - embedding-space payloads
  - entity timelines

## Key modules

- `src/fraud_graphs/synthetic.py`: synthetic IEEE-CIS-like data generator.
- `src/fraud_graphs/features.py`: UID aggregations + UID label blending.
- `src/fraud_graphs/graph_embeddings.py`: sparse graph embedding features.
- `src/fraud_graphs/modeling.py`: XGBoost training + asymmetric focal objective + explanation helper.
- `src/fraud_graphs/backend_builder.py`: builds full backend artifacts.
- `src/fraud_graphs/graph_module.py`: graph store and neighborhood/pattern queries.
- `src/fraud_graphs/api_server.py`: FastAPI app with all endpoints.

## Build backend artifacts (in `gpt2-pytorch`)

```bash
conda run -n gpt2-pytorch python scripts/build_backend_api_artifacts.py \
  --real-transaction-path data/train_transaction.csv \
  --real-identity-path data/train_identity.csv \
  --n-transactions 250000 \
  --output-dir outputs/backend_api
```

Artifacts are written under `outputs/backend_api/artifacts`.

## Run API

```bash
conda run -n gpt2-pytorch python scripts/run_api.py \
  --artifacts-dir outputs/backend_api/artifacts \
  --host 0.0.0.0 \
  --port 8000
```

Alternative:
```bash
conda run -n gpt2-pytorch uvicorn fraud_graphs.asgi:app --host 0.0.0.0 --port 8000
```

## Core endpoints

- `GET /health`
- `GET /api/v1/meta`
- `GET /api/v1/graph/overview`
- `GET /api/v1/graph/neighborhood?node_id=tx:2000001&hops=2`
- `GET /api/v1/graph/transaction/{transaction_id}`
- `GET /api/v1/graph/patterns/stars`
- `GET /api/v1/graph/patterns/rings`
- `GET /api/v1/transactions/{transaction_id}`
- `GET /api/v1/transactions/{transaction_id}/explain`
- `GET /api/v1/embedding-space?sample_size=2500`
- `GET /api/v1/timeline/entity?entity_type=DeviceInfo&entity_value=iOS`
- `GET /api/v1/search?q=2000`

Most endpoints accept:
- `dataset=main` (default)
- `dataset=demo` (small, idealized motif dataset for graph UX validation)

## API smoke test

```bash
conda run -n gpt2-pytorch python scripts/smoke_test_api.py \
  --artifacts-dir outputs/backend_api/artifacts
```

## Frontend (one-page explorer)

The `frontend/` directory contains a single-page Next.js investigation dashboard:
- force-directed graph explorer
- transaction explainability panel
- UID timeline view
- embedding scatter snapshot
- star/ring pattern lists

### Run frontend

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`.

### Static snapshot mode (Vercel-friendly)

Generate precomputed JSON snapshots locally:

```bash
conda run -n gpt2-pytorch python scripts/export_frontend_snapshots.py \
  --artifacts-dir outputs/backend_api/artifacts \
  --out-dir frontend/public/snapshots
```

Then configure frontend env for static serving only:

- `NEXT_PUBLIC_DATA_SOURCE=static`
- `NEXT_PUBLIC_FRAUD_API_BASE_URL` is ignored in static mode

In this mode, the frontend reads only files under `frontend/public/snapshots/` and does zero backend/model computation in cloud runtime.

### Vercel deployment (no cloud compute)

1. Commit `frontend/public/snapshots/*.json`.
2. In Vercel, set project root to `frontend/`.
3. Set env:
   - `NEXT_PUBLIC_DATA_SOURCE=static`
4. Deploy.

Because the frontend is exported static (`output: "export"`), Vercel serves files only. No model training/feature engineering/inference runs in cloud runtime.

### Frontend env

- `NEXT_PUBLIC_FRAUD_API_BASE_URL` default: `http://127.0.0.1:8000`
- `NEXT_PUBLIC_DATA_SOURCE`:
  - `api` = live API calls
  - `static` = precomputed snapshot files only (recommended for Vercel free tier)

Backend CORS now allows local frontend origins by default:
- `http://localhost:3000`
- `http://127.0.0.1:3000`

Override with:
- `FRAUD_API_CORS_ORIGINS=http://localhost:3000,http://your-host`
