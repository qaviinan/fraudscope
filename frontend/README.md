# FraudScope Frontend

Single-page Next.js dashboard for dynamic fraud graph exploration.

## Run

1. Choose a data source mode:
   - `api`: needs local backend API running.
   - `static`: reads precomputed snapshot files only.
2. Copy env file:

```bash
cp .env.example .env.local
```

3. Start development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Build

```bash
npm run build
```

Static output is emitted to `frontend/out/` (`output: "export"`).

## Environment

- `NEXT_PUBLIC_FRAUD_API_BASE_URL` default: `http://127.0.0.1:8000`
- `NEXT_PUBLIC_DATA_SOURCE`:
  - `api` (live backend requests)
  - `static` (precomputed local files in `public/snapshots`)

## Static Snapshot Workflow (recommended for Vercel)

From repo root, generate snapshots:

```bash
conda run -n gpt2-pytorch python scripts/export_frontend_snapshots.py \
  --artifacts-dir outputs/backend_api/artifacts \
  --out-dir frontend/public/snapshots
```

Set in `.env.local`:

```bash
NEXT_PUBLIC_DATA_SOURCE=static
```

Deploy only `frontend/` to Vercel. No model training, feature engineering, or inference runs in cloud runtime.

## Dataset Modes in UI

- `Demo Dataset`: small, interpretable motifs (star, ring, benign cluster) for validating graph behavior.
- `Main Dataset`: full synthetic backend artifacts for realistic/noisy structure.

## Libraries

- `react-force-graph` for interactive relationship graph
- `recharts` for waterfall/timeline/embedding charts
- `framer-motion` for staged reveal animations

## Main file

- `app/page.tsx`
