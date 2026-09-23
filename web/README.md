# React + FastAPI dashboard (experimental)

A parallel dashboard for the same traffic accident hotspot prediction results shown in the
project's Streamlit app (`../app.py`). This is an experiment to compare a React/FastAPI stack
against Streamlit — it does not replace or modify the Streamlit dashboard.

Both apps read the same files and never write to them:

- `../results/metrics.json`, `../results/frozen_config.json`, `../results/feature_importance.csv`
- `../outputs/forecast_2026-01.csv`, `../outputs/test_predictions_2025.parquet`

No model is loaded and no training code is imported by the backend — it only reads the files
above and serves them as JSON.

## Structure

```
web/
  backend/    FastAPI app (read-only JSON API)
  frontend/   React + Vite + TypeScript + Tailwind app
```

## Backend

Uses the project's existing Python virtual environment (`../../venv`) plus a couple of extra
packages.

```bash
cd web/backend
../../venv/Scripts/python.exe -m pip install -r requirements.txt   # first time only
../../venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000
```

On macOS/Linux, replace `../../venv/Scripts/python.exe` with `../../venv/bin/python`.

The API is served at `http://127.0.0.1:8000`. Endpoints:

- `GET /api/summary` — Home page headline metrics
- `GET /api/predictions` — Hotspot forecast table (`tiers`, `grid_id`, `page`, `page_size`)
- `GET /api/risk-map` — Map data (`source`=forecast|test, `month`, `tiers`)
- `GET /api/model-performance` — Confusion matrix, calibration, per-month AP, baselines
- `GET /api/feature-importance` — Permutation importance rows
- `GET /api/health` — liveness check

Interactive docs: `http://127.0.0.1:8000/docs`.

## Frontend

Requires Node.js 20+.

```bash
cd web/frontend
npm install       # first time only
npm run dev
```

Opens at `http://localhost:5173`. In dev mode, requests to `/api/*` are proxied to the backend
on port 8000 (see `vite.config.ts`), so start the backend first.

To type-check and production-build:

```bash
npm run build
```

## Running both together (recommended: one command)

From the project root, in PowerShell:

```powershell
.\run-web.ps1
```

This starts the backend (uvicorn, without `--reload` — it's a demo
launcher, not a dev-reload workflow) and the frontend (`npm run dev`) as
two child processes of the same console, waits for both ports to open,
then prints:

```
=================================================================
  Backend  API :  http://localhost:8000   (docs at /docs)
  Frontend App :  http://localhost:5173
=================================================================
```

Press **Ctrl+C** in that terminal to stop both. The script's `finally`
block force-stops both processes on Ctrl+C, so nothing is left listening
on port 8000 or 5173 afterward — verified by starting it, confirming both
ports respond, sending Ctrl+C, and checking `netstat -ano` shows neither
port still `LISTENING`.

Prerequisites (one-time): the Python venv set up (`Backend` section above)
and `npm install` already run in `web/frontend` (`Frontend` section above).
The script checks for both and exits with a clear message if either is
missing, rather than starting half of the stack.

### Fallback / troubleshooting: two terminals

If `run-web.ps1` doesn't work in your shell (e.g. you're not on Windows,
or you want `uvicorn --reload` for active backend development), run each
side manually:

```bash
# terminal 1
cd web/backend && ../../venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000

# terminal 2
cd web/frontend && npm run dev
```

Then open `http://localhost:5173` in a browser. Stop each with Ctrl+C in
its own terminal.

## Notes

- CORS is enabled on the backend for `http://localhost:5173` and `http://127.0.0.1:5173` only.
- The risk map uses Stadia Maps' `alidade_smooth_dark` tiles, which are free without an API key
  for local development (`localhost`/`127.0.0.1`). A production deployment on a real domain would
  need a Stadia Maps API key or a different tile provider.
- If `results/metrics.json` or the forecast/prediction files are missing, the relevant endpoints
  return HTTP 503 with a message pointing at the pipeline step that produces them (mirroring the
  Streamlit app's `st.info`/`st.error` messages).
