# React + FastAPI dashboard

The primary interface for the project: a read-only dashboard over the project's saved results. Seven pages:

| Page | What it shows |
|---|---|
| Home | Headline result, limitations, and links to the Action list, Risk map and Folium map |
| Risk Map | Next-month forecast or the 2025 retrospective on a map; tier chips, road-type and weather filters |
| Action list | The cells to prioritise next month, ranked, with probability, predicted count, a 12-month sparkline, descriptive drivers and a cell detail panel |
| Patterns | Temporal patterns (hour x weekday, seasonality, weekend, time of day) and DBSCAN clusters (map, sweep, Moran's I) |
| Regression track | Count-regression comparison (validation and test), the Jan 2026 count forecast, LOW/MEDIUM/HIGH quantile tiers next to the 4 rank tiers, and the three ablations |
| Model Performance | Test results against baselines, confusion matrix, calibration, per-month AP, validation, bootstrap CIs |
| Feature Importance | Permutation importance on 2024 validation |

Screenshots are in [`../docs/screenshots/`](../docs/screenshots/) and embedded in the [top-level README](../README.md#screenshots).
A click-by-click demo is in [`../docs/demo_script.md`](../docs/demo_script.md).

## What it reads

The backend only reads files that are tracked in git under `results/` and `outputs/`. It never reads `data/` or `models/`,
loads no model and imports no training code (tests enforce this, and that only GET endpoints exist):

- `results/metrics.json`, `results/frozen_config.json`, `results/feature_importance.csv`
- `results/dbscan_selection.json`, `results/dbscan_sweep.csv`
- `results/regression/` (`regression_metrics.json`, `quantile_tiers_2025.parquet`, `neighbor_features_ablation.json`) and `results/neighbor_features_ablation.json`
- `outputs/forecast_2026-01.csv`, `outputs/test_predictions_2025.parquet`, `outputs/regression_forecast_2026-01.csv` (+ `.meta.json`)
- `outputs/dashboard/` (per-cell drivers, DBSCAN clusters, temporal patterns, Moran's I)
- `outputs/report_compliance/hotspot_map.html` (the Folium map)

The `outputs/dashboard/` files and the regression forecast are produced by read-only scripts in
[`../src/dashboard_exports/`](../src/dashboard_exports/); they need the local `data/` folder and are already committed, so a fresh clone does not need to run them.
The regression forecast script loads the frozen regression model for inference only, checks its SHA-256 before and after, and never reads the 2025 test rows.

## Structure

```
web/
  backend/    FastAPI app (read-only JSON API) + tests
  frontend/   React + Vite + TypeScript + Tailwind app
```

## Quick start (Windows)

From the project root, in PowerShell:

```powershell
.\run-web.ps1
```

This starts the backend (uvicorn, no `--reload`) and the frontend (`npm run dev`), waits for both ports and prints:

```
  Backend  API :  http://localhost:8000   (docs at /docs)
  Frontend App :  http://localhost:5173
```

Press **Ctrl+C** to stop both; the script's `finally` block stops both processes, so nothing is left on ports 8000 or 5173.
One-time prerequisites: the Python venv with the backend packages, and `npm install` in `web/frontend` (the script checks for both and exits with a clear message).

### Setup (first time)

Python 3.13.1 and Node.js 20+ were used. From the project root:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -r web/backend/requirements.txt
cd web\frontend ; npm install
```

(`requirements-report-compliance.txt` is only needed to rebuild the Folium map, the regression track or the exports; the dashboard itself does not need it.)

### Two terminals (fallback / backend development)

```bash
# terminal 1
cd web/backend && ../../venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000
# terminal 2
cd web/frontend && npm run dev
```

On macOS/Linux use `../../venv/bin/python`. Then open `http://localhost:5173`. In dev mode `/api/*` is proxied to the backend on port 8000 (`vite.config.ts`).

## API

All endpoints are `GET`. Interactive docs: `http://127.0.0.1:8000/docs`.

| Endpoint | Purpose |
|---|---|
| `/api/health` | Liveness check |
| `/api/summary` | Home headline metrics |
| `/api/predictions` | Paged forecast table (`tiers`, `grid_id`, `page`, `page_size`); the Action list page uses `/api/action-list` instead |
| `/api/action-list` | Top-N cells for the forecast month: `n`, `offset`, `tiers`, `quantile_tiers`, `road`, `weather`, `min_adverse_share`, `min_single_share`, `sort`; returns drivers, 12-month history and share percentiles |
| `/api/cell/{grid_id}` | One forecast cell with drivers, history, classifier and regression forecasts |
| `/api/risk-map` | Map data: `source` (forecast or test), `month`, `tiers` (default Critical,High,Medium; empty means none), `road`, `weather`, `min_adverse_share`, `min_single_share`, `limit` (reports `total_matching` and `truncated`) |
| `/api/model-performance` | Confusion matrix, calibration, per-month AP, baselines, validation |
| `/api/feature-importance` | Permutation importance rows |
| `/api/temporal-patterns` | Hour x weekday matrix, monthly counts by year, weekend, time of day, clustered-vs-other hours |
| `/api/dbscan/summary` | Chosen parameters, selection criterion, fit statistics, the parameter sweep, Moran's I |
| `/api/dbscan/clusters` | Cluster table (`min_size`, `limit`) |
| `/api/regression/results` | Regression comparison on validation and test, selection decision |
| `/api/regression/forecast` | Jan 2026 count forecast (`n`, `offset`), labelled unvalidated |
| `/api/quantile-tiers` | LOW/MEDIUM/HIGH vs rank tiers: observed hotspot rates, crosstab, forecast counts |
| `/api/ablations` | DBSCAN and neighbour-cell ablations with intervals |
| `/api/map/folium` | The Folium map HTML (opened from the dashboard in a new tab) |

If a saved output is missing, empty or malformed, the endpoint returns HTTP 503 with a message naming the file and the step that produces it
(never a raw traceback), and the page shows that message in place of the affected view.

## Tests

```bash
cd web/backend && ../../venv/Scripts/python.exe -m pytest tests -q     # 111 tests
cd web/frontend && npm run build && npm run lint                        # type-check, bundle, lint
```

The backend tests check served values against the saved files, filters and pagination, bad parameters, and every endpoint's missing and malformed-file behaviour.

## Notes

- CORS is enabled for `http://localhost:5173` and `http://127.0.0.1:5173` only.
- The maps need an internet connection for their base tiles. The Risk Map and cluster map try Stadia Maps (free for `localhost` development without a key), then CARTO, then OpenStreetMap, after repeated tile errors; offline, the coloured points still appear on a blank background.
  A deployment on a real domain would need a Stadia Maps key or another provider. The Folium map also loads its scripts from CDNs, so it needs the internet too.
- Numbers shown in the UI are read from the endpoints; none are hard-coded.
- The Jan 2026 forecasts (classifier probability and regression count) are produced from data through Dec 2025 and have no outcomes yet, so they are labelled unvalidated.
