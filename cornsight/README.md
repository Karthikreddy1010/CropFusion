# CornSight

Agricultural intelligence and climate-risk decision support for maize in 583 counties of
Illinois, Indiana, Iowa, Minnesota, Missouri and Ohio (seasons 1985–2023).

CornSight runs locally. It reads the research project's outputs **read-only** and never sends the
data anywhere, except the questions you type into the optional AI assistant, which go to the
Anthropic API with only the data the assistant's tools return.

## First-time setup

```powershell
cd cornsight\backend
python -m venv .venv
.\.venv\Scripts\pip install fastapi "uvicorn[standard]" pandas pyarrow anthropic pytest httpx
python etl\build_store.py        # builds data_store/ from the repository root; or --paper3 <path>

cd ..\frontend
npm install
```

The data store is built locally from the research outputs and is not part of the repository.
The builder records the SHA-256 of every source file in `data_store/manifest.json`; the
Settings page and every CSV export show them.

## Run it

```powershell
# 1. API on http://127.0.0.1:8010
cd cornsight\backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8010

# 2. Web app on http://localhost:5180
cd cornsight\frontend
npm run dev
```

Or run `.\run.ps1` from `cornsight\` to start both.

### Turn on the AI assistant (optional)

Set an Anthropic API key for the backend process, then restart it:

```powershell
$env:ANTHROPIC_API_KEY = "<your key>"
```

Without a key the Assistant page offers **Data lookup**, which answers the same questions directly
from the data with no language model.

### Tests

```powershell
cd backend; .\.venv\Scripts\python.exe -m pytest -q      # units, traceability, risk classes, API
cd frontend; npx tsc -p tsconfig.app.json --noEmit         # type check
```

## What is real and what is planned

Real: observed yields, out-of-sample model estimates with 90% conformal intervals (2008–2023),
SPI/SPEI, heat, compound drought–heat, stage proxies, storms, soil, land cover, maps, rankings,
reports, assistant over these data.

Planned (shown as planned, never with placeholder numbers): current-season forecasts, weather
forecasts and alerts, satellite NDVI/EVI, field-level monitoring, observed planting/stage dates,
irrigation / fertilizer / pest recommendations, notifications.
