"""CornSight API."""
from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel

from .services import agdata, assistant, geo, guidance, risk, yields
from .services.integrations import PLANNED
from .store import DEFAULT_METHOD, METHODS, STATE_NAMES, VARS, get_store

import math as _math

import numpy as _np
from fastapi import encoders as _enc


def _np_float(v):
    v = float(v)
    return v if _math.isfinite(v) else None


for _t in (_np.float16, _np.float32, _np.float64):
    _enc.ENCODERS_BY_TYPE[_t] = _np_float
for _t in (_np.int8, _np.int16, _np.int32, _np.int64, _np.uint8, _np.uint16, _np.uint32, _np.uint64):
    _enc.ENCODERS_BY_TYPE[_t] = int
_enc.ENCODERS_BY_TYPE[_np.bool_] = bool
_enc.ENCODERS_BY_TYPE[float] = lambda v: v if _math.isfinite(v) else None

app = FastAPI(title="CornSight API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5180", "http://127.0.0.1:5180"],
                   allow_methods=["*"], allow_headers=["*"])


def S():
    return get_store()


def _fips(fips: str) -> str:
    f = fips.zfill(5)
    if not S().county_exists(f):
        raise HTTPException(404, f"County {fips} is not in the platform (six Corn Belt states only).")
    return f


def _season(year: int | None) -> int:
    s = S()
    y = year or s.pred_years[-1]
    if y not in s.years:
        raise HTTPException(404, f"No data for season {y}. Seasons available: {s.years[0]}–{s.years[-1]}.")
    return y


def _method(m: str | None) -> str:
    m = m or DEFAULT_METHOD
    if m not in METHODS:
        raise HTTPException(400, f"Unknown interval method. Choose one of: {', '.join(METHODS)}")
    return m


# ---------------------------------------------------------------- meta
@app.get("/api/meta")
def meta():
    s = S()
    m = s.manifest
    return {
        "product": "CornSight", "crop": "Maize (corn)", "states": STATE_NAMES,
        "seasons": [s.years[0], s.years[-1]], "estimate_seasons": [s.pred_years[0], s.pred_years[-1]],
        "default_season": s.pred_years[-1], "methods": METHODS, "default_method": DEFAULT_METHOD,
        "counties": m["checks"]["counties_in_panel"], "counties_with_estimates": m["checks"]["counties_with_predictions"],
        "data_built_utc": m["built_utc"], "constants": m["constants"],
        "sources": {k: {"path": v["path"], "modified_utc": v["modified_utc"], "sha256": v["sha256"][:12]} for k, v in m["sources"].items()},
        "framing": "All seasons are completed, historical seasons. Model estimates are out-of-sample reconstructions, not forecasts.",
        "variables": {k: {"label": a, "unit": b} for k, (a, b) in VARS.items()},
        "assistant_configured": assistant.configured(),
    }


@app.get("/api/counties")
def counties(state: str | None = None):
    s = S()
    c = s.counties
    if state:
        c = c[c["state"] == state]
    has = set(s.preds["county_fips"].unique())
    return [{"fips": f, "county": r["county"], "state": r["state"], "has_estimates": f in has}
            for f, r in c.sort_values(["state", "county"]).iterrows()]


@app.get("/api/counties/search")
def county_search(q: str, limit: int = 10):
    return S().find_counties(q, limit)


# ---------------------------------------------------------------- county intelligence
@app.get("/api/county/{fips}/profile")
def county_profile(fips: str, season: int | None = None, method: str | None = None, farm_acres: float | None = None):
    s = S()
    f, y, m = _fips(fips), _season(season), _method(method)
    info = agdata.county_info(s, f)
    row = s.row(f, y)
    yb = yields.yield_block(s, f, y, m, farm_acres)
    rb = risk.risk_block(row) if row else {}
    st = agdata.stages(s, f, y)
    return {
        "county": info, "season": y, "method": m, "framing": "historical season",
        "yield": yb, "risk": rb, "conditions": agdata.conditions(s, f, y), "stages": st,
        "alerts": guidance.alerts(info, yb, rb, st) if row else [],
        "recommendations": guidance.recommendations(yb, rb, st) if row else [],
        "reliability": yields.reliability(s, m, f),
    }


@app.get("/api/county/{fips}/series")
def county_series(fips: str, method: str | None = None):
    return {"fips": _fips(fips), "rows": yields.county_series(S(), _fips(fips), _method(method))}


@app.get("/api/county/{fips}/stages/history")
def county_stage_history(fips: str):
    return {"fips": _fips(fips), "note": agdata.STAGE_NOTE, "rows": agdata.stage_history(S(), _fips(fips))}


@app.get("/api/compare")
def compare(fips: str = Query(..., description="Comma-separated FIPS codes"), season: int | None = None, method: str | None = None):
    s = S()
    y, m = _season(season), _method(method)
    out = []
    for f in [x.strip() for x in fips.split(",") if x.strip()][:6]:
        f = _fips(f)
        row = s.row(f, y)
        rb = risk.risk_block(row) if row else {}
        out.append({"county": agdata.county_info(s, f), "yield": yields.yield_block(s, f, y, m),
                    "risk": {k: {"level": v.get("level"), "value": v.get("value"), "unit": v.get("unit")} for k, v in rb.items()},
                    "series": yields.county_series(s, f, m)})
    return {"season": y, "method": m, "counties": out}


# ---------------------------------------------------------------- regional
@app.get("/api/states/summary")
def states_summary(season: int | None = None, method: str | None = None):
    return {"season": _season(season), "states": yields.state_summary(S(), _season(season), _method(method))}


@app.get("/api/states/series")
def states_series(method: str | None = None):
    return {"rows": yields.state_series(S(), _method(method))}


@app.get("/api/rankings")
def rankings(metric: str, season: int | None = None, state: str | None = None, method: str | None = None,
             ascending: bool = False, limit: int = 25):
    try:
        return yields.rankings(S(), _season(season), metric, _method(method), state, ascending, min(limit, 600))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/rank-metrics")
def rank_metrics():
    return {k: {"label": a, "unit": b} for k, (a, b, _) in yields.RANK_METRICS.items()}


@app.get("/api/climate/compound-yield")
def compound_yield(state: str | None = None):
    return agdata.compound_yield_relation(S(), state)


@app.get("/api/model/reliability")
def model_reliability(method: str | None = None, fips: str | None = None):
    return yields.reliability(S(), _method(method), _fips(fips) if fips else None)


@app.get("/api/model/attribution")
def model_attribution():
    a = S().attribution
    return {"note": "Mean attribution across the four model backbones (NeuralCQR, LightGBM, CatBoost, XGBoost) from the "
                    "research run's SHAP consistency report. It shows which inputs the models relied on overall, not the "
                    "cause of any single county's yield.",
            "rows": a.to_dict(orient="records")}


# ---------------------------------------------------------------- map
@app.get("/api/geo/counties")
def geo_counties():
    return Response(S().geo_counties, media_type="application/geo+json")


@app.get("/api/geo/states")
def geo_states():
    return Response(S().geo_states, media_type="application/geo+json")


@app.get("/api/map/layers")
def map_layers():
    return geo.LAYERS


@app.get("/api/map/layer")
def map_layer(layer: str, season: int | None = None, method: str | None = None):
    try:
        return geo.layer_values(S(), layer, _season(season), _method(method))
    except ValueError as e:
        raise HTTPException(400, str(e))


# ---------------------------------------------------------------- integrations
@app.get("/api/integrations")
def integrations():
    return PLANNED


@app.api_route("/api/integrations/{name}/{rest:path}", methods=["GET", "POST"])
def integration_stub(name: str, rest: str = ""):
    raise HTTPException(501, f"The '{name}' integration is planned and not connected to a data source yet.")


# ---------------------------------------------------------------- assistant
class ChatIn(BaseModel):
    messages: list[dict]


@app.get("/api/assistant/status")
def assistant_status():
    return {"configured": assistant.configured(), "model": assistant.MODEL,
            "how_to_enable": "Set ANTHROPIC_API_KEY in backend/.env (or run `ant auth login` and set "
                             "CORNSIGHT_ASSISTANT_PROFILE=1), then restart the backend.",
            "tools": [{"name": t["name"], "description": t["description"]} for t in assistant.TOOLS]}


@app.post("/api/assistant/chat")
def assistant_chat(body: ChatIn):
    if not assistant.configured():
        raise HTTPException(503, "The assistant is not configured. Use Data lookup, or add an Anthropic API key.")
    import anthropic
    try:
        return assistant.chat(S(), body.messages)
    except anthropic.AuthenticationError:
        raise HTTPException(503, "The Anthropic credentials were rejected. Check ANTHROPIC_API_KEY.")
    except anthropic.RateLimitError:
        raise HTTPException(429, "The assistant is rate limited. Try again in a minute.")
    except anthropic.APIConnectionError:
        raise HTTPException(503, "Could not reach the Anthropic API. Check the internet connection.")
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"Assistant error ({e.status_code}).")


class LookupIn(BaseModel):
    tool: str
    args: dict = {}


@app.post("/api/assistant/lookup")
def assistant_lookup(body: LookupIn):
    if body.tool not in {t["name"] for t in assistant.TOOLS}:
        raise HTTPException(400, "Unknown lookup.")
    try:
        return assistant.run_tool(S(), body.tool, body.args)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, f"Missing or invalid input: {e}")


# ---------------------------------------------------------------- reports
def _csv(rows: list[dict], title: str, extra: list[str] | None = None) -> PlainTextResponse:
    s = S()
    buf = io.StringIO()
    buf.write(f"# CornSight report: {title}\n")
    buf.write(f"# Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from data store built {s.manifest['built_utc']}\n")
    for k, v in s.manifest["sources"].items():
        buf.write(f"# source {k}: {v['path']} sha256 {v['sha256'][:16]}\n")
    for line in extra or []:
        buf.write(f"# {line}\n")
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(buf, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    fname = title.lower().replace(" ", "_").replace(",", "") + ".csv"
    return PlainTextResponse(buf.getvalue(), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@app.get("/api/reports/county/{fips}.csv")
def report_county(fips: str, method: str | None = None):
    f = _fips(fips)
    m = _method(method)
    return _csv(yields.county_series(S(), f, m), f"County {S().county_label(f)}",
                [f"interval method: {METHODS[m]}; yields t/ha (bu/acre x 0.0627); estimates are out-of-sample reconstructions",
                 "exposure_class: Normal 0, Moderate 1-2, Extreme >=3 compound drought-heat days"])


@app.get("/api/reports/rankings.csv")
def report_rankings(metric: str, season: int | None = None, state: str | None = None, method: str | None = None, ascending: bool = False):
    r = rankings(metric, season, state, method, ascending, 600)
    return _csv(r["rows"], f"Ranking {r['label']} {r['season']}", [f"metric unit: {r['unit']}; order {r['order']}"])


@app.get("/api/reports/states.csv")
def report_states(season: int | None = None, method: str | None = None):
    y = _season(season)
    return _csv(yields.state_summary(S(), y, _method(method)), f"State summary {y}",
                ["acre-weighted means of panel counties, not official NASS state estimates"])


@app.get("/api/reports/region.csv")
def report_region(season: int | None = None, method: str | None = None):
    y, m = _season(season), _method(method)
    t = yields.county_table(S(), y, m)
    cols = ["county_fips", "county", "state", "year", "observed_t_ha", "prior5_avg_t_ha", "estimate_t_ha", "lower_t_ha",
            "upper_t_ha", "interval_width_t_ha", "stratum", "cdhw_days", "drought_days", "heat_days", "kdd30", "ppt_mm",
            "hail", "usda_corn_harvested_acres", "production_bu_estimated"]
    rows = t[cols].rename(columns={"stratum": "exposure_class"}).to_dict(orient="records")
    return _csv(rows, f"Regional county table {y}", [f"interval method: {METHODS[m]}"])
