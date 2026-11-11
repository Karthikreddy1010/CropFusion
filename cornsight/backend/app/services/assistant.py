"""AI agricultural assistant.

The model never sees the dataset directly: it can only call the data tools below, which return
platform data with their sources. Without credentials the same tools are available through the
"data lookup" endpoint, so nothing the assistant says depends on information the platform lacks.
"""
from __future__ import annotations

import json
import os

from ..store import DEFAULT_METHOD, STATE_NAMES, Store
from . import agdata, risk, yields
from .integrations import PLANNED

MODEL = os.environ.get("CORNSIGHT_MODEL", "claude-opus-5-5")

SRC_YIELD = "USDA NASS county yield survey (via Paper3 master panel)"
SRC_MODEL = "Rolling-origin LightGBM model estimates with conformal 90% intervals (Paper3 outputs)"
SRC_CLIMATE = "PRISM AN81d daily weather and NLDAS-2 forcing, SPI/SPEI and compound drought–heat indices (Paper3 master panel)"
SRC_STORM = "NOAA Storm Events Database (Paper3 master panel)"
SRC_SOIL = "Soil rasters, USGS 3DEP terrain, NLCD land cover (Paper3 master panel)"

SYSTEM = f"""You are the CornSight agricultural assistant for maize (corn) in 583 counties of Illinois, Indiana, Iowa,
Minnesota, Missouri and Ohio. Data cover seasons 1985–2023; model yield estimates with 90% prediction intervals exist
for 2008–2023 only.

Rules:
- Answer only from tool results. Call tools for every number you state. If a tool returns no data, say the data are
  not available; never estimate, guess or fill gaps from general knowledge.
- Always name the county (or state) and the season you are describing.
- Every season in the platform is a completed, historical season. Model estimates are out-of-sample reconstructions
  of past seasons, not forecasts. The platform has no current-season data, no weather forecast, no satellite data and
  no field-level data; say so if asked.
- When you give a model estimate, give its 90% prediction interval and say that it describes prediction uncertainty,
  not a guaranteed range.
- Explain indicators in plain words (for example: SPEI-30 below −1 means a 30-day water deficit; a compound
  drought–heat day has maximum temperature above 35 °C while SPEI-30 is below −1).
- Association in historical data is not proof of cause. Say "was associated with", not "caused".
- Do not give irrigation, fertilizer, pesticide or marketing prescriptions. You may point to the U.S. Drought Monitor,
  National Weather Service, USDA Risk Management Agency or the local extension service.
- Units: yield in t/ha and bu/acre (1 bu/acre = 0.0627 t/ha), precipitation in mm, temperature in °C.
- End every answer with a line "Sources:" listing the sources reported by the tools you used.
- Keep answers short and concrete: a few sentences or a small table.
"""

TOOLS = [
    {"name": "find_county", "description": "Find counties by name, e.g. 'Story, IA' or 'Champaign'. Returns FIPS codes.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}},
    {"name": "county_season", "description": "Yield (observed, model estimate with 90% interval, 5-season average), climate risk classes, and growth-stage stress for one county and season.",
     "input_schema": {"type": "object", "properties": {"fips": {"type": "string"}, "season": {"type": "integer"}}, "required": ["fips", "season"], "additionalProperties": False}},
    {"name": "county_history", "description": "Season-by-season observed yield, model estimate and interval (2008+), compound drought–heat days, drought days, heat days and precipitation for one county.",
     "input_schema": {"type": "object", "properties": {"fips": {"type": "string"}, "first_season": {"type": "integer"}, "last_season": {"type": "integer"}}, "required": ["fips"], "additionalProperties": False}},
    {"name": "rank_counties", "description": "Rank counties for one season by a metric. Metrics: " + ", ".join(yields.RANK_METRICS),
     "input_schema": {"type": "object", "properties": {"season": {"type": "integer"}, "metric": {"type": "string", "enum": list(yields.RANK_METRICS)},
                                                         "state": {"type": "string", "enum": list(STATE_NAMES)}, "lowest_first": {"type": "boolean"},
                                                         "limit": {"type": "integer"}}, "required": ["season", "metric"], "additionalProperties": False}},
    {"name": "state_season", "description": "Acre-weighted state yields (observed, estimate, interval), production and counts of high-risk counties for one season, all six states.",
     "input_schema": {"type": "object", "properties": {"season": {"type": "integer"}}, "required": ["season"], "additionalProperties": False}},
    {"name": "state_history", "description": "Season-by-season acre-weighted observed yield and model estimate for one state.",
     "input_schema": {"type": "object", "properties": {"state": {"type": "string", "enum": list(STATE_NAMES)}}, "required": ["state"], "additionalProperties": False}},
    {"name": "compound_vs_yield", "description": "How observed yield compared with the prior five-season average in seasons with Low, Moderate and High compound drought–heat exposure (1985–2023), optionally for one state.",
     "input_schema": {"type": "object", "properties": {"state": {"type": "string", "enum": list(STATE_NAMES)}}, "additionalProperties": False}},
    {"name": "model_reliability", "description": "How often past 90% prediction intervals contained the observed yield, by exposure class, optionally for one county.",
     "input_schema": {"type": "object", "properties": {"fips": {"type": "string"}}, "additionalProperties": False}},
    {"name": "platform_coverage", "description": "What data and capabilities the platform has, and which integrations are only planned.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
]


def _county_or_error(s: Store, fips: str):
    fips = str(fips).zfill(5)
    if not s.county_exists(fips):
        return None, {"error": f"No county with FIPS {fips} in the platform. Use find_county."}
    return fips, None


def run_tool(s: Store, name: str, args: dict) -> dict:
    method = DEFAULT_METHOD
    if name == "find_county":
        hits = s.find_counties(args.get("query", ""))
        return {"matches": hits, "sources": [SRC_SOIL]} if hits else {"matches": [], "note": "No matching county."}
    if name == "county_season":
        fips, err = _county_or_error(s, args["fips"])
        if err:
            return err
        y = int(args["season"])
        row = s.row(fips, y)
        if row is None:
            return {"error": f"No data for {s.county_label(fips)} in {y}. Seasons available: {s.years[0]}–{s.years[-1]}."}
        yb = yields.yield_block(s, fips, y, method)
        rb = risk.risk_block(row)
        st = agdata.stages(s, fips, y)
        return {"county": s.county_label(fips), "fips": fips, "season": y, "framing": "historical season",
                "yield": yb, "risk": {k: {kk: vv for kk, vv in v.items() if kk not in ("rule",)} for k, v in rb.items()},
                "risk_rules": {"compound": risk.CDHW_RULE, "drought_heat_rainfall": risk.RELATIVE_RULE},
                "most_stressed_stage": st.get("most_stressed"), "stage_note": agdata.STAGE_NOTE,
                "sources": [SRC_YIELD, SRC_MODEL, SRC_CLIMATE, SRC_STORM]}
    if name == "county_history":
        fips, err = _county_or_error(s, args["fips"])
        if err:
            return err
        ser = yields.county_series(s, fips, method)
        a, b = args.get("first_season", s.years[0]), args.get("last_season", s.years[-1])
        ser = [r for r in ser if a <= r["year"] <= b]
        keep = ["year", "observed_t_ha", "pred_t_ha", "lower_t_ha", "upper_t_ha", "exposure_class", "cdhw_days", "drought_days", "heat_days", "ppt_mm"]
        return {"county": s.county_label(fips), "fips": fips, "rows": [{k: r.get(k) for k in keep} for r in ser],
                "sources": [SRC_YIELD, SRC_MODEL, SRC_CLIMATE]}
    if name == "rank_counties":
        r = yields.rankings(s, int(args["season"]), args["metric"], method, args.get("state"),
                            ascending=bool(args.get("lowest_first", False)), limit=min(int(args.get("limit", 10)), 30))
        r["sources"] = [SRC_YIELD, SRC_MODEL, SRC_CLIMATE]
        return r
    if name == "state_season":
        y = int(args["season"])
        return {"season": y, "states": yields.state_summary(s, y, method), "sources": [SRC_YIELD, SRC_MODEL, SRC_CLIMATE]}
    if name == "state_history":
        rows = [r for r in yields.state_series(s, method) if r["state"] == args["state"]]
        return {"state": STATE_NAMES[args["state"]], "rows": rows, "definition": "Acre-weighted means of panel counties.",
                "sources": [SRC_YIELD, SRC_MODEL]}
    if name == "compound_vs_yield":
        r = agdata.compound_yield_relation(s, args.get("state"))
        r["sources"] = [SRC_YIELD, SRC_CLIMATE]
        return r
    if name == "model_reliability":
        fips = args.get("fips")
        if fips:
            fips, err = _county_or_error(s, fips)
            if err:
                return err
        r = yields.reliability(s, method, fips)
        r["sources"] = [SRC_MODEL]
        return r
    if name == "platform_coverage":
        m = s.manifest
        return {"panel_seasons": m["checks"]["panel_years"], "estimate_seasons": m["checks"]["prediction_years"],
                "counties": m["checks"]["counties_in_panel"], "counties_with_estimates": m["checks"]["counties_with_predictions"],
                "data_built": m["built_utc"], "planned_not_available": [p["name"] for p in PLANNED],
                "sources": ["CornSight data manifest"]}
    return {"error": f"Unknown tool {name}"}


def configured() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return os.environ.get("CORNSIGHT_ASSISTANT_PROFILE", "") == "1"  # set after `ant auth login`


def chat(s: Store, history: list[dict], max_rounds: int = 8) -> dict:
    """history: [{role: user|assistant, content: str}]. Returns answer text and the tool calls made."""
    import anthropic

    client = anthropic.Anthropic()
    messages = [{"role": m["role"], "content": m["content"]} for m in history if m.get("content")]
    trace = []
    for _ in range(max_rounds):
        resp = client.beta.messages.create(
            model=MODEL, max_tokens=16000, system=SYSTEM, tools=TOOLS, messages=messages,
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
        if resp.stop_reason == "refusal":
            return {"answer": "The assistant declined this request. Try rephrasing it as a question about the platform's data.",
                    "tool_calls": trace, "stop_reason": "refusal"}
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            text = "\n".join(b.text for b in resp.content if b.type == "text").strip()
            return {"answer": text, "tool_calls": trace, "stop_reason": resp.stop_reason}
        results = []
        for b in resp.content:
            if b.type != "tool_use":
                continue
            try:
                out = run_tool(s, b.name, dict(b.input))
                err = "error" in out
            except Exception as exc:  # report to the model, never crash the turn
                out, err = {"error": f"{type(exc).__name__}: {exc}"}, True
            trace.append({"tool": b.name, "input": dict(b.input), "ok": not err})
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out, default=str), "is_error": err})
        messages.append({"role": "user", "content": results})
    return {"answer": "The question needed more data lookups than allowed. Try a narrower question.", "tool_calls": trace,
            "stop_reason": "max_rounds"}
