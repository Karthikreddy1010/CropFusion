"""Contract and correctness tests for the CornSight API (run: python -m pytest -q)."""
import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import risk
from app.store import get_store

c = TestClient(app)
S = get_store()


def test_meta_framing_and_years():
    m = c.get("/api/meta").json()
    assert m["seasons"] == [1985, 2023]
    assert m["estimate_seasons"] == [2008, 2023]
    assert m["counties"] == 583
    assert "not forecasts" in m["framing"]


def test_unit_conversion_matches_model_outputs():
    p = S.preds.merge(S.cy[["county_fips", "year", "corn_yield_bu_acre"]], on=["county_fips", "year"])
    assert (p["observed_t_ha"] - p["corn_yield_bu_acre"] * 0.0627).abs().max() < 1e-9


def test_profile_values_trace_to_store():
    fips = S.preds["county_fips"].iloc[0]
    y = int(S.preds["year"].iloc[0])
    r = c.get(f"/api/county/{fips}/profile", params={"season": y}).json()
    p = S.preds[(S.preds.county_fips == fips) & (S.preds.year == y) & (S.preds.method == "group_conditional_cp")].iloc[0]
    e = r["yield"]["estimate"]
    assert math.isclose(e["t_ha"], p["pred_t_ha"]) and math.isclose(e["lower_t_ha"], p["lower_t_ha"])
    assert math.isclose(e["bu_acre"], p["pred_t_ha"] / 0.0627)
    assert e["nominal_coverage"] == 0.90
    assert e["lower_t_ha"] <= e["t_ha"] <= e["upper_t_ha"]


def test_prior_average_uses_only_earlier_seasons():
    fips = S.cy["county_fips"].iloc[0]
    d = S.cy[S.cy.county_fips == fips].set_index("year")
    exp = d.loc[2005:2009, "yield_t_ha"].dropna().mean()
    assert math.isclose(d.loc[2010, "prior5_avg_t_ha"], exp)


@pytest.mark.parametrize("days,level", [(0, "Low"), (1, "Moderate"), (2, "Moderate"), (3, "High"), (20, "High")])
def test_cdhw_classes_match_protocol(days, level):
    assert risk.cdhw_class(days) == level
    assert risk.model_stratum(days) == {"Low": "Normal", "Moderate": "Moderate", "High": "Extreme"}[level]


def test_profile_class_agrees_with_model_stratum():
    p = S.preds[S.preds.method == "group_conditional_cp"].merge(S.cy[["county_fips", "year", "cdhw_total_days_gs"]], on=["county_fips", "year"])
    assert (p["cdhw_total_days_gs"].map(risk.model_stratum) == p["stratum"]).mean() > 0.999


def test_relative_class_zero_is_low():
    assert risk.relative_class(0, 0.95) == "Low"
    assert risk.relative_class(5, 0.95) == "Very High"
    assert risk.relative_class(5, 0.80) == "High"


def test_season_without_estimate_says_so():
    fips = S.preds["county_fips"].iloc[0]
    r = c.get(f"/api/county/{fips}/profile", params={"season": 1995}).json()
    assert r["yield"]["estimate"] is None and "2008" in r["yield"]["unavailable_reason"]


def test_unknown_county_404():
    assert c.get("/api/county/99999/profile").status_code == 404


def test_rankings_sorted_and_layer_counts():
    r = c.get("/api/rankings", params={"metric": "cdhw_days", "season": 2012, "limit": 50}).json()
    v = [x["value"] for x in r["rows"]]
    assert v == sorted(v, reverse=True)
    lay = c.get("/api/map/layer", params={"layer": "estimate", "season": 2012}).json()
    assert lay["counties_with_value"] == int(((S.preds.year == 2012) & (S.preds.method == "group_conditional_cp")).sum())


def test_state_weighted_mean():
    st = c.get("/api/states/summary", params={"season": 2015}).json()["states"]
    ia = next(x for x in st if x["state"] == "IA")
    d = S.cy[(S.cy.state == "IA") & (S.cy.year == 2015)].dropna(subset=["yield_t_ha", "usda_corn_harvested_acres"])
    d = d[d.usda_corn_harvested_acres > 0]
    exp = (d.yield_t_ha * d.usda_corn_harvested_acres).sum() / d.usda_corn_harvested_acres.sum()
    assert math.isclose(ia["observed_t_ha_acre_weighted"], exp)


def test_planned_integrations_are_not_operational():
    assert all(p["status"] == "planned" for p in c.get("/api/integrations").json())
    assert c.get("/api/integrations/satellite/17001").status_code == 501


def test_lookup_tools_work_without_llm():
    r = c.post("/api/assistant/lookup", json={"tool": "find_county", "args": {"query": "Story, IA"}}).json()
    assert r["matches"][0]["fips"] == "19169"
    r = c.post("/api/assistant/lookup", json={"tool": "county_season", "args": {"fips": "19169", "season": 2012}}).json()
    assert r["season"] == 2012 and r["yield"]["estimate"] is not None


def test_report_has_provenance():
    t = c.get("/api/reports/county/19169.csv").text
    assert t.startswith("# CornSight report") and "sha256" in t
