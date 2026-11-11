"""Yield intelligence: observed yields, model estimates with prediction intervals, production, rankings."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..store import DEFAULT_METHOD, METHODS, STATE_NAMES, Store, clean

T_HA_PER_BU_ACRE = 0.0627
NOMINAL = 0.90
ACRES_PER_HA = 2.47105

INTERVAL_TEXT = ("The model's 90% prediction interval: intervals built this way aim to contain the observed yield "
                 "9 times in 10. How often they did in past seasons is shown in the reliability table. "
                 "It describes prediction uncertainty, not a guaranteed range.")
ESTIMATE_TEXT = ("Out-of-sample model estimate: for season {y} the model was fitted only on seasons up to {f} and "
                 "calibrated on {c1}–{c2}, then used that season's weather to estimate end-of-season yield. "
                 "It is a historical reconstruction, not a forecast of a future season.")


def bu(t_ha):
    return None if t_ha is None else t_ha / T_HA_PER_BU_ACRE


def yield_block(s: Store, fips: str, year: int, method: str = DEFAULT_METHOD, farm_acres: float | None = None) -> dict:
    row = s.row(fips, year) or {}
    p = s.pred(fips, year, method)
    obs = row.get("yield_t_ha")
    avg = row.get("prior5_avg_t_ha")
    out = {
        "season": year, "method": method, "method_label": METHODS.get(method, method),
        "observed": {"t_ha": obs, "bu_acre": row.get("corn_yield_bu_acre"), "source": "USDA NASS county survey"},
        "prior5_average": {"t_ha": avg, "bu_acre": bu(avg),
                           "definition": f"Mean observed yield of the five seasons {year-5}–{year-1} (at least 3 needed)"},
        "harvested_acres": row.get("usda_corn_harvested_acres"),
        "production_bu_observed": row.get("usda_corn_production_bu"),
        "estimate": None, "available": p is not None,
    }
    if obs is not None and avg:
        out["observed_vs_average_pct"] = 100 * (obs - avg) / avg
    if p is None:
        out["unavailable_reason"] = (
            f"No model estimate for {year}. Estimates exist for seasons {s.pred_years[0]}–{s.pred_years[-1]} "
            "and for counties with enough yield history."
            if year in s.years else f"No data for {year}.")
        return out
    est, lo, hi = p["pred_t_ha"], p["lower_t_ha"], p["upper_t_ha"]
    e = {
        "t_ha": est, "bu_acre": bu(est),
        "lower_t_ha": lo, "upper_t_ha": hi, "lower_bu_acre": bu(lo), "upper_bu_acre": bu(hi),
        "width_t_ha": hi - lo, "nominal_coverage": NOMINAL,
        "exposure_class": p["stratum"],
        "inside_interval": (lo <= obs <= hi) if obs is not None else None,
        "error_t_ha": (est - obs) if obs is not None else None,
        "explanation": ESTIMATE_TEXT.format(y=year, f=year - 5, c1=year - 2, c2=year - 1),
        "interval_explanation": INTERVAL_TEXT,
        "model": "LightGBM point model with conformal 90% interval (rolling-origin protocol)",
    }
    if avg:
        e["vs_average_pct"] = 100 * (est - avg) / avg
        e["lower_vs_average_pct"] = 100 * (lo - avg) / avg
    acres = row.get("usda_corn_harvested_acres")
    if acres:
        e["production_bu"] = bu(est) * acres
        e["production_bu_lower"] = bu(lo) * acres
        e["production_bu_upper"] = bu(hi) * acres
    if farm_acres:
        e["farm_production_bu"] = bu(est) * farm_acres
        e["farm_production_bu_lower"] = bu(lo) * farm_acres
        e["farm_production_bu_upper"] = bu(hi) * farm_acres
        e["farm_production_t"] = est * farm_acres / ACRES_PER_HA
    prev = s.pred(fips, year - 1, method)
    if prev:
        e["previous_season_estimate_t_ha"] = prev["pred_t_ha"]
        e["change_vs_previous_estimate_pct"] = 100 * (est - prev["pred_t_ha"]) / prev["pred_t_ha"]
    out["estimate"] = e
    return out


def county_series(s: Store, fips: str, method: str = DEFAULT_METHOD) -> list[dict]:
    cy = s.cy[s.cy["county_fips"] == fips]
    pr = s.preds[(s.preds["county_fips"] == fips) & (s.preds["method"] == method)].set_index("year")
    rows = []
    for _, r in cy.iterrows():
        y = int(r["year"])
        d = {"year": y, "observed_t_ha": clean(r["yield_t_ha"]), "prior5_avg_t_ha": clean(r["prior5_avg_t_ha"]),
             "cdhw_days": clean(r["cdhw_total_days_gs"]), "drought_days": clean(r["spei30_prism_nldas_drought_days_gs"]),
             "heat_days": clean(r["prism_tmax_days_gt35_gs"]), "ppt_mm": clean(r["prism_ppt_total_gs"]),
             "gdd": clean(r["prism_gdd_gs"]), "kdd30": clean(r["prism_kdd30_gs"]), "tmax_mean_c": clean(r["prism_tmax_mean_gs"]),
             "spei30_mean": clean(r["spei30_prism_nldas_mean_gs"]), "et0_mm": clean(r["nldas_et0_total_gs"]),
             "vpd_kpa": clean(r["nldas_vpd_mean_gs"]), "water_balance_mm": clean(r["water_balance_total_gs"]),
             "pred_t_ha": None, "lower_t_ha": None, "upper_t_ha": None, "exposure_class": None}
        if y in pr.index:
            p = pr.loc[y]
            d.update(pred_t_ha=clean(p["pred_t_ha"]), lower_t_ha=clean(p["lower_t_ha"]), upper_t_ha=clean(p["upper_t_ha"]),
                     exposure_class=p["stratum"])
        rows.append(d)
    return rows


def _weighted(df: pd.DataFrame, col: str, w: str = "usda_corn_harvested_acres") -> float | None:
    d = df.dropna(subset=[col, w])
    d = d[d[w] > 0]
    if d.empty:
        return None
    return float(np.average(d[col], weights=d[w]))


def state_summary(s: Store, year: int, method: str = DEFAULT_METHOD) -> list[dict]:
    cy = s.cy[s.cy["year"] == year]
    pr = s.preds[(s.preds["year"] == year) & (s.preds["method"] == method)]
    m = cy.merge(pr[["county_fips", "pred_t_ha", "lower_t_ha", "upper_t_ha", "stratum"]], on="county_fips", how="left")
    out = []
    for st in STATE_NAMES:
        d = m[m["state"] == st]
        dp = d.dropna(subset=["pred_t_ha"])
        acres_p = dp["usda_corn_harvested_acres"].fillna(0)
        rec = {
            "state": st, "state_name": STATE_NAMES[st], "season": year,
            "counties": int(d["county_fips"].nunique()),
            "counties_with_yield": int(d["yield_t_ha"].notna().sum()),
            "counties_with_estimate": int(len(dp)),
            "observed_t_ha_acre_weighted": _weighted(d, "yield_t_ha"),
            "prior5_avg_t_ha_acre_weighted": _weighted(d, "prior5_avg_t_ha"),
            "estimate_t_ha_acre_weighted": _weighted(dp, "pred_t_ha"),
            "lower_t_ha_acre_weighted": _weighted(dp, "lower_t_ha"),
            "upper_t_ha_acre_weighted": _weighted(dp, "upper_t_ha"),
            "production_bu_observed": clean(d["usda_corn_production_bu"].sum(min_count=1)),
            "production_bu_estimated": clean(float((dp["pred_t_ha"] / T_HA_PER_BU_ACRE * acres_p).sum())) if len(dp) else None,
            "harvested_acres": clean(d["usda_corn_harvested_acres"].sum(min_count=1)),
            "counties_compound_high": int((d["cdhw_total_days_gs"] >= 3).sum()),
            "counties_compound_moderate": int(((d["cdhw_total_days_gs"] >= 1) & (d["cdhw_total_days_gs"] < 3)).sum()),
            "mean_drought_days": clean(d["spei30_prism_nldas_drought_days_gs"].mean()),
            "mean_heat_days": clean(d["prism_tmax_days_gt35_gs"].mean()),
            "definition": "Acre-weighted means over the panel counties of the state (weights: USDA harvested acres). "
                          "Not the official NASS state estimate.",
        }
        out.append(rec)
    return out


def state_series(s: Store, method: str = DEFAULT_METHOD) -> list[dict]:
    cy = s.cy
    pr = s.preds[s.preds["method"] == method]
    m = cy.merge(pr[["county_fips", "year", "pred_t_ha", "lower_t_ha", "upper_t_ha"]], on=["county_fips", "year"], how="left")
    rows = []
    for (st, y), d in m.groupby(["state", "year"]):
        dp = d.dropna(subset=["pred_t_ha"])
        rows.append({"state": st, "year": int(y),
                     "observed_t_ha": _weighted(d, "yield_t_ha"),
                     "estimate_t_ha": _weighted(dp, "pred_t_ha") if len(dp) else None,
                     "lower_t_ha": _weighted(dp, "lower_t_ha") if len(dp) else None,
                     "upper_t_ha": _weighted(dp, "upper_t_ha") if len(dp) else None,
                     "production_bu": clean(d["usda_corn_production_bu"].sum(min_count=1)),
                     "compound_days_mean": clean(d["cdhw_total_days_gs"].mean())})
    return rows


RANK_METRICS = {
    "estimate_t_ha": ("Model yield estimate", "t/ha", False),
    "observed_t_ha": ("Observed yield", "t/ha", False),
    "estimate_vs_average_pct": ("Estimate vs 5-season average", "%", True),
    "observed_vs_average_pct": ("Observed vs 5-season average", "%", True),
    "interval_width_t_ha": ("Prediction interval width", "t/ha", False),
    "cdhw_days": ("Compound drought–heat days", "days", False),
    "drought_days": ("Drought days (SPEI-30 < −1)", "days", False),
    "heat_days": ("Days above 35 °C", "days", False),
    "kdd30": ("Killing degree days > 30 °C", "°C·day", False),
    "ppt_mm": ("Season precipitation", "mm", False),
    "hail": ("Hail reports", "events", False),
    "production_bu_estimated": ("Estimated production", "bu", False),
}


def county_table(s: Store, year: int, method: str = DEFAULT_METHOD) -> pd.DataFrame:
    cy = s.cy[s.cy["year"] == year].copy()
    pr = s.preds[(s.preds["year"] == year) & (s.preds["method"] == method)]
    m = cy.merge(pr[["county_fips", "pred_t_ha", "lower_t_ha", "upper_t_ha", "stratum"]], on="county_fips", how="left")
    m["estimate_t_ha"] = m["pred_t_ha"]
    m["observed_t_ha"] = m["yield_t_ha"]
    m["estimate_vs_average_pct"] = 100 * (m["pred_t_ha"] - m["prior5_avg_t_ha"]) / m["prior5_avg_t_ha"]
    m["observed_vs_average_pct"] = 100 * (m["yield_t_ha"] - m["prior5_avg_t_ha"]) / m["prior5_avg_t_ha"]
    m["interval_width_t_ha"] = m["upper_t_ha"] - m["lower_t_ha"]
    m["cdhw_days"] = m["cdhw_total_days_gs"]
    m["drought_days"] = m["spei30_prism_nldas_drought_days_gs"]
    m["heat_days"] = m["prism_tmax_days_gt35_gs"]
    m["kdd30"] = m["prism_kdd30_gs"]
    m["ppt_mm"] = m["prism_ppt_total_gs"]
    m["hail"] = m["noaa_hail_count_gs"]
    m["production_bu_estimated"] = m["pred_t_ha"] / T_HA_PER_BU_ACRE * m["usda_corn_harvested_acres"]
    return m


def rankings(s: Store, year: int, metric: str, method: str = DEFAULT_METHOD, state: str | None = None,
             ascending: bool | None = None, limit: int = 25) -> dict:
    if metric not in RANK_METRICS:
        raise ValueError(f"Unknown metric. Choose one of: {', '.join(RANK_METRICS)}")
    label, unit, _ = RANK_METRICS[metric]
    m = county_table(s, year, method)
    if state:
        m = m[m["state"] == state]
    m = m.dropna(subset=[metric])
    asc = ascending if ascending is not None else False
    m = m.sort_values(metric, ascending=asc)
    rows = [{"rank": i + 1, "fips": r["county_fips"], "county": r["county"], "state": r["state"],
             "value": clean(r[metric]), "estimate_t_ha": clean(r["estimate_t_ha"]),
             "lower_t_ha": clean(r["lower_t_ha"]), "upper_t_ha": clean(r["upper_t_ha"]),
             "observed_t_ha": clean(r["observed_t_ha"]), "cdhw_days": clean(r["cdhw_days"]),
             "drought_days": clean(r["drought_days"]), "heat_days": clean(r["heat_days"])}
            for i, (_, r) in enumerate(m.head(limit).iterrows())]
    return {"season": year, "metric": metric, "label": label, "unit": unit, "order": "ascending" if asc else "descending",
            "counties_ranked": int(len(m)), "rows": rows}


def reliability(s: Store, method: str = DEFAULT_METHOD, fips: str | None = None) -> dict:
    """How often past intervals contained the observed yield (row-weighted, by exposure class)."""
    p = s.preds[s.preds["method"] == method].copy()
    p["inside"] = (p["observed_t_ha"] >= p["lower_t_ha"]) & (p["observed_t_ha"] <= p["upper_t_ha"])
    p["below"] = p["observed_t_ha"] < p["lower_t_ha"]
    out = {"method": method, "method_label": METHODS.get(method, method), "nominal_coverage": NOMINAL,
           "seasons": [int(p["year"].min()), int(p["year"].max())],
           "note": "Share of past county-seasons whose observed yield fell inside the 90% interval. Row-weighted; "
                   "classes with few seasons, or dominated by one season, give uncertain rates.",
           "by_class": []}
    for cls in ["Normal", "Moderate", "Extreme", "ALL"]:
        d = p if cls == "ALL" else p[p["stratum"] == cls]
        out["by_class"].append({"exposure_class": cls, "rows": int(len(d)),
                                "inside_rate": clean(d["inside"].mean()) if len(d) else None,
                                "below_lower_rate": clean(d["below"].mean()) if len(d) else None,
                                "mean_width_t_ha": clean((d["upper_t_ha"] - d["lower_t_ha"]).mean()) if len(d) else None})
    if fips:
        d = p[p["county_fips"] == fips]
        out["county"] = {"fips": fips, "seasons": int(len(d)), "inside": int(d["inside"].sum()),
                         "below_lower": int(d["below"].sum())}
    return out
