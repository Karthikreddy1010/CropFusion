"""County profile, agricultural conditions and growth-stage (phenology proxy) analysis."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from ..store import Store, clean

STAGE_NOTE = ("Growth stages are modelled from accumulated growing degree days since 1 April "
              "(vegetative up to 700 GDD, silking 700–1400, grain filling above 1400 until 31 October). "
              "They approximate crop development; they are not observed planting or silking dates.")
STAGES = [("veg", "Vegetative"), ("silk", "Silking"), ("grain", "Grain filling")]


def doy_to_date(year: int, doy) -> str | None:
    if doy is None:
        return None
    return (date(year, 1, 1) + timedelta(days=int(doy) - 1)).isoformat()


def county_info(s: Store, fips: str) -> dict:
    r = s.counties.loc[fips]
    years = s.cy[(s.cy["county_fips"] == fips) & s.cy["yield_t_ha"].notna()]["year"]
    return {
        "fips": fips, "county": r["county"], "state": r["state"], "state_name": r["state_name"],
        "label": s.county_label(fips), "lat": clean(r["county_lat"]), "lon": clean(r["county_lon"]),
        "yield_years": [int(years.min()), int(years.max())] if len(years) else None,
        "yield_seasons": int(len(years)),
        "has_estimates": bool((s.preds["county_fips"] == fips).any()),
        "soil": {
            "available_water_capacity": {"value": clean(r["soil_awc"]), "unit": "units not verified in source"},
            "bulk_density": {"value": clean(r["soil_bulk_density"]), "unit": "cg/cm³"},
            "clay": {"value": clean(r["soil_clay"]), "unit": "g/kg"},
            "silt": {"value": clean(r["soil_silt"]), "unit": "g/kg"},
            "sand": {"value": clean(r["soil_sand"]), "unit": "g/kg"},
            "organic_carbon": {"value": clean(r["soil_organic_carbon"]), "unit": "dg/kg"},
        },
        "terrain": {"elevation_m": clean(r["elevation_mean_m"]), "elevation_sd_m": clean(r["elevation_std_m"]),
                    "slope_deg": clean(r["slope_mean"])},
        "land_cover": {k: clean(r[f"nlcd_{k}_frac"]) for k in
                       ["cropland", "pasture_hay", "forest", "grass_shrub", "developed", "water_wetland"]},
    }


def conditions(s: Store, fips: str, year: int) -> dict:
    """Season conditions compared with the county's own 1985-2023 record."""
    row = s.row(fips, year)
    if row is None:
        return {"available": False}
    hist = s.cy[s.cy["county_fips"] == fips]
    out = {"available": True, "season": year, "period": "1 April – 31 October", "items": []}
    for col, label, unit in [
        ("prism_ppt_total_gs", "Season precipitation", "mm"),
        ("prism_tmax_mean_gs", "Mean daily maximum temperature", "°C"),
        ("prism_tmin_mean_gs", "Mean daily minimum temperature", "°C"),
        ("prism_gdd_gs", "Growing degree days", "°C·day"),
        ("prism_kdd30_gs", "Killing degree days > 30 °C", "°C·day"),
        ("prism_tmax_days_gt35_gs", "Days above 35 °C", "days"),
        ("nldas_et0_total_gs", "Reference evapotranspiration", "mm"),
        ("water_balance_total_gs", "Water balance (P − ET₀)", "mm"),
        ("nldas_vpd_mean_gs", "Mean vapour-pressure deficit", "kPa"),
        ("spei90_prism_nldas_mean_gs", "Mean SPEI-90 (water availability)", "index"),
    ]:
        v = row.get(col)
        h = hist[col].dropna()
        out["items"].append({
            "key": col, "label": label, "unit": unit, "value": v,
            "county_mean_1985_2023": clean(h.mean()) if len(h) else None,
            "difference": (v - h.mean()) if (v is not None and len(h)) else None,
            "percentile": clean((h <= v).mean()) if (v is not None and len(h)) else None,
        })
    return out


def stages(s: Store, fips: str, year: int) -> dict:
    row = s.row(fips, year)
    if row is None:
        return {"available": False, "note": STAGE_NOTE}
    st = []
    for k, label in STAGES:
        st.append({
            "stage": k, "label": label,
            "start": doy_to_date(year, row.get(f"pheno_{k}_start_doy")),
            "end": doy_to_date(year, row.get(f"pheno_{k}_end_doy")),
            "days": row.get(f"pheno_{k}_days"),
            "tmax_mean_c": row.get(f"prism_tmax_mean_{k}"),
            "days_above_35c": row.get(f"prism_tmax_days_gt35_{k}"),
            "ppt_mm": row.get(f"prism_ppt_total_{k}"),
            "et0_mm": row.get(f"nldas_et0_total_{k}"),
            "water_balance_mm": row.get(f"water_balance_total_{k}"),
            "vpd_kpa": row.get(f"nldas_vpd_mean_{k}"),
            "spei30_mean": row.get(f"spei30_prism_nldas_mean_{k}"),
            "spei30_min": row.get(f"spei30_prism_nldas_min_{k}"),
            "spi1_min": row.get(f"spi1_prism_min_{k}"),
            "cdhw_days": row.get(f"cdhw_total_days_{k}"),
            "cdhw_episodes": row.get(f"cdhw_count_{k}"),
            "cdhw_severity": row.get(f"cdhw_cumulative_severity_{k}"),
        })

    def arg(key, best):
        vals = [(x[key], x["label"]) for x in st if x[key] is not None]
        if not vals:
            return None
        v, lab = best(vals)
        return {"stage": lab, "value": v}

    most = {
        "compound_days": arg("cdhw_days", max),
        "heat_days": arg("days_above_35c", max),
        "driest_spei30": arg("spei30_mean", min),
        "highest_vpd": arg("vpd_kpa", max),
    }
    if most["compound_days"] and not most["compound_days"]["value"]:
        most["compound_days"] = None
    if most["heat_days"] and not most["heat_days"]["value"]:
        most["heat_days"] = None
    return {"available": True, "season": year, "note": STAGE_NOTE, "stages": st, "most_stressed": most}


def stage_history(s: Store, fips: str) -> list[dict]:
    d = s.cy[s.cy["county_fips"] == fips]
    rows = []
    for _, r in d.iterrows():
        rec = {"year": int(r["year"]), "observed_t_ha": clean(r["yield_t_ha"])}
        for k, _ in STAGES:
            rec[f"{k}_cdhw_days"] = clean(r[f"cdhw_total_days_{k}"])
            rec[f"{k}_heat_days"] = clean(r[f"prism_tmax_days_gt35_{k}"])
            rec[f"{k}_spei30"] = clean(r[f"spei30_prism_nldas_mean_{k}"])
            rec[f"{k}_ppt_mm"] = clean(r[f"prism_ppt_total_{k}"])
        rows.append(rec)
    return rows


def compound_yield_relation(s: Store, state: str | None = None) -> dict:
    """Observed yield relative to the prior 5-season average, by compound drought-heat class (all seasons)."""
    d = s.cy.dropna(subset=["yield_t_ha", "prior5_avg_t_ha", "cdhw_total_days_gs"]).copy()
    if state:
        d = d[d["state"] == state]
    d["dev_pct"] = 100 * (d["yield_t_ha"] - d["prior5_avg_t_ha"]) / d["prior5_avg_t_ha"]
    d["cls"] = pd.cut(d["cdhw_total_days_gs"], [-0.5, 0.5, 2.5, 1e9], labels=["Low (0 days)", "Moderate (1–2)", "High (≥3)"])
    out = []
    for cls, g in d.groupby("cls", observed=True):
        out.append({"class": str(cls), "county_seasons": int(len(g)), "median_dev_pct": clean(g["dev_pct"].median()),
                    "p25_dev_pct": clean(g["dev_pct"].quantile(.25)), "p75_dev_pct": clean(g["dev_pct"].quantile(.75)),
                    "share_below_avg_10pct": clean((g["dev_pct"] < -10).mean())})
    return {"state": state, "seasons": [int(d["year"].min()), int(d["year"].max())], "classes": out,
            "note": "Observed yield vs the county's own prior five-season average. Describes association in the "
                    "historical record; it does not isolate the effect of compound drought–heat from other factors."}
