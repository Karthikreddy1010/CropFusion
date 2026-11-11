"""Map layers: one value per county for a layer and season."""
from __future__ import annotations

from ..store import DEFAULT_METHOD, Store, clean
from .yields import county_table

LAYERS = {
    "estimate": {"label": "Model yield estimate", "unit": "t/ha", "scale": "sequential", "seasonal": True, "needs_estimate": True},
    "observed": {"label": "Observed yield", "unit": "t/ha", "scale": "sequential", "seasonal": True},
    "anomaly": {"label": "Observed yield vs 5-season average", "unit": "%", "scale": "diverging", "seasonal": True},
    "estimate_anomaly": {"label": "Estimate vs 5-season average", "unit": "%", "scale": "diverging", "seasonal": True, "needs_estimate": True},
    "width": {"label": "Prediction interval width", "unit": "t/ha", "scale": "sequential", "seasonal": True, "needs_estimate": True},
    "drought": {"label": "Drought days (SPEI-30 < −1)", "unit": "days", "scale": "sequential", "seasonal": True},
    "heat": {"label": "Days above 35 °C", "unit": "days", "scale": "sequential", "seasonal": True},
    "compound": {"label": "Compound drought–heat days", "unit": "days", "scale": "classes", "seasonal": True,
                 "classes": [{"label": "Low (0)", "max": 0}, {"label": "Moderate (1–2)", "max": 2}, {"label": "High (≥3)", "max": None}]},
    "compound_severity": {"label": "Compound drought–heat severity", "unit": "SPEI·°C·day", "scale": "sequential", "seasonal": True},
    "storms": {"label": "Hail, wind and tornado reports", "unit": "events", "scale": "sequential", "seasonal": True},
    "ppt": {"label": "Season precipitation", "unit": "mm", "scale": "sequential", "seasonal": True},
    "soil_clay": {"label": "Soil clay content", "unit": "g/kg", "scale": "sequential", "seasonal": False},
    "soil_oc": {"label": "Soil organic carbon", "unit": "dg/kg", "scale": "sequential", "seasonal": False},
    "soil_awc": {"label": "Available water capacity (units unverified)", "unit": "", "scale": "sequential", "seasonal": False},
    "cropland": {"label": "Cropland share of county", "unit": "%", "scale": "sequential", "seasonal": False},
    "forest": {"label": "Forest share of county", "unit": "%", "scale": "sequential", "seasonal": False},
}


def layer_values(s: Store, layer: str, year: int, method: str = DEFAULT_METHOD) -> dict:
    if layer not in LAYERS:
        raise ValueError(f"Unknown layer. Choose one of: {', '.join(LAYERS)}")
    meta = LAYERS[layer]
    if meta["seasonal"]:
        t = county_table(s, year, method).set_index("county_fips")
        col = {
            "estimate": t["estimate_t_ha"], "observed": t["observed_t_ha"], "anomaly": t["observed_vs_average_pct"],
            "estimate_anomaly": t["estimate_vs_average_pct"], "width": t["interval_width_t_ha"],
            "drought": t["drought_days"], "heat": t["heat_days"], "compound": t["cdhw_days"],
            "compound_severity": t["cdhw_cumulative_severity_gs"],
            "storms": t[["noaa_hail_count_gs", "noaa_thunderstorm_wind_count_gs", "noaa_tornado_count_gs"]].sum(axis=1, min_count=1),
            "ppt": t["ppt_mm"],
        }[layer]
    else:
        c = s.counties
        col = {"soil_clay": c["soil_clay"], "soil_oc": c["soil_organic_carbon"], "soil_awc": c["soil_awc"],
               "cropland": 100 * c["nlcd_cropland_frac"], "forest": 100 * c["nlcd_forest_frac"]}[layer]
    values = {f: clean(v) for f, v in col.items() if clean(v) is not None}
    note = None
    if meta.get("needs_estimate") and year not in s.pred_years:
        note = f"Model estimates exist for {s.pred_years[0]}–{s.pred_years[-1]} only."
    return {"layer": layer, **meta, "season": year if meta["seasonal"] else None, "values": values,
            "counties_with_value": len(values), "note": note}
