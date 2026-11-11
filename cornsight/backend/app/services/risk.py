"""Climate risk classification.

Two kinds of classes, always labelled in the API response:
- DOCUMENTED: compound drought-heat classes from the research protocol (code/config.py):
  CDHW days in the growing season: 0 -> Normal, 1-2 -> Moderate, >= 3 -> Extreme.
  Shown as Low / Moderate / High. The protocol defines no fourth class, so "Very High" is never used.
- PROPOSED: drought, heat and rainfall have no documented thresholds in the protocol. The platform
  ranks the season against the county's own 1985-2023 record: <= 50th percentile Low,
  50-75 Moderate, 75-90 High, > 90 Very High. Zero exposure days are always Low.
"""
from __future__ import annotations

LEVELS = ["Low", "Moderate", "High", "Very High"]

CDHW_RULE = ("Documented model classes: 0 compound drought–heat days = Low (model 'Normal'), "
             "1–2 days = Moderate, 3 or more = High (model 'Extreme'). A compound day has "
             "maximum temperature above 35 °C while SPEI-30 is below −1.")
RELATIVE_RULE = ("Proposed scale, not from the model: this season ranked against the same county's "
                 "1985–2023 record. Up to the 50th percentile = Low, 50–75 = Moderate, 75–90 = High, "
                 "above 90 = Very High. Zero exposure is always Low.")


def cdhw_class(days):
    if days is None:
        return None
    if days >= 3:
        return "High"
    if days >= 1:
        return "Moderate"
    return "Low"


def model_stratum(days):
    if days is None:
        return None
    return "Extreme" if days >= 3 else ("Moderate" if days >= 1 else "Normal")


def relative_class(value, pct, zero_is_low: bool = True):
    if value is None or pct is None:
        return None
    if zero_is_low and value == 0:
        return "Low"
    if pct <= 0.50:
        return "Low"
    if pct <= 0.75:
        return "Moderate"
    if pct <= 0.90:
        return "High"
    return "Very High"


def risk_block(row: dict) -> dict:
    """All climate-risk indicators for one county-season row (from Store.row)."""
    if row is None:
        return {}
    drought_days = row.get("spei30_prism_nldas_drought_days_gs")
    heat_days = row.get("prism_tmax_days_gt35_gs")
    rain = row.get("prism_ppt_total_gs")
    cd = row.get("cdhw_total_days_gs")
    return {
        "drought": {
            "level": relative_class(drought_days, row.get("drought_pct")),
            "basis": "proposed", "rule": RELATIVE_RULE,
            "value": drought_days, "unit": "days with SPEI-30 < −1",
            "percentile": row.get("drought_pct"),
            "indices": {
                "SPI-1 (≈30 day) lowest": row.get("spi1_prism_min_gs"),
                "SPI-3 (≈90 day) lowest": row.get("spi3_prism_min_gs"),
                "SPI-6 (≈180 day) lowest": row.get("spi6_prism_min_gs"),
                "SPI-12 (≈365 day) lowest": row.get("spi12_prism_min_gs"),
                "SPEI-30 lowest": row.get("spei30_prism_nldas_min_gs"),
                "SPEI-30 mean": row.get("spei30_prism_nldas_mean_gs"),
                "SPEI-90 lowest": row.get("spei90_prism_nldas_min_gs"),
                "SPEI-90 mean": row.get("spei90_prism_nldas_mean_gs"),
            },
            "severity": row.get("spei30_prism_nldas_drought_severity_gs"),
        },
        "heat": {
            "level": relative_class(heat_days, row.get("heat_pct")),
            "basis": "proposed", "rule": RELATIVE_RULE,
            "value": heat_days, "unit": "days above 35 °C",
            "percentile": row.get("heat_pct"),
            "tmax_mean_c": row.get("prism_tmax_mean_gs"), "tmax_max_c": row.get("prism_tmax_max_gs"),
            "kdd30": row.get("prism_kdd30_gs"), "heat_exceedance": row.get("prism_heat_exceedance_total_gs"),
        },
        "rainfall": {
            "level": relative_class(rain, row.get("rain_pct"), zero_is_low=False),
            "basis": "proposed", "rule": RELATIVE_RULE.replace("Zero exposure is always Low.", "High = wetter than usual."),
            "value": rain, "unit": "mm, 1 Apr–31 Oct", "percentile": row.get("rain_pct"),
            "heavy_rain_reports": row.get("noaa_heavy_rain_count_gs"), "flood_reports": row.get("noaa_flood_count_gs"),
        },
        "compound": {
            "level": cdhw_class(cd), "model_class": model_stratum(cd),
            "basis": "documented", "rule": CDHW_RULE,
            "value": cd, "unit": "compound drought–heat days",
            "episodes": row.get("cdhw_count_gs"), "longest_episode_days": row.get("cdhw_max_duration_gs"),
            "cumulative_severity": row.get("cdhw_cumulative_severity_gs"),
        },
        "storms": {
            "basis": "historical reports", "level": None,
            "hail": row.get("noaa_hail_count_gs"), "thunderstorm_wind": row.get("noaa_thunderstorm_wind_count_gs"),
            "tornado": row.get("noaa_tornado_count_gs"), "max_hail_in": row.get("noaa_max_hail_size_gs"),
            "status": row.get("noaa_storm_data_status"),
            "note": "Counts of NOAA Storm Events reports during 1 Apr–31 Oct. Historical record, not a storm forecast.",
        },
    }
