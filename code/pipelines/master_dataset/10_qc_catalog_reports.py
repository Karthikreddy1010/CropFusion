"""
10_qc_catalog_reports.py -- QC, coverage, leakage, NLDAS completeness, variable catalog, provenance.

Outputs:
  audits/nldas_daily_completeness_report.csv        (final, post-backfill; section 24)
  audits/master_qc_report.csv                       (section 23; PASS/FAIL per check)
  audits/master_coverage_report.csv                 (section 25)
  audits/leakage_audit.csv                          (model-ready predictors)
  metadata/master_variable_catalog.csv              (section 21)
  metadata/master_data_provenance.json              (section 22)
  metadata/legacy_column_mapping.csv                (old MegaDataset name -> new master name)
  audits/qc_summary.json                            (numbers used by the final report)
"""
from __future__ import annotations

import datetime as dt
import json
import re

import numpy as np
import pandas as pd

from md_common import (AUDITS, FULL, INTER, META, MODEL, NLDAS_C_ROOTS, NLDAS_ELEV, NLDAS_G_ROOT, NOAA_COUNTY_YEAR,
                       NOAA_EVENTS, OLD_MEGA, PRISM_RAW, USDA_PROD, USDA_YIELD, COUNTY_GEOJSON, ARCGIS_STATIC, ARCGIS_IOWA_SAND, get_logger, write_json)

log = get_logger("10_qc_catalog_reports")
FULL_P = FULL / "Paper3_MasterDataset_PRISM_NLDAS_NOAA_USDA_FULL_1985_2023.parquet"
MODEL_P = MODEL / "Paper3_MasterDataset_PRISM_NLDAS_CDHWH_CORN_MODEL_1985_2023.parquet"
S = "code/pipelines/master_dataset/"
WINDOWS = {"ann": ("1 Jan-31 Dec", "annual"), "gs": ("1 Apr-31 Oct", "growing season"),
           "veg": ("1 Apr to cumulative GDD<=700 (vegetative / early season)", "vegetative"),
           "silk": ("700<cumulative GDD<=1400 (silking / flowering)", "silking"),
           "grain": ("cumulative GDD>1400 to 31 Oct (grain filling)", "grain fill")}
ID_COLS = ["state", "state_name", "state_fips", "county", "county_code", "county_fips", "year"]


# ----------------------------------------------------------------------------- NLDAS completeness
def nldas_completeness():
    frames = [pd.read_csv(f) for f in sorted((INTER / "nldas_county_daily").glob("nldas_day_status_*.csv"))]
    st = pd.concat(frames, ignore_index=True)
    all_dates = pd.DataFrame({"date": pd.date_range("1985-01-01", "2023-12-31").strftime("%Y-%m-%d")})
    st = all_dates.merge(st, on="date", how="left")
    st["year"] = st.date.str[:4].astype(int)
    st["hours_expected"] = 24
    st["complete_day"] = st.complete_day.fillna(False).astype(bool)
    st["hours_available"] = st.hours_available.fillna(0).astype(int)
    st["missing_hours"] = st.missing_hours.fillna("").astype(str)
    st.loc[st.failed_hours.fillna("").astype(str) != "", "missing_hours"] = st.failed_hours
    rep = st[["date", "year", "hours_available", "hours_expected", "complete_day", "missing_hours"]].copy()
    rep["backfilled_hours"] = st.backfilled_hours.fillna("").astype(str) if "backfilled_hours" in st else ""
    rep.to_csv(AUDITS / "nldas_daily_completeness_report.csv", index=False)
    return rep


# ----------------------------------------------------------------------------- metadata inference
def describe(v: str):
    """Rule-based metadata for every master variable (source, unit, calculation, aggregation...)."""
    d = dict(variable=v, source="", source_path="", description="", unit="", temporal_resolution="county-year",
             spatial_resolution="county", coverage="1985-2023", calculation="", aggregation="",
             static_or_dynamic="DYNAMIC", raw_or_derived="DERIVED", target_or_predictor="PREDICTOR",
             potential_leakage="NO", decision="KEEP", reason="", processing_script="")
    m = re.match(r"(.*)_(ann|gs|veg|silk|grain)$", v)
    base, win = (m.group(1), m.group(2)) if m else (v, None)
    wtxt = WINDOWS[win][0] if win else ""
    prism_src = ("PRISM Climate Group daily AN81d/AN91d 2.5 arc-min (4 km)", str(PRISM_RAW))
    nldas_src = ("NASA GES DISC NLDAS-2 primary forcing NLDAS_FORA0125_H.2.0 (hourly, 0.125 deg) + fixed NASA GSFC elevation",
                 f"{NLDAS_G_ROOT} (1985-2019); {'; '.join(str(p) for p in NLDAS_C_ROOTS.values())}; backfill: master_dataset/intermediate/nldas_backfill")
    county_agg = "county-polygon fractional-area weighted mean (exactextract coverage x cos(lat)) of daily grids"
    def setsrc(src, script):
        d["source"], d["source_path"] = src
        d["processing_script"] = script
    if v in ID_COLS:
        d.update(source="US Census county boundaries / study design", source_path=str(COUNTY_GEOJSON), static_or_dynamic="STATIC" if v != "year" else "DYNAMIC",
                 raw_or_derived="RAW", target_or_predictor="IDENTIFIER", description=f"Identifier: {v}", processing_script=S + "09_build_master.py")
        return d
    if v == "corn_yield_bu_acre":
        d.update(source="USDA NASS Quick Stats county survey: CORN, GRAIN - YIELD, MEASURED IN BU / ACRE", source_path=str(USDA_YIELD),
                 unit="bu/acre", raw_or_derived="RAW", target_or_predictor="TARGET", potential_leakage="TARGET",
                 description="Final published county corn grain yield", calculation="none (not imputed)", aggregation="county-year as published",
                 processing_script=S + "09_build_master.py")
        return d
    if v == "has_corn_yield":
        d.update(source="derived from USDA NASS target", source_path=str(USDA_YIELD), unit="0/1", target_or_predictor="TARGET_FLAG",
                 potential_leakage="TARGET", description="1 if corn_yield_bu_acre observed", calculation="notna(corn_yield_bu_acre)", processing_script=S + "09_build_master.py")
        return d
    if v.startswith("usda_"):
        units = {"usda_yield_cv_pct": "percent", "usda_corn_planted_acres": "acres", "usda_corn_harvested_acres": "acres", "usda_corn_production_bu": "bu"}
        d.update(source="USDA NASS Quick Stats county survey", source_path=str(USDA_YIELD if "cv" in v else USDA_PROD), unit=units.get(v, ""),
                 raw_or_derived="RAW", target_or_predictor="CONTEXT (FULL only)", potential_leakage="YES",
                 description=v.replace("_", " "), reason="Same survey/release as the target (production = yield x harvested acres; CV describes the target estimate).",
                 decision="KEEP (FULL only; excluded from MODEL-READY)", processing_script=S + "09_build_master.py")
        return d
    if base.startswith("prism_"):
        setsrc(prism_src, S + "04_prism_county_daily.py; " + S + "08_county_year_features.py")
        d["spatial_resolution"] = "4 km grid -> county"
        stats = {"tmax_mean": ("mean of daily Tmax", "degC"), "tmax_max": ("max of daily Tmax", "degC"),
                 "tmin_mean": ("mean of daily Tmin", "degC"), "tmin_min": ("min of daily Tmin", "degC"),
                 "ppt_total": ("sum of daily precipitation", "mm"), "dtr_mean": ("mean of daily Tmax-Tmin", "degC"),
                 "gdd": ("sum of daily GDD = max(0,(clip(Tmax,10,30)+clip(Tmin,10,30))/2-10)", "degC day"),
                 "kdd30": ("sum of max(Tmax-30,0)", "degC day"), "tmax_days_gt35": ("count of days Tmax>35 degC", "days"),
                 "heat_exceedance_total": ("sum of max(Tmax-35,0) (heat exceedance)", "degC day"),
                 "valid_days": ("days with valid Tmax, Tmin and ppt in window (QC)", "days")}
        k = base.replace("prism_", "")
        calc, unit = stats.get(k, (k, ""))
        d.update(unit=unit, calculation=f"{calc} over {wtxt}", aggregation=f"daily county PRISM -> {WINDOWS[win][1]}" if win else "",
                 description=f"PRISM {calc}, {wtxt}")
        if k == "valid_days":
            d["target_or_predictor"] = "QC"
    elif base.startswith("nldas_") or base.startswith("water_balance"):
        setsrc(nldas_src, S + "06_nldas_county_daily.py; " + S + "08_county_year_features.py")
        if base.startswith("water_balance"):
            d["source"] = "PRISM ppt - NLDAS-2 FAO-56 ET0"
        d["spatial_resolution"] = "0.125 deg grid -> county"
        stats = {"et0_mean": ("mean daily FAO-56 Penman-Monteith ET0 (validated fao56_et0.py, fixed elevation)", "mm/day"),
                 "et0_total": ("sum of daily FAO-56 ET0", "mm"), "rs_mean": ("mean daily shortwave radiation Rs = mean(SWdown)x0.0864", "MJ m-2 day-1"),
                 "humidity_mean": ("mean daily actual vapour pressure ea = mean(q P/(0.622+0.378q))/1000", "kPa"),
                 "vpd_mean": ("mean daily VPD = es(Tmax,Tmin) - ea (NLDAS)", "kPa"), "wind_mean": ("mean daily 2 m wind u2 = mean(0.748 sqrt(E^2+N^2))", "m s-1"),
                 "pressure_mean": ("mean daily surface pressure", "kPa"), "tair_mean": ("mean daily 2 m air temperature (QC vs PRISM)", "degC"),
                 "lwdown_mean": ("mean downward longwave flux (diagnostic)", "W m-2"), "valid_days": ("days with complete 24/24 h NLDAS in window (QC)", "days"),
                 "total": ("sum of daily P - ET0", "mm")}
        k = base.replace("nldas_", "").replace("water_balance_", "")
        calc, unit = stats.get(k, (k, ""))
        d.update(unit=unit, calculation=f"{calc} over {wtxt}", aggregation=f"hourly -> UTC daily grid -> county -> {WINDOWS[win][1]}" if win else "",
                 description=f"NLDAS-2 {calc}, {wtxt}")
        if k == "valid_days":
            d["target_or_predictor"] = "QC"
    elif base.startswith("spi"):
        setsrc(prism_src, S + "07_spi_spei_cdhw.py; " + S + "08_county_year_features.py")
        n = int(re.match(r"spi(\d+)", base).group(1))
        days = {1: 30, 3: 90, 6: 180, 12: 365}[n]
        stat = base.split("_")[-1]
        d.update(unit="dimensionless (standard normal)", spatial_resolution="4 km grid -> county",
                 calculation=f"SPI-{n}: trailing {days}-day PRISM ppt accumulation; gamma (Thom MLE) with zero-mass mixture, fitted per county x day-of-year slot (+/-7 d) on 1985-2013; Phi^-1; bounded +/-3.09",
                 aggregation=f"daily SPI -> {stat} over {wtxt}", description=f"{stat} of daily SPI-{n} (PRISM), {wtxt}")
        if n == 12 or n == 6:
            d["coverage"] = "1985-2023 (first 364/179 days of 1985 undefined: no pre-1985 PRISM)"
    elif base.startswith("spei"):
        setsrc(("PRISM ppt - NLDAS-2 FAO-56 ET0", prism_src[1] + " ; " + nldas_src[1]), S + "07_spi_spei_cdhw.py; " + S + "08_county_year_features.py")
        n = int(re.match(r"spei(\d+)", base).group(1))
        stat = base.split("prism_nldas_")[-1]
        calc = f"SPEI-{n}: trailing {n}-day accumulation of D = PRISM ppt - NLDAS ET0; log-logistic family fitted as generalized logistic (GLO, unbiased PWM / L-moments, as in the SPEI R package) per county x day-of-year slot (+/-7 d) on 1985-2013; Phi^-1; bounded +/-3.09"
        unit = "dimensionless (standard normal)"
        if stat == "drought_days":
            unit, calc = "days", calc + "; count of days SPEI-30 < -1"
        if stat == "drought_severity":
            unit, calc = "SPEI units x day", calc + "; sum of |SPEI-30| on days SPEI-30 < -1"
        d.update(unit=unit, spatial_resolution="county", calculation=calc, aggregation=f"daily SPEI -> {stat} over {wtxt}",
                 description=f"{stat} of daily SPEI-{n} (PRISM-NLDAS), {wtxt}")
    elif base.startswith("cdhw"):
        setsrc(("PRISM Tmax + PRISM-NLDAS SPEI-30", prism_src[1] + " ; " + nldas_src[1]), S + "07_spi_spei_cdhw.py; " + S + "08_county_year_features.py")
        stat = base.replace("cdhw_", "")
        calcs = {"count": ("number of CDHW episodes (runs of consecutive CDHW days)", "episodes"),
                 "total_days": ("number of CDHW days", "days"), "max_duration": ("longest CDHW episode", "days"),
                 "max_severity": ("max over episodes of sum(|SPEI30| x (Tmax-35))", "SPEI x degC day"),
                 "cumulative_severity": ("sum over days of |SPEI30| x (Tmax-35) on CDHW days (joint severity)", "SPEI x degC day"),
                 "valid_days": ("days with valid SPEI-30 and Tmax (QC)", "days"), "window_days": ("days in window (QC)", "days")}
        calc, unit = calcs.get(stat, (stat, ""))
        d.update(unit=unit, calculation=f"CDHW day: SPEI-30_PRISM-NLDAS < -1.0 AND PRISM Tmax > 35 degC; {calc}; {wtxt}",
                 aggregation=f"daily -> {WINDOWS[win][1]}" if win else "", description=f"CDHW {calc}, {wtxt}")
        if stat in ("valid_days", "window_days"):
            d["target_or_predictor"] = "QC"
    elif base.startswith("pheno_"):
        setsrc(prism_src, S + "08_county_year_features.py")
        d.update(unit="days" if base.endswith("_days") else "day of year",
                 calculation="GDD windows from 1 Apr cumulative PRISM GDD (base 10, cap 30) with project thresholds 700/1400",
                 description=v.replace("_", " "))
    elif base.startswith("noaa_"):
        d.update(source="NOAA NCEI Storm Events Database (processed: data/noaa_storm_pipeline)", source_path=str(NOAA_COUNTY_YEAR if win == "ann" or v == "noaa_storm_data_status" else NOAA_EVENTS),
                 spatial_resolution="county (county events + NWS zone-county crosswalk)", processing_script="code/pipelines/noaa_storm/*; " + S + "09_build_master.py")
        unit = "events"
        if "hail_size" in v:
            unit = "inches"
        elif "wind_mph" in v:
            unit = "mph"
        elif "tornado_scale" in v:
            unit = "F/EF scale 0-5"
        elif "damage" in v:
            unit = "USD (nominal)"
        elif "status" in v:
            unit = "category"
        d.update(unit=unit, calculation=("event counts by EVENT_TYPE with begin date 1 Apr-31 Oct (re-aggregated from cleaned events; annual recount validated against county-year product)"
                                         if win == "gs" else "county-year product (Jan-Dec)"), aggregation=f"events -> county-year ({wtxt or 'Jan-Dec'})",
                 description=v.replace("_", " "))
        if any(t in v for t in ("heavy_rain", "flood", "heat", "drought", "winter", "frost")):
            d["coverage"] = "1996-2023 (event type not recorded 1985-1995; NaN, never 0)"
        if "crop_damage" in v:
            d.update(potential_leakage="YES", reason="Crop damage estimates are post-event loss assessments tied to the yield outcome.")
        elif "damage" in v:
            d.update(potential_leakage="LOW", reason="Heavy-tailed nominal-dollar estimates; inconsistent reporting.")
        elif "drought" in v:
            d.update(potential_leakage="PARTIAL", reason="NOAA drought entries are zone-based impact reports partly informed by agricultural impacts.")
    elif base.startswith("soil_"):
        d.update(source="Soil rasters -> ArcGIS Pro zonal MEAN on TIGER/Line 2023 counties (stated source SSURGO; units match SoilGrids-style conventions; raster source/depth/scale UNVERIFIED)", source_path=str(ARCGIS_STATIC) + "; " + str(OLD_MEGA),
                 static_or_dynamic="STATIC", spatial_resolution="upstream grid -> county", coverage="static (valid for all years)",
                 processing_script=S + "02_audit_old_dataset.py; " + S + "09_build_master.py",
                 calculation="county zonal mean / f_texture, f_texture = (clay+silt+sand)/closure reference of counties with <2% water+dense urban (blank water/urban cells entered the means as 0); missing where f_texture < 0.25")
        units = {"soil_awc": "unverified (median ~356; not SSURGO AWC cm/cm units)", "soil_bulk_density": "cg/cm3", "soil_clay": "g/kg",
                 "soil_silt": "g/kg", "soil_sand": "g/kg", "soil_organic_carbon": "dg/kg", "soil_qc_flag": "category", "soil_valid_fraction": "fraction"}
        if v == "soil_sand":
            d.update(source=d["source"] + "; Iowa filled from the ArcGIS Iowa 'Soil' (sand) zonal table", source_path=str(ARCGIS_IOWA_SAND) + "; " + d["source_path"],
                     reason="FULL only: sand = ~1000 - clay - silt, so clay and silt carry texture in MODEL-READY.")
        d.update(unit=units.get(v, ""), description=v.replace("_", " "))
        if v in ("soil_qc_flag", "soil_valid_fraction"):
            d["target_or_predictor"] = "QC"
    elif v in ("elevation_mean_m", "slope_mean", "elevation_std_m"):
        if v == "elevation_std_m":
            d.update(source="NASA GSFC NLDAS fixed elevation (elevation_m, elevation_std_m, 0.125 deg)", source_path=str(NLDAS_ELEV),
                     unit="m", calculation="sqrt(sum w (sd_cell^2 + (mu_cell - mu_county)^2)); within- plus between-cell variability")
        else:
            d.update(source="USGS 3DEP DEM -> ArcGIS Pro (Surface > Slope in degrees for slope) -> zonal MEAN on TIGER/Line 2023 counties; DEM resolution not recorded", source_path=str(ARCGIS_STATIC) + "; " + str(OLD_MEGA),
                     unit="m" if v == "elevation_mean_m" else "degrees (per ArcGIS workflow; magnitudes consistent)",
                     calculation="old DEM_Mean (validated vs NASA GSFC county mean, r=0.995)" if v == "elevation_mean_m" else "old Slope_Mean")
        d.update(static_or_dynamic="STATIC", coverage="static", spatial_resolution="grid -> county", description=v.replace("_", " "),
                 processing_script=S + "09_build_master.py")
    elif v.startswith("nlcd_"):
        d.update(source="USGS NLCD 30 m class areas -> ArcGIS Pro zonal extraction on TIGER/Line 2023 counties (NLCD year not recorded, single epoch >= 2001)", source_path=str(ARCGIS_STATIC) + "; " + str(OLD_MEGA),
                 static_or_dynamic="STATIC (snapshot)", unit="fraction (0-1)", spatial_resolution="30 m -> county fraction",
                 coverage="one NLCD epoch applied to all years; not year-specific (NLCD has no pre-2001 epoch)",
                 calculation="class fraction" if not any(g in v for g in ("cropland", "pasture_hay", "forest_frac", "grass_shrub", "developed_frac", "water_wetland", "barren")) else "sum of NLCD class fractions (grouped)",
                 potential_leakage="LOW", reason="Static present-day snapshot applied to 1985-2023.", description=v.replace("_", " "),
                 processing_script=S + "02_audit_old_dataset.py; " + S + "09_build_master.py")
    elif v in ("county_lat", "county_lon", "county_land_area_m2", "county_water_area_m2"):
        d.update(source="US Census (old MegaDataset Lat/Lon/ALAND/AWATER)", source_path=str(OLD_MEGA), static_or_dynamic="STATIC", coverage="static",
                 unit="degrees" if "lat" in v or "lon" in v else "m2", raw_or_derived="RAW", description=v.replace("_", " "),
                 processing_script=S + "09_build_master.py")
        if "area" in v:
            d["target_or_predictor"] = "CONTEXT (FULL only)"
    elif v.startswith("oni_") or v.startswith("enso_"):
        d.update(source="NOAA CPC Oceanic Nino Index (old MegaDataset)", source_path=str(OLD_MEGA), spatial_resolution="global index (same for all counties)",
                 unit="degC anomaly" if v.startswith("oni") else "category", raw_or_derived="RAW", description=v.replace("_", " "),
                 processing_script=S + "09_build_master.py", target_or_predictor="CONTEXT (FULL only)")
        if v in ("oni_annual_mean", "oni_annual_max", "oni_annual_min", "enso_phase", "enso_anomalous_year"):
            d.update(potential_leakage="FUTURE_INFO", reason="Annual ONI includes the NDJ season (Jan of year+1), after harvest.")
        if v.startswith("enso_"):
            d["target_or_predictor"] = "EVALUATION STRATIFICATION (FULL only)"
    elif v.startswith("usdm_"):
        d.update(source="US Drought Monitor DSCI (old MegaDataset)", source_path=str(OLD_MEGA), coverage="2005-2023 in source file",
                 unit="DSCI 0-500" if "dsci" in v else "0/1", potential_leakage="PARTIAL", target_or_predictor="VALIDATION / CONTEXT (FULL only)",
                 reason="Expert-blended weekly product partly informed by observed impacts; not available before 2000.", description=v.replace("_", " "),
                 processing_script=S + "09_build_master.py")
    return d


# ----------------------------------------------------------------------------- QC
def qc_checks(full, model, nl):
    rows = []
    def add(check, passed, detail, severity="CRITICAL"):
        rows.append(dict(check=check, status="PASS" if passed else "FAIL", severity=severity, detail=detail))
    for name, df in (("FULL", full), ("MODEL-READY", model)):
        dup = int(df.duplicated(["county_fips", "year"]).sum())
        add(f"{name}: primary key county_fips+year unique", dup == 0, f"duplicates={dup}")
    y = full.corn_yield_bu_acre
    add("USDA: yield >= 0", bool((y.dropna() >= 0).all()), f"min={y.min()}, n={int(y.notna().sum())}")
    add("USDA: no imputed target (model-ready rows == observed yields)", len(model) == int(y.notna().sum()), f"model rows={len(model)}")
    # daily PRISM checks
    viol = nanc = neg = dup_d = n_days = 0
    for f in sorted((INTER / "prism_county_daily").glob("prism_county_daily_*.parquet")):
        p = pd.read_parquet(f)
        viol += int((p.prism_tmax < p.prism_tmin).sum())
        neg += int((p.prism_ppt < 0).sum())
        nanc += int(p[["prism_tmax", "prism_tmin", "prism_ppt"]].isna().sum().sum())
        dup_d += int(p.duplicated(["county_fips", "date"]).sum())
        n_days += p.date.nunique()
    add("PRISM daily: tmax >= tmin", viol == 0, f"violations={viol}")
    add("PRISM daily: ppt >= 0", neg == 0, f"negative={neg}")
    add("PRISM daily: no NaN", nanc == 0, f"NaN={nanc}")
    add("PRISM daily: no duplicate county-date", dup_d == 0, f"duplicates={dup_d}")
    add("PRISM daily: 14,244 dates", n_days == 14244, f"dates={n_days}")
    # NLDAS daily
    et0neg = nlnan = ninf = 0
    for f in sorted((INTER / "nldas_county_daily").glob("nldas_county_daily_*.parquet")):
        n = pd.read_parquet(f)
        num = n.select_dtypes("number")
        et0neg += int((n.nldas_et0 < 0).sum())
        nlnan += int(num.drop(columns=["nldas_rainf_qc"], errors="ignore").isna().sum().sum())
        ninf += int(np.isinf(num.to_numpy()).sum())
    add("NLDAS daily: ET0 >= 0", et0neg == 0, f"negative={et0neg}")
    add("NLDAS daily: no NaN/Inf in aggregated days", nlnan == 0 and ninf == 0, f"NaN={nlnan}, Inf={ninf}")
    inc = int((~nl.complete_day).sum())
    add("NLDAS: all 14,244 dates 24/24 hours after backfill", inc == 0, f"incomplete dates={inc}")
    # indices
    ix = pd.read_parquet(INTER / "indices" / "county_daily_indices_1985_2023.parquet",
                         columns=["date", "spi1_prism", "spi3_prism", "spi6_prism", "spi12_prism", "spei30_prism_nldas", "spei90_prism_nldas"])
    diag = pd.read_csv(AUDITS / "spi_spei_fit_diagnostics.csv")
    add("SPI/SPEI: all county x slot distribution fits valid", int(diag.counties_fit_failed.sum()) == 0,
        f"failed fits={int(diag.counties_fit_failed.sum())} of {int(diag.counties_fit_ok.sum() + diag.counties_fit_failed.sum())}")
    warm = {"spi1_prism": 29, "spi3_prism": 89, "spi6_prism": 179, "spi12_prism": 364, "spei30_prism_nldas": 29, "spei90_prism_nldas": 89}
    first = pd.Timestamp("1985-01-01")
    for c, wd in warm.items():
        v = ix[c].to_numpy()
        after = ix.date >= first + pd.Timedelta(days=wd)
        unexpected = int(np.isnan(v[after.to_numpy()]).sum())
        add(f"{c}: no Inf and no unexpected NaN after warm-up", int(np.isinf(v).sum()) == 0 and unexpected == 0,
            f"Inf={int(np.isinf(v).sum())}, NaN after warm-up={unexpected}, range=[{np.nanmin(v):.2f},{np.nanmax(v):.2f}]")
    del ix
    # CDHW
    for w in ("ann", "gs"):
        td, vd = full[f"cdhw_total_days_{w}"], full[f"cdhw_valid_days_{w}"]
        ok = bool(((td.dropna() >= 0).all()) and ((td <= vd) | td.isna()).all())
        add(f"CDHW {w}: 0 <= cdhw_total_days <= valid days", ok, f"max total={td.max()}, rows NaN={int(td.isna().sum())}")
        add(f"CDHW {w}: max_duration <= total_days and count <= total_days",
            bool(((full[f"cdhw_max_duration_{w}"] <= td) | td.isna()).all() and ((full[f"cdhw_count_{w}"] <= td) | td.isna()).all()), "")
    # NOAA
    cnt = [c for c in full.columns if c.startswith("noaa_") and c.endswith(("_count_ann", "_count_gs"))]
    add("NOAA: event counts >= 0", bool((full[cnt].fillna(0) >= 0).all().all()), f"{len(cnt)} count columns")
    # soil
    s_ok = bool(full.soil_bulk_density.dropna().between(100, 180).all() and full.soil_clay.dropna().between(0, 1000).all()
                and full.soil_silt.dropna().between(0, 1000).all() and (full.soil_clay + full.soil_silt).dropna().le(1000).all())
    add("SOIL: physically valid (BD 1.0-1.8 g/cm3, clay/silt 0-1000 g/kg, clay+silt<=1000)", s_ok,
        f"BD range=[{full.soil_bulk_density.min():.1f},{full.soil_bulk_density.max():.1f}], clay+silt max={(full.soil_clay + full.soil_silt).max():.1f}")
    tex = (full.soil_clay + full.soil_silt + full.soil_sand).dropna()
    sand_cov = full.loc[full.soil_clay.notna(), "soil_sand"].notna().all()
    add("SOIL: sand present wherever clay/silt are; clay+silt+sand within 850-1050 g/kg", bool(sand_cov and tex.between(850, 1050).all()),
        f"texture sum range=[{tex.min():.1f},{tex.max():.1f}], median={tex.median():.1f}; Iowa sand counties={int(full.loc[full.state_fips == '19', 'soil_sand'].notna().groupby(full.county_fips).any().sum())}")
    add("DEM: elevation within 0-700 m for study region", bool(full.elevation_mean_m.between(0, 700).all()),
        f"range=[{full.elevation_mean_m.min():.1f},{full.elevation_mean_m.max():.1f}]")
    lc = [c for c in full.columns if c.startswith("nlcd_") and c.endswith("_frac")]
    grouped = ["nlcd_cropland_frac", "nlcd_pasture_hay_frac", "nlcd_forest_frac", "nlcd_grass_shrub_frac", "nlcd_developed_frac", "nlcd_water_wetland_frac", "nlcd_barren_frac"]
    add("VEGETATION: fractions within [0,1]", bool(full[lc].stack().between(0, 1).all()), f"{len(lc)} columns")
    gsum = full[grouped].sum(axis=1)
    add("VEGETATION: grouped fractions sum to 1", bool(np.allclose(gsum, 1.0, atol=1e-6)), f"range=[{gsum.min():.6f},{gsum.max():.6f}]")
    # ERA5
    era5 = [c for c in full.columns if "era5" in c.lower()] + [c for c in model.columns if "era5" in c.lower()]
    add("No ERA5-derived variable in FULL or MODEL-READY", len(era5) == 0, f"found={era5}")
    add("MODEL-READY: no NaN/Inf in target", bool(np.isfinite(model.corn_yield_bu_acre).all()), "")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- coverage
def coverage(full, nl):
    counties = full.county_fips.nunique()
    src = {
        "USDA": full.corn_yield_bu_acre.notna(),
        "PRISM": full.prism_ppt_total_ann.notna() & full.prism_tmax_mean_ann.notna(),
        "NLDAS": full.nldas_et0_total_gs.notna(),
        "SPI": full.spi1_prism_mean_gs.notna(),
        "SPEI": full.spei30_prism_nldas_mean_gs.notna(),
        "CDHW": full.cdhw_total_days_gs.notna(),
        "NOAA": full.noaa_storm_count_gs.notna(),
        "SOIL": full.soil_awc.notna(),
        "DEM": full.elevation_mean_m.notna(),
        "VEGETATION": full.nlcd_cropland_frac.notna(),
    }
    rows = []
    for name, mask in src.items():
        for y in sorted(full.year.unique()):
            sel = full.year == y
            cy = int((mask & sel).sum())
            rows.append(dict(source=name, year=int(y), counties=int(full.loc[mask & sel, "county_fips"].nunique()),
                             county_years=cy, missing_county_years=int(sel.sum() - cy), coverage_percent=round(100 * cy / sel.sum(), 2)))
        rows.append(dict(source=name, year="ALL", counties=int(full.loc[mask, "county_fips"].nunique()), county_years=int(mask.sum()),
                         missing_county_years=int(len(full) - mask.sum()), coverage_percent=round(100 * mask.mean(), 2)))
    others = lambda n: np.logical_or.reduce([m for k, m in src.items() if k != n])
    for name, mask in src.items():
        only = mask & ~others(name)
        rows.append(dict(source=f"{name}-only (present in {name} but in no other source)", year="ALL", counties=int(full.loc[only, "county_fips"].nunique()),
                         county_years=int(only.sum()), missing_county_years="", coverage_percent=round(100 * only.mean(), 3)))
    inter = np.logical_and.reduce(list(src.values()))
    rows.append(dict(source="COMPLETE INTERSECTION (all sources)", year="ALL", counties=int(full.loc[inter, "county_fips"].nunique()),
                     county_years=int(inter.sum()), missing_county_years=int(len(full) - inter.sum()), coverage_percent=round(100 * inter.mean(), 2)))
    climate_all = np.logical_and.reduce([src[k] for k in ("PRISM", "NLDAS", "SPI", "SPEI", "CDHW")])
    rows.append(dict(source="USDA yield AND all climate sources", year="ALL", counties=int(full.loc[src["USDA"] & climate_all, "county_fips"].nunique()),
                     county_years=int((src["USDA"] & climate_all).sum()), missing_county_years="", coverage_percent=round(100 * (src["USDA"] & climate_all).mean(), 2)))
    cov = pd.DataFrame(rows)
    cov.to_csv(AUDITS / "master_coverage_report.csv", index=False)
    return cov, src, inter


# ----------------------------------------------------------------------------- leakage
def leakage(model, catalog):
    fit = model[model.year.between(1985, 2013)]
    preds = [c for c in model.columns if c not in ID_COLS + ["corn_yield_bu_acre"]]
    cat = catalog.set_index("variable")
    rows = []
    for p in preds:
        r = fit[[p, "corn_yield_bu_acre"]].corr().iloc[0, 1]
        win = re.match(r".*_(ann|gs|veg|silk|grain)$", p)
        timing = {"gs": "within season (<= 31 Oct)", "veg": "within season", "silk": "within season", "grain": "within season (<= 31 Oct)",
                  "ann": "includes Nov-Dec (post-harvest)"}.get(win.group(1) if win else "", "static / pre-season")
        rows.append(dict(predictor=p, source=cat.loc[p, "source"], temporal_window=timing,
                         derived_from_target_or_same_survey="NO", uses_post_harvest_information="YES" if "post-harvest" in timing else "NO",
                         uses_test_period_information_in_construction="NO (SPI/SPEI calibrated on 1985-2013)" if p.startswith(("spi", "spei", "cdhw")) else "NO",
                         pearson_r_with_target_fit_years=round(float(r), 3) if np.isfinite(r) else np.nan,
                         suspicious_abs_r_gt_0_9=bool(np.isfinite(r) and abs(r) > 0.9),
                         catalog_leakage_flag=cat.loc[p, "potential_leakage"]))
    lk = pd.DataFrame(rows)
    lk.to_csv(AUDITS / "leakage_audit.csv", index=False)
    return lk


def new_vs_old_validation(full):
    """VALIDATION ONLY: compare new PRISM/NLDAS products with the old ERA5-era variables (never merged)."""
    old = pd.read_parquet(INTER / "validation" / "old_era5_county_year_validation.parquet")
    m = full.merge(old, on=["county_fips", "year"], how="inner")
    pairs = [("corn_yield_bu_acre", "Corn_Yield_buacre", "USDA yield new vs old extract"),
             ("prism_ppt_total_gs", "Precip_growseason_mm", "PRISM vs ERA5 growing-season precipitation"),
             ("nldas_et0_total_gs", "ET0_growseason_mm", "NLDAS FAO-56 vs ERA5 growing-season ET0"),
             ("water_balance_total_gs", "WaterBalance_growseason_mm", "PRISM-NLDAS vs ERA5 water balance"),
             ("prism_gdd_gs", "GDD_Accumulated", "PRISM vs ERA5 GDD"),
             ("prism_tmax_mean_gs", "ERA5d_Tmax_mean_C", "PRISM vs ERA5 mean Tmax (growing season)"),
             ("prism_tmax_days_gt35_gs", "Tmax_Days_Above_35", "PRISM vs ERA5 days Tmax>35"),
             ("spei30_prism_nldas_min_gs", "SPEI_30_min", "PRISM-NLDAS vs ERA5 SPEI-30 min"),
             ("spei30_prism_nldas_mean_gs", "SPEI_30_mean", "PRISM-NLDAS vs ERA5 SPEI-30 mean"),
             ("cdhw_total_days_gs", "CDHW_Event_Count", "PRISM-NLDAS CDHW days vs ERA5 CDHW count"),
             ("cdhw_cumulative_severity_gs", "CDHW_Severity_Score", "PRISM-NLDAS vs ERA5 CDHW severity"),
             ("spei30_prism_nldas_mean_gs", "DSCI_growseason_mean", "SPEI-30 mean vs USDM DSCI (expect negative r)"),
             ("spei90_prism_nldas_min_gs", "DSCI_growseason_max", "SPEI-90 min vs USDM DSCI max (expect negative r)")]
    rows = []
    for new, oldc, label in pairs:
        if new not in m or oldc not in m:
            continue
        d = m[[new, oldc]].dropna()
        rows.append(dict(comparison=label, new_variable=new, old_variable=oldc, n=len(d),
                         pearson_r=round(float(d.corr().iloc[0, 1]), 3) if len(d) > 2 else np.nan,
                         spearman_r=round(float(d.corr(method="spearman").iloc[0, 1]), 3) if len(d) > 2 else np.nan,
                         new_mean=round(float(d[new].mean()), 3), old_mean=round(float(d[oldc].mean()), 3)))
    out = pd.DataFrame(rows)
    out.to_csv(AUDITS / "new_vs_old_era5_validation.csv", index=False)
    return out


def legacy_mapping():
    a = pd.read_csv(AUDITS / "old_dataset_variable_audit.csv")
    a[["variable", "decision", "era5_derived", "new_master_variable", "reason"]].rename(columns={"variable": "old_variable"}).to_csv(
        META / "legacy_column_mapping.csv", index=False)


def main():
    nl = nldas_completeness()
    full = pd.read_parquet(FULL_P)
    model = pd.read_parquet(MODEL_P)
    mr_cols = set(model.columns)
    cat = pd.DataFrame([describe(v) for v in full.columns])
    cat["missing_percent"] = [round(100 * full[v].isna().mean(), 3) for v in full.columns]
    cat["missing_percent_model_ready"] = [round(100 * model[v].isna().mean(), 3) if v in mr_cols else "" for v in full.columns]
    cat["in_model_ready"] = ["YES" if v in mr_cols else "NO" for v in full.columns]
    sel = json.loads((INTER / "model_ready_selection.json").read_text())
    for i, r in cat.iterrows():
        v = r.variable
        if r.decision == "KEEP" and not r.reason:
            if v in mr_cols:
                cat.at[i, "reason"] = "Scientifically valid; selected for MODEL-READY."
            elif v in sel["dropped_as_near_duplicate"]:
                cat.at[i, "reason"] = "Valid; FULL only -- near-duplicate (|r|>=0.98 on FIT years) of a retained model-ready predictor."
            elif v.endswith("_ann"):
                cat.at[i, "reason"] = "Valid; FULL only -- annual window includes post-harvest Nov-Dec."
            elif r.target_or_predictor in ("QC",):
                cat.at[i, "reason"] = "QC / coverage diagnostic; FULL only."
            else:
                cat.at[i, "reason"] = "Valid; FULL only (redundant with or less specific than a model-ready variable, or context)."
        if v not in mr_cols and r.decision == "KEEP":
            cat.at[i, "decision"] = "KEEP (FULL only)"
    cols = ["variable", "source", "source_path", "description", "unit", "temporal_resolution", "spatial_resolution", "coverage",
            "calculation", "aggregation", "static_or_dynamic", "raw_or_derived", "target_or_predictor", "potential_leakage",
            "missing_percent", "decision", "reason", "in_model_ready", "missing_percent_model_ready", "processing_script"]
    cat[cols].to_csv(META / "master_variable_catalog.csv", index=False)

    prov = {"generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "datasets": {"FULL": str(FULL_P), "MODEL_READY": str(MODEL_P)},
            "reproduce_from": "Paper3 project folder",
            "reproduce": [f"python {S}{step}" for step in (
                "01_audit_sources.py", "02_audit_old_dataset.py", "03_build_county_weights.py nldas",
                "03_build_county_weights.py prism", "04_prism_county_daily.py", "06_nldas_county_daily.py",
                "05_nldas_backfill.py", "06_nldas_county_daily.py --redo-incomplete --years <years with gaps>",
                "05_nldas_backfill.py --from-status --years <years with corrupt granules>",
                "06_nldas_county_daily.py --redo-incomplete --years <same years>",
                "07_spi_spei_cdhw.py", "08a_gdd_definition_check.py", "08_county_year_features.py", "09_build_master.py",
                "10_qc_catalog_reports.py", "11_final_report.py")],
            "variables": {}}
    for _, r in cat.iterrows():
        prov["variables"][r.variable] = dict(source=r.source, original_file=r.source_path.split(";")[0].strip(), original_path=r.source_path,
                                             processing_script=r.processing_script, calculation=r.calculation, aggregation=r.aggregation,
                                             units=r.unit, temporal_coverage=r.coverage, spatial_coverage="583 counties: IL, IN, IA, MN, MO, OH",
                                             spatial_resolution=r.spatial_resolution, in_model_ready=r.in_model_ready)
    write_json(prov, META / "master_data_provenance.json")

    qc = qc_checks(full, model, nl)
    qc.to_csv(AUDITS / "master_qc_report.csv", index=False)
    cov, src, inter = coverage(full, nl)
    lk = leakage(model, cat)
    val = new_vs_old_validation(full)
    log.info("New vs old (validation only):\n%s", val.to_string(index=False))
    legacy_mapping()
    summ = dict(
        full_shape=list(full.shape), model_shape=list(model.shape), counties=int(full.county_fips.nunique()),
        county_years=len(full), usda_obs=int(full.corn_yield_bu_acre.notna().sum()),
        prism_county_days=int(full.prism_valid_days_ann.sum()), nldas_complete_days=int(nl.complete_day.sum()),
        nldas_incomplete_days=int((~nl.complete_day).sum()),
        nldas_incomplete_dates=nl.loc[~nl.complete_day, "date"].tolist(),
        noaa_county_years=int(full.noaa_storm_count_ann.notna().sum()),
        qc_fail=qc.loc[qc.status == "FAIL", "check"].tolist(), qc_pass=int((qc.status == "PASS").sum()), qc_total=len(qc),
        complete_intersection=int(inter.sum()), leakage_suspicious=lk.loc[lk.suspicious_abs_r_gt_0_9, "predictor"].tolist(),
        post_harvest_predictors_in_model=lk.loc[lk.uses_post_harvest_information == "YES", "predictor"].tolist(),
        years_with_complete_usda=[int(y) for y, g in full.groupby("year") if g.corn_yield_bu_acre.notna().all()],
        usda_by_year=full.groupby("year").corn_yield_bu_acre.count().astype(int).to_dict(),
        era5_columns=[c for c in full.columns if "era5" in c.lower()],
    )
    write_json(summ, AUDITS / "qc_summary.json")
    log.info("QC: %d/%d PASS; FAIL=%s", summ["qc_pass"], summ["qc_total"], summ["qc_fail"])
    log.info("Catalog: %d variables (%d in model-ready)", len(cat), int((cat.in_model_ready == "YES").sum()))


if __name__ == "__main__":
    main()
