"""
09_build_master.py -- sequential, audited construction of the FULL and MODEL-READY master datasets.

Join order (key = state_fips + county_fips + year; every table unique on county_fips + year; left joins
onto the county-year skeleton; any row-count change raises and stops the build):
  0  skeleton (583 study counties x 39 years, US Census county geometry list)
  1  USDA NASS corn grain yield (target)            6  CDHW (PRISM-NLDAS)
  2  PRISM county-year features                     7  phenology-window features
  3  NLDAS / FAO-56 ET0 features                    8  NOAA Storm Events county-year (+ growing season)
  4  SPI (PRISM)                                    9  soil (dilution-corrected)
  5  SPEI (PRISM-NLDAS)                            10  DEM / elevation
                                                   11  vegetation / land cover (NLCD static)
  then context: USDA acreage/production (FULL only), ENSO/ONI + USDM (FULL only).

Outputs:
  full/Paper3_MasterDataset_PRISM_NLDAS_NOAA_USDA_FULL_1985_2023.parquet
  model_ready/Paper3_MasterDataset_PRISM_NLDAS_CDHWH_CORN_MODEL_1985_2023.parquet
  audits/master_merge_audit.csv, audits/noaa_gs_recount_validation.csv,
  audits/dem_validation_vs_nasa_gsfc.json, audits/model_ready_redundancy_check.csv
  intermediate/validation/master_validation_only_variables.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from md_common import (AUDITS, FULL, INTER, MODEL, NLDAS_ELEV, NOAA_COUNTY_YEAR, NOAA_EVENTS, OLD_MEGA, USDA_PROD,
                       USDA_YIELD, YEARS, MergeAuditor, assert_unique_key, get_logger, load_study_counties, write_json)

log = get_logger("09_build_master")
CY = INTER / "county_year"
FULL_NAME = "Paper3_MasterDataset_PRISM_NLDAS_NOAA_USDA_FULL_1985_2023.parquet"
MODEL_NAME = "Paper3_MasterDataset_PRISM_NLDAS_CDHWH_CORN_MODEL_1985_2023.parquet"
FIT_YEARS = (1985, 2013)
KEYS = ["state_fips", "county_fips", "year"]


def skeleton():
    c = load_study_counties().drop(columns="geometry")
    c["county_code"] = c.county_fips.str[2:]
    sk = c.merge(pd.DataFrame({"year": YEARS}), how="cross")
    sk = sk[["state", "state_name", "state_fips", "county", "county_code", "county_fips", "year"]]
    assert_unique_key(sk, ["county_fips", "year"], "skeleton")
    return sk


def with_state(df):
    df = df.copy()
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    df["state_fips"] = df["county_fips"].str[:2]
    df["year"] = df["year"].astype(int)
    return df


def usda():
    y = pd.read_parquet(USDA_YIELD)
    y = with_state(y.rename(columns={"yield_cv_pct": "usda_yield_cv_pct"}))
    y = y[["state_fips", "county_fips", "year", "corn_yield_bu_acre", "usda_yield_cv_pct"]]
    assert_unique_key(y, ["county_fips", "year"], "usda_yield")
    return y


def usda_context():
    p = with_state(pd.read_parquet(USDA_PROD))
    p = p.rename(columns={"planted_acres": "usda_corn_planted_acres", "harvested_acres": "usda_corn_harvested_acres",
                          "production_bu": "usda_corn_production_bu"})
    p = p[["state_fips", "county_fips", "year", "usda_corn_planted_acres", "usda_corn_harvested_acres", "usda_corn_production_bu"]]
    assert_unique_key(p, ["county_fips", "year"], "usda_production")
    return p


def feature_table(name):
    return with_state(pd.read_parquet(CY / f"{name}.parquet"))


# ----------------------------------------------------------------------------- NOAA
NOAA_ANN_MAP = {
    "storm_event_count": "noaa_storm_count_ann", "severe_convective_event_count": "noaa_severe_convective_count_ann",
    "hail_event_count": "noaa_hail_count_ann", "thunderstorm_wind_event_count": "noaa_thunderstorm_wind_count_ann",
    "tornado_event_count": "noaa_tornado_count_ann", "lightning_event_count": "noaa_lightning_count_ann",
    "heavy_rain_event_count": "noaa_heavy_rain_count_ann", "flood_event_count": "noaa_flood_count_ann",
    "flash_flood_event_count": "noaa_flash_flood_count_ann", "high_wind_event_count": "noaa_high_wind_count_ann",
    "strong_wind_event_count": "noaa_strong_wind_count_ann", "heat_event_count": "noaa_heat_count_ann",
    "excessive_heat_event_count": "noaa_excessive_heat_count_ann", "drought_event_count": "noaa_drought_count_ann",
    "winter_event_count": "noaa_winter_count_ann", "frost_freeze_event_count": "noaa_frost_freeze_count_ann",
    "maximum_hail_size": "noaa_max_hail_size_ann", "maximum_wind_magnitude": "noaa_max_wind_mph_ann",
    "maximum_tornado_scale": "noaa_max_tornado_scale_ann", "total_damage_property_usd": "noaa_property_damage_usd_ann",
    "total_damage_crops_usd": "noaa_crop_damage_usd_ann", "storm_data_status": "noaa_storm_data_status",
}
GS_TYPES = {"noaa_storm_count": None, "noaa_hail_count": ["Hail"], "noaa_thunderstorm_wind_count": ["Thunderstorm Wind"],
            "noaa_tornado_count": ["Tornado"], "noaa_high_wind_count": ["High Wind"], "noaa_heavy_rain_count": ["Heavy Rain"],
            "noaa_flood_count": ["Flood"], "noaa_flash_flood_count": ["Flash Flood"],
            "noaa_excessive_heat_count": ["Excessive Heat"], "noaa_drought_count": ["Drought"]}
PRE1996_UNTRACKED = {"noaa_heavy_rain_count", "noaa_flood_count", "noaa_flash_flood_count", "noaa_excessive_heat_count",
                     "noaa_drought_count"}


def noaa(counties):
    cy = pd.read_parquet(NOAA_COUNTY_YEAR)
    cy = with_state(cy.rename(columns={"GEOID": "county_fips", "Year": "year"}))
    cy = cy[cy.county_fips.isin(counties)]
    ann = cy[["state_fips", "county_fips", "year"] + list(NOAA_ANN_MAP)].rename(columns=NOAA_ANN_MAP)
    assert_unique_key(ann, ["county_fips", "year"], "noaa_ann")

    ev = pd.read_parquet(NOAA_EVENTS, columns=["EVENT_ID", "YEAR", "EVENT_TYPE", "BEGIN_DATE_TIME", "GEOID", "MAGNITUDE",
                                                "damage_property_usd", "damage_crops_usd"])
    ev = ev[ev.GEOID.notna()].copy()
    ev["county_fips"] = ev.GEOID.astype(str).str.zfill(5)
    ev = ev[ev.county_fips.isin(counties)]
    ev["year"] = ev.YEAR.astype(int)
    ev["begin"] = pd.to_datetime(ev.BEGIN_DATE_TIME, format="%d-%b-%y %H:%M:%S", errors="coerce")
    bad_dates = int(ev.begin.isna().sum())
    ev["month"] = ev.begin.dt.month
    # validation: annual recount from events must reproduce the county-year product
    rec = ev.groupby(["county_fips", "year"]).agg(storm=("EVENT_ID", "count"),
                                                  hail=("EVENT_TYPE", lambda s: (s == "Hail").sum()),
                                                  tornado=("EVENT_TYPE", lambda s: (s == "Tornado").sum())).reset_index()
    chk = ann.merge(rec, on=["county_fips", "year"], how="left").fillna({"storm": 0, "hail": 0, "tornado": 0})
    val = pd.DataFrame([dict(check=k, mismatching_county_years=int((chk[a] != chk[b]).sum()), county_years=len(chk))
                        for k, a, b in [("storm_count", "noaa_storm_count_ann", "storm"), ("hail_count", "noaa_hail_count_ann", "hail"),
                                        ("tornado_count", "noaa_tornado_count_ann", "tornado")]])
    val["unparseable_begin_dates"] = bad_dates
    val.to_csv(AUDITS / "noaa_gs_recount_validation.csv", index=False)
    log.info("NOAA annual recount validation:\n%s", val.to_string(index=False))

    gsev = ev[ev.month.between(4, 10)]
    base = ann[["state_fips", "county_fips", "year"]].copy()
    for name, types in GS_TYPES.items():
        sub = gsev if types is None else gsev[gsev.EVENT_TYPE.isin(types)]
        cnt = sub.groupby(["county_fips", "year"]).size().rename(f"{name}_gs")
        base = base.merge(cnt, on=["county_fips", "year"], how="left")
        base[f"{name}_gs"] = base[f"{name}_gs"].fillna(0.0)
        if name in PRE1996_UNTRACKED:
            base.loc[base.year < 1996, f"{name}_gs"] = np.nan
    hail = gsev[gsev.EVENT_TYPE == "Hail"].groupby(["county_fips", "year"]).MAGNITUDE.max().rename("noaa_max_hail_size_gs")
    base = base.merge(hail, on=["county_fips", "year"], how="left")
    base.loc[base.noaa_hail_count_gs == 0, "noaa_max_hail_size_gs"] = 0.0
    dmg = gsev.groupby(["county_fips", "year"]).agg(noaa_property_damage_usd_gs=("damage_property_usd", lambda s: s.sum(min_count=1)),
                                                    noaa_crop_damage_usd_gs=("damage_crops_usd", lambda s: s.sum(min_count=1)))
    base = base.merge(dmg, on=["county_fips", "year"], how="left")
    out = ann.merge(base, on=["state_fips", "county_fips", "year"], how="left", validate="one_to_one")
    assert_unique_key(out, ["county_fips", "year"], "noaa")
    return out


# ----------------------------------------------------------------------------- static
def static_tables(counties):
    st = pd.read_parquet(INTER / "static" / "old_static_county_variables.parquet")
    st = st[st.county_fips.isin(counties)].copy()
    # NASA GSFC NLDAS fixed elevation -> county mean / std (between + within cell) for DEM validation & std
    e = xr.open_dataset(NLDAS_ELEV)
    em, es = e.elevation_m.values.astype(float), e.elevation_std_m.values.astype(float)
    ncol = em.shape[1]
    e.close()
    w = pd.read_parquet(INTER / "weights" / "nldas_county_weights.parquet")
    w["mu"] = em[w.lat_idx, w.lon_idx]
    w["sd"] = es[w.lat_idx, w.lon_idx]
    def agg(g):
        wt = g.weight / g.weight.sum()
        mu = (wt * g.mu).sum()
        var = (wt * (g.sd ** 2 + (g.mu - mu) ** 2)).sum()
        return pd.Series({"nasa_gsfc_elevation_mean_m": mu, "elevation_std_m": np.sqrt(var)})
    ne = w.groupby("county_fips")[["weight", "mu", "sd"]].apply(agg).reset_index()
    st = st.merge(ne, on="county_fips", how="left", validate="one_to_one")
    r = float(np.corrcoef(st.elevation_mean_m, st.nasa_gsfc_elevation_mean_m)[0, 1])
    dem_val = dict(pearson_r_old_DEM_Mean_vs_NASA_GSFC=r,
                   mean_bias_m=float((st.elevation_mean_m - st.nasa_gsfc_elevation_mean_m).mean()),
                   mae_m=float((st.elevation_mean_m - st.nasa_gsfc_elevation_mean_m).abs().mean()),
                   max_abs_diff_m=float((st.elevation_mean_m - st.nasa_gsfc_elevation_mean_m).abs().max()),
                   corr_slope_vs_elevation_std=float(np.corrcoef(st.slope_mean, st.elevation_std_m)[0, 1]),
                   decision="elevation_mean_m (old DEM_Mean) retained if r >= 0.95")
    write_json(dem_val, AUDITS / "dem_validation_vs_nasa_gsfc.json")
    log.info("DEM validation: %s", dem_val)
    if r < 0.95:
        raise AssertionError(f"Old DEM_Mean fails validation against NASA GSFC elevation (r={r:.3f})")
    st["state_fips"] = st.county_fips.str[:2]
    soil = st[["state_fips", "county_fips", "soil_awc", "soil_bulk_density", "soil_clay", "soil_silt",
               "soil_organic_carbon", "soil_sand", "soil_qc_flag", "soil_valid_fraction"]]
    dem = st[["state_fips", "county_fips", "elevation_mean_m", "elevation_std_m", "slope_mean"]]
    lc_cols = [c for c in st.columns if c.startswith("nlcd_")]
    veg = st[["state_fips", "county_fips", "county_lat", "county_lon", "county_land_area_m2", "county_water_area_m2"] + lc_cols].copy()
    veg["nlcd_cropland_frac"] = st.nlcd_cultivatedcrops_frac
    veg["nlcd_pasture_hay_frac"] = st.nlcd_pasture_frac
    veg["nlcd_forest_frac"] = st.nlcd_deciduousforest_frac + st.nlcd_evergreenforest_frac + st.nlcd_mixedforest_frac
    veg["nlcd_grass_shrub_frac"] = st.nlcd_grassland_frac + st.nlcd_shrub_frac
    veg["nlcd_developed_frac"] = (st.nlcd_developed_openspace_frac + st.nlcd_developed_lowintensity_frac +
                                  st.nlcd_developed_mediumintensity_frac + st.nlcd_developed_highintensity_frac)
    veg["nlcd_water_wetland_frac"] = st.nlcd_open_water_frac + st.nlcd_woodywetlands_frac + st.nlcd_herbaceouswetlands_frac
    veg["nlcd_barren_frac"] = st.nlcd_barrenland_frac
    validation_static = st[["county_fips", "nasa_gsfc_elevation_mean_m"]]
    return soil, dem, veg, validation_static


def expand_static(df, sk):
    return sk[KEYS].merge(df, on=["state_fips", "county_fips"], how="left", validate="many_to_one")


def old_context(counties):
    o = pd.read_csv(OLD_MEGA, usecols=["GEOID", "Year", "ONI_annual_mean", "ONI_annual_max", "ONI_annual_min",
                                      "ONI_growseason_mean", "ONI_DJF", "ENSO_Phase", "ENSO_Anomalous_Year",
                                      "DSCI_growseason_mean", "DSCI_growseason_max", "Drought_Severe_Flag"])
    o["county_fips"] = o.GEOID.astype(str).str.zfill(5)
    o = o[o.county_fips.isin(counties)].rename(columns={
        "Year": "year", "ONI_annual_mean": "oni_annual_mean", "ONI_annual_max": "oni_annual_max",
        "ONI_annual_min": "oni_annual_min", "ONI_growseason_mean": "oni_growseason_mean", "ONI_DJF": "oni_djf",
        "ENSO_Phase": "enso_phase", "ENSO_Anomalous_Year": "enso_anomalous_year",
        "DSCI_growseason_mean": "usdm_dsci_gs_mean", "DSCI_growseason_max": "usdm_dsci_gs_max",
        "Drought_Severe_Flag": "usdm_drought_severe_flag"}).drop(columns="GEOID")
    return with_state(o)


# ----------------------------------------------------------------------------- model-ready selection
def model_ready_columns():
    ids = ["state", "state_name", "state_fips", "county", "county_fips", "year"]
    target = ["corn_yield_bu_acre"]
    geo = ["county_lat", "county_lon"]
    prism = ["prism_tmax_mean_gs", "prism_tmax_max_gs", "prism_tmin_mean_gs", "prism_tmin_min_gs", "prism_ppt_total_gs",
             "prism_gdd_gs", "prism_kdd30_gs", "prism_tmax_days_gt35_gs", "prism_heat_exceedance_total_gs"]
    nldas = ["nldas_et0_total_gs", "nldas_rs_mean_gs", "nldas_humidity_mean_gs", "nldas_vpd_mean_gs",
             "nldas_wind_mean_gs"]
    spi = [f"spi{k}_prism_{s}_gs" for k in (1, 3, 6, 12) for s in ("mean", "min")]
    spei = ["spei30_prism_nldas_mean_gs", "spei30_prism_nldas_min_gs", "spei30_prism_nldas_max_gs",
            "spei30_prism_nldas_drought_days_gs", "spei30_prism_nldas_drought_severity_gs",
            "spei90_prism_nldas_mean_gs", "spei90_prism_nldas_min_gs"]
    cdhw = ["cdhw_count_gs", "cdhw_total_days_gs", "cdhw_max_duration_gs", "cdhw_max_severity_gs", "cdhw_cumulative_severity_gs"]
    pheno = ["pheno_veg_days", "pheno_silk_days"]
    for s in ("veg", "silk", "grain"):
        pheno += [f"prism_tmax_mean_{s}", f"prism_tmax_days_gt35_{s}", f"prism_ppt_total_{s}", f"nldas_et0_total_{s}",
                  f"nldas_vpd_mean_{s}", f"spi1_prism_min_{s}", f"spei30_prism_nldas_mean_{s}", f"spei30_prism_nldas_min_{s}",
                  f"cdhw_count_{s}", f"cdhw_total_days_{s}", f"cdhw_cumulative_severity_{s}"]
    noaa = ["noaa_hail_count_gs", "noaa_thunderstorm_wind_count_gs", "noaa_tornado_count_gs", "noaa_max_hail_size_gs"]
    soil = ["soil_awc", "soil_bulk_density", "soil_clay", "soil_silt", "soil_organic_carbon"]
    dem = ["elevation_mean_m", "elevation_std_m", "slope_mean"]
    veg = ["nlcd_cropland_frac", "nlcd_pasture_hay_frac", "nlcd_forest_frac", "nlcd_grass_shrub_frac",
           "nlcd_developed_frac", "nlcd_water_wetland_frac"]
    # nldas_pressure_mean_gs is listed last: surface pressure is largely an elevation proxy, so the validated static
    # elevation is preferred if the two are near-duplicates.
    groups = dict(geography=geo, prism=prism, nldas=nldas, spi=spi, spei=spei, cdhw=cdhw, phenology=pheno, noaa=noaa,
                  soil=soil, dem=dem, vegetation=veg, low_priority=["nldas_pressure_mean_gs"])
    return ids, target, groups


# Priority when two candidates are near-duplicates on FIT years (|r| >= 0.98): drop the later one in this list.
REDUNDANCY_R = 0.98


def redundancy_filter(df, predictors):
    fit = df[df.year.between(*FIT_YEARS)][predictors]
    corr = fit.corr(min_periods=1000).abs()          # ignore pairs with too little overlap
    const = set(fit.columns[fit.nunique(dropna=True) <= 1])
    drops, rows = set(), []
    for i, a in enumerate(predictors):
        for b in predictors[i + 1:]:
            r = corr.loc[a, b]
            if a in const or b in const:
                continue
            if np.isfinite(r) and r >= REDUNDANCY_R:
                rows.append(dict(kept=a, candidate=b, abs_r_fit_years=round(float(r), 4),
                                 action="DROPPED" if a not in drops else "kept_partner_already_dropped"))
                if a not in drops:
                    drops.add(b)
    pd.DataFrame(rows, columns=["kept", "candidate", "abs_r_fit_years", "action"]).to_csv(
        AUDITS / "model_ready_redundancy_check.csv", index=False)
    return [p for p in predictors if p not in drops], sorted(drops)


def main():
    sk = skeleton()
    counties = set(sk.county_fips)
    ma = MergeAuditor()
    m = ma.merge(sk, usda(), "01_usda_corn_yield")
    m["has_corn_yield"] = m.corn_yield_bu_acre.notna().astype("int8")
    m = ma.merge(m, feature_table("prism_features"), "02_prism_features")
    m = ma.merge(m, feature_table("nldas_features"), "03_nldas_et0_features")
    m = ma.merge(m, feature_table("spi_features"), "04_spi_prism")
    m = ma.merge(m, feature_table("spei_features"), "05_spei_prism_nldas")
    m = ma.merge(m, feature_table("cdhw_features"), "06_cdhw_prism_nldas")
    m = ma.merge(m, feature_table("phenology_features"), "07_phenology_windows")
    m = ma.merge(m, noaa(counties), "08_noaa_storm_events")
    soil, dem, veg, val_static = static_tables(counties)
    m = ma.merge(m, expand_static(soil, sk), "09_soil")
    m = ma.merge(m, expand_static(dem, sk), "10_dem_elevation")
    m = ma.merge(m, expand_static(veg, sk), "11_vegetation_landcover")
    m = ma.merge(m, usda_context(), "12_context_usda_acreage_production")
    m = ma.merge(m, old_context(counties), "13_context_enso_usdm")
    ma.to_frame().to_csv(AUDITS / "master_merge_audit.csv", index=False)
    log.info("Merge audit:\n%s", ma.to_frame().to_string(index=False))
    assert_unique_key(m, ["county_fips", "year"], "FULL")

    # validation-only variables are kept OUT of FULL
    qc_only = [c for c in m.columns if c.endswith("_qc")]
    val = m[["county_fips", "year"] + qc_only].merge(val_static, on="county_fips", how="left")
    val.to_parquet(INTER / "validation" / "master_validation_only_variables.parquet", index=False)
    full = m.drop(columns=qc_only).sort_values(["state_fips", "county_fips", "year"]).reset_index(drop=True)
    full.to_parquet(FULL / FULL_NAME, index=False)
    log.info("FULL master: %d rows x %d columns", *full.shape)

    ids, target, groups = model_ready_columns()
    predictors = [c for g in groups.values() for c in g]
    missing_cols = [c for c in ids + target + predictors if c not in full.columns]
    if missing_cols:
        raise AssertionError(f"model-ready columns missing from FULL: {missing_cols}")
    kept, dropped = redundancy_filter(full, predictors)
    mr = full[full.corn_yield_bu_acre.notna()][ids + target + kept].reset_index(drop=True)
    assert_unique_key(mr, ["county_fips", "year"], "MODEL")
    mr.to_parquet(MODEL / MODEL_NAME, index=False)
    write_json(dict(groups=groups, dropped_as_near_duplicate=dropped, predictors_final=kept,
                    rows=len(mr), columns=mr.shape[1]), INTER / "model_ready_selection.json")
    log.info("MODEL-READY: %d rows x %d columns; near-duplicates dropped: %s", len(mr), mr.shape[1], dropped)


if __name__ == "__main__":
    main()
