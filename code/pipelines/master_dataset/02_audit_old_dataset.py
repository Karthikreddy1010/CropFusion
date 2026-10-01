"""
02_audit_old_dataset.py -- variable-by-variable audit of Paper3_MegaDataset_SPEI_FINAL.csv (READ ONLY).

Every one of the 85 columns is profiled (six study states only) and classified as
KEEP / RECOMPUTE / REPLACE / EXCLUDE / VALIDATION ONLY, with ERA5 lineage, temporal status
(static vs time-varying, valid years), leakage risk and the new master variable it maps to.

Outputs:
  audits/old_dataset_variable_audit.csv
  audits/old_dataset_evidence.json      (checks that justify the decisions)
  intermediate/static/old_static_county_variables.parquet (county-level static extract, six states)
  intermediate/validation/old_era5_county_year_validation.parquet (ERA5-era variables, VALIDATION ONLY,
      never merged into FULL or MODEL-READY)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from md_common import ARCGIS_IOWA_SAND, AUDITS, INTER, OLD_MEGA, STUDY_STATES, get_logger, write_json

log = get_logger("02_audit_old_dataset")
(INTER / "static").mkdir(exist_ok=True)
(INTER / "validation").mkdir(exist_ok=True)

ERA5_NOTE = ("ERA5-derived (methodology v4 section 4.1 / data table: SPEI-30 from ERA5 precipitation and "
             "Penman-Monteith ET0, 1981-2010 baseline; ERA5d_* from ERA5 daily)")

# variable -> (group, source, decision, era5_derived, leakage, new_variable, reason)
SPEC = {
    "GEOID": ("identifier", "US Census county FIPS", "KEEP", "NO", "NO", "county_fips", "Primary spatial key (5-digit FIPS string in master)."),
    "County_Name": ("identifier", "US Census", "KEEP", "NO", "NO", "county", "County name identifier."),
    "State": ("identifier", "US Census", "KEEP", "NO", "NO", "state_name", "State name identifier (needed by locked LOSO folds)."),
    "STATEFP": ("identifier", "US Census", "KEEP", "NO", "NO", "state_fips", "Join key component."),
    "COUNTYFP": ("identifier", "US Census", "KEEP", "NO", "NO", "county_code", "3-digit county code."),
    "Lat": ("geography", "County centroid (upstream, undocumented; static)", "KEEP", "NO", "LOW", "county_lat", "Static geographic context used by the existing model (STATIC_CONTEXT_COLS)."),
    "Lon": ("geography", "County centroid (upstream, undocumented; static)", "KEEP", "NO", "LOW", "county_lon", "Static geographic context used by the existing model."),
    "Year": ("identifier", "-", "KEEP", "NO", "NO", "year", "Primary temporal key."),
    "Split": ("model_artifact", "Old temporal split label", "EXCLUDE", "NO", "PROTOCOL", "", "Split assignment belongs to the locked code (splitting.py); a stored label could desynchronise the protocol."),
    "Corn_Yield_tha": ("target", "USDA NASS (old extract, x0.0627)", "VALIDATION ONLY", "NO", "TARGET", "corn_yield_bu_acre", "Target re-sourced from data/usda_nass processed parquet; old value used only to verify agreement."),
    "Corn_Yield_buacre": ("target", "USDA NASS (old extract)", "VALIDATION ONLY", "NO", "TARGET", "corn_yield_bu_acre", "Replaced by the processed USDA NASS corn grain yield; used to verify agreement."),
    "Soy_Yield_tha": ("other_crop_outcome", "USDA NASS soybean", "EXCLUDE", "NO", "HIGH", "", "Same-season outcome of another crop; co-determined with corn yield and released at the same time (proxy leakage)."),
    "Soy_Yield_buacre": ("other_crop_outcome", "USDA NASS soybean", "EXCLUDE", "NO", "HIGH", "", "Same-season outcome of another crop (proxy leakage)."),
    "Has_Corn_Yield": ("availability_flag", "derived", "RECOMPUTE", "NO", "TARGET", "has_corn_yield", "Recomputed from the new USDA target."),
    "Has_Soy_Yield": ("availability_flag", "derived", "EXCLUDE", "NO", "NO", "", "Soybean not part of this master."),
    "CDHW_Flag": ("CDHW", ERA5_NOTE, "REPLACE", "YES", "NO", "cdhw_any_gs", "ERA5-SPEI/ERA5-Tmax CDHW; replaced by PRISM-NLDAS CDHW (SPEI-30 PRISM-NLDAS < -1 AND PRISM Tmax > 35 C)."),
    "CDHW_Event_Count": ("CDHW", ERA5_NOTE, "REPLACE", "YES", "NO", "cdhw_total_days_gs / cdhw_count_gs", "ERA5 CDHW; methodology audit notes it is a day count, not an episode count. Replaced by explicit day and episode counts."),
    "CDHW_Severity_Score": ("CDHW", ERA5_NOTE, "REPLACE", "YES", "NO", "cdhw_cumulative_severity_gs", "ERA5 CDHW severity replaced by PRISM-NLDAS severity."),
    "CDHW_Severity_vegetative": ("CDHW", ERA5_NOTE, "REPLACE", "YES", "NO", "cdhw_cumulative_severity_veg", "Replaced by PRISM-NLDAS phenology-window CDHW."),
    "CDHW_Severity_silking": ("CDHW", ERA5_NOTE, "REPLACE", "YES", "NO", "cdhw_cumulative_severity_silk", "Replaced by PRISM-NLDAS phenology-window CDHW."),
    "CDHW_Severity_grainfill": ("CDHW", ERA5_NOTE, "REPLACE", "YES", "NO", "cdhw_cumulative_severity_grain", "Replaced by PRISM-NLDAS phenology-window CDHW."),
    "CDHW_Silking_Flag": ("CDHW", ERA5_NOTE, "REPLACE", "YES", "NO", "cdhw_total_days_silk", "Replaced by PRISM-NLDAS silking-window CDHW."),
    "SPEI_30_min": ("SPEI", ERA5_NOTE, "REPLACE", "YES", "NO", "spei30_prism_nldas_min_gs", "Old ERA5 SPEI-30; replaced by PRISM-NLDAS SPEI-30. Never mixed."),
    "SPEI_30_mean": ("SPEI", ERA5_NOTE, "REPLACE", "YES", "NO", "spei30_prism_nldas_mean_gs", "Old ERA5 SPEI-30; replaced."),
    "SPI_30_min": ("SPI", "ERA5 precipitation (upstream)", "REPLACE", "YES", "NO", "spi1_prism_min_gs", "Old ERA5 SPI-30; replaced by PRISM SPI."),
    "SPI_30_mean": ("SPI", "ERA5 precipitation (upstream)", "REPLACE", "YES", "NO", "spi1_prism_mean_gs", "Old ERA5 SPI-30; replaced by PRISM SPI."),
    "ET0_growseason_mm": ("ET0", ERA5_NOTE, "REPLACE", "YES", "NO", "nldas_et0_total_gs", "ERA5 ET0; replaced by NLDAS FAO-56 ET0."),
    "ET0_growseason_mean": ("ET0", ERA5_NOTE, "REPLACE", "YES", "NO", "nldas_et0_mean_gs", "ERA5 ET0; replaced by NLDAS FAO-56 ET0."),
    "WaterBalance_growseason_mm": ("water_balance", ERA5_NOTE, "REPLACE", "YES", "NO", "water_balance_total_gs", "ERA5 P - ERA5 ET0; replaced by PRISM P - NLDAS ET0."),
    "GDD_Accumulated": ("temperature", "ERA5 temperature (upstream)", "RECOMPUTE", "YES", "NO", "prism_gdd_gs", "Recomputed from PRISM Tmax/Tmin (base 10 C, cap 30 C, 1 Apr-31 Oct)."),
    "Tmax_Days_Above_35": ("temperature", "ERA5 temperature (upstream)", "RECOMPUTE", "YES", "NO", "prism_tmax_days_gt35_gs", "Recomputed from PRISM Tmax."),
    "Precip_growseason_mm": ("precipitation", "ERA5 precipitation (upstream)", "REPLACE", "YES", "NO", "prism_ppt_total_gs", "Replaced by PRISM precipitation."),
    "ERA5d_Tmax_mean_C": ("temperature", "ERA5 daily", "REPLACE", "YES", "NO", "prism_tmax_mean_gs", "ERA5 temperature replaced by PRISM (validation comparison only)."),
    "ERA5d_Tmax_max_C": ("temperature", "ERA5 daily", "REPLACE", "YES", "NO", "prism_tmax_max_gs", "ERA5 temperature replaced by PRISM."),
    "ERA5d_Tmin_mean_C": ("temperature", "ERA5 daily", "REPLACE", "YES", "NO", "prism_tmin_mean_gs", "ERA5 temperature replaced by PRISM."),
    "ERA5d_DiurnalRange_C": ("temperature", "ERA5 daily", "REPLACE", "YES", "NO", "prism_dtr_mean_gs", "ERA5 temperature replaced by PRISM."),
    "ONI_annual_mean": ("ENSO", "NOAA CPC Oceanic Nino Index", "KEEP", "NO", "FUTURE_INFO", "oni_annual_mean", "Year-level; annual mean includes the NDJ season (January of year+1) -> after harvest. FULL only, excluded from MODEL-READY."),
    "ONI_annual_max": ("ENSO", "NOAA CPC ONI", "KEEP", "NO", "FUTURE_INFO", "oni_annual_max", "Includes post-harvest seasons. FULL only."),
    "ONI_annual_min": ("ENSO", "NOAA CPC ONI", "KEEP", "NO", "FUTURE_INFO", "oni_annual_min", "Includes post-harvest seasons. FULL only."),
    "ONI_growseason_mean": ("ENSO", "NOAA CPC ONI", "KEEP", "NO", "LOW", "oni_growseason_mean", "Year-level large-scale climate context during the season (not ERA5)."),
    "ONI_DJF": ("ENSO", "NOAA CPC ONI", "KEEP", "NO", "NO", "oni_djf", "Dec(y-1)-Feb(y) ONI, known before planting (verified against CPC table)."),
    "ENSO_Phase": ("ENSO", "derived from ONI annual mean", "VALIDATION ONLY", "NO", "FUTURE_INFO", "enso_phase", "Evaluation stratification used by code/evaluation.py; derived from annual ONI (post-harvest months) so never a predictor."),
    "ENSO_Anomalous_Year": ("ENSO", "derived from ONI annual mean", "VALIDATION ONLY", "NO", "FUTURE_INFO", "enso_anomalous_year", "Evaluation stratification only; not a predictor."),
    "DSCI_growseason_mean": ("drought_monitor", "US Drought Monitor DSCI", "VALIDATION ONLY", "NO", "PARTIAL", "usdm_dsci_gs_mean", "Present only 2005-2023 in the old file (USDM itself starts 2000); expert-blended product partly informed by observed impacts. Used to validate new SPEI; FULL only."),
    "DSCI_growseason_max": ("drought_monitor", "US Drought Monitor DSCI", "VALIDATION ONLY", "NO", "PARTIAL", "usdm_dsci_gs_max", "2005-2023 only; FULL only."),
    "Drought_Severe_Flag": ("drought_monitor", "US Drought Monitor", "VALIDATION ONLY", "NO", "PARTIAL", "usdm_drought_severe_flag", "2005-2023 only; FULL only."),
    "Storm_Events_Total": ("NOAA", "NOAA Storm Events (old extract)", "REPLACE", "NO", "NO", "noaa_storm_count", "Old extract covers 2005-2023 only; replaced by data/noaa_storm_pipeline county-year product (1985-2023)."),
    "Storm_Hail_Count": ("NOAA", "NOAA Storm Events (old extract)", "REPLACE", "NO", "NO", "noaa_hail_count", "Replaced by NOAA pipeline."),
    "Storm_Flood_Count": ("NOAA", "NOAA Storm Events (old extract)", "REPLACE", "NO", "NO", "noaa_flood_count", "Replaced by NOAA pipeline."),
    "Storm_Heat_Count": ("NOAA", "NOAA Storm Events (old extract)", "REPLACE", "NO", "NO", "noaa_heat_count", "Replaced by NOAA pipeline (old column constant)."),
    "Storm_Drought_Event_Count": ("NOAA", "NOAA Storm Events (old extract)", "REPLACE", "NO", "NO", "noaa_drought_count", "Replaced by NOAA pipeline (old column constant)."),
    "Storm_Tornado_Count": ("NOAA", "NOAA Storm Events (old extract)", "REPLACE", "NO", "NO", "noaa_tornado_count", "Replaced by NOAA pipeline."),
    "Storm_Wind_Count": ("NOAA", "NOAA Storm Events (old extract)", "REPLACE", "NO", "NO", "noaa_thunderstorm_wind_count", "Replaced by NOAA pipeline."),
    "DEM_Mean": ("DEM", "USGS 3DEP DEM, ArcGIS Pro Zonal Statistics as Table (MEAN) on TIGER/Line 2023 counties, NAD83 (upstream table Merged_7States.csv; values identical to the old MegaDataset); DEM resolution not recorded", "KEEP", "NO", "NO", "elevation_mean_m", "Validated against NASA GSFC NLDAS fixed elevation county means (see evidence)."),
    "Slope_Mean": ("DEM", "Slope (degrees) from USGS 3DEP DEM via ArcGIS Surface > Slope, ArcGIS Pro Zonal Statistics as Table (MEAN) on TIGER/Line 2023 counties, NAD83 (upstream table Merged_7States.csv; values identical to the old MegaDataset)", "KEEP", "NO", "NO", "slope_mean", "Terrain / drainage modifier; checked against NLDAS GSFC slope grid."),
    "Soil_AWC_Mean": ("soil", "Soil rasters, ArcGIS Pro Zonal Statistics as Table (MEAN) on TIGER/Line 2023 counties, NAD83 (upstream table Merged_7States.csv; values identical to the old MegaDataset); stated source SSURGO, but values use SoilGrids-style units (texture g/kg summing to ~1000, BD cg/cm3) and raster scale/depth were not recorded -- source UNVERIFIED; AWC units unverified (median ~356 is closer to x1000 field-capacity water content than SSURGO AWC 0.1-0.25 cm/cm)", "RECOMPUTE", "NO", "NO", "soil_awc", "Nodata dilution detected (masked water/urban pixels averaged as 0). Recomputed as value / f_texture, f_texture = (clay+silt+sand)/closure of clean counties (texture-closure dilution factor); counties with f_texture < 0.25 set missing."),
    "Soil_BD_Mean": ("soil", "Soil rasters, ArcGIS Pro Zonal Statistics as Table (MEAN) on TIGER/Line 2023 counties, NAD83 (upstream table Merged_7States.csv; values identical to the old MegaDataset); stated source SSURGO, but values use SoilGrids-style units (texture g/kg summing to ~1000, BD cg/cm3) and raster scale/depth were not recorded -- source UNVERIFIED; cg/cm3", "RECOMPUTE", "NO", "NO", "soil_bulk_density", "Dilution-corrected; corrected BD also serves as the QC diagnostic (raw min 13.9 is not physical)."),
    "Soil_Clay_Mean": ("soil", "Soil rasters, ArcGIS Pro Zonal Statistics as Table (MEAN) on TIGER/Line 2023 counties, NAD83 (upstream table Merged_7States.csv; values identical to the old MegaDataset); stated source SSURGO, but values use SoilGrids-style units (texture g/kg summing to ~1000, BD cg/cm3) and raster scale/depth were not recorded -- source UNVERIFIED; g/kg", "RECOMPUTE", "NO", "NO", "soil_clay", "Dilution-corrected clay (texture)."),
    "Soil_Sand_Mean": ("soil", "Soil rasters, ArcGIS Pro Zonal Statistics as Table (MEAN) on TIGER/Line 2023 counties, NAD83 (upstream table Merged_7States.csv; values identical to the old MegaDataset); stated source SSURGO, but values use SoilGrids-style units (texture g/kg summing to ~1000, BD cg/cm3) and raster scale/depth were not recorded -- source UNVERIFIED; g/kg", "RECOMPUTE", "NO", "NO", "soil_sand", "Missing for all 99 Iowa counties in the old dataset because the ArcGIS Iowa sand table ('Soil') was never joined; filled from that table (validated: texture closes to ~1000 g/kg), then dilution-corrected. FULL only (sand ~ 1000 - clay - silt)."),
    "Soil_Silt_Mean": ("soil", "Soil rasters, ArcGIS Pro Zonal Statistics as Table (MEAN) on TIGER/Line 2023 counties, NAD83 (upstream table Merged_7States.csv; values identical to the old MegaDataset); stated source SSURGO, but values use SoilGrids-style units (texture g/kg summing to ~1000, BD cg/cm3) and raster scale/depth were not recorded -- source UNVERIFIED; g/kg", "RECOMPUTE", "NO", "NO", "soil_silt", "Dilution-corrected silt (texture)."),
    "Soil_OC_Mean": ("soil", "Soil rasters, ArcGIS Pro Zonal Statistics as Table (MEAN) on TIGER/Line 2023 counties, NAD83 (upstream table Merged_7States.csv; values identical to the old MegaDataset); stated source SSURGO, but values use SoilGrids-style units (texture g/kg summing to ~1000, BD cg/cm3) and raster scale/depth were not recorded -- source UNVERIFIED; dg/kg convention unverified", "RECOMPUTE", "NO", "NO", "soil_organic_carbon", "Dilution-corrected organic carbon (organic-matter proxy)."),
}
LC = ["Open_Water_frac", "Developed_Openspace_frac", "Developed_lowintensity_frac", "Developed_mediumintensity_frac",
      "Developed_highintensity_frac", "BarrenLand_frac", "DeciduousForest_frac", "EvergreenForest_frac", "MixedForest_frac",
      "Shrub_frac", "Grassland_frac", "Pasture_frac", "CultivatedCrops_frac", "WoodyWetlands_frac", "HerbaceousWetlands_frac"]
for c in LC:
    SPEC[c] = ("land_cover", "USGS NLCD 30 m class area / total NLCD area, ArcGIS Pro Zonal Statistics as Table (MEAN) on TIGER/Line 2023 counties, NAD83 (upstream table Merged_7States.csv; values identical to the old MegaDataset); NLCD year not recorded (single epoch >= 2001; static snapshot)", "KEEP", "NO", "LOW",
               "nlcd_" + c.replace("_frac", "").lower() + "_frac",
               "Static snapshot (constant across all 39 years). NLCD has no pre-2001 epoch, so it describes present-day land cover, not year-specific cover.")
SPEC.update({
    "ALAND": ("geography", "US Census TIGER land area (m2)", "KEEP", "NO", "NO", "county_land_area_m2", "Static administrative area; FULL only (not an environmental modifier)."),
    "AWATER": ("geography", "US Census TIGER water area (m2)", "KEEP", "NO", "NO", "county_water_area_m2", "Static; FULL only."),
    "Has_Drought": ("availability_flag", "derived", "EXCLUDE", "NO", "NO", "", "Old availability flag."),
    "Has_Storm": ("availability_flag", "derived", "EXCLUDE", "NO", "NO", "", "Old availability flag."),
    "Has_ERA5": ("availability_flag", "derived (ERA5)", "EXCLUDE", "YES", "NO", "", "ERA5 availability flag."),
    "Has_ERA5_daily": ("availability_flag", "derived (ERA5)", "EXCLUDE", "YES", "NO", "", "ERA5 availability flag."),
    "Has_CDHW": ("availability_flag", "derived (ERA5)", "EXCLUDE", "YES", "NO", "", "ERA5 CDHW availability flag."),
    "Has_SPEI": ("availability_flag", "derived (ERA5)", "EXCLUDE", "YES", "NO", "", "ERA5 SPEI availability flag."),
    "SPEI_30_min_dailycheck": ("SPEI", ERA5_NOTE, "EXCLUDE", "YES", "NO", "", "Duplicate QC recomputation of ERA5 SPEI_30_min."),
})


def main():
    old = pd.read_csv(OLD_MEGA, low_memory=False)
    assert list(SPEC.keys()) == list(old.columns) or set(SPEC) == set(old.columns), set(old.columns) ^ set(SPEC)
    six = old[old["STATEFP"].astype(str).str.zfill(2).isin(STUDY_STATES)].copy()
    ev = {"old_rows_all_states": len(old), "old_rows_six_states": len(six), "old_counties_six_states": int(six.GEOID.nunique()),
          "old_states": sorted(old.State.unique().tolist())}

    # --- evidence -----------------------------------------------------------
    gs_days = (six.ET0_growseason_mm / six.ET0_growseason_mean).round(3)
    ev["growing_season_length_days"] = gs_days.value_counts().to_dict()
    ev["waterbalance_identity_max_abs_err"] = float((six.WaterBalance_growseason_mm - (six.Precip_growseason_mm - six.ET0_growseason_mm)).abs().max())
    ev["corn_tha_per_buacre_factor"] = sorted((six.Corn_Yield_tha / six.Corn_Yield_buacre).dropna().round(5).unique().tolist())
    ev["year_level_constant"] = {c: bool(six.groupby("Year")[c].nunique(dropna=False).max() == 1)
                                 for c in ["ONI_annual_mean", "ONI_growseason_mean", "ONI_DJF", "ENSO_Phase", "ENSO_Anomalous_Year"]}
    ev["years_with_data"] = {c: sorted(six.loc[six[c].notna(), "Year"].unique().astype(int).tolist())[:1] + ["..."] +
                             sorted(six.loc[six[c].notna(), "Year"].unique().astype(int).tolist())[-1:]
                             for c in ["DSCI_growseason_mean", "Storm_Events_Total", "Corn_Yield_buacre", "SPEI_30_min"]}
    g = six.groupby("GEOID").first()
    m_mask = g.Open_Water_frac + g.Developed_mediumintensity_frac + g.Developed_highintensity_frac
    ev["soil_dilution"] = dict(
        corr_BD_vs_one_minus_water_devMH=float(np.corrcoef(g.Soil_BD_Mean, 1 - m_mask)[0, 1]),
        bd_min=float(g.Soil_BD_Mean.min()), bd_p05=float(g.Soil_BD_Mean.quantile(.05)), bd_median=float(g.Soil_BD_Mean.median()),
        counties_bd_below_120=int((g.Soil_BD_Mean < 120).sum()),
        sand_missing_counties_by_state=six[six.Soil_Sand_Mean.isna()].groupby("State").GEOID.nunique().to_dict(),
        texture_sum_min=float(g[["Soil_Clay_Mean", "Soil_Silt_Mean", "Soil_Sand_Mean"]].sum(axis=1, min_count=3).min()),
        interpretation="County soil means were computed with masked (water / urban) pixels counted as zero; every soil variable of a "
                       "county is scaled by the same valid-pixel fraction f. BD of mineral soils varies little, so X*BD_ref/BD removes f.")
    ev["landcover_fraction_sum"] = dict(min=float(g[LC].sum(axis=1).min()), max=float(g[LC].sum(axis=1).max()))
    ev["static_within_county"] = {c: int((six.groupby("GEOID")[c].nunique() > 1).sum()) for c in
                                  ["DEM_Mean", "Slope_Mean", "Soil_AWC_Mean", "CultivatedCrops_frac", "Lat", "ALAND"]}

    # --- profile + classification -----------------------------------------
    rows = []
    for c in old.columns:
        grp, src, dec, era5, leak, newv, reason = SPEC[c]
        s = six[c]
        within = six.groupby("GEOID")[c].nunique(dropna=True)
        across = six.groupby("Year")[c].nunique(dropna=True)
        if within.max() <= 1:
            temporal = "STATIC"
        elif across.max() <= 1:
            temporal = "TIME-VARYING (year-level, same for all counties)"
        else:
            temporal = "TIME-VARYING"
        yrs = six.loc[s.notna(), "Year"]
        num = pd.api.types.is_numeric_dtype(s)
        rows.append(dict(variable=c, group=grp, source=src, era5_derived=era5, decision=dec, new_master_variable=newv,
                         temporal_status=temporal, valid_years=f"{int(yrs.min())}-{int(yrs.max())}" if len(yrs) else "",
                         missing_percent_six_states=round(100 * s.isna().mean(), 2), n_unique=int(s.nunique()),
                         min=float(s.min()) if num else "", median=float(s.median()) if num else "",
                         max=float(s.max()) if num else "", potential_leakage=leak, reason=reason,
                         in_full_master="YES" if dec in ("KEEP", "RECOMPUTE") or (dec == "VALIDATION ONLY" and grp in ("ENSO", "drought_monitor")) else "NO",
                         in_model_ready="YES" if (dec in ("KEEP", "RECOMPUTE") and leak in ("NO", "LOW") and grp not in ("geography",) ) or c in ("Lat", "Lon") else "NO"))
    audit = pd.DataFrame(rows)
    # ALAND/AWATER are FULL-only; ONI annual (future info) FULL-only
    audit.loc[audit.variable.isin(["ALAND", "AWATER", "Soil_Sand_Mean"]), "in_model_ready"] = "NO"
    audit.to_csv(AUDITS / "old_dataset_variable_audit.csv", index=False)
    ev["decision_counts"] = audit.decision.value_counts().to_dict()
    ev["era5_derived_variables"] = audit.loc[audit.era5_derived == "YES", "variable"].tolist()

    # --- static county extract (KEEP/RECOMPUTE inputs) ----------------------
    st = g.reset_index()
    st["county_fips"] = st.GEOID.astype(str).str.zfill(5)
    # Dilution correction: valid-soil fraction f = 1 - (open water + developed medium + developed high)
    # (NLCD classes that soil grids mask). Chosen over BD-ratio normalisation, which over-corrects naturally
    # low-BD organic soils (evidence: soil_correction_candidates).
    cands = {"BD_ratio_to_clean_median": None, "W": g.Open_Water_frac,
             "W+DM+DH": m_mask, "W+DL+DM+DH": m_mask + g.Developed_lowintensity_frac}
    ev["soil_correction_candidates"] = {}
    for k, m in cands.items():
        if m is None:
            continue
        bdc = g.Soil_BD_Mean / (1 - m)
        ev["soil_correction_candidates"][k] = dict(cv=float(bdc.std() / bdc.mean()), n_below_115=int((bdc < 115).sum()),
                                                   n_above_160=int((bdc > 160).sum()), min=float(bdc.min()), max=float(bdc.max()))
    out = pd.DataFrame({"county_fips": st.county_fips, "county_lat": st.Lat, "county_lon": st.Lon,
                        "county_land_area_m2": st.ALAND, "county_water_area_m2": st.AWATER,
                        "elevation_mean_m": st.DEM_Mean, "slope_mean": st.Slope_Mean})
    mask_frac = (st.Open_Water_frac + st.Developed_mediumintensity_frac + st.Developed_highintensity_frac).values
    f = 1.0 - mask_frac
    # Iowa sand: missing from the old dataset because the ArcGIS "Soil" (sand) zonal table was never joined into
    # Iowa_final.csv. Filled from that table after validation (texture closure and implied-sand agreement).
    ia_sand = pd.read_csv(ARCGIS_IOWA_SAND, dtype={"GEOID": str})
    assert ia_sand.GEOID.is_unique and len(ia_sand) == 99 and ia_sand.GEOID.str.startswith("19").all()
    sand_map = dict(zip(ia_sand.GEOID, ia_sand.Soil_Mean))
    sand_raw = st.Soil_Sand_Mean.copy()
    fill = sand_raw.isna() & st.county_fips.isin(sand_map)
    sand_raw[fill] = st.loc[fill, "county_fips"].map(sand_map)
    iowa = st.county_fips.str.startswith("19")
    clean_ia = iowa & (m_mask.values < 0.02)
    clean_other = ~iowa & (m_mask.values < 0.02)
    tex_ia = st.Soil_Clay_Mean + st.Soil_Silt_Mean + sand_raw
    cell_deg = np.sqrt(ia_sand.AREA / ia_sand.COUNT)
    ev["iowa_sand_fill"] = dict(
        source=str(ARCGIS_IOWA_SAND), counties_filled=int(fill.sum()), remaining_sand_missing=int(sand_raw.isna().sum()),
        texture_sum_clean_iowa_median=float(tex_ia[clean_ia].median()), texture_sum_clean_other_states_median=float(tex_ia[clean_other].median()),
        corr_with_1000_minus_clay_silt_clean_iowa=float(np.corrcoef(sand_raw[clean_ia], 1000 - st.Soil_Clay_Mean[clean_ia] - st.Soil_Silt_Mean[clean_ia])[0, 1]),
        implied_raster_cell_deg=float(cell_deg.median()), implied_raster_cell_m_ns=float(cell_deg.median() * 111320),
        interpretation="Soil_Mean is sand: texture closes to ~1000 g/kg like the other states; ~250 m cells (SoilGrids-like grid; gSSURGO is 30 m).")
    assert abs(ev["iowa_sand_fill"]["texture_sum_clean_iowa_median"] - 1000) < 20, "Iowa Soil_Mean does not behave like sand"
    st = st.assign(Soil_Sand_Mean=sand_raw)
    # Texture-closure dilution factor (replaces the NLCD-mask factor once sand is complete for all counties):
    # all soil layers share one raster grid, and blank (water/urban) cells entered the county means as 0, so
    # f_texture = (clay + silt + sand) / reference closure, with the reference = median closure of counties with
    # < 2 % water + developed-medium/high land. Evidence: bulk-density CV 0.138 raw -> 0.066 (NLCD factor) -> 0.048
    # (texture factor); NLCD factor over-corrects some lake counties (texture sums up to 1173) and under-corrects
    # dense urban counties (e.g. DuPage NLCD 0.65 vs texture 0.28).
    tex_raw = st.Soil_Clay_Mean + st.Soil_Silt_Mean + st.Soil_Sand_Mean
    tex_ref = float(tex_raw[(m_mask.values < 0.02)].median())
    f_tex = (tex_raw / tex_ref).values
    for name, fac in (("raw", np.ones(len(st))), ("nlcd_mask_factor", f), ("texture_closure_factor", f_tex)):
        bdc = st.Soil_BD_Mean.values / fac
        ok = fac >= 0.25
        ev.setdefault("soil_dilution_factor_comparison", {})[name] = dict(
            bd_cv=float(np.std(bdc[ok]) / np.mean(bdc[ok])), bd_min=float(bdc[ok].min()), bd_max=float(bdc[ok].max()),
            n_bd_below_115=int((bdc[ok] < 115).sum()),
            texture_sum_min=float((tex_raw.values / fac)[ok].min()), texture_sum_max=float((tex_raw.values / fac)[ok].max()))
    ev["soil_dilution"]["texture_reference_g_per_kg"] = tex_ref
    ev["soil_dilution"]["corr_texture_factor_vs_nlcd_factor"] = float(np.corrcoef(f_tex, f)[0, 1])
    out["soil_valid_fraction"] = f_tex
    for src_c, new_c in [("Soil_AWC_Mean", "soil_awc"), ("Soil_BD_Mean", "soil_bulk_density"),
                         ("Soil_Clay_Mean", "soil_clay"), ("Soil_Sand_Mean", "soil_sand"), ("Soil_Silt_Mean", "soil_silt"),
                         ("Soil_OC_Mean", "soil_organic_carbon")]:
        out[new_c] = st[src_c].values / f_tex
    # fewer than a quarter of the county's soil cells carry data -> too few cells to represent the county
    failed = pd.Series(f_tex < 0.25, index=out.index)
    out["soil_qc_flag"] = np.select([failed, f_tex < 0.95], ["LOW_COVERAGE_SET_MISSING", "DILUTION_CORRECTED"], "OK")
    soil_cols = ["soil_awc", "soil_bulk_density", "soil_clay", "soil_sand", "soil_silt", "soil_organic_carbon"]
    out.loc[failed, soil_cols] = np.nan
    ev["soil_dilution"]["qc_flag_counts"] = out.soil_qc_flag.value_counts().to_dict()
    ev["soil_dilution"]["failed_counties"] = st.loc[failed.values, ["county_fips", "County_Name", "State"]].astype(str).values.tolist()
    ev["soil_dilution"]["corrected_bd_range"] = [float(out.soil_bulk_density.min()), float(out.soil_bulk_density.max())]
    for c in LC:
        out[SPEC[c][5]] = st[c]
    out.to_parquet(INTER / "static" / "old_static_county_variables.parquet", index=False)
    norm = out[soil_cols]
    ev["soil_corrected_summary"] = norm.describe().round(2).to_dict()
    ev["soil_corrected_corr"] = norm.corr().round(3).to_dict()
    ev["soil_texture_clay_plus_silt_corrected"] = out[["soil_clay", "soil_silt"]].sum(axis=1, min_count=2).describe().round(1).to_dict()
    ev["soil_texture_clay_silt_sand_corrected"] = out[["soil_clay", "soil_silt", "soil_sand"]].sum(axis=1, min_count=3).describe().round(1).to_dict()

    # --- ERA5-era validation extract (never merged into master) -------------
    era5_cols = ["GEOID", "Year", "Corn_Yield_buacre"] + audit.loc[audit.era5_derived == "YES", "variable"].tolist() + \
                ["DSCI_growseason_mean", "DSCI_growseason_max", "Storm_Events_Total", "Storm_Hail_Count", "Storm_Tornado_Count"]
    v = six[[c for c in dict.fromkeys(era5_cols)]].copy()
    v.insert(0, "county_fips", v.pop("GEOID").astype(str).str.zfill(5))
    v = v.rename(columns={"Year": "year"})
    v.to_parquet(INTER / "validation" / "old_era5_county_year_validation.parquet", index=False)

    write_json(ev, AUDITS / "old_dataset_evidence.json")
    log.info("Old dataset audit: %s", audit.decision.value_counts().to_dict())
    log.info("ERA5-derived: %s", ev["era5_derived_variables"])
    log.info("Soil dilution: %s", ev["soil_dilution"])


if __name__ == "__main__":
    main()
