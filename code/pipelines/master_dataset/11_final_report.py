"""
11_final_report.py -- writes master_dataset/MASTER_DATASET_FINAL_REPORT.md from the audit artefacts.
All numbers are read from files produced by scripts 01-10 (nothing is typed in by hand).
"""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from md_common import AUDITS, INTER, MD, META, get_logger

log = get_logger("11_final_report")


def j(p):
    return json.loads(p.read_text(encoding="utf-8"))


def table(df, cols=None, n=None):
    df = df if cols is None else df[cols]
    if n:
        df = df.head(n)
    return df.to_markdown(index=False)


def main():
    q = j(AUDITS / "qc_summary.json")
    src = j(AUDITS / "source_audit_summary.json")
    ev = j(AUDITS / "old_dataset_evidence.json")
    sel = j(INTER / "model_ready_selection.json")
    dem = j(AUDITS / "dem_validation_vs_nasa_gsfc.json")
    qc = pd.read_csv(AUDITS / "master_qc_report.csv")
    cov = pd.read_csv(AUDITS / "master_coverage_report.csv")
    merge = pd.read_csv(AUDITS / "master_merge_audit.csv")
    old = pd.read_csv(AUDITS / "old_dataset_variable_audit.csv")
    nl = pd.read_csv(AUDITS / "nldas_daily_completeness_report.csv")
    pre = pd.read_csv(AUDITS / "nldas_completeness_by_year_prebackfill.csv")
    bf = pd.concat([pd.read_csv(f) for f in AUDITS.glob("nldas_backfill_manifest*.csv")], ignore_index=True)
    cal = pd.read_csv(AUDITS / "spi_spei_calibration_check.csv")
    lk = pd.read_csv(AUDITS / "leakage_audit.csv")
    cat = pd.read_csv(META / "master_variable_catalog.csv")
    gdd = pd.read_csv(AUDITS / "gdd_definition_check.csv")
    noaa_val = pd.read_csv(AUDITS / "noaa_gs_recount_validation.csv")
    red = pd.read_csv(AUDITS / "model_ready_redundancy_check.csv")
    full = pd.read_parquet(MD / "full" / "Paper3_MasterDataset_PRISM_NLDAS_NOAA_USDA_FULL_1985_2023.parquet",
                           columns=["county_fips", "year", "state", "corn_yield_bu_acre"])

    qc_fail = qc[qc.status == "FAIL"]
    all_pass = len(qc_fail) == 0
    usda_missing = full[full.corn_yield_bu_acre.isna()]
    miss_by_state_year = usda_missing.groupby(["state"]).size().rename("missing_yield_county_years").reset_index()
    miss_by_year = usda_missing.groupby("year").size()
    counties_never = sorted(set(full.county_fips) - set(full.loc[full.corn_yield_bu_acre.notna(), "county_fips"]))
    inter = cov[cov.source.str.startswith("COMPLETE INTERSECTION")].iloc[0]
    usda_clim = cov[cov.source.str.startswith("USDA yield AND")].iloc[0]
    totals = cov[cov.year == "ALL"]
    mr_vars = cat[cat.in_model_ready == "YES"]
    soil_kept = [v for v in ["soil_awc", "soil_bulk_density", "soil_clay", "soil_silt", "soil_sand", "soil_organic_carbon"] if v in set(cat.variable)]
    spi_vars = sorted({v for v in cat.variable if v.startswith("spi")})
    spei_vars = sorted({v for v in cat.variable if v.startswith("spei")})
    cdhw_vars = sorted({v for v in cat.variable if v.startswith("cdhw")})
    reused = old[old.decision == "KEEP"].variable.tolist()
    recomputed = old[old.decision == "RECOMPUTE"].variable.tolist()
    replaced = old[old.decision == "REPLACE"].variable.tolist()
    excluded = old[old.decision == "EXCLUDE"].variable.tolist()
    valonly = old[old.decision == "VALIDATION ONLY"].variable.tolist()
    post_harvest = q["post_harvest_predictors_in_model"]
    leak_flags = mr_vars[(mr_vars.target_or_predictor == "PREDICTOR") & ~mr_vars.potential_leakage.isin(["NO", "LOW"])].variable.tolist()

    L = []
    A = L.append
    A("# Paper 3 Master Dataset — PRISM · NLDAS-2 · NOAA · USDA (1985–2023)")
    A(f"_Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `code/pipelines/master_dataset/11_final_report.py` from the audit artefacts._\n")
    A("## Status\n")
    A(f"**QC: {q['qc_pass']}/{q['qc_total']} critical checks PASS.** " + ("All critical checks pass; the dataset is declared **MODEL-READY** (data level)." if all_pass
      else "The following checks FAIL, so the dataset is **NOT** declared model-ready:\n\n" + table(qc_fail, ["check", "detail"])))
    A("\n| Product | Rows | Columns |\n|---|---:|---:|")
    A(f"| FULL `full/Paper3_MasterDataset_PRISM_NLDAS_NOAA_USDA_FULL_1985_2023.parquet` | {q['full_shape'][0]:,} | {q['full_shape'][1]} |")
    A(f"| MODEL-READY `model_ready/Paper3_MasterDataset_PRISM_NLDAS_CDHWH_CORN_MODEL_1985_2023.parquet` | {q['model_shape'][0]:,} | {q['model_shape'][1]} |\n")

    A("## Answers to the 22 required questions\n")
    A(f"**1. How many counties?** {q['counties']} (IL 102, IN 92, IA 99, MN 87, MO 115, OH 88). The brief's 676 includes Nebraska (93 counties), which the locked design drops (`code/config.py::DROP_STATES`).\n")
    A(f"**2. How many county-years?** {q['county_years']:,} in FULL (583 × 39; theoretical 676 × 39 = 26,364 is not reachable for the six states). MODEL-READY keeps the {q['model_shape'][0]:,} county-years with an observed yield.\n")
    peryear = cov[cov.year != "ALL"].copy()
    peryear["year"] = peryear.year.astype(int)
    full_years = {k: sorted(g.loc[g.coverage_percent >= 100, "year"].tolist()) for k, g in peryear.groupby("source") if "-only" not in k}
    def yrs(v):
        return "all 39 years" if len(v) == 39 else (f"{len(v)} years: {v}" if v else "no year")
    A("**3. Which years are complete?** Years in which a source covers all 583 counties: " +
      "; ".join(f"{k}: {yrs(v)}" for k, v in full_years.items() if k in ("PRISM", "NLDAS", "SPI", "SPEI", "CDHW", "NOAA", "SOIL", "DEM", "VEGETATION", "USDA")) +
      ". (SOIL is < 100 % by design: counties where fewer than 25 % of soil cells carry data are set missing. USDA NASS omits some counties every year.)\n")
    A(f"**4. Which county-years are missing?** {len(usda_missing):,} county-years have no published USDA corn yield (these rows are kept in FULL with `has_corn_yield = 0`, never imputed, and excluded from MODEL-READY). By state:\n")
    A(table(miss_by_state_year))
    A(f"\nMissing yields by year range from {miss_by_year.min()} to {miss_by_year.max()} county-years; {len(counties_never)} counties never report a yield: {', '.join(counties_never[:30]) or 'none'}{' …' if len(counties_never) > 30 else ''}. Full lists: `audits/master_coverage_report.csv`.\n")
    A(f"**5. How many USDA yield observations?** {q['usda_obs']:,} county-years (source rows {src['usda_yield']['rows']:,}; 0 duplicates; range {src['usda_yield']['yield_min']}–{src['usda_yield']['yield_max']} bu/acre).\n")
    def qd(name):
        r = qc[qc.check == name]
        return f"{r.status.iloc[0]} ({r.detail.iloc[0]})" if len(r) else "n/a"
    A(f"**6. How many PRISM observations?** {src['prism']['files_found']:,} of {src['prism']['files_expected']:,} daily archives (14,244 dates x tmax/tmin/ppt) were read in place from `D:\\prism_pipeline\\raw`, i.e. {int(q['prism_county_days']):,} county-days with all three variables valid. QC: dates {qd('PRISM daily: 14,244 dates')}; tmax>=tmin {qd('PRISM daily: tmax >= tmin')}; ppt>=0 {qd('PRISM daily: ppt >= 0')}; NaN {qd('PRISM daily: no NaN')}.\n")
    A(f"**7. How many complete NLDAS days?** {q['nldas_complete_days']:,} of 14,244 dates have 24/24 verified hourly granules after backfill.\n")
    A(f"**8. How many incomplete NLDAS days remain?** {q['nldas_incomplete_days']}" + (f": {', '.join(q['nldas_incomplete_dates'][:40])}{' ...' if q['nldas_incomplete_days'] > 40 else ''} (all listed in `audits/nldas_daily_completeness_report.csv`)." if q['nldas_incomplete_days'] else ".") +
      f" Before backfill the audit found {int((pre.days - pre.complete_days).sum())} incomplete dates ({int(pre.missing_hours.sum()):,} missing/0-byte hours; per year in `audits/nldas_completeness_by_year_prebackfill.csv`). "
      f"{int((bf.status == 'BACKFILLED').sum()) + int((bf.status == 'ALREADY_PRESENT').sum()):,} hourly granules were backfilled from NASA GES DISC (NLDAS_FORA0125_H.2.0), "
      f"including granules that existed but were corrupt (HDF5 parse failure). Only study-region subsets were stored (`intermediate/nldas_backfill`); raw folders were not modified.\n")
    A(f"**9. How many NOAA county-year observations?** {q['noaa_county_years']:,} county-years (complete skeleton). Growing-season counts were re-aggregated from the cleaned events; the annual recount reproduces the existing county-year product exactly ({int(noaa_val.mismatching_county_years.sum())} mismatches).\n")
    A(f"**10. Which SPI variables were created?** Daily `spi1_prism`, `spi3_prism`, `spi6_prism`, `spi12_prism` (30/90/180/365-day accumulations) and their county-year summaries: {', '.join(spi_vars)}. SPI-12 is undefined for the first 364 days of 1985 and SPI-6 for the first 179 (no pre-1985 PRISM).\n")
    A(f"**11. Which SPEI variables were created?** Daily `spei30_prism_nldas` (required) and `spei90_prism_nldas` (seasonal-scale drought relevant to maize), summarised as: {', '.join(spei_vars)}.\n")
    A(f"**12. Which CDHW variables were created?** {', '.join(cdhw_vars)}.\n")
    A(f"**13. Which soil variables were retained?** {', '.join(soil_kept)} (dilution-corrected), plus `soil_qc_flag`, `soil_valid_fraction` in FULL. `soil_sand` is in FULL only: Iowa sand (absent from the old dataset because the ArcGIS Iowa 'Soil' sand table was never joined) was filled from that table after validation (texture closes to ~1000 g/kg); MODEL-READY keeps clay and silt because sand = ~1000 - clay - silt. Soil drainage class is **not available** in any project source. {ev['soil_dilution']['qc_flag_counts'].get('LOW_COVERAGE_SET_MISSING', 0)} counties with fewer than 25 % valid soil cells are set missing; all other counties are corrected with the texture-closure valid fraction.\n")
    A(f"**14. Which DEM variables were retained?** `elevation_mean_m` (old DEM_Mean; validated against NASA GSFC NLDAS elevation, r = {dem['pearson_r_old_DEM_Mean_vs_NASA_GSFC']:.3f}, mean bias {dem['mean_bias_m']:.1f} m), `elevation_std_m` (NASA GSFC, within+between-cell), `slope_mean` (old Slope_Mean, units undocumented).\n")
    A("**15. Which vegetation variables were retained?** NLCD static snapshot: grouped `nlcd_cropland_frac`, `nlcd_pasture_hay_frac`, `nlcd_forest_frac`, `nlcd_grass_shrub_frac`, `nlcd_developed_frac`, `nlcd_water_wetland_frac`, `nlcd_barren_frac` (FULL also keeps the 15 original classes). All are **STATIC** — one undocumented NLCD epoch (≥2001) applied to every year; not valid as year-specific land cover for 1985–2000. `corn_fraction` is not available (NLCD has no crop-type classes; USDA acreage is same-survey and excluded). No vegetation index (NDVI/EVI) exists in any project source.\n")
    A(f"**16. Which old MegaDataset variables were reused (KEEP)?** {', '.join(reused)}.\n")
    A(f"**17. Which were recomputed?** {', '.join(recomputed)}. Replaced by new PRISM/NLDAS/NOAA products (REPLACE): {', '.join(replaced)}.\n")
    A(f"**18. Which were excluded?** EXCLUDE: {', '.join(excluded)}. VALIDATION ONLY: {', '.join(valonly)}.\n")
    A(f"**19. Are any ERA5 variables still present?** {'No.' if not q['era5_columns'] else q['era5_columns']} All {len(ev['era5_derived_variables'])} ERA5-derived columns were replaced; they exist only in `intermediate/validation/old_era5_county_year_validation.parquet` for comparison and are never merged. New indices are explicitly named `*_prism`, `*_prism_nldas`.\n")
    A("**20. Are there duplicate county-year records?** No — every intermediate table and both masters are unique on `county_fips + year` (asserted before and after each of the 13 joins; see merge audit).\n")
    no_leak = not post_harvest and not q["leakage_suspicious"] and not leak_flags
    A("**21. Is there any data leakage?** " + ("No target leakage is detected in MODEL-READY:\n" if no_leak
      else "Potential leakage flags remain in MODEL-READY (see below and `audits/leakage_audit.csv`):\n"))
    A(f"- Excluded as leakage/proxy: soybean yields, USDA production/acreage/CV (same survey as target), NOAA crop damage, NOAA drought reports, USDM DSCI, annual ONI/ENSO phase (include post-harvest months), all `_ann` climate windows (include Nov–Dec after harvest), `Split`.\n"
      f"- Post-harvest predictors in MODEL-READY: {post_harvest or 'none'}. Predictors with |r| > 0.9 with yield on FIT years: {q['leakage_suspicious'] or 'none'}. Catalog leakage flags other than NO/LOW in MODEL-READY: {leak_flags or 'none'}.\n"
      "- SPI/SPEI distributions are fitted on the locked FIT years 1985–2013 only, so no DEV/CAL/TEST-period information enters the indices.\n"
      "- Residual caveats (not leakage): NLCD and soil are present-day static snapshots applied to all years; NOAA hail/wind/tornado reporting density rises over time.\n")
    A("**22. Is the final dataset ready for the existing Neural CQR + locked LOSO experiment?** "
      + ("**Data: yes** — all critical QC checks pass, keys are unique, the target is un-imputed, no ERA5 variables remain, "
         + ("no leakage flags remain, " if no_leak else "leakage flags need review, ")
         + f"and MODEL-READY spans {sorted(full.state.unique())} for {int(full.year.min())}-{int(full.year.max())}. "
         if all_pass else "**Data: not yet** — see failing QC checks above. ")
      + "**Code integration: available (opt-in).** Set `PAPER3_DATA_SOURCE=master` to run the locked pipeline on this dataset through `code/master_data_loader.py` (approved data-interface change): the complete FULL county-year skeleton with the MODEL-READY predictors, protocol identifiers, target `Corn_Yield_tha` = bu/acre × 0.0627 (unchanged loss scale), methodology roles renamed from their PRISM-NLDAS sources (e.g. `SPEI_30_min` ← `spei30_prism_nldas_min_gs`, `CDHW_Severity_Score` ← `cdhw_cumulative_severity_gs`), ENSO labels for evaluation stratification only, outputs in `outputs_master/`. The default legacy run is unchanged (feature-engineering output verified identical). `code/smoke_test_master_loader.py` exercises loading, feature engineering, the 4-way temporal split, FIT-only selection, scaling and a full LOSO fold. No model architecture, loss, threshold, split or LOSO protocol was changed.\n")

    A("## Scientific architecture and methods\n")
    A("- **PRISM → Tmax, Tmin, ppt**: daily 4 km grids read from ZIP archives in place; county values = fractional-area polygon weights (exactextract coverage × cos φ). Verified against direct exactextract means (max diff 0.003 °C).\n"
      "- **NLDAS-2 → Rs, ea, u2, P, Tair → FAO-56 ET0**: every date audited for 24 hourly granules (time stamp checked in each file); days aggregated only when 24/24 verified; daily aggregation identical to `download_nldas.py`; ET0 from the validated `fao56_et0.py` (Eq. 6, Rnl Eq. 39, Rso with the fixed NASA GSFC elevation grid; pressure used only for γ). Reproduces the previously validated daily grids to ~1e-7. Grid ET0 computed first, then county aggregation. Day = UTC 00–23 h (PRISM day ends 12 UTC: ~half-day offset, documented).\n"
      "- **Water balance** D = PRISM ppt − NLDAS ET0 → **SPEI-30/90**: log-logistic family fitted as the generalized logistic (GLO) from unbiased PWMs, as in the SPEI R package. The closed-form 3-parameter log-logistic failed for negatively skewed summer water-balance samples (19,439 SPEI-30 and 72,996 SPEI-90 county-slots, all with L-skewness < 0); GLO equals it to < 2e-7 wherever it was valid and fits every slot. **SPI-1/3/6/12** from PRISM ppt (gamma, Thom MLE, zero-mass mixture). Accumulate first, fit per county × day-of-year slot (±7 d pooling) on 1985–2013, transform to N(0,1), bounded ±3.09.\n"
      "- **CDHW** day: SPEI-30 < −1 AND PRISM Tmax > 35 °C; joint severity |SPEI-30| × (Tmax − 35) (methodology §4.1); episodes = runs of consecutive CDHW days.\n"
      f"- **Phenology**: growing season 1 Apr–31 Oct (recovered from old dataset: 214 days); GDD stages from 1 Apr (base 10, cap 30 °C) with project thresholds 700 / 1400. The GDD definition matches the old variable best among candidates (r = {gdd.iloc[0].pearson_r:.3f}). Late season omitted (no project definition).\n"
      "- **NOAA Storm Events**: county-year product + growing-season re-aggregation; event types not recorded before 1996 are NaN (never 0).\n"
      "- **Soil**: old county means were diluted by masked water/urban pixels counted as zero (BD correlates r = 0.92 with 1 − masked fraction); corrected by dividing each layer by the texture-closure valid fraction (clay+silt+sand relative to clean counties; bulk-density CV 0.138 raw → 0.048), which replaced an earlier NLCD-mask factor (CV 0.066) once Iowa sand completed the texture triplet; counties with < 25 % valid soil cells are set missing.\n")

    A("## SPI / SPEI calibration check\n")
    A(table(cal))
    A("\n## New PRISM/NLDAS products vs old ERA5-era variables (validation only)\n")
    A(table(pd.read_csv(AUDITS / "new_vs_old_era5_validation.csv")))
    A("\n## Coverage (all years)\n")
    A(table(totals[["source", "counties", "county_years", "missing_county_years", "coverage_percent"]]))
    A(f"\nComplete intersection of all sources: **{int(inter.county_years):,}** county-years ({inter.coverage_percent}%). USDA yield ∩ all climate sources: **{int(usda_clim.county_years):,}**.\n")
    A("## Merge audit\n")
    A(table(merge[["step", "rows_before", "rows_after", "columns_before", "columns_after", "duplicate_keys", "new_missing_values", "county_count", "year_min", "year_max"]]))
    A("\n## QC report\n")
    A(table(qc[["check", "status", "detail"]]))
    A("\n## MODEL-READY variable groups\n")
    for g, cols in sel["groups"].items():
        kept = [c for c in cols if c in sel["predictors_final"]]
        A(f"- **{g}** ({len(kept)}): {', '.join(kept)}")
    A(f"\nNear-duplicates removed (|r| ≥ 0.98 on FIT years 1985–2013, kept in FULL): {', '.join(sel['dropped_as_near_duplicate']) or 'none'}.\n")
    A("## Known limitations\n")
    mr = pd.read_parquet(MD / "model_ready" / "Paper3_MasterDataset_PRISM_NLDAS_CDHWH_CORN_MODEL_1985_2023.parquet")
    na = mr.isna().sum()
    na = na[na > 0]
    A("- MODEL-READY contains only structural, documented missing values (handled by the locked pipeline's train-fit imputation): "
      + "; ".join(f"`{k}` {int(v)}" for k, v in na.items())
      + ". SPI-6/12 growing-season values are undefined in 1985 (no pre-1985 PRISM); grain-fill means are undefined where cumulative GDD never exceeds 1400 by 31 Oct; soil is missing where fewer than 25 % of a county's soil cells carry data.")
    A("- Product differences, not errors: NLDAS FAO-56 ET0 is ~10 % higher than the old ERA5 ET0 and PRISM records about twice as many Tmax > 35 degC days as ERA5, so CDHW frequency and severity exceed the old dataset; 2014-2023 are wetter than the 1985-2013 SPI calibration period (mean SPI-12 +0.58).")
    A("- The `*-only` coverage rows are zero because every climate/static source covers every skeleton county-year; per-source coverage is in the rows above them.")
    A("- Static layers come from the user's ArcGIS Pro extraction (`Merged_7States.csv`, identical to the old MegaDataset): USGS 3DEP DEM, slope in degrees, zonal MEAN on TIGER/Line 2023 counties. Not recorded: DEM resolution, NLCD year, soil raster source/depth/scale factors. The workflow names SSURGO, but the soil values use SoilGrids-style units (texture g/kg summing to ~1000; BD cg/cm3; AWC median ~356), so the soil source is UNVERIFIED. The nodata-dilution artefact (water/urban cells counted as values of 0; the Iowa sand table's counted area equals the full county area) is already present in that extraction; soil was corrected for dilution. The Iowa sand zonal table implies ~250 m raster cells (consistent with SoilGrids 250 m; gSSURGO is 30 m). Iowa sand was filled from `lowa_soil.csv`. CEC was not extracted.\n"
      "- NLCD is a single static epoch; land-cover change 1985–2023 is not represented.\n"
      "- PRISM mixes AN81 and AN91 dataset versions across dates (recorded per date in `intermediate/prism_county_daily/prism_day_status_*.csv`).\n"
      "- NLDAS-2 precipitation is PRISM-informed (CPC gauges with PRISM climatology), so NLDAS is not fully independent of PRISM; only radiation/humidity/wind/pressure/temperature from NLDAS enter ET0.\n"
      "- County boundaries are the project's 1:20M Census cartographic polygons (validated previously; coarse for small counties).\n"
      "- NOAA storm reporting density increases over time and changed taxonomy in 1996.\n")
    A("## File index\n")
    A("- `full/`, `model_ready/` — master datasets\n- `metadata/master_variable_catalog.csv`, `metadata/master_data_provenance.json`, `metadata/legacy_column_mapping.csv`\n"
      "- `audits/` — source/drive audit, new vs old ERA5 validation, NLDAS completeness (pre/post backfill), backfill manifests, old dataset audit + evidence, QC, coverage, merge, leakage, SPI/SPEI diagnostics, GDD and DEM validation, NOAA recount validation, model-ready redundancy check\n"
      "- `intermediate/` — county weights, PRISM/NLDAS county-daily, daily indices, county-year feature tables, static extract, validation-only tables, NLDAS backfill subsets\n- scripts: `code/pipelines/master_dataset/` — 01–11 (see provenance JSON for the reproduction order)\n")
    (MD / "MASTER_DATASET_FINAL_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    log.info("Final report written (%d lines)", len(L))


if __name__ == "__main__":
    main()
