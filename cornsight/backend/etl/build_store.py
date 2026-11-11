"""Build the CornSight data store from the Paper3 research outputs.

Reads Paper3 READ-ONLY and writes compact parquet/JSON files into backend/data_store/,
plus manifest.json with the sha256, size and modification time of every source, so each
number the platform shows can be traced to a file.

Usage:
    python etl/build_store.py                       # inside the repository: reads the repository root
    python etl/build_store.py --paper3 <research project root>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BU_ACRE_TO_T_HA = 0.0627  # Paper3 code/master_data_loader.py TONNES_PER_HA_PER_BU_ACRE
STATES = {"17": "IL", "18": "IN", "19": "IA", "27": "MN", "29": "MO", "39": "OH"}

SOURCES = {
    "panel_full": "master_dataset/full/Paper3_MasterDataset_PRISM_NLDAS_NOAA_USDA_FULL_1985_2023.parquet",
    "rolling_origin": "outputs_diagnostics/reports/rolling_origin_rows.csv",
    "attribution": "outputs_master/reports/shap_consistency_report.csv",
    "counties_geo": "outputs_master/geo_cache/counties_fips.json",
    "states_geo": "outputs_master/geo_cache/us_states.json",
    "config": "code/config.py",
}

STATIC_COLS = [
    "county_fips", "state", "state_name", "county", "county_lat", "county_lon", "county_land_area_m2",
    "soil_awc", "soil_bulk_density", "soil_clay", "soil_silt", "soil_sand", "soil_organic_carbon",
    "elevation_mean_m", "elevation_std_m", "slope_mean",
    "nlcd_cropland_frac", "nlcd_pasture_hay_frac", "nlcd_forest_frac", "nlcd_grass_shrub_frac",
    "nlcd_developed_frac", "nlcd_water_wetland_frac",
]
STAGES = ("veg", "silk", "grain")
YEAR_COLS = [
    "county_fips", "year", "corn_yield_bu_acre", "usda_yield_cv_pct",
    "usda_corn_planted_acres", "usda_corn_harvested_acres", "usda_corn_production_bu",
    # season (1 Apr - 31 Oct)
    "prism_tmax_mean_gs", "prism_tmax_max_gs", "prism_tmin_mean_gs", "prism_ppt_total_gs",
    "prism_gdd_gs", "prism_kdd30_gs", "prism_tmax_days_gt35_gs", "prism_heat_exceedance_total_gs",
    "nldas_et0_total_gs", "nldas_rs_mean_gs", "nldas_humidity_mean_gs", "nldas_vpd_mean_gs",
    "nldas_wind_mean_gs", "water_balance_total_gs",
    "spi1_prism_mean_gs", "spi1_prism_min_gs", "spi3_prism_mean_gs", "spi3_prism_min_gs",
    "spi6_prism_mean_gs", "spi6_prism_min_gs", "spi12_prism_mean_gs", "spi12_prism_min_gs",
    "spei30_prism_nldas_mean_gs", "spei30_prism_nldas_min_gs", "spei30_prism_nldas_drought_days_gs",
    "spei30_prism_nldas_drought_severity_gs", "spei90_prism_nldas_mean_gs", "spei90_prism_nldas_min_gs",
    "cdhw_count_gs", "cdhw_total_days_gs", "cdhw_max_duration_gs", "cdhw_max_severity_gs",
    "cdhw_cumulative_severity_gs",
    "noaa_hail_count_gs", "noaa_thunderstorm_wind_count_gs", "noaa_tornado_count_gs",
    "noaa_max_hail_size_gs", "noaa_heavy_rain_count_gs", "noaa_flood_count_gs", "noaa_storm_data_status",
    "usdm_dsci_gs_mean", "usdm_dsci_gs_max",
]
for s in STAGES:
    YEAR_COLS += [
        f"pheno_{s}_days", f"pheno_{s}_start_doy", f"pheno_{s}_end_doy",
        f"prism_tmax_mean_{s}", f"prism_tmax_days_gt35_{s}", f"prism_ppt_total_{s}",
        f"nldas_et0_total_{s}", f"water_balance_total_{s}", f"nldas_vpd_mean_{s}",
        f"spi1_prism_min_{s}", f"spei30_prism_nldas_mean_{s}", f"spei30_prism_nldas_min_{s}",
        f"cdhw_count_{s}", f"cdhw_total_days_{s}", f"cdhw_cumulative_severity_{s}",
    ]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def round_coords(geom: dict, nd: int = 4) -> dict:
    def r(c):
        return [round(c[0], nd), round(c[1], nd)] if isinstance(c[0], (int, float)) else [r(x) for x in c]
    return {"type": geom["type"], "coordinates": r(geom["coordinates"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper3", default=str(Path(__file__).resolve().parents[3]),
                    help="Research project root (default: the repository that contains cornsight/)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data_store"))
    a = ap.parse_args()
    root, out = Path(a.paper3), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "paper3_root": str(root), "sources": {}, "outputs": {}, "checks": {}}
    for key, rel in SOURCES.items():
        p = root / rel
        if not p.exists():
            print(f"MISSING source {key}: {p}", file=sys.stderr)
            return 2
        st = p.stat()
        manifest["sources"][key] = {"path": rel, "sha256": sha256(p), "bytes": st.st_size,
                                    "modified_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")}

    full = pd.read_parquet(root / SOURCES["panel_full"])
    full["county_fips"] = full["county_fips"].astype(str).str.zfill(5)
    full = full[full["county_fips"].str[:2].isin(STATES)]

    static = full.sort_values("year").groupby("county_fips")[STATIC_COLS[1:]].last().reset_index()
    static.to_parquet(out / "counties.parquet", index=False)

    cy = full[[c for c in YEAR_COLS if c in full.columns]].copy()
    missing = [c for c in YEAR_COLS if c not in full.columns]
    cy["yield_t_ha"] = cy["corn_yield_bu_acre"] * BU_ACRE_TO_T_HA
    cy.to_parquet(out / "county_year.parquet", index=False)

    ro = pd.read_csv(root / SOURCES["rolling_origin"])
    ro["county_fips"] = ro["GEOID"].astype(int).astype(str).str.zfill(5)
    ro = ro.rename(columns={"test_year": "year", "y": "observed_t_ha", "pred": "pred_t_ha",
                            "lo": "lower_t_ha", "hi": "upper_t_ha", "State": "state_abbr"})
    ro = ro[["county_fips", "year", "method", "stratum", "observed_t_ha", "pred_t_ha", "lower_t_ha", "upper_t_ha"]]
    ro.to_parquet(out / "predictions.parquet", index=False)

    # check: observed yield in predictions equals the panel yield after conversion
    chk = ro.merge(cy[["county_fips", "year", "yield_t_ha"]], on=["county_fips", "year"], how="left")
    manifest["checks"]["prediction_observed_matches_panel_max_abs_diff_t_ha"] = float((chk["observed_t_ha"] - chk["yield_t_ha"]).abs().max())
    manifest["checks"]["prediction_rows_without_panel_match"] = int(chk["yield_t_ha"].isna().sum())

    att = pd.read_csv(root / SOURCES["attribution"])
    att.to_csv(out / "attribution.csv", index=False)

    keep = set(static["county_fips"])
    g = json.loads((root / SOURCES["counties_geo"]).read_text(encoding="utf-8"))
    feats = []
    for f in g["features"]:
        fips = str(f["properties"]["STATE"]) + str(f["properties"]["COUNTY"])
        if fips in keep:
            feats.append({"type": "Feature", "id": fips,
                          "properties": {"fips": fips, "name": f["properties"]["NAME"]},
                          "geometry": round_coords(f["geometry"])})
    (out / "counties.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    sg = json.loads((root / SOURCES["states_geo"]).read_text(encoding="utf-8"))
    sfeats = []
    for f in sg["features"]:
        sf = str(f.get("id") or f["properties"].get("STATE") or f["properties"].get("STATEFP") or "").zfill(2)
        if sf in STATES:
            sfeats.append({"type": "Feature", "id": sf, "properties": {"fips": sf, "abbr": STATES[sf]},
                           "geometry": round_coords(f["geometry"])})
    (out / "states.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": sfeats}), encoding="utf-8")

    manifest["outputs"] = {
        "counties.parquet": len(static), "county_year.parquet": len(cy), "predictions.parquet": len(ro),
        "attribution.csv": len(att), "counties.geojson": len(feats), "states.geojson": len(sfeats),
    }
    manifest["checks"]["counties_in_panel"] = int(static["county_fips"].nunique())
    manifest["checks"]["counties_with_geometry"] = len(feats)
    manifest["checks"]["counties_with_predictions"] = int(ro["county_fips"].nunique())
    manifest["checks"]["year_columns_missing_from_panel"] = missing
    manifest["checks"]["prediction_years"] = [int(ro["year"].min()), int(ro["year"].max())]
    manifest["checks"]["panel_years"] = [int(cy["year"].min()), int(cy["year"].max())]
    manifest["constants"] = {"bu_acre_to_t_ha": BU_ACRE_TO_T_HA, "nominal_coverage": 0.90,
                             "gdd_vegetative_end": 700, "gdd_silking_end": 1400, "tmax_threshold_c": 35.0,
                             "spei_threshold": -1.0, "cdhw_moderate_min_days": 1, "cdhw_extreme_min_days": 3,
                             "growing_season": "1 Apr - 31 Oct"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"outputs": manifest["outputs"], "checks": manifest["checks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
