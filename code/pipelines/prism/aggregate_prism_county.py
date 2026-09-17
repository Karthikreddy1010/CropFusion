#!/usr/bin/env python3
"""
aggregate_prism_county.py

Aggregates raw PRISM daily grids (TMAX, TMIN, PPT) to county-level daily
values using exact fractional-area zonal extraction (exactextract).

Scientific Spatial Aggregation Method (Fixing Heuristic Limitations):
  - County Geometry: Loaded from authoritative local geo cache
    (outputs (1)/geo_cache/counties_fips.json), strictly matched against
    the 676 study counties by GEOID (STATE 2-digit + COUNTY 3-digit).
  - Raster Format: GeoTIFF (.tif) extracted from PRISM daily ZIP archives.
  - Spatial Reference: Reprojects county boundaries to match the native
    PRISM raster CRS (EPSG:4269 NAD83) prior to extraction.
  - Weighting Method: True fractional-area weighted zonal statistics via
    `exactextract` (exactextract-0.3.0), weighting each pixel by the exact
    fraction of its cell polygon intersecting each county boundary.
  - Nodata Handling: Nodata values (-9999.0) are strictly excluded from
    the weighted mean calculation, never treated as 0.

Output:
  - data/prism_pipeline/processed/prism_county_daily_1985_2023.parquet
    Columns: GEOID, County_Name, State, Date, Year, Month, Day, DOY, TMAX, TMIN, PPT
"""

from __future__ import annotations

import argparse
import zipfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
import exactextract

from prism_common import load_config, resolve_path, setup_logger, daterange


def load_county_boundaries(cfg: dict, logger) -> gpd.GeoDataFrame:
    """
    Loads county boundary polygons and restricts strictly to the 676
    study counties from the project's authoritative MegaDataset.
    """
    boundary_path = resolve_path(cfg, "paths.county_boundary_file")
    mega_csv_path = resolve_path(cfg, "paths.mega_dataset_csv")

    if not boundary_path.exists():
        raise FileNotFoundError(f"County boundary file not found at {boundary_path}")
    if not mega_csv_path.exists():
        raise FileNotFoundError(f"MegaDataset not found at {mega_csv_path}")

    counties_gdf = gpd.read_file(boundary_path)

    # Standardize GEOID
    if "STATE" in counties_gdf.columns and "COUNTY" in counties_gdf.columns:
        counties_gdf["GEOID"] = counties_gdf["STATE"].astype(str).str.zfill(2) + counties_gdf["COUNTY"].astype(str).str.zfill(3)
    elif "GEOID" in counties_gdf.columns:
        counties_gdf["GEOID"] = counties_gdf["GEOID"].astype(str).str.zfill(5)
    elif "GEO_ID" in counties_gdf.columns:
        counties_gdf["GEOID"] = counties_gdf["GEO_ID"].astype(str).str.split("US").str[-1].str.zfill(5)
    else:
        raise KeyError(f"No recognizable county GEOID column in {boundary_path}")

    mega_df = pd.read_csv(mega_csv_path, dtype={"GEOID": str}, usecols=["GEOID", "State", "County_Name"])
    mega_df["GEOID"] = mega_df["GEOID"].astype(str).str.zfill(5)
    study_counties = mega_df.drop_duplicates(subset=["GEOID"])
    study_geoids = set(study_counties["GEOID"])

    matched = counties_gdf[counties_gdf["GEOID"].isin(study_geoids)].copy().reset_index(drop=True)
    matched = matched.merge(study_counties, on="GEOID", how="inner")

    if len(matched) < len(study_geoids):
        missing = study_geoids - set(matched["GEOID"])
        raise ValueError(f"Matched only {len(matched)}/{len(study_geoids)} counties. Missing: {sorted(missing)[:10]}")

    logger.info(f"Loaded and verified {len(matched)} county polygons matching study counties. Base CRS: {matched.crs}")
    return matched


def extract_raster_from_zip(zip_path: Path, extract_dir: Path) -> Optional[Path]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
        names = zf.namelist()

    # Prioritize GeoTIFF, then BIL
    for ext in [".tif", ".tiff", ".bil", ".asc"]:
        for n in names:
            if n.lower().endswith(ext):
                return extract_dir / n
    return None


def exact_extract_for_date(raster_path: Path, counties_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Computes exact fractional-area weighted mean for each county polygon
    using exactextract.
    """
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    # Align CRS if necessary
    if counties_gdf.crs != raster_crs:
        target_gdf = counties_gdf.to_crs(raster_crs)
    else:
        target_gdf = counties_gdf

    # Run fractional-area extraction
    results = exactextract.exact_extract(str(raster_path), target_gdf, ["mean", "count"])

    out = counties_gdf[["GEOID", "State", "County_Name"]].copy().reset_index(drop=True)
    out["value_mean"] = [r["properties"]["mean"] for r in results]
    out["pixel_count"] = [r["properties"]["count"] for r in results]
    return out


def main():
    parser = argparse.ArgumentParser(description="Aggregate raw PRISM grids to county-daily values with exactextract")
    parser.add_argument("--config", default="config/prism_config.yaml")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--dates", nargs="+", default=None,
                        help="Explicit list of YYYY-MM-DD dates to aggregate")
    parser.add_argument("--variables", nargs="+", default=None)
    parser.add_argument("--mode", choices=["production", "staging"], default="staging",
                        help="Mode: 'production' requires all 14,244 dates before generating final dataset; 'staging' generates intermediate checkpoints")
    parser.add_argument("--all-available", action="store_true",
                        help="Scan raw_dir and aggregate all available dates on disk not yet in Parquet")
    parser.add_argument("--keep-scratch", action="store_true", default=False,
                        help="Keep uncompressed scratch GeoTIFFs on disk (default is to remove to save space)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("aggregate_prism_county", cfg, "aggregate_prism_county.log")
    variables = args.variables or cfg["prism"]["variables"]

    counties_gdf = load_county_boundaries(cfg, logger)

    raw_dir = resolve_path(cfg, "paths.raw_dir")
    scratch_dir = resolve_path(cfg, "paths.processed_dir") / "_scratch_extract"
    out_dir = resolve_path(cfg, "paths.county_daily_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "production":
        out_path = out_dir / "prism_county_daily_1985_2023.parquet"
    else:
        out_path = out_dir / "prism_county_daily_stage_checkpoint.parquet"

    already_done_dates = set()
    if out_path.exists():
        try:
            existing_pq = pd.read_parquet(out_path, columns=["Date"])
            already_done_dates = set(pd.to_datetime(existing_pq["Date"]).dt.date.unique())
        except Exception:
            pass

    # Determine date list
    if args.dates:
        dates = [pd.to_datetime(d).date() for d in args.dates]
    elif args.all_available:
        raw_dates = set()
        for v in variables:
            vdir = raw_dir / v
            if vdir.exists():
                for f in vdir.glob("*.zip"):
                    dstr = f.stem
                    try:
                        raw_dates.add(datetime.strptime(dstr, "%Y%m%d").date())
                    except ValueError:
                        pass
        dates = sorted(raw_dates - already_done_dates)
        logger.info(f"Auto-detected {len(raw_dates)} dates on disk. Already in Parquet: {len(already_done_dates)}. Pending: {len(dates)}")
    elif args.start_date and args.end_date:
        dates = list(daterange(args.start_date, args.end_date))
    else:
        dates = [pd.to_datetime(d).date() for d in cfg.get("validation_dates", [])]

    logger.info(f"Processing {len(dates)} dates across variables {variables}...")

    all_rows = []
    n_missing_files = 0

    for d in dates:
        datestr = d.strftime("%Y%m%d")
        day_frames = {}

        for variable in variables:
            zip_path = raw_dir / variable / f"{datestr}.zip"
            is_extracted_from_zip = False
            if not zip_path.exists():
                # Check unzipped directly or in processed
                alt_path = raw_dir / variable / f"{datestr}.tif"
                if not alt_path.exists():
                    n_missing_files += 1
                    continue
                raster_path = alt_path
            else:
                raster_path = extract_raster_from_zip(zip_path, scratch_dir / variable / datestr)
                is_extracted_from_zip = True

            if raster_path is None or not raster_path.exists():
                continue

            df = exact_extract_for_date(raster_path, counties_gdf)
            df = df.rename(columns={"value_mean": variable.upper(), "pixel_count": f"{variable}_pixel_count"})
            day_frames[variable] = df

            if is_extracted_from_zip and not args.keep_scratch:
                shutil.rmtree(scratch_dir / variable / datestr, ignore_errors=True)

        if not day_frames:
            continue

        merged = None
        for variable, df in day_frames.items():
            key_cols = ["GEOID", "State", "County_Name"]
            if merged is None:
                merged = df
            else:
                merged = merged.merge(df, on=key_cols, how="outer")

        merged["Date"] = d.isoformat()
        merged["Year"] = d.year
        merged["Month"] = d.month
        merged["Day"] = d.day
        merged["DOY"] = d.timetuple().tm_yday

        # Ensure all variable columns exist
        for v in ["TMAX", "TMIN", "PPT"]:
            if v not in merged.columns:
                merged[v] = np.nan

        # Clean column ordering
        req_cols = ["GEOID", "County_Name", "State", "Date", "Year", "Month", "Day", "DOY", "TMAX", "TMIN", "PPT"]
        extra_cols = [c for c in merged.columns if c not in req_cols]
        merged = merged[req_cols + extra_cols]
        all_rows.append(merged)

    if not all_rows:
        logger.warning(f"No county-day rows produced. Missing raw files: {n_missing_files}")
        return

    full_df = pd.concat(all_rows, ignore_index=True)

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        full_df = pd.concat([existing, full_df], ignore_index=True)
        full_df = full_df.drop_duplicates(subset=["GEOID", "Date"], keep="last")

    n_unique_dates = full_df['Date'].nunique()
    if args.mode == "production":
        if n_unique_dates < 14244:
            raise ValueError(
                f"PRODUCTION ERROR: Final production dataset requires all 14,244 dates. "
                f"Currently only {n_unique_dates} dates are present. Use --mode staging for intermediate checkpoints."
            )
        logger.info(f"PRODUCTION VERIFIED: Complete 14,244 dates present. Saving final dataset: {out_path}")
    else:
        logger.info(f"STAGING CHECKPOINT: Saving intermediate aggregation ({n_unique_dates}/14,244 dates) to {out_path}")

    full_df.to_parquet(out_path, index=False)
    logger.info(f"Successfully saved {len(full_df)} county-day rows to {out_path}")
    logger.info(f"Unique counties: {full_df['GEOID'].nunique()}, Unique dates: {n_unique_dates}")


if __name__ == "__main__":
    main()
