#!/usr/bin/env python3
"""
validate_storm_events.py

Independent Quality Control and Validation for the NOAA Storm Events pipeline:
  1. Event-level validation: missingness, duplicates, geography status breakdown.
  2. County-Year Skeleton validation (Part K):
     - Expected 26,364 rows (676 unique GEOIDs x 39 unique years).
     - 0 duplicate (GEOID, Year) records.
     - 0 missing GEOID, State, or Year values.
     - Distribution of storm_data_status (REPORTED_EVENTS, VALID_ZERO, RESTRICTED_TAXONOMY).
  3. Reporting coverage report:
     - Documents the documented 1996 taxonomy expansion break.

Outputs:
  - data/noaa_storm_pipeline/manifests/noaa_storm_quality_report.csv
  - data/noaa_storm_pipeline/manifests/storm_reporting_coverage_report.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from storm_common import load_config, resolve_path, setup_logger


def main():
    parser = argparse.ArgumentParser(description="QC for NOAA Storm Events pipeline")
    parser.add_argument("--config", default="config/storm_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("validate_storm_events", cfg, "validate_storm_events.log")

    processed_dir = resolve_path(cfg, "paths.processed_dir")
    clean_path = processed_dir / "noaa_storm_events_clean.parquet"
    county_year_path = processed_dir / "noaa_storm_county_year_1985_2023.parquet"

    if not clean_path.exists():
        logger.error(f"{clean_path} not found — run clean_storm_events.py first.")
        return

    df = pd.read_parquet(clean_path)
    logger.info(f"Loaded {len(df)} cleaned event rows")

    # --- Event-level Quality Report ---
    quality_rows = []
    for year, group in df.groupby("YEAR"):
        quality_rows.append({
            "year": int(year),
            "total_events": len(group),
            "county_direct": int((group["geography_status"] == "COUNTY_DIRECT").sum()),
            "zone_crosswalk_mapped": int((group["geography_status"] == "ZONE_CROSSWALK_MAPPED").sum()),
            "unresolved_geography": int((group["geography_status"] == "UNRESOLVED").sum()),
            "marine_excluded": int((group["geography_status"] == "MARINE_EXCLUDED").sum()),
            "missing_event_type": int(group["EVENT_TYPE"].isna().sum()),
            "missing_begin_time": int(group["BEGIN_DATE_TIME"].isna().sum()) if "BEGIN_DATE_TIME" in group else 0,
            "missing_end_time": int(group["END_DATE_TIME"].isna().sum()) if "END_DATE_TIME" in group else 0,
            "missing_damage_property": int(group["damage_property_usd"].isna().sum()),
            "missing_damage_crops": int(group["damage_crops_usd"].isna().sum()),
            "duplicate_event_ids": int(group["EVENT_ID"].duplicated().sum()) if "EVENT_ID" in group else 0,
        })
    quality_df = pd.DataFrame(quality_rows).sort_values("year")
    q_path = resolve_path(cfg, "manifest.quality_report_csv")
    quality_df.to_csv(q_path, index=False)
    logger.info(f"Wrote event quality report ({len(quality_df)} year-rows) to {q_path}")

    # --- Reporting coverage report: surfaces the 1996 structural break ---
    coverage_rows = []
    for year, group in df.groupby("YEAR"):
        regime = "RESTRICTED_TAXONOMY (Tornado/T-Wind/Hail only)" if year < 1996 else "FULL_TAXONOMY"
        coverage_rows.append({
            "year": int(year),
            "regime": regime,
            "total_events": len(group),
            "distinct_event_types": int(group["EVENT_TYPE"].nunique()),
            "counties_with_events": int(group.loc[group["GEOID"].notna(), "GEOID"].nunique()),
        })
    coverage_df = pd.DataFrame(coverage_rows).sort_values("year")
    c_path = resolve_path(cfg, "manifest.reporting_coverage_csv")
    coverage_df.to_csv(c_path, index=False)
    logger.info(f"Wrote reporting coverage report ({len(coverage_df)} year-rows) to {c_path}")

    # --- County-Year Skeleton Validation (Part K) ---
    if county_year_path.exists():
        cy_df = pd.read_parquet(county_year_path)
        logger.info(f"Loaded {len(cy_df)} county-year records from {county_year_path}")

        n_rows = len(cy_df)
        n_geoids = cy_df["GEOID"].nunique()
        n_years = cy_df["Year"].nunique()
        n_dups = int(cy_df.duplicated(subset=["GEOID", "Year"]).sum())
        n_null_geoid = int(cy_df["GEOID"].isna().sum())
        n_null_state = int(cy_df["State"].isna().sum())
        n_null_year = int(cy_df["Year"].isna().sum())

        logger.info(f"--- Part K County-Year Validation Checks ---")
        logger.info(f"Total rows: {n_rows} (Expected: 26364) -> {'PASS' if n_rows == 26364 else 'FAIL'}")
        logger.info(f"Unique GEOIDs: {n_geoids} (Expected: 676) -> {'PASS' if n_geoids == 676 else 'FAIL'}")
        logger.info(f"Unique Years: {n_years} (Expected: 39) -> {'PASS' if n_years == 39 else 'FAIL'}")
        logger.info(f"Duplicate (GEOID, Year): {n_dups} -> {'PASS' if n_dups == 0 else 'FAIL'}")
        logger.info(f"Missing GEOID: {n_null_geoid} -> {'PASS' if n_null_geoid == 0 else 'FAIL'}")
        logger.info(f"Missing State: {n_null_state} -> {'PASS' if n_null_state == 0 else 'FAIL'}")
        logger.info(f"Missing Year: {n_null_year} -> {'PASS' if n_null_year == 0 else 'FAIL'}")

        status_dist = cy_df["storm_data_status"].value_counts().to_dict()
        logger.info(f"storm_data_status distribution: {status_dist}")

        # Restricted taxonomy check (1985-1995)
        pre96 = cy_df[cy_df["Year"] < 1996]
        post96 = cy_df[cy_df["Year"] >= 1996]
        logger.info(f"Pre-1996 rows: {len(pre96)}, drought NaN rate: {pre96['drought_event_count'].isna().mean():.2%}")
        logger.info(f"Post-1996 rows: {len(post96)}, drought NaN rate: {post96['drought_event_count'].isna().mean():.2%}")
    else:
        logger.warning(f"{county_year_path} does not exist yet — run aggregate_storm_county_year.py")


if __name__ == "__main__":
    main()
