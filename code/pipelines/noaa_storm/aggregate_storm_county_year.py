#!/usr/bin/env python3
"""
aggregate_storm_county_year.py

Aggregates cleaned event-level NOAA Storm Events into the authoritative
county-year feature table.

Mandatory Protocol Requirements:
  1. Complete County-Year Skeleton:
     Constructed from the project's actual county list in Paper3_MegaDataset_SPEI_FINAL.csv:
     676 counties × 39 years (1985–2023) = 26,364 expected rows.
     Contains GEOID, County_Name, State, Year.
  2. Explicit Distinctions between Zero and Missing:
     - REPORTED_EVENTS: County has >= 1 reported storm event in that year.
     - VALID_ZERO: County has 0 reported storm events in a full-taxonomy year (>= 1996).
     - RESTRICTED_TAXONOMY: 1985–1995 regime where only Tornado, Hail, and
       Thunderstorm Wind were tracked. Untracked event categories (heat, drought,
       flood, winter) are set to NaN, never falsely imputed as 0.
  3. Numeric Tornado Intensity:
     Ordered integer scale (0–5) derived from F0-F5 and EF0-EF5.
  4. Durations & Damage:
     total_storm_event_hours, maximum_event_duration_hours,
     total_damage_property_usd, total_damage_crops_usd.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np

from storm_common import load_config, resolve_path, setup_logger


def build_county_year_skeleton(cfg: dict, logger) -> pd.DataFrame:
    mega_path = resolve_path(cfg, "paths.mega_dataset_csv")
    if not mega_path.exists():
        raise FileNotFoundError(f"Authoritative MegaDataset not found at {mega_path}")

    mega_df = pd.read_csv(mega_path, dtype={"GEOID": str}, usecols=["GEOID", "County_Name", "State"])
    mega_df["GEOID"] = mega_df["GEOID"].astype(str).str.zfill(5)
    counties = mega_df.drop_duplicates(subset=["GEOID"]).copy()

    start_year = cfg["study_period"]["start_year"]
    end_year = cfg["study_period"]["end_year"]
    years = list(range(start_year, end_year + 1))

    logger.info(f"Building county-year skeleton: {len(counties)} counties x {len(years)} years...")
    rows = []
    for yr in years:
        c_copy = counties.copy()
        c_copy["Year"] = yr
        rows.append(c_copy)

    skeleton = pd.concat(rows, ignore_index=True)
    expected_rows = len(counties) * len(years)
    assert len(skeleton) == expected_rows, f"Expected {expected_rows} skeleton rows, got {len(skeleton)}"
    logger.info(f"County-year skeleton built: {len(skeleton)} rows ({len(counties)} counties x {len(years)} years)")
    return skeleton


def normalize_wind_speed(magnitude, mag_type) -> float:
    if pd.isna(magnitude):
        return float("nan")
    try:
        val = float(magnitude)
        mtype = str(mag_type).strip().upper() if pd.notna(mag_type) else ""
        # 1 knot = 1.15078 mph; standardize to MPH
        if "KT" in mtype or "KNOT" in mtype:
            return float(val * 1.15078)
        return float(val)
    except (ValueError, TypeError):
        return float("nan")


def main():
    parser = argparse.ArgumentParser(description="Aggregate cleaned storm events to county-year skeleton")
    parser.add_argument("--config", default="config/storm_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("aggregate_storm_county_year", cfg, "aggregate_storm_county_year.log")

    processed_dir = resolve_path(cfg, "paths.processed_dir")
    clean_path = processed_dir / "noaa_storm_events_clean.parquet"
    if not clean_path.exists():
        logger.error(f"{clean_path} not found — run clean_storm_events.py first.")
        return

    # Load cleaned events
    events_df = pd.read_parquet(clean_path)
    logger.info(f"Loaded {len(events_df)} cleaned events from {clean_path}")

    # Keep only events with resolved GEOID
    events_with_geoid = events_df[events_df["GEOID"].notna()].copy()
    events_with_geoid["GEOID"] = events_with_geoid["GEOID"].astype(str).str.zfill(5)
    events_with_geoid["YEAR"] = events_with_geoid["YEAR"].astype(int)
    logger.info(f"{len(events_with_geoid)} events have valid GEOID and will be aggregated")

    # Filter events strictly to the 676 study counties
    skeleton = build_county_year_skeleton(cfg, logger)
    study_geoids = set(skeleton["GEOID"].unique())
    events_study = events_with_geoid[events_with_geoid["GEOID"].isin(study_geoids)].copy()
    logger.info(f"{len(events_study)} events belong to the 676 study counties and will be aggregated")

    # Categories from config
    cat_severe_conv = set(cfg["event_categories"]["SEVERE_CONVECTIVE"])
    cat_heat = set(cfg["event_categories"]["HEAT"])
    cat_flood = set(cfg["event_categories"]["PRECIPITATION_FLOOD"])
    cat_wind = set(cfg["event_categories"]["WIND"])
    cat_drought = set(cfg["event_categories"]["DROUGHT"])
    cat_winter = set(cfg["event_categories"]["WINTER"])

    # Vectorize wind speed (standardized to MPH)
    is_wind = events_study["EVENT_TYPE"].isin(["High Wind", "Thunderstorm Wind", "Strong Wind"])
    mag = pd.to_numeric(events_study["MAGNITUDE"], errors="coerce")
    mtype = events_study["MAGNITUDE_TYPE"].fillna("").astype(str).str.upper()
    is_knot = mtype.str.contains("KT|KNOT")
    events_study["wind_speed_mph"] = np.nan
    events_study.loc[is_wind & is_knot, "wind_speed_mph"] = mag * 1.15078
    events_study.loc[is_wind & ~is_knot, "wind_speed_mph"] = mag

    et = events_study["EVENT_TYPE"].astype(str)
    events_study["hail_size"] = np.where(et == "Hail", mag, np.nan)
    events_study["is_severe_conv"] = et.isin(cat_severe_conv).astype(int)
    events_study["is_hail"] = (et == "Hail").astype(int)
    events_study["is_tstm_wind"] = (et == "Thunderstorm Wind").astype(int)
    events_study["is_tornado"] = (et == "Tornado").astype(int)
    events_study["is_lightning"] = (et == "Lightning").astype(int)
    events_study["is_heavy_rain"] = (et == "Heavy Rain").astype(int)
    events_study["is_flood"] = (et == "Flood").astype(int)
    events_study["is_flash_flood"] = (et == "Flash Flood").astype(int)
    events_study["is_high_wind"] = (et == "High Wind").astype(int)
    events_study["is_strong_wind"] = (et == "Strong Wind").astype(int)
    events_study["is_heat"] = et.isin(cat_heat).astype(int)
    events_study["is_excessive_heat"] = (et == "Excessive Heat").astype(int)
    events_study["is_drought"] = et.isin(cat_drought).astype(int)
    events_study["is_winter"] = et.isin(cat_winter).astype(int)
    events_study["is_frost_freeze"] = (et == "Frost/Freeze").astype(int)

    events_study["tor_width"] = pd.to_numeric(events_study["TOR_WIDTH"], errors="coerce")
    events_study["tor_length"] = pd.to_numeric(events_study["TOR_LENGTH"], errors="coerce")
    events_study["tornado_scale"] = pd.to_numeric(events_study["tornado_scale_numeric"], errors="coerce")
    events_study["duration_hours"] = pd.to_numeric(events_study["event_duration_hours"], errors="coerce")

    # Vectorized GroupBy aggregation
    grouped = events_study.groupby(["GEOID", "YEAR"])
    agg_df = grouped.agg(
        storm_event_count=("EVENT_ID", "count"),
        unique_storm_episode_count=("EPISODE_ID", lambda s: s.nunique(dropna=True)),
        severe_convective_event_count=("is_severe_conv", "sum"),
        hail_event_count=("is_hail", "sum"),
        thunderstorm_wind_event_count=("is_tstm_wind", "sum"),
        tornado_event_count=("is_tornado", "sum"),
        lightning_event_count=("is_lightning", "sum"),
        heavy_rain_event_count=("is_heavy_rain", "sum"),
        flood_event_count=("is_flood", "sum"),
        flash_flood_event_count=("is_flash_flood", "sum"),
        high_wind_event_count=("is_high_wind", "sum"),
        strong_wind_event_count=("is_strong_wind", "sum"),
        heat_event_count=("is_heat", "sum"),
        excessive_heat_event_count=("is_excessive_heat", "sum"),
        drought_event_count=("is_drought", "sum"),
        winter_event_count=("is_winter", "sum"),
        frost_freeze_event_count=("is_frost_freeze", "sum"),
        maximum_hail_size=("hail_size", "max"),
        maximum_wind_magnitude=("wind_speed_mph", "max"),
        maximum_tornado_width=("tor_width", "max"),
        maximum_tornado_length=("tor_length", "max"),
        maximum_tornado_scale=("tornado_scale", "max"),
        total_damage_property_usd=("damage_property_usd", lambda s: s.sum(min_count=1)),
        total_damage_crops_usd=("damage_crops_usd", lambda s: s.sum(min_count=1)),
        total_storm_event_hours=("duration_hours", lambda s: s.sum(min_count=1)),
        maximum_event_duration_hours=("duration_hours", "max"),
    ).reset_index().rename(columns={"YEAR": "Year"})

    # Assemble onto mandatory county-year skeleton
    county_year_df = skeleton.merge(agg_df, on=["GEOID", "Year"], how="left")
    has_events = county_year_df["storm_event_count"].notna()
    is_pre_1996 = county_year_df["Year"] < 1996

    # Explicit data status classification
    county_year_df["storm_data_status"] = np.where(
        has_events,
        "REPORTED_EVENTS",
        np.where(is_pre_1996, "RESTRICTED_TAXONOMY", "VALID_ZERO")
    )

    # Base count columns that exist across all years (including 1985-1995)
    tracked_count_cols = [
        "storm_event_count", "unique_storm_episode_count", "severe_convective_event_count",
        "hail_event_count", "thunderstorm_wind_event_count", "tornado_event_count",
        "lightning_event_count", "high_wind_event_count", "strong_wind_event_count"
    ]
    for c in tracked_count_cols:
        county_year_df.loc[~has_events, c] = 0

    # Event types not tracked before 1996 MUST be NaN for pre-1996
    untracked_pre_1996_cols = [
        "heat_event_count", "excessive_heat_event_count", "drought_event_count",
        "winter_event_count", "frost_freeze_event_count", "flood_event_count",
        "flash_flood_event_count", "heavy_rain_event_count"
    ]
    for c in untracked_pre_1996_cols:
        county_year_df.loc[is_pre_1996, c] = np.nan

    # For post-1995 zero-event rows, all event counts are genuine 0
    post_1995_zero = (~has_events) & (~is_pre_1996)
    for c in untracked_pre_1996_cols:
        county_year_df.loc[post_1995_zero, c] = 0

    # Durations for zero-event rows are 0.0
    county_year_df.loc[~has_events, "total_storm_event_hours"] = 0.0
    county_year_df.loc[~has_events, "maximum_event_duration_hours"] = 0.0

    # Verification of skeleton integrity
    assert len(county_year_df) == 26364, f"Integrity check failed: Expected 26,364 rows, got {len(county_year_df)}"
    assert county_year_df["GEOID"].nunique() == 676, f"Expected 676 unique GEOIDs, got {county_year_df['GEOID'].nunique()}"
    assert county_year_df["Year"].nunique() == 39, f"Expected 39 years, got {county_year_df['Year'].nunique()}"
    assert not county_year_df.duplicated(subset=["GEOID", "Year"]).any(), "Found duplicate (GEOID, Year) records!"
    assert county_year_df["GEOID"].notna().all(), "Found null GEOID in skeleton!"
    assert county_year_df["State"].notna().all(), "Found null State in skeleton!"

    out_path = processed_dir / "noaa_storm_county_year_1985_2023.parquet"
    county_year_df.to_parquet(out_path, index=False)
    logger.info(f"Successfully saved {len(county_year_df)} county-year records to {out_path}")

    # Summary breakdown
    status_counts = county_year_df["storm_data_status"].value_counts().to_dict()
    logger.info(f"Storm Data Status Distribution: {status_counts}")


if __name__ == "__main__":
    main()
