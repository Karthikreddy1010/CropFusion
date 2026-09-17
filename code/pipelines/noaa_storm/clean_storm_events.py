#!/usr/bin/env python3
"""
clean_storm_events.py

Parses raw yearly NOAA StormEvents_details files into a clean,
event-level parquet dataset with validated county/zone geography,
numeric tornado scales, event durations, and damage in USD.

Scientific Geography Handling:
  - CZ_TYPE == 'C' (County): Validated direct FIPS concatenation:
      GEOID = STATE_FIPS (2-digit zfill) + CZ_FIPS (3-digit zfill).
      geography_status = "COUNTY_DIRECT".
  - CZ_TYPE == 'Z' (Zone): Mapped using the official NOAA/NWS Public
      Zone-to-County crosswalk (bp18mr25.dbx). When a zone encompasses
      multiple counties, the storm event exposure is mapped to each
      constituent county.
      geography_status = "ZONE_CROSSWALK_MAPPED".
      If a zone cannot be resolved reliably, the record is preserved with:
      GEOID = None / NaN and geography_status = "UNRESOLVED".
  - CZ_TYPE == 'M' (Marine): Marine zones are preserved with:
      GEOID = None / NaN and geography_status = "MARINE_EXCLUDED".
"""

from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from storm_common import load_config, resolve_path, setup_logger


DAMAGE_RE = re.compile(r"^\s*([\d.]+)\s*([KMB]?)\s*$", re.IGNORECASE)


def parse_damage(value, suffix_multipliers: dict) -> float:
    if pd.isna(value):
        return float("nan")
    s = str(value).strip()
    if s == "" or s.upper() == "NA":
        return float("nan")
    if s in ("0", "0.00", "0.0"):
        return 0.0
    match = DAMAGE_RE.match(s)
    if not match:
        return float("nan")
    number_str, suffix = match.groups()
    try:
        number = float(number_str)
    except ValueError:
        return float("nan")
    multiplier = suffix_multipliers.get(suffix.upper(), 1.0)
    return float(number * multiplier)


def parse_tornado_scale(val) -> Optional[int]:
    if pd.isna(val):
        return None
    s = str(val).strip().upper()
    match = re.search(r"[EF]?F([0-5])", s)
    if match:
        return int(match.group(1))
    return None


def calculate_duration_hours(begin_dt_str, end_dt_str, year_val) -> float:
    if pd.isna(begin_dt_str) or pd.isna(end_dt_str):
        return float("nan")
    try:
        b = pd.to_datetime(str(begin_dt_str).strip(), format="%d-%b-%y %H:%M:%S", errors="coerce")
        e = pd.to_datetime(str(end_dt_str).strip(), format="%d-%b-%y %H:%M:%S", errors="coerce")
        if pd.isna(b) or pd.isna(e):
            return float("nan")
        # Ensure year aligns with observation year
        if pd.notna(year_val):
            y = int(year_val)
            if b.year != y:
                b = b.replace(year=y)
            if e.year != y and abs(e.year - y) > 1:
                e = e.replace(year=y)
        diff_hours = (e - b).total_seconds() / 3600.0
        return max(0.0, float(diff_hours))
    except Exception:
        return float("nan")


def load_zone_crosswalk(cfg: dict, logger) -> Optional[pd.DataFrame]:
    path = resolve_path(cfg, "paths.zone_county_crosswalk_csv")
    if not path.exists():
        logger.warning(
            f"No zone-county crosswalk found at {path}. Zone(Z)-type events will be marked UNRESOLVED."
        )
        return None
    xwalk = pd.read_csv(path, dtype=str)
    # The NWS file has columns: STATE, ZONE_NUMBER, CWA, ZONE_NAME, STATE_ZONE, COUNTY_NAME, FIPS...
    # Derive STATE_FIPS (first 2 digits of FIPS) and 3-digit ZONE
    xwalk["state_fips"] = xwalk["FIPS"].str.zfill(5).str[:2]
    xwalk["zone"] = xwalk["ZONE_NUMBER"].str.zfill(3)
    xwalk["county_geoid"] = xwalk["FIPS"].str.zfill(5)
    dedup = xwalk[["state_fips", "zone", "county_geoid", "COUNTY_NAME", "STATE"]].drop_duplicates()
    logger.info(f"Loaded NWS zone-county crosswalk: {len(dedup)} unique (state, zone, county) mappings from {path}")
    return dedup


def clean_one_year(cfg: dict, logger, raw_path: Path, zone_xwalk: Optional[pd.DataFrame]) -> pd.DataFrame:
    with gzip.open(raw_path, "rt", errors="replace") as f:
        df = pd.read_csv(f, low_memory=False)

    # Normalize column names to uppercase for consistent access
    df.columns = [c.upper() for c in df.columns]

    keep_cols = [
        "EVENT_ID", "EPISODE_ID", "YEAR", "STATE", "STATE_FIPS",
        "CZ_TYPE", "CZ_FIPS", "CZ_NAME", "EVENT_TYPE",
        "BEGIN_DATE_TIME", "END_DATE_TIME",
        "BEGIN_LAT", "BEGIN_LON", "END_LAT", "END_LON",
        "INJURIES_DIRECT", "INJURIES_INDIRECT",
        "DEATHS_DIRECT", "DEATHS_INDIRECT",
        "DAMAGE_PROPERTY", "DAMAGE_CROPS",
        "MAGNITUDE", "MAGNITUDE_TYPE",
        "TOR_F_SCALE", "TOR_LENGTH", "TOR_WIDTH",
    ]
    available = [c for c in keep_cols if c in df.columns]
    missing = set(keep_cols) - set(available)
    if missing:
        logger.warning(f"[{raw_path.name}] fields absent: {missing}")

    out = df[available].copy()

    # Preserve original damage strings
    if "DAMAGE_PROPERTY" in out.columns:
        out["DAMAGE_PROPERTY_RAW"] = out["DAMAGE_PROPERTY"].astype(str)
        mult = cfg["damage_parsing"]["suffix_multipliers"]
        out["damage_property_usd"] = out["DAMAGE_PROPERTY"].apply(lambda v: parse_damage(v, mult))
    else:
        out["damage_property_usd"] = np.nan

    if "DAMAGE_CROPS" in out.columns:
        out["DAMAGE_CROPS_RAW"] = out["DAMAGE_CROPS"].astype(str)
        mult = cfg["damage_parsing"]["suffix_multipliers"]
        out["damage_crops_usd"] = out["DAMAGE_CROPS"].apply(lambda v: parse_damage(v, mult))
    else:
        out["damage_crops_usd"] = np.nan

    # Tornado numeric scale
    if "TOR_F_SCALE" in out.columns:
        out["tornado_scale_numeric"] = out["TOR_F_SCALE"].apply(parse_tornado_scale)
    else:
        out["tornado_scale_numeric"] = None

    # Event duration (vectorized)
    if "BEGIN_DATE_TIME" in out.columns and "END_DATE_TIME" in out.columns:
        b_dt = pd.to_datetime(out["BEGIN_DATE_TIME"].astype(str).str.strip(), format="%d-%b-%y %H:%M:%S", errors="coerce")
        e_dt = pd.to_datetime(out["END_DATE_TIME"].astype(str).str.strip(), format="%d-%b-%y %H:%M:%S", errors="coerce")
        diff_hours = (e_dt - b_dt).dt.total_seconds() / 3600.0
        out["event_duration_hours"] = diff_hours.clip(lower=0.0)
    else:
        out["event_duration_hours"] = np.nan

    # Normalize CZ_FIPS and STATE_FIPS
    out["CZ_FIPS_STR"] = out["CZ_FIPS"].fillna("").astype(str).str.strip().apply(
        lambda s: str(int(float(s))).zfill(3) if s and s.replace(".", "").isdigit() else ""
    )
    out["STATE_FIPS_STR"] = out["STATE_FIPS"].fillna("").astype(str).str.strip().apply(
        lambda s: str(int(float(s))).zfill(2) if s and s.replace(".", "").isdigit() else ""
    )
    out["CZ_TYPE_CLEAN"] = out["CZ_TYPE"].fillna("").astype(str).str.strip().str.upper()

    # County-type (C)
    c_mask = out["CZ_TYPE_CLEAN"] == "C"
    out["GEOID"] = None
    out["geography_status"] = "UNRESOLVED"

    # For C-type: valid FIPS concatenation
    direct_geoid = out["STATE_FIPS_STR"] + out["CZ_FIPS_STR"]
    valid_c = c_mask & (out["STATE_FIPS_STR"] != "") & (out["CZ_FIPS_STR"] != "")
    out.loc[valid_c, "GEOID"] = direct_geoid[valid_c]
    out.loc[valid_c, "geography_status"] = "COUNTY_DIRECT"

    # Marine-type (M)
    m_mask = out["CZ_TYPE_CLEAN"] == "M"
    out.loc[m_mask, "geography_status"] = "MARINE_EXCLUDED"

    # Zone-type (Z)
    z_mask = out["CZ_TYPE_CLEAN"] == "Z"
    
    non_z_df = out[~z_mask].copy()
    z_df = out[z_mask].copy()

    if len(z_df) > 0 and zone_xwalk is not None:
        # Match zone crosswalk on state_fips and zone
        merged_z = z_df.merge(
            zone_xwalk[["state_fips", "zone", "county_geoid"]],
            left_on=["STATE_FIPS_STR", "CZ_FIPS_STR"],
            right_on=["state_fips", "zone"],
            how="left",
        )
        resolved_mask = merged_z["county_geoid"].notna()
        merged_z.loc[resolved_mask, "GEOID"] = merged_z.loc[resolved_mask, "county_geoid"]
        merged_z.loc[resolved_mask, "geography_status"] = "ZONE_CROSSWALK_MAPPED"
        merged_z.loc[~resolved_mask, "geography_status"] = "UNRESOLVED"
        merged_z = merged_z.drop(columns=["state_fips", "zone", "county_geoid"], errors="ignore")
        final_df = pd.concat([non_z_df, merged_z], ignore_index=True)
    else:
        z_df["geography_status"] = "UNRESOLVED"
        final_df = pd.concat([non_z_df, z_df], ignore_index=True)

    final_df = final_df.drop(columns=["CZ_FIPS_STR", "STATE_FIPS_STR", "CZ_TYPE_CLEAN"], errors="ignore")

    # Enforce explicit dtypes to avoid PyArrow schema collisions across years
    str_cols = [
        "EVENT_ID", "EPISODE_ID", "STATE", "STATE_FIPS", "CZ_TYPE", "CZ_FIPS", "CZ_NAME",
        "EVENT_TYPE", "BEGIN_DATE_TIME", "END_DATE_TIME", "DAMAGE_PROPERTY", "DAMAGE_CROPS",
        "DAMAGE_PROPERTY_RAW", "DAMAGE_CROPS_RAW", "MAGNITUDE_TYPE", "TOR_F_SCALE", "geography_status"
    ]
    for sc in str_cols:
        if sc in final_df.columns:
            final_df[sc] = final_df[sc].fillna("").astype(str)

    # GEOID as nullable string
    if "GEOID" in final_df.columns:
        final_df["GEOID"] = final_df["GEOID"].replace({"": None, np.nan: None})

    float_cols = [
        "BEGIN_LAT", "BEGIN_LON", "END_LAT", "END_LON",
        "INJURIES_DIRECT", "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT",
        "damage_property_usd", "damage_crops_usd", "MAGNITUDE", "TOR_LENGTH", "TOR_WIDTH",
        "tornado_scale_numeric", "event_duration_hours"
    ]
    for fc in float_cols:
        if fc in final_df.columns:
            final_df[fc] = pd.to_numeric(final_df[fc], errors="coerce").astype("float64")

    if "YEAR" in final_df.columns:
        final_df["YEAR"] = pd.to_numeric(final_df["YEAR"], errors="coerce").astype("Int64")

    return final_df


def main():
    parser = argparse.ArgumentParser(description="Clean and standardize raw StormEvents files")
    parser.add_argument("--config", default="config/storm_config.yaml")
    parser.add_argument("--years", nargs="+", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("clean_storm_events", cfg, "clean_storm_events.log")

    years = args.years
    if not years:
        years = list(range(cfg["study_period"]["start_year"], cfg["study_period"]["end_year"] + 1))

    raw_dir = resolve_path(cfg, "paths.raw_dir")
    zone_xwalk = load_zone_crosswalk(cfg, logger)

    frames = []
    for year in sorted(years):
        matches = list(raw_dir.glob(f"StormEvents_details-ftp_v1.0_d{year}_c*.csv.gz"))
        if not matches:
            logger.warning(f"[{year}] No raw file in {raw_dir} — run download_storm_events.py first.")
            continue
        raw_file = sorted(matches)[-1]
        logger.info(f"[{year}] Cleaning {raw_file.name}...")
        df_year = clean_one_year(cfg, logger, raw_file, zone_xwalk)
        frames.append(df_year)

    if not frames:
        logger.error("No years produced clean data. Nothing written.")
        return

    clean_df = pd.concat(frames, ignore_index=True)

    n_direct = int((clean_df["geography_status"] == "COUNTY_DIRECT").sum())
    n_mapped = int((clean_df["geography_status"] == "ZONE_CROSSWALK_MAPPED").sum())
    n_unresolved = int((clean_df["geography_status"] == "UNRESOLVED").sum())
    n_marine = int((clean_df["geography_status"] == "MARINE_EXCLUDED").sum())
    n_geoid_valid = int(clean_df["GEOID"].notna().sum())

    logger.info(
        f"Geography Resolution Summary: COUNTY_DIRECT={n_direct}, ZONE_CROSSWALK_MAPPED={n_mapped}, "
        f"UNRESOLVED={n_unresolved}, MARINE_EXCLUDED={n_marine} | "
        f"Total rows with valid GEOID: {n_geoid_valid}/{len(clean_df)}"
    )

    processed_dir = resolve_path(cfg, "paths.processed_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "noaa_storm_events_clean.parquet"
    clean_df.to_parquet(out_path, index=False)
    logger.info(f"Wrote {len(clean_df)} cleaned event records to {out_path}")


if __name__ == "__main__":
    main()
