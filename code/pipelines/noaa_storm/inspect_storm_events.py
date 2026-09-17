#!/usr/bin/env python3
"""
inspect_storm_events.py

Inspects a real downloaded StormEvents_details .csv.gz file and reports
its actual schema, dtypes, value distributions for EVENT_TYPE and
CZ_TYPE, timestamp format, and damage-field format — for comparison
against NOAA's documented schema (config: event_type_cz_designator).

This is the required "schema compatibility report" step (task Section
26) before trusting the full downloader/cleaner.

Usage
-----
    python code/pipelines/noaa_storm/inspect_storm_events.py --config config/storm_config.yaml \
        --years 2010 2015 2019 2023
"""

from __future__ import annotations

import argparse
import gzip

import pandas as pd

from storm_common import load_config, resolve_path, setup_logger


def load_year_df(cfg: dict, year: int) -> pd.DataFrame | None:
    raw_dir = resolve_path(cfg, "paths.raw_dir")
    matches = list(raw_dir.glob(f"StormEvents_details-ftp_v1.0_d{year}_c*.csv.gz"))
    if not matches:
        return None
    path = matches[0]
    with gzip.open(path, "rt", errors="replace") as f:
        df = pd.read_csv(f, low_memory=False)
    return df


def inspect_year(cfg: dict, logger, year: int) -> dict:
    df = load_year_df(cfg, year)
    if df is None:
        logger.error(f"[{year}] no downloaded file found in raw_dir — run the downloader first.")
        return {"year": year, "found": False}

    report = {"year": year, "found": True, "n_rows": len(df), "columns": list(df.columns)}
    logger.info(f"[{year}] {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"[{year}] columns: {list(df.columns)}")

    # Confirm real column names against what the config/task assumed
    lower_cols = {c.lower() for c in df.columns}
    expected_lower = {"event_id", "cz_type", "cz_fips", "state_fips", "event_type",
                       "begin_date_time", "end_date_time", "damage_property", "damage_crops"}
    missing = expected_lower - lower_cols
    if missing:
        logger.warning(f"[{year}] expected fields not found (case-insensitive): {missing}")
    report["missing_expected_fields"] = list(missing)

    cz_col = next((c for c in df.columns if c.lower() == "cz_type"), None)
    if cz_col:
        vc = df[cz_col].value_counts(dropna=False).to_dict()
        logger.info(f"[{year}] CZ_TYPE distribution: {vc}")
        report["cz_type_distribution"] = vc

    et_col = next((c for c in df.columns if c.lower() == "event_type"), None)
    if et_col:
        vc = df[et_col].value_counts(dropna=False).to_dict()
        logger.info(f"[{year}] {df[et_col].nunique()} distinct EVENT_TYPE values")
        report["event_type_distribution"] = vc

    dp_col = next((c for c in df.columns if c.lower() == "damage_property"), None)
    if dp_col:
        sample = df[dp_col].dropna().astype(str).unique()[:10]
        logger.info(f"[{year}] DAMAGE_PROPERTY sample values: {list(sample)}")
        report["damage_property_sample"] = list(sample)

    bdt_col = next((c for c in df.columns if c.lower() == "begin_date_time"), None)
    if bdt_col:
        sample = df[bdt_col].dropna().astype(str).unique()[:5]
        logger.info(f"[{year}] BEGIN_DATE_TIME sample values: {list(sample)}")
        report["begin_date_time_sample"] = list(sample)

    eid_col = next((c for c in df.columns if c.lower() == "event_id"), None)
    epid_col = next((c for c in df.columns if c.lower() == "episode_id"), None)
    if eid_col:
        report["n_unique_event_id"] = int(df[eid_col].nunique())
        report["n_duplicate_event_id"] = int(df[eid_col].duplicated().sum())
    if epid_col:
        report["n_unique_episode_id"] = int(df[epid_col].nunique())

    return report


def main():
    parser = argparse.ArgumentParser(description="Inspect real StormEvents files across years")
    parser.add_argument("--config", default="config/storm_config.yaml")
    parser.add_argument("--years", nargs="+", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("inspect_storm_events", cfg, "inspect_storm_events.log")

    reports = [inspect_year(cfg, logger, y) for y in args.years]

    # Cross-year schema compatibility: do all requested years share the
    # same column set? Report differences explicitly rather than assuming.
    found_reports = [r for r in reports if r.get("found")]
    if len(found_reports) >= 2:
        col_sets = {r["year"]: set(c.lower() for c in r["columns"]) for r in found_reports}
        base_year = list(col_sets.keys())[0]
        base_cols = col_sets[base_year]
        for year, cols in col_sets.items():
            only_in_this = cols - base_cols
            only_in_base = base_cols - cols
            if only_in_this or only_in_base:
                logger.warning(
                    f"[{year}] schema differs from {base_year}: "
                    f"extra={only_in_this or 'none'} missing={only_in_base or 'none'}"
                )
            else:
                logger.info(f"[{year}] schema matches {base_year}")

    out_path = resolve_path(cfg, "manifest.schema_compatibility_csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"year": r["year"], "found": r.get("found", False), "n_rows": r.get("n_rows"),
         "n_columns": len(r.get("columns", [])), "missing_expected_fields": r.get("missing_expected_fields")}
        for r in reports
    ]).to_csv(out_path, index=False)
    logger.info(f"Wrote schema compatibility summary to {out_path}")


if __name__ == "__main__":
    main()
