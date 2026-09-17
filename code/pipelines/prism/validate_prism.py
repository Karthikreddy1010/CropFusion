#!/usr/bin/env python3
"""
validate_prism.py

Independent QC over the download manifest and the county-daily dataset.
Produces the required reports. Never deletes or "fixes" data — only
reports and flags, per the project's scientific rule against optimizing
results to look better.

Outputs
-------
    prism_quality_report.csv     — download-level QC (missing/dup/corrupt)
    prism_county_coverage.csv    — per county-year expected/available/missing days
    prism_validation_report.csv  — plausibility flags on weather values

Usage
-----
    python code/pipelines/prism/validate_prism.py --config config/prism_config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prism_common import load_config, resolve_path, setup_logger, daterange, expected_day_count


def download_level_qc(cfg: dict, logger) -> pd.DataFrame:
    manifest_path = resolve_path(cfg, "manifest.download_manifest_csv")
    if not manifest_path.exists():
        logger.error(f"No manifest found at {manifest_path} — nothing to validate yet.")
        return pd.DataFrame()

    df = pd.read_csv(manifest_path)
    logger.info(f"Loaded manifest with {len(df)} rows")

    start = cfg["study_period"]["start_date"]
    end = cfg["study_period"]["end_date"]
    all_dates = [d.isoformat() for d in daterange(start, end)]
    variables = cfg["prism"]["variables"]

    expected_index = pd.MultiIndex.from_product([all_dates, variables], names=["date", "variable"])
    expected_df = pd.DataFrame(index=expected_index).reset_index()

    dup_mask = df.duplicated(subset=["date", "variable"], keep=False)
    n_duplicates = int(dup_mask.sum())

    latest = (
        df.sort_values("download_timestamp")
        .drop_duplicates(subset=["date", "variable"], keep="last")
    )

    merged = expected_df.merge(latest, on=["date", "variable"], how="left", indicator=True)
    merged["present"] = merged["_merge"] == "both"

    n_expected = len(merged)
    n_present = int(merged["present"].sum())
    n_missing = n_expected - n_present
    n_success = int((merged["status"] == "SUCCESS").sum())
    n_failed = int((merged["status"] == "FAILED").sum())
    n_corrupt = int((merged["status"] == "CORRUPT").sum())

    logger.info(
        f"Expected={n_expected} present_in_manifest={n_present} missing_from_manifest={n_missing} "
        f"success={n_success} failed={n_failed} corrupt={n_corrupt} duplicate_manifest_rows={n_duplicates}"
    )

    by_year = merged.copy()
    by_year["year"] = pd.to_datetime(by_year["date"]).dt.year
    year_summary = (
        by_year.groupby(["year", "variable"])
        .agg(
            expected=("date", "count"),
            success=("status", lambda s: (s == "SUCCESS").sum()),
            failed=("status", lambda s: (s == "FAILED").sum()),
            corrupt=("status", lambda s: (s == "CORRUPT").sum()),
            missing_from_manifest=("present", lambda s: (~s).sum()),
        )
        .reset_index()
    )
    year_summary["coverage_fraction"] = year_summary["success"] / year_summary["expected"]

    out_path = resolve_path(cfg, "manifest.quality_report_csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    year_summary.to_csv(out_path, index=False)
    logger.info(f"Wrote download-level QC by year/variable to {out_path}")

    return merged


def county_coverage_qc(cfg: dict, logger) -> None:
    county_daily_path = resolve_path(cfg, "paths.county_daily_dir") / "prism_county_daily_1985_2023.parquet"
    if not county_daily_path.exists():
        logger.warning(f"{county_daily_path} not found — skipping county coverage QC (run aggregation first).")
        return

    df = pd.read_parquet(county_daily_path)
    df["Year"] = df["Year"].astype(int)

    rows = []
    for (geoid, state, county, year), group in df.groupby(["GEOID", "State", "County_Name", "Year"]):
        expected = expected_day_count(int(year))
        available_tmax = group["TMAX"].notna().sum() if "TMAX" in group else 0
        available_tmin = group["TMIN"].notna().sum() if "TMIN" in group else 0
        available_ppt = group["PPT"].notna().sum() if "PPT" in group else 0
        available_any = group[[c for c in ["TMAX", "TMIN", "PPT"] if c in group.columns]].notna().any(axis=1).sum()

        rows.append({
            "GEOID": geoid, "State": state, "County_Name": county, "Year": year,
            "expected_days": expected,
            "available_days_any_var": int(available_any),
            "available_days_tmax": int(available_tmax),
            "available_days_tmin": int(available_tmin),
            "available_days_ppt": int(available_ppt),
            "missing_days_any_var": expected - int(available_any),
            "coverage_fraction_any_var": available_any / expected,
        })

    coverage_df = pd.DataFrame(rows)
    out_path = resolve_path(cfg, "manifest.county_coverage_csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_df.to_csv(out_path, index=False)
    logger.info(
        f"Wrote county-year coverage for {coverage_df['GEOID'].nunique()} counties "
        f"({len(coverage_df)} county-year rows) to {out_path}"
    )
    logger.info(
        f"Coverage fraction — min={coverage_df['coverage_fraction_any_var'].min():.4f} "
        f"median={coverage_df['coverage_fraction_any_var'].median():.4f} "
        f"max={coverage_df['coverage_fraction_any_var'].max():.4f}"
    )

    # Coverage by Year
    manifest_dir = resolve_path(cfg, "paths.manifests_dir")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    by_year = coverage_df.groupby("Year").agg(
        total_counties=("GEOID", "nunique"),
        expected_days=("expected_days", "sum"),
        available_days_any=("available_days_any_var", "sum"),
        available_days_tmax=("available_days_tmax", "sum"),
        available_days_tmin=("available_days_tmin", "sum"),
        available_days_ppt=("available_days_ppt", "sum"),
    ).reset_index()
    by_year["coverage_fraction"] = by_year["available_days_any"] / by_year["expected_days"]
    by_year.to_csv(manifest_dir / "prism_coverage_by_year.csv", index=False)
    logger.info(f"Wrote coverage by year to {manifest_dir / 'prism_coverage_by_year.csv'}")

    # Coverage by State
    by_state = coverage_df.groupby("State").agg(
        total_counties=("GEOID", "nunique"),
        total_county_years=("Year", "count"),
        expected_days=("expected_days", "sum"),
        available_days_any=("available_days_any_var", "sum"),
        available_days_tmax=("available_days_tmax", "sum"),
        available_days_tmin=("available_days_tmin", "sum"),
        available_days_ppt=("available_days_ppt", "sum"),
    ).reset_index()
    by_state["coverage_fraction"] = by_state["available_days_any"] / by_state["expected_days"]
    by_state.to_csv(manifest_dir / "prism_coverage_by_state.csv", index=False)
    logger.info(f"Wrote coverage by state to {manifest_dir / 'prism_coverage_by_state.csv'}")

    # Coverage by Variable
    total_exp = coverage_df["expected_days"].sum()
    by_var = pd.DataFrame([
        {"variable": "TMAX", "available_days": coverage_df["available_days_tmax"].sum(), "expected_days": total_exp},
        {"variable": "TMIN", "available_days": coverage_df["available_days_tmin"].sum(), "expected_days": total_exp},
        {"variable": "PPT", "available_days": coverage_df["available_days_ppt"].sum(), "expected_days": total_exp},
    ])
    by_var["coverage_fraction"] = by_var["available_days"] / by_var["expected_days"]
    by_var.to_csv(manifest_dir / "prism_coverage_by_variable.csv", index=False)
    logger.info(f"Wrote coverage by variable to {manifest_dir / 'prism_coverage_by_variable.csv'}")


def plausibility_qc(cfg: dict, logger) -> None:
    county_daily_path = resolve_path(cfg, "paths.county_daily_dir") / "prism_county_daily_1985_2023.parquet"
    if not county_daily_path.exists():
        logger.warning(f"{county_daily_path} not found — skipping plausibility QC.")
        return

    df = pd.read_parquet(county_daily_path)
    v = cfg["validation"]
    flags = []

    def flag(mask, reason, colname):
        sub = df.loc[mask, ["GEOID", "State", "County_Name", "Date"]].copy()
        sub["variable"] = colname
        sub["reason"] = reason
        flags.append(sub)

    if "TMAX" in df.columns:
        flag(df["TMAX"] < v["tmax_min_c"], "below_plausible_min", "TMAX")
        flag(df["TMAX"] > v["tmax_max_c"], "above_plausible_max", "TMAX")
    if "TMIN" in df.columns:
        flag(df["TMIN"] < v["tmin_min_c"], "below_plausible_min", "TMIN")
        flag(df["TMIN"] > v["tmin_max_c"], "above_plausible_max", "TMIN")
    if "PPT" in df.columns:
        flag(df["PPT"] < v["ppt_min_mm"], "negative_precip", "PPT")
        flag(df["PPT"] > v["ppt_max_mm"], "above_plausible_max_flag_only", "PPT")
    if "TMAX" in df.columns and "TMIN" in df.columns:
        flag(df["TMIN"] > df["TMAX"], "tmin_exceeds_tmax", "TMIN_TMAX")

    dup_mask = df.duplicated(subset=["GEOID", "Date"], keep=False)
    flag(dup_mask, "duplicate_county_day", "ALL")

    if flags:
        flagged_df = pd.concat(flags, ignore_index=True)
    else:
        flagged_df = pd.DataFrame(columns=["GEOID", "State", "County_Name", "Date", "variable", "reason"])

    out_path = resolve_path(cfg, "manifest.validation_report_csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flagged_df.to_csv(out_path, index=False)
    logger.info(
        f"Plausibility QC flagged {len(flagged_df)} rows (flagged, NOT removed) -> {out_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run PRISM download + county QC")
    parser.add_argument("--config", default="config/prism_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("validate_prism", cfg, "validate_prism.log")

    download_level_qc(cfg, logger)
    county_coverage_qc(cfg, logger)
    plausibility_qc(cfg, logger)


if __name__ == "__main__":
    main()
