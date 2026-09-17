#!/usr/bin/env python3
"""
validate_combined_weather.py

Independent cross-dataset validation comparing:
  1. PRISM daily county weather (data/prism_pipeline/processed/prism_county_daily_1985_2023.parquet)
  2. NOAA CDO GHCND station county-daily weather (outputs/noaa_weather/noaa_daily_county_weather.csv)
  3. Master MegaDataset annual weather & ERA5 features (Paper3_MegaDataset_SPEI_FINAL.csv)

Outputs:
  - data/validation/weather_comparison/prism_vs_noaa_daily_comparison.csv
  - data/validation/weather_comparison/prism_vs_era5_annual_comparison.csv
  - data/validation/weather_comparison/weather_metrics_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats


def main():
    root = Path(__file__).resolve().parents[3]
    out_dir = root / "data" / "validation" / "weather_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    prism_path = root / "data" / "prism_pipeline" / "processed" / "prism_county_daily_1985_2023.parquet"
    noaa_daily_path = root / "outputs" / "noaa_weather" / "noaa_daily_county_weather.csv"
    mega_path = root / "Paper3_MegaDataset_SPEI_FINAL.csv"

    metrics = {}

    # --- 1. PRISM vs NOAA Daily Weather ---
    if prism_path.exists() and noaa_daily_path.exists():
        print("Comparing PRISM daily vs NOAA CDO daily...")
        prism_df = pd.read_parquet(prism_path)
        prism_df["GEOID"] = prism_df["GEOID"].astype(str).str.zfill(5)
        prism_df["Date"] = pd.to_datetime(prism_df["Date"]).dt.strftime("%Y-%m-%d")

        # Read matching dates from NOAA daily
        prism_dates = set(prism_df["Date"].unique())
        noaa_chunks = []
        for chunk in pd.read_csv(noaa_daily_path, dtype={"GEOID": str}, chunksize=200000):
            chunk["GEOID"] = chunk["GEOID"].astype(str).str.zfill(5)
            chunk["date"] = pd.to_datetime(chunk["date"]).dt.strftime("%Y-%m-%d")
            filtered = chunk[chunk["date"].isin(prism_dates)]
            if not filtered.empty:
                noaa_chunks.append(filtered)

        if noaa_chunks:
            noaa_df = pd.concat(noaa_chunks, ignore_index=True)
            merged_daily = prism_df.merge(
                noaa_df[["GEOID", "date", "TMAX_degC", "TMIN_degC", "PRCP_mm"]],
                left_on=["GEOID", "Date"],
                right_on=["GEOID", "date"],
                how="inner",
            )
            print(f"Matched {len(merged_daily)} county-days between PRISM and NOAA CDO.")

            daily_metrics = []
            for var_prism, var_noaa in [("TMAX", "TMAX_degC"), ("TMIN", "TMIN_degC"), ("PPT", "PRCP_mm")]:
                sub = merged_daily[[var_prism, var_noaa]].dropna()
                if len(sub) > 10:
                    p = sub[var_prism].values
                    n = sub[var_noaa].values
                    bias = float(np.mean(p - n))
                    mae = float(np.mean(np.abs(p - n)))
                    rmse = float(np.sqrt(np.mean((p - n) ** 2)))
                    r_pearson, _ = stats.pearsonr(p, n)
                    r_spearman, _ = stats.spearmanr(p, n)

                    daily_metrics.append({
                        "comparison": f"PRISM_{var_prism}_vs_NOAA_{var_noaa}",
                        "n_county_days": len(sub),
                        "mean_bias": bias,
                        "mae": mae,
                        "rmse": rmse,
                        "pearson_r": float(r_pearson),
                        "spearman_r": float(r_spearman),
                    })
            daily_df = pd.DataFrame(daily_metrics)
            daily_df.to_csv(out_dir / "prism_vs_noaa_daily_comparison.csv", index=False)
            metrics["daily_prism_vs_noaa"] = daily_metrics
            print(daily_df)
    else:
        print("PRISM daily or NOAA daily not found.")

    # --- 2. PRISM vs MegaDataset ERA5 Annual ---
    if prism_path.exists() and mega_path.exists():
        print("Comparing PRISM annual aggregated vs ERA5 annual...")
        prism_df = pd.read_parquet(prism_path)
        prism_df["GEOID"] = prism_df["GEOID"].astype(str).str.zfill(5)
        prism_df["Year"] = prism_df["Year"].astype(int)

        # Compute annual county metrics from PRISM
        annual_prism = prism_df.groupby(["GEOID", "Year"]).agg(
            PRISM_Tmax_mean=("TMAX", "mean"),
            PRISM_Tmax_max=("TMAX", "max"),
            PRISM_Tmin_mean=("TMIN", "mean"),
            PRISM_PPT_annual_sum=("PPT", "sum"),
            n_days=("TMAX", "count"),
        ).reset_index()

        mega_df = pd.read_csv(mega_path, dtype={"GEOID": str})
        mega_df["GEOID"] = mega_df["GEOID"].astype(str).str.zfill(5)
        mega_df["Year"] = mega_df["Year"].astype(int)

        merged_ann = annual_prism.merge(
            mega_df[["GEOID", "Year", "ERA5d_Tmax_mean_C", "ERA5d_Tmax_max_C", "ERA5d_Tmin_mean_C", "Precip_growseason_mm"]],
            on=["GEOID", "Year"],
            how="inner",
        )
        print(f"Matched {len(merged_ann)} county-years between PRISM and MegaDataset.")

        annual_metrics = []
        pairs = [
            ("PRISM_Tmax_mean", "ERA5d_Tmax_mean_C"),
            ("PRISM_Tmax_max", "ERA5d_Tmax_max_C"),
            ("PRISM_Tmin_mean", "ERA5d_Tmin_mean_C"),
        ]
        for p_col, e_col in pairs:
            sub = merged_ann[[p_col, e_col]].dropna()
            if len(sub) > 5:
                p = sub[p_col].values
                e = sub[e_col].values
                bias = float(np.mean(p - e))
                mae = float(np.mean(np.abs(p - e)))
                rmse = float(np.sqrt(np.mean((p - e) ** 2)))
                r_p, _ = stats.pearsonr(p, e)
                r_s, _ = stats.spearmanr(p, e)

                annual_metrics.append({
                    "comparison": f"{p_col}_vs_{e_col}",
                    "n_county_years": len(sub),
                    "mean_bias": bias,
                    "mae": mae,
                    "rmse": rmse,
                    "pearson_r": float(r_p),
                    "spearman_r": float(r_s),
                })

        ann_df = pd.DataFrame(annual_metrics)
        ann_df.to_csv(out_dir / "prism_vs_era5_annual_comparison.csv", index=False)
        metrics["annual_prism_vs_era5"] = annual_metrics
        print(ann_df)

    with open(out_dir / "weather_metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved weather metrics summary to {out_dir / 'weather_metrics_summary.json'}")


if __name__ == "__main__":
    main()
