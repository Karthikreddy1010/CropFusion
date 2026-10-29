"""
noaa_weather_validator.py - Rigorous statistical comparison between official NOAA GHCND
observations and existing repository climate features (ERA5 / SPEI dataset).

Compares:
1. Maximum Temperature: NOAA TMAX_max_growseason_C vs ERA5d_Tmax_max_C
2. Growing Season Precipitation: NOAA Precip_total_growseason_mm vs Precip_growseason_mm
3. Corn GDD: NOAA GDD_accum_may_sep vs GDD_Accumulated

Breakdowns:
- Global (all 583 counties, 1985-2023, 22,737 county-years)
- By Temporal Split (FIT: 1985-2013, DEV: 2014-2015, CAL: 2016-2018, TEST: 2019-2023)
- By State (IL, IN, IA, MN, MO, OH)
- Historical shock years (1988, 1993, 2012)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("noaa_validator")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute comparison metrics: Pearson r, R2, RMSE, MAE, Mean Bias, std ratio."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    n = len(yt)
    if n < 3:
        return {"n": n, "r": np.nan, "r2": np.nan, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "true_mean": np.nan, "pred_mean": np.nan, "true_std": np.nan, "pred_std": np.nan}

    r, _ = stats.pearsonr(yt, yp)
    bias = float(np.mean(yp - yt))
    mae = float(np.mean(np.abs(yp - yt)))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    ss_res = float(np.sum((yt - yp) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-9 else np.nan

    return {
        "n": int(n),
        "r": round(float(r), 4),
        "r2": round(float(r2), 4),
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "bias": round(float(bias), 4),
        "true_mean": round(float(np.mean(yt)), 2),
        "pred_mean": round(float(np.mean(yp)), 2),
        "true_std": round(float(np.std(yt)), 2),
        "pred_std": round(float(np.std(yp)), 2),
    }


def run_validation(
    repo_data_path: str = "Paper3_MegaDataset_SPEI_FINAL.csv",
    noaa_annual_path: str = "outputs/noaa_weather/noaa_county_weather_annual.csv",
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    """Run full statistical comparison and generate figures and summary tables."""
    out_path = Path(output_dir)
    figures_dir = out_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load repository dataset
    logger.info("Loading repository dataset: %s", repo_data_path)
    repo_df = pd.read_csv(repo_data_path, low_memory=False)
    study_states = ["Illinois", "Indiana", "Iowa", "Minnesota", "Missouri", "Ohio"]
    repo_df = repo_df[repo_df["State"].isin(study_states)].copy()
    repo_df["GEOID"] = repo_df["GEOID"].astype(int).apply(lambda x: f"{x:05d}")

    # 2. Load NOAA annual aggregated dataset
    logger.info("Loading NOAA annual weather dataset: %s", noaa_annual_path)
    noaa_df = pd.read_csv(noaa_annual_path)
    noaa_df["GEOID"] = noaa_df["GEOID"].astype(int).apply(lambda x: f"{x:05d}")

    # 3. Merge datasets on GEOID and Year
    merged = pd.merge(
        repo_df,
        noaa_df,
        on=["GEOID", "Year"],
        suffixes=("_repo", "_noaa"),
        how="inner",
    )
    logger.info("Successfully merged %d county-year records across %d counties", len(merged), merged["GEOID"].nunique())

    # Define temporal splits
    def assign_split(yr: int) -> str:
        if 1985 <= yr <= 2013:
            return "FIT"
        elif 2014 <= yr <= 2015:
            return "DEV"
        elif 2016 <= yr <= 2018:
            return "CAL"
        elif 2019 <= yr <= 2023:
            return "TEST"
        return "OTHER"

    merged["split"] = merged["Year"].apply(assign_split)

    # Variables to compare: (label, repo_col, noaa_col, unit)
    comparisons = [
        ("Tmax_Max", "ERA5d_Tmax_max_C", "TMAX_max_growseason_C", "°C"),
        ("Precip_Total", "Precip_growseason_mm", "Precip_total_growseason_mm", "mm"),
        ("GDD_Accum", "GDD_Accumulated", "GDD_accum_may_sep", "°C-days"),
    ]

    results: Dict[str, Any] = {"global": {}, "splits": {}, "states": {}, "extremes": {}}

    # Global Metrics
    for label, r_col, n_col, unit in comparisons:
        if r_col in merged.columns and n_col in merged.columns:
            m = compute_metrics(merged[r_col].to_numpy(), merged[n_col].to_numpy())
            results["global"][label] = m
            logger.info("Global %s: r=%.4f, R2=%.4f, RMSE=%.3f %s, Bias=%.3f %s", label, m["r"], m["r2"], m["rmse"], unit, m["bias"], unit)

    # Temporal Split Metrics
    for split_name in ["FIT", "DEV", "CAL", "TEST"]:
        split_df = merged[merged["split"] == split_name]
        results["splits"][split_name] = {}
        for label, r_col, n_col, unit in comparisons:
            if r_col in split_df.columns and n_col in split_df.columns:
                m = compute_metrics(split_df[r_col].to_numpy(), split_df[n_col].to_numpy())
                results["splits"][split_name][label] = m

    # State Metrics
    for state_name in study_states:
        st_df = merged[merged["State"] == state_name]
        results["states"][state_name] = {}
        for label, r_col, n_col, unit in comparisons:
            if r_col in st_df.columns and n_col in st_df.columns:
                m = compute_metrics(st_df[r_col].to_numpy(), st_df[n_col].to_numpy())
                results["states"][state_name][label] = m

    # Shock Years Comparison (1988, 1993, 2012)
    shock_years = [1988, 1993, 2012]
    for yr in shock_years:
        yr_df = merged[merged["Year"] == yr]
        results["extremes"][yr] = {}
        for label, r_col, n_col, unit in comparisons:
            if r_col in yr_df.columns and n_col in yr_df.columns:
                m = compute_metrics(yr_df[r_col].to_numpy(), yr_df[n_col].to_numpy())
                results["extremes"][yr][label] = m

    # Generate Figures
    # Figure 1: 3-panel scatter comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    for ax, (label, r_col, n_col, unit) in zip(axes, comparisons):
        if r_col in merged.columns and n_col in merged.columns:
            yt = merged[r_col].to_numpy()
            yp = merged[n_col].to_numpy()
            mask = ~np.isnan(yt) & ~np.isnan(yp)
            ax.scatter(yt[mask], yp[mask], alpha=0.15, s=10, c="#1f77b4", edgecolors="none")
            min_val = min(np.percentile(yt[mask], 0.5), np.percentile(yp[mask], 0.5))
            max_val = max(np.percentile(yt[mask], 99.5), np.percentile(yp[mask], 99.5))
            ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="1:1 Line")
            m = results["global"][label]
            ax.set_title(f"{label} ({unit})\nr={m['r']:.3f}, R²={m['r2']:.3f}, RMSE={m['rmse']:.2f}, Bias={m['bias']:+.2f}", fontsize=12)
            ax.set_xlabel(f"Existing Dataset ({r_col})", fontsize=11)
            ax.set_ylabel(f"Official NOAA GHCND ({n_col})", fontsize=11)
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1_path = figures_dir / "noaa_vs_era5_comparison.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    logger.info("Saved comparison scatter plot -> %s", fig1_path)

    # Figure 2: Time series of regional growing season trajectory
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, dpi=300)
    ts = merged.groupby("Year").agg(
        noaa_tmax=("TMAX_max_growseason_C", "mean"),
        repo_tmax=("ERA5d_Tmax_max_C", "mean"),
        noaa_prcp=("Precip_total_growseason_mm", "mean"),
        repo_prcp=("Precip_growseason_mm", "mean"),
    ).reset_index()

    ax1.plot(ts["Year"], ts["noaa_tmax"], "b-o", label="NOAA GHCND Tmax Max", lw=2)
    ax1.plot(ts["Year"], ts["repo_tmax"], "r--s", label="Existing ERA5 Tmax Max", lw=2)
    ax1.axvspan(1985, 2013, color="blue", alpha=0.06, label="FIT (1985-2013)")
    ax1.axvspan(2014, 2015, color="green", alpha=0.08, label="DEV (2014-2015)")
    ax1.axvspan(2016, 2018, color="orange", alpha=0.08, label="CAL (2016-2018)")
    ax1.axvspan(2019, 2023, color="red", alpha=0.08, label="TEST (2019-2023)")
    ax1.set_ylabel("Peak Tmax (°C)", fontsize=11)
    ax1.set_title("Historical Trajectory of Corn Belt Growing Season Climate: NOAA GHCND vs ERA5 (1985–2023)", fontsize=13)
    ax1.legend(loc="upper left", ncol=3)
    ax1.grid(True, alpha=0.3)

    ax2.plot(ts["Year"], ts["noaa_prcp"], "b-o", label="NOAA GHCND Precip Total (mm)", lw=2)
    ax2.plot(ts["Year"], ts["repo_prcp"], "r--s", label="Existing ERA5 Precip Total (mm)", lw=2)
    ax2.set_ylabel("Total Precip (mm)", fontsize=11)
    ax2.set_xlabel("Year", fontsize=11)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2_path = figures_dir / "noaa_historical_timeseries.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    logger.info("Saved time series plot -> %s", fig2_path)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_validation()
