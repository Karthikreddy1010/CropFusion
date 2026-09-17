#!/usr/bin/env python3
"""
download_nldas.py

Resumable, polite downloader and in-flight daily aggregator for NASA NLDAS-2
Primary Forcing (NLDAS_FORA0125_H.2.0) across the 7-state study domain (1985–2023).

Design:
-------
1. Authenticates via NASA Earthdata Bearer Token (strictly read from environment / .env).
2. For each date D in chronological order:
   a. Streams 24 hourly NetCDF granules into scratch space.
   b. Spatially slices the 7-state study domain (lat: [35.5, 50.0], lon: [-104.5, -80.0]).
   c. Extracts the minimum meteorological variables required for FAO-56 Penman–Monteith:
      - Shortwave Radiation (SWdown -> Rs, MJ/m2/day)
      - Longwave Radiation (LWdown -> Rnl, MJ/m2/day)
      - 10m Wind Speed (Wind_E, Wind_N -> 2m wind u2, m/s)
      - Specific Humidity & Surface Pressure (Qair, PSurf -> actual vapor pressure ea, kPa)
      - Surface Pressure (PSurf -> psychrometric constant gamma, kPa/C)
      - 2m Air Temperature (Tair -> deg C, for QC vs PRISM)
   d. Saves daily aggregated NetCDF: raw/YYYY/nldas_daily_YYYYMMDD.nc (~0.5 MB).
   e. Immediately deletes the 24 hourly raw files to prevent disk accumulation.
   f. Records validated metrics and SHA-256 in the manifest.
3. Pre-flight dynamic disk space check ensures safe pause if storage is insufficient.
"""

from __future__ import annotations

import argparse
import datetime as dt
from datetime import datetime, date, timezone
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import requests
import xarray as xr
import numpy as np

from nldas_common import (
    load_config,
    resolve_path,
    setup_logger,
    daterange,
    get_earthdata_token,
    NLDASManifestRecord,
    NLDASManifestWriter,
    load_nldas_manifest_dates,
)

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_replace(src: Path, dst: Path):
    for _ in range(10):
        try:
            if dst.exists():
                try:
                    dst.unlink()
                except PermissionError:
                    time.sleep(0.5)
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(0.5)
    os.replace(src, dst)

def build_granule_url(cfg: dict, d: date, hour: int) -> Tuple[str, str]:
    """
    Constructs the official NASA GES DISC HTTPS URL and filename for a specific hour.
    Format:
    https://data.gesdisc.earthdata.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.2.0/YYYY/DOY/NLDAS_FORA0125_H.AYYYYMMDD.HH00.020.nc
    """
    base_url = cfg["nldas"]["base_url"]
    year = d.year
    doy = d.timetuple().tm_yday
    datestr = d.strftime("%Y%m%d")
    hourstr = f"{hour:02d}00"
    fname = f"NLDAS_FORA0125_H.A{datestr}.{hourstr}.020.nc"
    url = f"{base_url}/{year}/{doy:03d}/{fname}"
    return url, fname

def download_one_hour(
    cfg: dict,
    session: requests.Session,
    logger,
    url: str,
    target_path: Path,
    token: str,
) -> bool:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")

    net_cfg = cfg["network"]
    max_retries = net_cfg.get("max_retries", 5)
    timeout_s = net_cfg.get("timeout_seconds", 60)
    backoff_factor = net_cfg.get("backoff_factor", 2.0)
    backoff_max = net_cfg.get("backoff_max_seconds", 120)

    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": net_cfg.get("user_agent", "NLDAS-Downloader"),
    }

    attempt = 0
    while attempt <= max_retries:
        try:
            resp = session.get(url, headers=headers, stream=True, timeout=timeout_s)
            if resp.status_code == 401 or resp.status_code == 403:
                logger.error(f"AUTHENTICATION / EULA ERROR: HTTP {resp.status_code} on {url}. Check Bearer Token.")
                return False

            if resp.status_code != 200:
                raise requests.HTTPError(f"HTTP {resp.status_code}")

            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)

            # Minimal check: file > 100 KB
            if tmp_path.stat().st_size < 100 * 1024:
                raise ValueError("Downloaded file too small (possible error payload)")

            safe_replace(tmp_path, target_path)
            return True

        except Exception as e:
            attempt += 1
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            if attempt > max_retries:
                logger.error(f"Failed to download {url} after {max_retries} attempts: {e}")
                return False
            sleep_s = min(backoff_factor ** attempt, backoff_max)
            logger.warning(f"Retry {attempt}/{max_retries} for {target_path.name} after error: {e} (sleeping {sleep_s:.1f}s)")
            time.sleep(sleep_s)

    return False

def aggregate_day_to_daily_grid(
    hourly_files: list[Path],
    cfg: dict,
    out_nc_path: Path,
    logger,
) -> Tuple[bool, dict]:
    """
    Spatially subsets to study region and computes daily mean meteorology for FAO-56 ET0.
    """
    region = cfg["study_region"]
    lat_slice = slice(region["lat_min"], region["lat_max"])
    lon_slice = slice(region["lon_min"], region["lon_max"])

    # Load 24 hourly files
    try:
        datasets = []
        for f in sorted(hourly_files):
            ds = xr.open_dataset(f)
            # Spatial subset
            ds_sub = ds.sel(lat=lat_slice, lon=lon_slice)
            datasets.append(ds_sub)

        # Concatenate along time
        combined = xr.concat(datasets, dim="time")

        # Compute hourly derived fields:
        # 1. Wind speed u10 -> u2
        w10 = np.sqrt(combined["Wind_E"]**2 + combined["Wind_N"]**2)
        u2_hourly = 0.748 * w10

        # 2. Actual vapor pressure ea (kPa)
        q = combined["Qair"]
        p_pa = combined["PSurf"]
        ea_hourly = (q * p_pa / (0.622 + 0.378 * q)) / 1000.0

        # 3. Psychrometric constant gamma (kPa / deg C)
        gamma_hourly = 0.000665 * (p_pa / 1000.0)

        # Daily aggregations (mean over 24 hours):
        # Solar radiation: W/m2 -> daily total MJ/m2/day = mean(W/m2) * 0.0864
        sw_mean_w = combined["SWdown"].mean(dim="time", skipna=True)
        rs_daily_mj = sw_mean_w * 0.0864

        lw_mean_w = combined["LWdown"].mean(dim="time", skipna=True)
        u2_daily_mean = u2_hourly.mean(dim="time", skipna=True)
        ea_daily_mean = ea_hourly.mean(dim="time", skipna=True)
        psurf_daily_mean = p_pa.mean(dim="time", skipna=True)
        tair_daily_mean_k = combined["Tair"].mean(dim="time", skipna=True)
        tair_daily_min_k = combined["Tair"].min(dim="time", skipna=True)
        tair_daily_max_k = combined["Tair"].max(dim="time", skipna=True)

        # Daily dataset assembly
        daily_ds = xr.Dataset(
            data_vars={
                "SWdown_mean": sw_mean_w.astype("float32"),
                "Rs_daily_MJ": rs_daily_mj.astype("float32"),
                "LWdown_mean": lw_mean_w.astype("float32"),
                "u2_daily_mean": u2_daily_mean.astype("float32"),
                "ea_daily_mean_kPa": ea_daily_mean.astype("float32"),
                "PSurf_daily_mean_Pa": psurf_daily_mean.astype("float32"),
                "Tair_daily_mean_C": (tair_daily_mean_k - 273.15).astype("float32"),
                "Tair_daily_min_C": (tair_daily_min_k - 273.15).astype("float32"),
                "Tair_daily_max_C": (tair_daily_max_k - 273.15).astype("float32"),
            },
            coords={
                "lat": combined.lat,
                "lon": combined.lon,
            },
            attrs={
                "title": "NLDAS-2 Daily Aggregated Meteorological Forcing for FAO-56 ET0",
                "study_region": "Corn Belt 7 States (IL, IN, IA, MN, MO, NE, OH)",
                "source": "NASA GES DISC NLDAS_FORA0125_H.2.0",
                "creation_date": datetime.now(timezone.utc).isoformat(),
            }
        )

        out_nc_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_out = out_nc_path.with_suffix(".nc.part")
        daily_ds.to_netcdf(tmp_out, format="NETCDF4")
        safe_replace(tmp_out, out_nc_path)

        # Close all datasets
        for ds in datasets:
            ds.close()
        daily_ds.close()

        # Compute summary metrics for manifest
        metrics = {
            "rs_mean_w_m2": float(sw_mean_w.mean().values),
            "u2_mean_m_s": float(u2_daily_mean.mean().values),
            "ea_mean_kpa": float(ea_daily_mean.mean().values),
            "psurf_mean_pa": float(psurf_daily_mean.mean().values),
            "tair_mean_c": float((tair_daily_mean_k - 273.15).mean().values),
        }
        return True, metrics

    except Exception as e:
        logger.error(f"Error during daily aggregation: {e}")
        return False, {}

def main():
    parser = argparse.ArgumentParser(description="NLDAS-2 Daily Meteorological Forcing Downloader")
    parser.add_argument("--config", default="data/nldas_pipeline/config/nldas_config.yaml")
    parser.add_argument("--mode", choices=["validation_dates", "full", "range"], default="range")
    parser.add_argument("--start-date", help="YYYY-MM-DD (mode=range)")
    parser.add_argument("--end-date", help="YYYY-MM-DD (mode=range)")
    parser.add_argument("--dates", nargs="+", default=None, help="Explicit list of YYYY-MM-DD dates")
    parser.add_argument("--dry-run", action="store_true", help="Audit planned dates and storage without downloading")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("download_nldas", cfg, "download_nldas.log")

    # Retrieve token securely (never prints or exposes token)
    token = get_earthdata_token(cfg)
    logger.info("NASA Earthdata Token securely detected.")

    # Determine date list
    if args.dates:
        dates = sorted([datetime.strptime(s.strip(), "%Y-%m-%d").date() for s in args.dates])
    elif args.mode == "validation_dates":
        # Standard benchmark validation dates
        val_dates = ["1985-07-15", "1995-07-15", "2005-07-15", "2015-07-15", "2019-07-15", "2023-07-15"]
        dates = sorted([datetime.strptime(s, "%Y-%m-%d").date() for s in val_dates])
    elif args.mode == "full":
        dates = list(daterange(cfg["study_period"]["start_date"], cfg["study_period"]["end_date"]))
    else:
        if not args.start_date or not args.end_date:
            parser.error("--start-date and --end-date are required for mode=range (or provide --dates)")
        dates = list(daterange(args.start_date, args.end_date))

    manifest_path = resolve_path(cfg, "manifest.download_manifest_csv")
    already_done = load_nldas_manifest_dates(manifest_path)
    pending_dates = [d for d in dates if d.strftime("%Y-%m-%d") not in already_done]

    raw_dir = resolve_path(cfg, "paths.raw_dir")
    scratch_dir = resolve_path(cfg, "paths.scratch_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Dynamic Storage Evaluation
    total_b, used_b, free_b = shutil.disk_usage(raw_dir)
    daily_grid_mb = 0.52  # empirical size per daily aggregated netCDF
    est_needed_gb = (len(pending_dates) * daily_grid_mb) / 1024.0
    safety_margin_gb = cfg["storage_safety"].get("min_free_gb", 10.0)
    scratch_buffer_gb = 0.5  # 24 hours of raw netcdf is ~45 MB
    total_required_gb = est_needed_gb + scratch_buffer_gb + safety_margin_gb
    free_gb = free_b / (1024 ** 3)
    deficit_gb = max(0.0, total_required_gb - free_gb)

    logger.info("=" * 72)
    logger.info("NLDAS-2 DYNAMIC STORAGE & PRE-FLIGHT AUDIT")
    logger.info("=" * 72)
    logger.info(f"Target Storage Location     : {raw_dir} (Drive: {raw_dir.drive or str(raw_dir)})")
    logger.info(f"Total Disk Capacity         : {total_b / (1024**3):.2f} GB")
    logger.info(f"Available Free Space        : {free_gb:.2f} GB")
    logger.info(f"Planned Acquisition Dates   : {len(dates):,} days")
    logger.info(f"Already Validated in Manifest: {len(already_done):,} days")
    logger.info(f"Pending Dates to Ingest     : {len(pending_dates):,} days")
    logger.info(f"Estimated Daily Grids Volume: {est_needed_gb:.2f} GB")
    logger.info(f"Hourly Scratch Buffer       : {scratch_buffer_gb:.2f} GB (purged daily)")
    logger.info(f"Configured Safety Margin    : {safety_margin_gb:.2f} GB")
    logger.info(f"Total Projected Storage Req : {total_required_gb:.2f} GB")
    logger.info(f"Storage Deficit             : {deficit_gb:.2f} GB")
    logger.info(f"Storage Sufficient          : {'YES' if free_gb >= total_required_gb else 'NO'}")
    logger.info("=" * 72)

    if free_gb < total_required_gb:
        logger.error("INSUFFICIENT STORAGE — NLDAS DOWNLOAD PAUSED")
        print(f"\nINSUFFICIENT STORAGE: Need {total_required_gb:.2f} GB, have {free_gb:.2f} GB.", file=sys.stderr)
        return

    if args.dry_run:
        logger.info(f"[DRY RUN COMPLETE] {len(pending_dates)} dates would be processed.")
        return

    session = requests.Session()
    delay_s = cfg["network"].get("request_delay_seconds", 0.5)

    with NLDASManifestWriter(manifest_path) as manifest:
        for d in pending_dates:
            datestr = d.strftime("%Y-%m-%d")
            year = d.year
            logger.info(f"[{datestr}] Starting in-flight acquisition (24 hourly granules)...")

            day_scratch = scratch_dir / datestr
            day_scratch.mkdir(parents=True, exist_ok=True)
            hourly_paths = []
            day_success = True

            # Step 1: Download 24 hours
            for hour in range(24):
                url, fname = build_granule_url(cfg, d, hour)
                h_target = day_scratch / fname
                if not h_target.exists():
                    ok = download_one_hour(cfg, session, logger, url, h_target, token)
                    if not ok:
                        day_success = False
                        logger.error(f"[{datestr}] Hour {hour:02d}:00 failed to download.")
                        break
                    time.sleep(delay_s)
                hourly_paths.append(h_target)

            if not day_success or len(hourly_paths) < 24:
                logger.error(f"[{datestr}] FAILED: Incomplete hourly set ({len(hourly_paths)}/24). Discarding scratch.")
                shutil.rmtree(day_scratch, ignore_errors=True)
                manifest.write(NLDASManifestRecord(
                    date=datestr, status="FAILED", hours_downloaded=len(hourly_paths),
                    hours_expected=24, daily_grid_path="", error_message="Incomplete hourly retrieval"
                ))
                continue

            # Step 2: Aggregate to daily NetCDF grid
            out_nc = raw_dir / str(year) / f"nldas_daily_{d.strftime('%Y%m%d')}.nc"
            agg_ok, metrics = aggregate_day_to_daily_grid(hourly_paths, cfg, out_nc, logger)

            # Step 3: Evict scratch hourly files immediately
            shutil.rmtree(day_scratch, ignore_errors=True)

            if not agg_ok or not out_nc.exists():
                logger.error(f"[{datestr}] FAILED: Daily aggregation error.")
                manifest.write(NLDASManifestRecord(
                    date=datestr, status="FAILED", hours_downloaded=24,
                    hours_expected=24, daily_grid_path="", error_message="Aggregation computation error"
                ))
                continue

            # Step 4: Record success
            checksum = sha256_of(out_nc)
            record = NLDASManifestRecord(
                date=datestr,
                status="SUCCESS",
                hours_downloaded=24,
                hours_expected=24,
                daily_grid_path=str(out_nc),
                file_size_bytes=out_nc.stat().st_size,
                checksum_sha256=checksum,
                rs_mean_w_m2=metrics.get("rs_mean_w_m2"),
                u2_mean_m_s=metrics.get("u2_mean_m_s"),
                ea_mean_kpa=metrics.get("ea_mean_kpa"),
                psurf_mean_pa=metrics.get("psurf_mean_pa"),
                tair_mean_c=metrics.get("tair_mean_c"),
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            )
            manifest.write(record)
            logger.info(f"[{datestr}] SUCCESS -> {out_nc.name} ({out_nc.stat().st_size} bytes, sha256={checksum[:12]}...)")

    logger.info("NLDAS-2 Acquisition batch finished.")

if __name__ == "__main__":
    main()
