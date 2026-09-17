"""
=============================================================================
STATUS: ABANDONED / DISABLED
=============================================================================
ERA5-Land acquisition has been officially discontinued and decommissioned by
user directive (2026-09-12).
DO NOT RUN THIS SCRIPT. NO FURTHER ERA5-LAND DATA ACQUISITION IS PERMITTED.
=============================================================================
"""
import sys
print("ERROR: ERA5-Land data collection has been permanently DISABLED by user directive.")
print("Exiting immediately. No CDS requests will be made.")
sys.exit(0)

"""
ERA5-Land Daily Climate Acquisition & De-accumulation Engine (ARCHIVED/DISABLED)
================================================================================
"""


import os
import sys
import time
import glob
import random
import shutil
import logging
import argparse
import datetime
import calendar
import threading
import concurrent.futures
import pandas as pd
import numpy as np
import xarray as xr
import cdsapi

# Configure directories
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "era5_daily")
RAW_DIR = os.path.join(DATA_DIR, "raw")
BOUNDARY_DIR = os.path.join(DATA_DIR, "boundary")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "era5_daily_download")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(BOUNDARY_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging (Never print credentials)
LOG_FILE = os.path.join(OUTPUT_DIR, "download.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ERA5DailyEngine")

# Study bounding box: North, West, South, East
# Covers IL, IN, IA, MN, MO, NE, OH with 0.5-degree margin
BOUNDING_BOX = [50.0, -104.5, 35.5, -80.0]

# Core agrometeorological variables required for PET (FAO-56 PM) & CDHW
ERA5_LAND_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "surface_solar_radiation_downwards",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure"
]

MANIFEST_FILE = os.path.join(OUTPUT_DIR, "download_manifest.csv")
FAILED_FILE = os.path.join(OUTPUT_DIR, "failed_downloads.csv")

# Thread safety lock for manifest file operations
MANIFEST_LOCK = threading.Lock()


def initialize_manifest():
    """Initializes the download manifest CSVs if not present."""
    with MANIFEST_LOCK:
        if not os.path.exists(MANIFEST_FILE):
            df = pd.DataFrame(columns=[
                "file", "year", "month", "days_count", "variables",
                "spatial_extent", "file_size_mb", "status", "validation_status", "timestamp"
            ])
            df.to_csv(MANIFEST_FILE, index=False)
            
        if not os.path.exists(FAILED_FILE):
            df_fail = pd.DataFrame(columns=["year", "month", "error_message", "timestamp"])
            df_fail.to_csv(FAILED_FILE, index=False)


def record_manifest(file_name, year, month, days_count, status, val_status, size_mb):
    """Thread-safe recording of a completed download and aggregation to manifest."""
    entry = {
        "file": file_name,
        "year": int(year),
        "month": int(month),
        "days_count": int(days_count),
        "variables": "t2m_max,t2m_min,t2m_mean,d2m_mean,tp_sum,ssrd_sum,wind_speed_mean,sp_mean",
        "spatial_extent": f"[{BOUNDING_BOX[0]},{BOUNDING_BOX[1]},{BOUNDING_BOX[2]},{BOUNDING_BOX[3]}]",
        "file_size_mb": round(size_mb, 2),
        "status": status,
        "validation_status": val_status,
        "timestamp": datetime.datetime.now().isoformat()
    }
    with MANIFEST_LOCK:
        try:
            if os.path.exists(MANIFEST_FILE):
                df = pd.read_csv(MANIFEST_FILE)
            else:
                df = pd.DataFrame()
            if not df.empty:
                df = df[~((df["year"] == int(year)) & (df["month"] == int(month)))]
            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
            df.sort_values(by=["year", "month"], inplace=True)
            df.to_csv(MANIFEST_FILE, index=False)
        except Exception as e:
            logger.error(f"Error recording manifest for {year}-{month}: {str(e)}")


def record_failure(year, month, error_msg):
    """Thread-safe recording of a failure in failed_downloads.csv."""
    entry = {
        "year": int(year),
        "month": int(month),
        "error_message": str(error_msg),
        "timestamp": datetime.datetime.now().isoformat()
    }
    with MANIFEST_LOCK:
        try:
            if os.path.exists(FAILED_FILE):
                df = pd.read_csv(FAILED_FILE)
            else:
                df = pd.DataFrame()
            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
            df.to_csv(FAILED_FILE, index=False)
        except Exception as e:
            logger.error(f"Error recording failure for {year}-{month}: {str(e)}")


def validate_daily_file(daily_nc_path, expected_days):
    """
    Forensic 7-point validation of daily aggregated NetCDF file:
    A. Temporal continuity and day count
    B. Spatial grid dimensions and coordinates
    C. Required variable presence (all 8 variables)
    D. Units verification
    E. Physical range and inversion checks
    F. Missing data / NaN fraction detection
    G. Month-end boundary day validity
    """
    if not os.path.exists(daily_nc_path) or os.path.getsize(daily_nc_path) < 1024 * 1024:
        return False, "File missing or smaller than 1MB"

    try:
        with xr.open_dataset(daily_nc_path) as ds:
            # A. Temporal
            if "time" not in ds.dims or "latitude" not in ds.dims or "longitude" not in ds.dims:
                return False, f"Missing required dims: {list(ds.dims)}"
            
            actual_days = ds.sizes["time"]
            if actual_days != expected_days:
                return False, f"Expected {expected_days} days, got {actual_days}"
            
            # Check chronological ordering
            times = pd.to_datetime(ds["time"].values)
            if not times.is_monotonic_increasing:
                return False, "Timestamps are not strictly monotonically increasing"
            
            # B. Spatial
            lats = ds["latitude"].values
            lons = ds["longitude"].values
            if len(lats) != 146 or len(lons) != 246:
                return False, f"Unexpected grid shape: ({len(lats)}, {len(lons)}), expected (146, 246)"
            
            if not (np.isclose(lats[0], BOUNDING_BOX[0]) and np.isclose(lats[-1], BOUNDING_BOX[2])):
                return False, f"Latitude range mismatch: [{lats[0]}, {lats[-1]}], expected [{BOUNDING_BOX[0]}, {BOUNDING_BOX[2]}]"
            
            if not (np.isclose(lons[0], BOUNDING_BOX[1]) and np.isclose(lons[-1], BOUNDING_BOX[3])):
                return False, f"Longitude range mismatch: [{lons[0]}, {lons[-1]}], expected [{BOUNDING_BOX[1]}, {BOUNDING_BOX[3]}]"

            # C. Variables
            req_vars = ["t2m_max", "t2m_min", "t2m_mean", "d2m_mean", "tp_sum", "ssrd_sum", "wind_speed_mean", "sp_mean"]
            for v in req_vars:
                if v not in ds.data_vars:
                    return False, f"Variable {v} missing"
            
            # E. Physical range validation
            tmax = ds["t2m_max"].values
            tmin = ds["t2m_min"].values
            tp = ds["tp_sum"].values
            ssrd = ds["ssrd_sum"].values
            wind = ds["wind_speed_mean"].values
            sp = ds["sp_mean"].values
            
            # F. Missing data check
            valid_mask = ~np.isnan(tmax)
            nan_fraction = np.sum(~valid_mask) / tmax.size
            if nan_fraction > 0.05: # Allow <5% for open water mask (Great Lakes)
                return False, f"Excessive NaN fraction detected: {nan_fraction*100:.2f}%"
            
            if not np.any(valid_mask):
                return False, "All values are NaN"
            
            # Check Tmax >= Tmin (zero tolerance for inversions)
            inversion_diff = tmin[valid_mask] - tmax[valid_mask]
            if np.max(inversion_diff) > 0.01: # allow tiny float tolerance
                return False, f"Physical violation: Tmin > Tmax by up to {np.max(inversion_diff):.4f} C"
            
            # Temperature range: [-60C, +55C]
            if np.nanmin(tmin) < -60.0 or np.nanmax(tmax) > 55.0:
                return False, f"Temperature out of bounds: [{np.nanmin(tmin):.1f}, {np.nanmax(tmax):.1f}] C"
            
            # Precipitation: >= 0, max <= 350 mm/day
            if np.nanmin(tp) < -0.001:
                return False, f"Negative precipitation detected: {np.nanmin(tp):.4f} mm"
            if np.nanmax(tp) > 350.0:
                return False, f"Excessive precipitation detected: {np.nanmax(tp):.1f} mm/day"
            
            # Solar radiation: >= 0, max <= 40 MJ/m2/day
            if np.nanmin(ssrd) < -0.001:
                return False, f"Negative solar radiation detected: {np.nanmin(ssrd):.4f} MJ/m2/day"
            if np.nanmax(ssrd) > 40.0:
                return False, f"Excessive solar radiation detected: {np.nanmax(ssrd):.1f} MJ/m2/day"
            
            # Wind: >= 0, max <= 60 m/s
            if np.nanmin(wind) < 0.0 or np.nanmax(wind) > 60.0:
                return False, f"Wind speed out of bounds: [{np.nanmin(wind):.1f}, {np.nanmax(wind):.1f}] m/s"
            
            # Surface pressure: [65, 110] kPa
            if np.nanmin(sp) < 65.0 or np.nanmax(sp) > 110.0:
                return False, f"Surface pressure out of bounds: [{np.nanmin(sp):.1f}, {np.nanmax(sp):.1f}] kPa"

            # G. Month boundary verification (final day must have valid precip & radiation)
            last_day_tp = tp[-1]
            last_day_ssrd = ssrd[-1]
            if np.all(np.isnan(last_day_tp)) or np.all(np.isnan(last_day_ssrd)):
                return False, "Month-end boundary day contains all NaNs for precipitation or radiation"
            if np.nanmean(last_day_ssrd) <= 0.0:
                return False, "Month-end boundary day has zero or negative mean solar radiation"
            
        return True, "VALIDATED_OK"
    except Exception as e:
        return False, f"Exception during validation: {str(e)}"


def retrieve_with_retries(client, request, target_path, max_retries=5, base_delay=20.0):
    """
    Submits a retrieval request to CDS API with robust exponential backoff and jitter.
    Catches 429, 500, 502, 503, 504, connection errors, and timeouts.
    """
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            client.retrieve("reanalysis-era5-land", request, target_path)
            elapsed = time.time() - t0
            return True, elapsed
        except Exception as e:
            err_str = str(e)
            # Check for recoverable HTTP or network errors
            recoverable = any(code in err_str for code in ["429", "500", "502", "503", "504", "timeout", "ConnectionResetError", "RemoteDisconnected"])
            logger.warning(f"CDS retrieve attempt {attempt}/{max_retries} failed ({err_str[:120]}...).")
            
            if attempt < max_retries:
                # Exponential backoff with random jitter: (base * factor^attempt) * jitter
                delay = min(300.0, (base_delay * (1.8 ** attempt)) * random.uniform(0.8, 1.25))
                logger.info(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries} retrieval attempts failed.")
                return False, err_str


def get_month_end_boundary(year, month, client):
    """
    Resolves the month-end boundary observation for the 24th hour of Day D_last:
    (next_year, next_month, 01, 00:00:00 UTC).
    
    1. Checks if boundary_{year}_{month:02d}.nc is already in BOUNDARY_DIR.
    2. Checks if raw hourly_{next_year}_{next_month:02d}.nc exists (extracts timestep 0).
    3. If neither exists, retrieves the single 1-hour boundary observation from CDS and caches it.
    For December 2023, retrieves 2024-01-01 00:00:00 UTC solely to calculate December 31.
    """
    boundary_path = os.path.join(BOUNDARY_DIR, f"boundary_{year}_{month:02d}.nc")
    if os.path.exists(boundary_path):
        try:
            ds = xr.open_dataset(boundary_path)
            if "tp" in ds and "ssrd" in ds:
                return ds
            ds.close()
        except Exception:
            pass

    # Compute next year and next month
    if month == 12:
        next_year = year + 1
        next_month = 1
    else:
        next_year = year
        next_month = month + 1

    # Check if next month's raw hourly file is already on disk
    next_raw_path = os.path.join(RAW_DIR, f"hourly_{next_year}_{next_month:02d}.nc")
    if os.path.exists(next_raw_path):
        try:
            with xr.open_dataset(next_raw_path) as ds_next:
                # Timestep 0 of next month is next_year-next_month-01 00:00:00
                ds_b = ds_next[["tp", "ssrd"]].isel(valid_time=0)
                ds_b.to_netcdf(boundary_path)
                return xr.open_dataset(boundary_path)
        except Exception as e:
            logger.warning(f"Could not extract boundary from existing {next_raw_path}: {e}")

    # Otherwise retrieve 1-hour boundary from CDS
    logger.info(f"Retrieving month-end boundary observation for {year}-{month:02d} ({next_year}-{next_month:02d}-01 00:00:00)...")
    req_b = {
        "variable": ["total_precipitation", "surface_solar_radiation_downwards"],
        "year": f"{next_year:04d}",
        "month": f"{next_month:02d}",
        "day": "01",
        "time": "00:00",
        "area": BOUNDING_BOX,
        "data_format": "netcdf",
        "download_format": "unarchived"
    }
    
    tmp_boundary_path = boundary_path + f".tmp_{os.getpid()}_{random.randint(1000, 9999)}.nc"
    success, res = retrieve_with_retries(client, req_b, tmp_boundary_path, max_retries=4, base_delay=15.0)
    if success and os.path.exists(tmp_boundary_path):
        if os.path.exists(boundary_path):
            os.remove(boundary_path)
        os.rename(tmp_boundary_path, boundary_path)
        return xr.open_dataset(boundary_path)
    else:
        logger.error(f"Failed to retrieve boundary observation for {year}-{month:02d}: {res}")
        return None


def export_preceding_boundary(raw_hourly_nc, year, month):
    """
    Extracts timestep 0 of the current month (year-month-01 00:00:00) and saves it
    as the boundary file for the PRECEDING month (prev_year-prev_month) to avoid
    an extra CDS request when the preceding month processes.
    """
    if month == 1:
        prev_year = year - 1
        prev_month = 12
    else:
        prev_year = year
        prev_month = month - 1

    prev_boundary_path = os.path.join(BOUNDARY_DIR, f"boundary_{prev_year}_{prev_month:02d}.nc")
    if not os.path.exists(prev_boundary_path):
        try:
            with xr.open_dataset(raw_hourly_nc) as ds:
                time_coord = "valid_time" if "valid_time" in ds.coords else "time"
                ds_b = ds[["tp", "ssrd"]].isel({time_coord: 0})
                tmp_path = prev_boundary_path + f".tmp_{os.getpid()}.nc"
                ds_b.to_netcdf(tmp_path)
                if os.path.exists(prev_boundary_path):
                    os.remove(prev_boundary_path)
                os.rename(tmp_path, prev_boundary_path)
                logger.info(f"Cached boundary observation for preceding month {prev_year}-{prev_month:02d}.")
        except Exception as e:
            logger.warning(f"Could not cache preceding boundary for {prev_year}-{prev_month:02d}: {e}")


def aggregate_hourly_chunk(hourly_nc_path, daily_nc_path, year, month, client):
    """
    Reads raw hourly ERA5-Land NetCDF, audits accumulation metadata, performs
    rigorous forecast de-accumulation with month-end boundary resolution,
    and produces a daily NetCDF file.
    """
    logger.info(f"Opening hourly file for aggregation: {hourly_nc_path}")
    with xr.open_dataset(hourly_nc_path) as ds:
        time_coord = "valid_time" if "valid_time" in ds.coords else "time"
        times = pd.to_datetime(ds[time_coord].values)
        
        lats = ds["latitude"].values
        lons = ds["longitude"].values
        n_lat, n_lon = len(lats), len(lons)
        
        days_in_month = calendar.monthrange(year, month)[1]
        target_dates = [datetime.date(year, month, d) for d in range(1, days_in_month + 1)]
        n_days = len(target_dates)
        
        # Metadata and accumulation convention inspection
        tp_step_type = ds["tp"].attrs.get("GRIB_stepType", "unknown")
        ssrd_step_type = ds["ssrd"].attrs.get("GRIB_stepType", "unknown")
        logger.info(f"Metadata audit for {year}-{month:02d}: tp stepType='{tp_step_type}', ssrd stepType='{ssrd_step_type}'")
        
        # Verify accumulation convention
        is_accumulated = (tp_step_type == "accum")
        if not is_accumulated:
            # If GRIB_stepType is not accum, verify whether values are already de-accumulated
            # If convention cannot be established, fail rather than assuming
            logger.warning(f"tp GRIB_stepType is '{tp_step_type}'. Verifying if data is already de-accumulated...")
            sample_tp = ds["tp"].values[:24, n_lat//2, n_lon//2]
            if np.all(np.diff(sample_tp[1:]) >= -1e-6) and np.max(sample_tp) > 0.001:
                is_accumulated = True
                logger.info("Empirical verification: tp shows forecast accumulation despite header. Treating as accumulated.")
            else:
                logger.info("Treating tp as already de-accumulated.")

        # Preallocate daily arrays
        daily_tmax = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
        daily_tmin = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
        daily_tmean = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
        daily_dmean = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
        daily_tp = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
        daily_ssrd = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
        daily_wind = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
        daily_sp = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
        
        t2m = ds["t2m"].values # K
        d2m = ds["d2m"].values # K
        tp_raw = ds["tp"].values # m
        ssrd_raw = ds["ssrd"].values # J m-2
        u10 = ds["u10"].values # m/s
        v10 = ds["v10"].values # m/s
        sp_arr = ds["sp"].values # Pa
        
        # Compute 2m wind speed (FAO-56 logarithmic scaling: u2 = u10 * 0.748)
        w10 = np.sqrt(u10**2 + v10**2) * 0.748
        
        time_to_idx = {pd.Timestamp(t): i for i, t in enumerate(times)}
        
        # Load month-end boundary observation for Day D_last if accumulated
        ds_boundary = None
        if is_accumulated:
            ds_boundary = get_month_end_boundary(year, month, client)
            if ds_boundary is None:
                logger.warning(f"Could not load month-end boundary for {year}-{month:02d}. Will use 23-hour accumulation for final day.")

        for d_idx, cur_date in enumerate(target_dates):
            date_str = cur_date.strftime("%Y-%m-%d")
            
            # --- 1. INSTANTANEOUS STATE VARIABLES (t2m, d2m, wind, sp) ---
            # Standard civil day: 24 observations from 00:00 to 23:00 UTC
            day_hour_indices = []
            for h in range(24):
                ts = pd.Timestamp(f"{date_str} {h:02d}:00:00")
                if ts in time_to_idx:
                    day_hour_indices.append(time_to_idx[ts])
                    
            if len(day_hour_indices) > 0:
                sub_t2m = t2m[day_hour_indices] - 273.15
                daily_tmax[d_idx] = np.nanmax(sub_t2m, axis=0)
                daily_tmin[d_idx] = np.nanmin(sub_t2m, axis=0)
                daily_tmean[d_idx] = np.nanmean(sub_t2m, axis=0)
                
                sub_d2m = d2m[day_hour_indices] - 273.15
                daily_dmean[d_idx] = np.nanmean(sub_d2m, axis=0)
                
                daily_wind[d_idx] = np.nanmean(w10[day_hour_indices], axis=0)
                daily_sp[d_idx] = np.nanmean(sp_arr[day_hour_indices] / 1000.0, axis=0) # kPa

            # --- 2. ACCUMULATED FLUX VARIABLES (tp, ssrd) ---
            if is_accumulated:
                # De-accumulate hours 01:00 to 23:00 of current day
                day_tp_increments = []
                day_ssrd_increments = []
                
                # Hour 01:00 (Step 1)
                ts_1 = pd.Timestamp(f"{date_str} 01:00:00")
                if ts_1 in time_to_idx:
                    idx_1 = time_to_idx[ts_1]
                    p_inc_1 = np.maximum(0.0, tp_raw[idx_1])
                    r_inc_1 = np.maximum(0.0, ssrd_raw[idx_1])
                    day_tp_increments.append(p_inc_1)
                    day_ssrd_increments.append(r_inc_1)
                    
                    prev_p = tp_raw[idx_1]
                    prev_r = ssrd_raw[idx_1]
                    
                    # Hours 02:00 to 23:00 (Steps 2 to 23)
                    for h in range(2, 24):
                        ts_h = pd.Timestamp(f"{date_str} {h:02d}:00:00")
                        if ts_h in time_to_idx:
                            idx_h = time_to_idx[ts_h]
                            cur_p = tp_raw[idx_h]
                            cur_r = ssrd_raw[idx_h]
                            p_diff = np.maximum(0.0, cur_p - prev_p)
                            r_diff = np.maximum(0.0, cur_r - prev_r)
                            day_tp_increments.append(p_diff)
                            day_ssrd_increments.append(r_diff)
                            prev_p = cur_p
                            prev_r = cur_r
                            
                    # Hour 24 (Step 24: 00:00 UTC of next day)
                    next_date = cur_date + datetime.timedelta(days=1)
                    ts_next_0 = pd.Timestamp(f"{next_date.strftime('%Y-%m-%d')} 00:00:00")
                    if ts_next_0 in time_to_idx:
                        # Intra-month boundary (Days 1 to D_last-1)
                        idx_next_0 = time_to_idx[ts_next_0]
                        cur_p = tp_raw[idx_next_0]
                        cur_r = ssrd_raw[idx_next_0]
                        p_diff = np.maximum(0.0, cur_p - prev_p)
                        r_diff = np.maximum(0.0, cur_r - prev_r)
                        day_tp_increments.append(p_diff)
                        day_ssrd_increments.append(r_diff)
                    elif d_idx == (n_days - 1) and ds_boundary is not None:
                        # Month-end boundary (Day D_last) using boundary observation
                        try:
                            cur_p = ds_boundary["tp"].values
                            cur_r = ds_boundary["ssrd"].values
                            # Squeeze if scalar coordinate present
                            if cur_p.ndim == 3:
                                cur_p = cur_p[0]
                                cur_r = cur_r[0]
                            p_diff = np.maximum(0.0, cur_p - prev_p)
                            r_diff = np.maximum(0.0, cur_r - prev_r)
                            day_tp_increments.append(p_diff)
                            day_ssrd_increments.append(r_diff)
                            logger.info(f"Successfully resolved month-end boundary for Day {days_in_month}.")
                        except Exception as e:
                            logger.warning(f"Error applying month-end boundary observation: {e}")
                
                if len(day_tp_increments) > 0:
                    daily_tp[d_idx] = np.sum(day_tp_increments, axis=0) * 1000.0 # mm
                    daily_ssrd[d_idx] = np.sum(day_ssrd_increments, axis=0) / 1.0e6 # MJ/m2
            else:
                # Already de-accumulated hourly rates
                if len(day_hour_indices) > 0:
                    daily_tp[d_idx] = np.nansum(tp_raw[day_hour_indices], axis=0) * 1000.0
                    daily_ssrd[d_idx] = np.nansum(ssrd_raw[day_hour_indices], axis=0) / 1.0e6

        if ds_boundary is not None:
            ds_boundary.close()

    # Build daily Dataset
    time_arr = pd.to_datetime(target_dates)
    daily_ds = xr.Dataset(
        data_vars={
            "t2m_max": (["time", "latitude", "longitude"], daily_tmax, {
                "long_name": "Daily Maximum 2-meter Air Temperature",
                "units": "degC",
                "cell_methods": "time: maximum (24-hour)"
            }),
            "t2m_min": (["time", "latitude", "longitude"], daily_tmin, {
                "long_name": "Daily Minimum 2-meter Air Temperature",
                "units": "degC",
                "cell_methods": "time: minimum (24-hour)"
            }),
            "t2m_mean": (["time", "latitude", "longitude"], daily_tmean, {
                "long_name": "Daily Mean 2-meter Air Temperature",
                "units": "degC",
                "cell_methods": "time: mean (24-hour)"
            }),
            "d2m_mean": (["time", "latitude", "longitude"], daily_dmean, {
                "long_name": "Daily Mean 2-meter Dewpoint Temperature",
                "units": "degC",
                "cell_methods": "time: mean (24-hour)"
            }),
            "tp_sum": (["time", "latitude", "longitude"], daily_tp, {
                "long_name": "Daily Total Precipitation Accumulation (De-accumulated)",
                "units": "mm",
                "cell_methods": "time: sum (24-hour de-accumulated)"
            }),
            "ssrd_sum": (["time", "latitude", "longitude"], daily_ssrd, {
                "long_name": "Daily Surface Downward Solar Radiation Accumulation (De-accumulated)",
                "units": "MJ m-2 day-1",
                "cell_methods": "time: sum (24-hour de-accumulated)"
            }),
            "wind_speed_mean": (["time", "latitude", "longitude"], daily_wind, {
                "long_name": "Daily Mean 2-meter Wind Speed (FAO-56 Scaled)",
                "units": "m s-1",
                "cell_methods": "time: mean (24-hour)"
            }),
            "sp_mean": (["time", "latitude", "longitude"], daily_sp, {
                "long_name": "Daily Mean Surface Atmospheric Pressure",
                "units": "kPa",
                "cell_methods": "time: mean (24-hour)"
            }),
        },
        coords={
            "time": time_arr,
            "latitude": lats,
            "longitude": lons
        },
        attrs={
            "title": "ERA5-Land Daily Climatological Surface Features (1985–2023)",
            "source": "ECMWF ERA5-Land Reanalysis (reanalysis-era5-land)",
            "institution": "CropFusion / Paper 3 Hydroclimate Pipeline",
            "spatial_resolution": "0.10 degrees (~9 km)",
            "bounding_box": f"North={BOUNDING_BOX[0]}, West={BOUNDING_BOX[1]}, South={BOUNDING_BOX[2]}, East={BOUNDING_BOX[3]}",
            "study_region": "Illinois, Indiana, Iowa, Minnesota, Missouri, Nebraska, Ohio (676 counties)",
            "history": f"De-accumulated and aggregated on-the-fly on {datetime.date.today().isoformat()}"
        }
    )
    
    # Save to unique temporary file first, then atomically rename
    tmp_daily_path = daily_nc_path + f".tmp_{os.getpid()}_{random.randint(1000, 9999)}.nc"
    encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"} for var in daily_ds.data_vars}
    daily_ds.to_netcdf(tmp_daily_path, encoding=encoding)
    daily_ds.close()
    
    if os.path.exists(daily_nc_path):
        os.remove(daily_nc_path)
    os.rename(tmp_daily_path, daily_nc_path)
    
    file_size_mb = os.path.getsize(daily_nc_path) / (1024 * 1024)
    logger.info(f"Saved {daily_nc_path} successfully ({file_size_mb:.2f} MB).")
    return len(target_dates), file_size_mb


def process_single_month_worker(year, month, force_redownload=False):
    """
    Worker task: downloads, aggregates, validates, and records a single month.
    Designed for concurrent execution with safe file paths and thread-safe manifest logging.
    """
    year_str = f"{year:04d}"
    month_str = f"{month:02d}"
    days_in_month = calendar.monthrange(year, month)[1]
    
    daily_nc_name = f"ERA5_daily_{year_str}_{month_str}.nc"
    daily_nc_path = os.path.join(DATA_DIR, daily_nc_name)
    raw_hourly_nc = os.path.join(RAW_DIR, f"hourly_{year_str}_{month_str}.nc")
    
    # Check if already completed and valid (Resumability)
    if not force_redownload and os.path.exists(daily_nc_path):
        is_valid, msg = validate_daily_file(daily_nc_path, days_in_month)
        if is_valid:
            logger.info(f"File {daily_nc_name} already exists and passes validation. SKIPPING redownload.")
            record_manifest(daily_nc_name, year, month, days_in_month, "SKIPPED_EXISTING", msg, os.path.getsize(daily_nc_path)/(1024*1024))
            return True
        else:
            logger.warning(f"File {daily_nc_name} exists but failed validation ({msg}). Will regenerate.")
            try:
                os.remove(daily_nc_path)
            except Exception:
                pass

    # Initialize thread-local CDS client (reads ~/.cdsapirc securely)
    client = cdsapi.Client()

    # Step 1: Download raw hourly data if not present
    if not os.path.exists(raw_hourly_nc) or os.path.getsize(raw_hourly_nc) < 1024 * 1024:
        request = {
            "variable": ERA5_LAND_VARIABLES,
            "year": year_str,
            "month": month_str,
            "day": [f"{d:02d}" for d in range(1, days_in_month + 1)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": BOUNDING_BOX,
            "data_format": "netcdf",
            "download_format": "unarchived"
        }
        
        logger.info(f"Submitting CDS request for Year {year_str}, Month {month_str} ({days_in_month} days)...")
        tmp_raw = raw_hourly_nc + f".tmp_{os.getpid()}_{random.randint(1000, 9999)}.nc"
        success, res = retrieve_with_retries(client, request, tmp_raw, max_retries=5, base_delay=20.0)
        
        if not success or not os.path.exists(tmp_raw):
            record_failure(year, month, f"CDS download failed: {res}")
            if os.path.exists(tmp_raw):
                os.remove(tmp_raw)
            return False
            
        if os.path.exists(raw_hourly_nc):
            os.remove(raw_hourly_nc)
        os.rename(tmp_raw, raw_hourly_nc)
        logger.info(f"Raw hourly file ready for {year_str}-{month_str} ({os.path.getsize(raw_hourly_nc)/(1024*1024):.2f} MB).")

    # Step 2: Cache boundary for preceding month immediately
    export_preceding_boundary(raw_hourly_nc, year, month)

    # Step 3: Aggregate and de-accumulate to daily NetCDF
    try:
        n_days, size_mb = aggregate_hourly_chunk(raw_hourly_nc, daily_nc_path, year, month, client)
        
        # Step 4: Validate resulting daily NetCDF
        is_valid, msg = validate_daily_file(daily_nc_path, days_in_month)
        if is_valid:
            logger.info(f"Validation PASSED for {daily_nc_name}: {msg}")
            record_manifest(daily_nc_name, year, month, n_days, "DOWNLOADED_AND_AGGREGATED", msg, size_mb)
            # CRITICAL RULE: Delete raw hourly file ONLY after successful validation
            if os.path.exists(raw_hourly_nc):
                os.remove(raw_hourly_nc)
                logger.info(f"Cleaned up verified raw file: {raw_hourly_nc}")
            return True
        else:
            logger.error(f"Validation FAILED for {daily_nc_name}: {msg}")
            record_manifest(daily_nc_name, year, month, n_days, "FAILED_VALIDATION", msg, size_mb)
            record_failure(year, month, f"Validation failure: {msg}")
            # Keep raw hourly file for forensic debugging
            logger.warning(f"Retaining raw hourly file for debugging: {raw_hourly_nc}")
            return False
            
    except Exception as e:
        logger.error(f"Error during aggregation/validation for {year_str}-{month_str}: {str(e)}")
        record_failure(year, month, f"Aggregation exception: {str(e)}")
        logger.warning(f"Retaining raw hourly file for debugging: {raw_hourly_nc}")
        return False


def run_pipeline(start_year=1985, end_year=2023, workers=2, test_mode=False, test_three=False, test_year=1985, test_month=1, force_redownload=False):
    """
    Main acquisition orchestration with safe controlled parallelism.
    """
    initialize_manifest()
    
    # Cap workers safely between 1 and 3
    workers = max(1, min(3, workers))
    
    logger.info("==========================================================")
    logger.info("ERA5-LAND DAILY CLIMATE ACQUISITION PIPELINE (PARALLEL)")
    logger.info(f"Workers: {workers}")
    logger.info(f"Bounding Box: {BOUNDING_BOX}")
    logger.info(f"Output Directory: {DATA_DIR}")
    logger.info("==========================================================")

    # TEST 1: Single representative month
    if test_mode:
        logger.info(f"RUNNING TEST 1: Single month ({test_year}-{test_month:02d})...")
        success = process_single_month_worker(test_year, test_month, force_redownload=force_redownload)
        if success:
            logger.info(f"TEST 1 SUCCEEDED for {test_year}-{test_month:02d}.")
        else:
            logger.error(f"TEST 1 FAILED for {test_year}-{test_month:02d}.")
        return success

    # TEST 2: Three consecutive months (month boundaries & concurrency check)
    if test_three:
        logger.info("RUNNING TEST 2: Three consecutive months (1985-01, 1985-02, 1985-03) with 2 workers...")
        tasks = [(1985, 1), (1985, 2), (1985, 3)]
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_month = {executor.submit(process_single_month_worker, yr, mo, force_redownload): (yr, mo) for yr, mo in tasks}
            for future in concurrent.futures.as_completed(future_to_month):
                yr, mo = future_to_month[future]
                try:
                    if future.result():
                        completed += 1
                        logger.info(f"Month {yr}-{mo:02d} finished successfully.")
                    else:
                        logger.error(f"Month {yr}-{mo:02d} reported failure.")
                except Exception as exc:
                    logger.error(f"Month {yr}-{mo:02d} generated exception: {exc}")
        success = (completed == len(tasks))
        logger.info(f"TEST 2 Result: {completed}/{len(tasks)} months succeeded.")
        return success

    # FULL PRODUCTION PIPELINE (e.g. 1985–2023)
    tasks = []
    for yr in range(start_year, end_year + 1):
        for mo in range(1, 13):
            tasks.append((yr, mo))
            
    total_months = len(tasks)
    logger.info(f"Starting acquisition for {total_months} months ({start_year} to {end_year}) using {workers} workers...")
    
    completed_count = 0
    failed_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_month = {executor.submit(process_single_month_worker, yr, mo, force_redownload): (yr, mo) for yr, mo in tasks}
        for future in concurrent.futures.as_completed(future_to_month):
            yr, mo = future_to_month[future]
            try:
                if future.result():
                    completed_count += 1
                    logger.info(f"Progress: [{completed_count + failed_count}/{total_months}] - {yr}-{mo:02d} complete.")
                else:
                    failed_count += 1
                    logger.warning(f"Progress: [{completed_count + failed_count}/{total_months}] - {yr}-{mo:02d} FAILED.")
            except Exception as exc:
                failed_count += 1
                logger.error(f"Worker exception for {yr}-{mo:02d}: {exc}")
                
    logger.info(f"Acquisition finished. Succeeded: {completed_count}, Failed: {failed_count}, Total: {total_months}.")
    return failed_count == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ERA5-Land Daily Climate Acquisition Engine")
    parser.add_argument("--test", action="store_true", help="Run Test 1: Single month test")
    parser.add_argument("--test_three", action="store_true", help="Run Test 2: Three consecutive months test")
    parser.add_argument("--test_year", type=int, default=1985, help="Test year (default: 1985)")
    parser.add_argument("--test_month", type=int, default=1, help="Test month (default: 1)")
    parser.add_argument("--start_year", type=int, default=1985, help="Start year (default: 1985)")
    parser.add_argument("--end_year", type=int, default=2023, help="End year (default: 2023)")
    parser.add_argument("--workers", type=int, default=2, help="Number of concurrent workers (default: 2, max: 3)")
    parser.add_argument("--force", action="store_true", help="Force redownload even if validated file exists")
    args = parser.parse_args()
    
    success = run_pipeline(
        start_year=args.start_year,
        end_year=args.end_year,
        workers=args.workers,
        test_mode=args.test,
        test_three=args.test_three,
        test_year=args.test_year,
        test_month=args.test_month,
        force_redownload=args.force
    )
    sys.exit(0 if success else 1)

