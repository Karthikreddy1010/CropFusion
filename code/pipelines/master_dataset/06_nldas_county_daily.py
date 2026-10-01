"""
06_nldas_county_daily.py -- NLDAS-2 hourly -> daily grid -> FAO-56 ET0 -> county daily.

Reads hourly granules IN PLACE (G:/My Drive/NLDAS/<Y>/raw/<date>/ for 1985-2019,
C:/nldas_curl_<Y>/ for 2020-2023, plus study-region subsets of backfilled hours in
master_dataset/intermediate/nldas_backfill/). Nothing is copied or written to raw locations.

Completeness rule: a date is aggregated ONLY if all 24 hourly granules (00Z..23Z) exist, parse,
carry the expected time stamp, and are non-empty. Otherwise the date is recorded as incomplete and
no daily values are produced (no silent filling).

Daily aggregation reproduces code/pipelines/nldas/download_nldas.py::aggregate_day_to_daily_grid:
  Rs = mean(SWdown) * 0.0864 ; LWdown mean ; u2 = mean(0.748*sqrt(E^2+N^2)) ;
  ea = mean(q*P/(0.622+0.378q))/1000 ; PSurf mean ; Tair mean/min/max (K-273.15).
ET0 uses the validated code/pipelines/nldas/fao56_et0.py (FAO-56 eq. 6, Rnl eq. 39,
Rso eq. 37 with the FIXED NASA GSFC NLDAS elevation grid) -- equations unchanged.
Day definition: UTC calendar day (00Z-23Z granules), as in the validated pipeline.

Usage:
  python 06_nldas_county_daily.py --years 1985 1986 ...   (default: all years, skip finished)
  python 06_nldas_county_daily.py --redo-incomplete --years 2020   (after backfill)
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import pandas as pd

from md_common import (BACKFILL_DIR, INTER, NLDAS_C_ROOTS, NLDAS_ELEV, NLDAS_G_ROOT, NLDAS_SCRIPTS, get_logger,
                       nldas_granule_name, storage_guard)

OUT = INTER / "nldas_county_daily"
OUT.mkdir(parents=True, exist_ok=True)
MIN_BYTES = 100 * 1024
LAT0, LAT1, LON0, LON1 = 84, 200, 164, 360   # study subset indices in the 224x464 CONUS grid (verified)
VARS = ["SWdown", "LWdown", "Wind_E", "Wind_N", "Qair", "PSurf", "Tair", "Rainf"]
BOUNDS = {"SWdown": (0, 1361.0 * 1.05), "LWdown": (50, 600), "Wind_E": (-50, 50), "Wind_N": (-50, 50),
          "Qair": (0, 0.040), "PSurf": (50_000, 108_000), "Tair": (180, 340), "Rainf": (0, 300)}
EPOCH = dt.datetime(1979, 1, 1)

_G = {}


def _init_worker():
    sys.path.insert(0, str(NLDAS_SCRIPTS))
    import xarray as xr
    import scipy.sparse as sp
    import fao56_et0  # validated implementation (unchanged)

    e = xr.open_dataset(NLDAS_ELEV)
    _G["lat"] = e.lat.values
    _G["lon"] = e.lon.values
    _G["elev"] = e.elevation_m.values.astype(np.float64)
    e.close()
    w = pd.read_parquet(INTER / "weights" / "nldas_county_weights.parquet")
    counties = np.array(sorted(w.county_fips.unique()))
    ci = {c: i for i, c in enumerate(counties)}
    ncell = len(_G["lat"]) * len(_G["lon"])
    cell = (w.lat_idx.values * len(_G["lon"]) + w.lon_idx.values).astype(np.int64)
    _G["W"] = sp.csr_matrix((w.weight.values, (w.county_fips.map(ci).values, cell)), shape=(len(counties), ncell))
    _G["counties"] = counties
    _G["fao"] = fao56_et0
    _G["xr"] = xr
    _G["pool"] = ThreadPoolExecutor(8)


def resolve_sources(d: dt.date) -> dict:
    """hour -> (kind, path) using granules >= 100 KB; backfilled subsets used only for hours not in raw."""
    ds = d.strftime("%Y-%m-%d")
    if d.year <= 2019:
        dirs = [NLDAS_G_ROOT / str(d.year) / "raw" / ds]
    else:
        r = NLDAS_C_ROOTS[d.year]
        dirs = [r / "raw" / ds, r / ds, r]
    found = {}
    for p in dirs:
        try:
            with os.scandir(p) as it:
                for e in it:
                    if e.is_file() and e.name.endswith(".nc"):
                        found.setdefault(e.name, e)
        except FileNotFoundError:
            continue
    src = {}
    for h in range(24):
        n = nldas_granule_name(d, h)
        e = found.get(n)
        b = BACKFILL_DIR / str(d.year) / n.replace(".nc", ".subset.nc")
        # A verified backfill subset exists only for hours whose raw granule was missing, empty or corrupt,
        # so it takes precedence over the raw file for that hour.
        if b.exists() and b.stat().st_size > 0:
            src[h] = ("backfill", str(b))
        elif e is not None and e.stat().st_size >= MIN_BYTES:
            src[h] = ("raw", e.path)
    return src


def _read_bytes(path):
    """I/O only (thread-safe). HDF5/netCDF4 parsing is NOT thread-safe and is done serially."""
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except Exception as ex:
        return ex


def _parse_hour(d, h, kind, buf):
    import netCDF4
    if isinstance(buf, Exception):
        return h, None, f"read_error:{type(buf).__name__}:{buf}"
    try:
        nc = netCDF4.Dataset("mem", memory=buf)
        try:
            t = float(nc["time"][:].ravel()[0])
            expected = (dt.datetime(d.year, d.month, d.day, h) - EPOCH).total_seconds() / 3600.0
            if abs(t - expected) > 1e-6:
                return h, None, f"time_mismatch({t} != {expected})"
            out = {}
            for v in VARS:
                var = nc[v]
                if kind == "raw":
                    a = var[0, LAT0:LAT1, LON0:LON1]
                else:
                    a = var[0, :, :]
                a = np.ma.filled(a.astype(np.float32), np.nan)
                if a.shape != (116, 196):
                    return h, None, f"bad_shape{a.shape}"
                out[v] = a
            return h, out, "ok"
        finally:
            nc.close()
    except Exception as ex:  # corrupt / unreadable granule -> hour unavailable
        return h, None, f"read_error:{type(ex).__name__}:{ex}"


def process_day(dstr: str):
    d = dt.date.fromisoformat(dstr)
    src = resolve_sources(d)
    status = dict(date=dstr, year=d.year, hours_expected=24, hours_with_file=len(src),
                  backfilled_hours=",".join(str(h) for h, s in sorted(src.items()) if s[0] == "backfill"))
    missing = [h for h in range(24) if h not in src]
    if missing:
        status.update(hours_available=len(src), complete_day=False, missing_hours=",".join(map(str, missing)),
                      failed_hours="", aggregated=False)
        return status, None
    items = sorted(src.items())
    bufs = list(_G["pool"].map(_read_bytes, [p for _, (_, p) in items]))
    results = [_parse_hour(d, h, k, b) for (h, (k, _)), b in zip(items, bufs)]
    del bufs
    failed = [(h, msg) for h, arr, msg in results if arr is None]
    if failed:
        status.update(hours_available=24 - len(failed), complete_day=False, missing_hours="",
                      failed_hours=";".join(f"{h}:{m}" for h, m in failed), aggregated=False)
        return status, None
    stack = {v: np.stack([r[1][v] for r in sorted(results, key=lambda x: x[0])]) for v in VARS}
    qc = {}
    for v, (lo, hi) in BOUNDS.items():
        a = stack[v]
        qc[f"qc_viol_{v}"] = int(np.sum((a < lo) | (a > hi)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # --- validated daily aggregation (download_nldas.py) ---
        u2_h = 0.748 * np.sqrt(stack["Wind_E"] ** 2 + stack["Wind_N"] ** 2)
        q, p = stack["Qair"], stack["PSurf"]
        ea_h = (q * p / (0.622 + 0.378 * q)) / 1000.0
        sw_mean = np.nanmean(stack["SWdown"], axis=0)
        daily = {
            "SWdown_mean": sw_mean,
            "Rs_daily_MJ": sw_mean * 0.0864,
            "LWdown_mean": np.nanmean(stack["LWdown"], axis=0),
            "u2_daily_mean": np.nanmean(u2_h, axis=0),
            "ea_daily_mean_kPa": np.nanmean(ea_h, axis=0),
            "PSurf_daily_mean_Pa": np.nanmean(p, axis=0),
            "Tair_daily_mean_C": np.nanmean(stack["Tair"], axis=0) - 273.15,
            "Tair_daily_min_C": np.nanmin(stack["Tair"], axis=0) - 273.15,
            "Tair_daily_max_C": np.nanmax(stack["Tair"], axis=0) - 273.15,
        }
        qair_mean = np.nanmean(q, axis=0)
        rainf_sum = np.nansum(stack["Rainf"], axis=0)
        rainf_sum[np.all(np.isnan(stack["Rainf"]), axis=0)] = np.nan
    xr = _G["xr"]
    coords = {"lat": _G["lat"], "lon": _G["lon"]}
    ds = xr.Dataset({k: (("lat", "lon"), v.astype(np.float32)) for k, v in daily.items()}, coords=coords)
    e = _G["fao"].calc_fao56_et0_from_dataset(ds, day_of_year=d.timetuple().tm_yday, elevation_m=_G["elev"])
    grid = {
        "nldas_tair_mean": e["Tair_daily_mean_C"].values,
        "nldas_tair_min": e["Tair_daily_min_C"].values,
        "nldas_tair_max": e["Tair_daily_max_C"].values,
        "nldas_rs": e["Rs_daily_MJ"].values,
        "nldas_lwdown": e["LWdown_mean"].values,
        "nldas_wind": e["u2_daily_mean"].values,
        "nldas_qair": qair_mean,
        "nldas_ea": e["ea_daily_mean_kPa"].values,
        "nldas_es": e["es_daily_kPa"].values,
        "nldas_vpd": np.maximum(e["es_daily_kPa"].values - e["ea_daily_mean_kPa"].values, 0.0),
        "nldas_pressure": e["PSurf_daily_mean_Pa"].values / 1000.0,
        "nldas_rso": e["Rso_daily_MJ"].values,
        "nldas_rnl": e["Rnl_fao56_daily_MJ"].values,
        "nldas_rn": e["Rn_daily_MJ"].values,
        "nldas_et0": e["ET0_fao56_daily_mm"].values,
        "nldas_rainf_qc": rainf_sum,
    }
    e.close()
    ds.close()
    W = _G["W"]
    out = {"county_fips": _G["counties"], "date": np.full(len(_G["counties"]), np.datetime64(dstr))}
    valid_frac = None
    for k, g in grid.items():
        flat = np.asarray(g, dtype=np.float64).ravel()
        ok = np.isfinite(flat)
        num = W @ np.where(ok, flat, 0.0)
        den = W @ ok.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[k] = (num / den).astype(np.float32)
        frac = den / np.asarray(W.sum(axis=1)).ravel()
        valid_frac = frac if valid_frac is None else np.minimum(valid_frac, frac)
    out["nldas_valid_area_frac"] = valid_frac.astype(np.float32)
    status.update(hours_available=24, complete_day=True, missing_hours="", failed_hours="", aggregated=True,
                  n_backfilled=len([s for s in src.values() if s[0] == "backfill"]), **qc)
    return status, pd.DataFrame(out)


def run_year(year: int, pool, log, redo_incomplete=False):
    pq = OUT / f"nldas_county_daily_{year}.parquet"
    st = OUT / f"nldas_day_status_{year}.csv"
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(f"{year}-01-01", f"{year}-12-31")]
    old_df, old_st = None, None
    if pq.exists() and st.exists():
        if not redo_incomplete:
            log.info("year %d already processed -> skip", year)
            return
        old_df = pd.read_parquet(pq)
        old_st = pd.read_csv(st)
        dates = old_st.loc[~old_st.complete_day.astype(bool), "date"].tolist()
        log.info("year %d: re-processing %d incomplete dates", year, len(dates))
        if not dates:
            return
    storage_guard(2.0, drives=("C:\\",), logger=log)
    t0 = time.time()
    frames, stats = [], []
    for i, (s, df) in enumerate(pool.imap(process_day, dates, chunksize=1), 1):
        stats.append(s)
        if df is not None:
            frames.append(df)
        if i % 30 == 0:
            el = time.time() - t0
            log.info("year %d: %d/%d dates, %.1f s/date, complete so far %d", year, i, len(dates), el / i,
                     sum(x["complete_day"] for x in stats))
            storage_guard(2.0, drives=("C:\\",), logger=log)
    new_st = pd.DataFrame(stats)
    new_df = pd.concat(frames, ignore_index=True) if frames else None
    if old_df is not None:
        new_st = pd.concat([old_st[~old_st.date.isin(new_st.date)], new_st], ignore_index=True)
        new_df = pd.concat([old_df] + ([new_df] if new_df is not None else []), ignore_index=True)
    new_st = new_st.sort_values("date").reset_index(drop=True)
    if new_df is not None:
        new_df = new_df.sort_values(["date", "county_fips"]).reset_index(drop=True)
        assert not new_df.duplicated(["county_fips", "date"]).any()
        tmp = pq.with_suffix(".parquet.part")
        new_df.to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, pq)
    new_st.to_csv(st, index=False)
    log.info("year %d done in %.1f min: %d/%d complete days", year, (time.time() - t0) / 60,
             int(new_st.complete_day.sum()), len(new_st))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", default=list(range(1985, 2024)))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--redo-incomplete", action="store_true")
    a = ap.parse_args()
    log = get_logger("06_nldas_county_daily")
    log.info("Start NLDAS county daily: years=%s workers=%d redo_incomplete=%s", a.years, a.workers, a.redo_incomplete)
    with Pool(a.workers, initializer=_init_worker) as pool:
        for y in a.years:
            run_year(y, pool, log, a.redo_incomplete)
    log.info("All requested years finished.")
