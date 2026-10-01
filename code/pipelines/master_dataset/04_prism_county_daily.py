"""
04_prism_county_daily.py -- PRISM daily 4 km (2.5 arc-min) Tmax/Tmin/PPT -> county daily.

Reads the ZIP archives IN PLACE from D:/prism_pipeline/raw/{tmax,tmin,ppt}/YYYYMMDD.zip (GeoTIFF member
read into memory; nothing extracted to disk, raw untouched). County values use the fractional-area
county-polygon weights from 03_build_county_weights.py (exactextract coverage x cos(lat)); nodata cells
(-9999) are excluded and the weights renormalised.

PRISM day definition: 24 h ending 12:00 UTC on the date (PRISM convention).

QC recorded per date: tmax >= tmin (cell and county level), ppt >= 0, plausible bounds, valid-area fraction,
PRISM dataset type / version / create date from the archive's .info.txt.

Outputs: intermediate/prism_county_daily/prism_county_daily_<Y>.parquet
         intermediate/prism_county_daily/prism_day_status_<Y>.csv
Usage:   python 04_prism_county_daily.py [--years ...] [--workers N]
"""
from __future__ import annotations

import argparse
import os
import time
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd

from md_common import INTER, PRISM_RAW, get_logger, storage_guard

OUT = INTER / "prism_county_daily"
OUT.mkdir(parents=True, exist_ok=True)
VARS = ("tmax", "tmin", "ppt")
BOUNDS = {"tmax": (-50.0, 55.0), "tmin": (-55.0, 45.0), "ppt": (0.0, 600.0)}
_G = {}


def _init():
    import scipy.sparse as sp
    from rasterio.io import MemoryFile

    w = pd.read_parquet(INTER / "weights" / "prism_county_weights.parquet")
    meta = pd.read_csv(INTER / "weights" / "prism_grid_metadata.csv").iloc[0]
    rows, cols = eval(meta["shape"])
    counties = np.array(sorted(w.county_fips.unique()))
    ci = {c: i for i, c in enumerate(counties)}
    # Restrict to the bounding window of used cells to save work
    r0, r1, c0, c1 = w.row.min(), w.row.max() + 1, w.col.min(), w.col.max() + 1
    cell = ((w.row.values - r0) * (c1 - c0) + (w.col.values - c0)).astype(np.int64)
    _G["W"] = sp.csr_matrix((w.weight.values, (w.county_fips.map(ci).values, cell)),
                            shape=(len(counties), (r1 - r0) * (c1 - c0)))
    _G["Wsum"] = np.asarray(_G["W"].sum(axis=1)).ravel()
    _G["win"] = (r0, r1, c0, c1)
    _G["shape"] = (rows, cols)
    _G["counties"] = counties
    _G["MemoryFile"] = MemoryFile


def _read(var, ds):
    z = zipfile.ZipFile(PRISM_RAW / var / f"{ds}.zip")
    try:
        names = z.namelist()
        tif = [n for n in names if n.endswith(".tif")][0]
        info = {}
        inf = [n for n in names if n.endswith(".info.txt")]
        if inf:
            for line in z.read(inf[0]).decode("utf-8", "replace").splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    info[k.strip()] = v.strip()
        with _G["MemoryFile"](z.read(tif)) as m, m.open() as r:
            if r.shape != _G["shape"]:
                raise ValueError(f"unexpected shape {r.shape}")
            r0, r1, c0, c1 = _G["win"]
            a = r.read(1)[r0:r1, c0:c1].astype(np.float64)
            a[a == r.nodata] = np.nan
        return a, info
    finally:
        z.close()


def process_date(ds: str):
    st = {"date": pd.Timestamp(ds).strftime("%Y-%m-%d")}
    grids = {}
    try:
        for v in VARS:
            a, info = _read(v, ds)
            grids[v] = a
            st[f"{v}_dataset_type"] = info.get("PRISM_DATASET_TYPE", "")
            st[f"{v}_version"] = info.get("PRISM_DATASET_VERSION", "")
            st[f"{v}_create_date"] = info.get("PRISM_DATASET_CREATE_DATE", "")
    except Exception as ex:
        st.update(ok=False, error=f"{type(ex).__name__}: {ex}")
        return st, None
    tx, tn, pp = grids["tmax"], grids["tmin"], grids["ppt"]
    both = np.isfinite(tx) & np.isfinite(tn)
    st["cell_tmax_lt_tmin"] = int(np.sum(tx[both] < tn[both]))
    st["cell_ppt_negative"] = int(np.nansum(pp < 0))
    for v in VARS:
        lo, hi = BOUNDS[v]
        g = grids[v]
        st[f"cell_{v}_out_of_bounds"] = int(np.nansum((g < lo) | (g > hi)))
    W = _G["W"]
    out = {"county_fips": _G["counties"], "date": np.full(len(_G["counties"]), np.datetime64(st["date"]))}
    frac = None
    for v in VARS:
        flat = grids[v].ravel()
        ok = np.isfinite(flat)
        num = W @ np.where(ok, flat, 0.0)
        den = W @ ok.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"prism_{v}"] = (num / den).astype(np.float32)
        f = den / _G["Wsum"]
        frac = f if frac is None else np.minimum(frac, f)
    out["prism_valid_area_frac"] = frac.astype(np.float32)
    df = pd.DataFrame(out)
    st["county_tmax_lt_tmin"] = int((df.prism_tmax < df.prism_tmin).sum())
    st["county_nan_values"] = int(df[[f"prism_{v}" for v in VARS]].isna().sum().sum())
    st["ok"] = True
    return st, df


def run_year(y, pool, log):
    pq = OUT / f"prism_county_daily_{y}.parquet"
    if pq.exists():
        log.info("PRISM %d exists -> skip", y)
        return
    storage_guard(1.5, logger=log)
    dates = [d.strftime("%Y%m%d") for d in pd.date_range(f"{y}-01-01", f"{y}-12-31")]
    t0 = time.time()
    stats, frames = [], []
    for s, df in pool.imap(process_date, dates, chunksize=2):
        stats.append(s)
        if df is not None:
            frames.append(df)
    st = pd.DataFrame(stats)
    df = pd.concat(frames, ignore_index=True).sort_values(["date", "county_fips"]).reset_index(drop=True)
    assert not df.duplicated(["county_fips", "date"]).any()
    tmp = pq.with_suffix(".parquet.part")
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, pq)
    st.to_csv(OUT / f"prism_day_status_{y}.csv", index=False)
    log.info("PRISM %d: %d/%d dates ok, %d county-days, %.1f min, cell tmax<tmin=%d, county tmax<tmin=%d, NaN=%d",
             y, int(st.ok.sum()), len(dates), len(df), (time.time() - t0) / 60, int(st.cell_tmax_lt_tmin.sum()),
             int(st.county_tmax_lt_tmin.sum()), int(st.county_nan_values.sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", default=list(range(1985, 2024)))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    log = get_logger("04_prism_county_daily")
    log.info("Start PRISM county daily years=%s workers=%d", a.years, a.workers)
    with Pool(a.workers, initializer=_init) as pool:
        for y in a.years:
            run_year(y, pool, log)
    log.info("PRISM all requested years finished")
