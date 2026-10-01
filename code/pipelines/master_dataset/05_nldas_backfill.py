"""
05_nldas_backfill.py -- backfill missing / 0-byte NLDAS-2 hourly granules (user-approved 2026-09-14).

Source: NASA GES DISC HTTPS, product NLDAS_FORA0125_H.2.0 (same product/version as the raw archive).
Each granule is streamed into memory, verified (size >= 100 KB, NetCDF parses, time stamp == expected
hour), and ONLY the study-region subset (116 x 196 cells; SWdown, LWdown, Wind_E, Wind_N, Qair, PSurf,
Tair, Rainf) is written, zlib-compressed, to master_dataset/intermediate/nldas_backfill/<Y>/.
Raw folders on G: and C: are never modified. SHA-256 of the original bytes is recorded for provenance.

Inputs : audits/nldas_hourly_inventory_prebackfill.csv
Outputs: intermediate/nldas_backfill/<Y>/NLDAS_FORA0125_H.A<YYYYMMDD>.<HH>00.020.subset.nc
         audits/nldas_backfill_manifest.csv
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

from md_common import AUDITS, BACKFILL_DIR, PAPER3, get_logger, nldas_granule_name, storage_guard

log = get_logger("05_nldas_backfill")
BASE = "https://data.gesdisc.earthdata.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.2.0"
LAT0, LAT1, LON0, LON1 = 84, 200, 164, 360
VARS = ["SWdown", "LWdown", "Wind_E", "Wind_N", "Qair", "PSurf", "Tair", "Rainf"]
EPOCH = dt.datetime(1979, 1, 1)
NC_LOCK = threading.Lock()
MANIFEST = AUDITS / "nldas_backfill_manifest.csv"


def token() -> str:
    for line in (PAPER3 / ".env").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("NASA_EARTHDATA_TOKEN"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("NASA_EARTHDATA_TOKEN not found in .env")


def targets(from_status: bool = False) -> list[tuple[dt.date, int]]:
    """Missing/0-byte hours from the pre-backfill inventory, or (from_status) hours whose raw granule
    exists but failed to parse (corrupt HDF5) as recorded by 06_nldas_county_daily.py."""
    out = []
    if not from_status:
        inv = pd.read_csv(AUDITS / "nldas_hourly_inventory_prebackfill.csv", dtype={"missing_hours": str})
        for _, r in inv[~inv.complete_day].iterrows():
            d = dt.date.fromisoformat(r.date)
            for h in sorted({int(x) for x in str(r.missing_hours).split(",") if x not in ("", "nan")}):
                out.append((d, h))
        return out
    from md_common import INTER
    for f in sorted((INTER / "nldas_county_daily").glob("nldas_day_status_*.csv")):
        st = pd.read_csv(f, dtype={"failed_hours": str, "missing_hours": str})
        for _, r in st[~st.complete_day.astype(bool)].iterrows():
            d = dt.date.fromisoformat(r.date)
            hrs = set()
            for item in str(r.failed_hours).split(";"):
                if ":" in item:
                    hrs.add(int(item.split(":")[0]))
            for x in str(r.missing_hours).split(","):
                if x not in ("", "nan"):
                    hrs.add(int(x))
            out += [(d, h) for h in sorted(hrs)]
    return out


def write_subset(buf: bytes, d: dt.date, h: int, dst):
    import netCDF4
    with NC_LOCK:
        src = netCDF4.Dataset("mem", memory=buf)
        try:
            t = float(src["time"][:].ravel()[0])
            expected = (dt.datetime(d.year, d.month, d.day, h) - EPOCH).total_seconds() / 3600.0
            if abs(t - expected) > 1e-6:
                raise ValueError(f"time mismatch {t} != {expected}")
            tmp = str(dst) + ".part"
            with netCDF4.Dataset(tmp, "w", format="NETCDF4") as out:
                out.setncatts({"title": "Study-region subset of NLDAS_FORA0125_H.2.0 hourly granule (backfill)",
                               "source_granule": nldas_granule_name(d, h), "source_product": "NLDAS_FORA0125_H.2.0",
                               "subset": f"lat[{LAT0}:{LAT1}] lon[{LON0}:{LON1}] of 224x464 CONUS grid",
                               "created": dt.datetime.now(dt.timezone.utc).isoformat()})
                out.createDimension("time", 1)
                out.createDimension("lat", LAT1 - LAT0)
                out.createDimension("lon", LON1 - LON0)
                for c, sl in (("lat", slice(LAT0, LAT1)), ("lon", slice(LON0, LON1))):
                    v = out.createVariable(c, "f4", (c,))
                    v[:] = src[c][sl]
                    v.units = src[c].units
                tv = out.createVariable("time", "f8", ("time",))
                tv[:] = t
                tv.units = src["time"].units
                for name in VARS:
                    s = src[name]
                    v = out.createVariable(name, "f4", ("time", "lat", "lon"), zlib=True, complevel=4,
                                           fill_value=np.float32(s._FillValue))
                    v[:] = s[0:1, LAT0:LAT1, LON0:LON1]
                    v.units = getattr(s, "units", "")
            os.replace(tmp, dst)
        finally:
            src.close()


def fetch(session, tok, d, h):
    name = nldas_granule_name(d, h)
    dst = BACKFILL_DIR / str(d.year) / name.replace(".nc", ".subset.nc")
    rec = dict(date=d.isoformat(), hour=h, granule=name, url=f"{BASE}/{d.year}/{d.timetuple().tm_yday:03d}/{name}",
               subset_path=str(dst))
    if dst.exists() and dst.stat().st_size > 0:
        rec.update(status="ALREADY_PRESENT")
        return rec
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 6):
        try:
            r = session.get(rec["url"], headers={"Authorization": f"Bearer {tok}"}, timeout=120)
            rec["http_status"] = r.status_code
            if r.status_code in (401, 403):
                rec.update(status="AUTH_ERROR")
                return rec
            if r.status_code == 404:
                rec.update(status="NOT_ON_SERVER")
                return rec
            r.raise_for_status()
            buf = r.content
            if len(buf) < 100 * 1024:
                raise ValueError(f"payload too small ({len(buf)} B)")
            rec["original_bytes"] = len(buf)
            rec["sha256_original"] = hashlib.sha256(buf).hexdigest()
            write_subset(buf, d, h, dst)
            rec.update(status="BACKFILLED", subset_bytes=dst.stat().st_size, attempts=attempt,
                       downloaded_at=dt.datetime.now().isoformat(timespec="seconds"))
            return rec
        except Exception as ex:
            rec.update(status="FAILED", error=f"{type(ex).__name__}: {ex}", attempts=attempt)
            time.sleep(min(2 ** attempt, 60))
    return rec


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-status", action="store_true", help="backfill hours that failed to parse (corrupt raw)")
    ap.add_argument("--years", type=int, nargs="*", default=None)
    a = ap.parse_args()
    tgts = targets(a.from_status)
    if a.years:
        tgts = [(d, h) for d, h in tgts if d.year in a.years]
    global MANIFEST
    if a.from_status:
        MANIFEST = AUDITS / f"nldas_backfill_manifest_corrupt_{'_'.join(map(str, a.years or ['all']))}.csv"
    log.info("Backfill targets: %d hours over %d dates", len(tgts), len({d for d, _ in tgts}))
    tok = token()
    session = requests.Session()
    session.headers["User-Agent"] = "Paper3-MasterDataset-NLDAS-Backfill/1.0 (research)"
    recs = []
    t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        futs = [ex.submit(fetch, session, tok, d, h) for d, h in tgts]
        for i, f in enumerate(as_completed(futs), 1):
            recs.append(f.result())
            if recs[-1]["status"] == "AUTH_ERROR":
                log.error("Authentication/EULA error -- stopping. %s", recs[-1]["url"])
                break
            if i % 100 == 0:
                storage_guard(2.0, logger=log)
                st = pd.Series([r["status"] for r in recs]).value_counts().to_dict()
                log.info("%d/%d done (%.1f/min) %s", i, len(tgts), i / ((time.time() - t0) / 60), st)
                pd.DataFrame(recs).to_csv(MANIFEST, index=False)
    m = pd.DataFrame(recs).sort_values(["date", "hour"])
    m.to_csv(MANIFEST, index=False)
    log.info("Backfill finished: %s", m.status.value_counts().to_dict())


if __name__ == "__main__":
    main()
