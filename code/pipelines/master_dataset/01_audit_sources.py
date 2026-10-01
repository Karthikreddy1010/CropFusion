"""
01_audit_sources.py -- Phase 1: audit drives and every raw/processed source BEFORE processing.

Outputs (master_dataset/audits/):
  drive_space_audit.csv
  prism_raw_inventory_summary.csv       (per variable x year)
  prism_missing_or_empty_files.csv
  nldas_hourly_inventory_prebackfill.csv (per date: hours found, missing hours, source dirs)
  nldas_daily_completeness_report_prebackfill.csv
  source_audit_summary.json
"""
from __future__ import annotations

import datetime as dt
import os
import zipfile
from collections import defaultdict

import pandas as pd

from md_common import (AUDITS, ALL_DATES, NLDAS_C_ROOTS, NLDAS_G_ROOT, PRISM_RAW, USDA_YIELD, USDA_PROD,
                       NOAA_COUNTY_YEAR, NOAA_EVENTS, NOAA_ENHANCED, OLD_MEGA, NLDAS_ELEV, get_logger,
                       free_gb, nldas_granule_name, write_json, load_study_counties)

log = get_logger("01_audit_sources")
MIN_GRANULE_BYTES = 100 * 1024


def audit_drives():
    rows = []
    for d in ("C:\\", "D:\\", "G:\\"):
        import shutil
        u = shutil.disk_usage(d)
        rows.append(dict(drive=d, total_gb=round(u.total / 1e9, 2), used_gb=round(u.used / 1e9, 2),
                         free_gb=round(u.free / 1e9, 2), audited_at=dt.datetime.now().isoformat(timespec="seconds")))
    df = pd.DataFrame(rows)
    df.to_csv(AUDITS / "drive_space_audit.csv", index=False)
    log.info("Drive space:\n%s", df.to_string(index=False))
    return df


def audit_prism():
    expected = {d.strftime("%Y%m%d") for d in ALL_DATES}
    summary, problems = [], []
    for var in ("tmax", "tmin", "ppt"):
        found = {}
        with os.scandir(PRISM_RAW / var) as it:
            for e in it:
                if e.is_file() and e.name.endswith(".zip"):
                    found[e.name[:-4]] = e.stat().st_size
        for ds in sorted(expected - found.keys()):
            problems.append(dict(variable=var, date=ds, issue="MISSING"))
        for ds, sz in found.items():
            if ds not in expected:
                problems.append(dict(variable=var, date=ds, issue="UNEXPECTED_DATE"))
            elif sz < 10_000:
                problems.append(dict(variable=var, date=ds, issue=f"TOO_SMALL_{sz}B"))
        by_year = defaultdict(lambda: [0, 0])
        for ds, sz in found.items():
            by_year[int(ds[:4])][0] += 1
            by_year[int(ds[:4])][1] += sz
        for y in range(1985, 2024):
            exp = sum(1 for d in expected if d.startswith(str(y)))
            summary.append(dict(variable=var, year=y, files_expected=exp, files_found=by_year[y][0],
                                size_gb=round(by_year[y][1] / 1e9, 3)))
        # structural check of first/mid/last archive
        for ds in ("19850101", "20040701", "20231231"):
            with zipfile.ZipFile(PRISM_RAW / var / f"{ds}.zip") as z:
                names = z.namelist()
            log.info("PRISM %s %s members: %s", var, ds, [n for n in names if n.endswith((".tif", ".bil"))])
    s = pd.DataFrame(summary)
    s.to_csv(AUDITS / "prism_raw_inventory_summary.csv", index=False)
    pd.DataFrame(problems, columns=["variable", "date", "issue"]).to_csv(AUDITS / "prism_missing_or_empty_files.csv", index=False)
    tot = s.groupby("variable")[["files_expected", "files_found", "size_gb"]].sum()
    log.info("PRISM inventory:\n%s\nproblems=%d", tot.to_string(), len(problems))
    return s, problems


def _scan_dir(p):
    out = {}
    try:
        with os.scandir(p) as it:
            for e in it:
                if e.is_file() and e.name.startswith("NLDAS_FORA0125_H.A") and e.name.endswith(".nc"):
                    out[e.name] = (os.path.join(p, e.name), e.stat().st_size)
    except FileNotFoundError:
        pass
    return out


def audit_nldas():
    rows = []
    root_cache = {}
    for y, root in NLDAS_C_ROOTS.items():   # stray granules stored directly under year roots
        root_cache[y] = _scan_dir(root)
    for ts in ALL_DATES:
        d = ts.date()
        ds = d.strftime("%Y-%m-%d")
        if d.year <= 2019:
            dirs = [NLDAS_G_ROOT / str(d.year) / "raw" / ds]
        else:
            r = NLDAS_C_ROOTS[d.year]
            dirs = [r / "raw" / ds, r / ds]
        found = {}
        dir_exists = []
        for p in dirs:
            dir_exists.append(p.is_dir())
            for k, v in _scan_dir(p).items():
                found.setdefault(k, v)
        if d.year >= 2020:
            for k, v in root_cache[d.year].items():
                if f".A{d.strftime('%Y%m%d')}." in k:
                    found.setdefault(k, v)
        present, small, missing = [], [], []
        for h in range(24):
            n = nldas_granule_name(d, h)
            if n in found and found[n][1] >= MIN_GRANULE_BYTES:
                present.append(h)
            elif n in found:
                small.append(h)
            else:
                missing.append(h)
        rows.append(dict(date=ds, year=d.year, source_dirs=";".join(str(p) for p in dirs),
                         any_dir_exists=any(dir_exists), hours_available=len(present), hours_expected=24,
                         complete_day=len(present) == 24,
                         missing_hours=",".join(str(h) for h in missing + small),
                         undersized_hours=",".join(str(h) for h in small)))
        if d.day == 1 and d.month == 1:
            log.info("NLDAS inventory reached %s", ds)
    df = pd.DataFrame(rows)
    df.to_csv(AUDITS / "nldas_hourly_inventory_prebackfill.csv", index=False)
    rep = df[["date", "year", "hours_available", "hours_expected", "complete_day", "missing_hours"]]
    rep.to_csv(AUDITS / "nldas_daily_completeness_report_prebackfill.csv", index=False)
    by_year = df.groupby("year").agg(days=("date", "size"), complete_days=("complete_day", "sum"),
                                     fully_missing_days=("hours_available", lambda s: int((s == 0).sum())),
                                     partial_days=("hours_available", lambda s: int(((s > 0) & (s < 24)).sum())),
                                     missing_hours=("hours_available", lambda s: int((24 - s).sum())))
    by_year.to_csv(AUDITS / "nldas_completeness_by_year_prebackfill.csv")
    log.info("NLDAS completeness by year (pre-backfill):\n%s", by_year.to_string())
    return df, by_year


def audit_processed():
    out = {}
    y = pd.read_parquet(USDA_YIELD)
    out["usda_yield"] = dict(path=str(USDA_YIELD), rows=len(y), counties=int(y.county_fips.nunique()),
                             years=[int(y.year.min()), int(y.year.max())],
                             duplicate_county_year=int(y.duplicated(["county_fips", "year"]).sum()),
                             yield_min=float(y.corn_yield_bu_acre.min()), yield_max=float(y.corn_yield_bu_acre.max()),
                             null_yield=int(y.corn_yield_bu_acre.isna().sum()))
    p = pd.read_parquet(USDA_PROD)
    out["usda_production"] = dict(path=str(USDA_PROD), rows=len(p), counties=int(p.county_fips.nunique()))
    n = pd.read_parquet(NOAA_COUNTY_YEAR)
    out["noaa_county_year"] = dict(path=str(NOAA_COUNTY_YEAR), rows=len(n), counties=int(n.GEOID.nunique()),
                                   states=sorted(n.State.unique().tolist()),
                                   duplicate_county_year=int(n.duplicated(["GEOID", "Year"]).sum()),
                                   status_counts=n.storm_data_status.value_counts().to_dict())
    out["noaa_events_clean"] = dict(path=str(NOAA_EVENTS), size_mb=round(NOAA_EVENTS.stat().st_size / 1e6, 2))
    out["noaa_enhanced_csv"] = dict(path=str(NOAA_ENHANCED), size_mb=round(NOAA_ENHANCED.stat().st_size / 1e6, 2))
    out["old_mega"] = dict(path=str(OLD_MEGA), size_mb=round(OLD_MEGA.stat().st_size / 1e6, 2))
    out["nldas_fixed_elevation"] = dict(path=str(NLDAS_ELEV), exists=NLDAS_ELEV.exists())
    c = load_study_counties()
    out["county_universe"] = dict(counties=len(c), by_state=c.groupby("state").size().to_dict(),
                                  theoretical_county_years=len(c) * 39,
                                  note="User-stated 676 counties includes Nebraska (93); the six study states contain 583.")
    return out


if __name__ == "__main__":
    summary = {"audited_at": dt.datetime.now().isoformat(timespec="seconds")}
    summary["drives"] = audit_drives().to_dict(orient="records")
    s, probs = audit_prism()
    summary["prism"] = dict(root=str(PRISM_RAW), files_found=int(s.files_found.sum()), files_expected=int(s.files_expected.sum()),
                            size_gb=round(float(s.size_gb.sum()), 2), problems=len(probs))
    df, by_year = audit_nldas()
    summary["nldas_prebackfill"] = dict(dates=len(df), complete_days=int(df.complete_day.sum()),
                                        fully_missing_days=int((df.hours_available == 0).sum()),
                                        partial_days=int(((df.hours_available > 0) & (df.hours_available < 24)).sum()),
                                        missing_hours=int((24 - df.hours_available).sum()),
                                        by_year=by_year.reset_index().to_dict(orient="records"))
    summary.update(audit_processed())
    write_json(summary, AUDITS / "source_audit_summary.json")
    log.info("Source audit complete.")
