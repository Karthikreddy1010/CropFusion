"""
md_common.py -- shared paths, study universe, storage guard, logging and merge
auditing for the Paper 3 PRISM-NLDAS-NOAA-USDA master dataset build.

This module never writes to raw data locations. All outputs go under
C:/Users/dukar/OneDrive/Desktop/Paper3/master_dataset/.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (read-only sources)
# ---------------------------------------------------------------------------
PAPER3 = Path(r"C:\Users\dukar\OneDrive\Desktop\Paper3")
OLD_MEGA = PAPER3 / "Paper3_MegaDataset_SPEI_FINAL.csv"
COUNTY_GEOJSON = PAPER3 / "outputs (1)" / "geo_cache" / "counties_fips.json"

PRISM_RAW = Path(r"D:\prism_pipeline\raw")
NLDAS_G_ROOT = Path(r"G:\My Drive\NLDAS")          # 1985-2019: <Y>/raw/<YYYY-MM-DD>/
NLDAS_C_ROOTS = {                                   # 2020-2023 on C:
    2020: Path(r"C:\nldas_curl_2020"),
    2021: Path(r"C:\nldas_curl_2021"),
    2022: Path(r"C:\nldas_curl_2022"),
    2023: Path(r"C:\nldas_curl_2023"),
}
NLDAS_ELEV = PAPER3 / "data" / "nldas_pipeline" / "config" / "nldas_grid_elevation.nc"
# User's ArcGIS Pro zonal-statistics extraction of the static layers (identical to the old MegaDataset values),
# copied from Downloads into the project for reproducibility (Final_ARCGIS.zip holds the per-state tables).
ARCGIS_DIR = PAPER3 / "master_dataset" / "intermediate" / "static" / "arcgis_source"
ARCGIS_STATIC = ARCGIS_DIR / "Merged_7States.csv"
# Iowa sand zonal table ("Soil" table in the user's Iowa ArcGIS project; never joined into Iowa_final.csv)
ARCGIS_IOWA_SAND = ARCGIS_DIR / "lowa_soil.csv"
NLDAS_SCRIPTS = PAPER3 / "code" / "pipelines" / "nldas"   # validated fao56_et0.py

USDA_YIELD = PAPER3 / "data" / "usda_nass" / "processed" / "usda_corn_yield_1985_2023.parquet"
USDA_PROD = PAPER3 / "data" / "usda_nass" / "processed" / "usda_corn_production_1985_2023.parquet"
NOAA_DIR = PAPER3 / "data" / "noaa_storm_pipeline" / "processed"
NOAA_COUNTY_YEAR = NOAA_DIR / "noaa_storm_county_year_1985_2023.parquet"
NOAA_EVENTS = NOAA_DIR / "noaa_storm_events_clean.parquet"
NOAA_ENHANCED = NOAA_DIR / "Paper3_NOAA_Storm_Enhanced.csv"

# ---------------------------------------------------------------------------
# Output workspace
# ---------------------------------------------------------------------------
MD = PAPER3 / "master_dataset"
FULL = MD / "full"
MODEL = MD / "model_ready"
META = MD / "metadata"
AUDITS = MD / "audits"
INTER = MD / "intermediate"
LOGS = MD / "logs"
for _p in (FULL, MODEL, META, AUDITS, INTER, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Study design
# ---------------------------------------------------------------------------
START = dt.date(1985, 1, 1)
END = dt.date(2023, 12, 31)
YEARS = list(range(1985, 2024))
STUDY_STATES = {  # state_fips -> (alpha, name)
    "17": ("IL", "Illinois"),
    "18": ("IN", "Indiana"),
    "19": ("IA", "Iowa"),
    "27": ("MN", "Minnesota"),
    "29": ("MO", "Missouri"),
    "39": ("OH", "Ohio"),
}
ALL_DATES = pd.date_range(START, END, freq="D")
assert len(ALL_DATES) == 14244

# Growing season recovered from the old dataset: ET0_growseason_mm / ET0_growseason_mean
# = 214 days = 1 April - 31 October (see old_dataset_variable_audit.csv).
GS_START_MMDD = (4, 1)
GS_END_MMDD = (10, 31)
# Existing project GDD phenology thresholds (code/config.py, unchanged)
GDD_VEGETATIVE_END = 700.0
GDD_SILKING_END = 1400.0
TMAX_THRESHOLD_C = 35.0
SPEI_THRESHOLD = -1.0


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(LOGS / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def free_gb(path: str | Path) -> float:
    return shutil.disk_usage(str(Path(path).anchor)).free / 1e9


def storage_guard(min_free_gb: float = 2.0, drives=("C:\\",), max_wait_s: int = 4 * 3600, logger=None) -> None:
    """Pause while any output/cache drive is below the free-space floor (other applications on this
    machine cause transient drops); raise only if space does not recover within max_wait_s."""
    import time

    waited = 0
    while True:
        low = [(d, free_gb(d)) for d in drives if free_gb(d) < min_free_gb]
        if not low:
            return
        if waited >= max_wait_s:
            raise RuntimeError(f"STORAGE GUARD: {low} below {min_free_gb} GB for {waited}s. Stopping.")
        msg = f"STORAGE GUARD: {low} below {min_free_gb} GB -- pausing 60 s (waited {waited}s)"
        (logger.warning(msg) if logger else print(msg, flush=True))
        time.sleep(60)
        waited += 60


def load_study_counties() -> "pd.DataFrame":
    """583 counties of the six study states from the project's validated county GeoJSON."""
    import geopandas as gpd

    gdf = gpd.read_file(COUNTY_GEOJSON)
    gdf["county_fips"] = gdf["STATE"].astype(str).str.zfill(2) + gdf["COUNTY"].astype(str).str.zfill(3)
    gdf = gdf[gdf["STATE"].isin(STUDY_STATES.keys())].copy()
    gdf["state_fips"] = gdf["STATE"]
    gdf["state"] = gdf["state_fips"].map(lambda s: STUDY_STATES[s][0])
    gdf["state_name"] = gdf["state_fips"].map(lambda s: STUDY_STATES[s][1])
    gdf["county"] = gdf["NAME"]
    gdf = gdf.sort_values("county_fips").reset_index(drop=True)
    return gdf[["state", "state_name", "state_fips", "county", "county_fips", "geometry"]]


def assert_unique_key(df: pd.DataFrame, keys=("county_fips", "year"), name: str = "") -> None:
    dup = int(df.duplicated(list(keys)).sum())
    if dup:
        raise AssertionError(f"[{name}] {dup} duplicate {keys} keys -- STOP")


class MergeAuditor:
    """Records rows/cols/duplicates/new-missing after every join; stops on row growth."""

    def __init__(self):
        self.records = []

    def merge(self, left: pd.DataFrame, right: pd.DataFrame, step: str, on=("state_fips", "county_fips", "year"),
              how: str = "left") -> pd.DataFrame:
        on = list(on)
        assert_unique_key(left, on, f"{step}:left")
        assert_unique_key(right, on, f"{step}:right")
        new_cols = [c for c in right.columns if c not in on]
        clash = [c for c in new_cols if c in left.columns]
        if clash:
            raise AssertionError(f"[{step}] column name collision: {clash}")
        missing_before = int(left.isna().sum().sum())
        out = left.merge(right, on=on, how=how, validate="one_to_one")
        if len(out) != len(left):
            raise AssertionError(f"[{step}] row count changed {len(left)} -> {len(out)} -- STOP (many-to-many?)")
        dup = int(out.duplicated(on).sum())
        rec = dict(
            step=step,
            rows_before=len(left),
            rows_after=len(out),
            columns_before=left.shape[1],
            columns_after=out.shape[1],
            columns_added=len(new_cols),
            duplicate_keys=dup,
            new_missing_values=int(out.isna().sum().sum()) - missing_before,
            new_missing_in_added_columns=int(out[new_cols].isna().sum().sum()),
            right_rows=len(right),
            right_rows_unmatched=int(len(right) - right.merge(left[on], on=on, how="inner").shape[0]),
            county_count=int(out["county_fips"].nunique()),
            year_min=int(out["year"].min()),
            year_max=int(out["year"].max()),
        )
        self.records.append(rec)
        return out

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


def write_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def nldas_hour_dirs(d: dt.date) -> list[Path]:
    """All candidate directories that may hold hourly granules for date d (primary first)."""
    ds = d.strftime("%Y-%m-%d")
    y = d.year
    if y <= 2019:
        return [NLDAS_G_ROOT / str(y) / "raw" / ds]
    root = NLDAS_C_ROOTS[y]
    cands = [root / "raw" / ds, root / ds, root]  # 2023 stores date dirs at root; stray granules at root
    return cands


def nldas_granule_name(d: dt.date, hour: int) -> str:
    return f"NLDAS_FORA0125_H.A{d.strftime('%Y%m%d')}.{hour:02d}00.020.nc"


BACKFILL_DIR = INTER / "nldas_backfill"   # study-region subsets of backfilled hours (never written to raw)
