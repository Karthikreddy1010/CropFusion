"""
master_data_loader.py - Data interface between the PRISM-NLDAS-NOAA-USDA master dataset and the
locked Paper 3 Neural CQR / ACI pipeline (active only when PAPER3_DATA_SOURCE=master).

The pipeline looks up methodology roles by their historical column names (e.g. SPEI_30_min,
CDHW_Severity_Score, GDD_Accumulated). This loader builds that view from the master dataset
WITHOUT changing any methodology:

1. Rows: the complete county-year skeleton of the FULL master (583 counties x 39 years), so the
   split-aware rolling climate windows see consecutive years exactly as with the old dataset.
   Unobserved yields stay NaN (never imputed; the splitters drop them).
2. Columns: only the MODEL-READY predictors (the leakage/redundancy-audited set), plus identifiers,
   the target and evaluation-only ENSO labels.
3. Identifiers renamed to the protocol names (GEOID, County_Name, State, STATEFP, COUNTYFP, Lat, Lon, Year).
   `Split` is derived from the locked config boundaries (TRAIN/VAL/TEST_YEARS), it is not stored data.
4. Target: Corn_Yield_tha = corn_yield_bu_acre x 0.0627 (the conversion used by the old dataset), so the
   loss scale and tuned hyperparameters are unchanged; Corn_Yield_buacre is kept as the redundant target.
5. Methodology roles are RENAMED (not duplicated) from their PRISM-NLDAS source columns via
   config.MASTER_ROLE_ALIASES. No ERA5-named column is ever created; the SPEI x Tmax interaction reads
   config.INTERACTION_TMAX_COL instead.
6. ENSO_Phase / ENSO_Anomalous_Year are loaded for evaluation stratification only: they are listed in
   config.ID_COLS (never features) and config.ENSO_AS_FEATURES=False stops the dummy encoding.

A manifest (source hashes, role map, row/column counts, checks) is written to
<OUTPUT_DIR>/audits/master_data_role_map.json on every load.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config as cfg

logger = logging.getLogger("paper3")

TONNES_PER_HA_PER_BU_ACRE = 0.0627   # identical to the factor in Paper3_MegaDataset_SPEI_FINAL.csv
MASTER_ID_MAP = {
    "county_fips": "GEOID", "county": "County_Name", "state_name": "State", "state_fips": "STATEFP",
    "county_code": "COUNTYFP", "county_lat": "Lat", "county_lon": "Lon", "year": "Year",
}
EVALUATION_ONLY_MAP = {"enso_phase": "ENSO_Phase", "enso_anomalous_year": "ENSO_Anomalous_Year"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_label(year: pd.Series) -> pd.Series:
    """Legacy 3-way Split label from the locked config boundaries (used by the compliance check only)."""
    return pd.Series(np.select(
        [year.between(*cfg.TRAIN_YEARS), year.between(*cfg.VAL_YEARS), year.between(*cfg.TEST_YEARS)],
        ["train", "val", "test"], default="unassigned"), index=year.index)


def load_master_dataset(full_path: Optional[Path] = None, model_ready_path: Optional[Path] = None) -> pd.DataFrame:
    full_path = Path(full_path or cfg.DATA_FILE)
    model_ready_path = Path(model_ready_path or cfg.MASTER_MODEL_READY_FILE)
    logger.info("Loading MASTER dataset (role-alias view): %s", full_path)

    model_cols = pq.read_schema(model_ready_path).names
    predictors = [c for c in model_cols if c not in MASTER_ID_MAP and c not in ("state", "corn_yield_bu_acre")]
    needed = list(MASTER_ID_MAP) + ["corn_yield_bu_acre"] + predictors + list(EVALUATION_ONLY_MAP)
    full = pd.read_parquet(full_path, columns=needed)

    # ── checks on the source before any renaming ───────────────────────────
    if full.duplicated(["county_fips", "year"]).any():
        raise ValueError("MASTER DATA ERROR: duplicate county_fips + year rows")
    era5 = [c for c in full.columns if "era5" in c.lower()]
    if era5:
        raise ValueError(f"MASTER DATA ERROR: ERA5 columns present: {era5}")
    missing_sources = [src for src in cfg.MASTER_ROLE_ALIASES if src not in full.columns]
    if missing_sources:
        raise ValueError(f"MASTER DATA ERROR: role-alias source columns absent from MODEL-READY set: {missing_sources}")
    clash = [dst for dst in cfg.MASTER_ROLE_ALIASES.values() if dst in full.columns]
    if clash:
        raise ValueError(f"MASTER DATA ERROR: alias targets already exist as columns: {clash}")

    df = full.rename(columns={**MASTER_ID_MAP, **cfg.MASTER_ROLE_ALIASES, **EVALUATION_ONLY_MAP})
    df["GEOID"] = df["GEOID"].astype("int64")
    df["STATEFP"] = df["STATEFP"].astype("int64")
    df["COUNTYFP"] = df["COUNTYFP"].astype("int64")
    df["Year"] = df["Year"].astype("int64")
    df["Split"] = _split_label(df["Year"])

    # target (t/ha, same conversion as the old dataset) and presence flag
    df["Corn_Yield_buacre"] = df.pop("corn_yield_bu_acre").astype("float64")
    df["Corn_Yield_tha"] = df["Corn_Yield_buacre"] * TONNES_PER_HA_PER_BU_ACRE
    df["Has_Corn_Yield"] = df["Corn_Yield_tha"].notna().astype("int64")

    # CDHW_Flag role (season-level: any CDHW day in the growing season), as in the old CDHW_COLS
    df["CDHW_Flag"] = (df["CDHW_Event_Count"] > 0).astype("int64")

    float_cols = df.select_dtypes(include=["float32"]).columns
    df[float_cols] = df[float_cols].astype("float64")

    front = ["GEOID", "County_Name", "State", "STATEFP", "COUNTYFP", "Lat", "Lon", "Year", "Split",
             "Corn_Yield_tha", "Corn_Yield_buacre", "Has_Corn_Yield", "ENSO_Phase", "ENSO_Anomalous_Year"]
    df = df[front + [c for c in df.columns if c not in front]].sort_values(["GEOID", "Year"]).reset_index(drop=True)

    # ── checks on the pipeline view ────────────────────────────────────────
    checks: Dict[str, Any] = {
        "rows": len(df), "counties": int(df.GEOID.nunique()), "years": [int(df.Year.min()), int(df.Year.max())],
        "states": sorted(df.State.unique().tolist()),
        "observed_yields": int(df.Corn_Yield_tha.notna().sum()),
        "duplicate_columns": df.columns[df.columns.duplicated()].tolist(),
        "unassigned_split_rows": int((df.Split == "unassigned").sum()),
        "legacy_era5_names_created": [c for c in df.columns if c.startswith("ERA5")],
    }
    expected_states = sorted(s for s in cfg.LOSO_STATES if s not in getattr(cfg, "DROP_STATES", []))
    problems = []
    if checks["duplicate_columns"]:
        problems.append(f"duplicate columns {checks['duplicate_columns']}")
    if checks["unassigned_split_rows"]:
        problems.append(f"{checks['unassigned_split_rows']} rows outside the configured split years")
    if checks["states"] != expected_states:
        problems.append(f"states {checks['states']} != configured {expected_states}")
    if checks["legacy_era5_names_created"]:
        problems.append("ERA5-named columns created")
    if problems:
        raise ValueError("MASTER DATA ERROR: " + "; ".join(problems))

    manifest = {
        "loaded_at": dt.datetime.now().isoformat(timespec="seconds"),
        "data_source": "master",
        "full_dataset": {"path": str(full_path), "sha256": _sha256(full_path)},
        "model_ready_column_set": {"path": str(model_ready_path), "sha256": _sha256(model_ready_path)},
        "row_policy": "complete FULL county-year skeleton; target NaN where USDA NASS did not publish a yield",
        "target": {"pipeline_column": "Corn_Yield_tha", "source": "corn_yield_bu_acre",
                   "conversion": f"x {TONNES_PER_HA_PER_BU_ACRE} (same factor as the old MegaDataset)"},
        "identifier_map": MASTER_ID_MAP,
        "role_aliases_renamed": cfg.MASTER_ROLE_ALIASES,
        "derived_columns": {"CDHW_Flag": "CDHW_Event_Count (cdhw_total_days_gs) > 0",
                            "Split": "derived from config TRAIN/VAL/TEST_YEARS (compliance label only)"},
        "evaluation_only_columns": EVALUATION_ONLY_MAP,
        "enso_policy": "ENSO_Phase/ENSO_Anomalous_Year listed in ID_COLS and ENSO_AS_FEATURES=False: stratification only, never features",
        "interaction_tmax_column": cfg.INTERACTION_TMAX_COL,
        "predictor_columns": [c for c in df.columns if c not in front],
        "checks": checks,
    }
    out = Path(cfg.AUDITS_DIR) / "master_data_role_map.json"
    out.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    logger.info("MASTER view: %d rows x %d cols, %d counties, %d observed yields; manifest -> %s",
                len(df), df.shape[1], checks["counties"], checks["observed_yields"], out)
    return df
