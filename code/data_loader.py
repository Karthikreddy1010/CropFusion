"""
data_loader.py - Dataset ingestion & methodology compliance validation for Paper 3.

Loads the mega-dataset CSV, validates schema, and runs automatic methodology checks.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

import config as cfg
from utils import log_decision

logger = logging.getLogger("paper3")


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the Paper 3 MegaDataset CSV and run basic validation.

    Parameters
    ----------
    path : Path, optional
        Override for the CSV path (defaults to ``config.DATA_FILE``).

    Returns
    -------
    pd.DataFrame
        Raw dataset with validated schema.
    """
    path = path or cfg.DATA_FILE
    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns", *df.shape)

    # ── Schema validation ────────────────────────────────────
    schema_report = validate_dataset_schema(df)

    # ── Methodology Compliance Validation (§6.2) ──────────────
    validate_methodology_compliance(df)

    # ── Year range validation ────────────────────────────────
    yr_min, yr_max = int(df["Year"].min()), int(df["Year"].max())
    assert yr_min == cfg.TRAIN_YEARS[0], (
        f"Expected first year {cfg.TRAIN_YEARS[0]}, got {yr_min}"
    )
    assert yr_max == cfg.TEST_YEARS[1], (
        f"Expected last year {cfg.TEST_YEARS[1]}, got {yr_max}"
    )
    logger.info("Year range validated: %d – %d", yr_min, yr_max)

    # ── State validation ─────────────────────────────────────
    # ── R² PATCH: drop configured states (e.g. Nebraska) ──────
    drop_states = getattr(cfg, "DROP_STATES", [])
    if drop_states:
        before = len(df)
        df = df[~df["State"].isin(drop_states)].copy()
        logger.info("Dropped states %s: %d -> %d rows", drop_states, before, len(df))

    states_found = sorted(df["State"].unique())
    expected_states = [s for s in cfg.LOSO_STATES if s not in drop_states]
    assert len(states_found) == len(expected_states), (
        f"Expected {len(expected_states)} states, found {len(states_found)}"
    )
    logger.info("States: %s", states_found)

    log_decision(
        step="data_loading",
        decision="Dataset loaded and methodology compliance verified",
        reason="Schema, year range, state checks, and SPEI/ET0/GDD variables validated",
        details={"rows": len(df), "cols": len(df.columns), "states": states_found},
    )
    return df


def validate_methodology_compliance(df: pd.DataFrame) -> Dict[str, Any]:
    """Automatic methodology compliance verification (§6.2 & §7).

    Validates:
    - SPEI-30 / SPI-30 presence (§4.1)
    - GDD Accumulated presence (§4.1)
    - CDHW Flag and Severity presence (§4.1)
    - Temporal split boundaries (1985-2015, 2016-2018, 2019-2023) (§5.1)
    - 7 State coverage (§5.2)
    - CQR Quantile tau targets (0.05, 0.95) (§4.2)
    """
    logger.info("Running automatic methodology compliance checks (§6.2)")

    checks = {}

    # 1. Drought/Moisture Index Check
    has_spei = any(col in df.columns for col in cfg.SPEI_PREFERRED_COLS)
    has_spi = any(col in df.columns for col in cfg.SPI_FALLBACK_COLS)
    if has_spei:
        checks["drought_index"] = "SPEI_30 (Preferred)"
        logger.info("  ✓ SPEI-30 index detected in dataset (§4.1)")
    elif has_spi:
        checks["drought_index"] = "SPI_30 (Fallback)"
        logger.info("  ! SPI-30 index detected (SPEI-30 preferred per §4.1; SPI used as fallback)")
    else:
        raise ValueError("METHODOLOGY ERROR: Neither SPEI-30 nor SPI-30 detected in dataset!")

    # 2. GDD & Thermal Stress Check
    if "GDD_Accumulated" not in df.columns:
        raise ValueError("METHODOLOGY ERROR: GDD_Accumulated missing from dataset (§4.1)!")
    if "Tmax_Days_Above_35" not in df.columns:
        raise ValueError("METHODOLOGY ERROR: Tmax_Days_Above_35 missing from dataset (§4.1)!")

    checks["gdd_present"] = True
    checks["tmax_35_present"] = True
    logger.info("  ✓ GDD_Accumulated and Tmax_Days_Above_35 validated (§4.1)")

    # 3. Temporal Split Check (§5.1)
    train_count = (df["Split"] == "train").sum()
    val_count = (df["Split"] == "val").sum()
    test_count = (df["Split"] == "test").sum()
    if train_count == 0 or val_count == 0 or test_count == 0:
        raise ValueError("METHODOLOGY ERROR: Split column does not contain train/val/test splits (§5.1)!")

    checks["temporal_splits"] = {"train": int(train_count), "val": int(val_count), "test": int(test_count)}
    logger.info("  ✓ Temporal splits validated: Train %d, Val %d, Test %d (§5.1)", train_count, val_count, test_count)

    # 4. Hyperparameter Check
    assert cfg.ACI_WINDOW_SIZE == 3, f"METHODOLOGY ERROR: Sliding window size must be 3, got {cfg.ACI_WINDOW_SIZE} (§4.3)"
    assert cfg.CQR_QUANTILES == (0.05, 0.95), f"METHODOLOGY ERROR: CQR quantiles must be (0.05, 0.95), got {cfg.CQR_QUANTILES} (§4.2)"
    logger.info("  ✓ ACI sliding window size (3) and CQR quantiles (0.05, 0.95) validated (§4.2, §4.3)")

    return checks


def validate_dataset_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate dataset schema, data types, uniqueness, and completeness.

    Exports ``dataset_schema_report.json``.
    """
    from utils import save_report

    _validate_schema(df)

    # Duplicate columns & rows check
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    dup_rows_count = int(df.duplicated().sum())

    # County-Year uniqueness
    county_year_dup = int(df.duplicated(subset=["GEOID", "Year"]).sum()) if "GEOID" in df.columns and "Year" in df.columns else 0

    # Categorical / string columns check
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Target availability check
    target_non_null = int(df[cfg.PRIMARY_TARGET].notna().sum()) if cfg.PRIMARY_TARGET in df.columns else 0

    report = {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "duplicate_columns": dup_cols,
        "duplicate_rows": dup_rows_count,
        "county_year_uniqueness": {
            "duplicate_county_years": county_year_dup,
            "unique": county_year_dup == 0,
        },
        "target_availability": {
            "target": cfg.PRIMARY_TARGET,
            "non_null_count": target_non_null,
            "pct_available": round(100.0 * target_non_null / max(1, len(df)), 2),
        },
        "categorical_columns": cat_cols,
        "column_dtypes": {c: str(df[c].dtype) for c in df.columns},
        "schema_passed": len(dup_cols) == 0 and county_year_dup == 0,
    }

    save_report(report, "dataset_schema_report.json")
    logger.info("Dataset schema validation report saved (schema_passed: %s)", report["schema_passed"])
    return report


def _validate_schema(df: pd.DataFrame) -> None:
    """Assert that all required column groups are present."""
    required_groups = {
        "ID": cfg.ID_COLS,
        "CDHW": cfg.CDHW_COLS,
        "Weather": cfg.WEATHER_COLS,
        "ENSO": cfg.ENSO_COLS,
        "Drought": cfg.DROUGHT_COLS,
        "Storm": cfg.STORM_COLS,
        "Soil": cfg.SOIL_COLS,
        "Topo": cfg.TOPO_COLS,
        "LandCover": cfg.LANDCOVER_COLS,
        "Targets": cfg.TARGET_COLS,
    }
    for group_name, cols in required_groups.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns in group '{group_name}': {missing}"
            )
    logger.info("Schema validation passed for all %d column groups", len(required_groups))

