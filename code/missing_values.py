"""
missing_values.py - Disaster-aware missing value handling and EDA profiling.

NOTE:
- Initial EDA & Profiling: Computes structural presence flags (Has_Storm, Has_Drought).
- Model Training Pipelines: All learned imputation parameters (medians, fallbacks, normals)
  are strictly fitted on training splits using `TrainFittedPreprocessor` in `preprocessor.py`.
- Target variables: NEVER imputed.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import config as cfg
from utils import log_decision, save_report

logger = logging.getLogger("paper3")


def handle_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply dataset-justified missing value handling.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        Processed DataFrame and missing-value report.
    """
    logger.info("=" * 60)
    logger.info("HANDLING MISSING VALUES")
    logger.info("=" * 60)

    df = df.copy()
    report: Dict[str, Any] = {"before": {}, "after": {}, "decisions": []}

    # Record pre-imputation state
    for col in df.columns:
        m = int(df[col].isnull().sum())
        if m > 0:
            report["before"][col] = {
                "missing_count": m,
                "missing_pct": round(m / len(df) * 100, 2),
            }

    # ── 1. Storm columns: structural zero-fill ───────────────
    df, storm_decisions = _handle_storm_missing(df)
    report["decisions"].extend(storm_decisions)

    # ── 2. Drought columns: structural zero-fill ─────────────
    df, drought_decisions = _handle_drought_missing(df)
    report["decisions"].extend(drought_decisions)

    # ── 3. Soil columns: documented for TrainFittedPreprocessor ──
    df, soil_decisions = _handle_soil_missing(df)
    report["decisions"].extend(soil_decisions)

    # ── 4. Target variables: NEVER impute ────────────────────
    target_decisions = _log_target_missing(df)
    report["decisions"].extend(target_decisions)

    # Record post-imputation state
    for col in df.columns:
        m = int(df[col].isnull().sum())
        if m > 0:
            report["after"][col] = {
                "missing_count": m,
                "missing_pct": round(m / len(df) * 100, 2),
            }

    remaining_missing = {
        col: int(df[col].isnull().sum())
        for col in df.columns if df[col].isnull().sum() > 0
    }
    report["remaining_missing_summary"] = remaining_missing
    logger.info("Remaining columns with missing values (to be handled by TrainFittedPreprocessor): %s", remaining_missing)

    save_report(report, "missing_values_report.json")

    # Generate missing_data_audit.json
    audit = {
        "missing_percentage_before": {c: v["missing_pct"] for c, v in report["before"].items()},
        "missing_mechanism": "Structural pre-2000 data gaps (NOAA Storm Events / USDM Drought Monitor)",
        "imputation_methods": {d["column"]: d.get("strategy") for d in report["decisions"]},
        "remaining_nans": remaining_missing,
        "columns_removed": [],
        "columns_preserved_despite_missingness": cfg.DATA_AVAILABILITY_FLAGS + cfg.STORM_COLS + cfg.DROUGHT_COLS,
        "disaster_columns_intentionally_retained": [
            "Has_Drought", "Has_Storm", "Has_ERA5", "Has_CDHW",
            "Storm_Events_Total", "DSCI_growseason_mean"
        ],
        "methodology_note": "Disaster columns intentionally retained with indicator flags per §4.1 methodology requirement; learned statistics strictly train-fitted.",
    }
    save_report(audit, "missing_data_audit.json")
    logger.info("Missing data audit report saved to missing_data_audit.json")
    return df, report


def _handle_storm_missing(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Zero-fill storm columns where Has_Storm == 0 (pre-2000 structural gap)."""
    decisions = []
    for col in cfg.STORM_COLS:
        if col in cfg.CONSTANT_COLS:
            # Storm_Heat_Count and Storm_Drought_Event_Count are constant (all zero)
            df[col] = df[col].fillna(0.0)
            log_decision(
                step="missing_values",
                decision=f"Zero-fill constant column {col}",
                reason="Column is constant (all zeros where present) and missing "
                       "is structural (pre-2000 NOAA data gap)",
            )
            decisions.append({
                "column": col,
                "strategy": "zero_fill",
                "reason": "Constant column with structural pre-2000 missingness",
            })
            continue

        before_missing = int(df[col].isnull().sum())
        # Fill missing where Has_Storm == 0 with 0
        mask_no_storm = df["Has_Storm"] == 0
        df.loc[mask_no_storm & df[col].isnull(), col] = 0.0

        # Any remaining missing (Has_Storm == 1 but still NaN) — also zero-fill
        remaining = int(df[col].isnull().sum())
        if remaining > 0:
            df[col] = df[col].fillna(0.0)

        after_missing = int(df[col].isnull().sum())
        log_decision(
            step="missing_values",
            decision=f"Zero-fill {col} ({before_missing} → {after_missing})",
            reason="Structural missingness: pre-2000 NOAA Storm Events not collected. "
                   "Has_Storm indicator flag preserved as companion feature.",
        )
        decisions.append({
            "column": col,
            "strategy": "zero_fill_with_indicator",
            "indicator_flag": "Has_Storm",
            "before_missing": before_missing,
            "after_missing": after_missing,
            "reason": "Structural pre-2000 data gap; scientifically essential",
        })
    return df, decisions


def _handle_drought_missing(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Zero-fill drought columns where Has_Drought == 0 (pre-2000 USDM gap)."""
    decisions = []
    for col in cfg.DROUGHT_COLS:
        before_missing = int(df[col].isnull().sum())
        mask_no_drought = df["Has_Drought"] == 0
        df.loc[mask_no_drought & df[col].isnull(), col] = 0.0

        # Any remaining
        remaining = int(df[col].isnull().sum())
        if remaining > 0:
            df[col] = df[col].fillna(0.0)

        after_missing = int(df[col].isnull().sum())
        log_decision(
            step="missing_values",
            decision=f"Zero-fill {col} ({before_missing} → {after_missing})",
            reason="Structural missingness: pre-2000 US Drought Monitor not available. "
                   "Has_Drought indicator flag preserved.",
        )
        decisions.append({
            "column": col,
            "strategy": "zero_fill_with_indicator",
            "indicator_flag": "Has_Drought",
            "before_missing": before_missing,
            "after_missing": after_missing,
            "reason": "Structural pre-2000 USDM data gap; scientifically essential",
        })
    return df, decisions


def _handle_soil_missing(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Document Soil_Sand_Mean missingness; learned imputation is handled by TrainFittedPreprocessor."""
    decisions = []
    col = "Soil_Sand_Mean"
    if col not in df.columns:
        return df, decisions

    missing_count = int(df[col].isnull().sum())
    if missing_count == 0:
        return df, decisions

    log_decision(
        step="missing_values",
        decision=f"Deferred learned state-median imputation for {col} ({missing_count} missing)",
        reason="Learned statistics (state medians & global medians) must be fitted strictly on train data only via TrainFittedPreprocessor to prevent data leakage.",
    )
    decisions.append({
        "column": col,
        "strategy": "train_fitted_state_median_imputation",
        "before_missing": missing_count,
        "after_missing": missing_count,
        "reason": "Learned state-median imputation deferred to TrainFittedPreprocessor (Zero Leakage)",
    })
    return df, decisions


def _log_target_missing(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Log target variable missingness without imputation."""
    decisions = []
    for t in cfg.TARGET_COLS:
        m = int(df[t].isnull().sum())
        if m > 0:
            log_decision(
                step="missing_values",
                decision=f"Target {t}: {m} missing values PRESERVED (no imputation)",
                reason="Target variables are never imputed. Missing rows are "
                       "excluded during crop-specific model fitting.",
            )
            decisions.append({
                "column": t,
                "strategy": "no_imputation",
                "missing_count": m,
                "reason": "Target variable — never imputed",
            })
    return decisions
