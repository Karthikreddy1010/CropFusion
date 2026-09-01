"""
eda.py - Automated Exploratory Data Analysis for Paper 3.

Generates comprehensive statistical summaries, missingness analysis,
distribution metrics, and exports a JSON report.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import config as cfg
from utils import log_decision, save_report

logger = logging.getLogger("paper3")


def run_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """Execute full EDA and return a structured report dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        Raw loaded dataset.

    Returns
    -------
    dict
        Nested report with dimensions, types, missingness, distributions, etc.
    """
    logger.info("=" * 60)
    logger.info("RUNNING EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)

    report: Dict[str, Any] = {}

    # ── 1. Dimensions & types ────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    report["dimensions"] = {
        "rows": len(df),
        "columns": len(df.columns),
        "numerical_columns": len(num_cols),
        "categorical_columns": len(cat_cols),
        "numerical_list": num_cols,
        "categorical_list": cat_cols,
    }
    logger.info("Dimensions: %d × %d (numerical: %d, categorical: %d)",
                len(df), len(df.columns), len(num_cols), len(cat_cols))

    # ── 2. Temporal & geographic ─────────────────────────────
    report["temporal"] = {
        "year_min": int(df["Year"].min()),
        "year_max": int(df["Year"].max()),
        "years_covered": int(df["Year"].nunique()),
    }
    report["geographic"] = {
        "states": df["State"].value_counts().to_dict(),
        "n_states": int(df["State"].nunique()),
        "n_counties": int(df["GEOID"].nunique()),
        "county_per_state": df.groupby("State")["GEOID"].nunique().to_dict(),
    }
    report["splits"] = df["Split"].value_counts().to_dict()

    # ── 3. Missing values ────────────────────────────────────
    report["missing_values"] = _analyze_missing(df)

    # ── 4. Duplicates & constant features ────────────────────
    report["duplicates"] = {
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_columns": int(df.columns.duplicated().sum()),
    }
    report["constant_features"] = _find_constant_features(df, num_cols)

    # ── 5. Distribution statistics ───────────────────────────
    report["distributions"] = _compute_distributions(df, num_cols)

    # ── 6. Target analysis ───────────────────────────────────
    report["targets"] = _analyze_targets(df)

    # ── 7. Disaster column analysis ──────────────────────────
    report["disaster_columns"] = _analyze_disaster_cols(df)

    # ── 8. Feature cardinality ───────────────────────────────
    report["cardinality"] = {
        col: int(df[col].nunique()) for col in df.columns
    }

    # ── 9. State-wise target distributions ───────────────────
    report["state_target_stats"] = _state_target_stats(df)

    # ── Save report ──────────────────────────────────────────
    save_report(report, "eda_report.json")
    logger.info("EDA complete. Report saved.")

    return report


def _analyze_missing(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze missing values across all columns."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    result = {}
    for col in df.columns:
        if missing[col] > 0:
            result[col] = {
                "count": int(missing[col]),
                "percent": round(float(missing_pct[col]), 2),
            }
    logger.info("Columns with missing values: %d", len(result))
    return result


def _find_constant_features(
    df: pd.DataFrame, num_cols: List[str]
) -> Dict[str, Any]:
    """Identify zero-variance and near-zero-variance columns."""
    zero_var = [c for c in num_cols if df[c].nunique() <= 1]
    near_zero = [
        c for c in num_cols
        if df[c].std() < 1e-4 and df[c].nunique() > 1
    ]
    logger.info("Constant columns: %s", zero_var)
    logger.info("Near-zero variance columns: %s", near_zero)
    return {"zero_variance": zero_var, "near_zero_variance": near_zero}


def _compute_distributions(
    df: pd.DataFrame, num_cols: List[str]
) -> List[Dict[str, Any]]:
    """Compute mean, std, skewness, kurtosis, and IQR outlier % for each
    numerical feature."""
    records = []
    for col in num_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        q25, q75 = np.percentile(s, [25, 75])
        iqr = q75 - q25
        if iqr > 0:
            outliers = int(((s < q25 - 1.5 * iqr) | (s > q75 + 1.5 * iqr)).sum())
        else:
            outliers = 0
        records.append({
            "feature": col,
            "count": int(len(s)),
            "mean": round(float(s.mean()), 6),
            "std": round(float(s.std()), 6),
            "min": round(float(s.min()), 6),
            "max": round(float(s.max()), 6),
            "skewness": round(float(s.skew()), 4),
            "kurtosis": round(float(s.kurtosis()), 4),
            "outlier_iqr_pct": round(outliers / len(s) * 100, 2),
        })
    return records


def _analyze_targets(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze target variable distributions."""
    result = {}
    for t in cfg.TARGET_COLS:
        if t not in df.columns:
            continue
        s = df[t].dropna()
        result[t] = {
            "valid_count": int(len(s)),
            "missing_count": int(df[t].isnull().sum()),
            "missing_pct": round(float(df[t].isnull().mean() * 100), 2),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "skewness": round(float(s.skew()), 4),
            "kurtosis": round(float(s.kurtosis()), 4),
        }
    return result


def _analyze_disaster_cols(df: pd.DataFrame) -> Dict[str, Any]:
    """Report missingness and stats for disaster-related columns."""
    all_disaster = cfg.CDHW_COLS + cfg.ENSO_COLS + cfg.DROUGHT_COLS + cfg.STORM_COLS
    result = {}
    for col in all_disaster:
        if col not in df.columns:
            continue
        m_cnt = int(df[col].isnull().sum())
        m_pct = round(float(df[col].isnull().mean() * 100), 2)
        result[col] = {
            "missing_count": m_cnt,
            "missing_pct": m_pct,
            "dtype": str(df[col].dtype),
            "nunique": int(df[col].nunique()),
            "retained": True,
            "retention_reason": (
                "Scientifically essential for Paper 3 CDHW/ACI objectives"
                if m_pct > 50
                else "Complete or near-complete data"
            ),
        }
    return result


def _state_target_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute per-state target mean, std, count."""
    result = {}
    for state in cfg.LOSO_STATES:
        state_df = df[df["State"] == state]
        result[state] = {}
        for t in [cfg.PRIMARY_TARGET, cfg.SECONDARY_TARGET]:
            s = state_df[t].dropna()
            result[state][t] = {
                "count": int(len(s)),
                "mean": round(float(s.mean()), 4) if len(s) > 0 else None,
                "std": round(float(s.std()), 4) if len(s) > 0 else None,
            }
    return result
