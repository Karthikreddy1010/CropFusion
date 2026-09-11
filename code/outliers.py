"""
outliers.py - Outlier detection and analysis for Paper 3.

IMPORTANT: Genuine climate extremes and disaster events are NEVER removed.
Only obvious measurement errors or physically impossible observations
are flagged for potential removal.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import config as cfg
from utils import log_decision, save_report

logger = logging.getLogger("paper3")

# Physical plausibility bounds for sanity checks
PLAUSIBILITY_BOUNDS = {
    "Corn_Yield_tha": (0.0, 25.0),
    "Soy_Yield_tha": (0.0, 8.0),
    "ERA5d_Tmax_mean_C": (-10.0, 50.0),
    "ERA5d_Tmax_max_C": (-5.0, 55.0),
    "ERA5d_Tmin_mean_C": (-20.0, 35.0),
    "Precip_growseason_mm": (0.0, 2000.0),
    "GDD_Accumulated": (0.0, 5000.0),
}


def analyze_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Analyze outliers across features; do NOT remove climate extremes.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset after missing value handling.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        DataFrame (unchanged unless impossible values found) and report.
    """
    logger.info("=" * 60)
    logger.info("OUTLIER ANALYSIS")
    logger.info("=" * 60)

    df = df.copy()
    report: Dict[str, Any] = {
        "iqr_analysis": {},
        "zscore_analysis": {},
        "isolation_forest": {},
        "plausibility_flags": {},
        "decision": "",
        "rows_removed": 0,
    }

    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in cfg.ID_COLS + cfg.CONSTANT_COLS
           + cfg.DATA_AVAILABILITY_FLAGS + cfg.YIELD_PRESENCE_COLS
    ]

    # ── 1. IQR-based outlier percentages ─────────────────────
    report["iqr_analysis"] = _iqr_analysis(df, num_cols)

    # ── 2. Z-score analysis ──────────────────────────────────
    report["zscore_analysis"] = _zscore_analysis(df, num_cols)

    # ── 3. Isolation Forest (anomaly fraction) ───────────────
    report["isolation_forest"] = _isolation_forest_analysis(df, num_cols)

    # ── 4. Physical plausibility check ───────────────────────
    impossible_mask = pd.Series(False, index=df.index)
    for col, (lo, hi) in PLAUSIBILITY_BOUNDS.items():
        if col not in df.columns:
            continue
        violations = (df[col].notna()) & ((df[col] < lo) | (df[col] > hi))
        n_violations = int(violations.sum())
        if n_violations > 0:
            logger.warning(
                "PLAUSIBILITY VIOLATION: %s has %d values outside [%.1f, %.1f]",
                col, n_violations, lo, hi,
            )
            report["plausibility_flags"][col] = {
                "bounds": [lo, hi],
                "violations": n_violations,
            }
            impossible_mask |= violations

    rows_to_remove = int(impossible_mask.sum())
    if rows_to_remove > 0:
        logger.warning(
            "Removing %d rows with physically impossible values", rows_to_remove
        )
        df = df[~impossible_mask].reset_index(drop=True)
        report["rows_removed"] = rows_to_remove
    else:
        logger.info("No physically impossible values detected — no rows removed")

    # ── Decision ─────────────────────────────────────────────
    report["decision"] = (
        "Outlier removal SKIPPED for climate features. "
        "Extreme values in CDHW_Severity_Score, Tmax_Days_Above_35, "
        "DSCI_growseason_mean, and Storm_Events_Total represent genuine "
        "compound climate events (1988 drought, 2012 drought, 2023 heat dome) "
        "and are scientifically essential for Paper 3 ACI evaluation. "
        f"Only {rows_to_remove} rows with physically impossible values removed."
    )
    log_decision(
        step="outlier_analysis",
        decision="Preserve all climate extremes",
        reason="Extreme weather events are scientifically essential for CDHW/ACI "
               "evaluation; only physically impossible values removed",
        details={"rows_removed": rows_to_remove},
    )

    save_report(report, "outlier_report.json")

    # Export outlier_audit.json
    audit = {
        "rows_flagged": int(sum(v.get("violations", 0) for v in report["plausibility_flags"].values())),
        "rows_removed": rows_to_remove,
        "rows_winsorized": 0,
        "reason": "Climate extremes preserved per methodology; only physically impossible measurement errors removed",
        "method": "Physical plausibility thresholding + IsolationForest diagnostic",
        "thresholds": PLAUSIBILITY_BOUNDS,
        "iqr_outliers_summary": {c: v["outlier_pct"] for c, v in report["iqr_analysis"].items() if v.get("outlier_pct", 0) > 0},
    }
    save_report(audit, "outlier_audit.json")
    logger.info("Outlier audit report saved to outlier_audit.json")
    return df, report


def _iqr_analysis(
    df: pd.DataFrame, cols: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Compute IQR-based outlier percentages per feature."""
    result = {}
    for col in cols:
        s = df[col].dropna()
        if len(s) < 4:
            continue
        q25, q75 = np.percentile(s, [25, 75])
        iqr = q75 - q25
        if iqr == 0:
            result[col] = {"outlier_count": 0, "outlier_pct": 0.0}
            continue
        n_outliers = int(((s < q25 - 1.5 * iqr) | (s > q75 + 1.5 * iqr)).sum())
        result[col] = {
            "q25": round(float(q25), 4),
            "q75": round(float(q75), 4),
            "iqr": round(float(iqr), 4),
            "outlier_count": n_outliers,
            "outlier_pct": round(n_outliers / len(s) * 100, 2),
        }
    return result


def _zscore_analysis(
    df: pd.DataFrame, cols: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Compute Z-score > 3 outlier percentages per feature."""
    result = {}
    for col in cols:
        s = df[col].dropna()
        if len(s) < 4 or s.std() == 0:
            continue
        z = np.abs((s - s.mean()) / s.std())
        n_outliers = int((z > 3).sum())
        result[col] = {
            "zscore_gt3_count": n_outliers,
            "zscore_gt3_pct": round(n_outliers / len(s) * 100, 2),
        }
    return result


def _isolation_forest_analysis(
    df: pd.DataFrame, cols: List[str]
) -> Dict[str, Any]:
    """Run Isolation Forest on key feature subsets for anomaly detection
    diagnostics (not for removal)."""
    # Use a subset of climate-relevant columns
    climate_cols = [
        c for c in (cfg.WEATHER_COLS + cfg.CDHW_COLS)
        if c in cols and c in df.columns
    ]
    if len(climate_cols) < 2:
        return {"status": "skipped", "reason": "Too few climate columns"}

    sub = df[climate_cols].dropna()
    if len(sub) < 100:
        return {"status": "skipped", "reason": "Too few complete rows"}

    iso = IsolationForest(
        contamination=0.05,
        random_state=cfg.RANDOM_SEED,
        n_jobs=-1,
    )
    preds = iso.fit_predict(sub)
    n_anomalies = int((preds == -1).sum())

    logger.info(
        "Isolation Forest: %d / %d flagged as anomalies (%.1f%%) — diagnostic only",
        n_anomalies, len(sub), n_anomalies / len(sub) * 100,
    )
    return {
        "status": "completed",
        "columns_used": climate_cols,
        "n_samples": len(sub),
        "n_anomalies": n_anomalies,
        "anomaly_pct": round(n_anomalies / len(sub) * 100, 2),
        "note": "Diagnostic only — no rows removed. Climate extremes preserved.",
    }
