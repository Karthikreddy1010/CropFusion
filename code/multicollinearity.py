"""
multicollinearity.py - Correlation and VIF analysis for Paper 3.

Computes Pearson/Spearman correlation matrices and Variance Inflation
Factors. Flags redundant columns but never removes scientifically
meaningful features without justification.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

import config as cfg
from utils import log_decision, save_report

logger = logging.getLogger("paper3")


def analyze_multicollinearity(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute correlations, VIF, and flag redundant features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset after missing-value handling.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        DataFrame with redundant columns dropped (if justified) and report.
    """
    logger.info("=" * 60)
    logger.info("MULTICOLLINEARITY ANALYSIS")
    logger.info("=" * 60)

    df = df.copy()
    report: Dict[str, Any] = {}

    # Build feature matrix (exclude IDs, targets, constants, flags)
    exclude = (
        cfg.ID_COLS + cfg.TARGET_COLS + cfg.CONSTANT_COLS
        + cfg.DATA_AVAILABILITY_FLAGS + cfg.YIELD_PRESENCE_COLS
        + cfg.REDUNDANT_TARGET_COLS
    )
    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]
    logger.info("Analyzing %d numerical features for multicollinearity",
                len(feature_cols))

    # ── 1. Pearson correlation ───────────────────────────────
    pearson_corr = df[feature_cols].corr(method="pearson")
    high_pearson = _find_high_correlations(pearson_corr, threshold=0.90)
    report["pearson"] = {
        "shape": list(pearson_corr.shape),
        "high_correlation_pairs": high_pearson,
    }
    logger.info("High Pearson correlation pairs (|r| > 0.90): %d",
                len(high_pearson))

    # ── 2. Spearman correlation ──────────────────────────────
    spearman_corr = df[feature_cols].corr(method="spearman")
    high_spearman = _find_high_correlations(spearman_corr, threshold=0.90)
    report["spearman"] = {
        "shape": list(spearman_corr.shape),
        "high_correlation_pairs": high_spearman,
    }
    logger.info("High Spearman correlation pairs (|ρ| > 0.90): %d",
                len(high_spearman))

    # ── 3. VIF ───────────────────────────────────────────────
    vif_results = _compute_vif(df, feature_cols)
    report["vif"] = vif_results
    high_vif = [r for r in vif_results if r["vif"] > 10]
    logger.info("Features with VIF > 10: %d", len(high_vif))

    # ── 4. Redundant column removal decisions ────────────────
    cols_to_drop: List[str] = []

    # 4a. Redundant unit columns (bu/acre duplicates of t/ha targets)
    for col in cfg.REDUNDANT_TARGET_COLS:
        if col in df.columns:
            cols_to_drop.append(col)
            log_decision(
                step="multicollinearity",
                decision=f"Drop redundant unit column {col}",
                reason="Perfect linear transform of t/ha target; "
                       "provides no additional information",
            )

    # 4b. Perfectly correlated feature pairs (r ≈ 1.0)
    #     Only drop if one is a strict subset / transform of the other
    for pair in high_pearson:
        if abs(pair["correlation"]) > 0.99:
            col1, col2 = pair["feature_1"], pair["feature_2"]
            # Keep the one with broader scientific meaning
            # E.g., keep Soil_Clay_Mean over Soil_Silt_Mean if near-perfect
            if col2 not in cols_to_drop and col1 not in cols_to_drop:
                # Do NOT auto-drop scientifically meaningful pairs
                logger.info(
                    "Near-perfect correlation: %s ↔ %s (r=%.4f) — "
                    "RETAINED (both may be scientifically meaningful)",
                    col1, col2, pair["correlation"],
                )

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors="ignore")
        logger.info("Dropped %d redundant columns: %s", len(cols_to_drop), cols_to_drop)

    report["dropped_columns"] = cols_to_drop
    report["decision"] = (
        f"Dropped {len(cols_to_drop)} redundant columns. "
        "High-VIF climate/disaster features retained because they carry "
        "distinct physical information required by Paper 3 methodology."
    )

    # Save correlation matrices as CSVs for visualization
    pearson_corr.to_csv(cfg.REPORT_DIR / "pearson_correlation.csv")
    spearman_corr.to_csv(cfg.REPORT_DIR / "spearman_correlation.csv")

    save_report(report, "multicollinearity_report.json")
    return df, report


def _find_high_correlations(
    corr: pd.DataFrame, threshold: float = 0.90
) -> List[Dict[str, Any]]:
    """Find pairs with absolute correlation above threshold."""
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                pairs.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "correlation": round(float(r), 4),
                })
    return sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)


def _compute_vif(
    df: pd.DataFrame, feature_cols: List[str]
) -> List[Dict[str, Any]]:
    """Compute VIF for each feature. Handles NaN by dropping incomplete rows."""
    sub = df[feature_cols].dropna()
    if len(sub) < len(feature_cols) + 1:
        logger.warning("Too few complete rows for VIF; skipping")
        return []

    # Add intercept
    sub_with_const = sub.copy()
    sub_with_const["_intercept"] = 1.0

    results = []
    for i, col in enumerate(feature_cols):
        try:
            vif = variance_inflation_factor(
                sub_with_const.values, sub_with_const.columns.get_loc(col)
            )
            results.append({
                "feature": col,
                "vif": round(float(vif), 2) if np.isfinite(vif) else 999.0,
            })
        except Exception as e:
            logger.warning("VIF computation failed for %s: %s", col, e)
            results.append({"feature": col, "vif": -1.0})

    results.sort(key=lambda x: x["vif"], reverse=True)
    return results
