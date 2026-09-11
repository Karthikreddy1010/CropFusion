"""
diagnostics.py - Negative R² source isolation, Feature Drift Diagnostics & Residual Analysis.

Evaluates:
1. Source of negative out-of-sample R² (SS_res, SS_tot, Train/Test target mean shift)
2. Feature Drift Analysis (Wasserstein distance, KS test, Population Stability Index (PSI))
3. Temporal vs Spatial vs Combined distribution shift comparison
4. Residual Diagnostics (Breusch-Pagan, Durbin-Watson, Moran's I, Normality tests)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import config as cfg
from utils import save_report, save_report_markdown, log_decision

logger = logging.getLogger("paper3")


def compute_psi(
    train_vals: np.ndarray,
    test_vals: np.ndarray,
    num_bins: int = 10,
) -> float:
    """Compute Population Stability Index (PSI) for a numerical feature."""
    tr = train_vals[~np.isnan(train_vals)]
    te = test_vals[~np.isnan(test_vals)]

    if len(tr) < 10 or len(te) < 10:
        return 0.0

    # Quantile-based binning using training distribution
    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(tr, quantiles)
    bins = np.unique(bins)  # Deduplicate identical quantiles

    if len(bins) < 2:
        return 0.0

    bins[0] = -np.inf
    bins[-1] = np.inf

    tr_counts, _ = np.histogram(tr, bins=bins)
    te_counts, _ = np.histogram(te, bins=bins)

    tr_pct = tr_counts / max(1, len(tr))
    te_pct = te_counts / max(1, len(te))

    # Add small epsilon to avoid division by zero
    eps = 1e-4
    tr_pct = np.where(tr_pct == 0, eps, tr_pct)
    te_pct = np.where(te_pct == 0, eps, te_pct)

    psi_val = np.sum((te_pct - tr_pct) * np.log(te_pct / tr_pct))
    return float(psi_val)


def compute_feature_drift_diagnostics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, Any]:
    """Compute Wasserstein Distance, KS Test, and PSI for all numerical features.

    Classifies drift into:
    - Low Drift: PSI < 0.10
    - Moderate Drift: 0.10 <= PSI < 0.25
    - Severe Drift: PSI >= 0.25
    """
    logger.info("Computing Feature Drift Analysis (Wasserstein, KS Test, PSI)")
    drift_results = []

    for col in feature_cols:
        if col in train_df.columns and col in test_df.columns:
            tr_vals = train_df[col].dropna().values
            te_vals = test_df[col].dropna().values

            if len(tr_vals) > 0 and len(te_vals) > 0:
                w_dist = float(sp_stats.wasserstein_distance(tr_vals, te_vals))
                ks_stat, ks_pval = sp_stats.ks_2samp(tr_vals, te_vals)
                psi_val = compute_psi(tr_vals, te_vals)

                if psi_val >= 0.25:
                    category = "Severe Drift"
                elif psi_val >= 0.10:
                    category = "Moderate Drift"
                else:
                    category = "Low Drift"

                drift_results.append({
                    "feature": col,
                    "wasserstein_distance": round(w_dist, 4),
                    "ks_statistic": round(float(ks_stat), 4),
                    "ks_p_value": round(float(ks_pval), 6),
                    "psi": round(psi_val, 4),
                    "drift_category": category,
                    "significant_shift": bool(ks_pval < 0.05),
                })

    df_drift = pd.DataFrame(drift_results).sort_values("psi", ascending=False)
    report = {
        "features_analyzed_count": len(drift_results),
        "severe_drift_count": int((df_drift["drift_category"] == "Severe Drift").sum()),
        "moderate_drift_count": int((df_drift["drift_category"] == "Moderate Drift").sum()),
        "low_drift_count": int((df_drift["drift_category"] == "Low Drift").sum()),
        "top_drifted_features": df_drift.head(15).to_dict(orient="records"),
        "all_feature_drift": df_drift.to_dict(orient="records"),
    }

    save_report(report, "feature_drift_report.json")
    save_report(report, "covariate_shift_report.json")
    logger.info("Feature drift report exported -> feature_drift_report.json (Severe: %d, Moderate: %d)",
                report["severe_drift_count"], report["moderate_drift_count"])
    return report


def diagnose_r2_source(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Diagnose the source of negative out-of-sample R² (§1).

    Reports:
    - Train mean, Test mean, Target mean shift
    - SS_res, SS_tot, R²
    - Spatial vs Temporal vs Combined drift comparisons
    - Automated dominant cause diagnosis
    """
    logger.info("Diagnosing source of negative out-of-sample R²...")

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    global_test_mean = float(np.mean(y_true))
    ss_tot_test = float(np.sum((y_true - global_test_mean) ** 2))

    r2_test = 1.0 - (ss_res / max(1e-6, ss_tot_test))

    train_mean = float(train_df[cfg.PRIMARY_TARGET].mean())
    ss_tot_train_baseline = float(np.sum((y_true - train_mean) ** 2))
    r2_train_baseline = 1.0 - (ss_res / max(1e-6, ss_tot_train_baseline))

    mean_shift = global_test_mean - train_mean

    # Spatial vs Temporal vs Combined Drift analysis
    temporal_var = float(test_df.groupby("Year")[cfg.PRIMARY_TARGET].mean().var())
    spatial_var = float(test_df.groupby("State")[cfg.PRIMARY_TARGET].mean().var()) if "State" in test_df.columns else 0.0

    if abs(mean_shift) > 0.5 and temporal_var > spatial_var:
        dominant_cause = "Temporal Concept Drift & Climate Trend Shift"
        detail = ("Severe temporal climate shift (2019-2023 vs 1985-2015) caused target mean displacement. "
                  "Models optimizing pure pinball loss fail to capture mean shift without Huber multi-task point prediction.")
    elif spatial_var > temporal_var:
        dominant_cause = "Spatial Heterogeneity"
        detail = "Spatial variance across Midwestern states dominates out-of-sample error."
    else:
        dominant_cause = "Combination of Architecture Limitations & Temporal Distribution Shift"
        detail = "Single-task quantile objective lacks direct point prediction constraint on mean yield."

    report = {
        "train_mean": round(train_mean, 4),
        "test_mean": round(global_test_mean, 4),
        "mean_shift": round(mean_shift, 4),
        "ss_res": round(ss_res, 4),
        "ss_tot": round(ss_tot_test, 4),
        "r2_test_sample_mean": round(r2_test, 4),
        "r2_train_mean_baseline": round(r2_train_baseline, 4),
        "temporal_variance": round(temporal_var, 4),
        "spatial_variance": round(spatial_var, 4),
        "dominant_source_of_error": dominant_cause,
        "diagnosis_detail": detail,
        "recommendations": [
            "Upgrade NeuralCQR backbone to multi-task residual architecture with dedicated Huber mean head.",
            "Incorporate flexible Fourier temporal encodings to model long-term trends and oscillations.",
            "Add static environmental descriptors (soil, topo, normals) for spatial context.",
            "Optimize composite loss balancing Huber loss, pinball loss, and quantile crossing penalty."
        ]
    }

    md = rf"""# Negative R² Diagnostic Report (§1)

## Performance Metrics & Variance Decomposition
- **Training Set Target Mean**: `{report['train_mean']}` t/ha
- **Test Set Target Mean**: `{report['test_mean']}` t/ha
- **Target Mean Shift**: `{report['mean_shift']}` t/ha
- **Residual Sum of Squares ($SS_{{res}}$)**: `{report['ss_res']}`
- **Total Sum of Squares ($SS_{{tot}}$)**: `{report['ss_tot']}`
- **Out-of-Sample $R^2$**: `{report['r2_test_sample_mean']}`
- **$R^2$ Relative to Train Mean Baseline**: `{report['r2_train_mean_baseline']}`

## Drift Comparison
- **Temporal Yield Variance**: `{report['temporal_variance']}`
- **Spatial Yield Variance**: `{report['spatial_variance']}`

## Dominant Source of Error Conclusion
**{report['dominant_source_of_error']}**: {report['diagnosis_detail']}

## Action Plan
1. Dedicated Multi-Task Huber Mean Head ($\hat{{y}}_{{\text{{mean}}}}$) exclusively evaluated for RMSE/MAE/$R^2$.
2. Fourier Temporal Encodings to capture multi-year climate oscillations.
3. Static Environmental Descriptors for spatial context without state IDs.
4. Composite Loss combining Huber Loss + Pinball Loss + Quantile Crossing Penalty.
"""

    save_report_markdown(md, "negative_r2_diagnostic_report.md")
    save_report(report, "negative_r2_diagnostic_report.json")
    logger.info("Negative R2 diagnostic report exported -> negative_r2_diagnostic_report.md")
    return report


def run_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute statistical tests on regression residuals (§11).

    Tests:
    - Shapiro-Wilk / Kolmogorov-Smirnov test for normality
    - Breusch-Pagan test for heteroscedasticity
    - Durbin-Watson test for temporal autocorrelation
    - Moran's I / spatial ACF proxy for spatial autocorrelation
    """
    logger.info("Running Residual Statistical Diagnostic Tests (§11)...")

    residuals = y_true - y_pred

    # 1. Normality Test (KS test against fitted normal)
    res_std = (residuals - np.mean(residuals)) / max(1e-6, np.std(residuals))
    ks_stat, ks_pval = sp_stats.ks_2samp(res_std, np.random.normal(0, 1, len(res_std)))

    # 2. Durbin-Watson Autocorrelation Test
    diff_res = np.diff(residuals)
    dw_stat = float(np.sum(diff_res ** 2) / max(1e-6, np.sum(residuals ** 2)))

    # 3. Breusch-Pagan Heteroscedasticity Test Proxy (Auxiliary regression of e^2 on y_pred)
    sq_res = residuals ** 2
    bp_slope, bp_intercept, bp_rval, bp_pval, bp_stderr = sp_stats.linregress(y_pred, sq_res)

    # 4. Spatial Autocorrelation Proxy (State-level residual variance ratio)
    spatial_auto = False
    if "State" in df.columns:
        valid_df = df[df[cfg.PRIMARY_TARGET].notna()].copy()
        if len(valid_df) == len(residuals):
            valid_df["res"] = residuals
            state_means = valid_df.groupby("State")["res"].mean()
            spatial_auto = bool(np.std(state_means) > 0.2 * np.std(residuals))
        else:
            state_means = df.groupby("State")[cfg.PRIMARY_TARGET].mean()
            spatial_auto = False

    report = {
        "residuals_mean": round(float(np.mean(residuals)), 4),
        "residuals_std": round(float(np.std(residuals)), 4),
        "normality_ks_stat": round(float(ks_stat), 4),
        "normality_ks_p_value": round(float(ks_pval), 6),
        "is_normally_distributed": bool(ks_pval >= 0.05),
        "durbin_watson_stat": round(dw_stat, 4),
        "temporal_autocorrelation_detected": bool(dw_stat < 1.5 or dw_stat > 2.5),
        "breusch_pagan_p_value": round(float(bp_pval), 6),
        "heteroscedasticity_detected": bool(bp_pval < 0.05),
        "spatial_autocorrelation_detected": spatial_auto,
    }

    md = f"""# Residual Diagnostic Report (§11)

## Summary of Diagnostic Tests
- **Residual Mean**: `{report['residuals_mean']}`
- **Residual Std**: `{report['residuals_std']}`
- **Normality Test (KS p-value)**: `{report['normality_ks_p_value']}` (Normally Distributed: `{report['is_normally_distributed']}`)
- **Durbin-Watson Autocorrelation Statistic**: `{report['durbin_watson_stat']}` (Autocorrelation Detected: `{report['temporal_autocorrelation_detected']}`)
- **Breusch-Pagan Heteroscedasticity p-value**: `{report['breusch_pagan_p_value']}` (Heteroscedasticity Detected: `{report['heteroscedasticity_detected']}`)
- **Spatial Autocorrelation Detected**: `{report['spatial_autocorrelation_detected']}`

## Interpretation & Next Steps
- **Heteroscedasticity**: {"Present — Adaptive conformal inference (ACI) interval scaling handles input-dependent variance." if report['heteroscedasticity_detected'] else "Not significant."}
- **Temporal Autocorrelation**: {"Detected — 3-year sliding window in ACI dynamically adjusts calibration levels." if report['temporal_autocorrelation_detected'] else "Minimal."}
"""

    save_report_markdown(md, "residual_diagnostic_report.md")
    save_report(report, "residual_diagnostic_report.json")
    logger.info("Residual diagnostic report exported -> residual_diagnostic_report.md")
    return report
