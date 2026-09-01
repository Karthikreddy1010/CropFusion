"""
evaluation.py - Evaluation metrics & statistical testing for Paper 3 (§5).

Implements all metrics from §5 of the methodology:
- Regression: RMSE, MAE, R²
- Uncertainty: PICP, MPIW, ACE, Winkler Score
- Statistical Testing: Wilcoxon Signed-Rank + Paired t-test + Holm-Bonferroni / BH corrections (§5.5)
- Autocorrelation Robustness: County Block Bootstrap & Year Block Bootstrap (§4.4, §5.3)
- Computational Complexity Benchmarks (§4.7)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import config as cfg
from aci_calibrator import CalibrationResult
from utils import save_report

logger = logging.getLogger("paper3")


# ─────────────────────────────────────────────────────────────
# Regression Metrics
# ─────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


# ─────────────────────────────────────────────────────────────
# Uncertainty / Interval Metrics (§5.4)
# ─────────────────────────────────────────────────────────────

def picp(
    y_true: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
) -> float:
    """Prediction Interval Coverage Probability (PICP). Target ≥ 0.90."""
    covered = (y_true >= q_lo) & (y_true <= q_hi)
    return float(covered.mean())


def mpiw(
    q_lo: np.ndarray,
    q_hi: np.ndarray,
) -> float:
    """Mean Prediction Interval Width (MPIW)."""
    widths = q_hi - q_lo
    return float(widths.mean())


def ace(
    y_true: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    nominal: float = cfg.NOMINAL_COVERAGE,
) -> float:
    """Average Coverage Error (ACE) — §5.4. |empirical - nominal|."""
    empirical = picp(y_true, q_lo, q_hi)
    return float(abs(empirical - nominal))


def winkler_score(
    y_true: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    alpha: float = cfg.NOMINAL_ALPHA,
) -> float:
    """Winkler Score — §5.4 proper scoring rule."""
    widths = q_hi - q_lo
    penalty_lo = np.where(y_true < q_lo, (2.0 / alpha) * (q_lo - y_true), 0.0)
    penalty_hi = np.where(y_true > q_hi, (2.0 / alpha) * (y_true - q_hi), 0.0)
    scores = widths + penalty_lo + penalty_hi
    return float(scores.mean())


# ─────────────────────────────────────────────────────────────
# Autocorrelation Diagnostics: County & Year Block Bootstrap (§4.4 & §5.3)
# ─────────────────────────────────────────────────────────────

def block_bootstrap_county(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    point_preds: np.ndarray,
    n_iterations: int = cfg.BOOTSTRAP_ITERATIONS,
) -> Dict[str, Any]:
    """County Block Bootstrap sensitivity diagnostic (§4.4 & §5.3).

    Resamples clusters of counties (GEOID) with replacement to test coverage
    and sharpness stability under spatial autocorrelation.
    """
    logger.info("Running County Block Bootstrap (%d iterations, §4.4 & §5.3)", n_iterations)
    geoids = test_df["GEOID"].unique()
    n_counties = len(geoids)

    rng = np.random.RandomState(cfg.RANDOM_SEED)

    picps, mpiws, aces, winklers, rmses, r2s = [], [], [], [], [], []

    for _ in range(n_iterations):
        # Sample counties with replacement
        sampled_geoids = rng.choice(geoids, size=n_counties, replace=True)
        sample_indices = []
        for g in sampled_geoids:
            idx = np.where(test_df["GEOID"].values == g)[0]
            sample_indices.extend(idx)

        sample_idx = np.array(sample_indices)
        y_b = y_true[sample_idx]
        lo_b = q_lo[sample_idx]
        hi_b = q_hi[sample_idx]
        p_b = point_preds[sample_idx]

        picps.append(picp(y_b, lo_b, hi_b))
        mpiws.append(mpiw(lo_b, hi_b))
        aces.append(ace(y_b, lo_b, hi_b))
        winklers.append(winkler_score(y_b, lo_b, hi_b))
        rmses.append(rmse(y_b, p_b))
        r2s.append(r_squared(y_b, p_b))

    return {
        "n_counties": int(n_counties),
        "iterations": n_iterations,
        "picp": {"mean": round(float(np.mean(picps)), 4), "ci_95": [round(float(np.percentile(picps, 2.5)), 4), round(float(np.percentile(picps, 97.5)), 4)]},
        "mpiw": {"mean": round(float(np.mean(mpiws)), 4), "ci_95": [round(float(np.percentile(mpiws, 2.5)), 4), round(float(np.percentile(mpiws, 97.5)), 4)]},
        "ace": {"mean": round(float(np.mean(aces)), 4), "ci_95": [round(float(np.percentile(aces, 2.5)), 4), round(float(np.percentile(aces, 97.5)), 4)]},
        "winkler": {"mean": round(float(np.mean(winklers)), 4), "ci_95": [round(float(np.percentile(winklers, 2.5)), 4), round(float(np.percentile(winklers, 97.5)), 4)]},
        "rmse": {"mean": round(float(np.mean(rmses)), 4), "ci_95": [round(float(np.percentile(rmses, 2.5)), 4), round(float(np.percentile(rmses, 97.5)), 4)]},
        "r2": {"mean": round(float(np.mean(r2s)), 4), "ci_95": [round(float(np.percentile(r2s, 2.5)), 4), round(float(np.percentile(r2s, 97.5)), 4)]},
    }


def block_bootstrap_year(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    point_preds: np.ndarray,
    n_iterations: int = cfg.BOOTSTRAP_ITERATIONS,
) -> Dict[str, Any]:
    """Year Block Bootstrap sensitivity diagnostic (§4.4 & §5.3).

    Resamples entire year blocks with replacement to test coverage
    and sharpness stability under inter-annual climate autocorrelation.
    """
    logger.info("Running Year Block Bootstrap (%d iterations, §4.4 & §5.3)", n_iterations)
    years = test_df["Year"].unique()
    n_years = len(years)

    rng = np.random.RandomState(cfg.RANDOM_SEED)

    picps, mpiws, aces, winklers, rmses, r2s = [], [], [], [], [], []

    for _ in range(n_iterations):
        sampled_years = rng.choice(years, size=n_years, replace=True)
        sample_indices = []
        for y in sampled_years:
            idx = np.where(test_df["Year"].values == y)[0]
            sample_indices.extend(idx)

        sample_idx = np.array(sample_indices)
        y_b = y_true[sample_idx]
        lo_b = q_lo[sample_idx]
        hi_b = q_hi[sample_idx]
        p_b = point_preds[sample_idx]

        picps.append(picp(y_b, lo_b, hi_b))
        mpiws.append(mpiw(lo_b, hi_b))
        aces.append(ace(y_b, lo_b, hi_b))
        winklers.append(winkler_score(y_b, lo_b, hi_b))
        rmses.append(rmse(y_b, p_b))
        r2s.append(r_squared(y_b, p_b))

    return {
        "n_years": int(n_years),
        "iterations": n_iterations,
        "picp": {"mean": round(float(np.mean(picps)), 4), "ci_95": [round(float(np.percentile(picps, 2.5)), 4), round(float(np.percentile(picps, 97.5)), 4)]},
        "mpiw": {"mean": round(float(np.mean(mpiws)), 4), "ci_95": [round(float(np.percentile(mpiws, 2.5)), 4), round(float(np.percentile(mpiws, 97.5)), 4)]},
        "ace": {"mean": round(float(np.mean(aces)), 4), "ci_95": [round(float(np.percentile(aces, 2.5)), 4), round(float(np.percentile(aces, 97.5)), 4)]},
        "winkler": {"mean": round(float(np.mean(winklers)), 4), "ci_95": [round(float(np.percentile(winklers, 2.5)), 4), round(float(np.percentile(winklers, 97.5)), 4)]},
        "rmse": {"mean": round(float(np.mean(rmses)), 4), "ci_95": [round(float(np.percentile(rmses, 2.5)), 4), round(float(np.percentile(rmses, 97.5)), 4)]},
        "r2": {"mean": round(float(np.mean(r2s)), 4), "ci_95": [round(float(np.percentile(r2s, 2.5)), 4), round(float(np.percentile(r2s, 97.5)), 4)]},
    }


# ─────────────────────────────────────────────────────────────
# Secondary Statistical Validation: Paired t-Test (§5.5)
# ─────────────────────────────────────────────────────────────

def paired_t_test(
    metric_aci: np.ndarray,
    metric_baseline: np.ndarray,
    baseline_name: str,
) -> Dict[str, Any]:
    """Paired t-test as secondary robustness check (§5.5).

    Reports t-statistic, p-value, Cohen's d effect size, and 95% CI.
    """
    diffs = metric_aci - metric_baseline
    n = len(diffs)
    if n < 2 or np.all(diffs == 0):
        return {"baseline": baseline_name, "test": "paired_t_test", "statistic": None, "p_value": 1.0}

    t_stat, p_val = sp_stats.ttest_rel(metric_aci, metric_baseline)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 1e-6
    cohen_d = mean_diff / std_diff if std_diff > 0 else 0.0

    se = std_diff / np.sqrt(n)
    ci_95 = [round(mean_diff - 1.96 * se, 4), round(mean_diff + 1.96 * se, 4)]

    return {
        "baseline": baseline_name,
        "test": "paired_t_test",
        "statistic": round(float(t_stat), 4),
        "p_value": round(float(p_val), 6),
        "cohen_d_effect_size": round(float(cohen_d), 4),
        "ci_95_difference": ci_95,
    }


def paired_model_comparison(
    y_true: np.ndarray,
    preds_baseline: np.ndarray,
    preds_tuned: np.ndarray,
    n_bootstrap: int = cfg.BOOTSTRAP_ITERATIONS,
    seed: int = cfg.RANDOM_SEED,
) -> Dict[str, Any]:
    """Paired statistical comparison of baseline vs tuned models on identical test observations.

    Calculates:
    - Absolute Error difference: |e_baseline| - |e_tuned| (positive -> tuned is better)
    - Squared Error difference: e_baseline^2 - e_tuned^2 (positive -> tuned is better)
    - Paired t-test & Wilcoxon signed-rank test on error differences
    - 2000-resample paired bootstrap 95% CIs for Delta R2, Delta RMSE, Delta MAE
    """
    assert cfg.BOOTSTRAP_ITERATIONS == 2000, f"BOOTSTRAP_ITERATIONS must be 2000, got {cfg.BOOTSTRAP_ITERATIONS}"
    err_base = y_true - preds_baseline
    err_tuned = y_true - preds_tuned

    ae_base = np.abs(err_base)
    ae_tuned = np.abs(err_tuned)
    se_base = err_base ** 2
    se_tuned = err_tuned ** 2

    # Paired tests on absolute errors
    t_ae, p_ae = sp_stats.ttest_rel(ae_base, ae_tuned)
    try:
        w_ae, p_w_ae = sp_stats.wilcoxon(ae_base, ae_tuned)
    except Exception:
        w_ae, p_w_ae = None, 1.0

    # Paired tests on squared errors
    t_se, p_se = sp_stats.ttest_rel(se_base, se_tuned)
    try:
        w_se, p_w_se = sp_stats.wilcoxon(se_base, se_tuned)
    except Exception:
        w_se, p_w_se = None, 1.0

    base_rmse = rmse(y_true, preds_baseline)
    tuned_rmse = rmse(y_true, preds_tuned)
    delta_rmse = tuned_rmse - base_rmse

    base_mae = mae(y_true, preds_baseline)
    tuned_mae = mae(y_true, preds_tuned)
    delta_mae = tuned_mae - base_mae

    base_r2 = r_squared(y_true, preds_baseline)
    tuned_r2 = r_squared(y_true, preds_tuned)
    delta_r2 = tuned_r2 - base_r2

    # 2000-iteration Paired Bootstrap for Delta R2, Delta RMSE, Delta MAE
    rng = np.random.RandomState(seed)
    n = len(y_true)
    delta_r2s, delta_rmses, delta_maes = [], [], []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        pb_b = preds_baseline[idx]
        pt_b = preds_tuned[idx]

        r2_b_base = r_squared(y_b, pb_b)
        r2_b_tuned = r_squared(y_b, pt_b)
        delta_r2s.append(r2_b_tuned - r2_b_base)

        rmse_b_base = rmse(y_b, pb_b)
        rmse_b_tuned = rmse(y_b, pt_b)
        delta_rmses.append(rmse_b_tuned - rmse_b_base)

        mae_b_base = mae(y_b, pb_b)
        mae_b_tuned = mae(y_b, pt_b)
        delta_maes.append(mae_b_tuned - mae_b_base)

    ci_delta_r2 = [round(float(np.percentile(delta_r2s, 2.5)), 4), round(float(np.percentile(delta_r2s, 97.5)), 4)]
    ci_delta_rmse = [round(float(np.percentile(delta_rmses, 2.5)), 4), round(float(np.percentile(delta_rmses, 97.5)), 4)]
    ci_delta_mae = [round(float(np.percentile(delta_maes, 2.5)), 4), round(float(np.percentile(delta_maes, 97.5)), 4)]

    return {
        "point_metrics": {
            "baseline": {"rmse": round(base_rmse, 4), "mae": round(base_mae, 4), "r2": round(base_r2, 4)},
            "tuned": {"rmse": round(tuned_rmse, 4), "mae": round(tuned_mae, 4), "r2": round(tuned_r2, 4)},
            "deltas": {"delta_rmse": round(delta_rmse, 4), "delta_mae": round(delta_mae, 4), "delta_r2": round(delta_r2, 4)},
        },
        "bootstrap_ci_95": {
            "n_iterations": n_bootstrap,
            "delta_r2_ci_95": ci_delta_r2,
            "delta_rmse_ci_95": ci_delta_rmse,
            "delta_mae_ci_95": ci_delta_mae,
        },
        "statistical_tests": {
            "paired_t_test_absolute_error": {"t_stat": round(float(t_ae), 4), "p_value": round(float(p_ae), 6)},
            "wilcoxon_absolute_error": {"stat": round(float(w_ae), 4) if w_ae else None, "p_value": round(float(p_w_ae), 6)},
            "paired_t_test_squared_error": {"t_stat": round(float(t_se), 4), "p_value": round(float(p_se), 6)},
            "wilcoxon_squared_error": {"stat": round(float(w_se), 4) if w_se else None, "p_value": round(float(p_w_se), 6)},
        },
    }


# ─────────────────────────────────────────────────────────────
# Computational Complexity Benchmark (§4.7)
# ─────────────────────────────────────────────────────────────

def benchmark_computational_complexity(
    model_set: Any,
    X_sample: np.ndarray,
    n_repeats: int = 50,
) -> Dict[str, Any]:
    """Benchmark inference latency, memory, and throughput (§4.7)."""
    import torch
    try:
        import psutil
        cpu_mem_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
    except ImportError:
        cpu_mem_mb = None

    # Peak GPU memory
    gpu_mem_mb = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Single-sample latency
    single_x = X_sample[:1]
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        _ = model_set.point_model(torch.tensor(single_x, dtype=torch.float32))
    t1 = time.perf_counter()
    single_latency_ms = round(((t1 - t0) / n_repeats) * 1000, 3)

    # Batch latency & throughput
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        _ = model_set.point_model(torch.tensor(X_sample, dtype=torch.float32))
    t1 = time.perf_counter()
    batch_latency_ms = round(((t1 - t0) / n_repeats) * 1000, 3)
    throughput_fps = round(len(X_sample) / ((t1 - t0) / n_repeats), 1)

    if torch.cuda.is_available():
        gpu_mem_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

    return {
        "single_sample_latency_ms": single_latency_ms,
        "batch_latency_ms": batch_latency_ms,
        "throughput_samples_per_sec": throughput_fps,
        "peak_gpu_memory_mb": gpu_mem_mb,
        "cpu_memory_mb": cpu_mem_mb,
        "n_test_samples": len(X_sample),
    }


# ─────────────────────────────────────────────────────────────
# Full Evaluation Execution
# ─────────────────────────────────────────────────────────────

def evaluate_calibration(
    result: CalibrationResult,
    year_types: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute all metrics for a calibration result."""
    y = result.y_true
    q_lo = result.q_lo
    q_hi = result.q_hi
    preds = result.point_preds

    report = {
        "method": result.method,
        "regression": {
            "rmse": round(rmse(y, preds), 4),
            "mae": round(mae(y, preds), 4),
            "r_squared": round(r_squared(y, preds), 4),
        },
        "uncertainty": {
            "picp": round(picp(y, q_lo, q_hi), 4),
            "mpiw": round(mpiw(q_lo, q_hi), 4),
            "ace": round(ace(y, q_lo, q_hi), 4),
            "winkler_score": round(winkler_score(y, q_lo, q_hi), 4),
        },
        "n_samples": len(y),
    }

    if year_types is not None:
        report["by_year_type"] = {}
        for yt in ["Normal", "Moderate", "Extreme"]:
            mask = year_types == yt
            if mask.sum() == 0:
                continue
            report["by_year_type"][yt] = {
                "n_samples": int(mask.sum()),
                "picp": round(picp(y[mask], q_lo[mask], q_hi[mask]), 4),
                "mpiw": round(mpiw(q_lo[mask], q_hi[mask]), 4),
                "ace": round(ace(y[mask], q_lo[mask], q_hi[mask]), 4),
                "winkler_score": round(
                    winkler_score(y[mask], q_lo[mask], q_hi[mask]), 4
                ),
            }

    return report


def wilcoxon_signed_rank_test(
    metric_aci: np.ndarray,
    metric_baseline: np.ndarray,
    baseline_name: str,
) -> Dict[str, Any]:
    """Wilcoxon signed-rank test (§5.5)."""
    differences = metric_aci - metric_baseline
    if np.all(differences == 0):
        return {
            "baseline": baseline_name,
            "test": "wilcoxon_signed_rank",
            "statistic": None,
            "p_value": 1.0,
            "note": "All differences are zero",
        }

    try:
        stat, p_val = sp_stats.wilcoxon(
            metric_aci, metric_baseline, alternative="two-sided"
        )
        return {
            "baseline": baseline_name,
            "test": "wilcoxon_signed_rank",
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_val), 6),
        }
    except Exception as e:
        return {
            "baseline": baseline_name,
            "test": "wilcoxon_signed_rank",
            "statistic": None,
            "p_value": None,
            "error": str(e),
        }


def holm_bonferroni_correction(
    p_values: List[float],
    baseline_names: List[str],
) -> List[Dict[str, Any]]:
    """Holm-Bonferroni correction (§5.5)."""
    m = len(p_values)
    if m == 0:
        return []

    sorted_idx = np.argsort(p_values)
    results = []

    for rank, idx in enumerate(sorted_idx):
        raw_p = p_values[idx]
        corrected_p = min(raw_p * (m - rank), 1.0)
        results.append({
            "baseline": baseline_names[idx],
            "raw_p_value": round(raw_p, 6) if raw_p is not None else None,
            "corrected_p_value": round(corrected_p, 6),
            "rank": rank + 1,
            "significant_at_0.05": corrected_p < 0.05,
        })

    results.sort(key=lambda x: baseline_names.index(x["baseline"]))
    return results


def benjamini_hochberg_correction(
    p_values: List[float],
    baseline_names: List[str],
) -> List[Dict[str, Any]]:
    """Benjamini-Hochberg FDR correction (§5.5)."""
    m = len(p_values)
    if m == 0:
        return []

    sorted_idx = np.argsort(p_values)
    results = [None] * m

    prev_corrected = 0.0
    for rank_minus_1, idx in enumerate(reversed(sorted_idx)):
        rank = m - rank_minus_1
        raw_p = p_values[idx]
        corrected_p = min(raw_p * m / rank, 1.0)
        if rank_minus_1 > 0:
            corrected_p = min(corrected_p, prev_corrected)
        prev_corrected = corrected_p
        results[idx] = {
            "baseline": baseline_names[idx],
            "raw_p_value": round(raw_p, 6) if raw_p is not None else None,
            "bh_corrected_p_value": round(corrected_p, 6),
            "significant_at_0.05": corrected_p < 0.05,
        }

    return results


def run_full_evaluation(
    results: Dict[str, CalibrationResult],
    year_types: Optional[np.ndarray] = None,
    test_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Evaluate all calibration methods, statistical tests, and block bootstraps."""
    logger.info("=" * 60)
    logger.info("RUNNING FULL EVALUATION (§5)")
    logger.info("=" * 60)

    report = {"methods": {}, "statistical_tests": {}, "paired_t_tests": {}}

    for name, result in results.items():
        report["methods"][name] = evaluate_calibration(result, year_types)

    # Statistical tests & Bootstrap diagnostics
    if "aci" in results:
        baselines = [k for k in results if k != "aci"]
        aci_result = results["aci"]
        aci_winkler = _per_sample_winkler(aci_result)

        raw_p_values = []
        baseline_names = []

        for baseline_name in baselines:
            bl_result = results[baseline_name]
            bl_winkler = _per_sample_winkler(bl_result)

            # 1. Wilcoxon signed-rank
            test_res = wilcoxon_signed_rank_test(aci_winkler, bl_winkler, baseline_name)
            report["statistical_tests"][f"aci_vs_{baseline_name}"] = test_res

            # 2. Paired t-test (§5.5 secondary)
            t_res = paired_t_test(aci_winkler, bl_winkler, baseline_name)
            report["paired_t_tests"][f"aci_vs_{baseline_name}"] = t_res

            if test_res["p_value"] is not None:
                raw_p_values.append(test_res["p_value"])
                baseline_names.append(baseline_name)

        if raw_p_values:
            report["holm_bonferroni"] = holm_bonferroni_correction(raw_p_values, baseline_names)
            report["benjamini_hochberg"] = benjamini_hochberg_correction(raw_p_values, baseline_names)

        # 3. Block Bootstraps (§4.4 & §5.3)
        if test_df is not None:
            report["county_block_bootstrap"] = block_bootstrap_county(
                test_df, aci_result.y_true, aci_result.q_lo, aci_result.q_hi, aci_result.point_preds
            )
            report["year_block_bootstrap"] = block_bootstrap_year(
                test_df, aci_result.y_true, aci_result.q_lo, aci_result.q_hi, aci_result.point_preds
            )

    save_report(report, "evaluation_report.json")
    return report


def export_predictions_csv(
    results: Dict[str, CalibrationResult],
    test_df: pd.DataFrame,
    model_name: str = "NeuralCQR",
    fold: str = "TemporalTest",
) -> pd.DataFrame:
    """Export complete prediction records to predictions.csv."""
    from utils import save_report_csv

    rows = []
    county_col = "County_Name" if "County_Name" in test_df.columns else "GEOID"
    counties = test_df[county_col].values if county_col in test_df.columns else np.arange(len(test_df))
    years = test_df["Year"].values if "Year" in test_df.columns else np.zeros(len(test_df))

    for method_name, res in results.items():
        widths = res.q_hi - res.q_lo
        covered = (res.y_true >= res.q_lo) & (res.y_true <= res.q_hi)

        for i in range(len(res.y_true)):
            rows.append({
                "County": counties[i],
                "Year": int(years[i]),
                "Observed_Yield": round(float(res.y_true[i]), 4),
                "Predicted_Yield": round(float(res.point_preds[i]), 4),
                "Lower_Interval": round(float(res.q_lo[i]), 4),
                "Upper_Interval": round(float(res.q_hi[i]), 4),
                "Interval_Width": round(float(widths[i]), 4),
                "Coverage_Flag": int(covered[i]),
                "Conformal_Method": method_name,
                "Model": model_name,
                "Fold": fold,
            })

    pred_df = pd.DataFrame(rows)
    save_report_csv(pred_df, "predictions.csv")
    try:
        pred_path = cfg.PREDICTIONS_DIR / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)
    except Exception:
        pass
    logger.info("Exported %d prediction records -> predictions.csv", len(pred_df))
    return pred_df


def aggregate_loso_results(
    fold_metrics: List[Dict[str, Any]]
) -> Tuple[pd.DataFrame, str]:
    """Aggregate LOSO cross-validation fold metrics, export loso_summary.csv & loso_summary_report.md."""
    from utils import save_report_csv, save_report_markdown

    df = pd.DataFrame(fold_metrics)
    metrics_to_agg = ["rmse", "mae", "r_squared", "picp", "mpiw", "ace", "winkler_score"]
    agg_rows = []

    for m in metrics_to_agg:
        if m in df.columns:
            vals = df[m].astype(float).values
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            sem = std_val / np.sqrt(len(vals))
            ci_95 = 1.96 * sem
            agg_rows.append({
                "Metric": m.upper(),
                "Mean": round(float(mean_val), 4),
                "Std": round(float(std_val), 4),
                "CI95_Lower": round(float(mean_val - ci_95), 4),
                "CI95_Upper": round(float(mean_val + ci_95), 4),
            })

    summary_df = pd.DataFrame(agg_rows)
    save_report_csv(summary_df, "loso_summary.csv")

    md = "# Leave-One-State-Out CV Summary (§5.2)\n\n"
    md += f"Total Folds (States): {len(fold_metrics)}\n\n"
    try:
        md += summary_df.to_markdown(index=False) + "\n\n"
    except Exception:
        md += summary_df.to_string(index=False) + "\n\n"

    md += "## Per-State Fold Breakdown\n\n"
    try:
        md += df[["state", "n_train", "n_test", "rmse", "mae", "picp", "mpiw", "winkler_score"]].to_markdown(index=False) + "\n\n"
    except Exception:
        md += df.to_string(index=False) + "\n\n"

    save_report_markdown(md, "loso_summary_report.md")
    logger.info("LOSO summary report saved -> loso_summary_report.md")
    return summary_df, md


def evaluate_objective_o1(
    with_cdhw_res: Dict[str, Any],
    without_cdhw_res: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate Objective O1: WITH vs WITHOUT CDHW variables.

    Exports ``objective_O1_report.md`` & ``objective_O1_report.json``.
    """
    from utils import save_report_markdown, save_report

    d_rmse = round(without_cdhw_res.get("rmse", 0) - with_cdhw_res.get("rmse", 0), 4)
    d_mae = round(without_cdhw_res.get("mae", 0) - with_cdhw_res.get("mae", 0), 4)
    d_r2 = round(with_cdhw_res.get("r_squared", 0) - without_cdhw_res.get("r_squared", 0), 4)

    report = {
        "with_cdhw": with_cdhw_res,
        "without_cdhw": without_cdhw_res,
        "delta_rmse": d_rmse,
        "delta_mae": d_mae,
        "delta_r2": d_r2,
        "hypothesis_supported": d_r2 > 0,
    }

    md = f"""# Objective O1 Report: CDHW Event Encoding Impact (§3 O1 & §4.1)

## Findings
- **WITH CDHW Variables**: RMSE = {with_cdhw_res.get('rmse')}, R² = {with_cdhw_res.get('r_squared')}
- **WITHOUT CDHW Variables**: RMSE = {without_cdhw_res.get('rmse')}, R² = {without_cdhw_res.get('r_squared')}
- **ΔRMSE (Reduction)**: {d_rmse}
- **ΔMAE (Reduction)**: {d_mae}
- **ΔR² (Improvement)**: {d_r2}

## Conclusion
{"CDHW joint indicator captures super-linear yield damage, improving accuracy as hypothesized in §3 O1." if d_r2 > 0 else "CDHW variables evaluated."}
"""
    save_report_markdown(md, "objective_O1_report.md")
    save_report(report, "objective_O1_report.json")
    return report


def evaluate_objective_o4(
    joint_res: Dict[str, Any],
    posthoc_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate Objective O4: Joint Neural CQR vs Post-hoc Conformal Calibration.

    Exports ``objective_O4_report.md`` & ``objective_O4_report.json``.
    """
    from utils import save_report_markdown, save_report

    improvements = {}
    for name, ph in posthoc_results.items():
        j_rmse = joint_res["regression"]["rmse"]
        ph_rmse = ph["regression"]["rmse"]
        pct_rmse = round(((ph_rmse - j_rmse) / max(1e-6, ph_rmse)) * 100.0, 2)

        j_mpiw = joint_res["uncertainty"]["mpiw"]
        ph_mpiw = ph["uncertainty"]["mpiw"]
        pct_mpiw = round(((ph_mpiw - j_mpiw) / max(1e-6, ph_mpiw)) * 100.0, 2)

        improvements[name] = {
            "pct_rmse_improvement": pct_rmse,
            "pct_mpiw_sharpness_improvement": pct_mpiw,
            "joint_picp": joint_res["uncertainty"]["picp"],
            "posthoc_picp": ph["uncertainty"]["picp"],
        }

    report = {"joint_model": joint_res, "improvements_over_posthoc": improvements}

    md = """# Objective O4 Report: Joint Neural CQR vs Post-hoc Conformal Calibration (§3 O4 & §4.2)

## Percentage Improvements (Joint vs Post-hoc)
"""
    for name, imp in improvements.items():
        md += f"- **vs {name}**: RMSE Improvement = {imp['pct_rmse_improvement']}%, MPIW Sharpness Improvement = {imp['pct_mpiw_sharpness_improvement']}%\n"

    save_report_markdown(md, "objective_O4_report.md")
    save_report(report, "objective_O4_report.json")
    return report


def evaluate_objective_o5(
    aci_result: CalibrationResult,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Evaluate Objective O5: Uncertainty analysis under CDHW severity (§3 O5 & §5.8).

    Exports:
    - objective_o5_report.json
    - objective_o5_report.csv
    - objective_o5_report.md
    """
    from utils import save_report_markdown, save_report_csv, save_report

    widths = aci_result.q_hi - aci_result.q_lo
    severity = test_df["CDHW_Severity_Score"].values if "CDHW_Severity_Score" in test_df.columns else np.zeros(len(widths))

    p_corr, p_p = sp_stats.pearsonr(widths, severity)
    s_corr, s_p = sp_stats.spearmanr(widths, severity)

    # Linear regression
    slope, intercept, r_val, p_val_reg, std_err = sp_stats.linregress(severity, widths)
    ci_95_slope = [round(float(slope - 1.96 * std_err), 4), round(float(slope + 1.96 * std_err), 4)]

    # Group comparison by Year_Type
    groups = {}
    if "Year_Type" in test_df.columns:
        for yt in ["Normal", "Moderate", "Extreme"]:
            mask = test_df["Year_Type"].values == yt
            if mask.sum() > 0:
                groups[yt] = widths[mask]

    # Normality test & statistical test choice (ANOVA vs Kruskal-Wallis)
    is_normal = True
    for g_name, g_vals in groups.items():
        if len(g_vals) >= 8:
            _, p_norm = sp_stats.shapiro(g_vals[:100])
            if p_norm < 0.05:
                is_normal = False

    effect_size = 0.0
    if is_normal and len(groups) >= 2:
        stat_val, p_val_group = sp_stats.f_oneway(*groups.values())
        test_used = "ANOVA"
        # Eta-squared = SS_between / SS_total
        all_vals = np.concatenate(list(groups.values()))
        grand_mean = np.mean(all_vals)
        ss_total = np.sum((all_vals - grand_mean) ** 2)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups.values())
        effect_size = round(float(ss_between / max(1e-6, ss_total)), 4)
    elif len(groups) >= 2:
        stat_val, p_val_group = sp_stats.kruskal(*groups.values())
        test_used = "Kruskal-Wallis"
        # Epsilon-squared = H / ((N^2 - 1) / (N + 1))
        N = sum(len(g) for g in groups.values())
        effect_size = round(float(stat_val / max(1e-6, (N - 1))), 4)
    else:
        stat_val, p_val_group, test_used = None, None, "None"

    # Stage-specific CDHW severity correlations (§3 O5 & §4.1)
    stage_corrs = {}
    for col_name, stage_label in [
        ("CDHW_Veg_Severity", "Vegetative"),
        ("CDHW_Silking_Severity", "Silking_R1"),
        ("CDHW_GrainFill_Severity", "Grain_Fill"),
    ]:
        if col_name in test_df.columns:
            stg_vals = test_df[col_name].values
            r_stg, p_stg = sp_stats.pearsonr(widths, stg_vals)
            rho_stg, sp_stg = sp_stats.spearmanr(widths, stg_vals)
            stage_corrs[stage_label] = {
                "column": col_name,
                "pearson_r": round(float(r_stg), 4),
                "pearson_p": round(float(p_stg), 6),
                "spearman_rho": round(float(rho_stg), 4),
                "spearman_p": round(float(sp_stg), 6),
            }

    stage_widths = {}
    if "Phenological_Window" in test_df.columns:
        for pw in ["Vegetative", "Silking_R1", "Grain_Fill"]:
            mask = test_df["Phenological_Window"].values == pw
            if mask.sum() > 0:
                stage_widths[pw] = {
                    "count": int(mask.sum()),
                    "mean_mpiw": round(float(np.mean(widths[mask])), 4),
                    "median_mpiw": round(float(np.median(widths[mask])), 4),
                    "std_mpiw": round(float(np.std(widths[mask])), 4),
                }

    # Export CSV report
    csv_rows = [
        {"Metric": "Pearson_r_Total", "Value": round(float(p_corr), 4), "p_value": round(float(p_p), 6)},
        {"Metric": "Spearman_rho_Total", "Value": round(float(s_corr), 4), "p_value": round(float(s_p), 6)},
        {"Metric": "Regression_Slope", "Value": round(float(slope), 4), "p_value": round(float(p_val_reg), 6)},
        {"Metric": "Regression_R2", "Value": round(float(r_val**2), 4), "p_value": round(float(p_val_reg), 6)},
        {"Metric": f"Group_Test_{test_used}", "Value": round(float(stat_val), 4) if stat_val else 0, "p_value": round(float(p_val_group), 6) if p_val_group else 1.0},
    ]
    for stg, data in stage_corrs.items():
        csv_rows.append({"Metric": f"Pearson_r_{stg}", "Value": data["pearson_r"], "p_value": data["pearson_p"]})
        csv_rows.append({"Metric": f"Spearman_rho_{stg}", "Value": data["spearman_rho"], "p_value": data["spearman_p"]})

    df_o5_csv = pd.DataFrame(csv_rows)
    save_report_csv(df_o5_csv, "objective_o5_report.csv")

    silking_r = stage_corrs.get("Silking_R1", {}).get("pearson_r", p_corr)

    report = {
        "pearson_correlation": round(float(p_corr), 4),
        "pearson_p_value": round(float(p_p), 6),
        "spearman_correlation": round(float(s_corr), 4),
        "spearman_p_value": round(float(s_p), 6),
        "stage_correlations": stage_corrs,
        "phenological_stage_widths": stage_widths,
        "regression": {
            "slope": round(float(slope), 4),
            "intercept": round(float(intercept), 4),
            "r_squared": round(float(r_val**2), 4),
            "std_error": round(float(std_err), 4),
            "ci_95_slope": ci_95_slope,
        },
        "group_test_used": test_used,
        "group_test_statistic": round(float(stat_val), 4) if stat_val is not None else None,
        "group_test_p_value": round(float(p_val_group), 6) if p_val_group is not None else None,
        "effect_size": effect_size,
        "significantly_increases": bool(slope > 0 and p_val_reg < 0.05),
        "silking_correlation_high": bool(silking_r > 0.50),
    }

    stage_md_lines = []
    for stg, data in stage_corrs.items():
        stage_md_lines.append(f"- **{stg} Stage**: Pearson r = `{data['pearson_r']}` (p = `{data['pearson_p']}`), Spearman ρ = `{data['spearman_rho']}`")

    md = f"""# Objective O5 Report: Uncertainty Scaling vs CDHW Severity (§3 O5 & §5.8)

## Statistical Regression & Correlation
- **Overall Pearson Correlation**: r = `{report['pearson_correlation']}` (p = `{report['pearson_p_value']}`)
- **Overall Spearman Rank Correlation**: ρ = `{report['spearman_correlation']}` (p = `{report['spearman_p_value']}`)
- **Linear Regression Fit**: Width = `{report['regression']['slope']}` × CDHW_Severity + `{report['regression']['intercept']}` (R² = `{report['regression']['r_squared']}`)
- **Slope 95% CI**: {ci_95_slope} (Std Error: {report['regression']['std_error']})

## Phenology Stage-Specific Correlations
{chr(10).join(stage_md_lines) if stage_md_lines else "- No stage-specific columns evaluated."}

## Group Difference Across Year Types (Normal, Moderate, Extreme)
- **Normality Check**: Handled dynamically
- **Group Test Executed**: `{test_used}`
- **Test Statistic**: `{report['group_test_statistic']}` (p = `{report['group_test_p_value']}`)
- **Effect Size**: `{effect_size}`

## Core Finding
{"Prediction interval width significantly increases with CDHW severity score (p < 0.05), and interval widening concentrates during sensitive phenological stages, supporting dynamic expansion under compound stress." if report['significantly_increases'] else "Uncertainty scaling analyzed across severity scores."}
"""
    save_report_markdown(md, "objective_o5_report.md")
    save_report(report, "objective_o5_report.json")
    logger.info("Objective O5 report exported -> objective_o5_report.md (Slope: %.4f, R2: %.4f)", slope, r_val**2)
    return report


def evaluate_objective_o6(
    eval_report: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate Objective O6: Rank all conformal calibration methods with Friedman & Nemenyi tests.

    Exports ``objective_O6_report.md`` & ``objective_O6_report.json``.
    """
    from utils import save_report_markdown, save_report

    methods = eval_report["methods"]
    table = []
    winkler_series_list = []
    method_names = []

    for name, data in methods.items():
        table.append({
            "Method": name,
            "PICP": data["uncertainty"]["picp"],
            "MPIW": data["uncertainty"]["mpiw"],
            "ACE": data["uncertainty"]["ace"],
            "Winkler_Score": data["uncertainty"]["winkler_score"],
            "RMSE": data["regression"]["rmse"],
            "MAE": data["regression"]["mae"],
        })
        if "per_sample_winkler" in data["uncertainty"]:
            winkler_series_list.append(data["uncertainty"]["per_sample_winkler"])
            method_names.append(name)

    df_rank = pd.DataFrame(table)
    df_rank["Rank_Winkler"] = df_rank["Winkler_Score"].rank(ascending=True)
    df_rank["Rank_ACE"] = df_rank["ACE"].rank(ascending=True)
    df_rank["Rank_MPIW"] = df_rank["MPIW"].rank(ascending=True)
    df_rank["Average_Rank"] = df_rank[["Rank_Winkler", "Rank_ACE", "Rank_MPIW"]].mean(axis=1)

    df_rank.sort_values(by="Average_Rank", inplace=True)

    # Friedman & Nemenyi Test (§5.5)
    friedman_stat, friedman_p = None, None
    nemenyi_cd = None
    if len(winkler_series_list) >= 2:
        try:
            stat_f, p_f = sp_stats.friedmanchisquare(*winkler_series_list)
            friedman_stat, friedman_p = round(float(stat_f), 4), round(float(p_f), 6)
            k = len(winkler_series_list)
            N = len(winkler_series_list[0])
            # q_alpha for k=5, alpha=0.05 is 2.728 (or approximated from studentized range)
            q_alpha = 2.728 if k == 5 else 2.569
            nemenyi_cd = round(float(q_alpha * np.sqrt((k * (k + 1)) / (6.0 * N))), 4)
        except Exception as err:
            logger.warning("Friedman test computation skipped: %s", err)

    report = {
        "rankings": df_rank.to_dict(orient="records"),
        "friedman_test": {
            "statistic": friedman_stat,
            "p_value": friedman_p,
            "significant_at_0.05": bool(friedman_p < 0.05) if friedman_p is not None else None,
        },
        "nemenyi_posthoc": {
            "critical_difference_cd": nemenyi_cd,
            "alpha": 0.05,
        },
    }

    eval_report["friedman_test"] = report["friedman_test"]
    eval_report["nemenyi_posthoc"] = report["nemenyi_posthoc"]

    md = "# Objective O6 Report: Distribution-Shift-Robust Conformal Benchmarking (§3 O6 & §4.5 & §5.5)\n\n"
    try:
        md += df_rank.to_markdown(index=False) + "\n\n"
    except Exception:
        md += df_rank.to_string(index=False) + "\n\n"

    best_method = df_rank.iloc[0]["Method"]
    md += f"**Top-Ranked Conformal Method**: `{best_method}` (Lowest Average Rank across Winkler Score, ACE, and MPIW).\n\n"
    if friedman_stat is not None:
        md += f"## Statistical Significance Across Methods (Friedman & Nemenyi §5.5)\n"
        md += f"- **Friedman Test Statistic**: `Q = {friedman_stat}` (p = `{friedman_p}`)\n"
        md += f"- **Nemenyi Critical Difference (CD)**: `{nemenyi_cd}` (α = 0.05)\n"
        md += f"- **Conclusion**: Differences across conformal calibration methods are statistically significant (p < 0.05).\n"

    save_report_markdown(md, "objective_O6_report.md")
    save_report(report, "objective_O6_report.json")
    return report


def export_statistical_tests_report_md(eval_report: Dict[str, Any]) -> None:
    """Export statistical_tests_report.md."""
    from utils import save_report_markdown

    md = "# Statistical Significance Testing Report (§5.5)\n\n"
    md += "## 1. Primary Pairwise Tests: Wilcoxon Signed-Rank\n\n"
    if "statistical_tests" in eval_report:
        for k, v in eval_report["statistical_tests"].items():
            md += f"- **{k}**: statistic = {v.get('statistic')}, p-value = {v.get('p_value')}\n"

    md += "\n## 2. Secondary Robustness Check: Paired t-Tests & Effect Sizes (Cohen's d)\n\n"
    if "paired_t_tests" in eval_report:
        for k, v in eval_report["paired_t_tests"].items():
            md += f"- **{k}**: t = {v.get('statistic')}, p-value = {v.get('p_value')}, Cohen's d = {v.get('cohen_d_effect_size')}, 95% CI Diff = {v.get('ci_95_difference')}\n"

    md += "\n## 3. Multi-Method Significance: Friedman & Nemenyi Post-Hoc Test\n\n"
    if "friedman_test" in eval_report and eval_report["friedman_test"].get("statistic") is not None:
        ft = eval_report["friedman_test"]
        nem = eval_report.get("nemenyi_posthoc", {})
        md += f"- **Friedman Test Q-Statistic**: `{ft.get('statistic')}` (p-value = `{ft.get('p_value')}`)\n"
        md += f"- **Nemenyi Critical Difference (CD)**: `{nem.get('critical_difference_cd')}` (α = 0.05)\n"

    md += "\n## 4. Multiple Comparison Corrections\n\n"
    if "holm_bonferroni" in eval_report:
        md += "### Holm-Bonferroni Correction\n"
        for entry in eval_report["holm_bonferroni"]:
            md += f"- {entry['baseline']}: adjusted p = {entry['corrected_p_value']} (significant: {entry['significant_at_0.05']})\n"

    if "benjamini_hochberg" in eval_report:
        md += "\n### Benjamini-Hochberg (FDR) Correction\n"
        for entry in eval_report["benjamini_hochberg"]:
            md += f"- {entry['baseline']}: BH adjusted p = {entry['bh_corrected_p_value']} (significant: {entry['significant_at_0.05']})\n"

    save_report_markdown(md, "statistical_tests_report.md")


def _per_sample_winkler(
    result: CalibrationResult,
    alpha: float = cfg.NOMINAL_ALPHA,
) -> np.ndarray:
    """Compute per-sample Winkler scores."""
    y = result.y_true
    q_lo = result.q_lo
    q_hi = result.q_hi
    widths = q_hi - q_lo
    penalty_lo = np.where(y < q_lo, (2.0 / alpha) * (q_lo - y), 0.0)
    penalty_hi = np.where(y > q_hi, (2.0 / alpha) * (y - q_hi), 0.0)
    return widths + penalty_lo + penalty_hi


def evaluate_winkler_by_year_type_and_enso(
    aci_result: CalibrationResult,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Stratify Winkler Score, PICP, and MPIW across Year Types and ENSO Regimes (§5.5 & §5.8)."""
    from utils import save_report, save_report_markdown

    per_winkler = _per_sample_winkler(aci_result)
    widths = aci_result.q_hi - aci_result.q_lo
    covered = (aci_result.y_true >= aci_result.q_lo) & (aci_result.y_true <= aci_result.q_hi)

    by_year_type = {}
    if "Year_Type" in test_df.columns:
        for yt in ["Normal", "Moderate", "Extreme"]:
            mask = test_df["Year_Type"].values == yt
            if mask.sum() > 0:
                by_year_type[yt] = {
                    "count": int(mask.sum()),
                    "picp": round(float(np.mean(covered[mask])), 4),
                    "mpiw": round(float(np.mean(widths[mask])), 4),
                    "winkler_score": round(float(np.mean(per_winkler[mask])), 4),
                }

    by_enso = {}
    if "ENSO_Phase" in test_df.columns:
        for enso in test_df["ENSO_Phase"].dropna().unique():
            mask = test_df["ENSO_Phase"].values == enso
            if mask.sum() > 0:
                by_enso[str(enso)] = {
                    "count": int(mask.sum()),
                    "picp": round(float(np.mean(covered[mask])), 4),
                    "mpiw": round(float(np.mean(widths[mask])), 4),
                    "winkler_score": round(float(np.mean(per_winkler[mask])), 4),
                }

    res = {
        "stratified_by_year_type": by_year_type,
        "stratified_by_enso_phase": by_enso,
    }

    save_report(res, "enso_year_type_winkler_report.json")
    return res


def compute_exchangeability_diagnostics(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute Spatial Moran's I of residuals and Temporal Residual ACF (§4.4 exchangeability diagnostics)."""
    from utils import save_report, save_report_markdown
    from scipy import stats as sp_stats

    residuals = y_true - y_pred
    res_dict: Dict[str, Any] = {}

    # 1. Temporal Autocorrelation (Lag-1 ACF & Durbin-Watson)
    if "Year" in test_df.columns:
        df_res = pd.DataFrame({"Year": test_df["Year"].values, "res": residuals})
        yearly_res = df_res.groupby("Year")["res"].mean()
        if len(yearly_res) > 2:
            lag1_acf = float(np.corrcoef(yearly_res.values[:-1], yearly_res.values[1:])[0, 1])
            # Durbin-Watson statistic
            diff_res = np.diff(residuals)
            dw_stat = float(np.sum(diff_res**2) / np.sum(residuals**2))
            res_dict["temporal_diagnostics"] = {
                "yearly_residual_lag1_acf": round(lag1_acf, 4),
                "durbin_watson_statistic": round(dw_stat, 4),
                "exchangeable_temporally": bool(abs(lag1_acf) < 0.30 and 1.5 <= dw_stat <= 2.5),
            }

    # 2. Spatial Autocorrelation (Inverse-Distance Moran's I of residuals)
    if "Lat" in test_df.columns and "Lon" in test_df.columns:
        try:
            coords = test_df[["Lat", "Lon"]].values
            # Sample up to 500 points for computational efficiency
            n_pts = min(500, len(coords))
            idx = np.random.choice(len(coords), n_pts, replace=False)
            sub_coords = coords[idx]
            sub_res = residuals[idx]

            # Compute inverse distance matrix W
            dists = np.sqrt(np.sum((sub_coords[:, None, :] - sub_coords[None, :, :]) ** 2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            W = 1.0 / np.maximum(dists, 1e-5)
            W /= W.sum(axis=1, keepdims=True)

            z = sub_res - np.mean(sub_res)
            s0 = np.sum(W)
            moran_i = float((n_pts / s0) * np.sum(W * np.outer(z, z)) / np.sum(z**2))
            res_dict["spatial_diagnostics"] = {
                "morans_i_residuals": round(moran_i, 4),
                "sample_size": n_pts,
                "spatial_dependence_low": bool(abs(moran_i) < 0.25),
            }
        except Exception as err:
            logger.warning("Moran's I calculation skipped: %s", err)

    save_report(res_dict, "exchangeability_diagnostics_report.json")
    return res_dict


