# -*- coding: utf-8 -*-
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
from utils import save_report, save_report_csv, save_report_markdown

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
    n_warmup: int = 10,
    batch_sizes: Optional[List[int]] = None,
    scaler: Any = None,
) -> Dict[str, Any]:
    """Benchmark inference latency, memory and throughput (§4.7).

    Corrections over the previous version, which produced optimistic and
    ambiguous numbers:

    * **Warm-up iterations** are now run and discarded. The first forward passes
      pay lazy initialisation, allocator and kernel-selection costs.
    * **``torch.no_grad()``** is used. The previous timing built an autograd
      graph on every pass, so it measured neither training nor deployment
      inference, and inflated memory.
    * **CUDA synchronisation** brackets every timed region when a GPU is
      present; CUDA launches are asynchronous, so timing without a sync measures
      queueing, not computation. The device actually used is recorded, so a CPU
      run can never be reported as GPU throughput.
    * **Batch size is explicit** and several are reported, because a single
      "throughput" figure derived from one whole-test-set batch is not
      comparable to per-request latency.
    * **Preprocessing, model inference and end-to-end latency are reported
      separately**, so model latency is never presented as full-pipeline
      latency.
    * **Memory methodology is stated**: process RSS delta for CPU,
      ``torch.cuda.max_memory_allocated`` for GPU.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_set.point_model
    is_neural = bool(getattr(model_set, "is_neural", False))
    if not is_neural:
        return {
            "status": "skipped",
            "reason": "benchmark targets the neural backbone; model_set is not neural",
        }

    # Remember where the model was so it can be restored afterwards. Leaving it
    # on CUDA would break every later predict_intervals() call that builds its
    # input tensor on the model's device.
    try:
        original_device = next(model.parameters()).device
    except StopIteration:
        original_device = torch.device("cpu")
    model = model.to(device).eval()
    X_sample = np.asarray(X_sample, dtype=np.float32)
    n_total = len(X_sample)
    if batch_sizes is None:
        batch_sizes = [b for b in (1, 32, 256, n_total) if b <= n_total]
        batch_sizes = sorted(set(batch_sizes))

    try:
        import psutil
        proc = psutil.Process()
        cpu_mem_before_mb = proc.memory_info().rss / (1024 * 1024)
    except ImportError:
        proc, cpu_mem_before_mb = None, None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    results_by_batch = []
    for bs in batch_sizes:
        xb = X_sample[:bs]
        t_xb = torch.tensor(xb, dtype=torch.float32, device=device)

        with torch.no_grad():
            for _ in range(n_warmup):          # warm-up, discarded
                _ = model(t_xb)
        _sync()

        with torch.no_grad():
            _sync()
            t0 = time.perf_counter()
            for _ in range(n_repeats):
                _ = model(t_xb)
            _sync()
            t1 = time.perf_counter()
        per_call_s = (t1 - t0) / n_repeats

        # End-to-end: numpy -> (optional scaling) -> tensor -> forward -> numpy.
        raw_xb = xb.astype(np.float64)
        _sync()
        t2 = time.perf_counter()
        for _ in range(n_repeats):
            arr = raw_xb
            if scaler is not None:
                arr = scaler.transform(arr)
            tt = torch.tensor(np.asarray(arr, dtype=np.float32), device=device)
            with torch.no_grad():
                out = model(tt)
            _ = out[0].detach().cpu().numpy()
        _sync()
        t3 = time.perf_counter()
        e2e_s = (t3 - t2) / n_repeats

        results_by_batch.append({
            "batch_size": int(bs),
            "model_inference_latency_ms": round(per_call_s * 1000, 4),
            "model_inference_per_sample_ms": round((per_call_s / bs) * 1000, 6),
            "model_inference_throughput_samples_per_sec": round(bs / per_call_s, 1),
            "end_to_end_latency_ms": round(e2e_s * 1000, 4),
            "end_to_end_throughput_samples_per_sec": round(bs / e2e_s, 1),
            "preprocessing_and_transfer_overhead_ms": round((e2e_s - per_call_s) * 1000, 4),
        })

    gpu_mem_mb = (round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
                  if device.type == "cuda" else None)
    cpu_mem_after_mb = proc.memory_info().rss / (1024 * 1024) if proc is not None else None

    n_params = sum(p.numel() for p in model.parameters())

    # Restore the model to the device it arrived on.
    model.to(original_device)

    single = next((r for r in results_by_batch if r["batch_size"] == 1), results_by_batch[0])
    largest = results_by_batch[-1]

    return {
        "status": "ok",
        "device": device.type,
        "device_name": (torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "n_parameters": int(n_params),
        "n_warmup_iterations": int(n_warmup),
        "n_timed_repeats": int(n_repeats),
        "cuda_synchronised": bool(device.type == "cuda"),
        "gradients_disabled": True,
        "per_batch_results": results_by_batch,
        "single_sample_latency_ms": single["model_inference_latency_ms"],
        "single_sample_end_to_end_latency_ms": single["end_to_end_latency_ms"],
        "batch_latency_ms": largest["model_inference_latency_ms"],
        "batch_size_for_batch_latency": largest["batch_size"],
        "throughput_samples_per_sec": largest["model_inference_throughput_samples_per_sec"],
        "throughput_measurement_note": (
            f"model-inference throughput at batch size {largest['batch_size']} on "
            f"{device.type}; this is NOT full-pipeline throughput -- see "
            f"end_to_end_throughput_samples_per_sec ("
            f"{largest['end_to_end_throughput_samples_per_sec']}/s) for that"
        ),
        "peak_gpu_memory_mb": gpu_mem_mb,
        "gpu_memory_method": ("torch.cuda.max_memory_allocated" if device.type == "cuda"
                              else "not applicable — no CUDA device present"),
        "cpu_memory_rss_before_mb": round(cpu_mem_before_mb, 2) if cpu_mem_before_mb else None,
        "cpu_memory_rss_after_mb": round(cpu_mem_after_mb, 2) if cpu_mem_after_mb else None,
        "cpu_memory_method": "psutil process RSS before/after the benchmark",
        "n_test_samples": int(n_total),
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
            # Per-observation Winkler scores. O6 needs these to build a
            # block-level Friedman test; without them the test silently never
            # ran and the report emitted null statistics.
            "per_sample_winkler": [float(v) for v in _per_sample_winkler(result)],
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

    report = {"methods": {}, "statistical_tests": {}, "paired_t_tests": {},
              "dependence_aware_comparisons": {}}

    for name, result in results.items():
        report["methods"][name] = evaluate_calibration(result, year_types)

    # Year labels for every test observation, so downstream tests can block on
    # the correct unit of independence rather than on individual rows.
    if test_df is not None and "Year" in test_df.columns:
        report["test_years"] = [int(v) for v in test_df["Year"].values]
        report["n_test_years"] = int(test_df["Year"].nunique())

    # Statistical tests & Bootstrap diagnostics
    primary_key = "sa_aci" if "sa_aci" in results else ("aci" if "aci" in results else None)
    if primary_key is not None:
        baselines = [k for k in results if k != primary_key]
        aci_result = results[primary_key]
        aci_winkler = _per_sample_winkler(aci_result)

        raw_p_values = []
        baseline_names = []

        for baseline_name in baselines:
            bl_result = results[baseline_name]
            bl_winkler = _per_sample_winkler(bl_result)

            # 1. Wilcoxon signed-rank
            test_res = wilcoxon_signed_rank_test(aci_winkler, bl_winkler, baseline_name)
            report["statistical_tests"][f"{primary_key}_vs_{baseline_name}"] = test_res

            # 2. Paired t-test (§5.5 secondary)
            t_res = paired_t_test(aci_winkler, bl_winkler, baseline_name)
            report["paired_t_tests"][f"{primary_key}_vs_{baseline_name}"] = t_res

            # 3. Dependence-aware comparison. The Wilcoxon and paired t-test
            #    above treat each county-year row as an independent sample,
            #    which inflates significance under the panel's spatial and
            #    temporal correlation. These are the comparisons to cite.
            if test_df is not None and "Year" in test_df.columns:
                from dependence_aware_stats import (
                    cluster_bootstrap_paired_difference, year_level_paired_test,
                )
                yrs = test_df["Year"].values
                report["dependence_aware_comparisons"][f"{primary_key}_vs_{baseline_name}"] = {
                    "metric": "per-observation Winkler score (lower is better)",
                    "year_block_cluster_bootstrap": cluster_bootstrap_paired_difference(
                        aci_winkler, bl_winkler, yrs,
                        n_boot=cfg.BOOTSTRAP_ITERATIONS, seed=cfg.RANDOM_SEED,
                        label_a=primary_key, label_b=baseline_name, cluster_unit="year",
                    ),
                    "year_level_paired_test": year_level_paired_test(
                        aci_winkler, bl_winkler, yrs,
                        label_a=primary_key, label_b=baseline_name,
                    ),
                    "naive_row_level_tests_are_anticonservative": True,
                }

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
    fold_metrics: List[Dict[str, Any]],
    loso_predictions: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, str]:
    """Aggregate LOSO cross-validation fold metrics.
    
    Generates:
      - Master Table: loso_summary.csv & loso_summary_report.md (Macro Mean, Macro 95% CI, Pooled)
      - Calibration Diagnostics: loso_calibration_diagnostics.csv
      - Ensemble Weight Audit: loso_ensemble_weights.csv
    """
    from utils import save_report_csv, save_report_markdown
    from pathlib import Path

    df = pd.DataFrame(fold_metrics)
    valid_df = df[df["rmse"].notna()].copy() if "rmse" in df.columns else df.copy()

    # If loso_predictions is not passed directly, attempt to load from disk
    if loso_predictions is None or len(loso_predictions) == 0:
        pred_path = getattr(cfg, "OUTPUT_DIR", Path("outputs")) / "predictions" / "loso_predictions.csv"
        if pred_path.exists():
            try:
                loso_predictions = pd.read_csv(pred_path)
            except Exception as _e_lp:
                logger.warning("Could not read %s: %s", pred_path, _e_lp)

    metric_cols = ["rmse", "mae", "r_squared", "picp", "ace", "mpiw", "winkler_score"]
    col_headers = ["RMSE", "MAE", "R²", "PICP", "ACE", "MPIW", "Winkler"]

    master_rows = []

    # 1. State-level fold rows
    for _, r in valid_df.iterrows():
        state_name = r.get("state", "Unknown")
        row_dict = {"Held-out State": state_name}
        for m_key, m_header in zip(metric_cols, col_headers):
            val = r.get(m_key, np.nan)
            row_dict[m_header] = f"{float(val):.4f}" if pd.notna(val) else "N/A"
        master_rows.append(row_dict)

    # 2. Macro-state aggregation (equal weight per state, df=5 Student's t CI)
    n_folds = len(valid_df)
    df_t = max(1, n_folds - 1)
    t_crit = float(sp_stats.t.ppf(0.975, df=df_t))

    macro_mean_dict = {"Held-out State": "Macro Mean"}
    macro_ci_dict = {"Held-out State": "Macro 95% CI"}

    for m_key, m_header in zip(metric_cols, col_headers):
        if m_key in valid_df.columns:
            vals = valid_df[m_key].astype(float).values
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            sem_v = std_v / np.sqrt(len(vals)) if len(vals) > 0 else 0.0
            ci_lo = mean_v - t_crit * sem_v
            ci_hi = mean_v + t_crit * sem_v
            macro_mean_dict[m_header] = f"{mean_v:.4f}"
            macro_ci_dict[m_header] = f"[{ci_lo:.4f}, {ci_hi:.4f}]"
        else:
            macro_mean_dict[m_header] = "N/A"
            macro_ci_dict[m_header] = "N/A"

    master_rows.append(macro_mean_dict)
    master_rows.append(macro_ci_dict)

    # 3. Pooled observation-level performance
    if loso_predictions is not None and len(loso_predictions) > 0:
        y_true = loso_predictions["y_true"].values
        y_pred = loso_predictions["y_pred"].values
        q_lo = loso_predictions["lower_bound"].values
        q_hi = loso_predictions["upper_bound"].values

        p_rmse = float(rmse(y_true, y_pred))
        p_mae = float(mae(y_true, y_pred))
        p_r2 = float(r_squared(y_true, y_pred))
        p_picp = float(picp(y_true, q_lo, q_hi))
        p_ace = float(ace(y_true, q_lo, q_hi))
        p_mpiw = float(mpiw(q_lo, q_hi))
        p_winkler = float(winkler_score(y_true, q_lo, q_hi))

        pooled_dict = {
            "Held-out State": "Pooled Performance",
            "RMSE": f"{p_rmse:.4f}",
            "MAE": f"{p_mae:.4f}",
            "R²": f"{p_r2:.4f}",
            "PICP": f"{p_picp:.4f}",
            "ACE": f"{p_ace:.4f}",
            "MPIW": f"{p_mpiw:.4f}",
            "Winkler": f"{p_winkler:.4f}",
        }
    else:
        pooled_dict = {
            "Held-out State": "Pooled Performance",
            "RMSE": "N/A", "MAE": "N/A", "R²": "N/A",
            "PICP": "N/A", "ACE": "N/A", "MPIW": "N/A", "Winkler": "N/A",
        }
    master_rows.append(pooled_dict)

    master_summary_df = pd.DataFrame(master_rows)
    save_report_csv(master_summary_df, "loso_summary.csv")

    # 4. Calibration diagnostics table
    diag_rows = []
    for _, r in valid_df.iterrows():
        diag_rows.append({
            "held_out_state": r.get("state", "Unknown"),
            "n_test": r.get("n_test", np.nan),
            "nominal_coverage": r.get("nominal_coverage", cfg.NOMINAL_COVERAGE),
            "PICP": r.get("picp", np.nan),
            "ACE": r.get("ace", np.nan),
            "MPIW": r.get("mpiw", np.nan),
            "Winkler": r.get("winkler_score", np.nan),
            "lower_violation_rate": r.get("lower_violation_rate", np.nan),
            "upper_violation_rate": r.get("upper_violation_rate", np.nan),
            "n_covered": r.get("n_covered", np.nan),
            "n_lower_violations": r.get("n_lower_violations", np.nan),
            "n_upper_violations": r.get("n_upper_violations", np.nan),
            "calibration_threshold_q": r.get("calibration_threshold_q", np.nan),
        })
    diag_df = pd.DataFrame(diag_rows)
    save_report_csv(diag_df, "loso_calibration_diagnostics.csv")

    # 5. Ensemble weights table
    ensemble_rows = []
    for _, r in valid_df.iterrows():
        ensemble_rows.append({
            "held_out_state": r.get("state", "Unknown"),
            "NeuralCQR_weight": r.get("neuralcqr_ensemble_weight", np.nan),
            "LightGBM_weight": r.get("lightgbm_ensemble_weight", np.nan),
            "val_r2_neural_alone": r.get("val_r2_neural_alone", np.nan),
            "val_r2_lgb_alone": r.get("val_r2_lgb_alone", np.nan),
            "val_r2_best_blend": r.get("val_r2_best_blend", np.nan),
            "validation_RMSE": r.get("validation_rmse", np.nan),
            "validation_MAE": r.get("validation_mae", np.nan),
            "selected_objective": r.get("selected_objective", "Maximize R2 on dev val_fold"),
        })
    ensemble_df = pd.DataFrame(ensemble_rows)
    save_report_csv(ensemble_df, "loso_ensemble_weights.csv")

    # 6. Build Master Markdown Report
    md = "# Leave-One-State-Out (LOSO) Cross-Validation Report (§5.2)\n\n"
    md += "## Master LOSO Results Table\n\n"
    try:
        md += master_summary_df.to_markdown(index=False) + "\n\n"
    except Exception:
        md += master_summary_df.to_string(index=False) + "\n\n"

    md += "> **Note on Uncertainty Summaries**: Confidence intervals summarize variation across the six held-out states and should not be interpreted as population-level uncertainty estimates.\n\n"

    md += "## Fold-Level Calibration Diagnostics\n\n"
    try:
        md += diag_df.to_markdown(index=False) + "\n\n"
    except Exception:
        md += diag_df.to_string(index=False) + "\n\n"

    md += "## Ensemble Weight Audit (Tuned on Dev Validation Only)\n\n"
    try:
        md += ensemble_df.to_markdown(index=False) + "\n\n"
    except Exception:
        md += ensemble_df.to_string(index=False) + "\n\n"

    save_report_markdown(md, "loso_summary_report.md")
    logger.info("LOSO summary report saved -> loso_summary_report.md")
    return master_summary_df, md


def evaluate_objective_o1(
    with_cdhw_res: Dict[str, Any],
    without_cdhw_res: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate Objective O1: WITH vs WITHOUT CDHW variables.

    Exports ``objective_O1_report.md`` & ``objective_O1_report.json``.
    """
    from utils import save_report_markdown, save_report, save_report_csv

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
    
    # Also save cdhw_ablation_results.csv
    ablation_rows = [
        {"Model": "Base_without_CDHW", "RMSE": without_cdhw_res.get("rmse"), "MAE": without_cdhw_res.get("mae"), "R2": without_cdhw_res.get("r_squared"), "Delta_RMSE": 0.0, "Delta_MAE": 0.0, "Delta_R2": 0.0},
        {"Model": "Base_with_CDHW", "RMSE": with_cdhw_res.get("rmse"), "MAE": with_cdhw_res.get("mae"), "R2": with_cdhw_res.get("r_squared"), "Delta_RMSE": d_rmse, "Delta_MAE": d_mae, "Delta_R2": d_r2},
    ]
    save_report_csv(pd.DataFrame(ablation_rows), "cdhw_ablation_results.csv")
    return report


def evaluate_objective_o4(o4_result: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate Objective O4 from the matched joint-vs-post-hoc experiment.

    Takes the output of ``o4_experiment.run_o4_joint_vs_posthoc`` -- a controlled
    paired experiment in which the *only* difference between the two arms is the
    training paradigm. The previous implementation compared calibration methods
    against each other, which share point predictions by construction and could
    therefore never answer this question.

    No positive conclusion is forced. The verdict follows the measured
    differences and their dependence-aware confidence intervals.

    Exports ``objective_O4_report.md`` & ``objective_O4_report.json``.
    """
    from utils import save_report_markdown, save_report

    comp = o4_result["comparison"]
    sig = o4_result["significance_dependence_aware"]
    distinct = o4_result["distinctness_check"]

    def _sig_line(key: str) -> str:
        b = sig[key]["year_block_cluster_bootstrap"]
        y = sig[key]["year_level_paired_test"]
        if not b.get("inference_supported"):
            return f"inference not supported ({b.get('reason')})"
        return (
            f"mean difference (joint − post-hoc) = `{b['estimate']}`, "
            f"95% CI `{b['ci_95']}` (year-block bootstrap, {b['n_clusters']} years, p = `{b['p_value']}`); "
            f"year-level paired t p = `{y.get('paired_t_p_value')}`"
        )

    # A claim of improvement requires BOTH a favourable point difference AND a
    # dependence-aware CI that excludes zero.
    def _supported(metric_key: str, sig_key: str) -> bool:
        b = sig[sig_key]["year_block_cluster_bootstrap"]
        if not (b.get("inference_supported") and b.get("ci_95")):
            return False
        excludes_zero = (b["ci_95"][0] > 0) or (b["ci_95"][1] < 0)
        return bool(comp[metric_key]["joint_better"] and excludes_zero)

    point_supported = _supported("rmse", "squared_error")
    interval_supported = (_supported("mpiw_static", "static_conformal_width")
                          or _supported("winkler_static", "static_conformal_winkler"))

    if distinct["status"].startswith("FAIL"):
        verdict = "INVALID — the two arms produced identical predictions"
    elif point_supported and interval_supported:
        verdict = "Supported — joint training improves both point prediction and interval quality"
    elif point_supported:
        verdict = "Partially supported — joint training improves point prediction only"
    elif interval_supported:
        verdict = "Partially supported — joint training improves interval quality only"
    else:
        verdict = "Not supported — joint training shows no reliable advantage over post-hoc"

    report = {
        "objective": "O4 — joint end-to-end Neural CQR vs post-hoc CQR",
        "design": o4_result["design"],
        "arms": o4_result["arms"],
        "distinctness_check": distinct,
        "comparison": comp,
        "significance_dependence_aware": sig,
        "point_prediction_improvement_supported": point_supported,
        "interval_quality_improvement_supported": interval_supported,
        "verdict": verdict,
    }

    j_fp = o4_result["arms"]["joint"]["paradigm_fingerprint"]
    p_fp = o4_result["arms"]["post_hoc"]["paradigm_fingerprint"]

    md = f"""# Objective O4 Report: Joint End-to-End Neural CQR vs Post-Hoc CQR (§3 O4 & §4.2)

## Research question
{o4_result['design']['question']}

## What actually differs between the two arms
| | Post-hoc (Condition A) | Joint (Condition B) |
|---|---|---|
| Training paradigm | `{o4_result['arms']['post_hoc']['training_paradigm']}` | `{o4_result['arms']['joint']['training_paradigm']}` |
| Optimization stages | {p_fp.get('n_optimization_stages')} | {j_fp.get('n_optimization_stages')} |
| Stage 1 loss terms | `{p_fp.get('stage1_loss_terms')}` | `{j_fp.get('stage1_loss_terms')}` |
| Stage 2 loss terms | `{p_fp.get('stage2_loss_terms')}` | `{j_fp.get('stage2_loss_terms')}` |
| Quantile gradients reach backbone | {p_fp.get('backbone_receives_quantile_gradient')} | {j_fp.get('backbone_receives_quantile_gradient')} |
| Backbone frozen for interval fitting | {p_fp.get('frozen_backbone_for_quantiles')} | {j_fp.get('frozen_backbone_for_quantiles')} |

**Distinctness check**: {distinct['status']} — max absolute difference in test point
predictions between arms = `{distinct['max_abs_point_difference']}`.

## Held constant across both arms
{chr(10).join('- ' + c for c in o4_result['design']['controlled_for'])}

Only difference: **{o4_result['design']['only_difference']}**.

## Point prediction
| Metric | Post-hoc | Joint | Absolute Δ (joint − post-hoc) | Relative Δ | Joint better? |
|---|---:|---:|---:|---:|---|
| RMSE | {comp['rmse']['post_hoc']} | {comp['rmse']['joint']} | {comp['rmse']['absolute_difference_joint_minus_posthoc']} | {comp['rmse']['relative_difference_pct']}% | {comp['rmse']['joint_better']} |
| MAE | {comp['mae']['post_hoc']} | {comp['mae']['joint']} | {comp['mae']['absolute_difference_joint_minus_posthoc']} | {comp['mae']['relative_difference_pct']}% | {comp['mae']['joint_better']} |
| R² | {comp['r_squared']['post_hoc']} | {comp['r_squared']['joint']} | {comp['r_squared']['absolute_difference_joint_minus_posthoc']} | {comp['r_squared']['relative_difference_pct']}% | {comp['r_squared']['joint_better']} |

- Squared-error difference: {_sig_line('squared_error')}
- Absolute-error difference: {_sig_line('absolute_error')}

## Uncertainty — static split conformal (identical calibration on both arms)
| Metric | Post-hoc | Joint | Absolute Δ | Relative Δ | Joint better? |
|---|---:|---:|---:|---:|---|
| PICP | {comp['picp_static']['post_hoc']} | {comp['picp_static']['joint']} | {comp['picp_static']['absolute_difference_joint_minus_posthoc']} | {comp['picp_static']['relative_difference_pct']}% | {comp['picp_static']['joint_better']} |
| MPIW | {comp['mpiw_static']['post_hoc']} | {comp['mpiw_static']['joint']} | {comp['mpiw_static']['absolute_difference_joint_minus_posthoc']} | {comp['mpiw_static']['relative_difference_pct']}% | {comp['mpiw_static']['joint_better']} |
| ACE | {comp['ace_static']['post_hoc']} | {comp['ace_static']['joint']} | {comp['ace_static']['absolute_difference_joint_minus_posthoc']} | {comp['ace_static']['relative_difference_pct']}% | {comp['ace_static']['joint_better']} |
| Winkler | {comp['winkler_static']['post_hoc']} | {comp['winkler_static']['joint']} | {comp['winkler_static']['absolute_difference_joint_minus_posthoc']} | {comp['winkler_static']['relative_difference_pct']}% | {comp['winkler_static']['joint_better']} |

- Winkler difference: {_sig_line('static_conformal_winkler')}
- Interval-width difference: {_sig_line('static_conformal_width')}

## Uncertainty — ACI (identical calibration on both arms)
| Metric | Post-hoc | Joint | Absolute Δ | Relative Δ | Joint better? |
|---|---:|---:|---:|---:|---|
| PICP | {comp['picp_aci']['post_hoc']} | {comp['picp_aci']['joint']} | {comp['picp_aci']['absolute_difference_joint_minus_posthoc']} | {comp['picp_aci']['relative_difference_pct']}% | {comp['picp_aci']['joint_better']} |
| MPIW | {comp['mpiw_aci']['post_hoc']} | {comp['mpiw_aci']['joint']} | {comp['mpiw_aci']['absolute_difference_joint_minus_posthoc']} | {comp['mpiw_aci']['relative_difference_pct']}% | {comp['mpiw_aci']['joint_better']} |
| ACE | {comp['ace_aci']['post_hoc']} | {comp['ace_aci']['joint']} | {comp['ace_aci']['absolute_difference_joint_minus_posthoc']} | {comp['ace_aci']['relative_difference_pct']}% | {comp['ace_aci']['joint_better']} |
| Winkler | {comp['winkler_aci']['post_hoc']} | {comp['winkler_aci']['joint']} | {comp['winkler_aci']['absolute_difference_joint_minus_posthoc']} | {comp['winkler_aci']['relative_difference_pct']}% | {comp['winkler_aci']['joint_better']} |

- ACI Winkler difference: {_sig_line('aci_winkler')}
- ACI width difference: {_sig_line('aci_width')}

## Verdict
**{verdict}**

- Point-prediction improvement supported by the data: **{point_supported}**
- Interval-quality improvement supported by the data: **{interval_supported}**

A metric counts as improved only when the joint arm is better *and* the
dependence-aware 95% confidence interval for the paired difference excludes
zero. With {sig['squared_error']['year_level_paired_test'].get('n_years')} test
years the unit of independence is the year, so this comparison has limited
power; a non-significant result is not evidence of equivalence.
"""

    save_report_markdown(md, "objective_O4_report.md")
    save_report(report, "objective_O4_report.json")
    logger.info("Objective O4 exported — verdict: %s", verdict)
    return report


def evaluate_objective_o5(
    aci_result: CalibrationResult,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Evaluate Objective O5: does interval width scale with CDHW severity? (§3 O5 & §5.8)

    Inference note
    --------------
    The previous version ran ``pearsonr`` / ``linregress`` / ANOVA over ~2.3k
    county-year rows as if they were independent draws. They are not: counties
    within a year share weather, and each county is a short autocorrelated time
    series. The naive p-values were therefore meaningless, and the naive slope
    CI far too narrow. The primary inference here is a **cluster-robust OLS with
    years as clusters**, corroborated by a **year-block cluster bootstrap**. The
    effective sample size is reported alongside so the reader can see the cost
    of the dependence.

    Statistical detectability and effect magnitude are reported as separate
    conclusions: a slope can be reliably non-zero while explaining almost none
    of the variance, and that is what the data show here.

    Exports objective_o5_report.{json,csv,md} and objective_o5_analysis.md.
    """
    from utils import save_report_markdown, save_report_csv, save_report
    from dependence_aware_stats import (
        cluster_robust_ols, cluster_bootstrap_mean, effective_sample_size,
        year_level_paired_test,
    )

    widths = np.asarray(aci_result.q_hi) - np.asarray(aci_result.q_lo)
    severity = (
        test_df["CDHW_Severity_Score"].values
        if "CDHW_Severity_Score" in test_df.columns else np.zeros(len(widths))
    )
    years = test_df["Year"].values if "Year" in test_df.columns else np.zeros(len(widths), dtype=int)
    geoids = test_df["GEOID"].values if "GEOID" in test_df.columns else np.arange(len(widths))

    # ── Descriptive association (no inferential claim attached) ──────────
    p_corr, p_p_naive = sp_stats.pearsonr(widths, severity)
    s_corr, s_p_naive = sp_stats.spearmanr(widths, severity)
    slope_n, intercept_n, r_val, p_val_naive, std_err_n = sp_stats.linregress(severity, widths)

    # ── PRIMARY inference: cluster-robust OLS, clustered by year ─────────
    cr_year = cluster_robust_ols(widths, severity, years,
                                 cluster_unit="year", x_name="CDHW_Severity_Score")
    # Secondary: clustered by county, which has many more clusters but
    # does not absorb the shared year-level shocks.
    cr_county = cluster_robust_ols(widths, severity, geoids,
                                   cluster_unit="county (GEOID)", x_name="CDHW_Severity_Score")

    # ── Corroboration: year-block cluster bootstrap of the OLS slope ─────
    uniq_years = np.unique(years)
    rng = np.random.RandomState(cfg.RANDOM_SEED)
    idx_by_year = {y: np.where(years == y)[0] for y in uniq_years}
    boot_slopes = []
    if len(uniq_years) >= 3:
        for _ in range(cfg.BOOTSTRAP_ITERATIONS):
            drawn = rng.choice(uniq_years, size=len(uniq_years), replace=True)
            idx = np.concatenate([idx_by_year[y] for y in drawn])
            if np.std(severity[idx]) < 1e-12:
                continue
            boot_slopes.append(float(np.polyfit(severity[idx], widths[idx], 1)[0]))
    if len(boot_slopes) >= 100:
        bs = np.array(boot_slopes)
        boot_ci = [round(float(np.percentile(bs, 2.5)), 6), round(float(np.percentile(bs, 97.5)), 6)]
        boot_p = min(1.0, max(2.0 * min(float(np.mean(bs <= 0)), float(np.mean(bs >= 0))),
                              1.0 / len(bs)))
        boot_block = {
            "method": "year-block cluster bootstrap of OLS slope",
            "cluster_unit": "year",
            "n_blocks": int(len(uniq_years)),
            "n_boot": int(len(bs)),
            "slope_mean": round(float(np.mean(bs)), 6),
            "ci_95": boot_ci,
            "p_value": round(float(boot_p), 6),
            "inference_supported": True,
        }
    else:
        boot_block = {
            "method": "year-block cluster bootstrap of OLS slope",
            "cluster_unit": "year",
            "n_blocks": int(len(uniq_years)),
            "inference_supported": False,
            "reason": "too few year blocks for a stable bootstrap",
        }

    ess = effective_sample_size(widths, years)

    # ── Group comparison across year types, at the correct unit ──────────
    # Year_Type is a year-level label, so the honest comparison is between
    # year-level mean widths, not between thousands of correlated rows.
    group_analysis: Dict[str, Any] = {"unit_of_independence": "year"}
    if "Year_Type" in test_df.columns:
        yt_df = pd.DataFrame({"year": years, "width": widths,
                              "year_type": test_df["Year_Type"].values})
        per_year = yt_df.groupby(["year", "year_type"], as_index=False)["width"].mean()
        group_analysis["per_year_mean_width"] = [
            {"year": int(r.year), "year_type": str(r.year_type), "mean_width": round(float(r.width), 4)}
            for r in per_year.itertuples()
        ]
        groups_y = {g: v["width"].values for g, v in per_year.groupby("year_type")}
        group_analysis["n_years_per_type"] = {k: int(len(v)) for k, v in groups_y.items()}
        usable = {k: v for k, v in groups_y.items() if len(v) >= 2}
        if len(usable) >= 2 and sum(len(v) for v in usable.values()) >= 5:
            stat_val, p_val_group = sp_stats.kruskal(*usable.values())
            group_analysis.update({
                "test_used": "Kruskal-Wallis on year-level means",
                "statistic": round(float(stat_val), 4),
                "p_value": round(float(p_val_group), 6),
                "inference_supported": True,
                "significant_at_0.05": bool(p_val_group < 0.05),
            })
        else:
            group_analysis.update({
                "test_used": None, "statistic": None, "p_value": None,
                "inference_supported": False,
                "reason": (
                    "with only "
                    f"{len(uniq_years)} test years there are too few independent "
                    "year-level observations per year type to support a group test; "
                    "per-year means are reported descriptively instead"
                ),
            })
    else:
        group_analysis.update({"inference_supported": False, "reason": "Year_Type column absent"})

    # ── Stage-specific severity, same treatment ──────────────────────────
    stage_corrs = {}
    for col_name, stage_label in [
        ("CDHW_Veg_Severity", "Vegetative"),
        ("CDHW_Silking_Severity", "Silking_R1"),
        ("CDHW_GrainFill_Severity", "Grain_Fill"),
    ]:
        if col_name in test_df.columns:
            stg_vals = test_df[col_name].values
            r_stg, _ = sp_stats.pearsonr(widths, stg_vals)
            rho_stg, _ = sp_stats.spearmanr(widths, stg_vals)
            cr_stg = cluster_robust_ols(widths, stg_vals, years,
                                        cluster_unit="year", x_name=col_name)
            stage_corrs[stage_label] = {
                "column": col_name,
                "pearson_r_descriptive": round(float(r_stg), 4),
                "spearman_rho_descriptive": round(float(rho_stg), 4),
                "cluster_robust_slope": cr_stg["slope"],
                "cluster_robust_p_value": cr_stg.get("p_value"),
                "cluster_robust_ci_95": cr_stg.get("ci_95"),
                "r_squared": cr_stg["r_squared"],
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

    r2 = float(cr_year["r_squared"])
    cr_p = cr_year.get("p_value")
    detectable = bool(cr_year.get("significant_at_0.05")) and bool(
        boot_block.get("inference_supported") and boot_block.get("p_value", 1.0) < 0.05
    )

    # ── CSV ──────────────────────────────────────────────────────────────
    csv_rows = [
        {"Metric": "Pearson_r_descriptive", "Value": round(float(p_corr), 4), "p_value": None,
         "Inference": "descriptive only (rows are not independent)"},
        {"Metric": "Spearman_rho_descriptive", "Value": round(float(s_corr), 4), "p_value": None,
         "Inference": "descriptive only (rows are not independent)"},
        {"Metric": "Slope_cluster_robust_year", "Value": cr_year["slope"], "p_value": cr_p,
         "Inference": "PRIMARY: OLS + CR1 SE clustered by year"},
        {"Metric": "Slope_cluster_robust_county", "Value": cr_county["slope"],
         "p_value": cr_county.get("p_value"), "Inference": "secondary: clustered by county"},
        {"Metric": "Slope_year_block_bootstrap", "Value": boot_block.get("slope_mean"),
         "p_value": boot_block.get("p_value"), "Inference": "corroborating year-block bootstrap"},
        {"Metric": "R_squared", "Value": round(r2, 4), "p_value": None,
         "Inference": "effect magnitude (unaffected by SE correction)"},
        {"Metric": "Effective_sample_size", "Value": ess.get("n_effective"), "p_value": None,
         "Inference": f"from n={ess.get('n_observations')} rows, ICC={ess.get('icc')}"},
    ]
    for stg, data in stage_corrs.items():
        csv_rows.append({"Metric": f"Slope_cluster_robust_{stg}", "Value": data["cluster_robust_slope"],
                         "p_value": data["cluster_robust_p_value"], "Inference": "clustered by year"})
    save_report_csv(pd.DataFrame(csv_rows), "objective_o5_report.csv")

    report = {
        "descriptive_association": {
            "pearson_r": round(float(p_corr), 4),
            "spearman_rho": round(float(s_corr), 4),
            "naive_ols_slope": round(float(slope_n), 6),
            "naive_ols_p_value": round(float(p_val_naive), 8),
            "naive_p_values_are_invalid_because": (
                "county-year rows are spatially and temporally dependent; the naive "
                "p-value assumes independent sampling and is reported only to show "
                "the size of the distortion"
            ),
        },
        "primary_inference": cr_year,
        "secondary_inference_county_clustered": cr_county,
        "corroborating_year_block_bootstrap": boot_block,
        "effective_sample_size": ess,
        "regression": {
            "slope": cr_year["slope"],
            "intercept": cr_year["intercept"],
            "r_squared": round(r2, 6),
            "std_error": cr_year.get("std_error"),
            "ci_95_slope": cr_year.get("ci_95"),
            "se_type": "cluster-robust (CR1), clustered by year",
        },
        "stage_correlations": stage_corrs,
        "phenological_stage_widths": stage_widths,
        "group_comparison_by_year_type": group_analysis,
        "statistically_detectable": detectable,
        "variance_explained_fraction": round(r2, 6),
        "effect_magnitude_label": (
            "negligible" if r2 < 0.02 else "weak" if r2 < 0.09
            else "moderate" if r2 < 0.25 else "substantial"
        ),
    }

    # ── Interpretation: detectability and magnitude stated separately ────
    direction = "positive" if cr_year["slope"] > 0 else "negative"
    if detectable:
        detect_txt = (
            f"statistically detectable under year-clustered inference "
            f"(cluster-robust slope = {cr_year['slope']:.4f}, "
            f"95% CI {cr_year.get('ci_95')}, p = {cr_p}, "
            f"{cr_year['n_clusters']} year clusters; year-block bootstrap p = "
            f"{boot_block.get('p_value')})"
        )
    else:
        detect_txt = (
            f"not statistically distinguishable from zero once dependence is "
            f"accounted for (cluster-robust slope = {cr_year['slope']:.4f}, "
            f"95% CI {cr_year.get('ci_95')}, p = {cr_p})"
        )
    finding_text = (
        f"Prediction interval width shows a {direction} association with CDHW severity that is "
        f"{detect_txt}. The association is {report['effect_magnitude_label']} in magnitude: "
        f"CDHW severity explains R² = {r2:.4f} "
        f"({100 * r2:.1f}%) of the variation in interval width, so severity alone accounts for "
        f"only a small fraction of the observed variation. Statistical detectability and effect "
        f"magnitude are distinct conclusions and are reported separately here."
    )
    report["interpretation"] = finding_text

    stage_md_lines = [
        f"- **{stg}**: cluster-robust slope = `{d['cluster_robust_slope']}` "
        f"(p = `{d['cluster_robust_p_value']}`, 95% CI `{d['cluster_robust_ci_95']}`), "
        f"R² = `{d['r_squared']}`, descriptive Pearson r = `{d['pearson_r_descriptive']}`"
        for stg, d in stage_corrs.items()
    ]

    md = f"""# Objective O5 Report: Uncertainty Scaling vs CDHW Severity (§3 O5 & §5.8)

## Unit of independence
County-year rows are **not** independent: counties share year-level weather shocks and
each county is an autocorrelated series. All inference below therefore clusters on
**year** ({cr_year['n_clusters']} clusters over {cr_year['n_observations']} rows).

- **Effective sample size**: `{ess.get('n_effective')}` (from `{ess.get('n_observations')}` rows;
  ICC = `{ess.get('icc')}`, design effect = `{ess.get('design_effect')}`)

## Primary inference — OLS with cluster-robust (CR1) SE, clustered by year
- **Slope**: `{cr_year['slope']}` per unit CDHW severity
- **95% CI**: `{cr_year.get('ci_95')}`
- **p-value**: `{cr_p}` (t = `{cr_year.get('t_statistic')}`, df = `{cr_year.get('degrees_of_freedom')}`)
- **R²**: `{round(r2, 4)}`

## Corroborating year-block cluster bootstrap
- **Slope**: `{boot_block.get('slope_mean')}`, **95% CI**: `{boot_block.get('ci_95')}`, **p**: `{boot_block.get('p_value')}`
- Resampling unit: `{boot_block.get('cluster_unit')}` ({boot_block.get('n_blocks')} blocks)

## Secondary inference — clustered by county
- **Slope**: `{cr_county['slope']}`, **95% CI**: `{cr_county.get('ci_95')}`, **p**: `{cr_county.get('p_value')}`
  ({cr_county.get('n_clusters')} county clusters)

## Descriptive association (NOT used for inference)
- Pearson r = `{round(float(p_corr), 4)}`, Spearman ρ = `{round(float(s_corr), 4)}`
- Naive OLS p-value = `{round(float(p_val_naive), 8)}` — **invalid**, shown only to
  document how much the independence assumption distorts the result.

## Phenology stage-specific associations
{chr(10).join(stage_md_lines) if stage_md_lines else "- No stage-specific columns evaluated."}

## Group differences across year types
- Unit of independence: `year`
- Test: `{group_analysis.get('test_used')}`, statistic = `{group_analysis.get('statistic')}`, p = `{group_analysis.get('p_value')}`
- Inference supported: `{group_analysis.get('inference_supported')}`{(" — " + str(group_analysis.get('reason'))) if group_analysis.get('reason') else ""}

## Interpretation
{finding_text}
"""

    save_report_markdown(md, "objective_o5_report.md")
    save_report_markdown(md, "objective_o5_analysis.md")
    save_report(report, "objective_o5_report.json")
    logger.info(
        "Objective O5 exported (cluster-robust slope=%.4f, p=%s, R2=%.4f, detectable=%s, magnitude=%s)",
        cr_year["slope"], cr_p, r2, detectable, report["effect_magnitude_label"],
    )
    return report


def evaluate_objective_o6(
    eval_report: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate Objective O6: rank the six conformal calibration methods (§3 O6, §4.5, §5.5).

    What changed and why
    --------------------
    * The Nemenyi critical value was hard-coded as ``2.728 if k == 5 else 2.569``.
      2.569 is the constant for k=4, so the six-method comparison used a critical
      difference that was far too small and over-declared significance. The
      constant is now looked up by the actual ``k`` (see
      ``dependence_aware_stats.nemenyi_critical_difference``).
    * Friedman was run over per-observation Winkler scores, treating thousands of
      correlated county-year rows as independent blocks. Blocks are now **years**.
    * The report previously printed "differences are statistically significant
      (p < 0.05)" whenever the test merely ran, and emitted null statistics as if
      the test had passed. The conclusion now follows the actual p-value, and if
      the test cannot be justified the ranking is explicitly labelled descriptive.
    * Nemenyi post-hoc pairwise comparisons are actually performed when the
      omnibus test rejects, instead of only printing a CD value.
    """
    from utils import save_report_markdown, save_report
    from dependence_aware_stats import friedman_on_blocks

    methods = eval_report["methods"]
    table = []
    per_sample: Dict[str, np.ndarray] = {}

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
            per_sample[name] = np.asarray(data["uncertainty"]["per_sample_winkler"])

    df_rank = pd.DataFrame(table)
    df_rank["Rank_Winkler"] = df_rank["Winkler_Score"].rank(ascending=True)
    df_rank["Rank_ACE"] = df_rank["ACE"].rank(ascending=True)
    df_rank["Rank_MPIW"] = df_rank["MPIW"].rank(ascending=True)
    df_rank["Average_Rank"] = df_rank[["Rank_Winkler", "Rank_ACE", "Rank_MPIW"]].mean(axis=1)
    df_rank.sort_values(by="Average_Rank", inplace=True)

    k_methods = len(df_rank)
    years_test = eval_report.get("test_years")

    friedman: Dict[str, Any]
    if per_sample and years_test is not None and len(per_sample) >= 3:
        blocks = np.asarray(years_test)
        lengths = {len(v) for v in per_sample.values()} | {len(blocks)}
        if len(lengths) == 1:
            friedman = friedman_on_blocks(per_sample, blocks, alpha=0.05)
        else:
            friedman = {
                "test_executed": False, "statistic": None, "p_value": None,
                "k": len(per_sample),
                "reason": "per-sample Winkler series have inconsistent lengths",
            }
    else:
        friedman = {
            "test_executed": False, "statistic": None, "p_value": None,
            "k": len(per_sample),
            "reason": (
                "per-sample Winkler scores or test-year labels unavailable; "
                "a block-level Friedman test cannot be constructed"
            ),
        }

    test_ran = bool(friedman.get("test_executed"))
    significant = bool(friedman.get("significant_at_alpha")) if test_ran else False
    ranking_status = (
        "statistical (Friedman rejected; Nemenyi post-hoc applied)"
        if (test_ran and significant) else "descriptive ranking only"
    )

    # Practical-significance context: how far apart are the methods really?
    wink = df_rank["Winkler_Score"].astype(float)
    spread = {
        "winkler_best": round(float(wink.min()), 4),
        "winkler_worst": round(float(wink.max()), 4),
        "winkler_relative_spread_pct": round(float(100.0 * (wink.max() - wink.min()) / max(1e-9, wink.max())), 2),
        "picp_min": round(float(df_rank["PICP"].astype(float).min()), 4),
        "picp_max": round(float(df_rank["PICP"].astype(float).max()), 4),
        "mpiw_relative_spread_pct": round(
            float(100.0 * (df_rank["MPIW"].astype(float).max() - df_rank["MPIW"].astype(float).min())
                  / max(1e-9, df_rank["MPIW"].astype(float).max())), 2),
    }

    report = {
        "n_methods_compared": int(k_methods),
        "methods_compared": df_rank["Method"].tolist(),
        "rankings": df_rank.to_dict(orient="records"),
        "ranking_basis": "mean of ranks on Winkler, ACE and MPIW",
        "ranking_status": ranking_status,
        "friedman_test": friedman,
        "nemenyi_posthoc": friedman.get("nemenyi") if test_ran else {
            "critical_difference": None,
            "reason": "omnibus test not executed, so no post-hoc comparison is performed",
        },
        "nemenyi_pairwise": friedman.get("nemenyi_pairwise", []) if (test_ran and significant) else [],
        "practical_significance": spread,
        "claims_statistical_superiority": bool(test_ran and significant
                                               and friedman.get("n_significant_pairs", 0) > 0),
    }

    eval_report["friedman_test"] = report["friedman_test"]
    eval_report["nemenyi_posthoc"] = report["nemenyi_posthoc"]

    best_method = df_rank.iloc[0]["Method"]
    md = "# Objective O6 Report: Distribution-Shift-Robust Conformal Benchmarking (§3 O6 & §4.5 & §5.5)\n\n"
    md += f"**Methods compared (k = {k_methods})**: {', '.join(df_rank['Method'].tolist())}\n\n"
    try:
        md += df_rank.to_markdown(index=False) + "\n\n"
    except Exception:
        md += df_rank.to_string(index=False) + "\n\n"

    md += f"**Lowest average rank**: `{best_method}` (mean rank over Winkler, ACE, MPIW).\n\n"
    md += f"**Status of this ranking**: {ranking_status}.\n\n"

    md += "## Statistical testing\n"
    if test_ran:
        md += (
            f"- **Blocking unit**: `{friedman.get('block_unit')}` "
            f"({friedman.get('n_blocks')} blocks) — per-observation blocking would treat "
            "correlated county-year rows as independent.\n"
            f"- **Friedman**: Q = `{friedman.get('statistic')}`, p = `{friedman.get('p_value')}`\n"
            f"- **Average ranks**: `{friedman.get('average_ranks')}`\n"
            f"- **Nemenyi CD** (k = {friedman.get('nemenyi', {}).get('k')}, "
            f"N = {friedman.get('nemenyi', {}).get('n_blocks')}, α = 0.05): "
            f"`{friedman.get('nemenyi', {}).get('critical_difference')}` "
            f"(q_α = `{friedman.get('nemenyi', {}).get('q_alpha')}`)\n"
        )
        if significant:
            n_sig = friedman.get("n_significant_pairs", 0)
            md += (
                f"- **Conclusion**: the omnibus test rejects at α = 0.05. "
                f"Nemenyi post-hoc finds **{n_sig}** of "
                f"{len(friedman.get('nemenyi_pairwise', []))} pairwise rank differences "
                f"exceeding the critical difference.\n"
            )
            if n_sig == 0:
                md += (
                    "- Note: no individual pair separates after the post-hoc correction, "
                    "so no method can be declared superior to another on this evidence.\n"
                )
        else:
            md += (
                f"- **Conclusion**: the omnibus test does **not** reject at α = 0.05 "
                f"(p = `{friedman.get('p_value')}`). No post-hoc comparison is performed and "
                "no method is claimed to be statistically superior. The ordering above is "
                "descriptive.\n"
            )
    else:
        md += (
            f"- **Not executed**: {friedman.get('reason')}\n"
            "- The ordering above is therefore a **descriptive ranking**, not evidence of "
            "statistical superiority. No test statistic or p-value is reported.\n"
        )

    md += (
        "\n## Practical significance\n"
        f"- Winkler score spans `{spread['winkler_best']}` to `{spread['winkler_worst']}` "
        f"({spread['winkler_relative_spread_pct']}% relative spread)\n"
        f"- PICP spans `{spread['picp_min']}` to `{spread['picp_max']}` "
        f"(nominal {cfg.NOMINAL_COVERAGE})\n"
        f"- MPIW relative spread: `{spread['mpiw_relative_spread_pct']}%`\n"
    )

    save_report_markdown(md, "objective_O6_report.md")
    save_report(report, "objective_O6_report.json")
    logger.info(
        "Objective O6 exported (k=%d, friedman_executed=%s, significant=%s, status=%s)",
        k_methods, test_ran, significant, ranking_status,
    )
    return report


def export_statistical_tests_report_md(eval_report: Dict[str, Any]) -> None:
    """Export statistical_tests_report.md (§5.5)."""
    from utils import save_report_markdown

    md = "# Statistical Significance Testing Report (§5.5)\n\n"
    md += "## 1. Primary Pairwise Tests: Wilcoxon Signed-Rank\n\n"
    if "statistical_tests" in eval_report and eval_report["statistical_tests"]:
        for k, v in eval_report["statistical_tests"].items():
            stat_str = f"`{v.get('statistic')}`" if v.get('statistic') is not None else "N/A"
            p_str = f"`{v.get('p_value')}`" if v.get('p_value') is not None else "N/A"
            md += f"- **{k}**: Wilcoxon W = {stat_str}, p-value = {p_str}\n"
    else:
        md += "- *Pairwise Wilcoxon signed-rank tests were not recorded or executed.*\n"

    md += "\n## 2. Secondary Robustness Check: Paired t-Tests & Effect Sizes (Cohen's d)\n\n"
    if "paired_t_tests" in eval_report and eval_report["paired_t_tests"]:
        for k, v in eval_report["paired_t_tests"].items():
            t_str = f"`{v.get('statistic')}`" if v.get('statistic') is not None else "N/A"
            p_str = f"`{v.get('p_value')}`" if v.get('p_value') is not None else "N/A"
            d_str = f"`{v.get('cohen_d_effect_size')}`" if v.get('cohen_d_effect_size') is not None else "N/A"
            ci_str = f"`{v.get('ci_95_difference')}`" if v.get('ci_95_difference') is not None else "N/A"
            md += f"- **{k}**: t = {t_str}, p-value = {p_str}, Cohen's d = {d_str}, 95% CI Diff = {ci_str}\n"
    else:
        md += "- *Paired t-tests were not recorded or executed.*\n"

    md += "\n## 3. Multi-Method Significance: Friedman & Nemenyi Post-Hoc Test\n\n"
    if "friedman_test" in eval_report and eval_report["friedman_test"].get("statistic") is not None:
        ft = eval_report["friedman_test"]
        nem = eval_report.get("nemenyi_posthoc", {})
        md += f"- **Friedman Test Q-Statistic**: `{ft.get('statistic')}` (p-value = `{ft.get('p_value')}`)\n"
        md += f"- **Nemenyi Critical Difference (CD)**: `{nem.get('critical_difference_cd')}` (α = 0.05)\n"
    else:
        md += "- *Omnibus Friedman/Nemenyi rank test not executed; pairwise Wilcoxon signed-rank and block bootstraps serve as authoritative statistical inference.*\n"

    md += "\n## 4. Multiple Comparison Corrections\n\n"
    if "holm_bonferroni" in eval_report and eval_report["holm_bonferroni"]:
        md += "### Holm-Bonferroni Correction\n"
        for entry in eval_report["holm_bonferroni"]:
            md += f"- **{entry['baseline']}**: adjusted p = `{entry['corrected_p_value']}` (significant at α=0.05: `{entry['significant_at_0.05']}`)\n"

    if "benjamini_hochberg" in eval_report and eval_report["benjamini_hochberg"]:
        md += "\n### Benjamini-Hochberg (FDR) Correction\n"
        for entry in eval_report["benjamini_hochberg"]:
            md += f"- **{entry['baseline']}**: BH adjusted p = `{entry['bh_corrected_p_value']}` (significant at FDR=0.05: `{entry['significant_at_0.05']}`)\n"

    md += "\n## 5. Primary Robustness Inference: Spatial & Temporal Clustered Block Bootstraps\n\n"
    if "county_block_bootstrap" in eval_report:
        cb = eval_report["county_block_bootstrap"]
        md += f"### County-Block Bootstrap ({cb.get('iterations', 2000)} iterations, {cb.get('n_counties', 0)} counties)\n"
        for m in ["r2", "rmse", "picp", "mpiw", "ace", "winkler"]:
            if m in cb:
                md += f"- **{m.upper()}**: Mean = `{cb[m]['mean']}`, 95% CI = `[{cb[m]['ci_95'][0]}, {cb[m]['ci_95'][1]}]`\n"

    if "year_block_bootstrap" in eval_report:
        yb = eval_report["year_block_bootstrap"]
        md += f"\n### Year-Block Bootstrap ({yb.get('iterations', 2000)} iterations, {yb.get('n_years', 0)} years)\n"
        for m in ["r2", "rmse", "picp", "mpiw", "ace", "winkler"]:
            if m in yb:
                md += f"- **{m.upper()}**: Mean = `{yb[m]['mean']}`, 95% CI = `[{yb[m]['ci_95'][0]}, {yb[m]['ci_95'][1]}]`\n"

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
    """Stratify Winkler Score, PICP, and MPIW across Year Types and ENSO Regimes (Section 5.5 & Section 5.8)."""
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
    """Compute Spatial Moran's I of residuals and Temporal Residual ACF (Section 4.4 exchangeability diagnostics)."""
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


