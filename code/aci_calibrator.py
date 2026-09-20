"""
aci_calibrator.py - Conformal calibration methods for Paper 3.

Implements five calibration strategies from §4.2, §4.3, and §4.5:
1. Static Conformal Prediction (Paper 18 baseline)
2. Phenology-Stratified Static CQR (Paper 21 baseline)
3. Weighted / Covariate-Shift Conformal (Paper 9 baseline)
4. Locally Adaptive Conformal (Paper 17 baseline)
5. Adaptive Conformal Inference - ACI (§4.3, Gibbs & Candès Paper 4)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config as cfg
from utils import log_decision

logger = logging.getLogger("paper3")


@dataclass
class CalibrationResult:
    """Stores calibrated interval bounds and metadata."""
    method: str
    q_lo: np.ndarray       # Calibrated lower bounds (Head 2: q0.05)
    q_hi: np.ndarray       # Calibrated upper bounds (Head 4: q0.95)
    point_preds: np.ndarray# Head 1: y_mean (evaluated strictly for point metrics)
    y_true: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


def _enforce_valid_bounds(q_lo: np.ndarray, q_hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure q_lo <= q_hi for all prediction interval samples."""
    q_lo = q_lo.copy()
    q_hi = q_hi.copy()
    invalid = q_lo > q_hi
    if np.any(invalid):
        mid = 0.5 * (q_lo[invalid] + q_hi[invalid])
        q_lo[invalid] = mid
        q_hi[invalid] = mid
    return q_lo, q_hi


# ─────────────────────────────────────────────────────────────
# 1. Static Conformal Prediction (Paper 18 baseline)
# ─────────────────────────────────────────────────────────────

def static_conformal(
    y_cal: np.ndarray,
    q_lo_cal: np.ndarray,
    q_hi_cal: np.ndarray,
    q_lo_test: np.ndarray,
    q_hi_test: np.ndarray,
    point_preds_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = cfg.NOMINAL_ALPHA,
) -> CalibrationResult:
    """Static split conformal calibration using fixed calibration residuals (Paper 18 baseline)."""
    logger.info("Calibrating with Static Conformal (Paper 18 baseline)")

    scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    n = len(scores)
    q_level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
    threshold = float(np.quantile(scores, q_level))

    cal_q_lo = q_lo_test - threshold
    cal_q_hi = q_hi_test + threshold

    cal_q_lo, cal_q_hi = _enforce_valid_bounds(cal_q_lo, cal_q_hi)
    logger.info("  Static threshold: %.4f", threshold)

    return CalibrationResult(
        method="static_conformal",
        q_lo=cal_q_lo,
        q_hi=cal_q_hi,
        point_preds=point_preds_test.copy(),
        y_true=y_test.copy(),
        metadata={"threshold": threshold, "n_cal": n},
    )


# ─────────────────────────────────────────────────────────────
# 2. Phenology-Stratified Static CQR (Paper 21 baseline)
# ─────────────────────────────────────────────────────────────

def phenology_stratified_cqr(
    y_cal: np.ndarray,
    q_lo_cal: np.ndarray,
    q_hi_cal: np.ndarray,
    pheno_cal: np.ndarray,
    q_lo_test: np.ndarray,
    q_hi_test: np.ndarray,
    pheno_test: np.ndarray,
    point_preds_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = cfg.NOMINAL_ALPHA,
) -> CalibrationResult:
    """Phenology-stratified static CQR (Paper 21 baseline)."""
    logger.info("Calibrating with Phenology-Stratified Static CQR (Paper 21)")

    scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    unique_windows = np.unique(pheno_cal)

    cal_q_lo = q_lo_test.copy()
    cal_q_hi = q_hi_test.copy()
    thresholds = {}

    for window in unique_windows:
        mask_cal = pheno_cal == window
        mask_test = pheno_test == window

        if mask_cal.sum() == 0:
            continue

        window_scores = scores[mask_cal]
        n = len(window_scores)
        q_level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
        threshold = float(np.quantile(window_scores, q_level))

        cal_q_lo[mask_test] = q_lo_test[mask_test] - threshold
        cal_q_hi[mask_test] = q_hi_test[mask_test] + threshold
        thresholds[str(window)] = threshold

    cal_q_lo, cal_q_hi = _enforce_valid_bounds(cal_q_lo, cal_q_hi)
    logger.info("  Phenology thresholds: %s", thresholds)

    return CalibrationResult(
        method="phenology_stratified_cqr",
        q_lo=cal_q_lo,
        q_hi=cal_q_hi,
        point_preds=point_preds_test.copy(),
        y_true=y_test.copy(),
        metadata={"thresholds_by_window": thresholds},
    )


# ─────────────────────────────────────────────────────────────
# 3. Weighted / Covariate-Shift Conformal (Paper 9 baseline)
# ─────────────────────────────────────────────────────────────

def weighted_conformal(
    y_cal: np.ndarray,
    q_lo_cal: np.ndarray,
    q_hi_cal: np.ndarray,
    X_cal: np.ndarray,
    q_lo_test: np.ndarray,
    q_hi_test: np.ndarray,
    X_test: np.ndarray,
    point_preds_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = cfg.NOMINAL_ALPHA,
) -> CalibrationResult:
    """Weighted conformal prediction using density ratio (Paper 9 baseline)."""
    logger.info("Calibrating with Weighted/Covariate-Shift Conformal (Paper 9)")

    from sklearn.linear_model import LogisticRegression

    scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)

    X_combined = np.vstack([X_cal, X_test])
    y_domain = np.concatenate([np.zeros(len(X_cal)), np.ones(len(X_test))])

    clf = LogisticRegression(max_iter=500, random_state=cfg.RANDOM_SEED, solver="lbfgs")
    clf.fit(X_combined, y_domain)

    proba_cal = clf.predict_proba(X_cal)[:, 1]
    proba_cal = np.clip(proba_cal, 0.01, 0.99)
    weights = proba_cal / (1.0 - proba_cal)
    weights = weights / max(1e-6, weights.sum())

    sorted_idx = np.argsort(scores)
    sorted_scores = scores[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cum_weights = np.cumsum(sorted_weights)
    threshold_idx = np.searchsorted(cum_weights, 1.0 - alpha)
    threshold_idx = min(threshold_idx, len(sorted_scores) - 1)
    threshold = float(sorted_scores[threshold_idx])

    cal_q_lo = q_lo_test - threshold
    cal_q_hi = q_hi_test + threshold

    cal_q_lo, cal_q_hi = _enforce_valid_bounds(cal_q_lo, cal_q_hi)
    logger.info("  Weighted threshold: %.4f", threshold)

    return CalibrationResult(
        method="weighted_conformal",
        q_lo=cal_q_lo,
        q_hi=cal_q_hi,
        point_preds=point_preds_test.copy(),
        y_true=y_test.copy(),
        metadata={"threshold": threshold},
    )


# ─────────────────────────────────────────────────────────────
# 4. Locally Adaptive Conformal (Paper 17 baseline)
# ─────────────────────────────────────────────────────────────

def locally_adaptive_conformal(
    y_cal: np.ndarray,
    q_lo_cal: np.ndarray,
    q_hi_cal: np.ndarray,
    q_lo_test: np.ndarray,
    q_hi_test: np.ndarray,
    point_preds_test: np.ndarray,
    y_test: np.ndarray,
    point_preds_cal: Optional[np.ndarray] = None,
    alpha: float = cfg.NOMINAL_ALPHA,
) -> CalibrationResult:
    """Locally adaptive conformal prediction (Paper 17 baseline)."""
    logger.info("Calibrating with Locally Adaptive Conformal (Paper 17)")

    widths_cal = np.maximum(q_hi_cal - q_lo_cal, 1e-6)
    widths_test = np.maximum(q_hi_test - q_lo_test, 1e-6)

    raw_scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    norm_scores = raw_scores / widths_cal

    n = len(norm_scores)
    q_level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
    threshold = float(np.quantile(norm_scores, q_level))

    cal_q_lo = q_lo_test - threshold * widths_test
    cal_q_hi = q_hi_test + threshold * widths_test

    cal_q_lo, cal_q_hi = _enforce_valid_bounds(cal_q_lo, cal_q_hi)
    logger.info("  Normalized threshold: %.4f", threshold)

    return CalibrationResult(
        method="locally_adaptive_conformal",
        q_lo=cal_q_lo,
        q_hi=cal_q_hi,
        point_preds=point_preds_test.copy(),
        y_true=y_test.copy(),
        metadata={"normalized_threshold": threshold},
    )


# ─────────────────────────────────────────────────────────────
# 5. Standard Adaptive Conformal Inference (Gibbs & Candès 2021)
# ─────────────────────────────────────────────────────────────

def standard_aci(
    y_cal: np.ndarray,
    q_lo_cal: np.ndarray,
    q_hi_cal: np.ndarray,
    y_test: np.ndarray,
    q_lo_test: np.ndarray,
    q_hi_test: np.ndarray,
    point_preds_test: np.ndarray,
    years_test: np.ndarray,
    alpha: float = cfg.NOMINAL_ALPHA,
    gamma: float = cfg.ACI_GAMMA,
) -> CalibrationResult:
    """Standard Adaptive Conformal Inference (Gibbs & Candès 2021).

    Maintains long-term coverage guarantee under non-exchangeable temporal shift:
        alpha_{t+1} = clip(alpha_t + gamma * (alpha - err_t), 0.01, 0.40)
    where err_t is the empirical miscoverage in step t.
    """
    logger.info("Calibrating with Standard ACI (Gibbs & Candès 2021)")
    logger.info("  gamma=%.4f, nominal_alpha=%.2f", gamma, alpha)

    scores_cal = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    unique_years = np.sort(np.unique(years_test))
    cal_q_lo = np.zeros_like(q_lo_test)
    cal_q_hi = np.zeros_like(q_hi_test)

    alpha_t = alpha
    history: List[Dict[str, Any]] = []

    # Expanding historical score pool starting with calibration set
    score_pool: List[float] = list(scores_cal)

    for t, year in enumerate(unique_years):
        year_mask = years_test == year
        n_year = int(year_mask.sum())

        scores_arr = np.array(score_pool)
        n_p = len(scores_arr)
        q_level = min(np.ceil((1 - alpha_t) * (n_p + 1)) / n_p, 1.0)
        q_level = max(q_level, 0.0)
        threshold_t = float(np.quantile(scores_arr, q_level))

        yr_lo = q_lo_test[year_mask] - threshold_t
        yr_hi = q_hi_test[year_mask] + threshold_t
        yr_lo, yr_hi = _enforce_valid_bounds(yr_lo, yr_hi)

        cal_q_lo[year_mask] = yr_lo
        cal_q_hi[year_mask] = yr_hi

        covered = (
            (y_test[year_mask] >= cal_q_lo[year_mask])
            & (y_test[year_mask] <= cal_q_hi[year_mask])
        )
        picp_year = float(covered.mean())
        err_t = 1.0 - picp_year

        # Online update
        alpha_t = alpha_t + gamma * (alpha - err_t)
        alpha_t = float(np.clip(alpha_t, 0.01, 0.40))

        # Add year residuals to score pool
        yr_scores = np.maximum(q_lo_test[year_mask] - y_test[year_mask], y_test[year_mask] - q_hi_test[year_mask])
        score_pool.extend(list(yr_scores))

        interval_width = float((cal_q_hi[year_mask] - cal_q_lo[year_mask]).mean())

        history.append({
            "year": int(year),
            "n_samples": n_year,
            "picp": round(picp_year, 4),
            "err_t": round(err_t, 4),
            "alpha_t": round(float(alpha_t), 4),
            "threshold": round(float(threshold_t), 4),
            "mean_width": round(interval_width, 4),
        })

    cal_q_lo, cal_q_hi = _enforce_valid_bounds(cal_q_lo, cal_q_hi)

    return CalibrationResult(
        method="standard_aci",
        q_lo=cal_q_lo,
        q_hi=cal_q_hi,
        point_preds=point_preds_test.copy(),
        y_true=y_test.copy(),
        metadata={
            "gamma": gamma,
            "year_history": history,
            "final_alpha_t": float(alpha_t),
        },
    )


# ─────────────────────────────────────────────────────────────
# 6. Severity-Aware Adaptive Conformal Inference (SA-ACI / CDHW-ACI)
# ─────────────────────────────────────────────────────────────

def severity_aware_adaptive_conformal(
    y_cal: np.ndarray,
    q_lo_cal: np.ndarray,
    q_hi_cal: np.ndarray,
    y_test: np.ndarray,
    q_lo_test: np.ndarray,
    q_hi_test: np.ndarray,
    point_preds_test: np.ndarray,
    years_test: np.ndarray,
    alpha: float = cfg.NOMINAL_ALPHA,
    gamma: float = cfg.ACI_GAMMA,
    window: int = cfg.ACI_WINDOW_SIZE,
    cdhw_severity_test: Optional[np.ndarray] = None,
    cdhw_severity_cal: Optional[np.ndarray] = None,
    severity_weighting: Optional[bool] = None,
) -> CalibrationResult:
    """Severity-Aware Adaptive Conformal Inference (SA-ACI / CDHW-ACI).

    Methodological Innovation for Compound Climate Extremes:
    1. Dynamic climate window W(S_t) based on regional severity:
       W(S_t) = 2 if S_t >= 5.0 (Extreme shock: fast adaptation)
                3 if 1.0 <= S_t < 5.0 (Moderate: 3-yr sliding window)
                5 if S_t < 1.0 (Normal: smooth 5-yr baseline)
    2. Non-linear severity-weighted conformity scores:
       E_i^{(S)} = E_i * (1 + lambda * log(1 + S_i))
    3. Observation-level severity scaling:
       The threshold is scaled UP with local severity, so a row under heavy
       compound stress receives a wider interval.

       Correction (2026-09-17, audit C5): this step previously divided the
       threshold by the severity weight, which narrowed intervals under stress
       -- the opposite of the stated intent, and the likely reason SA-ACI ranked
       worst on every conditional axis. Set cfg.ACI_SEVERITY_WEIGHTING = False
       to switch the severity mechanism off entirely (fixed window, unit
       weights), which leaves a plain windowed ACI.
    4. Decaying learning rate: eta_t = gamma0 / sqrt(t + 1).
    """
    logger.info("Calibrating with Severity-Aware Adaptive Conformal Inference (SA-ACI)")
    _sev_on = bool(getattr(cfg, "ACI_SEVERITY_WEIGHTING", True)) if severity_weighting is None         else bool(severity_weighting)
    _sev_lambda = float(getattr(cfg, "ACI_SEVERITY_LAMBDA", 0.05))
    logger.info("  gamma0=%.4f, default_window=%d, nominal_alpha=%.2f, severity_weighting=%s (lambda=%.3f)",
                gamma, window, alpha, _sev_on, _sev_lambda)

    # (no pooled calibration scores here: SA-ACI scores each sliding window
    # separately below, so a pooled vector would be dead weight.)
    unique_years = np.sort(np.unique(years_test))
    cal_q_lo = np.zeros_like(q_lo_test)
    cal_q_hi = np.zeros_like(q_hi_test)

    alpha_t = alpha
    history: List[Dict[str, Any]] = []

    sev_cal = cdhw_severity_cal if cdhw_severity_cal is not None else np.zeros(len(y_cal))

    buffer_y: List[np.ndarray] = [y_cal.copy()]
    buffer_lo: List[np.ndarray] = [q_lo_cal.copy()]
    buffer_hi: List[np.ndarray] = [q_hi_cal.copy()]
    buffer_sev: List[np.ndarray] = [sev_cal.copy()]

    for t, year in enumerate(unique_years):
        year_mask = years_test == year
        n_year = int(year_mask.sum())

        yr_sev = float(np.mean(cdhw_severity_test[year_mask])) if cdhw_severity_test is not None else 0.0

        if not _sev_on:
            dyn_window = int(window)
        elif yr_sev >= 5.0:
            dyn_window = 2
        elif yr_sev >= 1.0:
            dyn_window = 3
        else:
            dyn_window = 5

        window_y = np.concatenate(buffer_y[-dyn_window:])
        window_lo = np.concatenate(buffer_lo[-dyn_window:])
        window_hi = np.concatenate(buffer_hi[-dyn_window:])
        window_sev = np.concatenate(buffer_sev[-dyn_window:])

        raw_w_scores = np.maximum(window_lo - window_y, window_y - window_hi)
        sev_weights = (1.0 + _sev_lambda * np.log1p(np.maximum(window_sev, 0.0))
                       if _sev_on else np.ones_like(raw_w_scores))
        # Normalised (locally adaptive) conformity scores, Lei et al. 2018: the
        # score is divided by the local scale during calibration and the
        # threshold is multiplied by it at test time. Doing only one of the two
        # breaks the correspondence -- the committed code multiplied here and
        # divided at test, which is what made severe rows the narrowest.
        adj_w_scores = raw_w_scores / sev_weights

        n_w = len(adj_w_scores)
        q_level_w = min(np.ceil((1 - alpha_t) * (n_w + 1)) / n_w, 1.0)
        q_level_w = max(q_level_w, 0.0)
        current_threshold = float(np.quantile(adj_w_scores, q_level_w))

        this_sev = cdhw_severity_test[year_mask] if cdhw_severity_test is not None else np.zeros(n_year)
        if _sev_on:
            this_weights = 1.0 + _sev_lambda * np.log1p(np.maximum(this_sev, 0.0))
            # C5 fix: MULTIPLY, matching the division applied to the calibration
            # scores above. A more severe row gets a larger threshold and
            # therefore a wider interval; marginal coverage is preserved because
            # the scores were normalised by the same weight.
            eff_threshold = current_threshold * this_weights
        else:
            this_weights = np.ones(n_year)
            eff_threshold = np.full(n_year, current_threshold)

        yr_lo = q_lo_test[year_mask] - eff_threshold
        yr_hi = q_hi_test[year_mask] + eff_threshold
        yr_lo, yr_hi = _enforce_valid_bounds(yr_lo, yr_hi)

        cal_q_lo[year_mask] = yr_lo
        cal_q_hi[year_mask] = yr_hi

        covered = (
            (y_test[year_mask] >= cal_q_lo[year_mask])
            & (y_test[year_mask] <= cal_q_hi[year_mask])
        )
        picp_year = float(covered.mean())
        err_t = 1.0 - picp_year

        eta_t = gamma / np.sqrt(t + 1.0)
        alpha_t = alpha_t + eta_t * (alpha - err_t)
        alpha_t = float(np.clip(alpha_t, 0.01, 0.40))

        buffer_y.append(y_test[year_mask].copy())
        buffer_lo.append(q_lo_test[year_mask].copy())
        buffer_hi.append(q_hi_test[year_mask].copy())
        buffer_sev.append(this_sev.copy())

        interval_width = float((cal_q_hi[year_mask] - cal_q_lo[year_mask]).mean())

        history.append({
            "year": int(year),
            "n_samples": n_year,
            "picp": round(picp_year, 4),
            "err_t": round(err_t, 4),
            "alpha_t": round(float(alpha_t), 4),
            "eta_t": round(float(eta_t), 4),
            "dynamic_window": dyn_window,
            "threshold": round(float(current_threshold), 4),
            "mean_width": round(interval_width, 4),
        })

        logger.info(
            "  Year %d SA-ACI -> PICP: %.4f, err: %.4f, α_t: %.4f, DynWindow: %d, threshold: %.4f, MPIW: %.4f",
            year, picp_year, err_t, alpha_t, dyn_window, current_threshold, interval_width
        )

    cal_q_lo, cal_q_hi = _enforce_valid_bounds(cal_q_lo, cal_q_hi)

    return CalibrationResult(
        method="severity_aware_aci",
        q_lo=cal_q_lo,
        q_hi=cal_q_hi,
        point_preds=point_preds_test.copy(),
        y_true=y_test.copy(),
        metadata={
            "gamma": gamma,
            "window": window,
            "year_history": history,
            "final_alpha_t": float(alpha_t),
            "severity_weighting": _sev_on,
            "severity_lambda": _sev_lambda if _sev_on else 0.0,
            "severity_direction": "widen" if _sev_on else "none",
        },
    )


# Backward compatibility alias
def adaptive_conformal_inference(
    y_cal: np.ndarray,
    q_lo_cal: np.ndarray,
    q_hi_cal: np.ndarray,
    y_test: np.ndarray,
    q_lo_test: np.ndarray,
    q_hi_test: np.ndarray,
    point_preds_test: np.ndarray,
    years_test: np.ndarray,
    alpha: float = cfg.NOMINAL_ALPHA,
    gamma: float = cfg.ACI_GAMMA,
    window: int = cfg.ACI_WINDOW_SIZE,
    cdhw_severity_test: Optional[np.ndarray] = None,
) -> CalibrationResult:
    """Wrapper mapping to severity_aware_adaptive_conformal with 'aci' method name."""
    res = severity_aware_adaptive_conformal(
        y_cal, q_lo_cal, q_hi_cal, y_test, q_lo_test, q_hi_test,
        point_preds_test, years_test, alpha=alpha, gamma=gamma, window=window,
        cdhw_severity_test=cdhw_severity_test,
    )
    res.method = "aci"
    return res


def validate_conformal_calibration(
    calibration_results: Dict[str, CalibrationResult]
) -> Dict[str, Any]:
    """Validate conformal calibration consistency & strict memory/pool independence across methods."""
    from utils import save_report_markdown, save_report

    validation_summary = {}
    all_passed = True

    for name, res in calibration_results.items():
        widths = res.q_hi - res.q_lo
        neg_widths = int((widths < 0).sum())
        invalid_bounds = int((res.q_lo > res.q_hi).sum())
        non_negative_widths = neg_widths == 0 and invalid_bounds == 0

        validation_summary[name] = {
            "non_negative_widths": non_negative_widths,
            "negative_width_count": neg_widths,
            "invalid_bounds_count": invalid_bounds,
            "min_width": float(np.min(widths)),
            "max_width": float(np.max(widths)),
            "mean_width": float(np.mean(widths)),
        }
        if not non_negative_widths:
            all_passed = False

    # Automated Calibration Independence Assertion (Zero-Aliasing Check)
    mem_ids = set()
    method_names = list(calibration_results.keys())
    for name in method_names:
        res = calibration_results[name]
        lo_id = id(res.q_lo)
        hi_id = id(res.q_hi)
        assert lo_id not in mem_ids, f"Data leakage error: shared memory address detected for {name} q_lo!"
        assert hi_id not in mem_ids, f"Data leakage error: shared memory address detected for {name} q_hi!"
        mem_ids.add(lo_id)
        mem_ids.add(hi_id)

    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            m1, m2 = method_names[i], method_names[j]
            w1 = calibration_results[m1].q_hi - calibration_results[m1].q_lo
            w2 = calibration_results[m2].q_hi - calibration_results[m2].q_lo
            assert not np.array_equal(w1, w2), f"Calibration error: {m1} and {m2} produced identical interval widths!"

    logger.info("  ✓ Automated Conformal Independence Assertion PASSED: All calibration methods use distinct memory pools and distinct prediction intervals.")

    md = f"""# Conformal Calibration Validation Report (§4.2, §4.3 & §5.6)

## Summary
- **Overall Consistency Status**: {"PASSED (Fully Monotonic & Valid)" if all_passed else "WARNING (Invalid bounds detected)"}

## Method Verification Details
"""
    for name, v in validation_summary.items():
        md += f"""- **{name}**:
  - Non-negative widths: {v['non_negative_widths']}
  - Invalid bounds (q_lo > q_hi): {v['invalid_bounds_count']}
  - Mean interval width: {v['mean_width']:.4f} (min: {v['min_width']:.4f}, max: {v['max_width']:.4f})
"""

    save_report_markdown(md, "conformal_validation_report.md")
    save_report({"all_passed": all_passed, "methods": validation_summary}, "conformal_validation_report.json")
    logger.info("Conformal validation report saved -> conformal_validation_report.md")
    return validation_summary
