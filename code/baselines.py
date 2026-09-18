"""Baselines required by audit §17 C4.

Two of them decide how the paper's central claim reads:

* ``trend_county_mean`` — a linear FIT-only trend plus a county mean. If the
  model cannot beat this, the machine learning is not earning its place.
* ``group_conditional_cp`` — Mondrian / group-conditional conformal prediction
  with **pre-specified** groups (Vovk 2013; Gibbs, Cherian & Candès 2025). This
  is the published method that directly targets the gap the paper reports. If it
  closes the exposure gap, the paper's message becomes "use group-conditional
  calibration under stress"; if it does not, the gap is a property of the data
  rather than of the calibrator.

Neither reads TEST years or a held-out state for any choice.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import config as cfg

logger = logging.getLogger("paper3")


# ─────────────────────────────────────────────────────────────
# Point-prediction baselines
# ─────────────────────────────────────────────────────────────

def trend_county_mean(
    fit_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_col: str = cfg.PRIMARY_TARGET,
    county_col: str = "GEOID",
) -> np.ndarray:
    """Linear year trend fitted on FIT, plus each county's FIT-only residual mean.

    Counties absent from FIT fall back to the trend alone, which is what happens
    for every county in a leave-one-state-out fold.
    """
    coef = np.polyfit(fit_df["Year"].values, fit_df[target_col].values, 1)
    fit_resid = fit_df[target_col].values - np.polyval(coef, fit_df["Year"].values)
    county_mean = pd.Series(fit_resid, index=fit_df[county_col].values).groupby(level=0).mean()

    trend = np.polyval(coef, target_df["Year"].values)
    offset = target_df[county_col].map(county_mean).fillna(0.0).values
    return trend + offset


def trend_only(
    fit_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_col: str = cfg.PRIMARY_TARGET,
) -> np.ndarray:
    """Pooled linear year trend fitted on FIT only."""
    coef = np.polyfit(fit_df["Year"].values, fit_df[target_col].values, 1)
    return np.polyval(coef, target_df["Year"].values)


# ─────────────────────────────────────────────────────────────
# Group-conditional (Mondrian) conformal prediction
# ─────────────────────────────────────────────────────────────

def group_conditional_cp(
    y_cal: np.ndarray,
    q_lo_cal: np.ndarray,
    q_hi_cal: np.ndarray,
    groups_cal: np.ndarray,
    q_lo_test: np.ndarray,
    q_hi_test: np.ndarray,
    groups_test: np.ndarray,
    alpha: float = 1.0 - cfg.NOMINAL_COVERAGE,
    min_group_n: int = 20,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Mondrian conformal prediction with pre-specified groups.

    A separate conformal threshold is computed inside each group, so coverage is
    targeted *within* each group instead of only on average. Groups must be
    fixed in advance (here: the CDHW exposure class), never chosen after seeing
    test outcomes.

    Groups with fewer than ``min_group_n`` calibration rows fall back to the
    pooled threshold, and the fallback is recorded — with a small calibration
    group the finite-sample guarantee is too weak to be worth claiming.
    """
    scores_cal = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)

    def _threshold(scores: np.ndarray) -> float:
        n = len(scores)
        if n == 0:
            return float("nan")
        level = min(np.ceil((1.0 - alpha) * (n + 1)) / n, 1.0)
        return float(np.quantile(scores, max(level, 0.0)))

    pooled = _threshold(scores_cal)
    per_group: Dict[str, Any] = {}
    thresholds = np.full(len(q_lo_test), pooled, dtype=float)

    for g in np.unique(groups_test):
        mask_cal = groups_cal == g
        n_cal = int(mask_cal.sum())
        if n_cal >= min_group_n:
            t = _threshold(scores_cal[mask_cal])
            fallback = False
        else:
            t, fallback = pooled, True
        thresholds[groups_test == g] = t
        per_group[str(g)] = {
            "n_cal": n_cal,
            "n_test": int((groups_test == g).sum()),
            "threshold": round(float(t), 4),
            "fell_back_to_pooled": fallback,
        }

    lo = q_lo_test - thresholds
    hi = q_hi_test + thresholds
    cross = hi < lo
    if cross.any():
        mid = 0.5 * (lo[cross] + hi[cross])
        lo[cross], hi[cross] = mid, mid

    meta = {
        "method": "group_conditional_cp",
        "reference": "Vovk (2013) Mondrian CP; Gibbs, Cherian & Candes (2025)",
        "alpha": alpha,
        "pooled_threshold": round(float(pooled), 4),
        "min_group_n": min_group_n,
        "groups": per_group,
    }
    logger.info("Group-conditional CP: %d groups, thresholds %s",
                len(per_group), {k: v["threshold"] for k, v in per_group.items()})
    return lo, hi, meta


def rolling_static_cp(
    y_cal: np.ndarray,
    q_lo_cal: np.ndarray,
    q_hi_cal: np.ndarray,
    q_lo_test: np.ndarray,
    q_hi_test: np.ndarray,
    alpha: float = 1.0 - cfg.NOMINAL_COVERAGE,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Split conformal on the calibration rows supplied for this origin.

    Named "rolling" because the caller re-fits it at every rolling origin, which
    is what makes it comparable to ACI: both then see the same information.
    """
    scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    n = len(scores)
    level = min(np.ceil((1.0 - alpha) * (n + 1)) / n, 1.0) if n else 0.0
    t = float(np.quantile(scores, max(level, 0.0))) if n else 0.0
    lo, hi = q_lo_test - t, q_hi_test + t
    cross = hi < lo
    if cross.any():
        mid = 0.5 * (lo[cross] + hi[cross])
        lo[cross], hi[cross] = mid, mid
    return lo, hi, {"method": "rolling_static_cp", "threshold": round(t, 4), "n_cal": int(n)}
