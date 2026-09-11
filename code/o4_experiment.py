"""
o4_experiment.py - Objective O4: joint end-to-end Neural CQR vs post-hoc CQR.

What was wrong with the previous O4
-----------------------------------
``evaluate_objective_o4`` was handed one *calibration method* (SA-ACI) as the
"joint" arm and other *calibration methods* (static, phenology) as the
"post-hoc" arms, then reported a percentage RMSE improvement between them. All
of those share the same underlying point predictions by construction, so the
RMSE difference was identically zero and the comparison could never have
detected a training-paradigm effect in either direction. Conformal calibration
changes intervals, not point predictions -- so a calibration-vs-calibration
contrast is the wrong instrument for this question.

What this module does instead
-----------------------------
It runs the actual research question as a controlled paired experiment:

    Does genuine joint end-to-end Neural CQR outperform the intended post-hoc
    approach?

Both arms are held identical on everything except the training paradigm:

* same FIT / DEV / CAL / TEST partitions,
* same feature set and same fitted scaler,
* same architecture and capacity (hidden dims, dropout),
* same optimizer, learning rate, weight decay, batch size,
* same epoch budget and same early-stopping patience on the same DEV data,
* same random seed,
* same conformal calibration applied afterwards to both arms.

The only difference is the code path inside ``train_neural_cqr``:
``joint_training=True`` optimizes the composite objective end-to-end so quantile
gradients reach the shared backbone; ``joint_training=False`` trains the point
model alone, freezes it, then fits quantile heads on the fixed representation.

Significance is assessed with year-block cluster bootstrap and a year-level
paired test, because the county-year test rows are not independent.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import config as cfg

logger = logging.getLogger("paper3")


def run_o4_joint_vs_posthoc(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_cal: np.ndarray,
    y_cal_raw: np.ndarray,
    X_test: np.ndarray,
    y_test_raw: np.ndarray,
    feature_cols: List[str],
    scaler: Any,
    cal_trend: Optional[np.ndarray],
    test_trend: Optional[np.ndarray],
    years_test: np.ndarray,
    detrended: bool,
    epochs: int,
    seed: int = cfg.RANDOM_SEED,
) -> Dict[str, Any]:
    """Train both paradigms under matched conditions and evaluate both arms."""
    from model_training import train_neural_cqr, predict_intervals
    from aci_calibrator import static_conformal, standard_aci
    from evaluation import rmse, mae, r_squared, picp, mpiw, ace, winkler_score

    logger.info("=" * 60)
    logger.info("OBJECTIVE O4: JOINT END-TO-END vs POST-HOC (matched conditions)")
    logger.info("=" * 60)

    zero_cal = np.zeros_like(y_cal_raw)
    zero_test = np.zeros_like(y_test_raw)
    _cal_trend = cal_trend if (detrended and cal_trend is not None) else zero_cal
    _test_trend = test_trend if (detrended and test_trend is not None) else zero_test

    shared_config = {
        "epochs": epochs,
        "batch_size": int(cfg.BATCH_SIZE),
        "lr": float(cfg.LEARNING_RATE),
        "weight_decay": float(cfg.WEIGHT_DECAY),
        "dropout_rate": float(cfg.NEURAL_CQR_DROPOUT),
        "hidden_dims": list(cfg.NEURAL_CQR_HIDDEN_DIMS),
        "patience": int(cfg.EARLY_STOPPING_PATIENCE),
        "seed": int(seed),
        "n_features": len(feature_cols),
        "n_fit": int(len(X_fit)),
        "n_dev": int(len(X_dev)),
        "n_cal": int(len(X_cal)),
        "n_test": int(len(X_test)),
    }

    arms: Dict[str, Any] = {}
    per_obs: Dict[str, Dict[str, np.ndarray]] = {}

    for arm_name, is_joint in (("post_hoc", False), ("joint", True)):
        logger.info("--- O4 arm: %s (joint_training=%s) ---", arm_name, is_joint)
        models = train_neural_cqr(
            X_fit, y_fit, X_dev, y_dev, feature_cols,
            epochs=epochs,
            batch_size=cfg.BATCH_SIZE,
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY,
            dropout_rate=cfg.NEURAL_CQR_DROPOUT,
            hidden_dims=cfg.NEURAL_CQR_HIDDEN_DIMS,
            lambda_pinball=cfg.LAMBDA_PINBALL,
            lambda_huber=cfg.LAMBDA_HUBER,
            lambda_crossing=cfg.LAMBDA_CROSSING,
            lambda_width=cfg.LAMBDA_WIDTH,
            early_stopping_mode=cfg.EARLY_STOPPING_MODE,
            patience=cfg.EARLY_STOPPING_PATIENCE,
            joint_training=is_joint,
            scaler=scaler,
            seed=seed,
        )

        p_cal, qlo_cal, qhi_cal = predict_intervals(models, X_cal)
        p_te, qlo_te, qhi_te = predict_intervals(models, X_test)
        p_cal = p_cal + _cal_trend
        qlo_cal = qlo_cal + _cal_trend
        qhi_cal = qhi_cal + _cal_trend
        p_te = p_te + _test_trend
        qlo_te = qlo_te + _test_trend
        qhi_te = qhi_te + _test_trend

        # Identical calibration applied to both arms.
        static_res = static_conformal(
            y_cal_raw, qlo_cal, qhi_cal, qlo_te, qhi_te, p_te, y_test_raw,
            alpha=cfg.NOMINAL_ALPHA,
        )
        aci_res = standard_aci(
            y_cal_raw, qlo_cal, qhi_cal, y_test_raw, qlo_te, qhi_te,
            p_te, years_test, alpha=cfg.NOMINAL_ALPHA, gamma=cfg.ACI_GAMMA,
        )

        def _unc(res):
            return {
                "picp": round(float(picp(y_test_raw, res.q_lo, res.q_hi)), 4),
                "mpiw": round(float(mpiw(res.q_lo, res.q_hi)), 4),
                "ace": round(float(ace(y_test_raw, res.q_lo, res.q_hi)), 4),
                "winkler_score": round(float(winkler_score(y_test_raw, res.q_lo, res.q_hi)), 4),
            }

        fingerprint = getattr(models, "paradigm_fingerprint", None) or {}
        arms[arm_name] = {
            "training_paradigm": getattr(models, "training_paradigm", None),
            "paradigm_fingerprint": fingerprint,
            "epochs_trained": int(getattr(models, "epochs_trained", 0)),
            "best_epoch": int(getattr(models, "best_epoch", 0)),
            "point_prediction": {
                "rmse": round(float(rmse(y_test_raw, p_te)), 4),
                "mae": round(float(mae(y_test_raw, p_te)), 4),
                "r_squared": round(float(r_squared(y_test_raw, p_te)), 4),
            },
            "uncertainty_static_conformal": _unc(static_res),
            "uncertainty_aci": _unc(aci_res),
            "raw_interval_mpiw_before_calibration": round(float(np.mean(qhi_te - qlo_te)), 4),
            "prediction_checksum": round(float(np.sum(np.abs(p_te))), 6),
        }

        per_obs[arm_name] = {
            "point_pred": np.asarray(p_te),
            "sq_err": (np.asarray(y_test_raw) - np.asarray(p_te)) ** 2,
            "abs_err": np.abs(np.asarray(y_test_raw) - np.asarray(p_te)),
            "static_width": np.asarray(static_res.q_hi) - np.asarray(static_res.q_lo),
            "static_winkler": _winkler_per_obs(y_test_raw, static_res.q_lo, static_res.q_hi),
            "aci_width": np.asarray(aci_res.q_hi) - np.asarray(aci_res.q_lo),
            "aci_winkler": _winkler_per_obs(y_test_raw, aci_res.q_lo, aci_res.q_hi),
        }

    # ── Prove the two arms really are different computations ─────────────
    identical_points = bool(np.allclose(per_obs["joint"]["point_pred"],
                                        per_obs["post_hoc"]["point_pred"]))
    distinctness = {
        "identical_point_predictions": identical_points,
        "max_abs_point_difference": round(
            float(np.max(np.abs(per_obs["joint"]["point_pred"] - per_obs["post_hoc"]["point_pred"]))), 6),
        "joint_paradigm": arms["joint"]["training_paradigm"],
        "post_hoc_paradigm": arms["post_hoc"]["training_paradigm"],
        "joint_stages": arms["joint"]["paradigm_fingerprint"].get("n_optimization_stages"),
        "post_hoc_stages": arms["post_hoc"]["paradigm_fingerprint"].get("n_optimization_stages"),
        "joint_backbone_sees_quantile_gradient":
            arms["joint"]["paradigm_fingerprint"].get("backbone_receives_quantile_gradient"),
        "post_hoc_backbone_sees_quantile_gradient":
            arms["post_hoc"]["paradigm_fingerprint"].get("backbone_receives_quantile_gradient"),
        "status": "FAIL — arms are computationally identical" if identical_points else "PASS",
    }
    if identical_points:
        logger.error("O4 arms produced identical point predictions — the comparison is invalid.")

    # ── Dependence-aware paired significance ─────────────────────────────
    from dependence_aware_stats import (
        cluster_bootstrap_paired_difference, year_level_paired_test,
    )
    yrs = np.asarray(years_test)
    significance = {}
    for label, key in (("squared_error", "sq_err"),
                       ("absolute_error", "abs_err"),
                       ("static_conformal_winkler", "static_winkler"),
                       ("static_conformal_width", "static_width"),
                       ("aci_winkler", "aci_winkler"),
                       ("aci_width", "aci_width")):
        significance[label] = {
            "year_block_cluster_bootstrap": cluster_bootstrap_paired_difference(
                per_obs["joint"][key], per_obs["post_hoc"][key], yrs,
                n_boot=cfg.BOOTSTRAP_ITERATIONS, seed=seed,
                label_a="joint", label_b="post_hoc", cluster_unit="year",
            ),
            "year_level_paired_test": year_level_paired_test(
                per_obs["joint"][key], per_obs["post_hoc"][key], yrs,
                label_a="joint", label_b="post_hoc",
            ),
        }

    # ── Absolute and relative differences ────────────────────────────────
    def _delta(metric_path: List[str], lower_is_better: bool = True) -> Dict[str, Any]:
        j, p = arms["joint"], arms["post_hoc"]
        for k in metric_path:
            j, p = j[k], p[k]
        absolute = float(j) - float(p)
        rel = 100.0 * absolute / abs(p) if abs(p) > 1e-12 else None
        improved = (absolute < 0) if lower_is_better else (absolute > 0)
        return {
            "joint": round(float(j), 4),
            "post_hoc": round(float(p), 4),
            "absolute_difference_joint_minus_posthoc": round(absolute, 4),
            "relative_difference_pct": round(rel, 2) if rel is not None else None,
            "joint_better": bool(improved),
            "lower_is_better": lower_is_better,
        }

    comparison = {
        "rmse": _delta(["point_prediction", "rmse"]),
        "mae": _delta(["point_prediction", "mae"]),
        "r_squared": _delta(["point_prediction", "r_squared"], lower_is_better=False),
        "picp_static": _delta(["uncertainty_static_conformal", "picp"], lower_is_better=False),
        "mpiw_static": _delta(["uncertainty_static_conformal", "mpiw"]),
        "ace_static": _delta(["uncertainty_static_conformal", "ace"]),
        "winkler_static": _delta(["uncertainty_static_conformal", "winkler_score"]),
        "picp_aci": _delta(["uncertainty_aci", "picp"], lower_is_better=False),
        "mpiw_aci": _delta(["uncertainty_aci", "mpiw"]),
        "ace_aci": _delta(["uncertainty_aci", "ace"]),
        "winkler_aci": _delta(["uncertainty_aci", "winkler_score"]),
    }

    return {
        "design": {
            "question": ("Does genuine joint end-to-end Neural CQR provide an advantage "
                         "over the intended post-hoc approach?"),
            "controlled_for": [
                "identical FIT/DEV/CAL/TEST partitions",
                "identical feature set and fitted scaler",
                "identical architecture and capacity",
                "identical optimizer, lr, weight decay, batch size",
                "identical epoch budget and early-stopping patience on identical DEV data",
                "identical random seed",
                "identical conformal calibration applied to both arms",
            ],
            "only_difference": "training paradigm (joint end-to-end vs post-hoc two-stage)",
            "shared_configuration": shared_config,
        },
        "arms": arms,
        "distinctness_check": distinctness,
        "comparison": comparison,
        "significance_dependence_aware": significance,
    }


def _winkler_per_obs(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                     alpha: float = cfg.NOMINAL_ALPHA) -> np.ndarray:
    """Per-observation Winkler (interval) score."""
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    width = hi - lo
    score = width.copy()
    below = y < lo
    above = y > hi
    score[below] += (2.0 / alpha) * (lo[below] - y[below])
    score[above] += (2.0 / alpha) * (y[above] - hi[above])
    return score
