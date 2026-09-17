"""p4_neural_and_temporal.py - Phases 10 + 13 + the NeuralCQR half of Phase 4.

Answers, by measurement rather than assumption:
  * Why is temporal LightGBM R2 ~= 0.73 while LOSO LightGBM DEV R2 is negative?
  * Is the NeuralCQR DEV advantage on LOSO a level-tracking difference or a
    ranking difference?
  * Do the temporal and LOSO NeuralCQR paths use the same training regime?

Everything runs through the pipeline's own functions. Nothing is modified and
no result here is used to select anything -- DEV is the pipeline's own
selection partition, and the held-out state is only ever *reported*.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from _fold_lab import build_loso_fold, build_temporal, describe, cfg, DIAG_OUT

from model_training import train_lgbm_quantile, train_neural_cqr, predict_intervals
from evaluation import rmse, mae, r_squared


def _pred_stats(y_true, p, tag):
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    bias = float(np.mean(p - y_true))
    mse = float(np.mean((p - y_true) ** 2))
    return {
        "tag": tag,
        "r2": round(float(r_squared(y_true, p)), 4),
        "rmse": round(float(rmse(y_true, p)), 4),
        "mae": round(float(mae(y_true, p)), 4),
        "mean_bias": round(bias, 4),
        "bias_share_of_mse": round(float(bias ** 2 / mse), 4),
        "r2_after_removing_mean_bias": round(float(r_squared(y_true, p - bias)), 4),
        "pearson_r": round(float(np.corrcoef(p, y_true)[0, 1]), 4),
        "pred_mean": round(float(p.mean()), 4),
        "truth_mean": round(float(y_true.mean()), 4),
        "pred_std_over_truth_std": round(float(p.std() / y_true.std()), 4),
    }


def loso_neural(state: str) -> dict:
    """NeuralCQR on one LOSO fold, with main._run_loso_cv's exact arguments (line 1612)."""
    fold = build_loso_fold(state)
    t0 = time.time()
    models = train_neural_cqr(
        fold["X"]["tr_fit"], fold["y"]["tr_fit"], fold["X"]["es_val"], fold["y"]["es_val"],
        fold["feature_cols"],
        epochs=cfg.LOSO_MAX_EPOCHS, batch_size=cfg.LOSO_BATCH_SIZE, lr=cfg.LOSO_LEARNING_RATE,
        weight_decay=cfg.LOSO_WEIGHT_DECAY, early_stopping_mode=cfg.LOSO_EARLY_STOPPING_MODE,
        patience=cfg.LOSO_EARLY_STOPPING_PATIENCE, joint_training=True, scaler=fold["scaler"],
        hidden_dims=cfg.LOSO_HIDDEN_DIMS, dropout_rate=cfg.LOSO_DROPOUT,
    )
    p_dev, _, _ = predict_intervals(models, fold["X"]["dev"])
    p_te, _, _ = predict_intervals(models, fold["X"]["test"])
    y_dev = fold["y"]["dev"]
    yrs = fold["years"]["dev"]

    by_year = {}
    for yv in sorted(set(int(v) for v in yrs)):
        m = yrs == yv
        by_year[yv] = {"n": int(m.sum()), "y_mean": round(float(np.mean(y_dev[m])), 4),
                       "pred_mean": round(float(p_dev[m].mean()), 4),
                       "bias": round(float(np.mean(p_dev[m] - y_dev[m])), 4),
                       "r2_within_year": round(float(r_squared(y_dev[m], p_dev[m])), 4)}
    return {
        "state": state,
        "train_seconds": round(time.time() - t0, 1),
        "epochs_trained": int(getattr(models, "epochs_trained", -1) or -1),
        "best_epoch": int(getattr(models, "best_epoch", -1) or -1),
        "dev": _pred_stats(y_dev, p_dev, "NeuralCQR DEV"),
        "dev_pred_distribution": describe(p_dev, "NeuralCQR DEV prediction"),
        "test": _pred_stats(fold["y"]["test"], p_te, "NeuralCQR held-out-state TEST (uncalibrated point)"),
        "dev_by_year": by_year,
        "pct_dev_pred_above_train_target_max": round(
            100.0 * float((p_dev > np.max(fold["y"]["tr_fit"])).mean()), 3),
    }


def temporal_lightgbm() -> dict:
    """Phase 10. The temporal LightGBM path (main.py:384), plus the same path with
    the ONE variable that differs from LOSO flipped: detrending off."""
    tp = build_temporal()
    X, y, ydet, trend = tp["X"], tp["y"], tp["y_detrended"], tp["trend"]
    cols = tp["feature_cols"]
    out = {"n_features": len(cols), "trend_coef_t_ha_per_year": round(float(tp["trend_coef"][0]), 4),
           "partition_sizes": {k: int(len(v)) for k, v in y.items()}}

    # A. exactly as main.py runs it: fit on ALL of FIT, detrended target
    m_det = train_lgbm_quantile(X["fit"], ydet["fit"], X["dev"], ydet["dev"], cols)
    p_dev_det, _, _ = predict_intervals(m_det, X["dev"])
    p_test_det, _, _ = predict_intervals(m_det, X["test"])
    out["A_temporal_as_shipped_detrended"] = {
        "dev": _pred_stats(y["dev"], p_dev_det + trend["dev"], "temporal LGBM DEV (detrended, trend re-added)"),
        "test": _pred_stats(y["test"], p_test_det + trend["test"], "temporal LGBM TEST (detrended, trend re-added)"),
    }

    # B. same split, same features, same function: raw target (detrend OFF)
    m_raw = train_lgbm_quantile(X["fit"], y["fit"], X["dev"], y["dev"], cols)
    p_dev_raw, _, _ = predict_intervals(m_raw, X["dev"])
    p_test_raw, _, _ = predict_intervals(m_raw, X["test"])
    out["B_temporal_detrend_OFF"] = {
        "dev": _pred_stats(y["dev"], p_dev_raw, "temporal LGBM DEV (raw target)"),
        "test": _pred_stats(y["test"], p_test_raw, "temporal LGBM TEST (raw target)"),
    }

    out["target_distributions"] = {p: describe(y[p], f"temporal {p} target (raw)")
                                   for p in ("fit", "dev", "cal", "test")}
    out["lag1_nan_fraction_before_fill"] = {
        p: round(float(tp["frames"][p]["Yield_lag1"].isna().mean()), 4) if "Yield_lag1" in tp["frames"][p] else None
        for p in ("fit", "dev", "cal", "test")}
    return out


def loso_lightgbm_controlled(state: str) -> dict:
    """Controlled single-variable experiments on ONE fold's DEV partition.

    Variable 1: detrend the target on the fold's own FIT years (as the temporal
                path already does) -- trend re-added before scoring.
    Variable 2: train on the full FIT partition instead of the 90% chronological
                head that the internal early-stopping carve-out leaves behind.
    Nothing else changes: same fold, same features, same order, same scaler,
    same hyperparameters, same function.
    """
    fold = build_loso_fold(state)
    cols = fold["feature_cols"]
    X, y, yr = fold["X"], fold["y"], fold["years"]

    coef = np.polyfit(yr["fit"], y["fit"], 1)
    tr_trend = {k: np.polyval(coef, yr[k]) for k in ("tr_fit", "fit", "dev")}

    runs = {}
    combos = [
        ("E0_current_raw_90pct_fit", False, "tr_fit"),
        ("E1_raw_full_fit", False, "fit"),
        ("E2_detrended_90pct_fit", True, "tr_fit"),
        ("E3_detrended_full_fit", True, "fit"),
    ]
    for name, detrend, train_part in combos:
        y_tr = y[train_part] - tr_trend[train_part] if detrend else y[train_part]
        m = train_lgbm_quantile(X[train_part], y_tr, X["es_val"], y["es_val"], cols)
        p_dev, _, _ = predict_intervals(m, X["dev"])
        if detrend:
            p_dev = p_dev + tr_trend["dev"]
        runs[name] = _pred_stats(y["dev"], p_dev, name)
        runs[name]["detrend"] = detrend
        runs[name]["train_partition"] = train_part
        runs[name]["train_years"] = [int(yr[train_part].min()), int(yr[train_part].max())]
        runs[name]["n_train"] = int(len(y[train_part]))
    return runs


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {}
    if what in ("all", "temporal"):
        print("=== Phase 10: temporal LightGBM path ===", flush=True)
        res["temporal_lightgbm"] = temporal_lightgbm()
        print(json.dumps(res["temporal_lightgbm"]["A_temporal_as_shipped_detrended"], indent=1))
        print(json.dumps(res["temporal_lightgbm"]["B_temporal_detrend_OFF"], indent=1))
        print("lag1 NaN before fill:", res["temporal_lightgbm"]["lag1_nan_fraction_before_fill"])
    if what in ("all", "controlled"):
        print("\n=== Controlled LightGBM experiments (Minnesota DEV) ===", flush=True)
        res["loso_lgb_controlled_Minnesota"] = loso_lightgbm_controlled("Minnesota")
        for k, v in res["loso_lgb_controlled_Minnesota"].items():
            print(f"  {k}: R2={v['r2']} bias={v['mean_bias']} rmse={v['rmse']} years={v['train_years']}")
    if what in ("all", "neural"):
        print("\n=== NeuralCQR LOSO (Minnesota) ===", flush=True)
        res["loso_neural_Minnesota"] = loso_neural("Minnesota")
        print(json.dumps(res["loso_neural_Minnesota"], indent=1, default=float))

    out = DIAG_OUT / "reports" / f"p4_neural_and_temporal_{what}.json"
    out.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {out}")
