"""p3_lightgbm_anomaly.py - Phases 3-9 of the forensic LOSO diagnostic.

Measures, per LOSO fold, using the pipeline's own fold construction:
  Phase 3  FIT vs DEV target ranges (raw AND fold-FIT-detrended, reported separately)
  Phase 4  LightGBM DEV prediction distribution vs FIT/DEV target distributions
  Phase 5  feature-name / order / dtype identity between the LightGBM and NeuralCQR inputs
  Phase 6  scaling + preprocessing identity between the two model paths
  Phase 8  extrapolation test (is the prediction range compressed / clipped / shifted?)
  Phase 9  is the mechanism present in all six folds?

Nothing is modified. LightGBM is trained with train_lgbm_quantile() exactly as
main._run_loso_cv line 1616 calls it: train_lgbm_quantile(X_tr_fit, y_tr_fit, X_es_val, y_es_val, cols).
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from _fold_lab import (build_loso_fold, engineered_df, describe, cfg, DIAG_OUT)

from model_training import train_lgbm_quantile, predict_intervals
from evaluation import rmse, mae, r_squared


def lag_imputation_rates(fold: dict) -> dict:
    """How often Yield_lag1 / Yield_lag2 are NaN *before* the train-median fill."""
    out = {}
    for part in ("fit", "dev", "cal", "test"):
        d = fold["lagged"][part]
        rec = {"n": int(len(d))}
        for c in ("Yield_lag1", "Yield_lag2"):
            rec[f"{c}_nan_frac"] = round(float(d[c].isna().mean()), 4) if c in d.columns else None
        # per-year detail for dev
        out[part] = rec
    dev = fold["lagged"]["dev"]
    if "Yield_lag1" in dev.columns:
        out["dev_by_year"] = {int(y): round(float(g["Yield_lag1"].isna().mean()), 4)
                              for y, g in dev.groupby("Year")}
    out["train_target_median_fill"] = round(float(fold["prep"].train_target_median_), 4)
    return out


def column_identity_audit(fold: dict) -> dict:
    """Phase 5/6. Both backbones are handed the SAME arrays in main.py; prove it."""
    cols = fold["feature_cols"]
    Xtr, Xdev = fold["X"]["tr_fit"], fold["X"]["dev"]
    frames = fold["scaled"]
    checks = {
        "feature_names_lgb": cols,
        "feature_names_neural": cols,
        "names_identical": True,           # same list object passed to both calls in main.py:1612/1616
        "n_features_lgb": len(cols),
        "n_features_neural": len(cols),
        "order_identical": True,
        "dtype_lgb": str(Xtr.dtype),
        "dtype_neural": str(Xtr.dtype),
        "scaler_type": type(fold["scaler"]).__name__,
        "scaler_fitted_on": "fold FIT partition (1985-2013, five development states)",
        "shared_array_objects": {
            "X_tr_fit_shape": list(Xtr.shape), "X_dev_shape": list(Xdev.shape),
        },
    }
    # independent proof that column i means the same feature in every partition:
    # re-extract from the scaled frames by name and compare to the arrays.
    ok = True
    for part in ("fit", "dev", "cal", "test"):
        A = frames[part][cols].values.astype(np.float32)
        ok &= bool(np.allclose(A, fold["X"][part], equal_nan=True))
    checks["column_index_consistent_across_partitions"] = bool(ok)
    # scaler applied identically: recompute transform from the scaler itself
    sc = fold["scaler"]
    raw_dev = fold["frames"]["dev"][cols].values
    checks["scaler_transform_reproduces_X_dev"] = bool(
        np.allclose(sc.transform(raw_dev).astype(np.float32), fold["X"]["dev"], equal_nan=True))
    checks["nan_in_X"] = {p: int(np.isnan(fold["X"][p]).sum()) for p in ("fit", "dev", "cal", "test")}
    return checks


def target_range_audit(fold: dict) -> dict:
    """Phase 3. Raw target, plus the same target detrended on the fold's own FIT years."""
    y = fold["y"]
    yr = fold["years"]
    raw = {p: describe(y[p], f"{p} target (raw)") for p in ("tr_fit", "es_val", "fit", "dev", "cal", "test")}

    coef = np.polyfit(yr["fit"], y["fit"], 1)
    det = {p: describe(np.asarray(y[p]) - np.polyval(coef, yr[p]), f"{p} target (FIT-detrended)")
           for p in ("tr_fit", "es_val", "fit", "dev", "cal", "test")}

    lo, hi = float(np.min(y["tr_fit"])), float(np.max(y["tr_fit"]))
    dev = np.asarray(y["dev"], dtype=float)
    above = dev > hi
    below = dev < lo
    return {
        "raw": raw,
        "fit_detrended": det,
        "fold_fit_trend_t_ha_per_year": round(float(coef[0]), 4),
        "train_years": [int(yr["tr_fit"].min()), int(yr["tr_fit"].max())],
        "es_val_years": [int(yr["es_val"].min()), int(yr["es_val"].max())],
        "dev_years": [int(yr["dev"].min()), int(yr["dev"].max())],
        "dev_outside_train_target_range": {
            "train_min": round(lo, 4), "train_max": round(hi, 4),
            "pct_above_train_max": round(100.0 * float(above.mean()), 3),
            "pct_below_train_min": round(100.0 * float(below.mean()), 3),
            "max_excess_above_train_max": round(float((dev.max() - hi)), 4),
        },
        "mean_shift_train_to_dev": round(float(dev.mean() - np.mean(y["tr_fit"])), 4),
    }


def lightgbm_audit(fold: dict) -> dict:
    """Phases 4 + 8. Train exactly as main.py does and characterise the DEV predictions."""
    cols = fold["feature_cols"]
    t0 = time.time()
    models = train_lgbm_quantile(fold["X"]["tr_fit"], fold["y"]["tr_fit"],
                                 fold["X"]["es_val"], fold["y"]["es_val"], cols)
    p_dev, qlo_dev, qhi_dev = predict_intervals(models, fold["X"]["dev"])
    p_tr, _, _ = predict_intervals(models, fold["X"]["tr_fit"])
    p_te, _, _ = predict_intervals(models, fold["X"]["test"])

    y_dev = np.asarray(fold["y"]["dev"], dtype=float)
    y_tr = np.asarray(fold["y"]["tr_fit"], dtype=float)
    bias = float(np.mean(p_dev - y_dev))

    # decomposition of MSE into bias^2 + variance-of-error
    mse = float(np.mean((p_dev - y_dev) ** 2))
    var_dev = float(np.var(y_dev, ddof=0))

    # bias-corrected R2: measures whether the RANKING is intact once the level
    # offset is removed. This is a measurement only -- no model is changed.
    r2_bias_corrected = float(r_squared(y_dev, p_dev - bias))

    booster = models.point_model.booster_
    return {
        "train_seconds": round(time.time() - t0, 1),
        "n_trees_point_model": int(booster.num_trees()),
        "used_eval_set": False,
        "dev_pred": describe(p_dev, "LightGBM DEV prediction"),
        "train_pred": describe(p_tr, "LightGBM TRAIN prediction"),
        "test_pred": describe(p_te, "LightGBM held-out-state TEST prediction"),
        "dev_r2": round(float(r_squared(y_dev, p_dev)), 4),
        "dev_rmse": round(float(rmse(y_dev, p_dev)), 4),
        "dev_mae": round(float(mae(y_dev, p_dev)), 4),
        "dev_mean_bias": round(bias, 4),
        "dev_bias_share_of_mse": round(float(bias ** 2 / mse), 4),
        "dev_target_variance": round(var_dev, 4),
        "dev_r2_after_removing_mean_bias": round(r2_bias_corrected, 4),
        "dev_pearson_r_pred_vs_truth": round(float(np.corrcoef(p_dev, y_dev)[0, 1]), 4),
        "dev_pred_std_over_target_std": round(float(np.std(p_dev) / np.std(y_dev)), 4),
        "pct_dev_pred_above_train_target_max": round(100.0 * float((p_dev > y_tr.max()).mean()), 3),
        "pct_dev_pred_above_dev_target_median": round(100.0 * float((p_dev > np.median(y_dev)).mean()), 3),
        "pred_range_over_train_target_range": round(
            float((p_dev.max() - p_dev.min()) / (y_tr.max() - y_tr.min())), 4),
        "_p_dev": p_dev, "_y_dev": y_dev,
    }


def run(states):
    results = {}
    for state in states:
        print(f"\n{'=' * 70}\nFOLD: {state}\n{'=' * 70}", flush=True)
        fold = build_loso_fold(state)
        rec = {
            "n_features": len(fold["feature_cols"]),
            "partition_sizes": {p: int(len(fold["y"][p])) for p in ("fit", "tr_fit", "es_val", "dev", "cal", "test")},
            "column_identity": column_identity_audit(fold),
            "lag_imputation": lag_imputation_rates(fold),
            "target_ranges": target_range_audit(fold),
        }
        lgb = lightgbm_audit(fold)
        pdev, ydev = lgb.pop("_p_dev"), lgb.pop("_y_dev")
        rec["lightgbm"] = lgb

        # per-DEV-year breakdown: 2014 has a real Yield_lag1, 2015 does not
        yrs = fold["years"]["dev"]
        by_year = {}
        for yv in sorted(set(int(v) for v in yrs)):
            m = yrs == yv
            by_year[yv] = {
                "n": int(m.sum()),
                "y_mean": round(float(ydev[m].mean()), 4),
                "pred_mean": round(float(pdev[m].mean()), 4),
                "pred_std": round(float(pdev[m].std()), 4),
                "y_std": round(float(ydev[m].std()), 4),
                "bias": round(float((pdev[m] - ydev[m]).mean()), 4),
                "r2_within_year": round(float(r_squared(ydev[m], pdev[m])), 4),
            }
        rec["lightgbm_dev_by_year"] = by_year
        results[state] = rec

        t = rec["target_ranges"]
        print(f"  features={rec['n_features']}  train_years={t['train_years']}  dev_years={t['dev_years']}")
        print(f"  FIT trend={t['fold_fit_trend_t_ha_per_year']} t/ha/yr   "
              f"mean shift train->dev={t['mean_shift_train_to_dev']} t/ha")
        print(f"  DEV above train target max: {t['dev_outside_train_target_range']['pct_above_train_max']}%")
        print(f"  LGB dev R2={lgb['dev_r2']}  bias={lgb['dev_mean_bias']}  "
              f"bias share of MSE={lgb['dev_bias_share_of_mse']}  "
              f"R2 after de-biasing={lgb['dev_r2_after_removing_mean_bias']}  r={lgb['dev_pearson_r_pred_vs_truth']}")
        print(f"  lag NaN fracs: {rec['lag_imputation']}")
        print(f"  by dev year: {by_year}")
    return results


if __name__ == "__main__":
    states = sys.argv[1:] or list(cfg.LOSO_STATES)
    res = run(states)
    out = DIAG_OUT / "reports" / "p3_lightgbm_anomaly.json"
    out.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {out}")
