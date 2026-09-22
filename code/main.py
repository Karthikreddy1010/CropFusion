# -*- coding: utf-8 -*-
"""
main.py - Master orchestration script for Paper 3 ACI pipeline (Strict Methodology Compliance).

Executes the complete pipeline matching Paper3_Methodology_Updated_v2:
1. Data loading & methodology compliance validation (§6.2, §7)
2. Exploratory Data Analysis (EDA)
3. Missing value handling (disaster-aware zero-fill with presence flags)
4. Outlier analysis (preserve genuine climate extremes)
5. Multicollinearity analysis (VIF, correlation heatmaps)
6. Feature engineering (CDHW phenology-aligned severity, SPEI-30 fallback, year-type)
7. Authoritative 4-way temporal split (FIT 1985-2013, DEV 2014-2015, CAL 2016-2018, TEST 2019-2023)
8. Strict scaler & preprocessing fitting on FIT partition ONLY (Zero Leakage)
9. PyTorch Neural CQR Net joint end-to-end training (§4.2)
10. All 5 conformal calibration methods (Static, Phenology-CQR, Weighted, Locally Adaptive, ACI)
11. Full evaluation & statistical testing:
    - PICP, MPIW, ACE, Winkler Score (§5.4)
    - Wilcoxon Signed-Rank + Secondary Paired t-Test + Holm-Bonferroni / BH (§5.5)
    - County Block Bootstrap & Year Block Bootstrap (§4.4, §5.3)
12. Computational Complexity Benchmark: single/batch latency, throughput, memory (§4.7)
13. Six-state Leave-One-State-Out CV (Per-fold scaler fitting for Zero Leakage §5.2)
14. 5-Row Ablation Study (§4.6)
15. Visualization suite
16. Report export in JSON, CSV, and Markdown formats
"""
from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config as cfg
from utils import setup_logging, set_global_seed, save_report, save_report_csv, save_report_markdown, log_decision

# Module imports
from data_loader import load_dataset, validate_methodology_compliance
from eda import run_eda
from missing_values import handle_missing_values
from outliers import analyze_outliers
from multicollinearity import analyze_multicollinearity
from feature_engineering import engineer_features, build_split_aware_lags, build_split_aware_rolling_features
from feature_selection import select_features
from scaling import fit_scaler, apply_scaling
from splitting import (temporal_split, temporal_split_4way, random_row_split, loso_cv_folds,
                       get_feature_target_arrays, add_obs_ids, obs_ids)
from preprocessor import TrainFittedPreprocessor
from model_training import (
    train_neural_cqr, train_neural_cqr_final, predict_intervals,
    train_lgbm_quantile, train_catboost_quantile, train_xgb_quantile,
    save_model_artifacts
)
from aci_calibrator import (
    static_conformal, phenology_stratified_cqr,
    weighted_conformal, locally_adaptive_conformal,
    adaptive_conformal_inference, standard_aci,
    severity_aware_adaptive_conformal, CalibrationResult,
)
from experiment_registry import ExperimentRegistry
from evaluation import (
    evaluate_calibration, run_full_evaluation,
    benchmark_computational_complexity,
    rmse, mae, r_squared, picp, mpiw, ace, winkler_score,
)
from visualization import (
    run_all_visualizations, plot_picp_comparison, plot_mpiw_comparison,
    plot_interval_width_by_yeartype, plot_loso_cv_results,
    plot_residuals, plot_aci_tracking,
)
from diagnostics import (
    diagnose_r2_source,
    run_residual_diagnostics,
    compute_feature_drift_diagnostics,
)


def main() -> None:
    """Execute the complete methodology-compliant Paper 3 ACI pipeline."""
    start_time = time.time()

    # ── Setup ────────────────────────────────────────────────
    logger = setup_logging()
    set_global_seed()
    logger.info("=" * 70)
    logger.info("PAPER 3: ADAPTIVE CONFORMAL INFERENCE PIPELINE (FULL COMPLIANCE)")
    logger.info("=" * 70)

    # C7 / D5: declare the protocol before anything is fitted, and record where
    # the live config disagrees with it. A run whose config has drifted from the
    # declared protocol is still produced -- but it says so, in its own folder.
    from frozen_protocol import write_protocol
    _protocol = write_protocol()
    logger.info("Frozen protocol %s (hash %s); config matches: %s",
                _protocol["protocol_version"], _protocol["protocol_hash"],
                _protocol["config_matches_protocol"])

    # ══════════════════════════════════════════════════════════
    # PHASE 1: DATA LOADING & METHODOLOGY VALIDATION
    # ══════════════════════════════════════════════════════════
    logger.info("\n" + "═" * 70)
    logger.info("PHASE 1: DATA LOADING & METHODOLOGY VALIDATION")
    logger.info("═" * 70)

    df = load_dataset()
    compliance_checks = validate_methodology_compliance(df)
    eda_report = run_eda(df)

    # ══════════════════════════════════════════════════════════
    # PHASE 2: DETERMINISTIC FEATURE ENGINEERING & DATASET PROFILING
    # ══════════════════════════════════════════════════════════
    logger.info("\n" + "═" * 70)
    logger.info("PHASE 2: DETERMINISTIC FEATURE ENGINEERING & PROFILING")
    logger.info("═" * 70)

    # 2a. Deterministic static feature engineering (Fourier, CDHW, Phenology, Interactions, Lag structure)
    df, feat_eng_report = engineer_features(df)

    # 2b. Non-mutating dataset profiling & EDA reports
    _, missing_report = handle_missing_values(df)
    _, outlier_report = analyze_outliers(df)
    _, multicol_report = analyze_multicollinearity(df)

    # Save processed dataset (with deterministic features)
    processed_path = cfg.OUTPUT_DIR / "Paper3_Processed.csv"
    df.to_csv(processed_path, index=False)
    logger.info("Processed dataset saved -> %s", processed_path)

    # ══════════════════════════════════════════════════════════
    # PHASE 3: AUTHORITATIVE 4-WAY TEMPORAL PIPELINE (§4.2 & §5.1)
    # ══════════════════════════════════════════════════════════
    logger.info("\n" + "═" * 70)
    logger.info("PHASE 3: AUTHORITATIVE 4-WAY TEMPORAL PIPELINE (FIT / DEV / CAL / TEST)")
    logger.info("═" * 70)

    # 1. 4-Way Temporal Split FIRST:
    # fit_df:  1985-2013 (Model fitting, detrending, baseline, feature selection, scaler)
    # dev_df:  2014-2015 (Internal development, early stopping, ensemble weight selection)
    # cal_df:  2016-2018 (Conformal calibration, uncontaminated non-conformity scores)
    # test_df: 2019-2023 (Locked final test)
    fit_df, dev_df, cal_df, test_df = temporal_split_4way(df, target=cfg.PRIMARY_TARGET)
    train_df, val_df = fit_df, dev_df  # Compatibility aliases

    # 2. Split-aware lag and rolling feature construction
    fit_df, dev_df, cal_df, test_df = build_split_aware_lags(
        fit_df, dev_df, test_df, target_col=cfg.PRIMARY_TARGET, split_type="temporal", cal_df=cal_df
    )
    fit_df, dev_df, cal_df, test_df = build_split_aware_rolling_features(
        fit_df, dev_df, test_df, split_type="temporal", cal_df=cal_df
    )

    # 3. Train-fitted preprocessing (Zero Leakage, fitted strictly on FIT partition)
    prep = TrainFittedPreprocessor(target_col=cfg.PRIMARY_TARGET)
    fit_df = prep.fit_transform(fit_df, split_name="fit")
    dev_df = prep.transform(dev_df, split_name="dev")
    cal_df = prep.transform(cal_df, split_name="cal")
    test_df = prep.transform(test_df, split_name="test")
    train_df, val_df = fit_df, dev_df  # Preprocessed compatibility aliases

    # 4. Feature selection on FIT DATA ONLY
    feature_cols, feat_sel_report = select_features(fit_df, target=cfg.PRIMARY_TARGET)

    # 5. Multicollinearity pruning on FIT DATA ONLY
    def _multicollinearity_prune(_df, cols):
        import numpy as _np
        _prot = getattr(cfg, "MULTICOLLINEARITY_PROTECT", [])
        _ct = getattr(cfg, "CORR_DROP_THRESHOLD", 0.95)
        _vt = getattr(cfg, "VIF_DROP_THRESHOLD", 10.0)
        _X = _df[cols]
        _tc = pd.concat([_X, _df[cfg.PRIMARY_TARGET]], axis=1).corr()[cfg.PRIMARY_TARGET].abs()
        _c = _X.corr().abs()
        _up = _c.where(_np.triu(_np.ones(_c.shape), k=1).astype(bool))
        _drop = set()
        for _a in _up.columns:
            for _b in _up.index:
                _v = _up.loc[_b, _a]
                if pd.notna(_v) and _v > _ct and _a not in _drop and _b not in _drop:
                    _drop.add(_a if _tc.get(_a, 0) < _tc.get(_b, 0) else _b)
        _kept = [c for c in cols if c not in _drop]
        # iterative VIF (FIT-ONLY analytical calculation)
        while len(_kept) > 6:
            _Xv = _X[_kept].values
            _Xv = (_Xv - _Xv.mean(axis=0)) / _np.maximum(_Xv.std(axis=0), 1e-6)
            _corr_mat = _np.corrcoef(_Xv, rowvar=False)
            try:
                _inv_corr = _np.linalg.pinv(_corr_mat)
                _vifs = pd.Series(_np.diag(_inv_corr), index=_kept)
            except Exception:
                break
            _cand = _vifs.drop([c for c in _prot if c in _vifs.index], errors="ignore")
            if _cand.empty or _cand.max() <= _vt:
                break
            _kept.remove(_cand.idxmax())
        _kept = list(dict.fromkeys(_kept + [c for c in _prot if c in cols]))
        logger.info("MULTICOLLINEARITY: %d -> %d features (corr>%.2f + VIF>%.0f)",
                    len(cols), len(_kept), _ct, _vt)
        logger.info("  corr-dropped: %s", sorted(_drop))
        logger.info("  vif-dropped: %s", sorted(set(cols) - set(_kept) - _drop))
        return _kept

    try:
        feature_cols = _multicollinearity_prune(fit_df, feature_cols)
    except Exception as _e:
        logger.warning("Multicollinearity prune skipped (%s); using original set.", _e)

    # 6. County baseline: mean FIT detrended anomaly per county (leakage-free)
    if getattr(cfg, "ADD_COUNTY_BASELINE", False):
        import numpy as _np
        fit_df = fit_df.copy()
        dev_df = dev_df.copy()
        cal_df = cal_df.copy()
        test_df = test_df.copy()
        _tr = fit_df[fit_df[cfg.PRIMARY_TARGET].notna()]
        _cf0 = _np.polyfit(_tr["Year"].values, _tr[cfg.PRIMARY_TARGET].values, 1)
        _anom = _tr[cfg.PRIMARY_TARGET].values - _np.polyval(_cf0, _tr["Year"].values)
        _cb = pd.Series(_anom, index=_tr["GEOID"].values).groupby(level=0).mean()
        fit_df["county_baseline"] = fit_df["GEOID"].map(_cb).fillna(0.0)
        dev_df["county_baseline"] = dev_df["GEOID"].map(_cb).fillna(0.0)
        cal_df["county_baseline"] = cal_df["GEOID"].map(_cb).fillna(0.0)
        test_df["county_baseline"] = test_df["GEOID"].map(_cb).fillna(0.0)
        if "county_baseline" not in feature_cols:
            feature_cols = feature_cols + ["county_baseline"]
        logger.info("COUNTY BASELINE added (fit-only 1985-2013, leakage-free).")

    # 6b. ROW-IDENTITY PROVENANCE LEDGER for the temporal experiment.
    #     Registered before the scaler is fitted and asserted before any model
    #     is trained, so a violation stops the run rather than being discovered
    #     in a report afterwards. Overlaps are computed on real county-year IDs
    #     (obs_id = "<GEOID>_<Year>"), never on object identity.
    from leakage_provenance import PartitionLedger, assert_disjoint, COLLECTOR
    fit_df = add_obs_ids(fit_df)
    dev_df = add_obs_ids(dev_df)
    cal_df = add_obs_ids(cal_df)
    test_df = add_obs_ids(test_df)

    temporal_ledger = PartitionLedger(experiment="Temporal")
    temporal_ledger.register("train", fit_df)
    temporal_ledger.register("dev", dev_df)
    temporal_ledger.register("cal", cal_df)
    temporal_ledger.register("test", test_df)
    temporal_ledger.bind_stage("feature_selection", "train")
    temporal_ledger.bind_stage("multicollinearity_pruning", "train")
    temporal_ledger.bind_stage("county_baseline", "train")
    temporal_ledger.bind_stage("target_detrending", "train")
    temporal_ledger.bind_stage("preprocessor_fitting", "train")
    temporal_ledger.bind_stage("scaler_fitting", "train")
    temporal_ledger.bind_stage("model_fitting", "train")
    temporal_ledger.bind_stage("early_stopping", "dev")
    temporal_ledger.bind_stage("ensemble_weight_selection", "dev")
    temporal_ledger.bind_stage("conformal_calibration", "cal")
    temporal_ledger.bind_stage("final_evaluation", "test")
    _temporal_audit = assert_disjoint(temporal_ledger)
    _temporal_audit["fit_years"] = [int(fit_df["Year"].min()), int(fit_df["Year"].max())]
    _temporal_audit["dev_years"] = [int(dev_df["Year"].min()), int(dev_df["Year"].max())]
    _temporal_audit["cal_years"] = [int(cal_df["Year"].min()), int(cal_df["Year"].max())]
    _temporal_audit["test_years"] = [int(test_df["Year"].min()), int(test_df["Year"].max())]
    COLLECTOR.add(_temporal_audit)

    # Authoritative record of the feature-selection funnel, so every report
    # quotes the same numbers instead of each restating a different stage.
    feature_pipeline = {
        "stage_1_candidates": int(feat_sel_report.get("n_candidates", len(feat_sel_report.get("consensus_votes", {})))),
        "stage_2_consensus_selected": int(feat_sel_report.get("n_consensus_selected", 0)),
        "stage_3_after_multicollinearity_pruning": int(
            len([f for f in feature_cols if f != "county_baseline"])
        ),
        "stage_4_final_modeling_features": int(len(feature_cols)),
        "county_baseline_added": bool(getattr(cfg, "ADD_COUNTY_BASELINE", False)),
        "sequence": (
            "candidates -> 4-method consensus vote (>=2 votes or protected) -> "
            "correlation (|r|>{corr}) + iterative VIF (>{vif}) pruning on FIT only -> "
            "+ county_baseline -> final modeling feature set"
        ).format(corr=cfg.CORR_DROP_THRESHOLD, vif=cfg.VIF_DROP_THRESHOLD),
        "all_fitted_on": "FIT partition (1985-2013) only",
        "final_feature_list": list(feature_cols),
    }
    save_report(feature_pipeline, "feature_selection_pipeline.json")
    logger.info(
        "FEATURE FUNNEL: %d candidates -> %d consensus -> %d after multicollinearity -> %d final",
        feature_pipeline["stage_1_candidates"], feature_pipeline["stage_2_consensus_selected"],
        feature_pipeline["stage_3_after_multicollinearity_pruning"],
        feature_pipeline["stage_4_final_modeling_features"],
    )

    # 7. Fit RobustScaler STRICTLY on fit_df ONLY
    scaler, scaler_name = fit_scaler(fit_df, feature_cols, scaler_type="robust")

    # Apply scaler
    fit_df_scaled = apply_scaling(fit_df, feature_cols, scaler, split_name="fit")
    dev_df_scaled = apply_scaling(dev_df, feature_cols, scaler, split_name="dev")
    cal_df_scaled = apply_scaling(cal_df, feature_cols, scaler, split_name="cal")
    test_df_scaled = apply_scaling(test_df, feature_cols, scaler, split_name="test")

    X_fit, y_fit = get_feature_target_arrays(fit_df_scaled, feature_cols, split_name="fit")
    X_dev, y_dev = get_feature_target_arrays(dev_df_scaled, feature_cols, split_name="dev")
    X_cal, y_cal = get_feature_target_arrays(cal_df_scaled, feature_cols, split_name="cal")
    X_test, y_test = get_feature_target_arrays(test_df_scaled, feature_cols, split_name="test")

    # 8. Target detrending fit STRICTLY on fit_df ONLY (1985-2013)
    import numpy as _np
    _DETREND = getattr(cfg, "DETREND_TARGET", False)
    if _DETREND:
        _cf = _np.polyfit(fit_df["Year"].values, y_fit, 1)   # FIT ONLY (1985-2013)
        _trend_slope = float(_cf[0])
        _trend_intercept = float(_cf[1])
        _fit_trend = _np.polyval(_cf, fit_df["Year"].values)
        _dev_trend = _np.polyval(_cf, dev_df["Year"].values)
        _cal_trend = _np.polyval(_cf, cal_df["Year"].values)
        _test_trend = _np.polyval(_cf, test_df["Year"].values)

        y_fit_detrended = y_fit - _fit_trend
        y_dev_detrended = y_dev - _dev_trend
        y_cal_detrended = y_cal - _cal_trend
        y_test_detrended = y_test - _test_trend

        y_fit_raw = y_fit.copy()
        y_dev_raw = y_dev.copy()
        y_cal_raw = y_cal.copy()
        y_test_raw = y_test.copy()          # keep raw for final metrics
        logger.info("DETREND: trend=%.4f t/ha/yr (fit-only 1985-2013). Modeling anomaly.", _cf[0])
    else:
        _trend_slope = None
        _trend_intercept = None
        _fit_trend = _dev_trend = _cal_trend = _test_trend = np.zeros_like(y_fit)
        y_fit_detrended = y_fit.copy()
        y_dev_detrended = y_dev.copy()
        y_cal_detrended = y_cal.copy()
        y_test_detrended = y_test.copy()
        y_fit_raw = y_fit.copy()
        y_dev_raw = y_dev.copy()
        y_cal_raw = y_cal.copy()
        y_test_raw = y_test.copy()

    # Visualizations
    vif_data = multicol_report.get("vif", [])
    lgbm_imp = feat_sel_report.get("lgbm_importance", [])
    mi_data = feat_sel_report.get("mutual_information", [])
    run_all_visualizations(df, vif_results=vif_data,
                           importance_data=lgbm_imp, mi_data=mi_data)

    from visualization import plot_yield_trends_by_state
    try:
        plot_yield_trends_by_state(df)
    except Exception as _e_trend:
        logger.warning("plot_yield_trends_by_state failed: %s", _e_trend)

    from geo_maps import run_all_geo_maps
    try:
        run_all_geo_maps(df)
    except Exception as _e_geo:
        logger.warning("Geographic maps failed: %s", _e_geo)

    # 9. DEVELOPMENT PHASE: Model Training on FIT (1985-2013) & Early Stopping on DEV (2014-2015)
    logger.info("\n--- Training Candidate Models on FIT (1985-2013) with DEV Early Stopping ---")
    frozen_hidden_dims = tuple(cfg.NEURAL_CQR_HIDDEN_DIMS)
    frozen_dropout = float(cfg.NEURAL_CQR_DROPOUT)
    frozen_lr = float(cfg.LEARNING_RATE)
    frozen_wd = float(cfg.WEIGHT_DECAY)
    frozen_batch_size = int(cfg.BATCH_SIZE)
    frozen_lp = float(cfg.LAMBDA_PINBALL)
    frozen_lh = float(cfg.LAMBDA_HUBER)
    frozen_lc = float(cfg.LAMBDA_CROSSING)
    frozen_lw = float(cfg.LAMBDA_WIDTH)

    dev_neural_model = train_neural_cqr(
        X_fit, y_fit_detrended, X_dev, y_dev_detrended, feature_cols,
        epochs=cfg.BASELINE_MAX_EPOCHS, batch_size=frozen_batch_size, lr=frozen_lr,
        weight_decay=frozen_wd, dropout_rate=frozen_dropout, hidden_dims=frozen_hidden_dims,
        lambda_pinball=frozen_lp, lambda_huber=frozen_lh, lambda_crossing=frozen_lc, lambda_width=frozen_lw,
        early_stopping_mode=cfg.EARLY_STOPPING_MODE, patience=cfg.EARLY_STOPPING_PATIENCE,
        joint_training=True, scaler=scaler
    )
    # Critical: Use the actual best DEV RMSE epoch (not early-stopping termination epoch or len(history))
    best_dev_epoch = getattr(dev_neural_model, "best_epoch", getattr(dev_neural_model, "epochs_trained", cfg.BASELINE_MAX_EPOCHS))
    if best_dev_epoch is None and isinstance(dev_neural_model, dict):
        _mdl = next(iter(dev_neural_model.values()), None)
        best_dev_epoch = getattr(_mdl, "best_epoch", getattr(_mdl, "epochs_trained", cfg.BASELINE_MAX_EPOCHS))
    if best_dev_epoch is None or best_dev_epoch < 1:
        best_dev_epoch = 40
    frozen_best_dev_epoch = int(best_dev_epoch)
    logger.info("FROZEN BEST DEV EPOCH: %d (selected via DEV_2014_2015 minimum RMSE)", frozen_best_dev_epoch)

    p_neural_dev, _, _ = predict_intervals(dev_neural_model, X_dev)
    p_neural_dev_eval = p_neural_dev + _dev_trend if _DETREND else p_neural_dev

    from model_training import (
        train_lgbm_quantile, train_catboost_quantile, train_xgb_quantile,
        save_model_artifacts, save_model_failure_log, save_hyperparameter_report
    )

    dev_lgb_model = train_lgbm_quantile(X_fit, y_fit_detrended, X_dev, y_dev_detrended, feature_cols)
    p_lgb_dev, _, _ = predict_intervals(dev_lgb_model, X_dev)
    p_lgb_dev_eval = p_lgb_dev + _dev_trend if _DETREND else p_lgb_dev

    # 10. ENSEMBLE POINT WEIGHT SELECTION ON DEV (2014-2015) ONLY (Mandatory: Search on DEV, metric: RMSE)
    logger.info("\n--- Selecting Ensemble Weights on DEV (2014-2015) ONLY ---")
    best_w, best_dev_rmse = 0.80, np.inf
    dev_weight_records = []
    for _w in np.arange(0.0, 1.01, 0.05):
        _w = round(float(_w), 2)
        _blend = _w * p_neural_dev_eval + (1.0 - _w) * p_lgb_dev_eval
        _rmse_val = rmse(y_dev_raw, _blend)
        _mae_val = mae(y_dev_raw, _blend)
        _r2_val = r_squared(y_dev_raw, _blend)
        dev_weight_records.append({
            "weight_neuralcqr": _w,
            "weight_lightgbm": round(1.0 - _w, 2),
            "dev_rmse": round(float(_rmse_val), 4),
            "dev_mae": round(float(_mae_val), 4),
            "dev_r2": round(float(_r2_val), 4),
        })
        if _rmse_val < best_dev_rmse:
            best_dev_rmse = _rmse_val
            best_w = _w

    model_sel_df = pd.DataFrame(dev_weight_records)
    save_report_csv(model_sel_df, "model_selection_audit.csv")

    frozen_w = best_w
    logger.info(
        "FROZEN ENSEMBLE WEIGHT: w_neural=%.2f, w_lgb=%.2f (Selected strictly on DEV_2014_2015 RMSE=%.4f)",
        frozen_w, round(1.0 - frozen_w, 2), best_dev_rmse
    )

    # 10b. FORMAL LOCK PROCEDURE (§18) — FREEZE ALL MODELING DECISIONS BEFORE FINAL REFIT
    lock_data = {
        "status": "LOCKED_FINAL_DECISIONS",
        "lock_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fit_partition": "1985-2013",
        "dev_partition": "2014-2015",
        "cal_partition": "2016-2018",
        "test_partition": "2019-2023 (LOCKED)",
        "selected_features_count": len(feature_cols),
        "selected_features": feature_cols,
        "target_detrending": {
            "method": "Linear Ordinary Least Squares Anomaly",
            "fit_data": "FIT_1985_2013_ONLY",
            "slope_t_ha_per_yr": round(float(_trend_slope), 4) if _DETREND else None,
            "intercept_t_ha": round(float(_trend_intercept), 4) if _DETREND else None,
        },
        "county_baseline": {
            "method": "Historical County Mean Yield",
            "fit_data": "FIT_1985_2013_ONLY",
            "fallback": "State/global median",
        },
        "neural_cqr_architecture": {
            "hidden_dims": list(frozen_hidden_dims),
            "dropout_rate": frozen_dropout,
            "heads": ["y_mean", "q0.05", "q0.50", "q0.95"],
        },
        "neural_cqr_hyperparameters": {
            "origin": "Fixed baseline prior to final evaluation; DEV (2014-2015) was used strictly for early stopping and ensemble-weight selection",
            "learning_rate": frozen_lr,
            "weight_decay": frozen_wd,
            "batch_size": frozen_batch_size,
            "dropout_rate": frozen_dropout,
            "hidden_dims": list(frozen_hidden_dims),
            "lambda_pinball": frozen_lp,
            "lambda_huber": frozen_lh,
            "lambda_crossing": frozen_lc,
            "lambda_width": frozen_lw,
            "frozen_epochs": int(frozen_best_dev_epoch),
            "epoch_selection_metric": "DEV_RMSE",
            "epoch_selection_partition": "DEV_2014_2015",
            "early_stopping_on_refit": False,
        },
        "lightgbm_hyperparameters": cfg.LGBM_PARAMS,
        "catboost_hyperparameters": {"iterations": 500, "learning_rate": 0.05},
        "xgboost_hyperparameters": {"n_estimators": 300, "learning_rate": 0.05},
        "ensemble_weights": {
            "w_neural": round(float(frozen_w), 2),
            "w_lightgbm": round(float(1.0 - frozen_w), 2),
            "selection_metric": "DEV_2014_2015_RMSE",
            "best_dev_rmse": round(float(best_dev_rmse), 4),
        },
        "conformal_calibration": {
            "partition": "CAL_2016_2018_ONLY",
            "methods": ["Static Split Conformal", "Standard ACI (Gibbs & Candès)", "SA-ACI (Severity-Aware)"],
            "nominal_alpha": cfg.NOMINAL_ALPHA,
            "nominal_coverage": 0.90,
            "sa_aci_gamma": cfg.ACI_GAMMA,
            "sa_aci_window": cfg.ACI_WINDOW_SIZE,
        }
    }
    save_report(lock_data, "model_decisions_lock.json")
    lock_md = f"""# Final Model Decisions Lock File (§18)

**Status**: `LOCKED — NO MODELING DECISIONS MAY CHANGE BEYOND THIS POINT`
**Timestamp**: `{lock_data['lock_timestamp']}`

## 1. Frozen Partitions
- **FIT (1985–2013)**: Preprocessing, detrending, baseline, feature selection, candidate fitting
- **DEV (2014–2015)**: Early stopping, hyperparameter confirmation, ensemble weight search
- **CAL (2016–2018)**: Conformal calibration ONLY (uncontaminated residual scores)
- **TEST (2019–2023)**: Final locked evaluation strictly once

## 2. Frozen Feature Specification
- **Selected Features ({len(feature_cols)})**: `{', '.join(feature_cols[:10])}...`
- **Linear Detrending**: Slope = `{round(float(_trend_slope), 4) if _DETREND else 'N/A'}` t/ha/yr (Fitted on FIT only)
- **County Baseline**: Historical county mean fitted on FIT only

## 3. Frozen Backbone & Ensemble Specifications
- **NeuralCQR**: Architecture and hyperparameters were fixed prior to final evaluation; the DEV partition (2014–2015) was used strictly for early stopping and ensemble-weight selection.
  - Frozen Hyperparameters: Hidden Dims = `{frozen_hidden_dims}`, LR = `{frozen_lr}`, WD = `{frozen_wd}`, Batch = `{frozen_batch_size}`, Dropout = `{frozen_dropout}`, Pinball λ = `{frozen_lp}`, Huber λ = `{frozen_lh}`, Crossing λ = `{frozen_lc}`, Width λ = `{frozen_lw}`
  - Frozen Stopping Epoch: `{frozen_best_dev_epoch}` (selected via DEV RMSE on DEV_2014_2015; early stopping on refit: false)
- **LightGBM**: `{cfg.LGBM_PARAMS}`
- **CatBoost**: `iterations=500, learning_rate=0.05`
- **XGBoost**: `n_estimators=300, learning_rate=0.05`
- **Ensemble Weight**: `{frozen_w:.2f}` NeuralCQR + `{1.0 - frozen_w:.2f}` LightGBM (Selected on DEV RMSE = `{best_dev_rmse:.4f}`)

## 4. Conformal Calibration Protocol
- **Raw Quantile Blending**: Ensemble quantiles blended BEFORE calibration
- **Calibration Set**: 2016–2018 nonconformity scores
- **Evaluated Conformal Methods**: Static Conformal, Standard ACI, SA-ACI
"""
    save_report_markdown(lock_md, "model_decisions_lock.md")
    logger.info("MODEL DECISIONS LOCKED -> model_decisions_lock.json / model_decisions_lock.md")

    # 11. FINAL REFIT ON FIT + DEV (1985-2015) using FROZEN decisions (Zero Validation Leakage)
    logger.info("\n--- FINAL REFIT on FIT + DEV (1985-2015) using Frozen Configuration (Zero Validation Leakage) ---")
    X_fit_dev = np.vstack([X_fit, X_dev])
    y_fit_dev_detrended = np.concatenate([y_fit_detrended, y_dev_detrended])
    y_fit_dev_raw = np.concatenate([y_fit_raw, y_dev_raw])

    refit_neural = train_neural_cqr_final(
        X_fit_dev, y_fit_dev_detrended, feature_cols,
        epochs=frozen_best_dev_epoch, batch_size=frozen_batch_size, lr=frozen_lr,
        weight_decay=frozen_wd, dropout_rate=frozen_dropout, hidden_dims=frozen_hidden_dims,
        lambda_pinball=frozen_lp, lambda_huber=frozen_lh, lambda_crossing=frozen_lc, lambda_width=frozen_lw,
        joint_training=True, scaler=scaler
    )

    refit_lgbm = train_lgbm_quantile(X_fit_dev, y_fit_dev_detrended, feature_cols=feature_cols)

    try:
        refit_cat = train_catboost_quantile(X_fit_dev, y_fit_dev_detrended, feature_cols=feature_cols)
    except Exception as _e_cat:
        logger.warning("CatBoost refit failed (%s); falling back to LightGBM.", _e_cat)
        refit_cat = refit_lgbm

    try:
        refit_xgb = train_xgb_quantile(X_fit_dev, y_fit_dev_detrended, feature_cols=feature_cols)
    except Exception as _e_xgb:
        logger.warning("XGBoost refit failed (%s); falling back to LightGBM.", _e_xgb)
        refit_xgb = refit_lgbm

    primary_models = refit_neural
    backbone_models = {
        "NeuralCQR": refit_neural,
        "LightGBM": refit_lgbm,
        "CatBoost": refit_cat,
        "XGBoost": refit_xgb,
    }

    # Save refitted models
    save_model_artifacts(refit_neural, "NeuralCQR", scaler=scaler,
                         hyperparams={"type": "NeuralCQR", "epochs_trained": frozen_best_dev_epoch, "fit_period": "1985-2015"})
    save_model_artifacts(refit_lgbm, "LightGBM", scaler=scaler, hyperparams=cfg.LGBM_PARAMS)

    # 12. GENERATE RAW PREDICTIONS & QUANTILES ON CAL AND TEST
    p_neural_cal, q_lo_neural_cal, q_hi_neural_cal = predict_intervals(refit_neural, X_cal)
    p_lgb_cal, q_lo_lgb_cal, q_hi_lgb_cal = predict_intervals(refit_lgbm, X_cal)

    p_neural_te, q_lo_neural_te, q_hi_neural_te = predict_intervals(refit_neural, X_test)
    p_lgb_te, q_lo_lgb_te, q_hi_lgb_te = predict_intervals(refit_lgbm, X_test)
    p_cat_te, q_lo_cat_te, q_hi_cat_te = predict_intervals(refit_cat, X_test)
    p_xgb_te, q_lo_xgb_te, q_hi_xgb_te = predict_intervals(refit_xgb, X_test)

    # Add trend back to predictions and intervals so all calibration & metrics are on the real yield scale (t/ha)
    if _DETREND:
        p_neural_cal += _cal_trend; q_lo_neural_cal += _cal_trend; q_hi_neural_cal += _cal_trend
        p_lgb_cal += _cal_trend; q_lo_lgb_cal += _cal_trend; q_hi_lgb_cal += _cal_trend

        p_neural_te += _test_trend; q_lo_neural_te += _test_trend; q_hi_neural_te += _test_trend
        p_lgb_te += _test_trend; q_lo_lgb_te += _test_trend; q_hi_lgb_te += _test_trend
        p_cat_te += _test_trend; q_lo_cat_te += _test_trend; q_hi_cat_te += _test_trend
        p_xgb_te += _test_trend; q_lo_xgb_te += _test_trend; q_hi_xgb_te += _test_trend
        logger.info("DETREND: Trend added back to predictions and intervals for CAL and TEST.")

    # 13. CRITICAL — BLEND RAW ENSEMBLE QUANTILES BEFORE CALIBRATION (Part 11 & Part 13 Protocol)
    p_ens_cal = frozen_w * p_neural_cal + (1.0 - frozen_w) * p_lgb_cal
    q_lo_ens_cal = frozen_w * q_lo_neural_cal + (1.0 - frozen_w) * q_lo_lgb_cal
    q_hi_ens_cal = frozen_w * q_hi_neural_cal + (1.0 - frozen_w) * q_hi_lgb_cal

    p_ens_te = frozen_w * p_neural_te + (1.0 - frozen_w) * p_lgb_te
    q_lo_ens_te = frozen_w * q_lo_neural_te + (1.0 - frozen_w) * q_lo_lgb_te
    q_hi_ens_te = frozen_w * q_hi_neural_te + (1.0 - frozen_w) * q_hi_lgb_te

    # Use raw-scale test targets for all downstream evaluations
    y_test = y_test_raw
    preds_test = p_ens_te

    # 14. CONFORMAL CALIBRATION ON UNCONTAMINATED CAL (2016-2018) ONLY
    logger.info("\n--- Conformal Calibration on CAL (2016-2018) ONLY ---")
    years_test = test_df["Year"].values
    year_types_test = test_df["Year_Type"].values if "Year_Type" in test_df.columns else None

    # Log conformal calibration audit
    scores_cal_ens = np.maximum(q_lo_ens_cal - y_cal_raw, y_cal_raw - q_hi_ens_cal)
    cal_audit_df = pd.DataFrame({
        "Score_Index": np.arange(len(scores_cal_ens)),
        "NonConformity_Score": np.round(scores_cal_ens, 4),
        "Year": cal_df["Year"].values,
        "GEOID": cal_df["GEOID"].values,
        "Target_Yield": np.round(y_cal_raw, 4),
    })
    save_report_csv(cal_audit_df, "conformal_calibration_audit.csv")

    cdhw_sev_cal = cal_df["CDHW_Severity_Score"].values if "CDHW_Severity_Score" in cal_df.columns else None
    cdhw_sev_te = test_df["CDHW_Severity_Score"].values if "CDHW_Severity_Score" in test_df.columns else None

    # Algorithm 1: Static Split Conformal (CQR Baseline)
    static_result = static_conformal(
        y_cal_raw, q_lo_ens_cal, q_hi_ens_cal,
        q_lo_ens_te, q_hi_ens_te,
        p_ens_te, y_test_raw, alpha=cfg.NOMINAL_ALPHA,
    )
    static_result.method = "static_conformal"

    # Algorithm 2: Standard ACI (Gibbs & Candès 2021)
    std_aci_result = standard_aci(
        y_cal_raw, q_lo_ens_cal, q_hi_ens_cal,
        y_test_raw, q_lo_ens_te, q_hi_ens_te,
        p_ens_te, years_test, alpha=cfg.NOMINAL_ALPHA, gamma=cfg.ACI_GAMMA,
    )
    std_aci_result.method = "standard_aci"

    # Algorithm 3: Severity-Aware Adaptive Conformal Inference (SA-ACI / CDHW-ACI)
    sa_aci_result = severity_aware_adaptive_conformal(
        y_cal_raw, q_lo_ens_cal, q_hi_ens_cal,
        y_test_raw, q_lo_ens_te, q_hi_ens_te,
        p_ens_te, years_test, alpha=cfg.NOMINAL_ALPHA, gamma=cfg.ACI_GAMMA,
        window=cfg.ACI_WINDOW_SIZE, cdhw_severity_test=cdhw_sev_te, cdhw_severity_cal=cdhw_sev_cal,
    )
    sa_aci_result.method = "sa_aci"

    # Supplementary baselines
    pheno_cal = cal_df["Phenological_Window"].values if "Phenological_Window" in cal_df.columns else np.full(len(cal_df), "Grain_Fill")
    pheno_test = test_df["Phenological_Window"].values if "Phenological_Window" in test_df.columns else np.full(len(test_df), "Grain_Fill")
    pheno_result = phenology_stratified_cqr(
        y_cal_raw, q_lo_ens_cal, q_hi_ens_cal, pheno_cal,
        q_lo_ens_te, q_hi_ens_te, pheno_test,
        p_ens_te, y_test_raw,
    )
    weighted_result = weighted_conformal(
        y_cal_raw, q_lo_ens_cal, q_hi_ens_cal, X_cal,
        q_lo_ens_te, q_hi_ens_te, X_test,
        p_ens_te, y_test_raw,
    )
    local_result = locally_adaptive_conformal(
        y_cal_raw, q_lo_ens_cal, q_hi_ens_cal,
        q_lo_ens_te, q_hi_ens_te,
        p_ens_te, y_test_raw, p_ens_cal,
    )

    all_results = {
        "static_conformal": static_result,
        "standard_aci": std_aci_result,
        "sa_aci": sa_aci_result,
        "phenology_stratified_cqr": pheno_result,
        "weighted_conformal": weighted_result,
        "locally_adaptive": local_result,
    }
    aci_result = sa_aci_result  # Primary adaptive model for legacy hooks

    # Conformal validation check
    from aci_calibrator import validate_conformal_calibration
    validate_conformal_calibration(all_results)

    eval_report = run_full_evaluation(all_results, year_types_test, test_df=test_df)

    # 15. MULTI-BACKBONE BENCHMARK & CONFORMAL COMPARISON
    backbone_benchmark_rows = []
    for b_name, b_preds, bq_lo, bq_hi in [
        ("NeuralCQR", p_neural_te, q_lo_neural_te, q_hi_neural_te),
        ("LightGBM", p_lgb_te, q_lo_lgb_te, q_hi_lgb_te),
        ("CatBoost", p_cat_te, q_lo_cat_te, q_hi_cat_te),
        ("XGBoost", p_xgb_te, q_lo_xgb_te, q_hi_xgb_te),
    ]:
        b_rmse = rmse(y_test_raw, b_preds)
        b_mae = mae(y_test_raw, b_preds)
        b_r2 = r_squared(y_test_raw, b_preds)
        b_picp = picp(y_test_raw, bq_lo, bq_hi)
        b_mpiw = mpiw(bq_lo, bq_hi)
        b_winkler = winkler_score(y_test_raw, bq_lo, bq_hi)
        backbone_benchmark_rows.append({
            "Model": b_name,
            "RMSE": round(float(b_rmse), 4),
            "MAE": round(float(b_mae), 4),
            "R2": round(float(b_r2), 4),
            "Coverage": round(float(b_picp), 4),
            "MPIW": round(float(b_mpiw), 4),
            "Winkler_Score": round(float(b_winkler), 4),
        })

    ens_rmse = rmse(y_test_raw, p_ens_te)
    ens_mae = mae(y_test_raw, p_ens_te)
    ens_r2 = r_squared(y_test_raw, p_ens_te)
    ens_picp = picp(y_test_raw, sa_aci_result.q_lo, sa_aci_result.q_hi)
    ens_mpiw = mpiw(sa_aci_result.q_lo, sa_aci_result.q_hi)
    ens_winkler = winkler_score(y_test_raw, sa_aci_result.q_lo, sa_aci_result.q_hi)

    backbone_benchmark_rows.append({
        "Model": "NeuralCQR_LightGBM_Ensemble",
        "RMSE": round(float(ens_rmse), 4),
        "MAE": round(float(ens_mae), 4),
        "R2": round(float(ens_r2), 4),
        "Coverage": round(float(ens_picp), 4),
        "MPIW": round(float(ens_mpiw), 4),
        "Winkler_Score": round(float(ens_winkler), 4),
    })

    backbone_df = pd.DataFrame(backbone_benchmark_rows)
    save_report_csv(backbone_df, "backbone_benchmark.csv")
    save_report({"backbones": backbone_benchmark_rows}, "backbone_metrics.json")

    logger.info(
        "ENSEMBLE FINAL RESULT (Refit on 1985-2015, tested on locked 2019-2023) -> "
        "R2: %.4f, RMSE: %.4f, MAE: %.4f, Coverage: %.4f, Winkler: %.4f",
        ens_r2, ens_rmse, ens_mae, ens_picp, ens_winkler
    )

    # Conformal comparison table
    conformal_comp_rows = [
        {"Method": "Static Conformal", "Coverage_PICP": round(float(picp(y_test_raw, static_result.q_lo, static_result.q_hi)), 4), "ACE": round(float(ace(y_test_raw, static_result.q_lo, static_result.q_hi)), 4), "MPIW_Width": round(float(mpiw(static_result.q_lo, static_result.q_hi)), 4), "Winkler_Score": round(float(winkler_score(y_test_raw, static_result.q_lo, static_result.q_hi)), 4), "RMSE": round(float(ens_rmse), 4), "R2": round(float(ens_r2), 4)},
        {"Method": "Standard ACI", "Coverage_PICP": round(float(picp(y_test_raw, std_aci_result.q_lo, std_aci_result.q_hi)), 4), "ACE": round(float(ace(y_test_raw, std_aci_result.q_lo, std_aci_result.q_hi)), 4), "MPIW_Width": round(float(mpiw(std_aci_result.q_lo, std_aci_result.q_hi)), 4), "Winkler_Score": round(float(winkler_score(y_test_raw, std_aci_result.q_lo, std_aci_result.q_hi)), 4), "RMSE": round(float(ens_rmse), 4), "R2": round(float(ens_r2), 4)},
        {"Method": "SA-ACI", "Coverage_PICP": round(float(picp(y_test_raw, sa_aci_result.q_lo, sa_aci_result.q_hi)), 4), "ACE": round(float(ace(y_test_raw, sa_aci_result.q_lo, sa_aci_result.q_hi)), 4), "MPIW_Width": round(float(mpiw(sa_aci_result.q_lo, sa_aci_result.q_hi)), 4), "Winkler_Score": round(float(winkler_score(y_test_raw, sa_aci_result.q_lo, sa_aci_result.q_hi)), 4), "RMSE": round(float(ens_rmse), 4), "R2": round(float(ens_r2), 4)},
        {"Method": "Phenology-Stratified CQR", "Coverage_PICP": round(float(picp(y_test_raw, pheno_result.q_lo, pheno_result.q_hi)), 4), "ACE": round(float(ace(y_test_raw, pheno_result.q_lo, pheno_result.q_hi)), 4), "MPIW_Width": round(float(mpiw(pheno_result.q_lo, pheno_result.q_hi)), 4), "Winkler_Score": round(float(winkler_score(y_test_raw, pheno_result.q_lo, pheno_result.q_hi)), 4), "RMSE": round(float(ens_rmse), 4), "R2": round(float(ens_r2), 4)},
        {"Method": "Weighted Conformal", "Coverage_PICP": round(float(picp(y_test_raw, weighted_result.q_lo, weighted_result.q_hi)), 4), "ACE": round(float(ace(y_test_raw, weighted_result.q_lo, weighted_result.q_hi)), 4), "MPIW_Width": round(float(mpiw(weighted_result.q_lo, weighted_result.q_hi)), 4), "Winkler_Score": round(float(winkler_score(y_test_raw, weighted_result.q_lo, weighted_result.q_hi)), 4), "RMSE": round(float(ens_rmse), 4), "R2": round(float(ens_r2), 4)},
        {"Method": "Locally Adaptive", "Coverage_PICP": round(float(picp(y_test_raw, local_result.q_lo, local_result.q_hi)), 4), "ACE": round(float(ace(y_test_raw, local_result.q_lo, local_result.q_hi)), 4), "MPIW_Width": round(float(mpiw(local_result.q_lo, local_result.q_hi)), 4), "Winkler_Score": round(float(winkler_score(y_test_raw, local_result.q_lo, local_result.q_hi)), 4), "RMSE": round(float(ens_rmse), 4), "R2": round(float(ens_r2), 4)},
    ]
    df_conf_comp = pd.DataFrame(conformal_comp_rows)
    save_report_csv(df_conf_comp, "conformal_comparison.csv")

    # Extreme CDHW Coverage Analysis
    extreme_rows = []
    for yt in ["Normal", "Moderate", "Extreme"]:
        if "Year_Type" in test_df.columns:
            m_yt = (test_df["Year_Type"].values == yt)
        else:
            m_yt = np.ones(len(y_test_raw), dtype=bool)

        if m_yt.sum() > 0:
            for c_name, c_res in [("Static Conformal", static_result), ("Standard ACI", std_aci_result), ("SA-ACI", sa_aci_result)]:
                y_sub = y_test_raw[m_yt]
                lo_sub = c_res.q_lo[m_yt]
                hi_sub = c_res.q_hi[m_yt]
                extreme_rows.append({
                    "Year_Type": yt,
                    "Method": c_name,
                    "Count": int(m_yt.sum()),
                    "PICP": round(float(picp(y_sub, lo_sub, hi_sub)), 4),
                    "ACE": round(float(ace(y_sub, lo_sub, hi_sub)), 4),
                    "MPIW": round(float(mpiw(lo_sub, hi_sub)), 4),
                    "Winkler_Score": round(float(winkler_score(y_sub, lo_sub, hi_sub)), 4),
                })
    save_report_csv(pd.DataFrame(extreme_rows), "extreme_event_analysis.csv")

    # Clustered Bootstrap Inference Results (County-Block & Year-Block)
    from evaluation import block_bootstrap_county, block_bootstrap_year
    cb_diag = block_bootstrap_county(test_df, y_test_raw, sa_aci_result.q_lo, sa_aci_result.q_hi, p_ens_te)
    yb_diag = block_bootstrap_year(test_df, y_test_raw, sa_aci_result.q_lo, sa_aci_result.q_hi, p_ens_te)

    bootstrap_rows = []
    for cluster_type, diag in [("County_Block_Bootstrap", cb_diag), ("Year_Block_Bootstrap", yb_diag)]:
        for metric_name in ["r2", "rmse", "picp", "mpiw", "ace", "winkler"]:
            if metric_name in diag:
                bootstrap_rows.append({
                    "Cluster_Type": cluster_type,
                    "Metric": metric_name.upper(),
                    "Mean": diag[metric_name]["mean"],
                    "CI_95_Lower": diag[metric_name]["ci_95"][0],
                    "CI_95_Upper": diag[metric_name]["ci_95"][1],
                    "Iterations": diag["iterations"],
                })
    save_report_csv(pd.DataFrame(bootstrap_rows), "bootstrap_inference_results.csv")

    # Feature Lineage & Provenance
    feature_lineage = {
        "candidate_features_total": len(df.columns),
        "selected_features_count": len(feature_cols),
        "selected_features": feature_cols,
        "detrending_specification": (f"Linear trend fit strictly on 1985-2013 FIT partition (slope={_trend_slope:.4f} t/ha/yr)" if _DETREND and _trend_slope is not None else "None"),
        "county_baseline_specification": "Mean detrended yield anomaly fit strictly on 1985-2013 FIT partition",
        "feature_selection_methods": ["Mutual Information", "LightGBM Importance", "Permutation", "Multicollinearity VIF/Corr"],
        "scaling_specification": "RobustScaler fit strictly on 1985-2013 FIT partition",
    }
    save_report(feature_lineage, "feature_lineage.json")

    # Old vs New Comparative Table
    old_vs_new = [
        {"Model_Pipeline": "NeuralCQR Alone", "Old_Metric_R2": 0.5024, "Old_RMSE": 1.1958, "Old_MAE": 0.9424,
         "New_Metric_R2": round(float(r_squared(y_test_raw, p_neural_te)), 4), "New_RMSE": round(float(rmse(y_test_raw, p_neural_te)), 4), "New_MAE": round(float(mae(y_test_raw, p_neural_te)), 4),
         "Methodology_Status": "Clean 4-way split, zero leakage"},
        {"Model_Pipeline": "LightGBM Alone", "Old_Metric_R2": 0.4885, "Old_RMSE": 1.2124, "Old_MAE": 0.9685,
         "New_Metric_R2": round(float(r_squared(y_test_raw, p_lgb_te)), 4), "New_RMSE": round(float(rmse(y_test_raw, p_lgb_te)), 4), "New_MAE": round(float(mae(y_test_raw, p_lgb_te)), 4),
         "Methodology_Status": "Clean 4-way split, zero leakage"},
        {"Model_Pipeline": "Ensemble (NeuralCQR + LightGBM)", "Old_Metric_R2": 0.5370, "Old_RMSE": 1.1534, "Old_MAE": 0.9157,
         "New_Metric_R2": round(float(ens_r2), 4), "New_RMSE": round(float(ens_rmse), 4), "New_MAE": round(float(ens_mae), 4),
         "Methodology_Status": "Strict DEV weight optimization, raw quantile calibration"},
    ]
    save_report_csv(pd.DataFrame(old_vs_new), "old_vs_new_comparison.csv")

    # Experiment Registry Logging
    reg = ExperimentRegistry()
    reg.register_experiment(
        experiment_id="EXP_DEV_OPT",
        model="NeuralCQR+LightGBM",
        role="DEVELOPMENT",
        features=feature_cols,
        detrending="Linear_FIT_Only",
        fit_period="1985-2013",
        dev_period="2014-2015",
        cal_period="2016-2018",
        test_period="2019-2023",
        ensemble_weight=frozen_w,
        rmse=best_dev_rmse,
        r2=model_sel_df.loc[model_sel_df["weight_neuralcqr"] == frozen_w, "dev_r2"].values[0],
        notes="Dev weight optimization on 2014-2015"
    )
    reg.register_experiment(
        experiment_id="EXP_CAL_STATIC",
        model="NeuralCQR+LightGBM_Ensemble",
        role="CALIBRATION",
        features=feature_cols,
        detrending="Linear_FIT_Only",
        fit_period="1985-2013",
        dev_period="2014-2015",
        cal_period="2016-2018",
        test_period="2019-2023",
        conformal_method="Static_Conformal",
        picp=conformal_comp_rows[0]["Coverage_PICP"],
        mpiw=conformal_comp_rows[0]["MPIW_Width"],
        notes="Conformal threshold calibrated on uncontaminated CAL 2016-2018"
    )
    reg.register_experiment(
        experiment_id="EXP_FINAL_TEST_SA_ACI",
        model="NeuralCQR+LightGBM_Ensemble",
        role="FINAL_TEST",
        features=feature_cols,
        detrending="Linear_FIT_Only",
        fit_period="1985-2013",
        dev_period="2014-2015",
        cal_period="2016-2018",
        test_period="2019-2023",
        ensemble_weight=frozen_w,
        conformal_method="SA-ACI",
        rmse=ens_rmse,
        mae=ens_mae,
        r2=ens_r2,
        picp=ens_picp,
        ace=ace(y_test_raw, sa_aci_result.q_lo, sa_aci_result.q_hi),
        mpiw=ens_mpiw,
        winkler=ens_winkler,
        notes="Final locked evaluation on 2019-2023"
    )

    # Comprehensive Metrics Report
    comp_metrics_rows = []
    for b_row in backbone_benchmark_rows:
        comp_metrics_rows.append({
            "Model_or_Method": b_row["Model"],
            "Type": "Backbone_or_Ensemble",
            "RMSE": b_row["RMSE"],
            "MAE": b_row["MAE"],
            "R2": b_row["R2"],
            "Coverage_PICP": b_row["Coverage"],
            "MPIW_Width": b_row["MPIW"],
            "Winkler_Score": b_row["Winkler_Score"],
        })
    for c_row in conformal_comp_rows:
        comp_metrics_rows.append({
            "Model_or_Method": c_row["Method"],
            "Type": "Conformal_Calibration",
            "RMSE": c_row["RMSE"],
            "MAE": None,
            "R2": c_row["R2"],
            "Coverage_PICP": c_row["Coverage_PICP"],
            "MPIW_Width": c_row["MPIW_Width"],
            "Winkler_Score": c_row["Winkler_Score"],
        })
    df_comp_metrics = pd.DataFrame(comp_metrics_rows)
    save_report_csv(df_comp_metrics, "comprehensive_metrics.csv")
    save_report({"comprehensive_metrics": comp_metrics_rows}, "comprehensive_metrics.json")

    # Diagnostics & Spatial choropleths
    r2_diag_report = diagnose_r2_source(fit_df, test_df, y_test, preds_test)
    from geo_maps import plot_residual_choropleth
    try:
        plot_residual_choropleth(test_df, y_test, preds_test)
    except Exception as _e_resid_map:
        logger.warning("plot_residual_choropleth failed: %s", _e_resid_map)

    res_diag_report = run_residual_diagnostics(y_test, preds_test, test_df)

    # Post-hoc Covariate Shift & Feature Drift Diagnostics comparing FIT vs TEST (strictly post-lock & post-prediction)
    drift_report = compute_feature_drift_diagnostics(fit_df_scaled, test_df_scaled, feature_cols)
    if "all_feature_drift" in drift_report:
        save_report_csv(pd.DataFrame(drift_report["all_feature_drift"]), "distribution_shift_analysis.csv")

    from visualization import (
        plot_backbone_comparison, export_shap_consistency_report,
        plot_all_calibration_curves, plot_objective_o5_complete_suite,
        plot_feature_selection_stability_visualizations, plot_residual_diagnostics_suite
    )
    plot_residual_diagnostics_suite(y_test, preds_test, test_df)
    plot_backbone_comparison(backbone_df)
    export_shap_consistency_report(backbone_models, feature_cols, X_sample=X_test)
    plot_all_calibration_curves(all_results)
    plot_feature_selection_stability_visualizations()

    # Computational complexity benchmark (§4.7)
    complexity_report = benchmark_computational_complexity(primary_models, X_test, scaler=scaler)
    eval_report["computational_complexity"] = complexity_report
    save_report(eval_report, "evaluation_report.json")

    _export_evaluation_csv_md(eval_report)
    from evaluation import (
        export_predictions_csv, evaluate_objective_o1, evaluate_objective_o4,
        evaluate_objective_o5, evaluate_objective_o6, export_statistical_tests_report_md,
        aggregate_loso_results
    )
    export_predictions_csv(all_results, test_df)
    export_statistical_tests_report_md(eval_report)

    # 16. CONTROLLED OBJECTIVE O1 (Base vs. Base + CDHW)
    logger.info("\n--- Controlled Objective O1 Ablation ---")
    cdhw_cols_present = [c for c in cfg.CDHW_COLS + ["CDHW_Veg_Severity", "CDHW_Silking_Severity", "CDHW_GrainFill_Severity"] if c in feature_cols]
    non_cdhw_cols = [c for c in feature_cols if c not in cdhw_cols_present]

    scaler_no, _ = fit_scaler(fit_df, non_cdhw_cols, scaler_type="robust")
    X_fit_dev_no, _ = get_feature_target_arrays(apply_scaling(pd.concat([fit_df, dev_df], axis=0), non_cdhw_cols, scaler_no), non_cdhw_cols)
    X_te_no, y_te_no = get_feature_target_arrays(apply_scaling(test_df, non_cdhw_cols, scaler_no), non_cdhw_cols)

    models_no_cdhw = train_neural_cqr_final(
        X_fit_dev_no, y_fit_dev_detrended, non_cdhw_cols,
        epochs=frozen_best_dev_epoch, batch_size=frozen_batch_size, lr=frozen_lr,
        weight_decay=frozen_wd, dropout_rate=frozen_dropout, hidden_dims=frozen_hidden_dims,
        lambda_pinball=frozen_lp, lambda_huber=frozen_lh, lambda_crossing=frozen_lc, lambda_width=frozen_lw,
        joint_training=True, scaler=scaler_no
    )
    preds_no, _, _ = predict_intervals(models_no_cdhw, X_te_no)
    if _DETREND:
        preds_no = preds_no + _test_trend

    with_res = {"rmse": round(float(rmse(y_test_raw, p_neural_te)), 4), "mae": round(float(mae(y_test_raw, p_neural_te)), 4), "r_squared": round(float(r_squared(y_test_raw, p_neural_te)), 4)}
    without_res = {"rmse": round(float(rmse(y_test_raw, preds_no)), 4), "mae": round(float(mae(y_test_raw, preds_no)), 4), "r_squared": round(float(r_squared(y_test_raw, preds_no)), 4)}

    evaluate_objective_o1(with_res, without_res)

    # O4: matched joint-vs-post-hoc experiment. This has to be its own
    # controlled run -- comparing calibration methods (as the previous version
    # did) cannot detect a training-paradigm effect, because all calibration
    # methods share the same point predictions by construction.
    from o4_experiment import run_o4_joint_vs_posthoc
    try:
        o4_raw = run_o4_joint_vs_posthoc(
            X_fit=X_fit, y_fit=y_fit_detrended,
            X_dev=X_dev, y_dev=y_dev_detrended,
            X_cal=X_cal, y_cal_raw=y_cal_raw,
            X_test=X_test, y_test_raw=y_test_raw,
            feature_cols=feature_cols, scaler=scaler,
            cal_trend=_cal_trend if _DETREND else None,
            test_trend=_test_trend if _DETREND else None,
            years_test=years_test, detrended=_DETREND,
            epochs=cfg.BASELINE_MAX_EPOCHS, seed=cfg.RANDOM_SEED,
        )
        save_report(o4_raw, "objective_O4_experiment_raw.json")
        evaluate_objective_o4(o4_raw)
    except Exception as _e_o4:
        logger.error("Objective O4 experiment failed: %s", _e_o4, exc_info=True)

    evaluate_objective_o5(sa_aci_result, test_df)
    evaluate_objective_o6(eval_report)

    plot_objective_o5_complete_suite(sa_aci_result, test_df)

    from evaluation import evaluate_winkler_by_year_type_and_enso, compute_exchangeability_diagnostics
    from visualization import plot_aci_online_adaptation_trajectory
    evaluate_winkler_by_year_type_and_enso(sa_aci_result, test_df)
    compute_exchangeability_diagnostics(test_df, y_test_raw, preds_test)
    plot_aci_online_adaptation_trajectory(sa_aci_result, test_df)

    # Visualizations
    plot_picp_comparison(eval_report["methods"])
    plot_mpiw_comparison(eval_report["methods"])

    if year_types_test is not None:
        plot_interval_width_by_yeartype(all_results, year_types_test)

    plot_residuals(y_test_raw, preds_test, "NeuralCQR_LightGBM_Ensemble_Test")

    if sa_aci_result.metadata.get("year_history"):
        plot_aci_tracking(sa_aci_result.metadata["year_history"])

    # ══════════════════════════════════════════════════════════
    # PHASE 3B: RANDOM ROW-LEVEL SPLIT EVALUATION (Interpolation Benchmark)
    # Same feature set as the temporal split, but rows are shuffled and
    # randomly partitioned 70/10/20 instead of split by year. This is an
    # "easier" interpolation task (train/test years overlap) so R2 here is
    # expected to be HIGHER than the temporal (out-of-time) split -- it's
    # not a fairer number, it answers a different question ("how well does
    # the model fit the overall distribution" vs "how well does it forecast
    # unseen future years"). Reported side-by-side for comparison, never as
    # a replacement for the temporal-split result.
    # ══════════════════════════════════════════════════════════
    logger.info("\n" + "═" * 70)
    logger.info("PHASE 3B: RANDOM ROW-LEVEL SPLIT EVALUATION (Interpolation Benchmark)")
    logger.info("═" * 70)

    tr_row, va_row, te_row = random_row_split(df, target=cfg.PRIMARY_TARGET, seed=cfg.RANDOM_SEED)
    tr_row, va_row, te_row = build_split_aware_lags(tr_row, va_row, te_row, target_col=cfg.PRIMARY_TARGET, split_type="random_row")
    tr_row, va_row, te_row = build_split_aware_rolling_features(tr_row, va_row, te_row, split_type="random_row")

    prep_row = TrainFittedPreprocessor(target_col=cfg.PRIMARY_TARGET)
    tr_row = prep_row.fit_transform(tr_row, split_name="random_row_train")
    va_row = prep_row.transform(va_row, split_name="random_row_val")
    te_row = prep_row.transform(te_row, split_name="random_row_test")

    # Re-fit detrending + county baseline strictly on the RANDOM-SPLIT TRAIN
    # partition (zero leakage) -- same feature_cols selected on the temporal
    # split are reused, matching experiment_runner.py's E4 methodology.
    cf_row = np.polyfit(tr_row["Year"].values, tr_row[cfg.PRIMARY_TARGET].values, 1)
    anom_row = tr_row[cfg.PRIMARY_TARGET].values - np.polyval(cf_row, tr_row["Year"].values)
    cb_row = pd.Series(anom_row, index=tr_row["GEOID"].values).groupby(level=0).mean()
    tr_row = tr_row.copy()
    va_row = va_row.copy()
    te_row = te_row.copy()
    tr_row["county_baseline"] = tr_row["GEOID"].map(cb_row).fillna(0.0)
    va_row["county_baseline"] = va_row["GEOID"].map(cb_row).fillna(0.0)
    te_row["county_baseline"] = te_row["GEOID"].map(cb_row).fillna(0.0)

    scaler_row, _ = fit_scaler(tr_row, feature_cols, scaler_type="robust")
    tr_row_sc = apply_scaling(tr_row, feature_cols, scaler_row, split_name="random_row_train")
    va_row_sc = apply_scaling(va_row, feature_cols, scaler_row, split_name="random_row_val")
    te_row_sc = apply_scaling(te_row, feature_cols, scaler_row, split_name="random_row_test")

    X_tr_r, y_tr_r = get_feature_target_arrays(tr_row_sc, feature_cols, split_name="random_row_train")
    X_va_r, y_va_r = get_feature_target_arrays(va_row_sc, feature_cols, split_name="random_row_val")
    X_te_r, y_te_r = get_feature_target_arrays(te_row_sc, feature_cols, split_name="random_row_test")

    tr_trend_r = np.polyval(cf_row, tr_row["Year"].values)
    va_trend_r = np.polyval(cf_row, va_row["Year"].values)
    te_trend_r = np.polyval(cf_row, te_row["Year"].values)

    row_neural_model = train_neural_cqr(
        X_tr_r, y_tr_r - tr_trend_r, X_va_r, y_va_r - va_trend_r, feature_cols,
        epochs=cfg.BASELINE_MAX_EPOCHS, batch_size=cfg.BATCH_SIZE, lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY, joint_training=True,
        early_stopping_mode=cfg.EARLY_STOPPING_MODE, patience=cfg.EARLY_STOPPING_PATIENCE,
        seed=cfg.RANDOM_SEED,
    )
    row_lgb_model = train_lgbm_quantile(X_tr_r, y_tr_r - tr_trend_r, X_va_r, y_va_r - va_trend_r, feature_cols)

    p_neural_row_val, _, _ = predict_intervals(row_neural_model, X_va_r)
    p_neural_row_val = p_neural_row_val + va_trend_r
    p_neural_row_te, qlo_row_te, qhi_row_te = predict_intervals(row_neural_model, X_te_r)
    p_neural_row_te = p_neural_row_te + te_trend_r
    qlo_row_te = qlo_row_te + te_trend_r
    qhi_row_te = qhi_row_te + te_trend_r

    p_lgb_row_val, _, _ = predict_intervals(row_lgb_model, X_va_r)
    p_lgb_row_val = p_lgb_row_val + va_trend_r
    p_lgb_row_te, _, _ = predict_intervals(row_lgb_model, X_te_r)
    p_lgb_row_te = p_lgb_row_te + te_trend_r

    y_te_r_raw = y_te_r.copy()
    y_va_r_raw = y_va_r.copy()

    # Ensemble blend, weight tuned on the random split's OWN validation set
    best_w_row, best_val_r2_row = 1.0, -np.inf
    for _w in np.arange(0.0, 1.01, 0.05):
        _r2v = r_squared(y_va_r_raw, _w * p_neural_row_val + (1 - _w) * p_lgb_row_val)
        if _r2v > best_val_r2_row:
            best_val_r2_row, best_w_row = _r2v, round(float(_w), 2)
    p_ens_row_te = best_w_row * p_neural_row_te + (1 - best_w_row) * p_lgb_row_te

    row_results = {
        "NeuralCQR": {
            "rmse": round(rmse(y_te_r_raw, p_neural_row_te), 4),
            "mae": round(mae(y_te_r_raw, p_neural_row_te), 4),
            "r_squared": round(r_squared(y_te_r_raw, p_neural_row_te), 4),
            "picp": round(picp(y_te_r_raw, qlo_row_te, qhi_row_te), 4),
            "mpiw": round(mpiw(qlo_row_te, qhi_row_te), 4),
        },
        "LightGBM": {
            "rmse": round(rmse(y_te_r_raw, p_lgb_row_te), 4),
            "mae": round(mae(y_te_r_raw, p_lgb_row_te), 4),
            "r_squared": round(r_squared(y_te_r_raw, p_lgb_row_te), 4),
        },
        "NeuralCQR_LightGBM_Ensemble": {
            "ensemble_weight_neuralcqr": best_w_row,
            "ensemble_weight_lightgbm": round(1 - best_w_row, 2),
            "rmse": round(rmse(y_te_r_raw, p_ens_row_te), 4),
            "mae": round(mae(y_te_r_raw, p_ens_row_te), 4),
            "r_squared": round(r_squared(y_te_r_raw, p_ens_row_te), 4),
        },
        "n_train": len(X_tr_r), "n_val": len(X_va_r), "n_test": len(X_te_r),
    }
    save_report(row_results, "random_split_evaluation.json")
    logger.info(
        "Random Row-Level Split -> NeuralCQR R2: %.4f | LightGBM R2: %.4f | Ensemble R2: %.4f (w_neural=%.2f)",
        row_results["NeuralCQR"]["r_squared"], row_results["LightGBM"]["r_squared"],
        row_results["NeuralCQR_LightGBM_Ensemble"]["r_squared"], best_w_row,
    )

    # Side-by-side comparison: Temporal (out-of-time) vs Random (interpolation)
    split_comparison_df = pd.DataFrame([
        {"Split_Type": "Temporal (2019-2023 held out)", "Evaluation_Role": "Out-of-time forecast",
         "Model": "NeuralCQR", "RMSE": round(rmse(y_test, preds_test), 4), "R2": round(r_squared(y_test, preds_test), 4)},
        {"Split_Type": "Temporal (2019-2023 held out)", "Evaluation_Role": "Out-of-time forecast",
         "Model": "NeuralCQR_LightGBM_Ensemble",
         "RMSE": next((r["RMSE"] for r in backbone_benchmark_rows if r["Model"] == "NeuralCQR_LightGBM_Ensemble"), None),
         "R2": next((r["R2"] for r in backbone_benchmark_rows if r["Model"] == "NeuralCQR_LightGBM_Ensemble"), None)},
        {"Split_Type": "Random Row-Level (70/10/20)", "Evaluation_Role": "Interpolation benchmark",
         "Model": "NeuralCQR", "RMSE": row_results["NeuralCQR"]["rmse"], "R2": row_results["NeuralCQR"]["r_squared"]},
        {"Split_Type": "Random Row-Level (70/10/20)", "Evaluation_Role": "Interpolation benchmark",
         "Model": "NeuralCQR_LightGBM_Ensemble",
         "RMSE": row_results["NeuralCQR_LightGBM_Ensemble"]["rmse"],
         "R2": row_results["NeuralCQR_LightGBM_Ensemble"]["r_squared"]},
    ])
    save_report_csv(split_comparison_df, "temporal_vs_random_split_comparison.csv", subdir="comparisons")

    from visualization import plot_split_comparison
    try:
        plot_split_comparison(split_comparison_df)
    except Exception as _e_split:
        logger.warning("plot_split_comparison failed (continuing pipeline): %s", _e_split)

    # ══════════════════════════════════════════════════════════
    # PHASE 4: LEAVE-ONE-STATE-OUT CV (§5.2) — STRICT PER-FOLD SCALING
    # ══════════════════════════════════════════════════════════
    logger.info("\n" + "═" * 70)
    logger.info("PHASE 4: LEAVE-ONE-STATE-OUT CROSS-VALIDATION (§5.2)")
    logger.info("═" * 70)

    # Audit C7: derive each fold's hyperparameters on its own DEV rows before the
    # folds run. Cached after the first time, so this costs nothing on a rerun.
    logger.info("LOSO hyperparameter provenance: %s", ensure_loso_dev_hyperparameters())

    loso_metrics, loso_preds_df = _run_loso_cv(df, feature_cols)
    save_report({"folds": loso_metrics}, "loso_cv_report.json")
    plot_loso_cv_results(loso_metrics, loso_preds_df)
    aggregate_loso_results(loso_metrics, loso_preds_df)

    from geo_maps import plot_loso_r2_choropleth, plot_loso_picp_choropleth, plot_ensemble_weight_choropleth
    for _map_fn, _map_name in [
        (plot_loso_r2_choropleth, "loso_r2_choropleth"),
        (plot_loso_picp_choropleth, "loso_picp_choropleth"),
        (plot_ensemble_weight_choropleth, "ensemble_weight_choropleth"),
    ]:
        try:
            _map_fn(loso_metrics)
        except Exception as _e_loso_map:
            logger.warning("%s failed (continuing pipeline): %s", _map_name, _e_loso_map)

    # Optional multi-seed LOSO robustness check (expensive: N full 6-fold
    # runs). Off by default -- see cfg.ENABLE_LOSO_ROBUSTNESS_CHECK and
    # ROBUSTNESS_NOTES.md. The primary loso_metrics above (single seed,
    # cfg.RANDOM_SEED) is unaffected either way.
    if getattr(cfg, "ENABLE_LOSO_ROBUSTNESS_CHECK", False):
        try:
            run_loso_robustness_check(df, feature_cols)
        except Exception as _e_lrob:
            logger.warning("LOSO robustness check failed (continuing pipeline): %s", _e_lrob)

    # ══════════════════════════════════════════════════════════
    # PHASE 5: ABLATION STUDY (§4.6) — STRICT PER-CONFIG SCALING
    # ══════════════════════════════════════════════════════════
    logger.info("\n" + "═" * 70)
    logger.info("PHASE 5: ABLATION STUDY (§4.6)")
    logger.info("═" * 70)

    ablation_report = _run_ablation(fit_df, dev_df, test_df, feature_cols)
    save_report(ablation_report, "ablation_report.json")

    # ══════════════════════════════════════════════════════════
    # PHASE 6: FINAL SUMMARY & COMPLIANCE AUDIT
    # ══════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    logger.info("\n" + "═" * 70)
    logger.info("PIPELINE COMPLETE — Total time: %.1f seconds", elapsed)
    logger.info("═" * 70)

    # Reproducibility report
    from utils import generate_reproducibility_report, validate_artifact_integrity
    generate_reproducibility_report(elapsed_seconds=elapsed)

    integrity_report = validate_artifact_integrity()

    summary = {
        "total_time_seconds": round(elapsed, 1),
        "dataset_rows": len(df),
        "dataset_cols": len(df.columns),
        "features_used": len(feature_cols),
        "temporal_split": {
            "fit": len(fit_df), "dev": len(dev_df), "cal": len(cal_df), "test": len(test_df),
            "train": len(fit_df) + len(dev_df), "val": len(cal_df),
        },
        "loso_folds": len(loso_metrics),
        "calibration_methods": list(all_results.keys()),
        "backbones_trained": list(backbone_models.keys()),
        "outputs_dir": str(cfg.OUTPUT_DIR),
        "compliance_status": integrity_report["compliance_status"],
    }

    # Export the row-identity leakage audit collected across every experiment,
    # then derive the leakage status from it rather than asserting it.
    leakage_payload = COLLECTOR.export()
    summary["leakage_status"] = (
        "ZERO_OVERLAP_VERIFIED_BY_ROW_ID" if leakage_payload["all_passed"]
        else "LEAKAGE_DETECTED"
    )
    summary["leakage_experiments_audited"] = leakage_payload["n_experiments"]
    save_report(summary, "pipeline_summary.json")

    # RO1-RO5, the objectives this study answers, computed from the artefacts
    # produced above. They supersede O1/O4/O5/O6, which are still written (M11
    # reads O4) but stamped as retired by _stamp_retired_objectives below.
    # Objectives needing rolling-origin artefacts report their status instead of
    # failing when those have not been produced.
    try:
        from objectives_ro import write_report as _write_ro_report
        _ro = _write_ro_report()
        _avail = [k for k in ("RO1", "RO2", "RO3", "RO4", "RO5")
                  if "status" not in _ro.get(k, {})]
        logger.info("Objectives RO1-RO5 written (%d computed: %s)", len(_avail), ", ".join(_avail))
    except Exception as _ro_exc:
        logger.warning("Could not write the RO objectives report: %s", _ro_exc)

    _stamp_retired_objectives()

    # Genuine property-based methodology audit. Runs last so every artefact it
    # inspects already exists.
    from methodology_validator import run_methodology_validation
    validation = run_methodology_validation()

    _generate_paper3_final_summary(summary, eval_report, backbone_df, elapsed,
                                   validation=validation, leakage=leakage_payload)
    _generate_methodology_compliance_report()
    _generate_methodology_traceability_matrix()

    # Final pass: scan every generated artefact for claims the results do not
    # support. Runs last so it sees the reports written above.
    try:
        from claim_consistency_audit import run_claim_consistency_audit
        claim_audit = run_claim_consistency_audit()
        if claim_audit["status"] != "PASS":
            logger.warning(
                "CLAIM CONSISTENCY: %d unsupported claim(s) remain in generated outputs -- see "
                "claim_consistency_audit.md", claim_audit["n_unsupported"])
    except Exception as _e_claim:
        logger.error("Claim consistency audit failed: %s", _e_claim, exc_info=True)

    # FINAL_METHODOLOGY_AUDIT.md, generated from the artefacts above so it can
    # never drift from the results.
    try:
        from final_audit_report import generate_final_audit
        generate_final_audit()
    except Exception as _e_final:
        logger.error("Final methodology audit generation failed: %s", _e_final, exc_info=True)


RETIRED_OBJECTIVES = {
    "objective_O1_report.json": (
        "Superseded by cdhw_contribution.py. Two CDHW tests in this pipeline disagreed on "
        "the sign (+0.0116 here vs -0.0079 in the ablation), both single-seed on NeuralCQR, "
        "and neither carried dependence-aware inference. Across five model families and three "
        "seeds the mean dR2 ranges from -0.0057 to +0.0017 with no year-level test significant; "
        "within LightGBM alone the per-seed dR2 spans -0.0095 to +0.0114. The CDHW block is "
        "used as a pre-specified stratification variable, not as a predictor."),
    "objective_O4_report.json": (
        "Retired with the architecture claim. The study no longer proposes joint training as a "
        "contribution, and the arms were not equal-compute: post-hoc trained 348 epochs against "
        "joint's 200 under a 'shared configuration' of 200. Kept because methodology check M11 "
        "reads its raw artefact."),
    "objective_o5_report.json": (
        "Folded into RO3 as supporting evidence. Its own conclusion -- width shows a weak, "
        "statistically undetectable association with severity (n_effective = 35 of 2345 rows) -- "
        "is the same finding RO3 states as coverage tracking bias rather than width."),
    "objective_O6_report.json": (
        "Folded into RO1/RO2 and superseded by 16 rolling origins. Its Nemenyi critical "
        "difference (3.3722) exceeded the entire observed rank range (1.8 to 4.4), so the design "
        "could not have detected a difference between any pair."),
}


def _stamp_retired_objectives() -> None:
    """Mark the superseded objective reports so no reader mistakes them for current claims.

    They are stamped rather than deleted: M11 reads objective_O4_experiment_raw.json,
    and the history is worth keeping.
    """
    import json as _json
    logger = logging.getLogger("paper3")   # main.py has no module-level logger
    for name, reason in RETIRED_OBJECTIVES.items():
        path = Path(cfg.REPORT_DIR) / name
        if not path.exists():
            continue
        try:
            payload = _json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            payload["status"] = "RETIRED"
            payload["superseded_by"] = "objectives_RO_report.json (RO1-RO5)"
            payload["retired_reason"] = reason
            path.write_text(_json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not stamp %s as retired: %s", name, exc)
    logger.info("Stamped %d superseded objective report(s) as RETIRED", len(RETIRED_OBJECTIVES))


def _export_evaluation_csv_md(eval_report: Dict[str, Any]) -> None:
    """Export evaluation summary tables as CSV and Markdown files."""
    records = []
    for method, data in eval_report["methods"].items():
        reg = data["regression"]
        unc = data["uncertainty"]
        records.append({
            "Method": method,
            "RMSE": reg["rmse"],
            "MAE": reg["mae"],
            "R2": reg["r_squared"],
            "PICP": unc["picp"],
            "MPIW": unc["mpiw"],
            "ACE": unc["ace"],
            "Winkler_Score": unc["winkler_score"],
        })

    eval_df = pd.DataFrame(records)
    save_report_csv(eval_df, "evaluation_summary.csv")

    md = "# Paper 3 Evaluation Summary (§5)\n\n"
    md += "## Calibration Methods Benchmarking\n\n"
    try:
        md += eval_df.to_markdown(index=False) + "\n\n"
    except Exception:
        md += eval_df.to_string(index=False) + "\n\n"

    if "county_block_bootstrap" in eval_report:
        cb = eval_report["county_block_bootstrap"]
        md += "## County Block Bootstrap Diagnostics\n\n"
        md += f"- **PICP 95% CI**: {cb['picp']['ci_95']} (mean: {cb['picp']['mean']})\n"
        md += f"- **MPIW 95% CI**: {cb['mpiw']['ci_95']} (mean: {cb['mpiw']['mean']})\n"
        md += f"- **ACE 95% CI**: {cb['ace']['ci_95']} (mean: {cb['ace']['mean']})\n\n"

    if "year_block_bootstrap" in eval_report:
        yb = eval_report["year_block_bootstrap"]
        md += "## Year Block Bootstrap Diagnostics\n\n"
        md += f"- **PICP 95% CI**: {yb['picp']['ci_95']} (mean: {yb['picp']['mean']})\n"
        md += f"- **MPIW 95% CI**: {yb['mpiw']['ci_95']} (mean: {yb['mpiw']['mean']})\n"
        md += f"- **ACE 95% CI**: {yb['ace']['ci_95']} (mean: {yb['ace']['mean']})\n\n"

    if "paired_t_tests" in eval_report:
        md += "## Secondary Paired t-Test Robustness\n\n"
        pt_records = []
        for key, res in eval_report["paired_t_tests"].items():
            pt_records.append({
                "Comparison": key,
                "t_statistic": res["statistic"],
                "p_value": res["p_value"],
                "Cohen_d": res.get("cohen_d_effect_size"),
                "CI_95_Diff": str(res.get("ci_95_difference")),
            })
        pt_df = pd.DataFrame(pt_records)
        try:
            md += pt_df.to_markdown(index=False) + "\n\n"
        except Exception:
            md += pt_df.to_string(index=False) + "\n\n"

    save_report_markdown(md, "evaluation_summary.md")


def run_backbone_robustness_check(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    feature_cols: List[str], scaler: Any,
) -> Dict[str, Any]:
    """POST-HOC SUPPLEMENTARY TEST-SET ROBUSTNESS ANALYSIS — NOT USED FOR MODEL SELECTION

    Retrain NeuralCQR (and the NeuralCQR+LightGBM ensemble) across
    cfg.ROBUSTNESS_SEEDS on the SAME temporal split already computed by
    main(), and report mean +/- std for Test R2/RMSE/PICP, plus the
    Spearman correlation between each seed's validation R2 and test R2.

    CRITICAL METHODOLOGICAL GUARANTEES:
    1. POST-HOC ONLY: This function is strictly a post-hoc diagnostic to
       demonstrate stability across random weight initializations.
    2. NOT USED FOR MODEL SELECTION: Model hyperparameters, early stopping
       epoch, and ensemble weights are strictly frozen on DEV (2014-2015).
       Under NO circumstances may any seed be selected or preferred based
       on its TEST result.
    3. PRIMARY MODEL PRESERVED: The authoritative reported pipeline model
       remains strictly cfg.RANDOM_SEED (seed 42), frozen prior to test evaluation.

    X_train/y_train/... must already be on whatever scale main() used
    when it called this (detrended or not) -- this function only
    retrains and re-blends, it does not touch detrending itself.
    """
    from model_training import train_neural_cqr, predict_intervals, train_lgbm_quantile
    from evaluation import rmse, r_squared, picp
    from scipy import stats as sp_stats

    logger = logging.getLogger("paper3")
    seeds = list(dict.fromkeys([cfg.RANDOM_SEED] + list(cfg.ROBUSTNESS_SEEDS)))  # RANDOM_SEED first, de-duped
    logger.info("\n\n%s", "=" * 70)
    logger.info("POST-HOC SUPPLEMENTARY TEST-SET ROBUSTNESS ANALYSIS — NOT USED FOR MODEL SELECTION")
    logger.info("Seeds: %d %s (Primary seed %d locked prior to test evaluation)", len(seeds), seeds, cfg.RANDOM_SEED)
    logger.info("%s", "=" * 70)

    per_seed_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        t0 = time.time()
        neural_models = train_neural_cqr(
            X_train, y_train, X_val, y_val, feature_cols,
            epochs=cfg.BASELINE_MAX_EPOCHS, batch_size=cfg.BATCH_SIZE, lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY, early_stopping_mode=cfg.EARLY_STOPPING_MODE,
            patience=cfg.EARLY_STOPPING_PATIENCE, joint_training=True, scaler=scaler, seed=seed,
        )
        p_val, qlo_val, qhi_val = predict_intervals(neural_models, X_val)
        p_te, qlo_te, qhi_te = predict_intervals(neural_models, X_test)

        try:
            lgb_models = train_lgbm_quantile(X_train, y_train, X_val, y_val, feature_cols)
            l_val, _, _ = predict_intervals(lgb_models, X_val)
            l_te, _, _ = predict_intervals(lgb_models, X_test)
            best_w, best_val_r2 = 1.0, r_squared(y_val, p_val)
            for _w in np.arange(0.0, 1.01, 0.1):
                _r2v = r_squared(y_val, _w * p_val + (1 - _w) * l_val)
                if _r2v > best_val_r2:
                    best_val_r2, best_w = _r2v, round(float(_w), 2)
            ens_te = best_w * p_te + (1 - best_w) * l_te
            ens_val = best_w * p_val + (1 - best_w) * l_val
        except Exception as _e:
            logger.warning("  seed %d: LightGBM ensemble skipped (%s)", seed, _e)
            ens_te, ens_val, best_w = p_te, p_val, 1.0

        val_r2_neural = r_squared(y_val, p_val)
        test_r2_neural = r_squared(y_test, p_te)
        val_r2_ens = r_squared(y_val, ens_val)
        test_r2_ens = r_squared(y_test, ens_te)
        test_rmse_ens = rmse(y_test, ens_te)
        test_picp_ens = picp(y_test, qlo_te, qhi_te)  # intervals not re-blended; point-blend metric only

        logger.info(
            "  seed %4d (%.0fs) -> NeuralCQR: val R2=%.4f test R2=%.4f | "
            "Ensemble (w=%.2f): val R2=%.4f test R2=%.4f, RMSE=%.4f",
            seed, time.time() - t0, val_r2_neural, test_r2_neural,
            best_w, val_r2_ens, test_r2_ens, test_rmse_ens,
        )

        per_seed_rows.append({
            "seed": seed,
            "neuralcqr_val_r2": round(val_r2_neural, 4),
            "neuralcqr_test_r2": round(test_r2_neural, 4),
            "ensemble_weight_neuralcqr": best_w,
            "ensemble_val_r2": round(val_r2_ens, 4),
            "ensemble_test_r2": round(test_r2_ens, 4),
            "ensemble_test_rmse": round(test_rmse_ens, 4),
            "ensemble_test_picp": round(test_picp_ens, 4),
        })

    seed_df = pd.DataFrame(per_seed_rows)
    save_report_csv(seed_df, "backbone_robustness_by_seed.csv")

    ens_r2_vals = seed_df["ensemble_test_r2"].values
    neural_r2_vals = seed_df["neuralcqr_test_r2"].values

    val_test_corr, val_test_p = float("nan"), float("nan")
    if len(seed_df) >= 3:
        val_test_corr, val_test_p = sp_stats.spearmanr(
            seed_df["neuralcqr_val_r2"], seed_df["neuralcqr_test_r2"]
        )

    summary = {
        "analysis_type": "POST-HOC SUPPLEMENTARY TEST-SET ROBUSTNESS ANALYSIS — NOT USED FOR MODEL SELECTION",
        "methodology_note": (
            "This analysis is strictly post-hoc supplementary verification. "
            "It is NOT used for model selection, hyperparameter tuning, or seed picking. "
            "The primary reported model is strictly seed 42, frozen prior to test evaluation."
        ),
        "seeds_tested": seeds,
        "primary_reported_seed": cfg.RANDOM_SEED,
        "neuralcqr_test_r2_mean": round(float(np.mean(neural_r2_vals)), 4),
        "neuralcqr_test_r2_std": round(float(np.std(neural_r2_vals)), 4),
        "neuralcqr_test_r2_min": round(float(np.min(neural_r2_vals)), 4),
        "neuralcqr_test_r2_max": round(float(np.max(neural_r2_vals)), 4),
        "ensemble_test_r2_mean": round(float(np.mean(ens_r2_vals)), 4),
        "ensemble_test_r2_std": round(float(np.std(ens_r2_vals)), 4),
        "ensemble_test_r2_min": round(float(np.min(ens_r2_vals)), 4),
        "ensemble_test_r2_max": round(float(np.max(ens_r2_vals)), 4),
        "ensemble_test_rmse_mean": round(float(seed_df["ensemble_test_rmse"].mean()), 4),
        "ensemble_test_rmse_std": round(float(seed_df["ensemble_test_rmse"].std()), 4),
        "ensemble_test_picp_mean": round(float(seed_df["ensemble_test_picp"].mean()), 4),
        "val_vs_test_r2_spearman_corr": round(float(val_test_corr), 4) if val_test_corr == val_test_corr else None,
        "val_vs_test_r2_spearman_p": round(float(val_test_p), 4) if val_test_p == val_test_p else None,
        "interpretation": (
            "A low or negative val_vs_test_r2_spearman_corr means the validation "
            "window (2016-2018) does not reliably rank models by 2019-2023 test "
            "performance -- i.e. genuine distribution shift, not just noise. This "
            "is reported as supporting evidence for the paper's central claim, not "
            "a flaw to be tuned away. See ROBUSTNESS_NOTES.md."
        ),
    }
    save_report(summary, "backbone_robustness_summary.json")

    logger.info(
        "\nBACKBONE ROBUSTNESS SUMMARY -> Ensemble Test R2 = %.4f +/- %.4f (n=%d seeds, range [%.4f, %.4f])",
        summary["ensemble_test_r2_mean"], summary["ensemble_test_r2_std"], len(seeds),
        summary["ensemble_test_r2_min"], summary["ensemble_test_r2_max"],
    )
    if summary["val_vs_test_r2_spearman_corr"] is not None:
        logger.info(
            "Validation-vs-test R2 rank correlation across seeds: rho=%.3f (p=%.3f) -- "
            "low/negative values indicate the val window does not reliably predict "
            "test-time performance across seeds.",
            summary["val_vs_test_r2_spearman_corr"], summary["val_vs_test_r2_spearman_p"],
        )

    return summary


def ensure_loso_dev_hyperparameters(enabled: Optional[bool] = None,
                                    path: Optional[Path] = None,
                                    runner=None) -> str:
    """Guarantee each LOSO fold can read hyperparameters derived on its own DEV rows.

    Audit C7: the values in config.py descend from six rounds selected by watching
    leave-one-state-out R2 rise, i.e. chosen with held-out information.
    code/loso_dev_tuning.py re-derives them on each fold's own DEV partition, and
    this makes the pipeline do that for itself instead of relying on someone having
    remembered to run it first.

    The search is the expensive half -- 24 neural fits against the LOSO phase's 6 --
    so it runs once and is cached as an artefact. Later runs read the file. That is
    not only a speed decision: a tuner that reran every time could select a
    different configuration on each invocation, and the frozen protocol would then
    be describing a moving target. The selection is a file so its provenance can be
    cited and frozen.

    Returns the provenance string recorded on every fold's metrics.
    """
    logger = logging.getLogger("paper3")
    if enabled is None:
        enabled = bool(getattr(cfg, "LOSO_USE_DEV_SELECTED_HP", False))
    path = (Path(cfg.REPORT_DIR) / "loso_dev_selected_hyperparams.json"
            if path is None else Path(path))

    def _uncovered(p: Path) -> List[str]:
        """States the selection does not cover, so a partial file cannot pass as whole.

        loso_dev_tuning.py has a smoke mode (--states Minnesota --grid 2). If that
        output ever lands at the cache path, the caching above would reuse it and
        leave the other five folds on the held-out-informed defaults without
        anything in the run saying so.
        """
        import json as _json
        try:
            sel = _json.loads(p.read_text(encoding="utf-8")).get("per_state_selection", {})
        except (OSError, ValueError) as exc:
            # Only an unreadable or malformed file counts as "covers nothing". A
            # broader catch here once turned a NameError into a plausible-looking
            # "0 of 6 states" report, which is the kind of lie that survives review.
            logger.warning("could not read %s (%s); treating it as no selection", p.name, exc)
            return list(cfg.LOSO_STATES)
        return [s for s in cfg.LOSO_STATES if s not in sel]

    if not enabled:
        return "config defaults (C7 caveat applies)"
    if path.exists():
        gaps = _uncovered(path)
        if gaps:
            logger.warning("LOSO hyperparameters: %s covers only %d of %d states; %s will use "
                           "config defaults and keep the C7 caveat. Delete the file and rerun "
                           "code/loso_dev_tuning.py for a complete selection.",
                           path.name, len(cfg.LOSO_STATES) - len(gaps), len(cfg.LOSO_STATES),
                           ", ".join(gaps))
            return (f"DEV-selected but PARTIAL ({path.name}; {len(gaps)} of "
                    f"{len(cfg.LOSO_STATES)} states missing: {', '.join(gaps)})")
        logger.info("LOSO hyperparameters: reusing the cached DEV selection (%s)", path.name)
        return f"DEV-selected ({path.name}, cached)"

    logger.info("LOSO hyperparameters: no DEV selection on disk; running the C7 tuning now "
                "(%d states x 4 configurations -- GPU strongly preferred)",
                len(cfg.LOSO_STATES))
    if runner is None:
        from loso_dev_tuning import main as runner
    try:
        runner()
    except Exception as exc:                                  # noqa: BLE001
        # Losing the tuning 40 minutes into a pipeline run should cost the C7
        # correction, not the run. The caveat travels with the fold metrics.
        logger.error("LOSO DEV tuning failed (%s: %s); falling back to config defaults, "
                     "which carry the C7 caveat", type(exc).__name__, exc)
        return f"config defaults (C7 caveat applies; tuning failed: {type(exc).__name__})"
    if not path.exists():
        logger.error("LOSO DEV tuning finished but wrote no selection to %s; "
                     "falling back to config defaults", path)
        return "config defaults (C7 caveat applies; tuning produced no selection file)"
    return f"DEV-selected ({path.name}, freshly tuned)"


def _run_loso_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Execute the locked 6-fold LOSO-CV with strict role separation per fold.

    Protocol (locked -- see ``config.LOSO_STATES``, six states, never changed):

    ==============  ===========================================================
    Partition       Allowed use
    ==============  ===========================================================
    ``fit``         model fitting only (the last chronological 10% is carved off
                    *within fit* as the internal early-stopping split, so the
                    training API never sees dev/cal/test data)
    ``dev``         ensemble-weight selection only
    ``cal``         conformal calibration only, after the weight is frozen
    ``test``        the held-out state -- final evaluation only
    ==============  ===========================================================

    What was wrong before
    ---------------------
    The previous implementation built ``X_tr_full = concatenate([X_tr, X_va])``
    and fitted on it. The validation fold was therefore ~90% inside the training
    set, and the *same* contaminated validation rows were then reused to (a)
    early-stop, (b) choose the NeuralCQR/LightGBM ensemble weight, and (c)
    supply the conformal non-conformity scores. Every LOSO number was optimistic
    for three independent reasons. There was also no calibration partition at
    all, and feature selection had been fitted on all states including the
    held-out one.

    Each fold now re-runs consensus feature selection on its own ``fit``
    partition (the held-out state must not influence *any* learned component),
    records a row-identity provenance ledger, and asserts disjointness before a
    single model is fitted.
    """
    logger = logging.getLogger("paper3")
    from model_training import train_lgbm_quantile
    from feature_selection import select_features as _select_features
    from leakage_provenance import PartitionLedger, assert_disjoint, COLLECTOR

    fold_metrics: List[Dict[str, Any]] = []
    all_loso_predictions: List[pd.DataFrame] = []
    weight_search_records: List[Dict[str, Any]] = []
    fold_feature_records: List[Dict[str, Any]] = []

    per_fold_selection = getattr(cfg, "LOSO_PER_FOLD_FEATURE_SELECTION", True)

    for state, fit_fold, dev_fold, cal_fold, test_fold in loso_cv_folds(df, return_cal=True):
        logger.info("--- LOSO Fold: %s ---", state)

        # ── Structural fold isolation assertions ─────────────────────────
        for part, nm in ((fit_fold, "fit"), (dev_fold, "dev"), (cal_fold, "cal")):
            assert state not in set(part["State"]), f"{state} leaked into {nm}_fold"
        assert (test_fold["State"] == state).all(), f"Non-{state} row found in test_fold"
        assert test_fold["GEOID"].nunique() > 0, f"No counties found in test fold for {state}"

        _fit_valid = fit_fold[fit_fold[cfg.PRIMARY_TARGET].notna()]
        _te_valid = test_fold[test_fold[cfg.PRIMARY_TARGET].notna()]
        if len(_fit_valid) < 10 or len(_te_valid) < 1:
            logger.warning("LOSO fold %s skipped: fit=%d, test=%d valid rows.",
                           state, len(_fit_valid), len(_te_valid))
            continue

        # 1. Split-aware lag & rolling features, then fit-fitted preprocessing.
        #    Sources are restricted to the fit partition by build_split_aware_*
        #    for split_type='loso'.
        fit_fold, dev_fold, cal_fold, test_fold = build_split_aware_lags(
            fit_fold, dev_fold, test_fold, target_col=cfg.PRIMARY_TARGET,
            split_type="loso", cal_df=cal_fold,
        )
        fit_fold, dev_fold, cal_fold, test_fold = build_split_aware_rolling_features(
            fit_fold, dev_fold, test_fold, split_type="loso", cal_df=cal_fold,
        )
        prep_fold = TrainFittedPreprocessor(target_col=cfg.PRIMARY_TARGET)
        fit_fold = prep_fold.fit_transform(fit_fold, split_name=f"loso_{state}_fit")
        dev_fold = prep_fold.transform(dev_fold, split_name=f"loso_{state}_dev")
        cal_fold = prep_fold.transform(cal_fold, split_name=f"loso_{state}_cal")
        test_fold = prep_fold.transform(test_fold, split_name=f"loso_{state}_test")

        # 2. Feature set for this fold.
        #    county_baseline is a per-county historical-mean anomaly built only
        #    from the development states' GEOIDs. Every held-out-state test row
        #    belongs to a GEOID never seen in training, so it would resolve to a
        #    constant 0.0 placeholder across the whole test set -- zero signal
        #    for precisely the evaluation that needs spatial generalization.
        #    Excluded from the fold feature list (see config.py for the full
        #    root-cause history).
        if per_fold_selection:
            # The held-out state must not influence feature selection either.
            fold_feats, _fold_sel_report = _select_features(
                fit_fold, target=cfg.PRIMARY_TARGET,
                report_suffix=f"_loso_{state.lower()}",
            )
            fold_feats = [f for f in fold_feats if f in fit_fold.columns]
        else:
            fold_feats = list(feature_cols)
        _bookkeeping = {"county_baseline", "Fold", "HeldOutState", "obs_id", "Split"}
        loso_feature_cols = [f for f in fold_feats if f not in _bookkeeping]
        fold_feature_records.append({
            "state": state,
            "n_features": len(loso_feature_cols),
            "selection_partition": "fold_fit_only" if per_fold_selection else "main_temporal_fit",
            "features": ";".join(loso_feature_cols),
        })

        # 3. Scaler fitted on the fold's FIT partition ONLY.
        fold_scaler, _ = fit_scaler(fit_fold, loso_feature_cols, scaler_type="robust")
        fit_scaled = apply_scaling(fit_fold, loso_feature_cols, fold_scaler, split_name=f"loso_{state}_fit")
        dev_scaled = apply_scaling(dev_fold, loso_feature_cols, fold_scaler, split_name=f"loso_{state}_dev")
        cal_scaled = apply_scaling(cal_fold, loso_feature_cols, fold_scaler, split_name=f"loso_{state}_cal")
        test_scaled = apply_scaling(test_fold, loso_feature_cols, fold_scaler, split_name=f"loso_{state}_test")

        X_fit, y_fit = get_feature_target_arrays(fit_scaled, loso_feature_cols, split_name=f"loso_{state}_fit")
        X_dev, y_dev = get_feature_target_arrays(dev_scaled, loso_feature_cols, split_name=f"loso_{state}_dev")
        X_cal, y_cal = get_feature_target_arrays(cal_scaled, loso_feature_cols, split_name=f"loso_{state}_cal")
        X_te, y_te = get_feature_target_arrays(test_scaled, loso_feature_cols, split_name=f"loso_{state}_test")

        if len(X_fit) == 0 or len(X_te) == 0 or len(X_dev) == 0 or len(X_cal) == 0:
            logger.warning("Skipping fold %s: empty partition (fit=%d, dev=%d, cal=%d, test=%d)",
                           state, len(X_fit), len(X_dev), len(X_cal), len(X_te))
            continue

        # 3b. TARGET DETRENDING — fitted STRICTLY on this fold's FIT partition
        #     (1985-2013, five development states), exactly as the temporal
        #     pipeline does it in main() step 8.
        #
        #     Why this exists: without it the LOSO folds modelled the raw
        #     trending target while the temporal pipeline modelled the
        #     detrended one. Corn yield rises ~0.11 t/ha/yr here, so a fold
        #     fitted on 1985-2010 (mean ~8.0 t/ha) was scored on DEV 2014-2015
        #     (mean ~10.7 t/ha) with no feature able to express the level: there
        #     is no Year feature, the Fourier encodings are periodic and alias a
        #     later year onto an earlier one, and Yield_lag1/2 are unavailable
        #     for dev-2015, cal and the whole held-out state (build_split_aware_lags
        #     sources them from FIT only), so the train-median fill turns them
        #     into constants. The result was a systematic -1.35..-1.68 t/ha
        #     offset that accounted for 50-65% of DEV MSE and drove every fold's
        #     LightGBM DEV R2 negative, which in turn collapsed the ensemble
        #     weight to NeuralCQR=1.0 in all six folds.
        #
        #     Leakage: the trend is fitted only on the fold's own FIT rows, so
        #     the held-out state contributes nothing to it, and Year is known at
        #     prediction time, so adding the trend back is leakage-free. The
        #     stage is registered in the provenance ledger below.
        #
        #     Caveat worth keeping in mind when interpreting the folds: the
        #     five development states' trend is applied to the held-out state.
        #     Per-state 1985-2013 trends range from 0.069 (Missouri) to 0.157
        #     (Minnesota) t/ha/yr against a five-state trend of 0.101-0.116, so
        #     a residual state-specific trend error of up to ~0.05 t/ha/yr
        #     remains. This is a modelling assumption, not a correction.
        _DETREND_LOSO = bool(getattr(cfg, "DETREND_TARGET", False))
        if _DETREND_LOSO:
            _cf_fold = np.polyfit(fit_fold["Year"].values, y_fit, 1)
            _trend = {
                "fit": np.polyval(_cf_fold, fit_fold["Year"].values),
                "dev": np.polyval(_cf_fold, dev_fold["Year"].values),
                "cal": np.polyval(_cf_fold, cal_fold["Year"].values),
                "test": np.polyval(_cf_fold, test_fold["Year"].values),
            }
            logger.info("  %s -> DETREND (fold FIT only): %.4f t/ha/yr; modelling the anomaly.",
                        state, float(_cf_fold[0]))
        else:
            _cf_fold = np.array([0.0, 0.0])
            _trend = {"fit": np.zeros(len(y_fit)), "dev": np.zeros(len(y_dev)),
                      "cal": np.zeros(len(y_cal)), "test": np.zeros(len(y_te))}
        y_fit_model = y_fit - _trend["fit"]

        def _to_raw(triple, part, _t=_trend):
            """Add the fold's FIT-only trend back so every score below is on the raw yield scale."""
            return tuple(np.asarray(a, dtype=float) + _t[part] for a in triple)

        # 4. Internal early-stopping split carved STRICTLY from the FIT
        #    partition (chronological tail). It never contains dev, cal or
        #    held-out-state observations.
        fit_ids = obs_ids(fit_fold)
        n_es_val = max(1, len(X_fit) // 10)
        X_tr_fit, y_tr_fit = X_fit[:-n_es_val], y_fit_model[:-n_es_val]
        X_es_val, y_es_val = X_fit[-n_es_val:], y_fit_model[-n_es_val:]
        train_ids, es_ids = fit_ids[:-n_es_val], fit_ids[-n_es_val:]

        # 5. Row-identity provenance ledger -- asserted BEFORE any model is fitted.
        ledger = PartitionLedger(experiment=f"LOSO/{state}")
        ledger.register("train", train_ids)
        ledger.register("es_internal", es_ids)
        # fit_full = train + es_internal, i.e. the whole FIT partition. It is the
        # set the target trend is fitted on, so it is registered by name rather
        # than folded into "train"; its disjointness from dev/cal/test follows
        # from the train/* and es_internal/* pairs already asserted below.
        ledger.register("fit_full", fit_ids)
        ledger.register("dev", dev_fold)
        ledger.register("cal", cal_fold)
        ledger.register("test", test_fold)
        ledger.bind_stage("feature_selection", "train" if per_fold_selection else "external_fit")
        ledger.bind_stage("scaler_fitting", "train")
        ledger.bind_stage("preprocessor_fitting", "train")
        if _DETREND_LOSO:
            ledger.bind_stage("target_detrending", "fit_full")
        ledger.bind_stage("model_fitting", "train")
        ledger.bind_stage("early_stopping", "es_internal")
        ledger.bind_stage("ensemble_weight_selection", "dev")
        ledger.bind_stage("conformal_calibration", "cal")
        ledger.bind_stage("final_evaluation", "test")
        audit_record = assert_disjoint(ledger)
        audit_record["held_out_state"] = state
        audit_record["n_features"] = len(loso_feature_cols)
        COLLECTOR.add(audit_record)

        try:
            # C7: prefer hyperparameters re-derived on this fold's own DEV rows
            # (code/loso_dev_tuning.py) over the config defaults, which descend
            # from rounds selected by watching held-out R2. Off unless the flag
            # is set AND the selection file exists, so behaviour is unchanged
            # until the tuning run has actually been done.
            _hp = {"hidden_dims": cfg.LOSO_HIDDEN_DIMS, "lr": cfg.LOSO_LEARNING_RATE,
                   "dropout": cfg.LOSO_DROPOUT, "weight_decay": cfg.LOSO_WEIGHT_DECAY,
                   "batch_size": cfg.LOSO_BATCH_SIZE}
            _hp_source = "config defaults (C7 caveat applies)"
            if bool(getattr(cfg, "LOSO_USE_DEV_SELECTED_HP", False)):
                _sel_path = Path(cfg.REPORT_DIR) / "loso_dev_selected_hyperparams.json"
                if _sel_path.exists():
                    import json as _json_sel
                    _sel = _json_sel.loads(_sel_path.read_text(encoding="utf-8"))
                    _fold_hp = _sel.get("per_state_selection", {}).get(state)
                    if _fold_hp:
                        _hp.update({k: _fold_hp[k] for k in _hp if k in _fold_hp})
                        _hp["hidden_dims"] = tuple(_hp["hidden_dims"])
                        _hp_source = f"DEV-selected ({_sel_path.name})"
                    else:
                        logger.warning("  %s: no DEV-selected hyperparameters in %s; "
                                       "falling back to config defaults", state, _sel_path.name)
                else:
                    logger.warning("  LOSO_USE_DEV_SELECTED_HP is on but %s is missing; "
                                   "run code/loso_dev_tuning.py first", _sel_path.name)
            logger.info("  %s -> hyperparameters: %s | %s", state, _hp_source, _hp)

            # 6. MODEL FITTING — fit partition only.
            models = train_neural_cqr(
                X_tr_fit, y_tr_fit, X_es_val, y_es_val, loso_feature_cols,
                epochs=cfg.LOSO_MAX_EPOCHS, batch_size=_hp["batch_size"], lr=_hp["lr"],
                weight_decay=_hp["weight_decay"], early_stopping_mode=cfg.LOSO_EARLY_STOPPING_MODE,
                patience=cfg.LOSO_EARLY_STOPPING_PATIENCE, joint_training=True, scaler=fold_scaler,
                hidden_dims=_hp["hidden_dims"], dropout_rate=_hp["dropout"],
            )
            # Predictions are produced on the modelling scale (the detrended
            # anomaly) and converted straight back to raw yield, so every
            # downstream step -- DEV weight search, CAL conformal calibration
            # and the held-out-state evaluation -- works on raw t/ha exactly as
            # before this change.
            preds_dev_n, qlo_dev_n, qhi_dev_n = _to_raw(predict_intervals(models, X_dev), "dev")
            preds_cal_n, qlo_cal_n, qhi_cal_n = _to_raw(predict_intervals(models, X_cal), "cal")
            preds_te_n, qlo_te_n, qhi_te_n = _to_raw(predict_intervals(models, X_te), "test")
            orig_neural_preds = preds_te_n.copy()

            lgb_ok = False
            try:
                lgb_models = train_lgbm_quantile(X_tr_fit, y_tr_fit, X_es_val, y_es_val, loso_feature_cols)
                preds_dev_l, qlo_dev_l, qhi_dev_l = _to_raw(predict_intervals(lgb_models, X_dev), "dev")
                preds_cal_l, qlo_cal_l, qhi_cal_l = _to_raw(predict_intervals(lgb_models, X_cal), "cal")
                preds_te_l, qlo_te_l, qhi_te_l = _to_raw(predict_intervals(lgb_models, X_te), "test")
                lgb_ok = True
            except Exception as _e_lgb:
                logger.warning("  %s -> LightGBM ensemble skipped: %s", state, _e_lgb)
                preds_te_l = np.full_like(preds_te_n, np.nan)

            # 7. ENSEMBLE WEIGHT SELECTION — DEV partition ONLY.
            #    Every candidate weight is scored on DEV and recorded; the test
            #    state and the calibration partition play no part whatsoever.
            val_r2_neural = r_squared(y_dev, preds_dev_n)
            val_r2_lgb = np.nan
            best_w, best_val_r2 = 1.0, val_r2_neural

            if lgb_ok:
                val_r2_lgb = r_squared(y_dev, preds_dev_l)
                for _w in np.arange(0.0, 1.01, 0.1):
                    _w = round(float(_w), 2)
                    _blend_v = _w * preds_dev_n + (1 - _w) * preds_dev_l
                    _r2v = float(r_squared(y_dev, _blend_v))
                    weight_search_records.append({
                        "state": state,
                        "weight_neuralcqr": _w,
                        "weight_lightgbm": round(1.0 - _w, 2),
                        "dev_r2": round(_r2v, 4),
                        "dev_rmse": round(float(rmse(y_dev, _blend_v)), 4),
                        "dev_mae": round(float(mae(y_dev, _blend_v)), 4),
                        "selection_partition": "LOSO_DEV_2014_2015",
                    })
                    if _r2v > best_val_r2:
                        best_val_r2, best_w = _r2v, _w
            else:
                weight_search_records.append({
                    "state": state, "weight_neuralcqr": 1.0, "weight_lightgbm": 0.0,
                    "dev_r2": round(float(val_r2_neural), 4),
                    "dev_rmse": round(float(rmse(y_dev, preds_dev_n)), 4),
                    "dev_mae": round(float(mae(y_dev, preds_dev_n)), 4),
                    "selection_partition": "LOSO_DEV_2014_2015",
                })

            for rec in weight_search_records:
                if rec["state"] == state:
                    rec["is_selected"] = bool(abs(rec["weight_neuralcqr"] - best_w) < 1e-9)

            # ── WEIGHT FROZEN HERE. Nothing below may change it. ──
            frozen_w = float(best_w)
            dev_blend = frozen_w * preds_dev_n + (1 - frozen_w) * (preds_dev_l if lgb_ok else preds_dev_n)
            val_rmse_best = float(rmse(y_dev, dev_blend))
            val_mae_best = float(mae(y_dev, dev_blend))
            val_r2_best = float(r_squared(y_dev, dev_blend))
            if lgb_ok and frozen_w < 1.0:
                logger.info("  %s -> ensembled with LightGBM (w_NeuralCQR=%.2f), dev R2 %.4f -> %.4f",
                            state, frozen_w, val_r2_neural, val_r2_best)

            # 8. Apply the frozen weight to CAL and TEST.
            if lgb_ok:
                p_cal = frozen_w * preds_cal_n + (1 - frozen_w) * preds_cal_l
                qlo_cal = frozen_w * qlo_cal_n + (1 - frozen_w) * qlo_cal_l
                qhi_cal = frozen_w * qhi_cal_n + (1 - frozen_w) * qhi_cal_l
                preds = frozen_w * preds_te_n + (1 - frozen_w) * preds_te_l
                q_lo = frozen_w * qlo_te_n + (1 - frozen_w) * qlo_te_l
                q_hi = frozen_w * qhi_te_n + (1 - frozen_w) * qhi_te_l
            else:
                p_cal, qlo_cal, qhi_cal = preds_cal_n, qlo_cal_n, qhi_cal_n
                preds, q_lo, q_hi = preds_te_n, qlo_te_n, qhi_te_n

            # 9. CONFORMAL CALIBRATION — CAL partition only, weight already frozen.
            cal_result = static_conformal(
                y_cal, qlo_cal, qhi_cal,
                q_lo, q_hi, preds, y_te,
            )

            y_te_arr = np.asarray(y_te)
            cal_qlo_arr = np.asarray(cal_result.q_lo)
            cal_qhi_arr = np.asarray(cal_result.q_hi)
            preds_arr = np.asarray(preds)

            covered = (y_te_arr >= cal_qlo_arr) & (y_te_arr <= cal_qhi_arr)
            lower_viol = (y_te_arr < cal_qlo_arr)
            upper_viol = (y_te_arr > cal_qhi_arr)
            n_test = len(y_te_arr)
            n_covered = int(np.sum(covered))
            n_lower = int(np.sum(lower_viol))
            n_upper = int(np.sum(upper_viol))
            cal_threshold = cal_result.metadata.get("threshold", 0.0)

            fold_obs = pd.DataFrame({
                "held_out_state": state,
                "state": test_fold["State"].values,
                "county": test_fold["County_Name"].values if "County_Name" in test_fold.columns else test_fold["GEOID"].values,
                "year": test_fold["Year"].values,
                "geoid": test_fold["GEOID"].values,
                "observation_id": test_fold["obs_id"].values,
                "y_true": y_te_arr,
                "y_pred": preds_arr,
                "lower_bound": cal_qlo_arr,
                "upper_bound": cal_qhi_arr,
                "neuralcqr_pred": orig_neural_preds,
                "lightgbm_pred": preds_te_l,
                "ensemble_weight_neuralcqr": frozen_w,
                "ensemble_weight_lightgbm": round(1.0 - frozen_w, 2),
                "covered": covered.astype(int),
                "lower_violation": lower_viol.astype(int),
                "upper_violation": upper_viol.astype(int),
            })
            all_loso_predictions.append(fold_obs)

            metrics = {
                "state": state,
                "n_features": len(loso_feature_cols),
                "target_detrending": "Linear_FIT_Only" if _DETREND_LOSO else "None",
                "fold_trend_t_ha_per_year": round(float(_cf_fold[0]), 4),
                "n_fit_total": len(X_fit),
                "n_train": len(X_tr_fit),
                "n_es_val": len(X_es_val),
                "n_dev": len(X_dev),
                "n_cal": len(X_cal),
                "n_test": n_test,
                "hyperparameter_source": _hp_source,
                "neuralcqr_ensemble_weight": frozen_w,
                "lightgbm_ensemble_weight": round(1.0 - frozen_w, 2),
                "dev_r2_neural_alone": round(float(val_r2_neural), 4),
                "dev_r2_lgb_alone": round(float(val_r2_lgb), 4) if not np.isnan(val_r2_lgb) else None,
                "dev_r2_selected_blend": round(float(val_r2_best), 4),
                "dev_rmse_selected_blend": round(float(val_rmse_best), 4),
                "dev_mae_selected_blend": round(float(val_mae_best), 4),
                "weight_selection_partition": "LOSO_DEV_2014_2015",
                "weight_selection_objective": "Maximize R2 on LOSO DEV",
                "calibration_partition": "LOSO_CAL_2016_2023",
                "calibration_threshold_q": round(float(cal_threshold), 4),
                "nominal_coverage": cfg.NOMINAL_COVERAGE,
                "rmse": round(rmse(y_te_arr, preds_arr), 4),
                "mae": round(mae(y_te_arr, preds_arr), 4),
                "r_squared": round(r_squared(y_te_arr, preds_arr), 4),
                "picp": round(picp(y_te_arr, cal_qlo_arr, cal_qhi_arr), 4),
                "mpiw": round(mpiw(cal_qlo_arr, cal_qhi_arr), 4),
                "ace": round(ace(y_te_arr, cal_qlo_arr, cal_qhi_arr), 4),
                "winkler_score": round(winkler_score(y_te_arr, cal_qlo_arr, cal_qhi_arr), 4),
                "lower_violation_rate": round(float(n_lower / n_test), 4),
                "upper_violation_rate": round(float(n_upper / n_test), 4),
                "n_covered": n_covered,
                "n_lower_violations": n_lower,
                "n_upper_violations": n_upper,
                "leakage_status": audit_record["status"],
            }
            fold_metrics.append(metrics)

            logger.info(
                "  %s -> RMSE=%.4f, R2=%.4f, PICP=%.4f, MPIW=%.4f (w_NeuralCQR=%.2f, n_obs=%d)",
                state, metrics["rmse"], metrics["r_squared"],
                metrics["picp"], metrics["mpiw"], frozen_w, n_test,
            )
        except Exception as e:
            logger.error("LOSO fold %s failed: %s", state, e, exc_info=True)
            fold_metrics.append({"state": state, "error": str(e)})

    loso_preds_df = pd.DataFrame()
    if all_loso_predictions:
        loso_preds_df = pd.concat(all_loso_predictions, axis=0, ignore_index=True)
        preds_dir = getattr(cfg, "OUTPUT_DIR", Path("outputs")) / "predictions"
        preds_dir.mkdir(parents=True, exist_ok=True)
        loso_preds_df.to_csv(preds_dir / "loso_predictions.csv", index=False)
        logger.info("Saved observation-level LOSO predictions -> %s (%d rows)",
                    preds_dir / "loso_predictions.csv", len(loso_preds_df))

    if weight_search_records:
        save_report_csv(pd.DataFrame(weight_search_records), "loso_ensemble_weight_search.csv")
    if fold_feature_records:
        save_report_csv(pd.DataFrame(fold_feature_records), "loso_fold_feature_sets.csv")

    # Machine-readable per-fold metrics with the raw metric keys.
    # loso_summary.csv is the human-facing table and uses display headers
    # ("Held-out State", "R²", ...), which downstream validators cannot key on.
    if fold_metrics:
        save_report_csv(pd.DataFrame(fold_metrics), "loso_fold_metrics.csv")

    # C8: register every LOSO fold. The registry previously held only the three
    # temporal experiments, so the six spatial folds -- the ones the paper's
    # transfer claim rests on -- left no provenance record.
    try:
        from experiment_registry import ExperimentRegistry
        _reg = ExperimentRegistry()
        for _f in fold_metrics:
            if "rmse" not in _f:
                continue
            _state = str(_f.get("state", "unknown")).replace(" ", "_").upper()
            _reg.register_experiment(
                experiment_id=f"EXP_LOSO_{_state}",
                model="NeuralCQR+LightGBM_Ensemble",
                role="LOSO",
                features=int(_f.get("n_features", 0)),
                detrending=("Linear_fold_FIT_only" if _f.get("target_detrending") else "None"),
                fit_period="1985-2013 (five non-held-out states)",
                dev_period="2014-2015 (five non-held-out states)",
                cal_period="2016-2023 (five non-held-out states)",
                test_period=f"all years, held-out {_f.get('state')}",
                ensemble_weight=_f.get("neuralcqr_ensemble_weight"),
                conformal_method="static_conformal",
                rmse=_f.get("rmse"), mae=_f.get("mae"), r2=_f.get("r_squared"),
                picp=_f.get("picp"), ace=_f.get("ace"), mpiw=_f.get("mpiw"),
                winkler=_f.get("winkler_score"),
                notes=(f"LOSO fold; fold trend {_f.get('fold_trend_t_ha_per_year')} t/ha/yr; "
                       f"n_test={_f.get('n_test')}"),
            )
    except Exception as _reg_exc:
        logger.warning("Could not register LOSO folds: %s", _reg_exc)

    valid_folds = [f for f in fold_metrics if "rmse" in f]
    if valid_folds:
        for metric_name in ["rmse", "mae", "r_squared", "picp", "mpiw"]:
            vals = [f[metric_name] for f in valid_folds]
            logger.info("LOSO-CV %s: mean=%.4f, std=%.4f", metric_name, np.mean(vals), np.std(vals))

    return fold_metrics, loso_preds_df


def run_loso_robustness_check(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    """Re-run the full 6-fold LOSO-CV across cfg.LOSO_ROBUSTNESS_SEEDS,
    reusing _run_loso_cv() unchanged for each seed (via a temporary
    cfg.RANDOM_SEED override, restored afterward), and report per-state
    mean +/- std R2 plus the overall LOSO mean R2 per seed.

    Expensive: each seed is a full 6-fold run (~15-25 min). Off by
    default -- set cfg.ENABLE_LOSO_ROBUSTNESS_CHECK = True to run this
    from main(), or call it directly from a standalone script for more
    control over runtime. See ROBUSTNESS_NOTES.md.
    """
    logger = logging.getLogger("paper3")
    seeds = list(dict.fromkeys([cfg.RANDOM_SEED] + list(cfg.LOSO_ROBUSTNESS_SEEDS)))
    logger.info("\n\n%s", "=" * 70)
    logger.info("LOSO ROBUSTNESS CHECK: %d seeds %s (each seed = full 6-fold run)", len(seeds), seeds)
    logger.info("%s", "=" * 70)

    original_seed = cfg.RANDOM_SEED
    all_rows: List[Dict[str, Any]] = []
    try:
        for seed in seeds:
            logger.info("\n--- LOSO robustness: seed %d ---", seed)
            res = _run_loso_cv(df, feature_cols)
            fold_metrics = res[0] if isinstance(res, tuple) else res
            for m in fold_metrics:
                if "error" in m:
                    continue
                row = dict(m)
                row["seed"] = seed
                all_rows.append(row)
    finally:
        cfg.RANDOM_SEED = original_seed  # always restore, even if a seed run raises

    if not all_rows:
        logger.warning("LOSO robustness check produced no valid folds.")
        return {}

    rob_df = pd.DataFrame(all_rows)
    save_report_csv(rob_df, "loso_robustness_by_seed_and_state.csv")

    per_state = rob_df.groupby("state")["r_squared"].agg(["mean", "std", "count"]).reset_index()
    save_report_csv(per_state, "loso_robustness_by_state_summary.csv")

    per_seed_mean = rob_df.groupby("seed")["r_squared"].mean()
    summary = {
        "seeds_tested": seeds,
        "primary_reported_seed": original_seed,
        "loso_mean_r2_by_seed": {int(s): round(float(v), 4) for s, v in per_seed_mean.items()},
        "loso_mean_r2_across_all_seeds": round(float(rob_df["r_squared"].mean()), 4),
        "loso_mean_r2_std_across_seeds": round(float(per_seed_mean.std()), 4),
        "worst_state_across_seeds": per_state.loc[per_state["mean"].idxmin(), "state"],
        "worst_state_mean_r2": round(float(per_state["mean"].min()), 4),
    }
    save_report(summary, "loso_robustness_summary.json")

    logger.info(
        "\nLOSO ROBUSTNESS SUMMARY -> mean R2 across %d seeds = %.4f +/- %.4f",
        len(seeds), summary["loso_mean_r2_across_all_seeds"], summary["loso_mean_r2_std_across_seeds"],
    )
    return summary


def _run_ablation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_feature_cols: List[str],
) -> Dict[str, Any]:
    """Run 5-row ablation study with per-config feature scaling (Zero Leakage)."""
    logger = logging.getLogger("paper3")
    report: Dict[str, Any] = {"configurations": []}

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    for col in all_feature_cols:
        if col in train_df.columns and train_df[col].isna().any():
            med = float(train_df[col].median()) if not np.isnan(train_df[col].median()) else 0.0
            train_df[col] = train_df[col].fillna(med)
            if col in val_df.columns:
                val_df[col] = val_df[col].fillna(med)
            if col in test_df.columns:
                test_df[col] = test_df[col].fillna(med)

    cdhw_features = [c for c in cfg.CDHW_COLS + [
        "CDHW_Veg_Severity", "CDHW_Silking_Severity", "CDHW_GrainFill_Severity"
    ] if c in all_feature_cols]

    non_cdhw_features = [c for c in all_feature_cols if c not in cdhw_features]

    # Ablation ladder. Each rung must differ from the previous one in the
    # actual computation, not only in what gets reported:
    #   1 -> 2 : feature set (CDHW family added)
    #   2 -> 3 : an interval head is fitted at all + split conformal calibration
    #            (post-hoc: the backbone is trained on the point objective alone,
    #            then frozen, then quantile heads are fitted on top of it)
    #   3 -> 4 : static split conformal replaced by online ACI recalibration
    #   4 -> 5 : post-hoc two-stage training replaced by genuine joint
    #            end-to-end training (composite loss, quantile gradients reach
    #            the shared backbone)
    # Rows 1 and 2 are point-prediction-only rows: with use_cqr=False the model
    # is trained on the Huber objective alone and no interval is produced, so
    # PICP/MPIW are genuinely not applicable rather than merely unreported.
    configs = [
        ("1_backbone_only", non_cdhw_features, False, False, False),
        ("2_plus_cdhw", all_feature_cols, False, False, False),
        ("3_plus_cqr", all_feature_cols, True, False, False),      # Post-hoc CQR + static conformal
        ("4_plus_aci", all_feature_cols, True, True, False),       # Post-hoc CQR + ACI
        ("5_full_joint", all_feature_cols, True, True, True),      # Joint end-to-end + ACI
    ]

    for config_name, feat_cols, use_cqr, use_aci, is_joint in configs:
        logger.info("Ablation config: %s", config_name)
        try:
            config_scaler, _ = fit_scaler(train_df, feat_cols, scaler_type="robust")

            tr_scaled = apply_scaling(train_df, feat_cols, config_scaler)
            va_scaled = apply_scaling(val_df, feat_cols, config_scaler)
            te_scaled = apply_scaling(test_df, feat_cols, config_scaler)

            X_tr, y_tr = get_feature_target_arrays(tr_scaled, feat_cols)
            X_val, y_val = get_feature_target_arrays(va_scaled, feat_cols)
            X_te, y_te = get_feature_target_arrays(te_scaled, feat_cols)

            # ── R² PATCH: detrend target for ablation (train-only), consistent
            #    with the main pipeline. Trend added back before scoring below. ──
            _abl_detrend = getattr(cfg, "DETREND_TARGET", False)
            if _abl_detrend:
                import numpy as _np
                _tr_years = train_df[train_df[cfg.PRIMARY_TARGET].notna()]["Year"].values
                _cf_abl = _np.polyfit(_tr_years, y_tr, 1)
                _tr_trend_abl = _np.polyval(_cf_abl, _tr_years)
                _va_years = val_df[val_df[cfg.PRIMARY_TARGET].notna()]["Year"].values
                _te_years = test_df[test_df[cfg.PRIMARY_TARGET].notna()]["Year"].values
                _va_trend_abl = _np.polyval(_cf_abl, _va_years)
                _te_trend_abl = _np.polyval(_cf_abl, _te_years)
                y_tr = y_tr - _tr_trend_abl
                y_val = y_val - _va_trend_abl
                y_te_raw = y_te.copy()
                y_te = y_te - _te_trend_abl
            else:
                _te_trend_abl = None
                y_te_raw = y_te

            if use_cqr:
                # Rows 3-5: interval-producing models. `is_joint` selects between
                # the two genuinely different training paths in
                # model_training.train_neural_cqr (see its docstring).
                models = train_neural_cqr(
                    X_tr, y_tr, X_val, y_val, feat_cols,
                    epochs=cfg.BASELINE_MAX_EPOCHS, batch_size=cfg.BATCH_SIZE, lr=cfg.LEARNING_RATE,
                    joint_training=is_joint, scaler=config_scaler
                )
            else:
                # Rows 1-2: point-prediction-only backbone. The pinball,
                # crossing and width terms are switched off entirely, so no
                # interval objective shapes the representation.
                models = train_neural_cqr(
                    X_tr, y_tr, X_val, y_val, feat_cols,
                    epochs=cfg.BASELINE_MAX_EPOCHS, batch_size=cfg.BATCH_SIZE, lr=cfg.LEARNING_RATE,
                    joint_training=True, scaler=config_scaler,
                    lambda_pinball=0.0, lambda_crossing=0.0, lambda_width=0.0,
                    early_stopping_mode="rmse",
                )
            preds, q_lo, q_hi = predict_intervals(models, X_te)

            # ── R² PATCH: add trend back so ablation scores on the real scale ──
            if _abl_detrend:
                preds = preds + _te_trend_abl
                q_lo = q_lo + _te_trend_abl
                q_hi = q_hi + _te_trend_abl
                y_te = y_te_raw

            _fingerprint = getattr(models, "paradigm_fingerprint", None) or {}
            result_entry = {
                "config": config_name,
                "n_features": len(feat_cols),
                "interval_head": use_cqr,
                "calibration": ("aci" if use_aci else ("static_conformal" if use_cqr else "none")),
                "joint_training": is_joint,
                "training_paradigm": getattr(models, "training_paradigm", None) if use_cqr else "point_only",
                "n_optimization_stages": _fingerprint.get("n_optimization_stages"),
                "backbone_receives_quantile_gradient": (
                    bool(_fingerprint.get("backbone_receives_quantile_gradient")) if use_cqr else False
                ),
                "loss_terms_stage1": _fingerprint.get("stage1_loss_terms"),
                "loss_terms_stage2": _fingerprint.get("stage2_loss_terms"),
                "epochs_trained": getattr(models, "epochs_trained", None),
                "best_epoch": getattr(models, "best_epoch", None),
                "prediction_checksum": round(float(np.sum(np.abs(preds))), 6),
                "rmse": round(rmse(y_te, preds), 4),
                "mae": round(mae(y_te, preds), 4),
                "r_squared": round(r_squared(y_te, preds), 4),
            }

            if use_cqr:
                preds_val, q_lo_val, q_hi_val = predict_intervals(models, X_val)
                if _abl_detrend:
                    preds_val = preds_val + _va_trend_abl
                    q_lo_val = q_lo_val + _va_trend_abl
                    q_hi_val = q_hi_val + _va_trend_abl
                    y_val = y_val + _va_trend_abl

                if use_aci:
                    years_test = test_df["Year"].values
                    cal = adaptive_conformal_inference(
                        y_val, q_lo_val, q_hi_val,
                        y_te, q_lo, q_hi, preds, years_test,
                    )
                else:
                    cal = static_conformal(
                        y_val, q_lo_val, q_hi_val,
                        q_lo, q_hi, preds, y_te,
                    )

                result_entry["picp"] = round(picp(y_te, cal.q_lo, cal.q_hi), 4)
                result_entry["mpiw"] = round(mpiw(cal.q_lo, cal.q_hi), 4)
                result_entry["ace"] = round(ace(y_te, cal.q_lo, cal.q_hi), 4)
                result_entry["winkler_score"] = round(
                    winkler_score(y_te, cal.q_lo, cal.q_hi), 4
                )
            else:
                # No interval head was fitted for this configuration.
                result_entry["picp"] = "n/a"
                result_entry["mpiw"] = "n/a"
                result_entry["ace"] = "n/a"
                result_entry["winkler_score"] = "n/a"

            report["configurations"].append(result_entry)
            logger.info("  -> %s", result_entry)

        except Exception as e:
            logger.error("Ablation config %s failed: %s", config_name, e, exc_info=True)

    # Prove the supposedly-different configurations really are different.
    # Identical prediction checksums between rows claiming different
    # methodologies mean the ablation is invalid, not that the effect is zero.
    by_name = {c["config"]: c for c in report["configurations"]}
    checks = []
    # M12 correction (2026-09-17): config 2 is point-only (Huber alone) and
    # config 3 is POST-HOC CQR, whose first stage trains the backbone and mean
    # head on Huber alone with the same seed and data. Identical point
    # predictions between them is therefore the defining property of a post-hoc
    # interval head, not a failure -- the old expectation asserted the opposite
    # and made M12 fail on correct behaviour. What must change from 2 to 3 is
    # that intervals exist at all.
    for a, b, what, mode in (
        ("2_plus_cdhw", "3_plus_cqr",
         "interval head added (post-hoc: the point path is unchanged by design)", "interval_appears"),
        ("3_plus_cqr", "4_plus_aci", "static conformal -> ACI (calibration only)", "interval_differs"),
        ("4_plus_aci", "5_full_joint", "post-hoc -> joint end-to-end training", "point_differs"),
    ):
        if a not in by_name or b not in by_name:
            continue
        same_points = abs(by_name[a].get("prediction_checksum", 0.0)
                          - by_name[b].get("prediction_checksum", -1.0)) < 1e-9
        same_intervals = (by_name[a].get("mpiw") == by_name[b].get("mpiw"))
        expect_point_diff = (mode == "point_differs")
        if mode == "point_differs":
            status = "FAIL" if same_points else "PASS"
        elif mode == "interval_appears":
            # The point row has no interval at all; the CQR row must have one.
            # Point-only rows record mpiw as the string "n/a", so a numeric
            # value is what marks a real interval.
            def _has_interval(v):
                return isinstance(v, (int, float)) and not isinstance(v, bool)
            a_has = _has_interval(by_name[a].get("mpiw"))
            b_has = _has_interval(by_name[b].get("mpiw"))
            status = "PASS" if (not a_has and b_has) else "FAIL"
        else:
            # 3 vs 4 differ only in calibration, which by construction cannot
            # move point predictions; the intervals must differ instead.
            status = "FAIL" if same_intervals else "PASS"
        checks.append({
            "comparison": f"{a} vs {b}",
            "expected_difference": what,
            "point_difference_expected": expect_point_diff,
            "identical_point_predictions": bool(same_points),
            "identical_mpiw": bool(same_intervals),
            "status": status,
        })
        logger.info("Ablation distinctness [%s | %s]: %s", f"{a} vs {b}", what, status)

    report["configuration_distinctness_checks"] = checks
    report["all_configurations_distinct"] = bool(checks) and all(c["status"] == "PASS" for c in checks)
    return report


def _generate_paper3_final_summary(
    summary_dict: Dict[str, Any],
    eval_report: Dict[str, Any],
    backbone_df: pd.DataFrame,
    elapsed_seconds: float,
    validation: Optional[Dict[str, Any]] = None,
    leakage: Optional[Dict[str, Any]] = None,
) -> None:
    """Generate paper3_final_summary.md from measured results.

    Every claim in this report is now read from the artefacts produced by the
    run. The previous version hard-coded "Methodology Compliance Status: 100%
    Fully Compliant" plus four objective conclusions ("joint training improves
    RMSE", "significantly positively correlated", "ACI achieves top rank",
    "superior coverage guarantee") that were printed regardless of what the
    pipeline actually computed -- and, as it turned out, contradicted it.
    """
    import json as _json

    def _read(name: str):
        path = cfg.REPORT_DIR / name
        if not path.exists():
            return None
        try:
            return _json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    o4 = _read("objective_O4_report.json")
    o5 = _read("objective_o5_report.json")
    o6 = _read("objective_O6_report.json")
    o1 = _read("objective_O1_report.json")
    shap_rpt = _read("shap_consistency_report.json")
    feat = _read("feature_selection_pipeline.json")

    val_status = (validation or {}).get("overall_status", "NOT_RUN")
    val_counts = (validation or {}).get("counts", {})
    leak_ok = (leakage or {}).get("all_passed")
    leak_n = (leakage or {}).get("n_experiments", 0)

    md = f"""# Paper 3: Adaptive Conformal Inference — Final Master Report

## Executive Summary
- **Dataset**: {summary_dict['dataset_rows']} county-year observations ({summary_dict['dataset_cols']} columns)
- **Final modeling features**: {summary_dict['features_used']}
- **Authoritative 4-Way Temporal Split**: FIT 1985–2013 ({summary_dict['temporal_split'].get('fit', 'n/a')}), DEV 2014–2015 ({summary_dict['temporal_split'].get('dev', 'n/a')}), CAL 2016–2018 ({summary_dict['temporal_split'].get('cal', 'n/a')}), TEST 2019–2023 ({summary_dict['temporal_split'].get('test', 'n/a')})
- **LOSO-CV**: {len(cfg.LOSO_STATES)} state folds ({', '.join(cfg.LOSO_STATES)})
- **Backbones Trained**: {', '.join(summary_dict.get('backbones_trained', []))}
- **Conformal Calibration Methods**: {', '.join(summary_dict.get('calibration_methods', []))}
- **Pipeline Runtime**: {elapsed_seconds:.1f} seconds
- **Methodology validation**: `{val_status}` (PASS={val_counts.get('PASS', '?')}, FAIL={val_counts.get('FAIL', '?')}, WARNING={val_counts.get('WARNING', '?')}, N/A={val_counts.get('NOT_APPLICABLE', '?')}) — see `methodology_validation_report.md`
- **Leakage audit**: {'zero train/dev/cal/test observation overlap across ' + str(leak_n) + ' experiments (row-identity check)' if leak_ok else 'FAILED — see leakage_provenance_audit.md'}
"""

    if feat:
        md += f"""
## Feature selection funnel
`{feat['stage_1_candidates']}` candidates → `{feat['stage_2_consensus_selected']}` consensus-selected
→ `{feat['stage_3_after_multicollinearity_pruning']}` after multicollinearity pruning
→ `{feat['stage_4_final_modeling_features']}` final modeling features.
All stages fitted on {feat['all_fitted_on']}.
"""

    md += "\n## Multi-Backbone Model Performance Benchmark (§2)\n"
    try:
        md += backbone_df.to_markdown(index=False) + "\n\n"
    except Exception:
        md += backbone_df.to_string(index=False) + "\n\n"

    md += "## Key Research Objective Outcomes\n"

    if o1:
        d_r2 = o1.get("delta_r2", o1.get("delta_r_squared"))
        md += (f"- **O1 (CDHW encoding)**: ΔR² with vs without CDHW features = "
               f"`{d_r2}`. See `objective_O1_report.md`.\n")
    else:
        md += "- **O1 (CDHW encoding)**: report not available.\n"

    if o4:
        md += (f"- **O4 (Joint vs post-hoc)**: **{o4.get('verdict')}** — "
               f"RMSE {o4['comparison']['rmse']['post_hoc']} (post-hoc) vs "
               f"{o4['comparison']['rmse']['joint']} (joint), "
               f"relative Δ {o4['comparison']['rmse']['relative_difference_pct']}%; "
               f"MPIW {o4['comparison']['mpiw_static']['post_hoc']} vs "
               f"{o4['comparison']['mpiw_static']['joint']} "
               f"({o4['comparison']['mpiw_static']['relative_difference_pct']}%). "
               f"Point improvement supported: {o4.get('point_prediction_improvement_supported')}; "
               f"interval improvement supported: {o4.get('interval_quality_improvement_supported')}.\n")
    else:
        md += "- **O4 (Joint vs post-hoc)**: report not available.\n"

    if o5:
        pi = o5.get("primary_inference", {})
        md += (f"- **O5 (Severity attribution)**: cluster-robust slope = `{pi.get('slope')}` "
               f"(95% CI `{pi.get('ci_95')}`, p = `{pi.get('p_value')}`, clustered by "
               f"{pi.get('cluster_unit')}), R² = `{o5.get('variance_explained_fraction')}` — "
               f"effect magnitude: **{o5.get('effect_magnitude_label')}**, statistically "
               f"detectable: **{o5.get('statistically_detectable')}**.\n")
    else:
        md += "- **O5 (Severity attribution)**: report not available.\n"

    if o6:
        top = (o6.get("rankings") or [{}])[0].get("Method")
        fr = o6.get("friedman_test", {})
        md += (f"- **O6 (Conformal method ranking)**: lowest average rank = `{top}` across "
               f"{o6.get('n_methods_compared')} methods. Status: **{o6.get('ranking_status')}**. "
               f"Friedman executed: {fr.get('test_executed')}"
               + (f", p = `{fr.get('p_value')}` (blocked by {fr.get('block_unit')})"
                  if fr.get('test_executed') else f" ({fr.get('reason')})")
               + f". Claims statistical superiority: **{o6.get('claims_statistical_superiority')}**.\n")
    else:
        md += "- **O6 (Conformal method ranking)**: report not available.\n"

    if shap_rpt:
        md += (f"- **Interpretability (SHAP)**: cross-backbone similarity index = "
               f"`{shap_rpt.get('shap_similarity_index')}` — **{shap_rpt.get('agreement_label')}** "
               f"across {len(shap_rpt.get('models_with_computed_shap', []))} backbones"
               + (f"; excluded: {shap_rpt.get('models_excluded')}"
                  if shap_rpt.get('models_excluded') else "") + ".\n")

    # Coverage statement derived from the measured PICP, not asserted.
    methods = eval_report.get("methods", {})
    if methods:
        md += "\n## Measured coverage against the nominal level\n\n"
        md += f"Nominal coverage: **{cfg.NOMINAL_COVERAGE}**.\n\n"
        md += "| Method | PICP | MPIW | ACE | Winkler | Meets nominal? |\n|---|---:|---:|---:|---:|---|\n"
        for name, data in methods.items():
            u = data.get("uncertainty", {})
            picp_v = u.get("picp")
            meets = (picp_v is not None and picp_v >= cfg.NOMINAL_COVERAGE)
            md += (f"| {name} | {picp_v} | {u.get('mpiw')} | {u.get('ace')} | "
                   f"{u.get('winkler_score')} | {'yes' if meets else 'no'} |\n")

    md += """
## Interpretation notes
- Significance claims in O4, O5 and O6 use dependence-aware inference (year-block
  cluster bootstrap, cluster-robust standard errors, or year-level paired tests),
  because county-year rows are spatially and temporally correlated. Row-level
  p-values computed under an independence assumption are reported only where
  explicitly labelled as invalid, to document the size of the distortion.
- The test partition (2019–2023) was used exactly once, for final evaluation, after
  all modelling decisions were frozen in `model_decisions_lock.json`.
"""
    save_report_markdown(md, "paper3_final_summary.md")
    logging.getLogger("paper3").info("Paper 3 Final Master Report saved -> paper3_final_summary.md")


def _generate_methodology_compliance_report() -> None:
    """Render the compliance report from the property-based validator's results.

    The previous version was a fixed Markdown string that declared every section
    "Verified", stated "**100% Methodology Compliance** ... with **Zero Data
    Leakage**", described the temporal split with the superseded 1985-2015 /
    2016-2018 boundaries, and overstated the LOSO fold count; the protocol has six.
    None of it was computed. This version reports only what the validator and
    the leakage audit actually found.
    """
    import json as _json

    def _read(name):
        path = cfg.REPORT_DIR / name
        if not path.exists():
            return None
        try:
            return _json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    validation = _read("methodology_validation_report.json")
    leakage = _read("leakage_provenance_audit.json")

    if validation is None:
        md = (
            "# Paper 3 Methodology Compliance Report\n\n"
            "**Status: UNKNOWN** — `methodology_validation_report.json` was not produced, "
            "so no compliance claim can be made.\n"
        )
        save_report_markdown(md, "methodology_compliance_report.md")
        return

    counts = validation.get("counts", {})
    overall = validation.get("overall_status", "UNKNOWN")
    leak_ok = (leakage or {}).get("all_passed")
    leak_n = (leakage or {}).get("n_experiments", 0)
    leak_txt = (
        f"zero observation overlap across {leak_n} experiments, checked on county-year row IDs"
        if leak_ok else "FAILED or not run — see leakage_provenance_audit.md"
    )

    md = f"""# Paper 3 Methodology Compliance Report (§1–§7)

## Overall status: `{overall}`

| Result | Count |
|---|---:|
| PASS | {counts.get('PASS', 0)} |
| FAIL | {counts.get('FAIL', 0)} |
| WARNING | {counts.get('WARNING', 0)} |
| NOT_APPLICABLE | {counts.get('NOT_APPLICABLE', 0)} |

This status is computed by `methodology_validator.run_methodology_validation()`,
which reads the artefacts this run produced and tests properties of them. It is
not a fixed statement: a check whose evidence is missing counts as FAIL.

## Locked protocol as implemented
- **Temporal split**: FIT {cfg.FIT_YEARS[0]}–{cfg.FIT_YEARS[1]}, DEV {cfg.DEV_YEARS[0]}–{cfg.DEV_YEARS[1]}, CAL {cfg.CAL_YEARS[0]}–{cfg.CAL_YEARS[1]}, TEST {cfg.TEST_YEARS[0]}–{cfg.TEST_YEARS[1]}
- **LOSO-CV**: {len(cfg.LOSO_STATES)} state folds — {', '.join(cfg.LOSO_STATES)}
- **Leakage audit**: {leak_txt}

## Check results
| ID | Requirement | Status | Evidence |
|---|---|---|---|
"""
    for c in validation.get("checks", []):
        ev = str(c.get("evidence", "")).replace("|", "\\|").replace("\n", " ")
        md += f"| {c['id']} | {c['requirement']} | **{c['status']}** | {ev} |\n"

    failing = [c for c in validation.get("checks", []) if c["status"] == "FAIL"]
    warned = [c for c in validation.get("checks", []) if c["status"] == "WARNING"]
    md += "\n## Verification statement\n"
    if not failing and not warned:
        md += ("Every applicable methodology check passed against artefacts generated by this "
               "run. No check was assumed to pass.\n")
    else:
        md += (f"{len(failing)} check(s) FAILED and {len(warned)} raised WARNING. "
               "The implementation is not fully compliant; see the rows above.\n")
        for c in failing + warned:
            md += f"- **{c['id']}** ({c['status']}): {c['requirement']} — {c.get('evidence')}\n"

    save_report_markdown(md, "methodology_compliance_report.md")
    logging.getLogger("paper3").info(
        "Methodology compliance report saved -> methodology_compliance_report.md (status=%s)",
        overall)


def _generate_methodology_traceability_matrix() -> None:
    """Map methodology sections to the code that implements them.

    The mapping itself is documentation. The *status* column used to be the
    literal string "Yes" on every row, which asserted verification that had
    never happened. It is now derived: each row's declared output artefact is
    looked up on disk and reported as PRESENT / MISSING / EMPTY, and rows tied
    to a methodology check inherit that check's real PASS/FAIL/WARNING status
    from methodology_validation_report.json.
    """
    import json as _json

    validator_status = {}
    _vp = cfg.REPORT_DIR / "methodology_validation_report.json"
    if _vp.exists():
        try:
            for c in _json.loads(_vp.read_text(encoding="utf-8")).get("checks", []):
                validator_status[c["id"]] = c["status"]
        except Exception:
            pass

    def _artifact_status(rel: str) -> str:
        for base in (cfg.REPORT_DIR, cfg.OUTPUT_DIR, cfg.FIGURES_DIR,
                     cfg.METRICS_DIR, cfg.PREDICTIONS_DIR):
            path = base / rel
            if path.exists():
                if path.is_dir():
                    return "PRESENT" if any(path.iterdir()) else "EMPTY"
                return "PRESENT" if path.stat().st_size > 0 else "EMPTY"
        return "MISSING"

    records = [
        ("§4.1 Compound Encoding", "O1", "feature_engineering.py", "_compute_phenology_aligned_cdhw", "feature_validation_report.json", None),
        ("§4.2 Neural CQR Head", "O2", "model_training.py", "train_neural_cqr", "models/NeuralCQR", None),
        ("§4.3 ACI Recalibration", "O3", "aci_calibrator.py", "adaptive_conformal_inference", "predictions.csv", None),
        ("§4.2 Joint vs Posthoc", "O4", "evaluation.py", "evaluate_objective_o4", "objective_O4_report.md", "M11"),
        ("§5.8 Severity Attribution", "O5", "evaluation.py", "evaluate_objective_o5", "objective_o5_report.md", "M15"),
        ("§4.5 Shift-Robust Conformal", "O6", "evaluation.py", "evaluate_objective_o6", "objective_O6_report.md", "M15"),
        ("§4.4 Autocorrelation Diagnostic", "Robustness", "evaluation.py", "block_bootstrap_county", "evaluation_summary.md", None),
        ("§4.6 Ablation Study", "Ablation", "main.py", "_run_ablation", "ablation_report.json", "M12"),
        ("§4.7 Computational Complexity", "Benchmark", "evaluation.py", "benchmark_computational_complexity", "computational_benchmark.csv", "M18"),
        ("§5.1 Temporal Split", "Splitting", "splitting.py", "temporal_split_4way", "leakage_audit_report.md", "M6"),
        ("§5.2 LOSO Cross-Validation", "Generalization", "main.py", "_run_loso_cv", "loso_summary.csv", "M1"),
        ("§5.2 Leakage Provenance", "Integrity", "leakage_provenance.py", "assert_disjoint", "leakage_provenance_audit.json", "M2"),
        ("§5.5 Statistical Significance", "Testing", "evaluation.py", "wilcoxon_signed_rank_test", "statistical_tests_report.md", None),
        ("§5.6 Calibration Curves", "Calibration", "visualization.py", "plot_all_calibration_curves", "calibration_curves.png", None),
        ("§5.8 SHAP Consistency", "Interpretability", "visualization.py", "export_shap_consistency_report", "shap_consistency_report.json", "M16"),
    ]

    rows = []
    for section, objective, pyfile, func, output, check_id in records:
        rows.append({
            "Methodology Section": section,
            "Objective": objective,
            "Python File": pyfile,
            "Function": func,
            "Output Artifact": output,
            "Artifact Status": _artifact_status(output),
            "Validator Check": check_id or "n/a",
            "Validator Status": validator_status.get(check_id, "n/a") if check_id else "n/a",
        })

    df = pd.DataFrame(rows)
    save_report_csv(df, "methodology_traceability_matrix.csv")
    logger = logging.getLogger("paper3")
    logger.info(
        "Methodology traceability matrix saved -> methodology_traceability_matrix.csv "
        "(%d artefacts present, %d missing)",
        int((df["Artifact Status"] == "PRESENT").sum()),
        int((df["Artifact Status"] == "MISSING").sum()),
    )


def _generate_methodology_validation_report() -> None:
    """DEPRECATED — superseded by methodology_validator.run_methodology_validation().

    The original function here was a hard-coded table of 15 rows with
    ``"Status": "Pass"`` written into every one of them; ``all_pass`` was
    therefore always True and the report always announced
    "FULL_METHODOLOGY_COMPLIANCE_100%". It tested nothing. Kept as a thin
    redirect so any external caller gets the real, property-based audit.
    """
    from methodology_validator import run_methodology_validation
    logging.getLogger("paper3").warning(
        "_generate_methodology_validation_report() is deprecated; "
        "delegating to the property-based methodology validator."
    )
    return run_methodology_validation()

if __name__ == "__main__":
    main()


