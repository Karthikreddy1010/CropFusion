"""
smoke_test_master_loader.py - end-to-end data-path check for PAPER3_DATA_SOURCE=master.

Runs the locked pipeline's own functions (no re-implementation) up to, but not including, model
training: load + compliance + EDA, deterministic feature engineering, profiling, the 4-way temporal
split with split-aware lags/rolling features, train-fitted preprocessing, FIT-only feature selection,
scaling and array extraction, and one complete LOSO fold. Every check is asserted; the summary is
written to outputs_master/audits/master_loader_smoke_test.json.

Usage (from the Paper3 folder; the master dataset is selected automatically when its files are present):
    python code/smoke_test_master_loader.py
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("PAPER3_DATA_SOURCE", "master")
if os.environ["PAPER3_DATA_SOURCE"].lower() != "master":
    sys.exit("This smoke test checks the master data source; unset PAPER3_DATA_SOURCE or set it to 'master'.")

import numpy as np
import pandas as pd

import config as cfg
from utils import setup_logging, set_global_seed
from data_loader import load_dataset, validate_methodology_compliance
from eda import run_eda
from missing_values import handle_missing_values
from outliers import analyze_outliers
from multicollinearity import analyze_multicollinearity
from feature_engineering import engineer_features, build_split_aware_lags, build_split_aware_rolling_features
from feature_selection import select_features
from scaling import fit_scaler, apply_scaling
from splitting import temporal_split_4way, loso_cv_folds, get_feature_target_arrays
from preprocessor import TrainFittedPreprocessor

NON_FEATURES = {"GEOID", "County_Name", "State", "STATEFP", "COUNTYFP", "Year", "Split", "ENSO_Phase",
                "ENSO_Anomalous_Year", "Corn_Yield_tha", "Corn_Yield_buacre", "Has_Corn_Yield"}


def main():
    setup_logging()
    set_global_seed()
    t0 = time.time()
    s = {"data_source": cfg.DATA_SOURCE, "output_dir": str(cfg.OUTPUT_DIR), "stages": {}}
    assert cfg.DATA_SOURCE == "master" and cfg.OUTPUT_DIR.name == "outputs_master"

    # Phase 1
    df = load_dataset()
    compliance = validate_methodology_compliance(df)
    run_eda(df)
    assert len(df) == 22737 and df.GEOID.nunique() == 583 and df.duplicated(["GEOID", "Year"]).sum() == 0
    assert not any(c.startswith("ERA5") for c in df.columns)
    s["stages"]["load"] = dict(rows=len(df), cols=df.shape[1], observed_yields=int(df.Corn_Yield_tha.notna().sum()),
                               compliance=compliance, seconds=round(time.time() - t0, 1))

    # Phase 2
    df, fe_report = engineer_features(df)
    for col in ["Phenological_Window", "CDHW_Veg_Severity", "CDHW_Silking_Severity", "CDHW_GrainFill_Severity",
                "Year_Type", "Inter_SPEI_Tmax", "Inter_Heat_Precip", "Inter_CDHW_GDD"]:
        assert col in df.columns, f"engineered feature missing: {col}"
    assert not any(c.startswith("ENSO_") and c not in ("ENSO_Phase", "ENSO_Anomalous_Year") for c in df.columns), \
        "ENSO dummy features created although ENSO_AS_FEATURES=False"
    assert np.allclose(df["Inter_SPEI_Tmax"], np.round(df["SPEI_30_min"] * df["prism_tmax_max_gs"], 4), equal_nan=True)
    handle_missing_values(df.copy())
    analyze_outliers(df.copy())
    analyze_multicollinearity(df.copy())
    s["stages"]["feature_engineering"] = dict(
        added=fe_report["features_added"], phenological_window=df.Phenological_Window.value_counts().to_dict(),
        year_type=df.Year_Type.value_counts().to_dict(), seconds=round(time.time() - t0, 1))

    # Phase 3: temporal
    fit_df, dev_df, cal_df, test_df = temporal_split_4way(df, target=cfg.PRIMARY_TARGET)
    fit_df, dev_df, cal_df, test_df = build_split_aware_lags(fit_df, dev_df, test_df, target_col=cfg.PRIMARY_TARGET,
                                                             split_type="temporal", cal_df=cal_df)
    fit_df, dev_df, cal_df, test_df = build_split_aware_rolling_features(fit_df, dev_df, test_df, split_type="temporal", cal_df=cal_df)
    for c in ["Yield_lag1", "Rolling2yr_Precip_growseason_mm_mean", "Rolling3yr_GDD_Accumulated_mean"]:
        assert c in fit_df.columns, f"missing split-aware column {c}"
    prep = TrainFittedPreprocessor(target_col=cfg.PRIMARY_TARGET)
    fit_df = prep.fit_transform(fit_df, split_name="fit")
    dev_df, cal_df, test_df = (prep.transform(d, split_name=n) for d, n in ((dev_df, "dev"), (cal_df, "cal"), (test_df, "test")))
    feats, _ = select_features(fit_df, target=cfg.PRIMARY_TARGET)
    leaked = sorted(set(feats) & NON_FEATURES)
    assert not leaked, f"non-feature columns selected: {leaked}"
    scaler, _ = fit_scaler(fit_df, feats, scaler_type="robust")
    arrays = {}
    for name, part in (("fit", fit_df), ("dev", dev_df), ("cal", cal_df), ("test", test_df)):
        X, y = get_feature_target_arrays(apply_scaling(part, feats, scaler, split_name=name), feats, split_name=name)
        assert np.isfinite(X).all() and np.isfinite(y).all(), f"non-finite values in {name} arrays"
        arrays[name] = [int(X.shape[0]), int(X.shape[1])]
    s["stages"]["temporal"] = dict(
        years={n: [int(d.Year.min()), int(d.Year.max())] for n, d in (("fit", fit_df), ("dev", dev_df), ("cal", cal_df), ("test", test_df))},
        arrays=arrays, n_selected_features=len(feats), selected_features=feats,
        role_aliases_selected=sorted(set(feats) & set(cfg.MASTER_ROLE_ALIASES.values())),
        seconds=round(time.time() - t0, 1))

    # LOSO: first fold end-to-end data path
    state, fit_f, dev_f, cal_f, test_f = next(iter(loso_cv_folds(df, return_cal=True)))
    for part in (fit_f, dev_f, cal_f):
        assert state not in set(part.State)
    assert (test_f.State == state).all()
    fit_f, dev_f, cal_f, test_f = build_split_aware_lags(fit_f, dev_f, test_f, target_col=cfg.PRIMARY_TARGET, split_type="loso", cal_df=cal_f)
    fit_f, dev_f, cal_f, test_f = build_split_aware_rolling_features(fit_f, dev_f, test_f, split_type="loso", cal_df=cal_f)
    prep_f = TrainFittedPreprocessor(target_col=cfg.PRIMARY_TARGET)
    fit_f = prep_f.fit_transform(fit_f, split_name=f"loso_{state}_fit")
    dev_f, cal_f, test_f = (prep_f.transform(d, split_name=f"loso_{state}_{n}") for d, n in ((dev_f, "dev"), (cal_f, "cal"), (test_f, "test")))
    fold_feats, _ = select_features(fit_f, target=cfg.PRIMARY_TARGET, report_suffix=f"_loso_{state.lower()}_smoke")
    fold_feats = [f for f in fold_feats if f not in {"county_baseline", "Fold", "HeldOutState", "obs_id", "Split"}]
    assert not set(fold_feats) & NON_FEATURES
    sc_f, _ = fit_scaler(fit_f, fold_feats, scaler_type="robust")
    fold_arrays = {}
    for name, part in (("fit", fit_f), ("dev", dev_f), ("cal", cal_f), ("test", test_f)):
        X, y = get_feature_target_arrays(apply_scaling(part, fold_feats, sc_f, split_name=name), fold_feats, split_name=name)
        assert np.isfinite(X).all() and np.isfinite(y).all()
        fold_arrays[name] = [int(X.shape[0]), int(X.shape[1])]
    s["stages"]["loso_first_fold"] = dict(held_out_state=state, arrays=fold_arrays, n_features=len(fold_feats),
                                          seconds=round(time.time() - t0, 1))
    s["result"] = "PASS"
    out = cfg.AUDITS_DIR / "master_loader_smoke_test.json"
    out.write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    print(f"SMOKE TEST PASS in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
