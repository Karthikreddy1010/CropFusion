# Final Model Decisions Lock File (§18)

**Status**: `LOCKED — NO MODELING DECISIONS MAY CHANGE BEYOND THIS POINT`
**Timestamp**: `2026-09-22T22:30:56Z`

## 1. Frozen Partitions
- **FIT (1985–2013)**: Preprocessing, detrending, baseline, feature selection, candidate fitting
- **DEV (2014–2015)**: Early stopping, hyperparameter confirmation, ensemble weight search
- **CAL (2016–2018)**: Conformal calibration ONLY (uncontaminated residual scores)
- **TEST (2019–2023)**: Final locked evaluation strictly once

## 2. Frozen Feature Specification
- **Selected Features (94)**: `prism_tmax_mean_gs, prism_tmax_max_gs, prism_tmin_mean_gs, prism_tmin_min_gs, Precip_growseason_mm, prism_kdd30_gs, Tmax_Days_Above_35, prism_heat_exceedance_total_gs, nldas_rs_mean_gs, nldas_humidity_mean_gs...`
- **Linear Detrending**: Slope = `0.1094` t/ha/yr (Fitted on FIT only)
- **County Baseline**: Historical county mean fitted on FIT only

## 3. Frozen Backbone & Ensemble Specifications
- **NeuralCQR**: Architecture and hyperparameters were fixed prior to final evaluation; the DEV partition (2014–2015) was used strictly for early stopping and ensemble-weight selection.
  - Frozen Hyperparameters: Hidden Dims = `(64, 32)`, LR = `3e-05`, WD = `0.001`, Batch = `128`, Dropout = `0.2`, Pinball λ = `1.0`, Huber λ = `1.0`, Crossing λ = `10.0`, Width λ = `0.005`
  - Frozen Stopping Epoch: `187` (selected via DEV RMSE on DEV_2014_2015; early stopping on refit: false)
- **LightGBM**: `{'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 7, 'num_leaves': 63, 'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}`
- **CatBoost**: `iterations=500, learning_rate=0.05`
- **XGBoost**: `n_estimators=300, learning_rate=0.05`
- **Ensemble Weight**: `0.85` NeuralCQR + `0.15` LightGBM (Selected on DEV RMSE = `1.0250`)

## 4. Conformal Calibration Protocol
- **Raw Quantile Blending**: Ensemble quantiles blended BEFORE calibration
- **Calibration Set**: 2016–2018 nonconformity scores
- **Evaluated Conformal Methods**: Static Conformal, Standard ACI, SA-ACI
