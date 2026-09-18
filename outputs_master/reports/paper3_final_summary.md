# Paper 3: Adaptive Conformal Inference — Final Master Report

## Executive Summary
- **Dataset**: 22737 county-year observations (121 columns)
- **Final modeling features**: 94
- **Authoritative 4-Way Temporal Split**: FIT 1985–2013 (15751), DEV 2014–2015 (978), CAL 2016–2018 (1433), TEST 2019–2023 (2345)
- **LOSO-CV**: 6 state folds (Illinois, Indiana, Iowa, Minnesota, Missouri, Ohio)
- **Backbones Trained**: NeuralCQR, LightGBM, CatBoost, XGBoost
- **Conformal Calibration Methods**: static_conformal, standard_aci, sa_aci, phenology_stratified_cqr, weighted_conformal, locally_adaptive
- **Pipeline Runtime**: 3338.1 seconds
- **Methodology validation**: `COMPLIANT` (PASS=18, FAIL=0, WARNING=0, N/A=0) — see `methodology_validation_report.md`
- **Leakage audit**: zero train/dev/cal/test observation overlap across 7 experiments (row-identity check)

## Feature selection funnel
`117` candidates → `100` consensus-selected
→ `93` after multicollinearity pruning
→ `94` final modeling features.
All stages fitted on FIT partition (1985-2013) only.

## Multi-Backbone Model Performance Benchmark (§2)
| Model                       |   RMSE |    MAE |     R2 |   Coverage |   MPIW |   Winkler_Score |
|:----------------------------|-------:|-------:|-------:|-----------:|-------:|----------------:|
| NeuralCQR                   | 0.9529 | 0.7328 | 0.684  |     0.7552 | 2.0973 |          4.7188 |
| LightGBM                    | 0.8788 | 0.6732 | 0.7313 |     0.661  | 1.77   |          5.2387 |
| CatBoost                    | 0.8908 | 0.6918 | 0.7238 |     0.6725 | 1.9123 |          5.1755 |
| XGBoost                     | 0.9029 | 0.6924 | 0.7163 |     0.7505 | 2.5516 |          4.9962 |
| NeuralCQR_LightGBM_Ensemble | 0.9237 | 0.7091 | 0.7031 |     0.9241 | 3.3203 |          4.0396 |

## Key Research Objective Outcomes
- **O1 (CDHW encoding)**: ΔR² with vs without CDHW features = `0.0116`. See `objective_O1_report.md`.
- **O4 (Joint vs post-hoc)**: **Partially supported — joint training improves interval quality only** — RMSE 0.9629 (post-hoc) vs 0.963 (joint), relative Δ 0.01%; MPIW 4.7056 vs 3.7938 (-19.38%). Point improvement supported: False; interval improvement supported: True.
- **O5 (Severity attribution)**: cluster-robust slope = `0.024459` (95% CI `[-0.001738, 0.050655]`, p = `0.060539`, clustered by year), R² = `0.075887` — effect magnitude: **weak**, statistically detectable: **False**.
- **O6 (Conformal method ranking)**: lowest average rank = `sa_aci` across 6 methods. Status: **descriptive ranking only**. Friedman executed: True, p = `0.199346` (blocked by year). Claims statistical superiority: **False**.
- **Interpretability (SHAP)**: cross-backbone similarity index = `0.6046` — **strong agreement** across 4 backbones.

## Measured coverage against the nominal level

Nominal coverage: **0.9**.

| Method | PICP | MPIW | ACE | Winkler | Meets nominal? |
|---|---:|---:|---:|---:|---|
| static_conformal | 0.9488 | 3.6493 | 0.0488 | 4.1473 | yes |
| standard_aci | 0.9296 | 3.4151 | 0.0296 | 4.0788 | yes |
| sa_aci | 0.9241 | 3.3203 | 0.0241 | 4.0396 | yes |
| phenology_stratified_cqr | 0.9493 | 3.6506 | 0.0493 | 4.1475 | yes |
| weighted_conformal | 0.9471 | 3.6367 | 0.0471 | 4.1413 | yes |
| locally_adaptive | 0.9488 | 3.6987 | 0.0488 | 4.167 | yes |

## Interpretation notes
- Significance claims in O4, O5 and O6 use dependence-aware inference (year-block
  cluster bootstrap, cluster-robust standard errors, or year-level paired tests),
  because county-year rows are spatially and temporally correlated. Row-level
  p-values computed under an independence assumption are reported only where
  explicitly labelled as invalid, to document the size of the distortion.
- The test partition (2019–2023) was used exactly once, for final evaluation, after
  all modelling decisions were frozen in `model_decisions_lock.json`.
