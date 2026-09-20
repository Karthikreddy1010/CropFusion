"""
config.py - Central configuration for Paper 3 ACI pipeline.

All paths, feature lists, hyperparameters, temporal boundaries,
and random seeds are defined here for reproducibility.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent
DATA_FILE = PROJECT_ROOT / "Paper3_MegaDataset_SPEI_FINAL.csv"
if not DATA_FILE.exists():
    DATA_FILE = PROJECT_ROOT / "Paper3_Processed.csv"

# ─────────────────────────────────────────────────────────────────────
# Data source selection.
#
# master: the PRISM-NLDAS-NOAA-USDA master dataset (the project's final data),
#   loaded through master_data_loader.py with the feature groups redefined
#   role-for-role at the end of this file. Outputs go to outputs_master/.
# legacy: the original ERA5-era MegaDataset CSV, exactly as before.
#
# PAPER3_DATA_SOURCE=master|legacy forces a source. When it is unset (or "auto"),
# the master dataset is used whenever its files are found, and the legacy CSV
# only when the master files are absent but the CSV is present.
#
# Master files are looked up in master_dataset/full|model_ready/, then directly
# in PROJECT_ROOT or master_dataset/ (flat Colab uploads), unless given
# explicitly via PAPER3_MASTER_FULL / PAPER3_MASTER_MODEL_READY.
# Only the data interface changes: model architecture, losses, thresholds,
# splits and the LOSO protocol are untouched.
# ─────────────────────────────────────────────────────────────────────
MASTER_DATASET_DIR = PROJECT_ROOT / "master_dataset"
MASTER_FULL_NAME = "Paper3_MasterDataset_PRISM_NLDAS_NOAA_USDA_FULL_1985_2023.parquet"
MASTER_MODEL_READY_NAME = "Paper3_MasterDataset_PRISM_NLDAS_CDHWH_CORN_MODEL_1985_2023.parquet"


def _locate_master_file(name: str, subdir: str, env_var: str) -> Path:
    if os.environ.get(env_var):
        return Path(os.environ[env_var])
    candidates = [MASTER_DATASET_DIR / subdir / name, PROJECT_ROOT / name, MASTER_DATASET_DIR / name]
    return next((c for c in candidates if c.exists()), candidates[0])


MASTER_FULL_FILE = _locate_master_file(MASTER_FULL_NAME, "full", "PAPER3_MASTER_FULL")
MASTER_MODEL_READY_FILE = _locate_master_file(MASTER_MODEL_READY_NAME, "model_ready", "PAPER3_MASTER_MODEL_READY")
_master_available = MASTER_FULL_FILE.exists() and MASTER_MODEL_READY_FILE.exists()
_legacy_available = (PROJECT_ROOT / "Paper3_MegaDataset_SPEI_FINAL.csv").exists()

_requested_source = (os.environ.get("PAPER3_DATA_SOURCE") or "auto").strip().lower()
if _requested_source not in ("auto", "legacy", "master"):
    raise ValueError(f"PAPER3_DATA_SOURCE must be 'master', 'legacy' or 'auto', got '{_requested_source}'")
if _requested_source == "auto":
    _soy_run = (os.environ.get("PAPER3_CROP_TARGET") or "corn").strip().lower() == "soy"   # master has no soybean target
    DATA_SOURCE = "legacy" if (_soy_run or (not _master_available and _legacy_available)) else "master"
    DATA_SOURCE_SELECTED_BY = "auto (master files %s, legacy CSV %s)" % (
        "found" if _master_available else "not found", "found" if _legacy_available else "not found")
else:
    DATA_SOURCE = _requested_source
    DATA_SOURCE_SELECTED_BY = "PAPER3_DATA_SOURCE"
if DATA_SOURCE == "master":
    DATA_FILE = MASTER_FULL_FILE

# ─────────────────────────────────────────────────────────────────────
# Crop target selection (parallel single-crop runs, not multi-task).
#
# Set PAPER3_CROP_TARGET=soy (env var) to run the identical pipeline
# for soybean instead of corn. Each crop writes to its own outputs_<crop>/
# directory so a soybean run can never overwrite corn results (or vice
# versa) -- see run_both_crops.py to run both crops back-to-back and
# CROP_COMPARISON_NOTES.md for why this is a parallel run, not a shared
# multi-task backbone.
#
# Nothing else in the pipeline is crop-specific: TARGET_COLS below
# already excludes BOTH crops' yield columns from the feature set
# regardless of which is primary, outliers.py already has bounds
# defined for both, and no crop-specific physics constants (e.g. the
# FAO Ky coefficients mentioned in the methodology doc) are hardcoded
# anywhere in this pipeline -- that PICA-level physics integration
# belongs to Paper 1's architecture, not this one. So swapping targets
# here is a genuine like-for-like comparison, not a partial port.
# ─────────────────────────────────────────────────────────────────────
_CROP_TARGET = os.environ.get("PAPER3_CROP_TARGET", "corn").strip().lower()
if _CROP_TARGET not in ("corn", "soy"):
    raise ValueError(f"PAPER3_CROP_TARGET must be 'corn' or 'soy', got '{_CROP_TARGET}'")

OUTPUT_DIR = PROJECT_ROOT / "outputs" if _CROP_TARGET == "corn" else PROJECT_ROOT / f"outputs_{_CROP_TARGET}"
if DATA_SOURCE == "master":
    if _CROP_TARGET != "corn":
        raise ValueError("PAPER3_DATA_SOURCE=master supports corn only (the master dataset has no soybean target).")
    OUTPUT_DIR = PROJECT_ROOT / "outputs_master"
FIGURES_DIR = OUTPUT_DIR / "figures"
PLOT_DIR = FIGURES_DIR  # Alias for backward compatibility
REPORT_DIR = OUTPUT_DIR / "reports"
METRICS_DIR = OUTPUT_DIR / "metrics"
BENCHMARKS_DIR = OUTPUT_DIR / "benchmarks"
CALIBRATION_DIR = OUTPUT_DIR / "calibration"
OBJECTIVES_DIR = OUTPUT_DIR / "objectives"
STATISTICAL_TESTS_DIR = OUTPUT_DIR / "statistical_tests"
COMPUTATIONAL_DIR = OUTPUT_DIR / "computational"
FEATURE_ENG_DIR = OUTPUT_DIR / "feature_engineering"
LOGS_DIR = OUTPUT_DIR / "logs"
MODELS_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

AUDITS_DIR = OUTPUT_DIR / "audits"
SPLITS_DIR = OUTPUT_DIR / "splits"
TUNING_DIR = OUTPUT_DIR / "tuning"
RANDOM_SPLIT_DIR = OUTPUT_DIR / "random_split"
COMPARISONS_DIR = OUTPUT_DIR / "comparisons"

ALL_OUTPUT_DIRS = [
    OUTPUT_DIR, FIGURES_DIR, REPORT_DIR, METRICS_DIR, BENCHMARKS_DIR,
    CALIBRATION_DIR, OBJECTIVES_DIR, STATISTICAL_TESTS_DIR, COMPUTATIONAL_DIR,
    FEATURE_ENG_DIR, LOGS_DIR, MODELS_DIR, PREDICTIONS_DIR,
    AUDITS_DIR, SPLITS_DIR, TUNING_DIR, RANDOM_SPLIT_DIR, COMPARISONS_DIR,
]

for _d in ALL_OUTPUT_DIRS:
    _d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Reproducibility & Bootstrap Settings
# ──────────────────────────────────────────────────────────────
RANDOM_SEED: int = 42

# ─────────────────────────────────────────────────────────────────────
# Multi-seed robustness check (backbone + ensemble)
#
# Motivation: an exploratory test (not part of the main pipeline) found
# NeuralCQR test R2 on this temporal split ranged from 0.30 to 0.46
# across 4 random seeds -- a bigger swing than any single modeling
# change tried. It also found validation R2 and test R2 were NOT
# reliably correlated across seeds (the seed with the best val R2 was
# NOT the seed with the best test R2), meaning the 2016-2018 validation
# window does not reliably rank models for 2019-2023 test performance.
# This is itself evidence for the paper's central claim (static
# calibration/selection doesn't transfer under distribution shift), so
# it's reported here as a formal robustness check rather than hidden.
#
# POST-HOC SUPPLEMENTARY TEST-SET ROBUSTNESS ANALYSIS — NOT USED FOR MODEL SELECTION
# When enabled, main() retrains NeuralCQR (and the NeuralCQR+LightGBM
# ensemble) across ROBUSTNESS_SEEDS on the SAME temporal split, and
# reports mean +/- std for Test R2, RMSE, and PICP, plus the Spearman
# correlation between per-seed val R2 and per-seed test R2.
# CRITICAL: This is strictly post-hoc supplementary analysis. It is NOT
# used for model selection. Never choose the best seed based on test results.
# ─────────────────────────────────────────────────────────────────────
ENABLE_BACKBONE_ROBUSTNESS_CHECK: bool = False  # Post-hoc supplementary only; primary locked model strictly uses seed 42
ROBUSTNESS_SEEDS: List[int] = [42, 7, 123]  # RANDOM_SEED (42) always included first

# LOSO robustness is far more expensive (6 states x N seeds, each fold
# already ~3-5 min), so it defaults OFF. Enable explicitly if you have
# the compute budget -- see run_loso_robustness_check() in main.py.
ENABLE_LOSO_ROBUSTNESS_CHECK: bool = False
LOSO_ROBUSTNESS_SEEDS: List[int] = [42, 7]
SEED: int = 42
STABILITY_SEEDS: List[int] = [42, 123, 2024, 3407, 7]
BOOTSTRAP_ITERATIONS: int = 2000

# ──────────────────────────────────────────────────────────────
# Training Epoch Budgets & Architecture Parameters (§2, §3, §4.2)
# ──────────────────────────────────────────────────────────────
# Architectural configuration rationale:
# Historical exploratory result from an earlier experiment; not used for current model selection.
# The original exploratory config (lr=1e-3, hidden=(256,128,64,32), batch=64,
# es_mode="pinball", patience=20) caused validation RMSE to peak early and overfit
# on tabular input. Selected architecture uses lower LR, a compact 2-layer backbone (64, 32),
# gradient stability via larger batches, weight decay, and early stopping tracked on
# DEV RMSE.
BASELINE_MAX_EPOCHS: int = 200
OPTUNA_MAX_EPOCHS: int = 200
MAX_EPOCHS: int = 200  # Alias for BASELINE_MAX_EPOCHS
MAX_TUNING_EPOCHS: int = 200  # Alias for OPTUNA_MAX_EPOCHS
EARLY_STOPPING_PATIENCE: int = 60
EARLY_STOPPING_MODE: str = "rmse"
LEARNING_RATE: float = 3e-5
WEIGHT_DECAY: float = 1e-3
DROPOUT: float = 0.2
BATCH_SIZE: int = 128
HIDDEN_DIMS: Tuple[int, ...] = (64, 32)
NEURAL_CQR_HIDDEN_DIMS: Tuple[int, ...] = (64, 32)
NEURAL_CQR_DROPOUT: float = 0.2
LAMBDA_PINBALL: float = 1.0
LAMBDA_HUBER: float = 1.0
LAMBDA_CROSSING: float = 10.0
LAMBDA_WIDTH: float = 0.005

# ─────────────────────────────────────────────────────────────────────
# W3 / F0 ladder flags (added 2026-09-17)
#
# Every flag below defaults to the behaviour of the 2026-09-16 run, so
# enabling one is a single, DEV-testable experimental variable (spec
# docs/superpowers/specs/2026-09-17-paper3-audit-remediation-design.md,
# rules D2/D3). Nothing here may be selected on TEST years or on a
# held-out LOSO state.
# ─────────────────────────────────────────────────────────────────────

# F0a — target standardisation and Huber delta.
#   The detrended FIT target has sd = 1.785 t/ha, so the legacy
#   delta = 1.0 sits at 0.56 sigma: Huber runs in its linear (MAE)
#   region for 58.8% of FIT rows, which makes the mean head estimate a
#   conditional median while RMSE/R2 (and LightGBM's default L2
#   objective) target the mean. With residual skew -0.79 the two differ
#   most in the low-yield years.
STANDARDIZE_TARGET: bool = False      # True -> train on (y - mu)/sigma from the TRAIN partition only
HUBER_DELTA: float = 1.0              # used when STANDARDIZE_TARGET is False (legacy value)
HUBER_DELTA_SIGMA: float = 1.345      # used when STANDARDIZE_TARGET is True; classic robust choice

# F0b — exponential moving average of weights.
#   ModelEMA kept the shadow but never applied it, so the final refit
#   (no validation, no early stopping) used the last epoch's weights.
USE_EMA_WEIGHTS: bool = False         # True -> the no-validation refit returns EMA-averaged weights
EMA_DECAY: float = 0.99

# F0c — learning-rate schedule.
#   CosineAnnealingLR(eta_min=1e-5) with LEARNING_RATE=3e-5 decays only
#   from 3e-5 to 1e-5. T_max also equals the epoch budget, so the final
#   refit follows a different LR curve than the DEV run that chose its
#   epoch count.
LR_ETA_MIN: float = 1e-5              # legacy absolute floor
LR_SCHEDULE_MATCH_DEV: bool = False   # True -> refit reuses the DEV run's T_max so the LR curve matches

# F1 — monotone quantile heads.
#   q50 = f, q05 = f - softplus(a), q95 = f + softplus(b): crossing
#   becomes impossible by construction instead of being penalised.
NEURAL_MONOTONE_HEADS: bool = False

# LightGBM validation handling.
#   train_lgbm_quantile accepted X_val/y_val and ignored them, so every caller
#   that passed a validation set got the full tree budget anyway. Default False
#   keeps that behaviour (and the locked results); True actually early-stops on
#   the supplied set, which is a protocol change and must be declared.
LGBM_DEV_EARLY_STOPPING: bool = False

# C7 — use LOSO hyperparameters re-derived on each fold's own DEV rows.
#   Produce the selection file first:  python code/loso_dev_tuning.py
#   Then set this True and rerun. Off by default so results do not change
#   until the tuning run exists. The LOSO macro R2 is expected to FALL: the
#   incumbent settings were chosen by watching held-out performance.
LOSO_USE_DEV_SELECTED_HP: bool = False

# C5 — SA-ACI severity weighting (audit blocker).
#   The published implementation DIVIDED the conformal threshold by the
#   severity weight, so intervals got NARROWER exactly where the compound
#   drought-heat stress was worst. The methodology text claimed the
#   weighting was disabled, but no switch existed. Both are fixed here:
#   the weight now widens, and the switch is real.
ACI_SEVERITY_WEIGHTING: bool = True   # False -> fixed window, unit weights (windowed ACI)
ACI_SEVERITY_LAMBDA: float = 0.05     # lambda in E_i * (1 + lambda * log1p(S_i))

# LOSO-SPECIFIC HYPERPARAMETERS
#
# C7 (2026-09-18): the selection history for these values lived here and
# documented six rounds judged by leave-one-state-out R2 -- i.e. chosen with
# held-out information. It now lives in
# docs/supplement/development_history.md, disclosed rather than deleted.
#
# These values are PENDING re-derivation on each fold's own DEV rows. Until
# that lands, every LOSO number carries the C7 caveat, and
# code/frozen_protocol.py records them as PENDING.
LOSO_LEARNING_RATE: float = 1e-3
LOSO_BATCH_SIZE: int = 64
LOSO_WEIGHT_DECAY: float = 1e-4
LOSO_MAX_EPOCHS: int = 200
LOSO_EARLY_STOPPING_PATIENCE: int = 20
LOSO_EARLY_STOPPING_MODE: str = "pinball"

# LOSO architecture: kept separate from the temporal-split net. Selection
# history (including the round that motivated the larger net) is in
# docs/supplement/development_history.md. Also PENDING re-derivation on DEV.
LOSO_HIDDEN_DIMS: Tuple[int, ...] = (256, 128, 64, 32)
LOSO_DROPOUT: float = 0.25

# ─────────────────────────────────────────────────────────────────────
# LOSO role separation (audit fix)
#
# The LOSO folds previously had only train/val/test, and the val fold was
# concatenated into training before being reused for early stopping, ensemble
# weight selection AND conformal calibration. Roles are now separated using the
# project's own locked temporal boundaries within the five non-held-out states:
#   fit  = FIT_YEARS   (1985-2013)  -> model fitting (internal ES carve-out)
#   dev  = DEV_YEARS   (2014-2015)  -> ensemble-weight selection only
#   cal  = CAL_YEARS[0]+ (2016-2023)-> conformal calibration only
#   test = held-out state, all years-> final evaluation only
# The six-state list above is untouched.
#
# LOSO_PER_FOLD_FEATURE_SELECTION: the locked protocol says the held-out state
# must never enter "any other learned component" before final evaluation.
# Consensus feature selection is a learned component, so each fold re-runs it on
# its own fit partition instead of inheriting the main temporal pipeline's
# feature set (which was selected using all states, held-out one included).
LOSO_PER_FOLD_FEATURE_SELECTION: bool = True

# ──────────────────────────────────────────────────────────────
# Temporal Split Boundaries (Strict 4-Way Temporal Structure)
# ──────────────────────────────────────────────────────────────
# 1985-2013: Model fitting (weights, detrending, county baseline, feature selection, scaler)
# 2014-2015: Internal development & early stopping & ensemble weight selection
# 2016-2018: Conformal calibration (purely fresh, zero model selection reuse)
# 2019-2023: Final test (locked until evaluation)
FIT_YEARS: Tuple[int, int] = (1985, 2013)
DEV_YEARS: Tuple[int, int] = (2014, 2015)
CAL_YEARS: Tuple[int, int] = (2016, 2018)
TEST_YEARS: Tuple[int, int] = (2019, 2023)

# LEGACY — NOT USED BY AUTHORITATIVE TEMPORAL PIPELINE:
# The authoritative main pipeline strictly uses FIT_YEARS, DEV_YEARS, CAL_YEARS, TEST_YEARS.
# Kept strictly for backward-compatible unit tests / legacy loaders.
TRAIN_YEARS: Tuple[int, int] = (1985, 2015)  # Historical 3-way train (FIT + DEV)
VAL_YEARS: Tuple[int, int] = (2016, 2018)    # Historical 3-way val (CAL)
CALIBRATION_YEARS: Tuple[int, int] = CAL_YEARS

# ──────────────────────────────────────────────────────────────
# LOSO-CV states (§5.2 of methodology)
# ──────────────────────────────────────────────────────────────
LOSO_STATES: List[str] = [
    # Nebraska removed from LOSO: dropped via DROP_STATES (irrigated, breaks
    # drought-yield transfer -> empty fold). This is 6-state LOSO-CV.
    "Illinois", "Indiana", "Iowa", "Minnesota",
    "Missouri", "Ohio",
]

# ──────────────────────────────────────────────────────────────
# Target variables
# ──────────────────────────────────────────────────────────────
PRIMARY_TARGET: str = "Corn_Yield_tha" if _CROP_TARGET == "corn" else "Soy_Yield_tha"

# SECONDARY_TARGET is a leftover from the two-crop era. The PRISM-NLDAS master
# dataset carries CORN ONLY -- there is no Soy_Yield_tha column in
# Paper3_Processed.csv -- so under the master data source this constant resolves
# to a column that does not exist. It is kept because eda.py and
# visualization.py iterate over both targets, and both already skip a missing
# column. Nothing in the paper is computed on soybean; adding it would mean
# re-extracting USDA NASS and rebuilding the master dataset.
SECONDARY_TARGET: str = "Soy_Yield_tha" if _CROP_TARGET == "corn" else "Corn_Yield_tha"
TARGET_COLS: List[str] = [
    "Corn_Yield_tha", "Corn_Yield_buacre",
    "Soy_Yield_tha", "Soy_Yield_buacre",
]
YIELD_PRESENCE_COLS: List[str] = ["Has_Corn_Yield", "Has_Soy_Yield"]
PRIMARY_YIELD_PRESENCE_COL: str = "Has_Corn_Yield" if _CROP_TARGET == "corn" else "Has_Soy_Yield"

# ──────────────────────────────────────────────────────────────
# R² IMPROVEMENT PATCH (backbone performance)  --  added per request
# All toggles below are leakage-free (fit on TRAIN split only).
# ──────────────────────────────────────────────────────────────
# 1. Drop Nebraska (heavily irrigated; breaks drought-yield transfer)
DROP_STATES: List[str] = ["Nebraska"]

# 2. Detrend the target: model technology-adjusted yield anomaly, add
#    the linear yield trend (fit on TRAIN years only) back at predict time.
#    This is the single biggest R² fix (train yields ~8.2, test ~11.3 t/ha).
DETREND_TARGET: bool = True

# 3. Multicollinearity removal thresholds (applied on TRAIN only)
CORR_DROP_THRESHOLD: float = 0.95      # drop one of any |r|>0.95 pair
VIF_DROP_THRESHOLD: float = 10.0       # iteratively drop VIF>10
# Core scientific features protected from removal even if flagged:
MULTICOLLINEARITY_PROTECT: List[str] = [
    "CDHW_Severity_Score", "CDHW_Event_Count", "SPEI_30_min", "CDHW_Severity_silking",
]

# 4. Add a leakage-free county baseline feature (each county's mean TRAIN
#    detrended anomaly; unseen counties -> 0).
ADD_COUNTY_BASELINE: bool = True

# ──────────────────────────────────────────────────────────────
# Feature groups
# ──────────────────────────────────────────────────────────────
ID_COLS: List[str] = [
    "GEOID", "County_Name", "State", "STATEFP", "COUNTYFP",
    "Lat", "Lon", "Year", "Split",
]

CDHW_COLS: List[str] = [
    "CDHW_Flag", "CDHW_Event_Count", "CDHW_Severity_Score",
]

# SPEI-30 / SPI-30 features (§4.1 methodology mandate: SPEI-30 supersedes SPI-30)
SPEI_PREFERRED_COLS: List[str] = ["SPEI_30_min", "SPEI_30_mean"]
SPI_FALLBACK_COLS: List[str] = ["SPI_30_min", "SPI_30_mean"]

WEATHER_COLS: List[str] = [
    "SPI_30_min", "SPI_30_mean",
    "GDD_Accumulated", "Tmax_Days_Above_35", "Precip_growseason_mm",
    "ERA5d_Tmax_mean_C", "ERA5d_Tmax_max_C",
    "ERA5d_Tmin_mean_C", "ERA5d_DiurnalRange_C",
]

ENSO_COLS: List[str] = [
    "ONI_annual_mean", "ONI_annual_max", "ONI_annual_min",
    "ONI_growseason_mean", "ONI_DJF",
    "ENSO_Phase", "ENSO_Anomalous_Year",
]

DROUGHT_COLS: List[str] = [
    "DSCI_growseason_mean", "DSCI_growseason_max",
    "Drought_Severe_Flag",
]

STORM_COLS: List[str] = [
    "Storm_Events_Total",
    "Storm_Hail_Count", "Storm_Flood_Count",
    "Storm_Heat_Count", "Storm_Drought_Event_Count",
    "Storm_Tornado_Count", "Storm_Wind_Count",
]

DISASTER_INDICATOR_FLAGS: Dict[str, str] = {
    "drought": "Has_Drought",
    "storm": "Has_Storm",
}

SOIL_COLS: List[str] = [
    "Soil_AWC_Mean", "Soil_BD_Mean",
    "Soil_Clay_Mean", "Soil_Sand_Mean",
    "Soil_Silt_Mean", "Soil_OC_Mean",
]

TOPO_COLS: List[str] = ["DEM_Mean", "Slope_Mean"]

LANDCOVER_COLS: List[str] = [
    "Open_Water_frac", "Developed_Openspace_frac",
    "Developed_lowintensity_frac", "Developed_mediumintensity_frac",
    "Developed_highintensity_frac", "BarrenLand_frac",
    "DeciduousForest_frac", "EvergreenForest_frac",
    "MixedForest_frac", "Shrub_frac", "Grassland_frac",
    "Pasture_frac", "CultivatedCrops_frac",
    "WoodyWetlands_frac", "HerbaceousWetlands_frac",
]

AREA_COLS: List[str] = ["ALAND", "AWATER"]

DATA_AVAILABILITY_FLAGS: List[str] = [
    "Has_Drought", "Has_Storm", "Has_ERA5", "Has_ERA5_daily", "Has_CDHW",
]

CONSTANT_COLS: List[str] = [
    "Storm_Heat_Count", "Storm_Drought_Event_Count",
    "Has_ERA5_daily", "Has_CDHW",
]

REDUNDANT_TARGET_COLS: List[str] = [
    "Corn_Yield_buacre", "Soy_Yield_buacre",
]

ANOMALOUS_YEARS: List[int] = [1988, 1997, 1998, 2012, 2015, 2016]

# NeuralCQR Architecture & Loss Hyperparameters (§2 & §3) -- see R2 FIX note above
NEURAL_CQR_HIDDEN_DIMS: Tuple[int, ...] = (64, 32)
NEURAL_CQR_DROPOUT: float = 0.2
LAMBDA_PINBALL: float = 1.0
LAMBDA_HUBER: float = 1.0
LAMBDA_CROSSING: float = 10.0
LAMBDA_WIDTH: float = 0.005

# Fourier Time Encoding (§5)
FOURIER_PERIODS: List[float] = [1.0, 3.0, 5.0, 7.0, 11.0, 19.0]

# Static Environmental Context Descriptors (§6)
STATIC_CONTEXT_COLS: List[str] = [
    "Lat", "Lon", "DEM_Mean", "Slope_Mean",
    "Soil_AWC_Mean", "Soil_BD_Mean", "Soil_Clay_Mean",
    "Soil_Sand_Mean", "Soil_Silt_Mean", "Soil_OC_Mean",
    "CultivatedCrops_frac", "Pasture_frac",
]

# CQR / ACI hyperparameters (§4.2 & §4.3 of methodology)
CQR_QUANTILES: Tuple[float, float] = (0.05, 0.95)
NOMINAL_COVERAGE: float = 0.90
NOMINAL_ALPHA: float = 1.0 - NOMINAL_COVERAGE  # 0.10
ACI_WINDOW_SIZE: int = 3  # 3-year sliding window (§4.3)
ACI_GAMMA: float = 0.05   # step-size for online update (§4.3)

# Bootstrap settings (§4.4 & §5.3) - Authoritative definition at top (2000)

LGBM_PARAMS: Dict = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 7,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

CALIBRATION_LEVELS: List[float] = [0.80, 0.85, 0.90, 0.95]

PLOT_DPI: int = 300
PLOT_FORMAT: str = "png"

# ──────────────────────────────────────────────────────────────
# Centralized Phenology & CDHW Thresholds (§4.1)
# ──────────────────────────────────────────────────────────────
GDD_VEGETATIVE_END: int = 700      # End of vegetative stage
GDD_SILKING_END: int = 1400        # End of silking / R1 stage
TMAX_THRESHOLD_C: float = 35.0      # Heat stress threshold (°C)
SPEI_THRESHOLD: float = -1.0        # Drought stress threshold

# Phenology stage weights for severity scoring
VEG_WEIGHT_PRIMARY: float = 1.0
VEG_WEIGHT_SECONDARY: float = 0.2
SILKING_WEIGHT_PRIMARY: float = 1.5
SILKING_WEIGHT_SECONDARY: float = 0.2
GRAIN_WEIGHT_PRIMARY: float = 0.8
GRAIN_WEIGHT_SECONDARY: float = 0.2

# Anomaly event thresholds
EXTREME_EVENT_COUNT_THRESHOLD: int = 3
MODERATE_EVENT_COUNT_THRESHOLD: int = 1



# ──────────────────────────────────────────────────────────────
# Data-interface switches (legacy defaults reproduce the original behaviour)
# ──────────────────────────────────────────────────────────────
ENSO_AS_FEATURES: bool = True                    # one-hot ENSO_Phase into features (legacy behaviour)
INTERACTION_TMAX_COL: str = "ERA5d_Tmax_max_C"   # Tmax column used by Inter_SPEI_Tmax
MASTER_ROLE_ALIASES: Dict[str, str] = {}         # master column -> methodology role name (master only)

# ──────────────────────────────────────────────────────────────
# MASTER DATA PROFILE (PAPER3_DATA_SOURCE=master)
#
# Feature groups are mapped role-for-role onto the PRISM-NLDAS-NOAA-USDA master
# dataset. Source-neutral methodology roles keep their historical names through
# MASTER_ROLE_ALIASES (a load-time rename, never a duplicate column); everything
# else keeps its master name. No ERA5-named column is created: the one ERA5-named
# role (Inter_SPEI_Tmax) reads INTERACTION_TMAX_COL instead. ENSO labels are
# evaluation-only metadata (ID_COLS) because annual ONI includes post-harvest
# months. See master_dataset/metadata/ for the variable catalog and provenance.
# ──────────────────────────────────────────────────────────────
if DATA_SOURCE == "master":
    MASTER_ROLE_ALIASES = {
        "cdhw_cumulative_severity_gs": "CDHW_Severity_Score",       # sum |SPEI30| x (Tmax-35), growing season
        "cdhw_total_days_gs": "CDHW_Event_Count",                   # CDHW day count (old column was a day count too)
        "cdhw_cumulative_severity_veg": "CDHW_Severity_vegetative",
        "cdhw_cumulative_severity_silk": "CDHW_Severity_silking",
        "cdhw_cumulative_severity_grain": "CDHW_Severity_grainfill",
        "spei30_prism_nldas_min_gs": "SPEI_30_min",
        "spei30_prism_nldas_mean_gs": "SPEI_30_mean",
        "prism_gdd_gs": "GDD_Accumulated",                          # season-total GDD (drives Phenological_Window)
        "prism_tmax_days_gt35_gs": "Tmax_Days_Above_35",
        "prism_ppt_total_gs": "Precip_growseason_mm",
    }
    ENSO_AS_FEATURES = False
    INTERACTION_TMAX_COL = "prism_tmax_max_gs"

    ID_COLS = ID_COLS + ["ENSO_Phase", "ENSO_Anomalous_Year"]      # evaluation stratification only
    TARGET_COLS = ["Corn_Yield_tha", "Corn_Yield_buacre"]
    YIELD_PRESENCE_COLS = ["Has_Corn_Yield"]
    REDUNDANT_TARGET_COLS = ["Corn_Yield_buacre"]

    # CDHW_COLS and SPEI_PREFERRED_COLS keep their names (role aliases); MULTICOLLINEARITY_PROTECT likewise.
    SPI_FALLBACK_COLS = ["spi1_prism_min_gs", "spi1_prism_mean_gs"]
    WEATHER_COLS = [                                                # role-for-role with the legacy list
        "spi1_prism_min_gs", "spi1_prism_mean_gs",                  # SPI_30_min / SPI_30_mean
        "GDD_Accumulated", "Tmax_Days_Above_35", "Precip_growseason_mm",
        "prism_tmax_mean_gs", "prism_tmax_max_gs",                  # ERA5d_Tmax_mean_C / ERA5d_Tmax_max_C
        "prism_tmin_mean_gs",                                       # ERA5d_Tmin_mean_C (diurnal range = tmax - tmin, not kept)
    ]
    ENSO_COLS = []                                                  # not features in the master profile
    DROUGHT_COLS = []                                               # USDM excluded (2000+ only, impact-informed)
    STORM_COLS = ["noaa_hail_count_gs", "noaa_thunderstorm_wind_count_gs", "noaa_tornado_count_gs", "noaa_max_hail_size_gs"]
    DISASTER_INDICATOR_FLAGS = {}
    SOIL_COLS = ["soil_awc", "soil_bulk_density", "soil_clay", "soil_silt", "soil_organic_carbon"]
    TOPO_COLS = ["elevation_mean_m", "elevation_std_m", "slope_mean"]
    LANDCOVER_COLS = ["nlcd_cropland_frac", "nlcd_pasture_hay_frac", "nlcd_forest_frac", "nlcd_grass_shrub_frac",
                      "nlcd_developed_frac", "nlcd_water_wetland_frac"]
    AREA_COLS = []
    DATA_AVAILABILITY_FLAGS = []
    CONSTANT_COLS = []
    STATIC_CONTEXT_COLS = [
        "Lat", "Lon", "elevation_mean_m", "slope_mean",
        "soil_awc", "soil_bulk_density", "soil_clay", "soil_silt", "soil_organic_carbon",
        "nlcd_cropland_frac", "nlcd_pasture_hay_frac",
    ]
