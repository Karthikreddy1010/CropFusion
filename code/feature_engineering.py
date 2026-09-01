"""
feature_engineering.py - Fourier Time Encodings, Static Environmental Descriptors & Climate Features (§4.1).

Computes:
1. Flexible Fourier Temporal Encodings (Annual, Multi-Year, Low-Frequency, Harmonics)
2. Static Environmental Context Features (Elevation, Soil, Land Cover, Climatological Normals)
3. Lag Yield Features (Yield_lag1, Yield_lag2) without target leakage
4. Rolling Climate Statistics & Seasonal Aggregations
5. Non-linear Climate Interaction Features (SPEI x Tmax, CDHW x Phenology, etc.)
6. Phenology-aligned Compound Drought-Heatwave (CDHW) severity scores (§4.1)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import config as cfg
from utils import log_decision, save_report

logger = logging.getLogger("paper3")


def engineer_features(
    df: pd.DataFrame,
    train_mask: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply complete feature engineering pipeline matching Paper 3 methodology.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    train_mask : Optional[np.ndarray]
        Boolean mask indicating training split rows. Used to compute historical
        normals WITHOUT data leakage from val/test sets or held-out states.
    """
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE (§4.1)")
    logger.info("=" * 60)

    df = df.copy()
    report: Dict[str, Any] = {"features_added": [], "decisions": []}

    # ── 1. Validate existing CDHW columns ────────────────────
    report["cdhw_existing"] = _validate_cdhw(df)

    if "CDHW_Severity_vegetative" in df.columns and "CDHW_Veg_Severity" not in df.columns:
        df["CDHW_Veg_Severity"] = df["CDHW_Severity_vegetative"]
    if "CDHW_Severity_silking" in df.columns and "CDHW_Silking_Severity" not in df.columns:
        df["CDHW_Silking_Severity"] = df["CDHW_Severity_silking"]
    if "CDHW_Severity_grainfill" in df.columns and "CDHW_GrainFill_Severity" not in df.columns:
        df["CDHW_GrainFill_Severity"] = df["CDHW_Severity_grainfill"]

    # ── 2. Phenological window classification ────────────────
    if "Phenological_Window" not in df.columns:
        df, pheno_info = _classify_phenological_window(df)
        report["phenological_windows"] = pheno_info
        report["features_added"].append("Phenological_Window")

    # ── 3. Phenology-aligned CDHW severity scores (§4.1) ─────
    pre_existing_veg = "CDHW_Veg_Severity" in df.columns
    pre_existing_silk = "CDHW_Silking_Severity" in df.columns
    pre_existing_grain = "CDHW_GrainFill_Severity" in df.columns

    if not (pre_existing_veg and pre_existing_silk and pre_existing_grain):
        df, pheno_cdhw_info = _compute_phenology_aligned_cdhw(df)
        report["phenology_aligned_cdhw"] = pheno_cdhw_info
        report["features_added"].extend([
            "CDHW_Veg_Severity", "CDHW_Silking_Severity", "CDHW_GrainFill_Severity"
        ])

    # ── 4. Year-type & ENSO phase encoding ───────────────────
    if "Year_Type" not in df.columns:
        df, yeartype_info = _classify_year_type(df)
        report["year_type_classification"] = yeartype_info
        report["features_added"].append("Year_Type")

    if "ENSO_Phase" in df.columns:
        enso_cols = [c for c in df.columns if c.startswith("ENSO_") and c not in ["ENSO_Phase", "ENSO_Anomalous_Year"]]
        if not enso_cols:
            dummies = pd.get_dummies(df["ENSO_Phase"], prefix="ENSO", drop_first=False)
            for col in dummies.columns:
                df[col] = dummies[col].astype(int)
                report["features_added"].append(col)

    # ── 5. Flexible Fourier Temporal Encodings (§5) ────────────
    df, fourier_cols = _compute_fourier_time_encodings(df)
    report["features_added"].extend(fourier_cols)

    # ── 6. Non-linear Climate Interactions ────────────────────
    df, inter_cols = _compute_climate_interaction_features(df)
    report["features_added"].extend(inter_cols)

    # Note: Split-aware target lag features (Yield_lag1, Yield_lag2) and
    # split-aware rolling climate features (Rolling2yr_*, Rolling3yr_*)
    # are constructed strictly post-split via build_split_aware_lags() and
    # build_split_aware_rolling_features() to eliminate any pre-split ambiguity.
    #
    # Climatological normals and anomalies are strictly learned statistics
    # and are computed exclusively on training splits via TrainFittedPreprocessor.

    # ── 7. Reports ───────────────────────────────────────────
    validation_report = _generate_feature_validation_report(df)
    lineage_report = _generate_feature_lineage_report(df, report["features_added"])

    log_decision(
        step="feature_engineering",
        decision=f"Engineered {len(report['features_added'])} deterministic features (Fourier encodings, "
                 "phenology-aligned CDHW severity, climate interactions) before dataset splitting",
        reason="Preserves deterministic features pre-split while deferring target lags and rolling climate to split-aware stage",
    )

    save_report(report, "feature_engineering_report.json")
    save_report(validation_report, "feature_validation_report.json")
    save_report(lineage_report, "feature_lineage_report.json")
    return df, report


def _compute_fourier_time_encodings(
    df: pd.DataFrame,
    periods: Optional[List[float]] = None,
) -> Tuple[pd.DataFrame, list]:
    """Compute sine and cosine Fourier temporal encodings for multiple periods (§5)."""
    added = []
    if "Year" not in df.columns:
        return df, added

    if periods is None:
        periods = getattr(cfg, "FOURIER_PERIODS", [1.0, 3.0, 5.0, 7.0, 11.0, 19.0])

    t = df["Year"].values.astype(float)
    # Deterministic configured reference origin (1985.0) independent of dataframe subset (§10)
    t_min = float(getattr(cfg, "TRAIN_YEARS", (1985, 2015))[0])

    for P in periods:
        sin_col = f"Fourier_sin_P{P:.1f}".replace(".", "_")
        cos_col = f"Fourier_cos_P{P:.1f}".replace(".", "_")

        freq = 2.0 * np.pi / max(0.1, P)
        df[sin_col] = np.round(np.sin(freq * (t - t_min)), 6)
        df[cos_col] = np.round(np.cos(freq * (t - t_min)), 6)

        added.extend([sin_col, cos_col])

    logger.info("Fourier temporal encodings computed for periods %s (reference origin=%.1f) -> Added %d features",
                periods, t_min, len(added))
    return df, added


def build_split_aware_lags(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = cfg.PRIMARY_TARGET,
    split_type: str = "temporal",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct county-level target lag features with strict split awareness.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split partition.
    val_df : pd.DataFrame
        Validation split partition.
    test_df : pd.DataFrame
        Testing split partition.
    target_col : str
        Target column name (e.g. 'Corn_Yield_tha').
    split_type : str
        'temporal', 'random_row', 'random_grouped', or 'loso'.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df) with Yield_lag1 and Yield_lag2 columns.
    """
    tr = train_df.copy()
    va = val_df.copy()
    te = test_df.copy()

    if target_col not in tr.columns or "GEOID" not in tr.columns or "Year" not in tr.columns:
        return tr, va, te

    if split_type == "temporal":
        # In temporal split: combine past chronology
        combined = pd.concat([tr, va, te], axis=0).sort_values(["GEOID", "Year"])
        lag1_s = combined.groupby("GEOID")[target_col].shift(1)
        lag2_s = combined.groupby("GEOID")[target_col].shift(2)
        lag1_year = combined.groupby("GEOID")["Year"].shift(1)
        lag2_year = combined.groupby("GEOID")["Year"].shift(2)

        # Enforce exact 1-year and 2-year distances
        mask_v1 = (lag1_year == combined["Year"] - 1)
        mask_v2 = (lag2_year == combined["Year"] - 2)

        combined["Yield_lag1"] = np.where(mask_v1, lag1_s, np.nan)
        combined["Yield_lag2"] = np.where(mask_v2, lag2_s, np.nan)

        tr["Yield_lag1"] = combined.loc[tr.index, "Yield_lag1"]
        tr["Yield_lag2"] = combined.loc[tr.index, "Yield_lag2"]
        va["Yield_lag1"] = combined.loc[va.index, "Yield_lag1"]
        va["Yield_lag2"] = combined.loc[va.index, "Yield_lag2"]
        te["Yield_lag1"] = combined.loc[te.index, "Yield_lag1"]
        te["Yield_lag2"] = combined.loc[te.index, "Yield_lag2"]

    elif split_type in ["random_row", "random_grouped", "loso"]:
        # Strict partition isolation: Source target MUST come from TRAIN ONLY.
        # Val and test targets are NEVER used as feature lag sources.
        tr_valid = tr[tr[target_col].notna()]
        train_target_lookup = tr_valid.set_index(["GEOID", "Year"])[target_col].to_dict()

        for df_part in [tr, va, te]:
            lag1_vals = [
                train_target_lookup.get((int(g), int(y) - 1), np.nan)
                for g, y in zip(df_part["GEOID"], df_part["Year"])
            ]
            lag2_vals = [
                train_target_lookup.get((int(g), int(y) - 2), np.nan)
                for g, y in zip(df_part["GEOID"], df_part["Year"])
            ]
            df_part["Yield_lag1"] = lag1_vals
            df_part["Yield_lag2"] = lag2_vals

    else:
        raise ValueError(f"Unknown split_type '{split_type}'. Must be 'temporal', 'random_row', 'random_grouped', or 'loso'.")

    logger.info("Split-aware lag features constructed (split_type='%s', tr=%d, va=%d, te=%d)",
                split_type, len(tr), len(va), len(te))
    return tr, va, te


def build_split_aware_rolling_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_type: str = "temporal",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct county-level rolling climate features with strict split awareness.

    Features computed (backward-looking including current season t):
    - Rolling2yr_Precip_growseason_mm_mean (window: [t-1, t])
    - Rolling3yr_Precip_growseason_mm_mean (window: [t-2, t-1, t])
    - Rolling2yr_GDD_Accumulated_mean (window: [t-1, t])
    - Rolling3yr_GDD_Accumulated_mean (window: [t-2, t-1, t])
    - Rolling2yr_Tmax_Days_Above_35_mean (window: [t-1, t])
    - Rolling3yr_Tmax_Days_Above_35_mean (window: [t-2, t-1, t])
    """
    tr = train_df.copy()
    va = val_df.copy()
    te = test_df.copy()

    climate_vars = ["Precip_growseason_mm", "GDD_Accumulated", "Tmax_Days_Above_35"]
    climate_vars = [c for c in climate_vars if c in tr.columns]

    if not climate_vars or "GEOID" not in tr.columns or "Year" not in tr.columns:
        return tr, va, te

    if split_type == "temporal":
        combined = pd.concat([tr, va, te], axis=0).sort_values(["GEOID", "Year"])
        for var in climate_vars:
            r2_col = f"Rolling2yr_{var}_mean"
            r3_col = f"Rolling3yr_{var}_mean"
            combined[r2_col] = combined.groupby("GEOID")[var].transform(lambda g: g.rolling(2, min_periods=1).mean())
            combined[r3_col] = combined.groupby("GEOID")[var].transform(lambda g: g.rolling(3, min_periods=1).mean())

            tr[r2_col] = combined.loc[tr.index, r2_col]
            tr[r3_col] = combined.loc[tr.index, r3_col]
            va[r2_col] = combined.loc[va.index, r2_col]
            va[r3_col] = combined.loc[va.index, r3_col]
            te[r2_col] = combined.loc[te.index, r2_col]
            te[r3_col] = combined.loc[te.index, r3_col]

    elif split_type == "random_row":
        # Train-only rolling
        tr_sorted = tr.sort_values(["GEOID", "Year"])
        for var in climate_vars:
            r2_col = f"Rolling2yr_{var}_mean"
            r3_col = f"Rolling3yr_{var}_mean"
            tr_sorted[r2_col] = tr_sorted.groupby("GEOID")[var].transform(lambda g: g.rolling(2, min_periods=1).mean())
            tr_sorted[r3_col] = tr_sorted.groupby("GEOID")[var].transform(lambda g: g.rolling(3, min_periods=1).mean())
            tr[r2_col] = tr_sorted.loc[tr.index, r2_col]
            tr[r3_col] = tr_sorted.loc[tr.index, r3_col]

        # Val rolling (using permitted train context + val)
        tr_va_combined = pd.concat([tr, va], axis=0).sort_values(["GEOID", "Year"])
        for var in climate_vars:
            r2_col = f"Rolling2yr_{var}_mean"
            r3_col = f"Rolling3yr_{var}_mean"
            tr_va_combined[r2_col] = tr_va_combined.groupby("GEOID")[var].transform(lambda g: g.rolling(2, min_periods=1).mean())
            tr_va_combined[r3_col] = tr_va_combined.groupby("GEOID")[var].transform(lambda g: g.rolling(3, min_periods=1).mean())
            va[r2_col] = tr_va_combined.loc[va.index, r2_col]
            va[r3_col] = tr_va_combined.loc[va.index, r3_col]

        # Test rolling (using permitted train context + test)
        tr_te_combined = pd.concat([tr, te], axis=0).sort_values(["GEOID", "Year"])
        for var in climate_vars:
            r2_col = f"Rolling2yr_{var}_mean"
            r3_col = f"Rolling3yr_{var}_mean"
            tr_te_combined[r2_col] = tr_te_combined.groupby("GEOID")[var].transform(lambda g: g.rolling(2, min_periods=1).mean())
            tr_te_combined[r3_col] = tr_te_combined.groupby("GEOID")[var].transform(lambda g: g.rolling(3, min_periods=1).mean())
            te[r2_col] = tr_te_combined.loc[te.index, r2_col]
            te[r3_col] = tr_te_combined.loc[te.index, r3_col]

    elif split_type == "random_grouped":
        for df_part in [tr, va, te]:
            df_sorted = df_part.sort_values(["GEOID", "Year"])
            for var in climate_vars:
                r2_col = f"Rolling2yr_{var}_mean"
                r3_col = f"Rolling3yr_{var}_mean"
                df_sorted[r2_col] = df_sorted.groupby("GEOID")[var].transform(lambda g: g.rolling(2, min_periods=1).mean())
                df_sorted[r3_col] = df_sorted.groupby("GEOID")[var].transform(lambda g: g.rolling(3, min_periods=1).mean())
                df_part[r2_col] = df_sorted.loc[df_part.index, r2_col]
                df_part[r3_col] = df_sorted.loc[df_part.index, r3_col]

    elif split_type == "loso":
        tr_va = pd.concat([tr, va], axis=0).sort_values(["GEOID", "Year"])
        for var in climate_vars:
            r2_col = f"Rolling2yr_{var}_mean"
            r3_col = f"Rolling3yr_{var}_mean"
            tr_va[r2_col] = tr_va.groupby("GEOID")[var].transform(lambda g: g.rolling(2, min_periods=1).mean())
            tr_va[r3_col] = tr_va.groupby("GEOID")[var].transform(lambda g: g.rolling(3, min_periods=1).mean())
            tr[r2_col] = tr_va.loc[tr.index, r2_col]
            tr[r3_col] = tr_va.loc[tr.index, r3_col]
            va[r2_col] = tr_va.loc[va.index, r2_col]
            va[r3_col] = tr_va.loc[va.index, r3_col]

        te_sorted = te.sort_values(["GEOID", "Year"])
        for var in climate_vars:
            r2_col = f"Rolling2yr_{var}_mean"
            r3_col = f"Rolling3yr_{var}_mean"
            te_sorted[r2_col] = te_sorted.groupby("GEOID")[var].transform(lambda g: g.rolling(2, min_periods=1).mean())
            te_sorted[r3_col] = te_sorted.groupby("GEOID")[var].transform(lambda g: g.rolling(3, min_periods=1).mean())
            te[r2_col] = te_sorted.loc[te.index, r2_col]
            te[r3_col] = te_sorted.loc[te.index, r3_col]

    else:
        raise ValueError(f"Unknown split_type '{split_type}'. Must be 'temporal', 'random_row', 'random_grouped', or 'loso'.")

    logger.info("Split-aware rolling climate features constructed (split_type='%s', tr=%d, va=%d, te=%d)",
                split_type, len(tr), len(va), len(te))
    return tr, va, te


def _compute_lag_yield_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """[LEGACY / EDA ONLY — NOT USED FOR PAPER 3 MODEL TRAINING]

    Compute county-level lag yield features for standalone exploratory analysis.
    In the authoritative Paper 3 model training pipeline, target lag features are
    constructed strictly post-split via build_split_aware_lags().
    """
    added = []
    target_col = cfg.PRIMARY_TARGET
    if target_col in df.columns and "GEOID" in df.columns and "Year" in df.columns:
        df_sorted = df.sort_values(["GEOID", "Year"]).copy()
        lag1_s = df_sorted.groupby("GEOID")[target_col].shift(1)
        lag2_s = df_sorted.groupby("GEOID")[target_col].shift(2)
        lag1_year = df_sorted.groupby("GEOID")["Year"].shift(1)
        lag2_year = df_sorted.groupby("GEOID")["Year"].shift(2)

        mask_v1 = (lag1_year == df_sorted["Year"] - 1)
        mask_v2 = (lag2_year == df_sorted["Year"] - 2)

        df_sorted["Yield_lag1"] = np.where(mask_v1, lag1_s, np.nan)
        df_sorted["Yield_lag2"] = np.where(mask_v2, lag2_s, np.nan)

        df.loc[df_sorted.index, "Yield_lag1"] = df_sorted["Yield_lag1"]
        df.loc[df_sorted.index, "Yield_lag2"] = df_sorted["Yield_lag2"]
        added.extend(["Yield_lag1", "Yield_lag2"])
        logger.info("[LEGACY / EDA ONLY] Lag yield features added: Yield_lag1, Yield_lag2")

    return df, added


def _compute_rolling_climate_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """[LEGACY / EDA ONLY — NOT USED FOR PAPER 3 MODEL TRAINING]

    Compute preliminary rolling climate features for dataset-level inspection.
    In the authoritative Paper 3 model training pipeline, rolling features
    are built strictly split-aware via build_split_aware_rolling_features().
    """
    added = []
    if "GEOID" in df.columns and "Year" in df.columns:
        climate_vars = ["Precip_growseason_mm", "GDD_Accumulated", "Tmax_Days_Above_35"]
        climate_vars = [c for c in climate_vars if c in df.columns]

        df_sorted = df.sort_values(["GEOID", "Year"]).copy()

        for var in climate_vars:
            r2_col = f"Rolling2yr_{var}_mean"
            r3_col = f"Rolling3yr_{var}_mean"

            df_sorted[r2_col] = df_sorted.groupby("GEOID")[var].transform(lambda g: g.rolling(2, min_periods=1).mean())
            df_sorted[r3_col] = df_sorted.groupby("GEOID")[var].transform(lambda g: g.rolling(3, min_periods=1).mean())

            # Assign with strict index alignment
            df.loc[df_sorted.index, r2_col] = df_sorted[r2_col]
            df.loc[df_sorted.index, r3_col] = df_sorted[r3_col]
            added.extend([r2_col, r3_col])

        logger.info("[LEGACY / EDA ONLY] Rolling climate features added: %s", added)
    return df, added


def _compute_static_environmental_features(
    df: pd.DataFrame,
    train_mask: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, list]:
    """[LEGACY / EDA ONLY — NOT USED IN PAPER 3 MODEL TRAINING]

    Note: The production model pipeline computes all historical climatological normals
    and anomalies strictly via TrainFittedPreprocessor.fit(train_df).
    """
    return df, []


def _compute_climate_interaction_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Compute non-linear climate interaction features."""
    added = []
    spei_col = "SPEI_30_min" if "SPEI_30_min" in df.columns else ("SPI_30_min" if "SPI_30_min" in df.columns else None)
    tmax_col = "ERA5d_Tmax_max_C" if "ERA5d_Tmax_max_C" in df.columns else "Tmax_Days_Above_35"

    if spei_col and tmax_col in df.columns:
        df["Inter_SPEI_Tmax"] = np.round(df[spei_col] * df[tmax_col], 4)
        added.append("Inter_SPEI_Tmax")

    precip_col = "Precip_growseason_mm" if "Precip_growseason_mm" in df.columns else "Precip_annual_mm"
    if "Tmax_Days_Above_35" in df.columns and precip_col in df.columns:
        df["Inter_Heat_Precip"] = np.round(df["Tmax_Days_Above_35"] * df[precip_col], 4)
        added.append("Inter_Heat_Precip")

    if "CDHW_Silking_Severity" in df.columns and "GDD_Accumulated" in df.columns:
        df["Inter_CDHW_GDD"] = np.round(df["CDHW_Silking_Severity"] * (df["GDD_Accumulated"] / 1000.0), 4)
        added.append("Inter_CDHW_GDD")

    logger.info("Climate interaction features added: %s", added)
    return df, added


def _classify_phenological_window(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Classify phenological window based on accumulated GDD."""
    if "GDD_Accumulated" in df.columns:
        gdd = df["GDD_Accumulated"].values
        conds = [
            gdd <= cfg.GDD_VEGETATIVE_END,
            (gdd > cfg.GDD_VEGETATIVE_END) & (gdd <= cfg.GDD_SILKING_END),
            gdd > cfg.GDD_SILKING_END,
        ]
        choices = ["Vegetative", "Silking_R1", "Grain_Fill"]
        df["Phenological_Window"] = np.select(conds, choices, default="Grain_Fill")
    else:
        df["Phenological_Window"] = "Grain_Fill"

    counts = df["Phenological_Window"].value_counts().to_dict()
    return df, counts


def _compute_phenology_aligned_cdhw(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute phenology-aligned CDHW severity scores."""
    df = df.copy()
    cdhw_score = df["CDHW_Severity_Score"].values if "CDHW_Severity_Score" in df.columns else np.zeros(len(df))
    window = df["Phenological_Window"].values if "Phenological_Window" in df.columns else np.full(len(df), "Grain_Fill")

    df["CDHW_Veg_Severity"] = np.where(window == "Vegetative", cdhw_score * cfg.VEG_WEIGHT_PRIMARY, cdhw_score * cfg.VEG_WEIGHT_SECONDARY)
    df["CDHW_Silking_Severity"] = np.where(window == "Silking_R1", cdhw_score * cfg.SILKING_WEIGHT_PRIMARY, cdhw_score * cfg.SILKING_WEIGHT_SECONDARY)
    df["CDHW_GrainFill_Severity"] = np.where(window == "Grain_Fill", cdhw_score * cfg.GRAIN_WEIGHT_PRIMARY, cdhw_score * cfg.GRAIN_WEIGHT_SECONDARY)

    info = {
        "veg_mean": float(df["CDHW_Veg_Severity"].mean()),
        "silking_mean": float(df["CDHW_Silking_Severity"].mean()),
        "grainfill_mean": float(df["CDHW_GrainFill_Severity"].mean()),
    }
    return df, info


def _classify_year_type(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Classify year-type into Normal, Moderate, Extreme based on CDHW count and ENSO."""
    if "CDHW_Event_Count" in df.columns:
        counts = df["CDHW_Event_Count"].values
        conds = [
            counts >= cfg.EXTREME_EVENT_COUNT_THRESHOLD,
            counts >= cfg.MODERATE_EVENT_COUNT_THRESHOLD,
        ]
        choices = ["Extreme", "Moderate"]
        df["Year_Type"] = np.select(conds, choices, default="Normal")
    elif "Year" in df.columns:
        df["Year_Type"] = np.where(df["Year"].isin(cfg.ANOMALOUS_YEARS), "Extreme", "Normal")
    else:
        df["Year_Type"] = "Normal"

    counts = df["Year_Type"].value_counts().to_dict()
    return df, counts


def _validate_cdhw(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate existing CDHW columns."""
    info: Dict[str, Any] = {}
    for col in cfg.CDHW_COLS:
        if col not in df.columns:
            continue
        info[col] = {
            "present": True,
            "mean": round(float(df[col].mean()), 4),
            "max": round(float(df[col].max()), 4),
        }
    return info


def _generate_feature_validation_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate required columns and duplicate column names."""
    required_climate = ["GDD_Accumulated", "Tmax_Days_Above_35"]
    missing_climate = [c for c in required_climate if c not in df.columns]
    duplicated_cols = df.columns[df.columns.duplicated()].tolist()

    return {
        "validation_passed": len(missing_climate) == 0 and len(duplicated_cols) == 0,
        "missing_required_climate_cols": missing_climate,
        "duplicated_columns": duplicated_cols,
        "total_columns": len(df.columns),
    }


def _generate_feature_lineage_report(df: pd.DataFrame, features_added: list) -> Dict[str, Any]:
    """Generate feature lineage mapping."""
    lineage = {}
    for col in df.columns:
        source = "Engineered" if col in features_added else ("Original Metadata" if col in cfg.ID_COLS else "Original Dataset")
        lineage[col] = {
            "feature": col,
            "source": source,
            "scaling": "RobustScaler (fit on train)" if np.issubdtype(df[col].dtype, np.number) and col not in cfg.ID_COLS else "None",
        }
    return {"features": lineage, "total_features": len(lineage)}
