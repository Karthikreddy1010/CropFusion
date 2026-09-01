"""
preprocessor.py - Reusable, train-fitted preprocessor for zero data leakage.

Guarantees:
1. Imputation statistics (state medians, global medians, lag fallbacks) are fitted strictly on train partition.
2. Validation, testing, and held-out LOSO states only use pre-fitted training statistics.
3. Target variables (Corn_Yield_tha, Soy_Yield_tha) are NEVER imputed.
4. Deterministic fallback for unobserved states/counties.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config as cfg
from utils import log_decision

logger = logging.getLogger("paper3")


class TrainFittedPreprocessor:
    """Reusable preprocessing object fitted strictly on training data."""

    def __init__(self, target_col: str = cfg.PRIMARY_TARGET) -> None:
        self.target_col = target_col
        self.state_medians_: Dict[str, Dict[str, float]] = {}
        self.global_medians_: Dict[str, float] = {}
        self.train_target_median_: float = 0.0
        self.hist_normals_: Dict[str, Dict[Any, float]] = {}
        self.train_climate_means_: Dict[str, float] = {}
        self.fitted_train_rows_: int = 0
        self.fitted_: bool = False

    def fit(self, train_df: pd.DataFrame) -> "TrainFittedPreprocessor":
        """Fit all preprocessing parameters strictly on training observations."""
        self.fitted_train_rows_ = len(train_df)

        # 1. State-specific medians for spatial features (e.g. Soil_Sand_Mean)
        for col in ["Soil_Sand_Mean"]:
            if col in train_df.columns and "State" in train_df.columns:
                valid_state_meds = train_df.groupby("State")[col].median().dropna().to_dict()
                self.state_medians_[col] = valid_state_meds

        # 2. Global training medians for all numerical feature columns
        num_cols = train_df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if col not in cfg.TARGET_COLS and col not in cfg.REDUNDANT_TARGET_COLS:
                med = float(train_df[col].median())
                self.global_medians_[col] = 0.0 if np.isnan(med) else med

        # 3. Training target median for lag feature fallback
        if self.target_col in train_df.columns:
            valid_targets = train_df[self.target_col].dropna()
            self.train_target_median_ = float(valid_targets.median()) if len(valid_targets) > 0 else 0.0
        else:
            self.train_target_median_ = 0.0

        # 4. Historical climate normals strictly from training partition
        if "GEOID" in train_df.columns:
            for col in ["Precip_growseason_mm", "GDD_Accumulated"]:
                if col in train_df.columns:
                    self.hist_normals_[col] = train_df.groupby("GEOID")[col].mean().dropna().to_dict()
                    self.train_climate_means_[col] = float(train_df[col].mean()) if len(train_df) > 0 else 0.0

        self.fitted_ = True
        logger.info(
            "TrainFittedPreprocessor fitted on %d training rows. (Target median fallback: %.4f)",
            self.fitted_train_rows_, self.train_target_median_
        )
        return self

    def transform(self, df: pd.DataFrame, split_name: str = "unknown") -> pd.DataFrame:
        """Transform dataframe using strictly pre-fitted training statistics."""
        if not self.fitted_:
            raise RuntimeError("TrainFittedPreprocessor must be fitted on training data before transform().")

        out = df.copy()

        # 1. Structural Storm zero-fill where Has_Storm == 0 (and constant storm cols)
        for col in cfg.STORM_COLS:
            if col in out.columns:
                if col in cfg.CONSTANT_COLS:
                    out[col] = out[col].fillna(0.0)
                else:
                    if "Has_Storm" in out.columns:
                        mask_no_storm = out["Has_Storm"] == 0
                        out.loc[mask_no_storm & out[col].isnull(), col] = 0.0
                    out[col] = out[col].fillna(0.0)

        # 2. Structural Drought zero-fill where Has_Drought == 0
        for col in cfg.DROUGHT_COLS:
            if col in out.columns:
                if "Has_Drought" in out.columns:
                    mask_no_drought = out["Has_Drought"] == 0
                    out.loc[mask_no_drought & out[col].isnull(), col] = 0.0
                out[col] = out[col].fillna(0.0)

        # 3. State-median imputation for Soil_Sand_Mean using pre-fitted state medians
        if "Soil_Sand_Mean" in out.columns:
            state_meds = self.state_medians_.get("Soil_Sand_Mean", {})
            global_med = self.global_medians_.get("Soil_Sand_Mean", 0.0)

            # Map pre-fitted state median
            mapped_state_med = out["State"].map(state_meds) if "State" in out.columns else pd.Series(np.nan, index=out.index)
            out["Soil_Sand_Mean"] = out["Soil_Sand_Mean"].fillna(mapped_state_med).fillna(global_med)

        # 4. Lag yield feature fallback using pre-fitted train target median
        for lag_col in ["Yield_lag1", "Yield_lag2"]:
            if lag_col in out.columns:
                out[lag_col] = out[lag_col].fillna(self.train_target_median_)

        # 5. Historical climate normals and anomalies using pre-fitted training statistics
        if "GEOID" in out.columns:
            for col, normals in self.hist_normals_.items():
                if col in out.columns:
                    normal_col = f"Hist_Normal_{col}"
                    anom_col = f"Anom_{col}"
                    g_mean = self.train_climate_means_.get(col, 0.0)
                    out[normal_col] = out["GEOID"].map(normals).fillna(g_mean)
                    out[anom_col] = np.round(out[col] - out[normal_col], 4)

        # 6. General numerical feature imputation using pre-fitted training medians
        for col, med_val in self.global_medians_.items():
            if col in out.columns and col not in cfg.TARGET_COLS:
                out[col] = out[col].fillna(med_val)

        # 7. Fail-fast validation: Ensure zero unexpected NaNs remain in feature columns
        num_cols = out.select_dtypes(include=[np.number]).columns
        feat_check_cols = [c for c in num_cols if c not in cfg.TARGET_COLS and c not in cfg.REDUNDANT_TARGET_COLS and c not in cfg.ID_COLS]
        for col in feat_check_cols:
            null_mask = out[col].isnull()
            if null_mask.any():
                bad_row = out[null_mask].iloc[0]
                geoid_val = bad_row.get("GEOID", "unknown")
                year_val = bad_row.get("Year", "unknown")
                raise ValueError(
                    f"PREPROCESSING ERROR: Remaining NaN detected in feature column '{col}' after train-fitted imputation! "
                    f"[Split: {split_name}, GEOID: {geoid_val}, Year: {year_val}]"
                )

        # 8. Targets are NEVER imputed - verify target nulls are preserved
        for t_col in cfg.TARGET_COLS:
            if t_col in df.columns:
                orig_nulls = df[t_col].isnull().sum()
                trans_nulls = out[t_col].isnull().sum()
                if orig_nulls != trans_nulls:
                    raise RuntimeError(f"CRITICAL LEAKAGE BUG: Target column {t_col} was modified during imputation!")

        return out

    def fit_transform(self, train_df: pd.DataFrame, split_name: str = "train") -> pd.DataFrame:
        """Fit on train_df and transform train_df."""
        return self.fit(train_df).transform(train_df, split_name=split_name)
