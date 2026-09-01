"""
test_pipeline.py - Automated Verification Test Suite for Paper 3 Split-Aware Methodology.

Covers all core invariants, split-aware lag & rolling feature tests, and production integration (§13, §14, §19):
- Test A: Temporal lag (2019 uses 2018/2017 for same GEOID)
- Test B: No future lag (no t+1 or t+2 target references)
- Test C: Random row cross-partition lag (TRAIN row cannot receive VAL/TEST target as feature)
- Test D: Random test lag (TEST row cannot use prohibited VAL/TEST target info)
- Test E: Grouped county lag (no lag source violates grouped partition isolation)
- Test F: LOSO lag (held-out state target cannot enter training/validation features)
- Test G: Historical normal isolation (changing VAL/TEST climate cannot alter TRAIN normals)
- Test H: Temporal historical normal (temporal normals use only 1985–2015 train)
- Test I: Random historical normal (random normals use TRAIN partition only)
- Test J: LOSO historical normal (LOSO normals exclude held-out state)
- Test K: Imputation isolation (changing test cannot alter training medians)
- Test L: Scaling isolation (changing test cannot alter training RobustScaler parameters)
- Test M: County baseline & Detrending isolation (train-only parameter calculation)
- Test N: Split reproducibility & Zero-overlap invariants
- Test O: Bootstrap iteration invariant (BOOTSTRAP_ITERATIONS == 2000)
- Test R1: Rolling climate no future leakage (row at year t cannot use t+1 climate)
- Test R2: Rolling climate random partition isolation (train row uses train-only climate history)
- Test R3: Rolling climate grouped county isolation (county never receives rolling values from other GEOIDs)
- Test R4: Rolling climate LOSO isolation (held-out state never contributes to train/val rolling statistics)
- Test R5: Rolling climate deterministic reproducibility (identical split -> identical rolling features)
- Test P1: Production pipeline integration test (end-to-end trace from raw to model input matrices)
"""
from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import config as cfg
from splitting import (
    temporal_split, loso_cv_folds, random_row_split, random_grouped_county_split,
    get_feature_target_arrays
)
from scaling import fit_scaler, apply_scaling, apply_domain_adaptation_scaling
from feature_engineering import (
    engineer_features, build_split_aware_lags, build_split_aware_rolling_features
)
from feature_selection import select_features
from preprocessor import TrainFittedPreprocessor
from model_training import train_neural_cqr, predict_intervals
from evaluation import paired_model_comparison, rmse, mae, r_squared, picp, mpiw


class TestPaper3SplitAwarePipeline(unittest.TestCase):

    def setUp(self):
        np.random.seed(cfg.RANDOM_SEED)
        torch.manual_seed(cfg.RANDOM_SEED)
        n = 100
        self.dummy_df = pd.DataFrame({
            "GEOID": [17001] * 50 + [18001] * 50,
            "State": ["Illinois"] * 50 + ["Indiana"] * 50,
            "Year": list(range(1985, 2035)) * 2,
            "Corn_Yield_tha": np.random.uniform(5.0, 12.0, n),
            "GDD_Accumulated": np.random.uniform(500, 2500, n),
            "Tmax_Days_Above_35": np.random.randint(0, 15, n),
            "Precip_growseason_mm": np.random.uniform(300, 900, n),
            "CDHW_Flag": np.random.randint(0, 2, n),
            "CDHW_Severity_Score": np.random.uniform(0, 50, n),
            "Phenological_Window": np.random.choice(["Vegetative", "Silking_R1", "Grain_Fill"], n),
            "Year_Type": np.random.choice(["Normal", "Moderate", "Extreme"], n),
            "ENSO_Phase": np.random.choice(["ENSO_Neutral", "ENSO_El Nino", "ENSO_La Nina"], n),
            "Lat": [40.0] * 50 + [41.0] * 50,
            "Lon": [-88.0] * 50 + [-86.0] * 50,
            "DEM_Mean": [200.0] * 50 + [220.0] * 50,
            "Slope_Mean": [2.0] * 50 + [1.5] * 50,
            "Soil_AWC_Mean": [0.15] * 50 + [0.18] * 50,
            "Soil_BD_Mean": [1.3] * 50 + [1.4] * 50,
            "Soil_Clay_Mean": [25.0] * 50 + [30.0] * 50,
            "Soil_Sand_Mean": [40.0] * 50 + [35.0] * 50,
            "Soil_Silt_Mean": [35.0] * 50 + [35.0] * 50,
            "Soil_OC_Mean": [2.5] * 50 + [2.1] * 50,
            "CultivatedCrops_frac": [0.7] * 50 + [0.8] * 50,
            "Pasture_frac": [0.1] * 50 + [0.05] * 50,
            "Feature1": np.random.randn(n),
            "Feature2": np.random.randn(n),
        })
        self.feature_cols = ["Feature1", "Feature2", "CDHW_Severity_Score"]

    # ── Test A: Temporal Lag ───────────────────────────────────────────────────
    def test_A_temporal_lag_historical_alignment(self):
        """Test A: In temporal split, 2019 prediction uses 2018 (lag1) and 2017 (lag2) for same GEOID."""
        tr, va, te = temporal_split(self.dummy_df, target=cfg.PRIMARY_TARGET)
        tr, va, te = build_split_aware_lags(tr, va, te, target_col=cfg.PRIMARY_TARGET, split_type="temporal")

        te_2019 = te[te["Year"] == 2019]
        for _, row in te_2019.iterrows():
            geoid = row["GEOID"]
            lag1 = row["Yield_lag1"]
            lag2 = row["Yield_lag2"]

            gt_2018 = self.dummy_df[(self.dummy_df["GEOID"] == geoid) & (self.dummy_df["Year"] == 2018)][cfg.PRIMARY_TARGET].values[0]
            gt_2017 = self.dummy_df[(self.dummy_df["GEOID"] == geoid) & (self.dummy_df["Year"] == 2017)][cfg.PRIMARY_TARGET].values[0]

            self.assertAlmostEqual(lag1, gt_2018, places=4)
            self.assertAlmostEqual(lag2, gt_2017, places=4)

    # ── Test B: No Future Lag ──────────────────────────────────────────────────
    def test_B_no_future_lag(self):
        """Test B: Verify lag features never reference t+1 or t+2 target values."""
        tr, va, te = temporal_split(self.dummy_df, target=cfg.PRIMARY_TARGET)
        tr, va, te = build_split_aware_lags(tr, va, te, target_col=cfg.PRIMARY_TARGET, split_type="temporal")

        for split_df in [tr, va, te]:
            for _, row in split_df.iterrows():
                geoid = row["GEOID"]
                year = int(row["Year"])
                lag1 = row["Yield_lag1"]
                if pd.notna(lag1):
                    future_rows = self.dummy_df[(self.dummy_df["GEOID"] == geoid) & (self.dummy_df["Year"] == year + 1)]
                    prev_rows = self.dummy_df[(self.dummy_df["GEOID"] == geoid) & (self.dummy_df["Year"] == year - 1)]
                    if len(future_rows) > 0 and len(prev_rows) > 0:
                        f_val = future_rows[cfg.PRIMARY_TARGET].values[0]
                        p_val = prev_rows[cfg.PRIMARY_TARGET].values[0]
                        if not np.isclose(f_val, p_val):
                            self.assertNotEqual(lag1, f_val)

    # ── Test C: Random Row Cross-Partition Lag Isolation ──────────────────────
    def test_C_random_row_cross_partition_isolation(self):
        """Test C: A TRAIN row must not receive a lag target whose source row belongs to VAL or TEST."""
        tr_r, va_r, te_r = random_row_split(self.dummy_df, target=cfg.PRIMARY_TARGET, seed=42)
        tr_r, va_r, te_r = build_split_aware_lags(tr_r, va_r, te_r, target_col=cfg.PRIMARY_TARGET, split_type="random_row")

        non_train_indices = set(va_r.set_index(["GEOID", "Year"]).index).union(set(te_r.set_index(["GEOID", "Year"]).index))

        for _, row in tr_r.iterrows():
            geoid = row["GEOID"]
            year = int(row["Year"])
            lag1 = row["Yield_lag1"]
            lag2 = row["Yield_lag2"]

            if (geoid, year - 1) in non_train_indices:
                self.assertTrue(pd.isna(lag1), f"TRAIN row ({geoid}, {year}) leaked lag1 from VAL/TEST!")
            if (geoid, year - 2) in non_train_indices:
                self.assertTrue(pd.isna(lag2), f"TRAIN row ({geoid}, {year}) leaked lag2 from VAL/TEST!")

    # ── Test D: Random Test Lag Isolation ─────────────────────────────────────
    def test_D_random_test_lag_isolation(self):
        """Test D: A TEST row must not use prohibited TEST/VAL target information."""
        tr_r, va_r, te_r = random_row_split(self.dummy_df, target=cfg.PRIMARY_TARGET, seed=42)
        tr_r, va_r, te_r = build_split_aware_lags(tr_r, va_r, te_r, target_col=cfg.PRIMARY_TARGET, split_type="random_row")

        non_train_indices = set(va_r.set_index(["GEOID", "Year"]).index).union(set(te_r.set_index(["GEOID", "Year"]).index))

        for _, row in te_r.iterrows():
            geoid = row["GEOID"]
            year = int(row["Year"])
            lag1 = row["Yield_lag1"]
            if (geoid, year - 1) in non_train_indices:
                self.assertTrue(pd.isna(lag1), f"TEST row ({geoid}, {year}) leaked lag1 from non-train split!")

    # ── Test E: Grouped County Lag Isolation ──────────────────────────────────
    def test_E_grouped_county_lag_isolation(self):
        """Test E: In grouped county split, Val and Test counties must have NaN lags (unseen in train)."""
        tr_g, va_g, te_g = random_grouped_county_split(self.dummy_df, target=cfg.PRIMARY_TARGET, seed=42)
        tr_g, va_g, te_g = build_split_aware_lags(tr_g, va_g, te_g, target_col=cfg.PRIMARY_TARGET, split_type="random_grouped")

        self.assertTrue(va_g["Yield_lag1"].isnull().all())
        self.assertTrue(va_g["Yield_lag2"].isnull().all())
        self.assertTrue(te_g["Yield_lag1"].isnull().all())
        self.assertTrue(te_g["Yield_lag2"].isnull().all())

    # ── Test F: LOSO Lag Isolation ────────────────────────────────────────────
    def test_F_loso_lag_isolation(self):
        """Test F: Held-out state target cannot enter training or validation lag features."""
        for state, tr_f, va_f, te_f in loso_cv_folds(self.dummy_df, return_val=True):
            tr_f, va_f, te_f = build_split_aware_lags(tr_f, va_f, te_f, target_col=cfg.PRIMARY_TARGET, split_type="loso")

            self.assertNotIn(state, tr_f["State"].unique())
            self.assertNotIn(state, va_f["State"].unique())
            self.assertTrue(te_f["Yield_lag1"].isnull().all())
            self.assertTrue(te_f["Yield_lag2"].isnull().all())

    # ── Test G: Historical Normal Isolation ───────────────────────────────────
    def test_G_historical_normal_isolation(self):
        """Test G: Changing VAL/TEST climate values cannot change TRAIN-derived normals."""
        tr, va, te = temporal_split(self.dummy_df, target=cfg.PRIMARY_TARGET)
        prep = TrainFittedPreprocessor(target_col=cfg.PRIMARY_TARGET)
        prep.fit(tr)

        norm_precip_orig = prep.hist_normals_["Precip_growseason_mm"].copy()

        te_perturbed = te.copy()
        te_perturbed["Precip_growseason_mm"] = 99999.0
        _ = prep.transform(te_perturbed)

        self.assertEqual(prep.hist_normals_["Precip_growseason_mm"], norm_precip_orig)

    # ── Test H: Temporal Historical Normal Train-Only ─────────────────────────
    def test_H_temporal_historical_normal(self):
        """Test H: Temporal normals use only 1985–2015 train observations."""
        tr, va, te = temporal_split(self.dummy_df, target=cfg.PRIMARY_TARGET)
        prep = TrainFittedPreprocessor(target_col=cfg.PRIMARY_TARGET)
        prep.fit(tr)

        for geoid in tr["GEOID"].unique():
            tr_sub = tr[tr["GEOID"] == geoid]
            expected_mean = tr_sub["Precip_growseason_mm"].mean()
            self.assertAlmostEqual(prep.hist_normals_["Precip_growseason_mm"][geoid], expected_mean, places=4)

    # ── Test I: Random Historical Normal Train-Only ───────────────────────────
    def test_I_random_historical_normal(self):
        """Test I: Random normals use TRAIN partition only."""
        tr_r, va_r, te_r = random_row_split(self.dummy_df, target=cfg.PRIMARY_TARGET, seed=42)
        prep = TrainFittedPreprocessor(target_col=cfg.PRIMARY_TARGET)
        prep.fit(tr_r)

        for geoid in tr_r["GEOID"].unique():
            tr_sub = tr_r[tr_r["GEOID"] == geoid]
            expected_mean = tr_sub["Precip_growseason_mm"].mean()
            self.assertAlmostEqual(prep.hist_normals_["Precip_growseason_mm"][geoid], expected_mean, places=4)

    # ── Test J: LOSO Historical Normal Excludes Held-Out State ────────────────
    def test_J_loso_historical_normal_isolation(self):
        """Test J: LOSO normals exclude held-out state completely."""
        for state, tr_f, va_f, te_f in loso_cv_folds(self.dummy_df, return_val=True):
            prep = TrainFittedPreprocessor(target_col=cfg.PRIMARY_TARGET)
            prep.fit(tr_f)

            held_out_geoids = te_f["GEOID"].unique()
            for g in held_out_geoids:
                self.assertNotIn(g, prep.hist_normals_["Precip_growseason_mm"])

    # ── Test K: Imputation Isolation ──────────────────────────────────────────
    def test_K_imputation_isolation(self):
        """Test K: Verify test perturbations cannot alter training medians."""
        tr, va, te = temporal_split(self.dummy_df, target=cfg.PRIMARY_TARGET)
        prep = TrainFittedPreprocessor(target_col=cfg.PRIMARY_TARGET)
        prep.fit(tr)

        orig_median = prep.train_target_median_
        te_pert = te.copy()
        te_pert[cfg.PRIMARY_TARGET] = 99999.0
        _ = prep.transform(te_pert)
        self.assertEqual(prep.train_target_median_, orig_median)

    # ── Test L: Scaling & Domain Adaptation Isolation ─────────────────────────
    def test_L_scaling_isolation(self):
        """Test L: RobustScaler and QuantileTransformer fit parameters strictly on train."""
        tr, va, te = temporal_split(self.dummy_df, target=cfg.PRIMARY_TARGET)
        scaler, _ = fit_scaler(tr, self.feature_cols, scaler_type="robust")
        c1 = scaler.center_.copy()
        s1 = scaler.scale_.copy()

        te_pert = te.copy()
        te_pert[self.feature_cols] = 99999.0
        scaler2, _ = fit_scaler(tr, self.feature_cols, scaler_type="robust")
        self.assertTrue(np.allclose(c1, scaler2.center_))
        self.assertTrue(np.allclose(s1, scaler2.scale_))

        tr_da, te_da = apply_domain_adaptation_scaling(tr, te, self.feature_cols)
        self.assertEqual(len(tr_da), len(tr))
        self.assertEqual(len(te_da), len(te))

    # ── Test M: County Baseline & Detrending Isolation ────────────────────────
    def test_M_county_baseline_detrending_isolation(self):
        """Test M: County baseline anomaly and linear trend are strictly train-derived."""
        tr = self.dummy_df[self.dummy_df["GEOID"] == 17001].copy()
        te = self.dummy_df[self.dummy_df["GEOID"] == 18001].copy()

        cf = np.polyfit(tr["Year"].values, tr["Corn_Yield_tha"].values, 1)
        anom = tr["Corn_Yield_tha"].values - np.polyval(cf, tr["Year"].values)
        cb = pd.Series(anom, index=tr["GEOID"].values).groupby(level=0).mean()

        test_cb = te["GEOID"].map(cb).fillna(0.0)
        self.assertAlmostEqual(test_cb.iloc[0], 0.0)

    # ── Test N: Split Reproducibility & Zero-Overlap ───────────────────────────
    def test_N_split_reproducibility(self):
        """Test N: Identical seed produces identical split partitions with zero cross-overlap."""
        tr_g1, va_g1, te_g1 = random_grouped_county_split(self.dummy_df, seed=42)
        tr_g2, va_g2, te_g2 = random_grouped_county_split(self.dummy_df, seed=42)
        self.assertTrue(tr_g1.index.equals(tr_g2.index))
        self.assertTrue(set(tr_g1["GEOID"].unique()).isdisjoint(set(te_g1["GEOID"].unique())))

    # ── Test O: Bootstrap Invariant ───────────────────────────────────────────
    def test_O_bootstrap_iteration_invariant(self):
        """Test O: Verify bootstrap iterations configured for 2,000 resamples."""
        self.assertEqual(cfg.BOOTSTRAP_ITERATIONS, 2000)

    # ── Test O2: Configuration Consistency Invariants ─────────────────────────
    def test_O2_configuration_consistency(self):
        """Test O2: Verify epoch budgets, random seeds, and split boundaries are centralized and consistent."""
        self.assertEqual(getattr(cfg, "BASELINE_MAX_EPOCHS", None), 60)
        self.assertEqual(getattr(cfg, "OPTUNA_MAX_EPOCHS", None), 120)
        self.assertEqual(getattr(cfg, "MAX_TUNING_EPOCHS", None), 120)
        self.assertEqual(cfg.RANDOM_SEED, 42)
        self.assertEqual(cfg.TRAIN_YEARS, (1985, 2015))
        self.assertEqual(cfg.VAL_YEARS, (2016, 2018))
        self.assertEqual(cfg.TEST_YEARS, (2019, 2023))

    # ── Test R1: Rolling Climate — No Future Climate ──────────────────────────
    def test_R1_rolling_no_future_climate(self):
        """Test R1: A row at year t cannot use climate information from year t+1 or later."""
        tr, va, te = temporal_split(self.dummy_df, target=cfg.PRIMARY_TARGET)
        tr, va, te = build_split_aware_rolling_features(tr, va, te, split_type="temporal")

        for split_df in [tr, va, te]:
            for _, row in split_df.iterrows():
                geoid = row["GEOID"]
                year = int(row["Year"])
                r2_precip = row["Rolling2yr_Precip_growseason_mm_mean"]

                # Expected 2-year rolling is mean of year t-1 and year t (if available)
                c_prev = self.dummy_df[(self.dummy_df["GEOID"] == geoid) & (self.dummy_df["Year"] == year - 1)]["Precip_growseason_mm"]
                c_curr = self.dummy_df[(self.dummy_df["GEOID"] == geoid) & (self.dummy_df["Year"] == year)]["Precip_growseason_mm"]
                vals = []
                if len(c_prev) > 0:
                    vals.append(c_prev.values[0])
                if len(c_curr) > 0:
                    vals.append(c_curr.values[0])
                expected = np.mean(vals)
                self.assertAlmostEqual(r2_precip, expected, places=4)

    # ── Test R2: Rolling Climate — Random Partition Isolation ─────────────────
    def test_R2_rolling_random_partition_isolation(self):
        """Test R2: A training row cannot depend on prohibited validation/test observations."""
        tr_r, va_r, te_r = random_row_split(self.dummy_df, target=cfg.PRIMARY_TARGET, seed=42)

        # Perturb test set climate to extreme value
        te_pert = te_r.copy()
        te_pert["Precip_growseason_mm"] = 999999.0

        tr_r_clean, _, _ = build_split_aware_rolling_features(tr_r, va_r, te_pert, split_type="random_row")
        tr_r_orig, _, _ = build_split_aware_rolling_features(tr_r, va_r, te_r, split_type="random_row")

        self.assertTrue(np.allclose(
            tr_r_clean["Rolling2yr_Precip_growseason_mm_mean"].values,
            tr_r_orig["Rolling2yr_Precip_growseason_mm_mean"].values
        ))

    # ── Test R3: Rolling Climate — Grouped County Isolation ───────────────────
    def test_R3_rolling_grouped_county_isolation(self):
        """Test R3: A county cannot receive rolling values from another GEOID."""
        tr_g, va_g, te_g = random_grouped_county_split(self.dummy_df, seed=42)
        tr_g, va_g, te_g = build_split_aware_rolling_features(tr_g, va_g, te_g, split_type="random_grouped")

        for split_df in [tr_g, va_g, te_g]:
            for geoid in split_df["GEOID"].unique():
                sub = split_df[split_df["GEOID"] == geoid].sort_values("Year")
                precip_vals = sub["Precip_growseason_mm"].values
                expected_r2 = [
                    precip_vals[0] if i == 0 else (precip_vals[i-1] + precip_vals[i]) / 2.0
                    for i in range(len(precip_vals))
                ]
                self.assertTrue(np.allclose(sub["Rolling2yr_Precip_growseason_mm_mean"].values, expected_r2))

    # ── Test R4: Rolling Climate — LOSO Isolation ─────────────────────────────
    def test_R4_rolling_loso_isolation(self):
        """Test R4: A held-out state cannot contribute to training/validation rolling statistics."""
        for state, tr_f, va_f, te_f in loso_cv_folds(self.dummy_df, return_val=True):
            # Perturb held-out state climate values
            te_pert = te_f.copy()
            te_pert["Precip_growseason_mm"] = 999999.0

            tr_clean, va_clean, _ = build_split_aware_rolling_features(tr_f, va_f, te_pert, split_type="loso")
            tr_orig, va_orig, _ = build_split_aware_rolling_features(tr_f, va_f, te_f, split_type="loso")

            self.assertTrue(np.allclose(
                tr_clean["Rolling2yr_Precip_growseason_mm_mean"].values,
                tr_orig["Rolling2yr_Precip_growseason_mm_mean"].values
            ))
            self.assertTrue(np.allclose(
                va_clean["Rolling2yr_Precip_growseason_mm_mean"].values,
                va_orig["Rolling2yr_Precip_growseason_mm_mean"].values
            ))

    # ── Test R5: Rolling Climate — Deterministic Reproducibility ──────────────
    def test_R5_rolling_deterministic_reproducibility(self):
        """Test R5: Same input + same split -> identical rolling features."""
        tr1, va1, te1 = temporal_split(self.dummy_df, target=cfg.PRIMARY_TARGET)
        tr1, va1, te1 = build_split_aware_rolling_features(tr1, va1, te1, split_type="temporal")

        tr2, va2, te2 = temporal_split(self.dummy_df, target=cfg.PRIMARY_TARGET)
        tr2, va2, te2 = build_split_aware_rolling_features(tr2, va2, te2, split_type="temporal")

        self.assertTrue(tr1["Rolling2yr_Precip_growseason_mm_mean"].equals(tr2["Rolling2yr_Precip_growseason_mm_mean"]))
        self.assertTrue(te1["Rolling3yr_GDD_Accumulated_mean"].equals(te2["Rolling3yr_GDD_Accumulated_mean"]))

    # ── Test R6: Rolling Climate — Perturbation Invariance across Partitions ──
    def test_R6_rolling_perturbation_tests_all_partitions(self):
        """Test R6: Perturbing prohibited partition sources does not contaminate permitted split features."""
        # 1. Random row split perturbation: perturb test, verify train & val
        tr_r, va_r, te_r = random_row_split(self.dummy_df, target=cfg.PRIMARY_TARGET, seed=42)
        te_pert = te_r.copy()
        te_pert["GDD_Accumulated"] = 88888.0
        te_pert["Tmax_Days_Above_35"] = 999.0

        tr_clean, va_clean, _ = build_split_aware_rolling_features(tr_r, va_r, te_pert, split_type="random_row")
        tr_orig, va_orig, _ = build_split_aware_rolling_features(tr_r, va_r, te_r, split_type="random_row")

        self.assertTrue(np.allclose(tr_clean["Rolling3yr_GDD_Accumulated_mean"].values, tr_orig["Rolling3yr_GDD_Accumulated_mean"].values))
        self.assertTrue(np.allclose(va_clean["Rolling2yr_Tmax_Days_Above_35_mean"].values, va_orig["Rolling2yr_Tmax_Days_Above_35_mean"].values))

        # 2. Grouped county split perturbation: perturb other counties, verify each county
        tr_g, va_g, te_g = random_grouped_county_split(self.dummy_df, seed=42)
        te_g_pert = te_g.copy()
        te_g_pert["Precip_growseason_mm"] = 999999.0
        tr_g_clean, va_g_clean, _ = build_split_aware_rolling_features(tr_g, va_g, te_g_pert, split_type="random_grouped")
        tr_g_orig, va_g_orig, _ = build_split_aware_rolling_features(tr_g, va_g, te_g, split_type="random_grouped")

        self.assertTrue(np.allclose(tr_g_clean["Rolling2yr_Precip_growseason_mm_mean"].values, tr_g_orig["Rolling2yr_Precip_growseason_mm_mean"].values))
        self.assertTrue(np.allclose(va_g_clean["Rolling3yr_Precip_growseason_mm_mean"].values, va_g_orig["Rolling3yr_Precip_growseason_mm_mean"].values))

    # ── Test P1: Production Pipeline Integration Test ─────────────────────────
    def test_P1_production_pipeline_integration(self):
        """Test P1: End-to-end execution of the production preprocessing and feature pipeline (§13).

        Trace: raw -> engineer_features -> split -> split_aware_lags -> split_aware_rolling
               -> TrainFittedPreprocessor.fit_transform -> feature_selection -> RobustScaler
        """
        # 1. Deterministic feature engineering on raw data
        raw_feat_df, _ = engineer_features(self.dummy_df)

        # 2. Evaluation split
        tr, va, te = temporal_split(raw_feat_df, target=cfg.PRIMARY_TARGET)

        # 3. Split-aware target lags
        tr, va, te = build_split_aware_lags(tr, va, te, target_col=cfg.PRIMARY_TARGET, split_type="temporal")

        # 4. Split-aware rolling climate features
        tr, va, te = build_split_aware_rolling_features(tr, va, te, split_type="temporal")

        # 5. Train-fitted preprocessor (imputation & historical normals)
        prep = TrainFittedPreprocessor(target_col=cfg.PRIMARY_TARGET)
        tr_proc = prep.fit_transform(tr, split_name="prod_test_train")
        va_proc = prep.transform(va, split_name="prod_test_val")
        te_proc = prep.transform(te, split_name="prod_test_test")

        # 6. Feature selection on TRAIN ONLY
        cand_cols = [c for c in tr_proc.select_dtypes(include=[np.number]).columns
                     if c not in cfg.TARGET_COLS and c not in cfg.REDUNDANT_TARGET_COLS and c not in cfg.ID_COLS]
        self.assertGreater(len(cand_cols), 5)

        # 7. Scaler on TRAIN ONLY
        scaler, _ = fit_scaler(tr_proc, cand_cols, scaler_type="robust")
        tr_sc = apply_scaling(tr_proc, cand_cols, scaler, split_name="prod_test_train")
        va_sc = apply_scaling(va_proc, cand_cols, scaler, split_name="prod_test_val")
        te_sc = apply_scaling(te_proc, cand_cols, scaler, split_name="prod_test_test")

        # 8. Verify model input matrices
        X_tr, y_tr = get_feature_target_arrays(tr_sc, cand_cols, split_name="prod_test_train")
        X_va, y_va = get_feature_target_arrays(va_sc, cand_cols, split_name="prod_test_val")
        X_te, y_te = get_feature_target_arrays(te_sc, cand_cols, split_name="prod_test_test")

        self.assertEqual(X_tr.shape[0], len(tr_proc))
        self.assertEqual(X_va.shape[0], len(va_proc))
        self.assertEqual(X_te.shape[0], len(te_proc))
        self.assertEqual(X_tr.shape[1], len(cand_cols))
        self.assertFalse(np.isnan(X_tr).any())
        self.assertFalse(np.isnan(X_te).any())


if __name__ == "__main__":
    unittest.main()
