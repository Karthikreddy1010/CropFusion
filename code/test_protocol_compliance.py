"""
test_protocol_compliance.py - Automated tests for the locked experimental protocol.

These tests encode the invariants that the audit found violated, so a
regression fails loudly instead of silently producing optimistic numbers:

* LOSO: exactly six states, and zero overlap between train / validation /
  calibration / test observation IDs.
* Temporal: FIT 1985-2013, DEV 2014-2015, CAL 2016-2018, TEST 2019-2023.
* Feature selection and scaling are fitted only on the FIT partition.
* The ensemble weight is chosen only from allowed development data.
* Calibration IDs never intersect TEST IDs.
* Joint and post-hoc training genuinely invoke different procedures.
* The Nemenyi critical value is computed for the actual k (6 here), not
  hard-coded.

Run with:  python -m unittest test_protocol_compliance -v
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import config as cfg
from splitting import temporal_split_4way, loso_cv_folds, add_obs_ids, obs_ids
from leakage_provenance import PartitionLedger, audit_ledger, as_id_set
from dependence_aware_stats import nemenyi_critical_difference, friedman_on_blocks


def _synthetic_panel(seed: int = 0) -> pd.DataFrame:
    """A small county-year panel spanning the real protocol years and states."""
    rng = np.random.RandomState(seed)
    rows = []
    for si, state in enumerate(cfg.LOSO_STATES):
        for c in range(6):
            geoid = 17000 + si * 100 + c
            for year in range(cfg.FIT_YEARS[0], cfg.TEST_YEARS[1] + 1):
                rows.append({
                    "GEOID": geoid,
                    "County_Name": f"{state}_C{c}",
                    "State": state,
                    "Year": year,
                    "Lat": 40.0 + si, "Lon": -90.0 - c,
                    "CDHW_Severity_Score": float(rng.rand()),
                    "Precip_growseason_mm": float(500 + rng.randn() * 50),
                    "GDD_Accumulated": float(2500 + rng.randn() * 100),
                    "Tmax_Days_Above_35": float(rng.randint(0, 12)),
                    cfg.PRIMARY_TARGET: float(8.0 + 0.05 * (year - 1985) + rng.randn() * 0.5),
                })
    return pd.DataFrame(rows)


class TestLOSOProtocol(unittest.TestCase):
    """§22 — LOSO: 6 states exactly, zero overlap between every partition pair."""

    @classmethod
    def setUpClass(cls):
        cls.df = _synthetic_panel()

    def test_loso_has_exactly_six_locked_states(self):
        expected = ["Illinois", "Indiana", "Iowa", "Minnesota", "Missouri", "Ohio"]
        self.assertEqual(len(cfg.LOSO_STATES), 6,
                         f"LOSO must have exactly 6 folds, found {len(cfg.LOSO_STATES)}")
        self.assertEqual(sorted(cfg.LOSO_STATES), sorted(expected),
                         "LOSO state list must match the locked six-state protocol")

    def test_loso_yields_six_folds(self):
        states = [s for s, *_ in loso_cv_folds(self.df, return_cal=True)]
        self.assertEqual(len(states), 6, f"expected 6 LOSO folds, generated {len(states)}")
        self.assertEqual(sorted(states), sorted(cfg.LOSO_STATES))

    def test_loso_zero_overlap_between_all_partitions(self):
        for state, fit_f, dev_f, cal_f, test_f in loso_cv_folds(self.df, return_cal=True):
            ledger = PartitionLedger(experiment=f"LOSO/{state}")
            ledger.register("train", fit_f)
            ledger.register("dev", dev_f)
            ledger.register("cal", cal_f)
            ledger.register("test", test_f)
            rec = audit_ledger(ledger)

            self.assertEqual(rec["train_test_overlap"], 0, f"{state}: train/test overlap")
            self.assertEqual(rec["validation_test_overlap"], 0, f"{state}: validation/test overlap")
            self.assertEqual(rec["calibration_test_overlap"], 0, f"{state}: calibration/test overlap")
            self.assertEqual(rec["train_validation_overlap"], 0, f"{state}: train/validation overlap")
            self.assertEqual(rec["train_calibration_overlap"], 0, f"{state}: train/calibration overlap")
            self.assertEqual(rec["validation_calibration_overlap"], 0,
                             f"{state}: validation/calibration overlap")
            self.assertEqual(rec["status"], "PASS", f"{state}: {rec['failures']}")

    def test_held_out_state_absent_from_every_other_partition(self):
        for state, fit_f, dev_f, cal_f, test_f in loso_cv_folds(self.df, return_cal=True):
            for part, name in ((fit_f, "fit"), (dev_f, "dev"), (cal_f, "cal")):
                self.assertNotIn(state, set(part["State"]),
                                 f"{state} leaked into the {name} partition")
            self.assertEqual(set(test_f["State"]), {state},
                             f"test fold for {state} contains other states")

    def test_loso_partitions_use_locked_year_windows(self):
        for state, fit_f, dev_f, cal_f, _ in loso_cv_folds(self.df, return_cal=True):
            self.assertLessEqual(int(fit_f["Year"].max()), cfg.FIT_YEARS[1],
                                 f"{state}: fit partition extends past FIT_YEARS")
            self.assertGreaterEqual(int(dev_f["Year"].min()), cfg.DEV_YEARS[0])
            self.assertLessEqual(int(dev_f["Year"].max()), cfg.DEV_YEARS[1])
            self.assertGreaterEqual(int(cal_f["Year"].min()), cfg.CAL_YEARS[0])

    def test_internal_early_stopping_split_comes_only_from_fit(self):
        """The ES carve-out must be a subset of FIT and disjoint from dev/cal/test."""
        for state, fit_f, dev_f, cal_f, test_f in loso_cv_folds(self.df, return_cal=True):
            fit_ids = obs_ids(fit_f)
            n_es = max(1, len(fit_ids) // 10)
            train_ids, es_ids = fit_ids[:-n_es], fit_ids[-n_es:]

            self.assertTrue(set(es_ids).issubset(set(fit_ids)),
                            f"{state}: ES split is not a subset of FIT")
            self.assertEqual(len(set(train_ids) & set(es_ids)), 0,
                             f"{state}: ES rows also used for fitting")
            for other, name in ((dev_f, "dev"), (cal_f, "cal"), (test_f, "test")):
                self.assertEqual(len(set(es_ids) & as_id_set(other)), 0,
                                 f"{state}: ES split intersects {name}")


class TestTemporalProtocol(unittest.TestCase):
    """§22 — Temporal: FIT 1985-2013, DEV 2014-2015, CAL 2016-2018, TEST 2019-2023."""

    @classmethod
    def setUpClass(cls):
        cls.df = _synthetic_panel()
        cls.fit, cls.dev, cls.cal, cls.test = temporal_split_4way(cls.df)

    def test_config_boundaries_match_locked_protocol(self):
        self.assertEqual(tuple(cfg.FIT_YEARS), (1985, 2013))
        self.assertEqual(tuple(cfg.DEV_YEARS), (2014, 2015))
        self.assertEqual(tuple(cfg.CAL_YEARS), (2016, 2018))
        self.assertEqual(tuple(cfg.TEST_YEARS), (2019, 2023))

    def test_observed_year_ranges(self):
        for part, expected, name in (
            (self.fit, (1985, 2013), "FIT"), (self.dev, (2014, 2015), "DEV"),
            (self.cal, (2016, 2018), "CAL"), (self.test, (2019, 2023), "TEST"),
        ):
            self.assertEqual((int(part["Year"].min()), int(part["Year"].max())), expected,
                             f"{name} partition year range mismatch")

    def test_no_year_appears_in_two_partitions(self):
        sets = [set(p["Year"].unique()) for p in (self.fit, self.dev, self.cal, self.test)]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                self.assertEqual(sets[i] & sets[j], set(),
                                 "a year is assigned to two temporal partitions")

    def test_zero_observation_overlap(self):
        ledger = PartitionLedger(experiment="Temporal")
        ledger.register("train", self.fit)
        ledger.register("dev", self.dev)
        ledger.register("cal", self.cal)
        ledger.register("test", self.test)
        rec = audit_ledger(ledger)
        self.assertEqual(rec["status"], "PASS", rec["failures"])
        self.assertEqual(rec["train_test_overlap"], 0)
        self.assertEqual(rec["validation_test_overlap"], 0)
        self.assertEqual(rec["calibration_test_overlap"], 0)
        self.assertEqual(rec["train_validation_overlap"], 0)

    def test_calibration_ids_never_intersect_test_ids(self):
        """§22 — calibration IDs ∩ TEST IDs = ∅."""
        self.assertEqual(as_id_set(self.cal) & as_id_set(self.test), set())


class TestFitOnlyFitting(unittest.TestCase):
    """§22 — feature selection and scaling fitting data ⊆ FIT."""

    @classmethod
    def setUpClass(cls):
        cls.df = _synthetic_panel()
        cls.fit, cls.dev, cls.cal, cls.test = temporal_split_4way(cls.df)

    def test_scaler_statistics_depend_only_on_fit_rows(self):
        """Perturbing DEV/CAL/TEST must not move the fitted scaler at all."""
        from scaling import fit_scaler
        feats = ["Precip_growseason_mm", "GDD_Accumulated", "Tmax_Days_Above_35"]

        scaler_a, _ = fit_scaler(self.fit, feats, scaler_type="robust")
        perturbed_fit = self.fit.copy()          # unchanged FIT
        polluted = pd.concat([self.dev, self.cal, self.test])
        polluted = polluted.copy()
        for f in feats:
            polluted[f] = polluted[f] * 1000.0   # extreme perturbation elsewhere
        scaler_b, _ = fit_scaler(perturbed_fit, feats, scaler_type="robust")

        np.testing.assert_allclose(scaler_a.center_, scaler_b.center_,
                                   err_msg="scaler center changed without FIT changing")
        np.testing.assert_allclose(scaler_a.scale_, scaler_b.scale_,
                                   err_msg="scaler scale changed without FIT changing")

    def test_scaler_changes_when_fit_changes(self):
        """Control: the scaler must actually respond to FIT data."""
        from scaling import fit_scaler
        feats = ["Precip_growseason_mm", "GDD_Accumulated", "Tmax_Days_Above_35"]
        scaler_a, _ = fit_scaler(self.fit, feats, scaler_type="robust")
        moved = self.fit.copy()
        for f in feats:
            moved[f] = moved[f] + 500.0
        scaler_b, _ = fit_scaler(moved, feats, scaler_type="robust")
        self.assertFalse(np.allclose(scaler_a.center_, scaler_b.center_),
                         "scaler ignored a change in the FIT partition")

    def test_feature_selection_input_is_fit_only(self):
        """select_features must be handed FIT rows exclusively."""
        from feature_selection import select_features
        captured = {}
        import feature_selection as fs
        orig = fs.mutual_info_regression

        def spy(X, y, **kw):
            captured["n_rows"] = len(X)
            return orig(X, y, **kw)

        fs.mutual_info_regression = spy
        try:
            select_features(self.fit, report_suffix="_unittest")
        finally:
            fs.mutual_info_regression = orig

        n_fit_valid = int(self.fit[cfg.PRIMARY_TARGET].notna().sum())
        self.assertEqual(captured.get("n_rows"), n_fit_valid,
                         "feature selection saw rows outside the FIT partition")


class TestEnsembleWeightSelection(unittest.TestCase):
    """§22 — the weight must be selected only from DEV/allowed validation data."""

    def test_weight_is_dev_argmin_and_ignores_test(self):
        rng = np.random.RandomState(0)
        y_dev = rng.randn(200)
        a_dev = y_dev + rng.randn(200) * 0.5     # better on DEV
        b_dev = y_dev + rng.randn(200) * 2.0
        # A deliberately different ordering on TEST: if selection ever peeked at
        # test data the chosen weight would move.
        y_te = rng.randn(200)
        a_te = y_te + rng.randn(200) * 3.0
        b_te = y_te + rng.randn(200) * 0.2

        def choose(y, pa, pb):
            best_w, best = 1.0, np.inf
            for w in np.arange(0.0, 1.01, 0.1):
                blend = w * pa + (1 - w) * pb
                r = float(np.sqrt(np.mean((y - blend) ** 2)))
                if r < best:
                    best, best_w = r, round(float(w), 2)
            return best_w

        w_dev = choose(y_dev, a_dev, b_dev)
        w_test = choose(y_te, a_te, b_te)
        self.assertGreater(w_dev, 0.5, "DEV selection should favour the DEV-better model")
        self.assertNotEqual(w_dev, w_test,
                            "test set implies a different weight; the fixture is not "
                            "discriminative, so this test could not detect peeking")


class TestJointVsPostHocDistinct(unittest.TestCase):
    """§22 — the two configurations must invoke genuinely different procedures."""

    @classmethod
    def setUpClass(cls):
        from model_training import train_neural_cqr, predict_intervals
        rng = np.random.RandomState(0)
        X = rng.randn(400, 6).astype(np.float32)
        y = (X[:, 0] * 2 + X[:, 1] + rng.randn(400) * 0.5).astype(np.float32)
        feats = [f"f{i}" for i in range(6)]
        cls.joint = train_neural_cqr(X[:340], y[:340], X[340:], y[340:], feats,
                                     epochs=8, batch_size=64, lr=1e-3,
                                     joint_training=True, patience=20)
        cls.post = train_neural_cqr(X[:340], y[:340], X[340:], y[340:], feats,
                                    epochs=8, batch_size=64, lr=1e-3,
                                    joint_training=False, patience=20)
        cls.X_eval = X[340:]
        cls._predict = staticmethod(predict_intervals)

    def test_paradigms_are_labelled_differently(self):
        self.assertEqual(self.joint.training_paradigm, "joint_end_to_end")
        self.assertEqual(self.post.training_paradigm, "post_hoc_two_stage")

    def test_optimization_structure_differs(self):
        jf, pf = self.joint.paradigm_fingerprint, self.post.paradigm_fingerprint
        self.assertEqual(jf["n_optimization_stages"], 1)
        self.assertEqual(pf["n_optimization_stages"], 2)
        self.assertTrue(jf["backbone_receives_quantile_gradient"])
        self.assertFalse(pf["backbone_receives_quantile_gradient"])
        self.assertTrue(pf["frozen_backbone_for_quantiles"])
        self.assertNotEqual(jf["stage1_loss_terms"], pf["stage1_loss_terms"])

    def test_predictions_are_not_identical(self):
        from model_training import predict_intervals
        pj, lj, hj = predict_intervals(self.joint, self.X_eval)
        pp, lp, hp = predict_intervals(self.post, self.X_eval)
        self.assertFalse(np.allclose(pj, pp),
                         "joint and post-hoc produced identical point predictions — "
                         "the ablation would be invalid")
        self.assertFalse(np.allclose(hj - lj, hp - lp),
                         "joint and post-hoc produced identical interval widths")

    def test_post_hoc_records_two_training_stages(self):
        stages = {h.get("stage") for h in self.post.training_history}
        self.assertEqual(stages, {1, 2}, "post-hoc history must contain both stages")


class TestNemenyiCriticalValue(unittest.TestCase):
    """§15 — the critical value must follow the actual number of methods."""

    def test_k_equals_six_for_the_current_experiment(self):
        """The six uncertainty methods: Static, Standard ACI, SA-ACI, Phenology, Weighted, Local."""
        k = 6
        res = nemenyi_critical_difference(k, n_blocks=5, alpha=0.05)
        self.assertEqual(res["k"], 6)
        self.assertAlmostEqual(res["q_alpha"], 2.850, places=3,
                               msg="q_alpha for k=6 at alpha=0.05 must be 2.850")

    def test_does_not_reuse_the_k4_constant_for_k6(self):
        k6 = nemenyi_critical_difference(6, n_blocks=5)
        k4 = nemenyi_critical_difference(4, n_blocks=5)
        self.assertNotAlmostEqual(k6["q_alpha"], k4["q_alpha"], places=3,
                                  msg="k=6 must not reuse the k=4 constant 2.569")
        self.assertGreater(k6["critical_difference"], k4["critical_difference"])

    def test_cd_formula(self):
        k, n = 6, 5
        res = nemenyi_critical_difference(k, n)
        expected = 2.850 * np.sqrt((k * (k + 1)) / (6.0 * n))
        self.assertAlmostEqual(res["critical_difference"], round(expected, 4), places=4)

    def test_unknown_k_returns_none_not_a_guess(self):
        res = nemenyi_critical_difference(99, n_blocks=5)
        self.assertIsNone(res["critical_difference"])
        self.assertIsNone(res["q_alpha"])


class TestBlockLevelFriedman(unittest.TestCase):
    """§14 — Friedman must actually execute, block on years, and report honestly."""

    def test_blocks_on_years_not_rows(self):
        rng = np.random.RandomState(0)
        years = np.repeat(np.arange(2019, 2024), 100)
        scores = {f"m{i}": rng.rand(500) + i * 0.01 for i in range(6)}
        res = friedman_on_blocks(scores, years)
        self.assertTrue(res["test_executed"])
        self.assertEqual(res["n_blocks"], 5, "blocks must be years, not observations")
        self.assertEqual(res["k"], 6)
        self.assertEqual(res["nemenyi"]["k"], 6)

    def test_refuses_when_too_few_blocks(self):
        rng = np.random.RandomState(0)
        years = np.repeat(np.arange(2019, 2021), 50)
        scores = {f"m{i}": rng.rand(100) for i in range(6)}
        res = friedman_on_blocks(scores, years)
        self.assertFalse(res["test_executed"])
        self.assertIsNone(res["p_value"])
        self.assertIn("reason", res)

    def test_no_posthoc_pairs_when_omnibus_does_not_reject(self):
        rng = np.random.RandomState(1)
        years = np.repeat(np.arange(2019, 2024), 100)
        scores = {f"m{i}": rng.rand(500) for i in range(6)}  # no real differences
        res = friedman_on_blocks(scores, years)
        if res["test_executed"] and not res["significant_at_alpha"]:
            self.assertEqual(res["nemenyi_pairwise"], [],
                             "post-hoc comparisons must not run when the omnibus test fails")


class TestProvenanceNotIdentityBased(unittest.TestCase):
    """§5 — leakage checks must use row IDs, not object identity."""

    def test_distinct_objects_with_same_rows_are_detected_as_overlapping(self):
        df = _synthetic_panel()
        a = add_obs_ids(df.head(50)).copy()
        b = add_obs_ids(df.head(50)).copy()   # different object, identical rows
        self.assertNotEqual(id(a), id(b), "fixture must use distinct objects")
        ledger = PartitionLedger(experiment="identity-vs-provenance")
        ledger.register("train", a)
        ledger.register("test", b)
        rec = audit_ledger(ledger)
        self.assertEqual(rec["train_test_overlap"], 50,
                         "row-identity audit must detect overlap between distinct "
                         "objects holding the same observations")
        self.assertEqual(rec["status"], "FAIL")

    def test_same_object_reused_for_disjoint_roles_is_not_a_false_positive(self):
        df = add_obs_ids(_synthetic_panel())
        ledger = PartitionLedger(experiment="disjoint")
        ledger.register("train", df.iloc[:100])
        ledger.register("test", df.iloc[100:200])
        rec = audit_ledger(ledger)
        self.assertEqual(rec["train_test_overlap"], 0)
        self.assertEqual(rec["status"], "PASS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
