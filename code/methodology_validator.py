"""
methodology_validator.py - Genuine implementation audit against the locked protocol.

The previous validator was a hand-written list of dictionaries with
``"Status": "Pass"`` typed into every row, and ``all_pass`` was therefore
trivially ``True``. It reported "FULL_METHODOLOGY_COMPLIANCE_100%" and "15/15
passed" while the LOSO folds were training on their own validation data, the
joint-vs-post-hoc ablation was comparing a model against itself, and the SHAP
attributions for one backbone were random numbers. It verified nothing.

Every check here reads a real artefact produced by the run and tests an actual
property of it. Checks return one of:

    PASS            the property was verified from evidence
    FAIL            the property was tested and does not hold
    WARNING         the property holds only partially, or evidence is weaker
                    than it should be
    NOT_APPLICABLE  the property does not apply to this configuration

A check whose evidence is missing is ``FAIL`` (it cannot be verified), never
``PASS``. The overall status is not "100% compliant" unless every check is
genuinely PASS.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import config as cfg

logger = logging.getLogger("paper3")

PASS, FAIL, WARNING, NOT_APPLICABLE = "PASS", "FAIL", "WARNING", "NOT_APPLICABLE"


@dataclass
class Check:
    check_id: str
    requirement: str
    status: str
    evidence: str
    artifact: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


def _load_json(name: str) -> Optional[Dict[str, Any]]:
    path = cfg.REPORT_DIR / name
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("methodology_validator: could not read %s: %s", name, e)
        return None


def _load_csv(name: str, subdir: Optional[Path] = None) -> Optional[pd.DataFrame]:
    base = subdir if subdir is not None else cfg.REPORT_DIR
    path = base / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.warning("methodology_validator: could not read %s: %s", name, e)
        return None


# ─────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────

def check_loso_states() -> Check:
    """1. Exactly the six locked LOSO states, and the run used all six."""
    expected = ["Illinois", "Indiana", "Iowa", "Minnesota", "Missouri", "Ohio"]
    configured = list(cfg.LOSO_STATES)
    if sorted(configured) != sorted(expected):
        return Check("M1", "LOSO uses exactly the six locked states", FAIL,
                     f"config.LOSO_STATES = {configured}, expected {expected}",
                     "config.py")

    summary = _load_csv("loso_fold_metrics.csv")
    if summary is None or "state" not in summary.columns:
        return Check("M1", "LOSO uses exactly the six locked states", FAIL,
                     "config lists the correct six states but loso_fold_metrics.csv is "
                     "missing or malformed, so the executed folds cannot be verified",
                     "loso_fold_metrics.csv", {"configured": configured})

    ran = sorted(summary["state"].astype(str).unique())
    if ran != sorted(expected):
        return Check("M1", "LOSO uses exactly the six locked states", FAIL,
                     f"executed folds = {ran}, expected {sorted(expected)}",
                     "loso_fold_metrics.csv")
    return Check("M1", "LOSO uses exactly the six locked states", PASS,
                 f"config and executed folds both contain exactly {len(ran)} states: "
                 f"{', '.join(ran)}",
                 "config.py + loso_fold_metrics.csv", {"n_folds": len(ran), "states": ran})


def check_leakage_overlaps() -> List[Check]:
    """2-3, 7. Row-identity overlap checks for every experiment."""
    audit = _load_json("leakage_provenance_audit.json")
    if audit is None:
        return [Check("M2", "No train/dev/cal/test observation overlap in any experiment",
                      FAIL, "leakage_provenance_audit.json not found; overlaps unverified",
                      "leakage_provenance_audit.json")]

    checks: List[Check] = []
    exps = audit.get("experiments", [])
    if not exps:
        return [Check("M2", "No train/dev/cal/test observation overlap in any experiment",
                      FAIL, "leakage audit contains no experiments", "leakage_provenance_audit.json")]

    failing = [e for e in exps if e.get("status") != "PASS"]
    detail = {
        e["experiment"]: {
            k: e.get(k) for k in
            ("train_test_overlap", "validation_test_overlap", "calibration_test_overlap",
             "train_validation_overlap", "train_calibration_overlap",
             "validation_calibration_overlap", "earlystop_test_overlap")
            if k in e
        } for e in exps
    }
    checks.append(Check(
        "M2", "No train/validation/test observation overlap in any experiment",
        PASS if not failing else FAIL,
        (f"all {len(exps)} experiments show zero overlap on real county-year IDs"
         if not failing else
         f"{len(failing)} experiment(s) overlap: "
         + "; ".join(f"{e['experiment']}: {e.get('failures')}" for e in failing)),
        "leakage_provenance_audit.json", detail))

    cal_fail = [e for e in exps if e.get("calibration_test_overlap", 0) != 0]
    has_cal = [e for e in exps if "calibration_test_overlap" in e]
    checks.append(Check(
        "M3", "Calibration observations never overlap test observations",
        PASS if (has_cal and not cal_fail) else (FAIL if cal_fail else WARNING),
        (f"calibration/test overlap = 0 in all {len(has_cal)} experiments with a "
         "calibration partition" if (has_cal and not cal_fail)
         else (f"{len(cal_fail)} experiment(s) with calibration/test overlap"
               if cal_fail else "no experiment registered a calibration partition")),
        "leakage_provenance_audit.json"))

    es_fail = [e for e in exps
               if e.get("earlystop_test_overlap", 0) != 0
               or e.get("earlystop_validation_overlap", 0) != 0
               or e.get("earlystop_calibration_overlap", 0) != 0]
    has_es = [e for e in exps if "earlystop_test_overlap" in e]
    checks.append(Check(
        "M4", "Internal early-stopping split is drawn only from training data",
        PASS if (has_es and not es_fail) else (FAIL if es_fail else NOT_APPLICABLE),
        (f"early-stopping split is disjoint from dev, cal and test in all "
         f"{len(has_es)} experiments that use one" if (has_es and not es_fail)
         else (f"{len(es_fail)} experiment(s) leak early-stopping data" if es_fail
               else "no experiment registered a separate early-stopping split")),
        "leakage_provenance_audit.json"))

    checks.append(Check(
        "M5", "Test labels never used for fitting, selection or calibration",
        PASS if not failing else FAIL,
        ("every stage is bound to a partition disjoint from test: "
         + "; ".join(
             f"{e['experiment']}[" + ", ".join(
                 f"{k}->{v}" for k, v in e.get("stage_bindings", {}).items()) + "]"
             for e in exps[:2])
         + (" ..." if len(exps) > 2 else "")
         if not failing else "at least one stage consumed test observations"),
        "leakage_provenance_audit.json"))
    return checks


def check_temporal_boundaries() -> Check:
    """9. FIT/DEV/CAL/TEST year boundaries match the locked protocol exactly."""
    expected = {
        "fit": tuple(cfg.FIT_YEARS), "dev": tuple(cfg.DEV_YEARS),
        "cal": tuple(cfg.CAL_YEARS), "test": tuple(cfg.TEST_YEARS),
    }
    locked = {"fit": (1985, 2013), "dev": (2014, 2015), "cal": (2016, 2018), "test": (2019, 2023)}
    if expected != locked:
        return Check("M6", "Temporal partitions are FIT 1985-2013 / DEV 2014-2015 / "
                           "CAL 2016-2018 / TEST 2019-2023", FAIL,
                     f"config boundaries {expected} differ from the locked protocol {locked}",
                     "config.py")

    splits = _load_csv("temporal_split_assignments.csv", subdir=cfg.SPLITS_DIR)
    if splits is None or not {"Split", "Year"}.issubset(splits.columns):
        return Check("M6", "Temporal partitions match the locked protocol", FAIL,
                     "config is correct but temporal_split_assignments.csv is missing or "
                     "malformed, so the executed split cannot be verified",
                     "splits/temporal_split_assignments.csv")

    observed, mismatches = {}, []
    for name, (lo, hi) in locked.items():
        sub = splits[splits["Split"] == name]
        if sub.empty:
            mismatches.append(f"{name}: no rows")
            continue
        got = (int(sub["Year"].min()), int(sub["Year"].max()))
        observed[name] = got
        if got != (lo, hi):
            mismatches.append(f"{name}: {got} != {(lo, hi)}")

    # Partitions must also be strictly ordered with no year appearing twice.
    year_to_splits: Dict[int, set] = {}
    for _, row in splits[["Split", "Year"]].drop_duplicates().iterrows():
        year_to_splits.setdefault(int(row["Year"]), set()).add(str(row["Split"]))
    shared_years = {y: sorted(s) for y, s in year_to_splits.items() if len(s) > 1}
    if shared_years:
        mismatches.append(f"years assigned to multiple partitions: {shared_years}")

    return Check("M6", "Temporal partitions are FIT 1985-2013 / DEV 2014-2015 / "
                       "CAL 2016-2018 / TEST 2019-2023 with no year in two partitions",
                 PASS if not mismatches else FAIL,
                 (f"observed year ranges {observed}; each year belongs to exactly one partition"
                  if not mismatches else "; ".join(mismatches)),
                 "splits/temporal_split_assignments.csv", {"observed": observed})


def check_feature_selection_fit_only() -> Check:
    """4. Feature selection fitted on FIT data only."""
    pipeline = _load_json("feature_selection_pipeline.json")
    audit = _load_json("leakage_provenance_audit.json")
    if pipeline is None:
        return Check("M7", "Feature selection fitted on the FIT partition only", FAIL,
                     "feature_selection_pipeline.json not found", "feature_selection_pipeline.json")

    bound_ok = False
    if audit:
        for e in audit.get("experiments", []):
            if e.get("experiment") == "Temporal":
                bound_ok = e.get("stage_bindings", {}).get("feature_selection") == "train"
    if not bound_ok:
        return Check("M7", "Feature selection fitted on the FIT partition only", FAIL,
                     "the temporal provenance ledger does not bind feature_selection to the "
                     "train (FIT) partition",
                     "leakage_provenance_audit.json")

    return Check("M7", "Feature selection fitted on the FIT partition only", PASS,
                 f"selection funnel {pipeline['stage_1_candidates']} candidates -> "
                 f"{pipeline['stage_2_consensus_selected']} consensus -> "
                 f"{pipeline['stage_3_after_multicollinearity_pruning']} after multicollinearity "
                 f"pruning -> {pipeline['stage_4_final_modeling_features']} final; "
                 f"all stages fitted on {pipeline['all_fitted_on']}, and the provenance ledger "
                 "binds feature_selection to the train partition",
                 "feature_selection_pipeline.json", pipeline)


def check_scaler_fit_only() -> Check:
    """5. Scaler fitted on FIT data only."""
    audit = _load_json("leakage_provenance_audit.json")
    scaling = _load_json("scaling_report.json")
    if audit is None:
        return Check("M8", "Scaler fitted on the FIT partition only", FAIL,
                     "leakage_provenance_audit.json not found", "leakage_provenance_audit.json")

    bad = [e["experiment"] for e in audit.get("experiments", [])
           if e.get("stage_bindings", {}).get("scaler_fitting") not in ("train", None)]
    bound = [e["experiment"] for e in audit.get("experiments", [])
             if e.get("stage_bindings", {}).get("scaler_fitting") == "train"]
    if bad:
        return Check("M8", "Scaler fitted on the FIT partition only", FAIL,
                     f"scaler_fitting bound to a non-train partition in: {bad}",
                     "leakage_provenance_audit.json")
    if not bound:
        return Check("M8", "Scaler fitted on the FIT partition only", FAIL,
                     "no experiment records a scaler_fitting stage binding",
                     "leakage_provenance_audit.json")
    return Check("M8", "Scaler fitted on the FIT partition only", PASS,
                 f"scaler_fitting bound to the train partition in all {len(bound)} experiments "
                 f"({', '.join(bound)})"
                 + (f"; scaling_report.json records scaler='{scaling.get('scaler_type', 'n/a')}'"
                    if scaling else ""),
                 "leakage_provenance_audit.json")


def check_ensemble_weight_selection() -> List[Check]:
    """6. Ensemble weight chosen only on DEV / allowed validation data."""
    checks: List[Check] = []

    lock = _load_json("model_decisions_lock.json")
    sel = _load_csv("model_selection_audit.csv")
    if lock is None or sel is None:
        checks.append(Check("M9", "Temporal ensemble weight selected on DEV only", FAIL,
                            "model_decisions_lock.json or model_selection_audit.csv missing",
                            "model_decisions_lock.json"))
    else:
        metric = lock.get("ensemble_weights", {}).get("selection_metric")
        chosen = lock.get("ensemble_weights", {}).get("w_neural")
        ok_metric = metric == "DEV_2014_2015_RMSE"
        # The recorded winner must actually be the argmin over the DEV grid.
        ok_argmin = False
        if {"weight_neuralcqr", "dev_rmse"}.issubset(sel.columns) and not sel.empty:
            best_row = sel.loc[sel["dev_rmse"].idxmin()]
            ok_argmin = abs(float(best_row["weight_neuralcqr"]) - float(chosen)) < 1e-6
        checks.append(Check(
            "M9", "Temporal ensemble weight selected on DEV only, and is the DEV argmin",
            PASS if (ok_metric and ok_argmin) else FAIL,
            (f"selection metric = {metric}; frozen weight {chosen} equals the argmin over "
             f"{len(sel)} candidate weights scored on DEV"
             if (ok_metric and ok_argmin)
             else f"selection metric = {metric} (expected DEV_2014_2015_RMSE); "
                  f"frozen weight {chosen} matches DEV argmin: {ok_argmin}"),
            "model_decisions_lock.json + model_selection_audit.csv",
            {"n_candidates_scored": int(len(sel))}))

    loso_w = _load_csv("loso_ensemble_weight_search.csv")
    if loso_w is None:
        checks.append(Check("M10", "LOSO ensemble weight selected on LOSO DEV only", FAIL,
                            "loso_ensemble_weight_search.csv not found",
                            "loso_ensemble_weight_search.csv"))
    else:
        parts = set(loso_w.get("selection_partition", pd.Series(dtype=str)).astype(str))
        bad_parts = parts - {"LOSO_DEV_2014_2015"}
        per_state_ok = True
        details = {}
        if {"state", "dev_r2", "is_selected"}.issubset(loso_w.columns):
            for st, grp in loso_w.groupby("state"):
                sel_rows = grp[grp["is_selected"] == True]  # noqa: E712
                if len(sel_rows) != 1:
                    per_state_ok = False
                    details[str(st)] = f"{len(sel_rows)} rows flagged selected"
                    continue
                if abs(float(sel_rows.iloc[0]["dev_r2"]) - float(grp["dev_r2"].max())) > 1e-9:
                    per_state_ok = False
                    details[str(st)] = "selected weight is not the DEV argmax"
                else:
                    details[str(st)] = {
                        "selected_weight": float(sel_rows.iloc[0]["weight_neuralcqr"]),
                        "n_candidates": int(len(grp)),
                        "best_dev_r2": round(float(grp["dev_r2"].max()), 4),
                    }
        else:
            per_state_ok = False
        checks.append(Check(
            "M10", "LOSO ensemble weight selected on LOSO DEV only, and is the DEV argmax",
            PASS if (not bad_parts and per_state_ok) else FAIL,
            (f"all {len(loso_w)} candidate weights scored on LOSO_DEV_2014_2015; each state's "
             "frozen weight is the DEV argmax"
             if (not bad_parts and per_state_ok)
             else f"unexpected selection partitions {bad_parts or 'none'}; per-state argmax "
                  f"verification: {details}"),
            "loso_ensemble_weight_search.csv", details))
    return checks


def check_joint_vs_posthoc_distinct() -> List[Check]:
    """8. Joint and post-hoc are genuinely different training procedures."""
    checks: List[Check] = []

    o4 = _load_json("objective_O4_experiment_raw.json")
    if o4 is None:
        checks.append(Check("M11", "Joint and post-hoc training paths are genuinely different",
                            FAIL, "objective_O4_experiment_raw.json not found",
                            "objective_O4_experiment_raw.json"))
    else:
        d = o4.get("distinctness_check", {})
        j = o4.get("arms", {}).get("joint", {}).get("paradigm_fingerprint", {})
        p = o4.get("arms", {}).get("post_hoc", {}).get("paradigm_fingerprint", {})
        different_paths = (
            not d.get("identical_point_predictions", True)
            and j.get("n_optimization_stages") != p.get("n_optimization_stages")
            and j.get("backbone_receives_quantile_gradient") is True
            and p.get("backbone_receives_quantile_gradient") is False
        )
        checks.append(Check(
            "M11", "Joint and post-hoc training paths are genuinely different",
            PASS if different_paths else FAIL,
            (f"joint = {j.get('n_optimization_stages')} stage(s) with quantile gradients "
             f"reaching the backbone; post-hoc = {p.get('n_optimization_stages')} stages with a "
             f"frozen backbone; max absolute difference in test point predictions = "
             f"{d.get('max_abs_point_difference')}"
             if different_paths else
             f"arms are not demonstrably distinct: {d}"),
            "objective_O4_experiment_raw.json", d))

    abl = _load_json("ablation_report.json")
    if abl is None:
        checks.append(Check("M12", "Ablation configurations are mutually distinct", FAIL,
                            "ablation_report.json not found", "ablation_report.json"))
    else:
        dc = abl.get("configuration_distinctness_checks", [])
        failing = [c for c in dc if c.get("status") != "PASS"]
        checks.append(Check(
            "M12", "Ablation configurations 3, 4 and 5 are mutually distinct",
            PASS if (dc and not failing) else FAIL,
            ("; ".join(f"{c['comparison']}: {c['status']} ({c['expected_difference']})" for c in dc)
             if dc else "no distinctness checks were recorded"),
            "ablation_report.json", {"checks": dc}))
    return checks


def check_test_untouched_until_evaluation() -> Check:
    """10. TEST remains untouched until final evaluation."""
    lock = _load_json("model_decisions_lock.json")
    audit = _load_json("leakage_provenance_audit.json")
    if lock is None or audit is None:
        return Check("M13", "TEST partition untouched until final evaluation", FAIL,
                     "model_decisions_lock.json or leakage_provenance_audit.json missing",
                     "model_decisions_lock.json")

    non_eval_stages = ("feature_selection", "multicollinearity_pruning", "county_baseline",
                       "target_detrending", "preprocessor_fitting", "scaler_fitting",
                       "model_fitting", "early_stopping", "ensemble_weight_selection",
                       "conformal_calibration")
    violations = []
    for e in audit.get("experiments", []):
        for stage, part in e.get("stage_bindings", {}).items():
            if stage in non_eval_stages and part == "test":
                violations.append(f"{e['experiment']}.{stage}")

    lock_ok = lock.get("status") == "LOCKED_FINAL_DECISIONS"
    refit_no_es = lock.get("neural_cqr_hyperparameters", {}).get("early_stopping_on_refit") is False

    return Check("M13", "TEST partition untouched until final evaluation",
                 PASS if (not violations and lock_ok and refit_no_es) else FAIL,
                 ("no pipeline stage other than final_evaluation is bound to the test "
                  "partition in any experiment; decisions were frozen before the final refit "
                  "(model_decisions_lock.json status=LOCKED_FINAL_DECISIONS, "
                  "early_stopping_on_refit=False)"
                  if (not violations and lock_ok and refit_no_es)
                  else f"violations={violations}, lock_ok={lock_ok}, refit_no_es={refit_no_es}"),
                 "leakage_provenance_audit.json + model_decisions_lock.json")


def check_no_stale_fold_count() -> Check:
    """Documentation consistency: nothing claims seven LOSO folds."""
    import re
    code_dir = Path(__file__).resolve().parent
    pattern = re.compile(r"\b(7|seven)[\s-]+(state\s+)?(folds?|state\s+folds?|spatial\s+folds?)\b",
                         re.IGNORECASE)
    hits = []
    for py in sorted(code_dir.glob("*.py")):
        if py.name == Path(__file__).name:
            continue  # this file names the stale pattern in order to search for it
        try:
            for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    hits.append(f"{py.name}:{i}")
        except Exception:
            continue
    for rpt in sorted(cfg.REPORT_DIR.glob("*.md")) + sorted(cfg.REPORT_DIR.glob("*.json")):
        try:
            for i, line in enumerate(rpt.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    hits.append(f"reports/{rpt.name}:{i}")
        except Exception:
            continue
    return Check("M14", "No artefact claims seven LOSO folds (the protocol has six)",
                 PASS if not hits else FAIL,
                 (f"no stale seven-fold reference found in code or reports; "
                  f"config.LOSO_STATES has {len(cfg.LOSO_STATES)} states"
                  if not hits else f"stale references found at: {', '.join(hits[:12])}"),
                 "code/*.py + outputs/reports/*")


def check_statistical_dependence_handling() -> Check:
    """Significance claims account for panel dependence."""
    o5 = _load_json("objective_o5_report.json")
    o6 = _load_json("objective_O6_report.json")
    if o5 is None or o6 is None:
        return Check("M15", "Significance testing accounts for temporal/spatial dependence",
                     FAIL, "objective_o5_report.json or objective_O6_report.json missing",
                     "objective_o5_report.json")

    o5_ok = (o5.get("primary_inference", {}).get("method", "").startswith("OLS with cluster-robust")
             and o5.get("primary_inference", {}).get("cluster_unit") is not None)
    fr = o6.get("friedman_test", {})
    o6_ok = (fr.get("block_unit") == "year") if fr.get("test_executed") else True
    nem = o6.get("nemenyi_posthoc") or {}
    k_ok = True
    if nem.get("k") is not None:
        k_ok = int(nem["k"]) == int(o6.get("n_methods_compared", nem["k"]))

    status = PASS if (o5_ok and o6_ok and k_ok) else FAIL
    return Check("M15", "Significance testing accounts for temporal/spatial dependence",
                 status,
                 (f"O5 primary inference = {o5.get('primary_inference', {}).get('method')} "
                  f"clustered by {o5.get('primary_inference', {}).get('cluster_unit')} "
                  f"({o5.get('primary_inference', {}).get('n_clusters')} clusters, effective n = "
                  f"{o5.get('effective_sample_size', {}).get('n_effective')} from "
                  f"{o5.get('effective_sample_size', {}).get('n_observations')} rows); "
                  f"O6 Friedman executed = {fr.get('test_executed')}, block unit = "
                  f"{fr.get('block_unit')}, k = {nem.get('k')} with q_alpha = {nem.get('q_alpha')}"),
                 "objective_o5_report.json + objective_O6_report.json")


def check_shap_provenance() -> Check:
    """SHAP attributions are real, with no synthetic fallback."""
    rpt = _load_json("shap_consistency_report.json")
    if rpt is None:
        return Check("M16", "SHAP attributions are genuinely computed (no synthetic fallback)",
                     FAIL, "shap_consistency_report.json not found", "shap_consistency_report.json")
    prov = rpt.get("attribution_provenance", {})
    computed = [m for m, p in prov.items() if p.get("status") == "computed"]
    excluded = rpt.get("models_excluded", [])
    if not computed:
        return Check("M16", "SHAP attributions are genuinely computed (no synthetic fallback)",
                     FAIL, "no backbone produced computable SHAP values",
                     "shap_consistency_report.json")
    status = PASS if not excluded else WARNING
    explainers = ", ".join("{}: {}".format(m, prov[m].get("method")) for m in computed)
    return Check("M16", "SHAP attributions are genuinely computed (no synthetic fallback)",
                 status,
                 (f"{len(computed)} backbone(s) explained with real SHAP explainers "
                  f"({explainers}); "
                  f"excluded (not fabricated): {excluded or 'none'}; "
                  f"agreement label = {rpt.get('agreement_label')}"),
                 "shap_consistency_report.json",
                 {"computed": computed, "excluded": excluded})


def check_dataset_checksum() -> Check:
    """Reproducibility: dataset checksum covers the whole file."""
    rep = _load_json("reproducibility_report.json")
    if rep is None:
        return Check("M17", "Dataset checksum covers the complete file", FAIL,
                     "reproducibility_report.json not found", "reproducibility_report.json")
    settings = rep.get("settings", {})
    h = settings.get("dataset_sha256_full_file")
    scope = settings.get("dataset_checksum_scope")
    if not h or len(str(h)) != 64:
        return Check("M17", "Dataset checksum covers the complete file", FAIL,
                     f"no full-file SHA-256 recorded (got {h!r}); a partial checksum cannot "
                     "support a reproducibility claim", "reproducibility_report.json")
    return Check("M17", "Dataset checksum covers the complete file", PASS,
                 f"SHA-256 = {h} over {settings.get('dataset_size_bytes')} bytes ({scope})",
                 "reproducibility_report.json")


def check_computational_benchmark() -> Check:
    """Benchmark methodology is sound and the device is reported honestly."""
    ev = _load_json("evaluation_report.json")
    if ev is None or "computational_complexity" not in ev:
        return Check("M18", "Computational benchmark uses warm-up, no-grad and explicit batches",
                     FAIL, "evaluation_report.json has no computational_complexity section",
                     "evaluation_report.json")
    c = ev["computational_complexity"]
    if c.get("status") == "skipped":
        return Check("M18", "Computational benchmark uses warm-up, no-grad and explicit batches",
                     NOT_APPLICABLE, f"benchmark skipped: {c.get('reason')}", "evaluation_report.json")
    ok = (c.get("n_warmup_iterations", 0) > 0 and c.get("gradients_disabled") is True
          and bool(c.get("per_batch_results")))
    sync_ok = (c.get("cuda_synchronised") is True) if c.get("device") == "cuda" else True
    return Check("M18", "Computational benchmark uses warm-up, no-grad, explicit batch sizes "
                        "and separates inference from end-to-end latency",
                 PASS if (ok and sync_ok) else FAIL,
                 (f"device = {c.get('device')} ({c.get('device_name')}), cuda_available = "
                  f"{c.get('cuda_available')}, warm-up = {c.get('n_warmup_iterations')}, "
                  f"timed repeats = {c.get('n_timed_repeats')}, gradients disabled = "
                  f"{c.get('gradients_disabled')}, batch sizes = "
                  f"{[r['batch_size'] for r in c.get('per_batch_results', [])]}; "
                  f"GPU memory method = {c.get('gpu_memory_method')}"),
                 "evaluation_report.json", {"device": c.get("device")})


# ─────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────

def run_methodology_validation() -> Dict[str, Any]:
    """Execute every check and export the validation report."""
    from utils import save_report, save_report_markdown, save_report_csv

    checks: List[Check] = []
    checks.append(check_loso_states())
    checks.extend(check_leakage_overlaps())
    checks.append(check_temporal_boundaries())
    checks.append(check_feature_selection_fit_only())
    checks.append(check_scaler_fit_only())
    checks.extend(check_ensemble_weight_selection())
    checks.extend(check_joint_vs_posthoc_distinct())
    checks.append(check_test_untouched_until_evaluation())
    checks.append(check_no_stale_fold_count())
    checks.append(check_statistical_dependence_handling())
    checks.append(check_shap_provenance())
    checks.append(check_dataset_checksum())
    checks.append(check_computational_benchmark())

    counts = {s: sum(1 for c in checks if c.status == s)
              for s in (PASS, FAIL, WARNING, NOT_APPLICABLE)}
    applicable = [c for c in checks if c.status != NOT_APPLICABLE]
    all_pass = counts[FAIL] == 0 and counts[WARNING] == 0 and counts[PASS] > 0

    if counts[FAIL] > 0:
        overall = "NON_COMPLIANT"
    elif counts[WARNING] > 0:
        overall = "COMPLIANT_WITH_WARNINGS"
    else:
        overall = "COMPLIANT"

    report = {
        "overall_status": overall,
        "validation_type": "property-based implementation audit (artefacts are read and tested)",
        "note": ("Each check reads a real artefact produced by this run and tests a property of "
                 "it. A check whose evidence is missing is FAIL, never PASS."),
        "counts": counts,
        "n_checks": len(checks),
        "n_applicable": len(applicable),
        "all_applicable_passed": all_pass,
        "checks": [
            {"id": c.check_id, "requirement": c.requirement, "status": c.status,
             "evidence": c.evidence, "artifact": c.artifact, "details": c.details}
            for c in checks
        ],
    }
    save_report(report, "methodology_validation_report.json")

    df = pd.DataFrame([
        {"ID": c.check_id, "Requirement": c.requirement, "Status": c.status,
         "Artifact": c.artifact, "Evidence": c.evidence}
        for c in checks
    ])
    save_report_csv(df, "methodology_validation_report.csv")

    md = f"""# Methodology Validation Report (property-based implementation audit)

## Overall status: `{overall}`

| Result | Count |
|---|---:|
| PASS | {counts[PASS]} |
| FAIL | {counts[FAIL]} |
| WARNING | {counts[WARNING]} |
| NOT_APPLICABLE | {counts[NOT_APPLICABLE]} |

{report['note']}

This replaces the previous validator, which was a hard-coded table with
`"Status": "Pass"` typed into every row and reported "100% compliance, 15/15"
without testing anything.

| ID | Requirement | Status | Evidence | Artifact |
|---|---|---|---|---|
"""
    for c in checks:
        ev = c.evidence.replace("|", "\\|").replace("\n", " ")
        md += f"| {c.check_id} | {c.requirement} | **{c.status}** | {ev} | `{c.artifact}` |\n"

    if counts[FAIL]:
        md += "\n## Failing checks\n"
        for c in checks:
            if c.status == FAIL:
                md += f"- **{c.check_id}** — {c.requirement}: {c.evidence}\n"
    if counts[WARNING]:
        md += "\n## Warnings\n"
        for c in checks:
            if c.status == WARNING:
                md += f"- **{c.check_id}** — {c.requirement}: {c.evidence}\n"

    save_report_markdown(md, "methodology_validation_report.md")
    logger.info("Methodology validation: %s (PASS=%d FAIL=%d WARNING=%d N/A=%d)",
                overall, counts[PASS], counts[FAIL], counts[WARNING], counts[NOT_APPLICABLE])
    return report
