"""
final_audit_report.py - Generate FINAL_METHODOLOGY_AUDIT.md from run artefacts.

Every number and every verdict in the generated document is read from the JSON
and CSV files the pipeline produced. Nothing is typed in by hand, so the
document cannot drift from the results the way the previous hard-coded
"100% compliance" reports did.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import config as cfg

logger = logging.getLogger("paper3")


def _read_json(name: str) -> Optional[Dict[str, Any]]:
    p = cfg.REPORT_DIR / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv(name: str, base: Optional[Path] = None) -> Optional[pd.DataFrame]:
    p = (base or cfg.REPORT_DIR) / name
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def generate_final_audit(output_path: Optional[Path] = None) -> Path:
    """Write FINAL_METHODOLOGY_AUDIT.md next to the project root."""
    validation = _read_json("methodology_validation_report.json") or {}
    leakage = _read_json("leakage_provenance_audit.json") or {}
    claims = _read_json("claim_consistency_audit.json") or {}
    o4 = _read_json("objective_O4_report.json") or {}
    o5 = _read_json("objective_o5_report.json") or {}
    o6 = _read_json("objective_O6_report.json") or {}
    shap_rpt = _read_json("shap_consistency_report.json") or {}
    feat = _read_json("feature_selection_pipeline.json") or {}
    ablation = _read_json("ablation_report.json") or {}
    evaluation = _read_json("evaluation_report.json") or {}
    repro = _read_json("reproducibility_report.json") or {}
    loso = _read_csv("loso_fold_metrics.csv")

    checks = validation.get("checks", [])
    counts = validation.get("counts", {})

    lines: List[str] = []
    A = lines.append

    A("# FINAL METHODOLOGY AUDIT")
    A("")
    A("Generated directly from the artefacts of the corrected pipeline run. "
      "Every status below is computed, not asserted.")
    A("")
    A(f"- **Overall validator status**: `{validation.get('overall_status', 'NOT_RUN')}`")
    A(f"- **Checks**: PASS {counts.get('PASS', 0)} · FAIL {counts.get('FAIL', 0)} · "
      f"WARNING {counts.get('WARNING', 0)} · NOT_APPLICABLE {counts.get('NOT_APPLICABLE', 0)}")
    A(f"- **Leakage audit**: {'PASS' if leakage.get('all_passed') else 'FAIL'} "
      f"across {leakage.get('n_experiments', 0)} experiments (row-identity check on "
      f"`obs_id = <GEOID>_<Year>`)")
    A(f"- **Claim consistency audit**: `{claims.get('status', 'NOT_RUN')}` "
      f"({claims.get('n_unsupported', '?')} unsupported claim(s) across "
      f"{claims.get('files_scanned', '?')} generated files)")
    A("")

    # ── 1. Requirement-by-requirement status ─────────────────────────
    A("## 1. Requirement status")
    A("")
    A("| ID | Requirement | Status | Evidence |")
    A("|---|---|---|---|")
    for c in checks:
        ev = str(c.get("evidence", "")).replace("|", "\\|").replace("\n", " ")
        A(f"| {c['id']} | {c['requirement']} | **{c['status']}** | {ev} |")
    A("")

    # ── 2. Leakage table ─────────────────────────────────────────────
    A("## 2. Leakage audit (computed observation-ID overlaps)")
    A("")
    A("| Experiment | Train/Test | Val/Test | Cal/Test | Train/Val | Train/Cal | Val/Cal | Status |")
    A("|---|---:|---:|---:|---:|---:|---:|---|")
    for e in leakage.get("experiments", []):
        A(f"| {e['experiment']} | {e.get('train_test_overlap', 'n/a')} "
          f"| {e.get('validation_test_overlap', 'n/a')} "
          f"| {e.get('calibration_test_overlap', 'n/a')} "
          f"| {e.get('train_validation_overlap', 'n/a')} "
          f"| {e.get('train_calibration_overlap', 'n/a')} "
          f"| {e.get('validation_calibration_overlap', 'n/a')} "
          f"| **{e['status']}** |")
    A("")
    A("Overlaps are intersections of real county-year identifiers. No check in this "
      "audit uses Python object identity or NumPy memory addresses.")
    A("")

    # ── 3. LOSO results ──────────────────────────────────────────────
    A(f"## 3. LOSO-CV results ({len(cfg.LOSO_STATES)} folds — locked protocol)")
    A("")
    if loso is not None and not loso.empty:
        cols = [c for c in ["state", "rmse", "mae", "r_squared", "picp", "mpiw", "ace",
                            "winkler_score", "neuralcqr_ensemble_weight", "n_train",
                            "n_dev", "n_cal", "n_test", "n_features", "leakage_status"]
                if c in loso.columns]
        A(loso[cols].to_markdown(index=False))
        A("")
        num = [c for c in ["rmse", "mae", "r_squared", "picp", "mpiw", "ace", "winkler_score"]
               if c in loso.columns]
        if num:
            A("**Macro means across folds**: " + ", ".join(
                f"{c} = {loso[c].mean():.4f} (sd {loso[c].std():.4f})" for c in num))
            A("")
    else:
        A("_loso_fold_metrics.csv not available._")
        A("")

    # ── 4. Ablation ──────────────────────────────────────────────────
    A("## 4. Ablation study")
    A("")
    cfgs = ablation.get("configurations", [])
    if cfgs:
        keep = ["config", "n_features", "training_paradigm", "n_optimization_stages",
                "calibration", "rmse", "mae", "r_squared", "picp", "mpiw", "ace",
                "winkler_score"]
        rows = [{k: c.get(k) for k in keep if k in c} for c in cfgs]
        A(pd.DataFrame(rows).to_markdown(index=False))
        A("")
        A(f"**All configurations mutually distinct**: "
          f"`{ablation.get('all_configurations_distinct')}`")
        A("")
        dchecks = ablation.get("configuration_distinctness_checks", [])
        if dchecks:
            A("| Comparison | Expected difference | Identical points? | Identical MPIW? | Status |")
            A("|---|---|---|---|---|")
            for d in dchecks:
                A(f"| {d['comparison']} | {d['expected_difference']} | "
                  f"{d['identical_point_predictions']} | {d['identical_mpiw']} | **{d['status']}** |")
            A("")
    else:
        A("_ablation_report.json not available._")
        A("")

    # ── 5. O4 ────────────────────────────────────────────────────────
    A("## 5. O4 — joint end-to-end vs post-hoc")
    A("")
    if o4:
        comp = o4.get("comparison", {})
        A(f"**Verdict**: {o4.get('verdict')}")
        A("")
        A(f"- Point-prediction improvement supported: `{o4.get('point_prediction_improvement_supported')}`")
        A(f"- Interval-quality improvement supported: `{o4.get('interval_quality_improvement_supported')}`")
        A(f"- Distinctness of the two arms: `{o4.get('distinctness_check', {}).get('status')}` "
          f"(max |point difference| = {o4.get('distinctness_check', {}).get('max_abs_point_difference')})")
        A("")
        A("| Metric | Post-hoc | Joint | Absolute Δ | Relative Δ | Joint better? |")
        A("|---|---:|---:|---:|---:|---|")
        for key, label in (("rmse", "RMSE"), ("mae", "MAE"), ("r_squared", "R²"),
                           ("picp_static", "PICP (static)"), ("mpiw_static", "MPIW (static)"),
                           ("ace_static", "ACE (static)"), ("winkler_static", "Winkler (static)"),
                           ("picp_aci", "PICP (ACI)"), ("mpiw_aci", "MPIW (ACI)"),
                           ("ace_aci", "ACE (ACI)"), ("winkler_aci", "Winkler (ACI)")):
            d = comp.get(key)
            if not d:
                continue
            A(f"| {label} | {d['post_hoc']} | {d['joint']} | "
              f"{d['absolute_difference_joint_minus_posthoc']} | "
              f"{d['relative_difference_pct']}% | {d['joint_better']} |")
        A("")
    else:
        A("_objective_O4_report.json not available._")
        A("")

    # ── 6. O5 ────────────────────────────────────────────────────────
    A("## 6. O5 — interval width vs CDHW severity")
    A("")
    if o5:
        pi = o5.get("primary_inference", {})
        ess = o5.get("effective_sample_size", {})
        bb = o5.get("corroborating_year_block_bootstrap", {})
        A(f"- **Primary inference**: {pi.get('method')} clustered by `{pi.get('cluster_unit')}` "
          f"({pi.get('n_clusters')} clusters over {pi.get('n_observations')} rows)")
        A(f"- **Slope**: `{pi.get('slope')}`, 95% CI `{pi.get('ci_95')}`, p = `{pi.get('p_value')}`")
        A(f"- **Year-block bootstrap**: slope `{bb.get('slope_mean')}`, CI `{bb.get('ci_95')}`, "
          f"p = `{bb.get('p_value')}`")
        A(f"- **R²**: `{o5.get('variance_explained_fraction')}` — magnitude "
          f"**{o5.get('effect_magnitude_label')}**")
        A(f"- **Effective sample size**: `{ess.get('n_effective')}` of `{ess.get('n_observations')}` "
          f"rows (ICC = `{ess.get('icc')}`, design effect = `{ess.get('design_effect')}`)")
        A(f"- **Statistically detectable**: `{o5.get('statistically_detectable')}`")
        A("")
        A(f"> {o5.get('interpretation')}")
        A("")
        A(f"For contrast, the naive independence-assuming p-value was "
          f"`{o5.get('descriptive_association', {}).get('naive_ols_p_value')}` — reported only to "
          f"document the distortion, not used for any claim.")
        A("")
    else:
        A("_objective_o5_report.json not available._")
        A("")

    # ── 7. O6 ────────────────────────────────────────────────────────
    A("## 7. O6 — conformal method comparison")
    A("")
    if o6:
        fr = o6.get("friedman_test", {})
        A(f"- **Methods compared (k)**: `{o6.get('n_methods_compared')}` — "
          f"{', '.join(o6.get('methods_compared', []))}")
        A(f"- **Ranking status**: {o6.get('ranking_status')}")
        A(f"- **Friedman executed**: `{fr.get('test_executed')}`"
          + (f", Q = `{fr.get('statistic')}`, p = `{fr.get('p_value')}`, "
             f"blocked by `{fr.get('block_unit')}` ({fr.get('n_blocks')} blocks)"
             if fr.get("test_executed") else f" — {fr.get('reason')}"))
        nem = o6.get("nemenyi_posthoc") or {}
        if nem.get("critical_difference") is not None:
            A(f"- **Nemenyi CD**: `{nem.get('critical_difference')}` "
              f"(k = {nem.get('k')}, q_α = {nem.get('q_alpha')}, N = {nem.get('n_blocks')})")
        A(f"- **Claims statistical superiority**: `{o6.get('claims_statistical_superiority')}`")
        A("")
        rk = o6.get("rankings", [])
        if rk:
            A(pd.DataFrame(rk).to_markdown(index=False))
            A("")
        ps = o6.get("practical_significance", {})
        if ps:
            A(f"**Practical spread**: Winkler {ps.get('winkler_best')}–{ps.get('winkler_worst')} "
              f"({ps.get('winkler_relative_spread_pct')}% relative), PICP "
              f"{ps.get('picp_min')}–{ps.get('picp_max')} against a nominal "
              f"{cfg.NOMINAL_COVERAGE}, MPIW spread {ps.get('mpiw_relative_spread_pct')}%.")
            A("")
    else:
        A("_objective_O6_report.json not available._")
        A("")

    # ── 8. SHAP ──────────────────────────────────────────────────────
    A("## 8. SHAP attribution consistency")
    A("")
    if shap_rpt:
        A(f"- **Backbones explained**: {', '.join(shap_rpt.get('models_with_computed_shap', []))}")
        A(f"- **Excluded (not fabricated)**: {shap_rpt.get('models_excluded') or 'none'}")
        A(f"- **Similarity index**: `{shap_rpt.get('shap_similarity_index')}` "
          f"(Spearman `{shap_rpt.get('mean_spearman_correlation')}`, "
          f"Kendall `{shap_rpt.get('mean_kendall_tau_correlation')}`)")
        A(f"- **Agreement label**: **{shap_rpt.get('agreement_label')}** "
          f"(bands: {shap_rpt.get('agreement_bands')})")
        A("")
        A(f"> {shap_rpt.get('interpretation')}")
        A("")
    else:
        A("_shap_consistency_report.json not available._")
        A("")

    # ── 9. Feature funnel ────────────────────────────────────────────
    A("## 9. Feature-selection funnel")
    A("")
    if feat:
        A(f"`{feat.get('stage_1_candidates')}` candidates → "
          f"`{feat.get('stage_2_consensus_selected')}` consensus-selected → "
          f"`{feat.get('stage_3_after_multicollinearity_pruning')}` after multicollinearity "
          f"pruning → `{feat.get('stage_4_final_modeling_features')}` final modeling features.")
        A("")
        A(f"Sequence: {feat.get('sequence')}")
        A("")
        A(f"All stages fitted on: **{feat.get('all_fitted_on')}**")
        A("")
    else:
        A("_feature_selection_pipeline.json not available._")
        A("")

    # ── 10. Computational benchmark ──────────────────────────────────
    A("## 10. Computational benchmark")
    A("")
    comp = evaluation.get("computational_complexity", {})
    if comp and comp.get("status") == "ok":
        A(f"- **Device**: `{comp.get('device')}` ({comp.get('device_name')}), "
          f"CUDA available: `{comp.get('cuda_available')}`, torch `{comp.get('torch_version')}`")
        A(f"- **Method**: {comp.get('n_warmup_iterations')} warm-up iterations discarded, "
          f"{comp.get('n_timed_repeats')} timed repeats, gradients disabled: "
          f"`{comp.get('gradients_disabled')}`, CUDA synchronised: `{comp.get('cuda_synchronised')}`")
        A(f"- **Parameters**: {comp.get('n_parameters')}")
        A(f"- **GPU memory method**: {comp.get('gpu_memory_method')}")
        A(f"- **CPU memory method**: {comp.get('cpu_memory_method')}")
        A("")
        pb = comp.get("per_batch_results", [])
        if pb:
            A(pd.DataFrame(pb).to_markdown(index=False))
            A("")
        A(f"**Note**: {comp.get('throughput_measurement_note')}")
        A("")
    else:
        A(f"_Benchmark status: {comp.get('status', 'not available')}_")
        A("")

    # ── 11. Reproducibility ──────────────────────────────────────────
    A("## 11. Reproducibility")
    A("")
    st = repro.get("settings", {})
    if st:
        A(f"- **Dataset**: `{st.get('dataset_file')}`")
        A(f"- **SHA-256 (complete file)**: `{st.get('dataset_sha256_full_file')}`")
        A(f"- **Size**: `{st.get('dataset_size_bytes')}` bytes — checksum scope: "
          f"{st.get('dataset_checksum_scope')}")
        A(f"- **Global seed**: `{st.get('global_random_seed')}`")
        A("")

    # ── 12. Unsupported claims ───────────────────────────────────────
    A("## 12. Claim consistency across generated outputs")
    A("")
    if claims:
        A(f"Status: **{claims.get('status')}** — {claims.get('n_unsupported')} unsupported "
          f"claim(s) of {claims.get('n_matches')} phrase matches across "
          f"{claims.get('files_scanned')} files.")
        A("")
        unsupported = [f for f in claims.get("findings", []) if f["status"] == "UNSUPPORTED"]
        if unsupported:
            A("| File | Line | Claim type | Text |")
            A("|---|---:|---|---|")
            for f in unsupported[:100]:
                A(f"| `{f['file']}` | {f['line']} | {f['claim_type']} | "
                  f"{f['matched_text'].replace('|', chr(92) + '|')} |")
        else:
            A("No unsupported claims were found in the generated outputs.")
        A("")

    # ── 13. Failures / warnings summary ──────────────────────────────
    fails = [c for c in checks if c["status"] == "FAIL"]
    warns = [c for c in checks if c["status"] == "WARNING"]
    A("## 13. Outstanding issues")
    A("")
    if not fails and not warns:
        A("No failing or warning checks.")
    else:
        for c in fails:
            A(f"- **FAIL — {c['id']}**: {c['requirement']} — {c.get('evidence')}")
        for c in warns:
            A(f"- **WARNING — {c['id']}**: {c['requirement']} — {c.get('evidence')}")
    A("")

    out = output_path or (cfg.PROJECT_ROOT / "FINAL_METHODOLOGY_AUDIT.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Final methodology audit written -> %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(generate_final_audit())
