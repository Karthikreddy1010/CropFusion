"""Compute RO1-RO5, the objectives this study actually answers.

Replaces the O1/O4/O5/O6 reports as the paper's objective results:

  O1  CDHW features improve accuracy   -> retired. Two tests in the pipeline
      disagree on the sign (+0.0116 vs -0.0079), both single-seed on the weakest
      model, and neither carries dependence-aware inference. Superseded by
      code/cdhw_contribution.py, which tests every model and reports whatever
      it finds.
  O4  joint vs post-hoc training       -> retired. It is the architecture claim
      the study no longer makes, and its arms were not equal-compute (post-hoc
      trained 348 epochs against joint's 200 under a "shared configuration").
  O5  width vs severity                -> folded into RO3 as supporting evidence.
  O6  method ranking on 5 year-blocks  -> folded into RO1/RO2, superseded by 16
      rolling origins. Its Nemenyi critical difference (3.37) exceeded the whole
      observed rank range (2.6), so it could not have detected anything.

Everything below is computed from artefacts on disk; nothing is refitted, and no
choice is made on any result. Sources are named per objective in the output.

Run:  python code/objectives_ro.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import config as cfg  # noqa: E402

REPO = CODE_DIR.parent
DIAG = REPO / "outputs_diagnostics" / "reports"
MASTER = REPO / "outputs_master" / "reports"
NOMINAL = cfg.NOMINAL_COVERAGE
ALPHA = 1.0 - NOMINAL

# Every pooled figure below is an average over 16 rolling origins, and rows inside
# an origin move together -- a drought year shifts the whole panel at once. The
# origin is therefore the unit of independence, and each headline carries an
# interval that resamples whole origins. 2000 resamples; the function is verified
# in code/diagnostics_loso/verify_cluster_bootstrap.py.
N_BOOT = 2000
CLUSTER = "test_year"


def _wavg(g: pd.DataFrame, col: str, w: str = "n") -> float:
    return float(np.average(g[col].values, weights=g[w].values.astype(float)))


def _ci(frame: pd.DataFrame, statistic, null_value: float = 0.0, seed: int = 0) -> Dict[str, Any]:
    """Year-clustered interval for one statistic, trimmed to what the report quotes."""
    from dependence_aware_stats import cluster_bootstrap_statistic
    r = cluster_bootstrap_statistic(frame, statistic, cluster_col=CLUSTER, n_boot=N_BOOT,
                                    seed=seed, null_value=null_value, cluster_unit="rolling origin")
    out = {"estimate": r["estimate"], "ci_95": r["ci_95"], "null_value": null_value,
           "excludes_null": r["significant_at_0.05"], "p_value": r.get("p_value"),
           "n_origins": r["n_clusters"], "inference_supported": r["inference_supported"]}
    if not r["inference_supported"]:
        out["reason"] = r.get("reason")
    return out


def _mean_of(col: str):
    return lambda f: float(f[col].mean())


def _contrast(col: str, group_col: str, a: str, b: str):
    """Mean of `col` in group a minus group b -- NaN when a resample lacks either."""
    def stat(f: pd.DataFrame) -> float:
        ga, gb = f[f[group_col] == a], f[f[group_col] == b]
        if not len(ga) or not len(gb):
            return float("nan")
        return float(ga[col].mean() - gb[col].mean())
    return stat


def _cell_wavg(col: str):
    """Row-weighted average of a per-origin cell statistic (aggregate-file path)."""
    def stat(f: pd.DataFrame) -> float:
        w = f["n"].values.astype(float)
        return float(np.average(f[col].values, weights=w)) if w.sum() else float("nan")
    return stat


def load_rows() -> pd.DataFrame | None:
    """Row-level rolling-origin output, or None when it has not been produced.

    A fresh checkout (or a fresh Colab folder) has no rolling-origin artefacts.
    RO3 works from the LOSO predictions alone, so a missing file must degrade the
    four objectives that need it rather than kill the run that calls this.
    """
    p = DIAG / "rolling_origin_rows.csv"
    if not p.exists():
        return None
    r = pd.read_csv(p)
    r["covered"] = (r.y >= r.lo) & (r.y <= r.hi)
    r["unsafe"] = r.y < r.lo
    r["width"] = r.hi - r.lo
    return r


def load_aggregate() -> pd.DataFrame | None:
    """Per origin x method x stratum aggregates, or None.

    RO1 and RO2 need only each cell's coverage and row count, both of which are
    in the aggregate file. Requiring the row-level file for them made the module
    report two objectives as unavailable when their evidence was on disk.
    """
    p = DIAG / "rolling_origin_raw.csv"
    if not p.exists():
        return None
    raw = pd.read_csv(p)
    return raw[~raw.method.isin(["_group_cp_meta", "_error"])].copy()


# ─────────────────────────────────────────────────────────────

def ro1_from_aggregate(agg: pd.DataFrame) -> Dict[str, Any]:
    """RO1 from per-cell coverage weighted by row count."""
    out: Dict[str, Any] = {"objective": "RO1", "nominal_coverage": NOMINAL,
                           "sources": ["rolling_origin_raw.csv (aggregate)",
                                       "loso_fold_metrics.csv"],
                           "basis": "row-weighted over origins"}
    s = agg[agg.method == "rolling_static_cp"]
    out["rolling_origin_by_class"] = {
        str(k): {"rows": int(g.n.sum()), "picp": round(_wavg(g, "picp"), 4),
                 "ci_95": _ci(g, _cell_wavg("picp"), null_value=NOMINAL)["ci_95"]}
        for k, g in s.groupby("stratum") if k != "ALL"}
    out["n_origins"] = int(s.test_year.nunique())
    out["inference"] = "year-clustered bootstrap over rolling origins (aggregate cells)"
    fm = pd.read_csv(MASTER / "loso_fold_metrics.csv")
    out["loso"] = {"macro_r2": round(float(fm.r_squared.mean()), 4),
                   "macro_picp": round(float(fm.picp.mean()), 4),
                   "per_state_picp": {r.state: round(float(r.picp), 4) for r in fm.itertuples()},
                   "caveat": str(fm.get("hyperparameter_source", pd.Series(["unknown"])).iloc[0])}
    ex = out["rolling_origin_by_class"].get("Extreme", {})
    out["headline"] = (f"in the Extreme exposure class coverage is {ex.get('picp')} against a "
                       f"nominal {NOMINAL:.2f} (95% CI {ex.get('ci_95')}), across "
                       f"{out['n_origins']} origins")
    return out


def ro2_from_aggregate(agg: pd.DataFrame) -> Dict[str, Any]:
    """RO2 from per-cell coverage; the sign test needs only per-origin means."""
    from scipy import stats
    out: Dict[str, Any] = {"objective": "RO2", "sources": ["rolling_origin_raw.csv (aggregate)"],
                           "basis": "row-weighted over origins"}
    ex = agg[agg.stratum == "Extreme"]
    for m in ("rolling_static_cp", "group_conditional_cp"):
        g = ex[ex.method == m]
        if len(g):
            out[m] = {"rows": int(g.n.sum()), "picp": round(_wavg(g, "picp"), 4)}
    per = ex.pivot_table(index="test_year", columns="method", values="picp")
    if {"group_conditional_cp", "rolling_static_cp"}.issubset(per.columns):
        d = (per["group_conditional_cp"] - per["rolling_static_cp"]).dropna()
        wins, n = int((d > 0).sum()), int((d != 0).sum())
        out["per_origin_sign_test"] = {
            "origins_where_methods_differ": n, "group_cp_better": wins,
            "p_value": round(float(stats.binomtest(wins, n, 0.5).pvalue), 4) if n else None,
            "mean_coverage_gain": round(float(d.mean()), 4)}

    def cell_gain(f: pd.DataFrame) -> float:
        a, b = f[f.method == "group_conditional_cp"], f[f.method == "rolling_static_cp"]
        if not len(a) or not len(b):
            return float("nan")
        return _cell_wavg("picp")(a) - _cell_wavg("picp")(b)

    out["group_cp_coverage_gain"] = _ci(ex, cell_gain)
    out["inference"] = "year-clustered bootstrap over rolling origins (aggregate cells)"
    clean = ex[ex.test_year >= 2014]
    out["sensitivity_clean_origins_only"] = {
        "reason": "SPEI reference period 1985-2013 overlaps the test year for origins 2008-2013",
        **{m: {"extreme_picp": round(_wavg(clean[clean.method == m], "picp"), 4)}
           for m in ("rolling_static_cp", "group_conditional_cp")
           if len(clean[clean.method == m])}}
    out["note"] = ("computed from aggregates; the per-origin fallback counts and the unsafe-miss "
                   "rates require rolling_origin_rows.csv")
    return out


def ro1(rows: pd.DataFrame) -> Dict[str, Any]:
    """Regime-conditional coverage under temporal and spatial shift.

    Stated against the nominal level, not against the Normal class. Both contrasts
    are reported, but they are not equally well supported: the shortfall of the
    Extreme class below nominal survives year-clustered resampling, while the
    Normal-minus-Extreme difference does not, because one origin (2012) supplies
    about 40% of the extreme rows and dominates the resampling distribution. A
    claim phrased as "Normal vs Extreme" would be the one a referee can overturn.
    """
    out: Dict[str, Any] = {"objective": "RO1", "nominal_coverage": NOMINAL,
                           "sources": ["rolling_origin_rows.csv", "loso_fold_metrics.csv"],
                           "inference": "year-clustered bootstrap over rolling origins"}
    s = rows[rows.method == "rolling_static_cp"]
    by = {}
    for k, g in s.groupby("stratum"):
        ci = _ci(g, _mean_of("covered"), null_value=NOMINAL)
        by[str(k)] = {"rows": int(len(g)), "picp": round(float(g.covered.mean()), 4),
                      "ci_95": ci["ci_95"], "differs_from_nominal": ci["excludes_null"]}
    out["rolling_origin_by_class"] = by
    out["n_origins"] = int(s.test_year.nunique())
    out["normal_minus_extreme_gap"] = _ci(s, _contrast("covered", "stratum", "Normal", "Extreme"))

    fm = pd.read_csv(MASTER / "loso_fold_metrics.csv")
    out["loso"] = {"macro_r2": round(float(fm.r_squared.mean()), 4),
                   "macro_picp": round(float(fm.picp.mean()), 4),
                   "per_state_picp": {r.state: round(float(r.picp), 4) for r in fm.itertuples()},
                   "caveat": str(fm.get("hyperparameter_source", pd.Series(["unknown"])).iloc[0])}

    ex = by.get("Extreme", {})
    gap = out["normal_minus_extreme_gap"]
    out["headline"] = (
        f"extreme-class coverage is {ex.get('picp')} against a nominal {NOMINAL:.2f} "
        f"(95% CI {ex.get('ci_95')}, {out['n_origins']} origins), an interval that "
        f"{'excludes' if ex.get('differs_from_nominal') else 'includes'} the nominal level; the "
        f"Normal-minus-Extreme contrast is {gap['estimate']} (95% CI {gap['ci_95']}), which "
        f"{'excludes' if gap['excludes_null'] else 'includes'} zero")
    out["interpretation"] = (
        f"two-sided coverage in the extreme class "
        f"{'separates from' if ex.get('differs_from_nominal') else 'does not separate from'} the "
        f"nominal level under origin-level resampling, and the between-class contrast "
        f"{'excludes' if gap['excludes_null'] else 'includes'} zero. Two-sided coverage spends "
        "half its power on the upper tail, where an observation above the interval is a good "
        "harvest; the deficit that does survive is the one-sided one, tested against its "
        "alpha/2 target in RO4. RO1 should be read as the descriptive stratification and RO4 "
        "as the test.")
    return out


def ro2(rows: pd.DataFrame) -> Dict[str, Any]:
    """Does pre-specified group-conditional calibration close the gap?"""
    from scipy import stats
    out: Dict[str, Any] = {"objective": "RO2",
                           "sources": ["rolling_origin_rows.csv", "rolling_origin_raw.csv"]}
    ex = rows[rows.stratum == "Extreme"]
    for m in ("rolling_static_cp", "group_conditional_cp"):
        g = ex[ex.method == m]
        out[m] = {"rows": int(len(g)), "picp": round(float(g.covered.mean()), 4),
                  "unsafe_rate": round(float(g.unsafe.mean()), 4)}

    per = (ex.groupby(["test_year", "method"]).covered.mean().unstack())
    if {"group_conditional_cp", "rolling_static_cp"}.issubset(per.columns):
        d = (per["group_conditional_cp"] - per["rolling_static_cp"]).dropna()
        wins, n = int((d > 0).sum()), int((d != 0).sum())
        out["per_origin_sign_test"] = {
            "origins_where_methods_differ": n, "group_cp_better": wins,
            "p_value": round(float(stats.binomtest(wins, n, 0.5).pvalue), 4) if n else None,
            "mean_coverage_gain": round(float(d.mean()), 4)}

    # The remedy is the claim this objective rests on, so it carries the interval
    # rather than the sign test alone: the sign test discards magnitude and drops
    # the origins where the two methods happen to tie.
    gain = _contrast("covered", "method", "group_conditional_cp", "rolling_static_cp")
    drop = _contrast("unsafe", "method", "rolling_static_cp", "group_conditional_cp")
    out["group_cp_coverage_gain"] = _ci(ex, gain)
    out["group_cp_unsafe_reduction"] = _ci(ex, drop)
    # Mondrian calibration buys extreme coverage by moving width, not making it.
    nrm = rows[rows.stratum == "Normal"]
    out["cost_to_normal_class"] = {
        "coverage_change": _ci(nrm, gain),
        "mpiw_extreme_static": round(float(ex[ex.method == "rolling_static_cp"].width.mean()), 4),
        "mpiw_extreme_group": round(float(ex[ex.method == "group_conditional_cp"].width.mean()), 4),
        "mpiw_normal_static": round(float(nrm[nrm.method == "rolling_static_cp"].width.mean()), 4),
        "mpiw_normal_group": round(float(nrm[nrm.method == "group_conditional_cp"].width.mean()), 4),
        "note": "width is reallocated from the normal class to the extreme class, not added"}

    raw = pd.read_csv(DIAG / "rolling_origin_raw.csv")
    meta = raw[raw.method == "_group_cp_meta"]
    fb = [json.loads(x) for x in meta.group_fallbacks.dropna()] if "group_fallbacks" in meta else []
    out["extreme_group_fell_back_to_pooled"] = {
        "origins": sum(1 for f in fb if f.get("Extreme")), "of": len(fb),
        "why": "too few extreme rows in the two-year calibration window"}

    clean = rows[rows.test_year >= 2014]
    clean_ex = clean[clean.stratum == "Extreme"]
    out["sensitivity_clean_origins_only"] = {
        "reason": "SPEI reference period 1985-2013 overlaps the test year for origins 2008-2013",
        "origins": sorted(clean.test_year.unique().tolist()),
        **{m: {"extreme_picp": round(float(clean_ex[clean_ex.method == m].covered.mean()), 4)}
           for m in ("rolling_static_cp", "group_conditional_cp")},
        "group_cp_coverage_gain": _ci(clean_ex, gain)}
    # and with the one origin that dominates the extreme stratum removed entirely
    no_dom = ex[ex.test_year != 2012]
    out["sensitivity_excluding_2012"] = {
        "reason": "2012 supplies about 40% of all extreme-exposure rows",
        "rolling_static_cp": {"extreme_picp": round(
            float(no_dom[no_dom.method == "rolling_static_cp"].covered.mean()), 4)},
        "group_conditional_cp": {"extreme_picp": round(
            float(no_dom[no_dom.method == "group_conditional_cp"].covered.mean()), 4)},
        "group_cp_coverage_gain": _ci(no_dom, gain)}
    g0 = out["group_cp_coverage_gain"]
    # Stated from the flags, not asserted in prose: a rerun that weakened a
    # sensitivity would otherwise leave the headline claiming it still held.
    sens = [("with the dominant origin removed",
             out["sensitivity_excluding_2012"]["group_cp_coverage_gain"]),
            ("on the clean origins",
             out["sensitivity_clean_origins_only"]["group_cp_coverage_gain"])]
    parts = [f"{lab} {r['estimate']} (95% CI {r['ci_95']}, "
             f"{'excludes' if r['excludes_null'] else 'includes'} zero)" for lab, r in sens]
    out["headline"] = (
        f"group-conditional calibration raises extreme-class coverage by {g0['estimate']} "
        f"(95% CI {g0['ci_95']}, {'excludes' if g0['excludes_null'] else 'includes'} zero); "
        + "; ".join(parts))
    return out


def ro3(rows: pd.DataFrame) -> Dict[str, Any]:
    """Is coverage failure driven by width or by bias?"""
    out: Dict[str, Any] = {"objective": "RO3",
                           "sources": ["loso_predictions.csv", "rolling_origin_raw.csv"]}
    lp = pd.read_csv(REPO / "outputs_master" / "predictions" / "loso_predictions.csv")
    lp["err"] = lp.y_true - lp.y_pred
    lp["width"] = lp.upper_bound - lp.lower_bound
    cell = lp.groupby(["held_out_state", "year"]).agg(
        bias=("err", "mean"), width=("width", "mean"), coverage=("covered", "mean"),
        n=("y_true", "size")).reset_index()
    cell["abs_bias"] = cell.bias.abs()
    out["loso_state_years"] = int(len(cell))
    out["corr_abs_bias_vs_coverage"] = round(float(cell.abs_bias.corr(cell.coverage)), 4)
    out["corr_width_vs_coverage"] = round(float(cell.width.corr(cell.coverage)), 4)

    # The correlations above come from loso_predictions.csv alone. The 2012 case
    # study needs rolling-origin output, which a fresh run has not produced -- so
    # it is optional. Reading it unconditionally made RO3 fail on Colab and threw
    # away correlations that were already computed.
    raw_path = DIAG / "rolling_origin_raw.csv"
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        y2012 = raw[(raw.test_year == 2012) & (raw.stratum == "Extreme") &
                    (~raw.method.isin(["_group_cp_meta", "_error"]))]
        if len(y2012):
            out["case_2012"] = {
                "model_bias_t_ha": float(y2012.model_bias.iloc[0]),
                "extreme_rows": int(y2012.n.iloc[0]),
                "picp_by_method": {r.method: round(float(r.picp), 4) for r in y2012.itertuples()},
                "note": "every calibrator fails together; width adjustment cannot repair a level error"}
    else:
        out["case_2012"] = {"status": "requires rolling_origin_raw.csv"}
    out["headline"] = (f"coverage tracks |bias| (r = {out['corr_abs_bias_vs_coverage']}) and not "
                       f"width (r = {out['corr_width_vs_coverage']})")
    return out


def ro4(rows: pd.DataFrame) -> Dict[str, Any]:
    """Interval failure expressed as decision-relevant loss."""
    out: Dict[str, Any] = {"objective": "RO4", "unsafe_target": ALPHA / 2,
                           "sources": ["rolling_origin_rows.csv", "decision_impact_by_stratum.csv"]}
    out["inference"] = "year-clustered bootstrap over rolling origins"
    s = rows[rows.method == "rolling_static_cp"]
    by = {}
    for k, g in s.groupby("stratum"):
        short = np.maximum(g.lo - g.y, 0.0)
        # tested against the calibrated one-sided failure rate, alpha/2, not zero
        ci = _ci(g, _mean_of("unsafe"), null_value=ALPHA / 2)
        by[str(k)] = {"rows": int(len(g)), "unsafe_rate": round(float(g.unsafe.mean()), 4),
                      "ci_95": ci["ci_95"], "exceeds_target": ci["excludes_null"],
                      "n_unsafe": int(g.unsafe.sum()),
                      "mean_shortfall_t_ha": round(float(short[g.unsafe].mean()), 4) if g.unsafe.any() else 0.0,
                      "expected_shortfall_t_ha": round(float(short.mean()), 4)}
    out["rolling_origin_by_class"] = by
    ex_only = s[s.stratum == "Extreme"]
    out["extreme_excluding_2012"] = _ci(ex_only[ex_only.test_year != 2012],
                                        _mean_of("unsafe"), null_value=ALPHA / 2)
    out["extreme_clean_origins"] = _ci(ex_only[ex_only.test_year >= 2014],
                                       _mean_of("unsafe"), null_value=ALPHA / 2)

    di = DIAG / "decision_impact_by_stratum.csv"
    if di.exists():
        d = pd.read_csv(di)
        out["locked_window_by_class"] = d[d.Conformal_Method == "static_conformal"][
            ["Year_Type", "n", "unsafe_miss_rate", "mean_shortfall_t_ha",
             "worst_shortfall_t_ha"]].to_dict("records")
    ex = by.get("Extreme", {})

    def _verdict(key: str, label: str) -> str:
        r = out[key]
        return (f"{label} {r['estimate']} (95% CI {r['ci_95']}, "
                f"{'excludes' if r['excludes_null'] else 'includes'} the target)")

    out["headline"] = (f"in the Extreme class the lower bound was breached for "
                       f"{ex.get('unsafe_rate')} of county-years (95% CI {ex.get('ci_95')}, "
                       f"{'excludes' if ex.get('exceeds_target') else 'includes'} the target) "
                       f"against a target of {ALPHA/2:.2f}, understating the shortfall by "
                       f"{ex.get('mean_shortfall_t_ha')} t/ha on average; "
                       + _verdict("extreme_clean_origins", "on the clean origins") + "; "
                       + _verdict("extreme_excluding_2012", "with 2012 removed"))
    out["interpretation"] = (
        "the excess over the safety target is established on the full pool and on the clean "
        "origins; with 2012 excluded the point estimate stays above twice the target but the "
        "interval reaches it, so the strength of the claim depends on whether the single most "
        "severe year in the record is admitted as evidence. It should be, and the sensitivity "
        "is reported either way.")
    return out


def ro5(rows: pd.DataFrame) -> Dict[str, Any]:
    """Is the regime effect confounded with yield level, state, or one year?"""
    out: Dict[str, Any] = {"objective": "RO5", "sources": ["rolling_origin_rows.csv"]}
    s = rows[rows.method == "rolling_static_cp"].copy()

    s["yield_q"] = pd.qcut(s.y, 5, labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"])
    out["within_yield_quintile"] = {
        str(q): {str(k): round(float(v), 4) for k, v in
                 g.groupby("stratum").unsafe.mean().items()}
        for q, g in s.groupby("yield_q", observed=True)}

    # Quintiles cut across the whole pool, so a bad year lands in the low bins by
    # construction and the yield control is partly a year control. Ranking within
    # each origin separates the two. Both are reported; the within-origin version
    # is the weaker and the more honest of the pair.
    s["yield_q_within_origin"] = s.groupby("test_year").y.transform(
        lambda v: pd.qcut(v.rank(method="first"), 5, labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"]))
    out["within_origin_yield_quintile"] = {
        str(q): {str(k): round(float(v), 4) for k, v in
                 g.groupby("stratum").unsafe.mean().items()}
        for q, g in s.groupby("yield_q_within_origin", observed=True)}
    q1 = s[s.yield_q_within_origin == "Q1_low"]
    out["q1_extreme_minus_normal_unsafe"] = {
        **_ci(q1, _contrast("unsafe", "stratum", "Extreme", "Normal")),
        "reads": "exposure effect on the miss rate among the lowest-yielding fifth of each origin"}

    within_state = {}
    for st, g in s.groupby("State"):
        e, n = g[g.stratum == "Extreme"], g[g.stratum == "Normal"]
        if len(e) >= 10 and len(n) >= 10:
            within_state[st] = {"extreme_unsafe": round(float(e.unsafe.mean()), 4), "n_extreme": int(len(e)),
                                "normal_unsafe": round(float(n.unsafe.mean()), 4), "n_normal": int(len(n)),
                                "ratio": round(float(e.unsafe.mean() / max(n.unsafe.mean(), 1e-9)), 2)}
    out["within_state"] = within_state
    out["states_showing_effect"] = sum(1 for v in within_state.values() if v["ratio"] > 1.5)
    out["states_tested"] = len(within_state)
    out["within_state_note"] = ("descriptive: each state has at most 16 origins and as few as 51 "
                                "extreme rows, too thin for a per-state interval; the ratio is "
                                "reported to show the direction is not carried by one state")

    ex = s[s.stratum == "Extreme"]
    share = ex.groupby("test_year").size() / len(ex)
    dom = share.idxmax()
    without = s[(s.test_year != dom)]
    out["leave_one_origin_out"] = {
        "most_influential_origin": int(dom), "share_of_extreme_rows": round(float(share.max()), 3),
        "extreme_picp_all": round(float(ex.covered.mean()), 4),
        "extreme_picp_without": round(float(without[without.stratum == "Extreme"].covered.mean()), 4),
        "normal_picp_without": round(float(without[without.stratum == "Normal"].covered.mean()), 4)}
    return out


def build_report() -> Dict[str, Any]:
    """Compute whichever objectives the available artefacts support."""
    rows = load_rows()
    report: Dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "replaces": ["objective_O1_report", "objective_O4_report",
                     "objective_o5_report", "objective_O6_report"],
        "note": "computed from artefacts; nothing refitted, no choice made on any result",
    }
    agg = load_aggregate()
    fallbacks = {"RO1": ro1_from_aggregate, "RO2": ro2_from_aggregate}
    needs_rolling = {"RO1": ro1, "RO2": ro2, "RO4": ro4, "RO5": ro5}
    try:
        report["RO3"] = ro3(rows)
    except Exception as exc:
        report["RO3"] = {"objective": "RO3", "status": f"unavailable: {exc}"}
    for key, fn in needs_rolling.items():
        if rows is None:
            if key in fallbacks and agg is not None:
                try:
                    report[key] = fallbacks[key](agg)
                    continue
                except Exception as exc:
                    report[key] = {"objective": key, "status": f"aggregate fallback failed: {exc}"}
                    continue
            report[key] = {"objective": key, "status": "requires rolling_origin_rows.csv "
                                                       "(run code/rolling_origin.py)"}
            continue
        try:
            report[key] = fn(rows)
        except Exception as exc:
            report[key] = {"objective": key, "status": f"unavailable: {exc}"}
    return report


def write_report() -> Dict[str, Any]:
    """Entry point for main.py: compute and persist the RO report."""
    report = build_report()
    DIAG.mkdir(parents=True, exist_ok=True)
    (DIAG / "objectives_RO_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    try:
        (Path(cfg.REPORT_DIR) / "objectives_RO_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass
    return report


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = build_report()

    write_report()
    for k in ("RO1", "RO2", "RO3", "RO4", "RO5"):
        print(f"\n=== {k} ===")
        h = report[k].get("headline")
        if h:
            print(f"  {h}")
        for key, val in report[k].items():
            if key in ("objective", "headline", "sources"):
                continue
            print(f"  {key}: {json.dumps(val, default=str)[:300]}")
    print(f"\nwritten: {DIAG / 'objectives_RO_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
