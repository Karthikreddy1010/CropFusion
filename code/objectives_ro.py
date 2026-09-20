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


def _wavg(g: pd.DataFrame, col: str, w: str = "n") -> float:
    return float(np.average(g[col].values, weights=g[w].values.astype(float)))


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
        str(k): {"rows": int(g.n.sum()), "picp": round(_wavg(g, "picp"), 4)}
        for k, g in s.groupby("stratum") if k != "ALL"}
    out["n_origins"] = int(s.test_year.nunique())
    fm = pd.read_csv(MASTER / "loso_fold_metrics.csv")
    out["loso"] = {"macro_r2": round(float(fm.r_squared.mean()), 4),
                   "macro_picp": round(float(fm.picp.mean()), 4),
                   "per_state_picp": {r.state: round(float(r.picp), 4) for r in fm.itertuples()},
                   "caveat": str(fm.get("hyperparameter_source", pd.Series(["unknown"])).iloc[0])}
    normal = out["rolling_origin_by_class"].get("Normal", {}).get("picp")
    extreme = out["rolling_origin_by_class"].get("Extreme", {}).get("picp")
    out["headline"] = (f"coverage falls from {normal} in the Normal class to {extreme} in the "
                       f"Extreme class at nominal {NOMINAL:.2f}, across {out['n_origins']} origins")
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
    """Regime-conditional coverage under temporal and spatial shift."""
    out: Dict[str, Any] = {"objective": "RO1", "nominal_coverage": NOMINAL,
                           "sources": ["rolling_origin_rows.csv", "loso_fold_metrics.csv"]}
    s = rows[rows.method == "rolling_static_cp"]
    out["rolling_origin_by_class"] = {
        str(k): {"rows": int(len(g)), "picp": round(float(g.covered.mean()), 4)}
        for k, g in s.groupby("stratum")}
    out["n_origins"] = int(s.test_year.nunique())

    fm = pd.read_csv(MASTER / "loso_fold_metrics.csv")
    out["loso"] = {"macro_r2": round(float(fm.r_squared.mean()), 4),
                   "macro_picp": round(float(fm.picp.mean()), 4),
                   "per_state_picp": {r.state: round(float(r.picp), 4) for r in fm.itertuples()},
                   "caveat": str(fm.get("hyperparameter_source", pd.Series(["unknown"])).iloc[0])}
    normal = out["rolling_origin_by_class"].get("Normal", {}).get("picp")
    extreme = out["rolling_origin_by_class"].get("Extreme", {}).get("picp")
    out["headline"] = (f"coverage falls from {normal} in the Normal class to {extreme} in the "
                       f"Extreme class at nominal {NOMINAL:.2f}, across {out['n_origins']} origins")
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

    raw = pd.read_csv(DIAG / "rolling_origin_raw.csv")
    meta = raw[raw.method == "_group_cp_meta"]
    fb = [json.loads(x) for x in meta.group_fallbacks.dropna()] if "group_fallbacks" in meta else []
    out["extreme_group_fell_back_to_pooled"] = {
        "origins": sum(1 for f in fb if f.get("Extreme")), "of": len(fb),
        "why": "too few extreme rows in the two-year calibration window"}

    clean = rows[rows.test_year >= 2014]
    out["sensitivity_clean_origins_only"] = {
        "reason": "SPEI reference period 1985-2013 overlaps the test year for origins 2008-2013",
        "origins": sorted(clean.test_year.unique().tolist()),
        **{m: {"extreme_picp": round(float(clean[(clean.method == m) &
                                                 (clean.stratum == "Extreme")].covered.mean()), 4)}
           for m in ("rolling_static_cp", "group_conditional_cp")}}
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
    s = rows[rows.method == "rolling_static_cp"]
    by = {}
    for k, g in s.groupby("stratum"):
        short = np.maximum(g.lo - g.y, 0.0)
        by[str(k)] = {"rows": int(len(g)), "unsafe_rate": round(float(g.unsafe.mean()), 4),
                      "n_unsafe": int(g.unsafe.sum()),
                      "mean_shortfall_t_ha": round(float(short[g.unsafe].mean()), 4) if g.unsafe.any() else 0.0,
                      "expected_shortfall_t_ha": round(float(short.mean()), 4)}
    out["rolling_origin_by_class"] = by

    di = DIAG / "decision_impact_by_stratum.csv"
    if di.exists():
        d = pd.read_csv(di)
        out["locked_window_by_class"] = d[d.Conformal_Method == "static_conformal"][
            ["Year_Type", "n", "unsafe_miss_rate", "mean_shortfall_t_ha",
             "worst_shortfall_t_ha"]].to_dict("records")
    ex = by.get("Extreme", {})
    out["headline"] = (f"in the Extreme class the lower bound was breached for "
                       f"{ex.get('unsafe_rate')} of county-years against a target of {ALPHA/2:.2f}, "
                       f"understating the shortfall by {ex.get('mean_shortfall_t_ha')} t/ha on average")
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
