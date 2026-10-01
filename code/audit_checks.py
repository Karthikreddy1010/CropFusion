"""Publication-audit checks computed from the final run's artefacts (no model is refitted).

Answers questions the manuscript leaves open, for the publication audit:
  1. 2012: per-origin Extreme-class size and unsafe miss rate (static calibration), and
     the share of Extreme county-years each origin contributes.
  2. Bias versus width: the LOSO state-year correlations with year-clustered bootstrap
     intervals, Spearman versions, and how much |bias| and width actually vary.
  3. LOSO ensemble weights per fold (the manuscript quotes the locked-window weight).
  4. Multiplicity: Holm and Benjamini-Hochberg adjustments of the reported p-values,
     which the frozen protocol declares but the manuscript does not apply.
  5. The phenology label used by phenology-stratified CQR: its distribution.
  6. Referee-review checks (2026-09-26): locked-window marginal unsafe miss rates and the
     years behind the Extreme class; the rolling-origin Extreme miss rate with test years
     weighted equally; whether LOSO interval width grows with |bias|; the size of each
     attribution "stressed" group.

Inputs (all from the final runs of 2026-09-22/23):
  outputs_diagnostics/reports/rolling_origin_raw.csv
  outputs_diagnostics/reports/objectives_RO_report.json
  outputs_diagnostics/reports/phenology_coverage.json
  outputs_diagnostics/reports/rolling_origin_rows.csv
  outputs_master/predictions/loso_predictions.csv
  outputs_master/predictions/predictions.csv
  outputs_master/Paper3_Processed.csv
  master_dataset/model_ready/*.parquet
Output: outputs_diagnostics/reports/audit_checks.json

Run:  python code/audit_checks.py
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "outputs_diagnostics" / "reports"
sys.path.insert(0, str(ROOT / "code"))
import config as cfg  # noqa: E402

N_BOOT, SEED = 2000, 20260925


def holm(p: list) -> list:
    order = np.argsort(p)
    m, adj, running = len(p), [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = round(running, 4)
    return adj


def bh(p: list) -> list:
    order = np.argsort(p)[::-1]
    m, adj, running = len(p), [0.0] * len(p), 1.0
    for rank_from_top, i in enumerate(order):
        rank = m - rank_from_top
        running = min(running, p[i] * m / rank)
        adj[i] = round(min(1.0, running), 4)
    return adj


def review_checks(lp: pd.DataFrame) -> dict:
    """Checks raised by the referee-style review of 2026-09-26, from stored row-level outputs only."""
    res: dict = {}
    panel = pd.read_csv(ROOT / "outputs_master" / "Paper3_Processed.csv")

    # 6a. Locked window: marginal unsafe miss rate per method, Extreme county-years by year.
    # predictions.csv has no GEOID, so rows are aligned to the panel's test rows by order,
    # and the alignment is asserted on county name, year and yield for every method.
    pr = pd.read_csv(ROOT / "outputs_master" / "predictions" / "predictions.csv")
    test = panel[panel.Year.between(2019, 2023) & panel.Corn_Yield_tha.notna()].sort_values(["GEOID", "Year"])
    test = test.reset_index(drop=True)
    cls = np.where(test.CDHW_Event_Count >= 3, "Extreme", np.where(test.CDHW_Event_Count >= 1, "Moderate", "Normal"))
    locked = {}
    for meth, d in pr.groupby("Conformal_Method", sort=False):
        d = d.reset_index(drop=True)
        assert (d.County.values == test.County_Name.values).all() and (d.Year.values == test.Year.values).all()
        assert np.allclose(d.Observed_Yield.values, test.Corn_Yield_tha.values, atol=1e-4)
        low = d.Observed_Yield < d.Lower_Interval
        locked[meth] = {"marginal_unsafe_miss_rate": round(float(low.mean()), 4),
                        "extreme_unsafe_miss_rate": round(float(low[cls == "Extreme"].mean()), 4),
                        "rows_with_yield_equal_to_lower_bound": int((d.Observed_Yield == d.Lower_Interval).sum())}
    ex_years = test.Year[cls == "Extreme"].value_counts().sort_index()
    res["locked_window"] = {
        "source": "outputs_master/predictions/predictions.csv (bounds stored to 4 decimals), ensemble point model",
        "by_method": locked,
        "extreme_county_years_by_year": {int(k): int(v) for k, v in ex_years.items()},
        "extreme_county_years_2022_2023": int(ex_years.get(2022, 0) + ex_years.get(2023, 0)),
    }

    # 6b. Rolling origin: Extreme-class unsafe miss rate with each test year weighted equally
    rows = pd.read_csv(D / "rolling_origin_rows.csv")
    s = rows[(rows.method == "rolling_static_cp") & (rows.stratum == "Extreme")]
    per = (s.y < s.lo).groupby(s.test_year).mean()
    years = np.arange(2008, 2024)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(N_BOOT):
        v = [per[y] for y in rng.choice(years, size=len(years), replace=True) if y in per.index]
        if v:
            boot.append(float(np.mean(v)))
    k, n = int((per > 0.05).sum()), int(len(per))
    sign_p = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    res["rolling_extreme_unsafe_year_weighted"] = {
        "origins": n, "row_weighted": round(float((s.y < s.lo).mean()), 4),
        "year_weighted_mean": round(float(per.mean()), 4),
        "year_weighted_ci95": [round(float(np.percentile(boot, q)), 3) for q in (2.5, 97.5)],
        "median_over_origins": round(float(per.median()), 4),
        "year_weighted_mean_excluding_2012": round(float(per.drop(2012).mean()), 4),
        "origins_above_0.05": k, "sign_test_p_one_sided": round(sign_p, 3),
        "bootstrap": f"{N_BOOT} resamples of the 16 test years, seed {SEED}",
    }

    # 6c. LOSO: does interval width grow where the point prediction is biased?
    lp = lp.copy()
    lp["err"] = lp.y_true - lp.y_pred
    lp["width"] = lp.upper_bound - lp.lower_bound
    cell = lp.groupby(["held_out_state", "year"]).agg(bias=("err", "mean"), width=("width", "mean"),
                                                      coverage=("covered", "mean")).reset_index()
    cell["abs_bias"] = cell.bias.abs()
    yrs = cell.year.unique()
    boot = []
    for _ in range(N_BOOT):
        b = pd.concat([cell[cell.year == y] for y in rng.choice(yrs, size=len(yrs), replace=True)])
        boot.append(b.abs_bias.corr(b.width))
    widest = cell.loc[cell.width.idxmax()]
    rest = cell.drop(index=cell.width.idxmax())
    res["loso_bias_width"] = {
        "pearson_abs_bias_width": round(float(cell.abs_bias.corr(cell.width)), 4),
        "pearson_abs_bias_width_ci95": [round(float(np.percentile(boot, q)), 3) for q in (2.5, 97.5)],
        "spearman_abs_bias_width": round(float(cell.abs_bias.corr(cell.width, method="spearman")), 4),
        "widest_cell": {"state": str(widest.held_out_state), "year": int(widest.year),
                        "width_t_ha": round(float(widest.width), 2), "coverage": round(float(widest.coverage), 3)},
        "without_widest_cell": {"sd_width_t_ha": round(float(rest.width.std()), 3),
                                "pearson_width_coverage": round(float(rest.width.corr(rest.coverage)), 3),
                                "pearson_abs_bias_coverage": round(float(rest.abs_bias.corr(rest.coverage)), 3)},
    }

    # 6d. Attribution: share of county-years in the "stressed" group (ties in day counts shrink it)
    ph = json.loads((D / "phenology_coverage.json").read_text(encoding="utf-8"))
    r = rows[rows.method == "rolling_static_cp"][["GEOID", "test_year"]]
    shares = {}
    for t in ph["by_exposure"]:
        col = t["variable"]
        if col not in panel.columns:
            continue
        j = r.merge(panel[["GEOID", "Year", col]], left_on=["GEOID", "test_year"], right_on=["GEOID", "Year"])
        v = j[col]
        stressed = v < t["threshold"] if t["stressed_tail"] == "lower" else v > t["threshold"]
        shares[col] = round(float(stressed.mean()), 3)
    res["attribution_stressed_share"] = shares
    return res


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out: dict = {"generated_from": "final run artefacts, 2026-09-22/23; no model refitted"}

    # 1. 2012 and the per-origin Extreme class ------------------------------------------
    raw = pd.read_csv(D / "rolling_origin_raw.csv")
    ex = raw[(raw.method == "rolling_static_cp") & (raw.stratum == "Extreme")].sort_values("test_year")
    total = ex.n.sum()
    out["per_origin_extreme_static"] = [
        {"test_year": int(r.test_year), "n_extreme": int(r.n), "share_of_extreme": round(r.n / total, 3),
         "unsafe_miss_rate": round(float(r.unsafe_miss_rate), 4), "picp": round(float(r.picp), 4),
         "model_bias_t_ha": round(float(r.model_bias), 4)} for r in ex.itertuples()]
    out["origins_with_extreme"] = int(len(ex))
    out["origins_extreme_unsafe_above_0.05"] = int((ex.unsafe_miss_rate > 0.05).sum())

    # 2. Bias versus width on LOSO state-years ----------------------------------------
    lp = pd.read_csv(ROOT / "outputs_master" / "predictions" / "loso_predictions.csv")
    lp["err"] = lp.y_true - lp.y_pred
    lp["width"] = lp.upper_bound - lp.lower_bound
    cell = lp.groupby(["held_out_state", "year"]).agg(bias=("err", "mean"), width=("width", "mean"),
                                                      coverage=("covered", "mean")).reset_index()
    cell["abs_bias"] = cell.bias.abs()
    years = cell.year.unique()
    rng = np.random.default_rng(SEED)
    boots = {"abs_bias": [], "width": []}
    for _ in range(N_BOOT):
        pick = rng.choice(years, size=len(years), replace=True)
        b = pd.concat([cell[cell.year == y] for y in pick])
        boots["abs_bias"].append(b.abs_bias.corr(b.coverage))
        boots["width"].append(b.width.corr(b.coverage))
    out["bias_vs_width"] = {
        "unit": "held-out state x year cell (LOSO, ensemble, static split calibration)",
        "cells": int(len(cell)), "years": int(len(years)),
        "pearson_abs_bias_coverage": round(float(cell.abs_bias.corr(cell.coverage)), 4),
        "pearson_width_coverage": round(float(cell.width.corr(cell.coverage)), 4),
        "spearman_abs_bias_coverage": round(float(cell.abs_bias.corr(cell.coverage, method="spearman")), 4),
        "spearman_width_coverage": round(float(cell.width.corr(cell.coverage, method="spearman")), 4),
        "year_cluster_ci95_abs_bias": [round(float(np.percentile(boots["abs_bias"], q)), 3) for q in (2.5, 97.5)],
        "year_cluster_ci95_width": [round(float(np.percentile(boots["width"], q)), 3) for q in (2.5, 97.5)],
        "sd_abs_bias_t_ha": round(float(cell.abs_bias.std()), 4),
        "sd_width_t_ha": round(float(cell.width.std()), 4),
        "range_abs_bias_t_ha": [round(float(cell.abs_bias.min()), 3), round(float(cell.abs_bias.max()), 3)],
        "range_width_t_ha": [round(float(cell.width.min()), 3), round(float(cell.width.max()), 3)],
        "bootstrap": f"{N_BOOT} resamples of whole years, seed {SEED}",
    }

    # 3. LOSO ensemble weights per fold ------------------------------------------------
    w = lp.groupby("held_out_state").ensemble_weight_neuralcqr.agg(["min", "max"])
    out["loso_ensemble_weight_neuralcqr"] = {s: float(r["min"]) if r["min"] == r["max"] else [float(r["min"]),
                                                                                             float(r["max"])]
                                             for s, r in w.iterrows()}

    # 4. Multiplicity -----------------------------------------------------------------
    ph = json.loads((D / "phenology_coverage.json").read_text(encoding="utf-8"))
    rows = [r for r in ph["by_exposure"] if r.get("p_value") is not None]
    p = [float(r["p_value"]) for r in rows]
    out["attribution_multiplicity"] = [
        {"label": r["label"], "p": pv, "holm": h, "bh": b}
        for r, pv, h, b in zip(rows, p, holm(p), bh(p))]
    ro = json.loads((D / "objectives_RO_report.json").read_text(encoding="utf-8"))
    tests = {
        "RO1 Normal minus Extreme coverage": ro["RO1"]["normal_minus_extreme_gap"],
        "RO2 group-conditional gain": ro["RO2"]["group_cp_coverage_gain"],
        "RO2 gain excluding 2012": ro["RO2"]["sensitivity_excluding_2012"]["group_cp_coverage_gain"],
        "RO2 gain clean origins": ro["RO2"]["sensitivity_clean_origins_only"]["group_cp_coverage_gain"],
        "RO4 Extreme unsafe vs 0.05, excluding 2012": ro["RO4"]["extreme_excluding_2012"],
        "RO4 Extreme unsafe vs 0.05, clean origins": ro["RO4"]["extreme_clean_origins"],
        "RO5 lowest-quintile Extreme minus Normal": ro["RO5"]["q1_extreme_minus_normal_unsafe"],
    }
    names = [k for k, v in tests.items() if isinstance(v, dict) and v.get("p_value") is not None]
    tp = [float(tests[k]["p_value"]) for k in names]
    out["objective_tests_multiplicity"] = [{"test": n, "p": pv, "holm": h, "bh": b}
                                           for n, pv, h, b in zip(names, tp, holm(tp), bh(tp))]

    # 5. Phenology label used by phenology-stratified CQR ------------------------------
    df = pd.read_parquet(glob.glob(str(ROOT / "master_dataset" / "model_ready" / "*.parquet"))[0])
    g = df["prism_gdd_gs"]
    lab = pd.cut(g, [-np.inf, cfg.GDD_VEGETATIVE_END, cfg.GDD_SILKING_END, np.inf],
                 labels=["Vegetative", "Silking_R1", "Grain_Fill"])
    out["phenology_label_from_season_total_gdd"] = {
        "rule": f"season-total GDD <= {cfg.GDD_VEGETATIVE_END} vegetative, <= {cfg.GDD_SILKING_END} silking, "
                "else grain fill (feature_engineering._classify_phenological_window)",
        "counts": {k: int(v) for k, v in lab.value_counts().items()},
        "season_total_gdd_mean": round(float(g.mean()), 1),
        "season_total_gdd_min_max": [round(float(g.min()), 1), round(float(g.max()), 1)],
    }

    # 6. Referee-review checks (2026-09-26) ----------------------------------------------
    out["review_2026_09_26"] = review_checks(lp)

    (D / "audit_checks.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
