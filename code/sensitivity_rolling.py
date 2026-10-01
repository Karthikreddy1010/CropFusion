"""Rolling-origin sensitivity analyses declared in manuscript/SENSITIVITY_PLAN_2026-09-26.md.

Reads the row-level outputs of code/run_rolling_sensitivity.sh (R0 LightGBM, R1 CatBoost,
R2 XGBoost, R3 LightGBM with four calibration seasons) and the primary run, and computes
the statistics the plan declares, with the same year-clustered bootstrap and seeds as the
primary summaries so that R0 must reproduce the paper's figures exactly. Also computes the
exposure-threshold sensitivity (>=1, >=3, >=5 compound days) from the primary rows.
No model is fitted here.

Inputs:  outputs_diagnostics/reports/{rolling_origin,ro_sens_*}_rows.csv
         outputs_master/Paper3_Processed.csv (CDHW_Event_Count, for the thresholds)
Output:  outputs_diagnostics/reports/sensitivity_rolling.json

Run:  python code/sensitivity_rolling.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from dependence_aware_stats import cluster_bootstrap_statistic  # noqa: E402

ROOT = CODE.parent
D = ROOT / "outputs_diagnostics" / "reports"
RUNS = {"R0": ("LightGBM, 2 calibration seasons (primary design)", "ro_sens_lgbm"),
        "R1": ("CatBoost, 2 calibration seasons", "ro_sens_catboost"),
        "R2": ("XGBoost, 2 calibration seasons", "ro_sens_xgboost"),
        "R3": ("LightGBM, 4 calibration seasons", "ro_sens_cal4")}
METHODS = ["rolling_static_cp", "group_conditional_cp", "standard_aci", "sa_aci",
           "sa_aci_severity_score", "asymmetric_cqr", "asymmetric_group_cqr"]
PRIMARY_METHODS = METHODS[:4]
N_BOOT = 2000
YW_SEED = 20260925          # year-weighted interval: same stream as audit_checks.py section 6b


def prep(rows: pd.DataFrame) -> pd.DataFrame:
    r = rows.copy()
    r["covered"] = ((r.y >= r.lo) & (r.y <= r.hi)).astype(float)
    r["unsafe"] = (r.y < r.lo).astype(float)
    r["width"] = r.hi - r.lo
    return r


def ci(frame: pd.DataFrame, statistic, null_value: float = 0.0) -> dict:
    r = cluster_bootstrap_statistic(frame, statistic, cluster_col="test_year", n_boot=N_BOOT, seed=0,
                                    null_value=null_value, cluster_unit="rolling origin")
    return {"estimate": r["estimate"], "ci_95": [round(v, 6) for v in r["ci_95"]] if r["ci_95"] else None,
            "excludes_null": r["significant_at_0.05"], "p_value": r.get("p_value"), "n_origins": r["n_clusters"]}


def mean_of(col: str):
    return lambda f: float(f[col].mean()) if len(f) else float("nan")


def contrast(col: str, a: str, b: str):
    def stat(f: pd.DataFrame) -> float:
        ga, gb = f[f.method == a], f[f.method == b]
        return float(ga[col].mean() - gb[col].mean()) if len(ga) and len(gb) else float("nan")
    return stat


def year_weighted(sub: pd.DataFrame) -> dict:
    per = sub.groupby("test_year").unsafe.mean()
    years = np.arange(2008, 2024)
    rng = np.random.default_rng(YW_SEED)
    boot = []
    for _ in range(N_BOOT):
        v = [per[y] for y in rng.choice(years, size=len(years), replace=True) if y in per.index]
        if v:
            boot.append(float(np.mean(v)))
    return {"estimate": round(float(per.mean()), 4),
            "ci_95": [round(float(np.percentile(boot, q)), 3) for q in (2.5, 97.5)],
            "median_over_origins": round(float(per.median()), 4), "origins": int(len(per)),
            "origins_above_0.05": int((per > 0.05).sum())}


def summarise_run(r: pd.DataFrame) -> dict:
    out: dict = {}
    for m in [m for m in METHODS if m in set(r.method)]:
        rm = r[r.method == m]
        cls = {}
        for s in ("Normal", "Moderate", "Extreme", "ALL"):
            sub = rm if s == "ALL" else rm[rm.stratum == s]
            cls[s] = {"rows": int(len(sub)), "picp": ci(sub, mean_of("covered")),
                      "unsafe": ci(sub, mean_of("unsafe"), null_value=0.05),
                      "mpiw": round(float(sub.width.mean()), 4)}
        cls["Extreme"]["unsafe_year_weighted"] = year_weighted(rm[rm.stratum == "Extreme"])
        out[m] = cls
    if {"rolling_static_cp", "group_conditional_cp"} <= set(r.method):
        ex, nrm = r[r.stratum == "Extreme"], r[r.stratum == "Normal"]
        out["group_conditional_vs_static"] = {
            "extreme_coverage_gain": ci(ex, contrast("covered", "group_conditional_cp", "rolling_static_cp")),
            "extreme_unsafe_reduction": ci(ex, contrast("unsafe", "rolling_static_cp", "group_conditional_cp")),
            "normal_coverage_change": ci(nrm, contrast("covered", "group_conditional_cp", "rolling_static_cp"))}
    point = r[r.method == "rolling_static_cp"]
    err = point.y - point.pred
    per_year = point.groupby("test_year").apply(
        lambda g: 1 - ((g.y - g.pred) ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum(), include_groups=False)
    out["point_model"] = {"rmse_all_rows": round(float(np.sqrt((err ** 2).mean())), 4),
                          "mean_r2_over_origins": round(float(per_year.mean()), 4),
                          "bias_2012": round(float(err[point.test_year == 2012].mean()), 4)}
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    primary = prep(pd.read_csv(D / "rolling_origin_rows.csv"))
    res: dict = {"plan": "manuscript/SENSITIVITY_PLAN_2026-09-26.md", "runs": {}}

    for key, (label, tag) in RUNS.items():
        path = D / f"{tag}_rows.csv"
        if not path.exists():
            res["runs"][key] = {"label": label, "status": f"missing {path.name}"}
            continue
        r = prep(pd.read_csv(path))
        entry = {"label": label, "tag": tag, "rows": int(len(r) // r.method.nunique())}
        if key == "R0":             # plan: R0's primary calibrators must reproduce the primary run exactly
            diffs = {}
            for m in PRIMARY_METHODS:
                a = primary[primary.method == m].reset_index(drop=True)
                b = r[r.method == m].reset_index(drop=True)
                same = len(a) == len(b) and (a.GEOID.values == b.GEOID.values).all()
                diffs[m] = float(max(np.abs(a.lo - b.lo).max(), np.abs(a.hi - b.hi).max())) if same else None
            entry["reproduces_primary"] = {"max_abs_difference_by_method": diffs,
                                           "exact": all(v == 0.0 for v in diffs.values())}
        entry.update(summarise_run(r))
        res["runs"][key] = entry

    # exposure-threshold sensitivity, primary rows, static calibration
    panel = pd.read_csv(ROOT / "outputs_master" / "Paper3_Processed.csv", usecols=["GEOID", "Year", "CDHW_Event_Count"])
    s = primary[primary.method == "rolling_static_cp"].merge(
        panel, left_on=["GEOID", "test_year"], right_on=["GEOID", "Year"], how="left")
    thr = {}
    for k in (1, 3, 5):
        sub = s[s.CDHW_Event_Count >= k]
        thr[f">={k}"] = {"rows": int(len(sub)), "unsafe": ci(sub, mean_of("unsafe"), null_value=0.05),
                         "unsafe_year_weighted": year_weighted(sub),
                         "picp": ci(sub, mean_of("covered"))}
    res["threshold_sensitivity_static"] = thr
    res["threshold_sensitivity_static"]["missing_day_counts"] = int(s.CDHW_Event_Count.isna().sum())

    (D / "sensitivity_rolling.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    def fmt(c: dict) -> str:
        return f"{c['estimate']:.4f} {c['ci_95']}" if c and c.get("ci_95") else str(c.get("estimate") if c else c)
    for key, e in res["runs"].items():
        print(f"\n{key}: {e['label']}")
        if "status" in e:
            print("  ", e["status"]); continue
        if "reproduces_primary" in e:
            print("   reproduces primary exactly:", e["reproduces_primary"]["exact"])
        for m in METHODS:
            if m in e:
                x = e[m]
                print(f"   {m:24s} Extreme UMR {fmt(x['Extreme']['unsafe'])} | year-wtd "
                      f"{x['Extreme']['unsafe_year_weighted']['estimate']} {x['Extreme']['unsafe_year_weighted']['ci_95']}"
                      f" | ALL PICP {fmt(x['ALL']['picp'])} | ALL UMR {x['ALL']['unsafe']['estimate']}")
        if "group_conditional_vs_static" in e:
            g = e["group_conditional_vs_static"]
            print(f"   group-cond gain {fmt(g['extreme_coverage_gain'])} | Normal change {fmt(g['normal_coverage_change'])}")
        print("   point model:", e["point_model"])
    print("\nthresholds:", {k: (v["rows"], fmt(v["unsafe"])) for k, v in thr.items() if isinstance(v, dict)})
    print(f"written: {D / 'sensitivity_rolling.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
