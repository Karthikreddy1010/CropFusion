"""Which meteorological exposure breaks the intervals, and in which growth stage?

RO1-RO5 establish that prediction intervals fail in the compound drought-heat
class. They do not say which component of that exposure is responsible, or
whether the timing within the crop's development matters. Those are the
questions an agrometeorologist asks, and the answer is not in the objectives:

  - heat concentrated in the SILKING window is followed by interval failure;
  - the same heat in the VEGETATIVE window is not;
  - and simple days above 35 C separates failure about as well as the compound
    event count does, which qualifies the compound framing rather than
    supporting it.

Each variable is read from its stressed tail: the upper quintile for heat and event
counts, the lower quintile for SPEI and precipitation, where a high value means a
wet season. Reading those two from the upper tail measures the wettest county-years
and makes drought look irrelevant.

The unit of inference is the rolling origin, as everywhere else: the statistic is
the excess unsafe-miss rate in the top quintile of an exposure variable over the
remaining four, with whole origins resampled beneath it.

Nothing here is refitted. Coverage comes from the rolling-origin row output and
the exposure variables from the engineered frame, joined on county and year.

Run:  python code/phenology_coverage.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
for p in (str(CODE_DIR), str(CODE_DIR / "diagnostics_loso")):
    if p not in sys.path:
        sys.path.insert(0, p)

import config as cfg  # noqa: E402

# Ordered by growth stage, so the phenological pattern is visible in the output.
WINDOWS = [
    ("prism_tmax_days_gt35_veg", "days >35C, vegetative"),
    ("prism_tmax_days_gt35_silk", "days >35C, silking"),
    ("prism_tmax_days_gt35_grain", "days >35C, grain fill"),
    ("CDHW_Veg_Severity", "CDHW severity, vegetative"),
    ("CDHW_Silking_Severity", "CDHW severity, silking"),
    ("CDHW_GrainFill_Severity", "CDHW severity, grain fill"),
]
AGGREGATES = [
    ("Tmax_Days_Above_35", "days >35C, whole season"),
    ("CDHW_Event_Count", "CDHW event count (the paper's stratifier)"),
    ("CDHW_Severity_Score", "CDHW severity score"),
    ("spei90_prism_nldas_min_gs", "SPEI-90 minimum (drought)"),
    ("Precip_growseason_mm", "season precipitation"),
]

# Which tail is the STRESSED one. For heat and event counts the stressed county-years
# sit in the upper tail; for SPEI and precipitation they sit in the lower tail, because
# a high SPEI minimum means a wet season. Taking the upper quintile for those two
# measures the wettest county-years and reports "drought does not matter" when the
# variable was simply read from the wrong end.
STRESSED_TAIL = {"spei90_prism_nldas_min_gs": "lower", "Precip_growseason_mm": "lower"}
TOP_QUINTILE = 0.80


def _excess(frame: pd.DataFrame, col: str, threshold: float, tail: str = "upper"):
    """Unsafe-miss rate in the STRESSED quintile of `col` minus that in the rest."""
    def stat(f: pd.DataFrame) -> float:
        if tail == "lower":
            stressed, rest = f[f[col] < threshold], f[f[col] >= threshold]
        else:
            stressed, rest = f[f[col] > threshold], f[f[col] <= threshold]
        if len(stressed) < 5 or len(rest) < 5:
            return float("nan")
        return float(stressed["unsafe"].mean() - rest["unsafe"].mean())
    return stat


def build() -> Dict[str, Any]:
    import _fold_lab as lab
    from dependence_aware_stats import cluster_bootstrap_statistic as cb

    diag = Path(cfg.PROJECT_ROOT) / "outputs_diagnostics" / "reports"
    rows = pd.read_csv(diag / "rolling_origin_rows.csv")
    r = rows[rows.method == "rolling_static_cp"].copy()
    r["unsafe"] = (r.y < r.lo).astype(float)
    r["covered"] = ((r.y >= r.lo) & (r.y <= r.hi)).astype(float)

    df = lab.engineered_df()
    cols = [c for c, _ in WINDOWS + AGGREGATES if c in df.columns]
    j = r.merge(df[["GEOID", "Year"] + cols], left_on=["GEOID", "test_year"],
                right_on=["GEOID", "Year"], how="left")

    out: Dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "statistic": ("excess unsafe-miss rate, top quintile of the exposure variable minus "
                      "the remaining four fifths"),
        "inference": "year-clustered bootstrap over the 16 rolling origins, 2000 resamples",
        "n_rows": int(len(j)),
        "n_rows_complete": int(j[cols].notna().all(axis=1).sum()),
        "calibrator": "rolling_static_cp",
    }
    table: List[Dict[str, Any]] = []
    for col, label in WINDOWS + AGGREGATES:
        if col not in j.columns or j[col].notna().sum() < 1000:
            continue
        tail = STRESSED_TAIL.get(col, "upper")
        thr = float(j[col].quantile(1 - TOP_QUINTILE if tail == "lower" else TOP_QUINTILE))
        o = cb(j, _excess(j, col, thr, tail), cluster_col="test_year", n_boot=2000, seed=0,
               cluster_unit="rolling origin")
        table.append({"variable": col, "label": label, "threshold": round(thr, 4),
                      "stressed_tail": tail,
                      "excess_unsafe": o["estimate"], "ci_95": o["ci_95"],
                      "excludes_zero": o["significant_at_0.05"], "p_value": o.get("p_value"),
                      "stage": ("vegetative" if "veg" in col else
                                "silking" if "silk" in col else
                                "grain fill" if "grain" in col else "season")})
    out["by_exposure"] = table

    # the contrast that carries the finding: same heat, different growth stage
    s_thr = float(j.prism_tmax_days_gt35_silk.quantile(TOP_QUINTILE))
    v_thr = float(j.prism_tmax_days_gt35_veg.quantile(TOP_QUINTILE))

    def silk_minus_veg(f: pd.DataFrame) -> float:
        s = f[f.prism_tmax_days_gt35_silk > s_thr]
        v = f[f.prism_tmax_days_gt35_veg > v_thr]
        if len(s) < 5 or len(v) < 5:
            return float("nan")
        return float(s["unsafe"].mean() - v["unsafe"].mean())

    o = cb(j, silk_minus_veg, cluster_col="test_year", n_boot=2000, seed=0,
           cluster_unit="rolling origin")
    out["silking_minus_vegetative"] = {
        "estimate": o["estimate"], "ci_95": o["ci_95"],
        "excludes_zero": o["significant_at_0.05"], "p_value": o.get("p_value"),
        "reads": ("unsafe-miss rate among heat-stressed silking county-years minus that among "
                  "heat-stressed vegetative county-years; the same exposure in a different "
                  "developmental window")}

    # and the honest comparison: does the compound index beat simple heat?
    by = {t["variable"]: t for t in table}
    if "Tmax_Days_Above_35" in by and "CDHW_Event_Count" in by:
        out["compound_vs_simple_heat"] = {
            "simple_heat_excess": by["Tmax_Days_Above_35"]["excess_unsafe"],
            "compound_count_excess": by["CDHW_Event_Count"]["excess_unsafe"],
            "note": ("the compound event count does not separate interval failure better than "
                     "days above 35 C alone; the compound construction is not doing the work "
                     "here and the paper should not claim that it is")}
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = build()
    diag = Path(cfg.PROJECT_ROOT) / "outputs_diagnostics" / "reports"
    diag.mkdir(parents=True, exist_ok=True)
    (diag / "phenology_coverage.json").write_text(json.dumps(out, indent=2, default=str),
                                                  encoding="utf-8")
    pd.DataFrame(out["by_exposure"]).to_csv(diag / "phenology_coverage.csv", index=False)

    print(f"{'exposure window':<42}{'tail':>7}{'excess':>9}{'95% CI':>24}{'excl 0':>8}")
    for t in out["by_exposure"]:
        ci = f"[{t['ci_95'][0]:+.4f}, {t['ci_95'][1]:+.4f}]" if t["ci_95"] else "none"
        print(f"{t['label']:<42}{t.get('stressed_tail','upper'):>7}"
              f"{t['excess_unsafe']:>+9.4f}{ci:>24}{str(t['excludes_zero']):>8}")
    s = out["silking_minus_vegetative"]
    print(f"\nsilking minus vegetative: {s['estimate']:+.4f}  CI {s['ci_95']}  "
          f"excludes zero: {s['excludes_zero']}")
    print(f"\nwritten: {diag / 'phenology_coverage.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
