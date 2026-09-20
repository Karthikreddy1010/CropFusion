"""Translate interval coverage into consequence (audit follow-up, Addition 2).

Coverage is abstract: 0.876 in the extreme class does not tell an agronomist or
an insurer what went wrong. What matters for risk is the one-sided failure —
the observed yield falling BELOW the stated lower bound, i.e. the loss being
worse than the interval admitted — and by how much.

Metrics, per method and per CDHW exposure class:

    unsafe_miss_rate      P(y < lower). For a 90% two-sided interval the
                          calibrated value is 0.05; anything above means the
                          downside was understated more often than advertised.
    mean_shortfall        mean(lower - y) among unsafe misses, in t/ha
    worst_shortfall       the largest single shortfall
    expected_shortfall    mean(max(lower - y, 0)) over ALL rows, so rate and
                          magnitude combine into one number
    over_rate             P(y > upper), reported for symmetry; harmless for risk

Run:  python code/decision_impact.py
Writes decision_impact_{by_method,by_stratum,by_year}.csv and a markdown summary.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
REPO = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import config as cfg  # noqa: E402

OUT_DIR = REPO / "outputs_diagnostics" / "reports"
PRED = REPO / "outputs_master" / "reports" / "predictions.csv"
PROC = REPO / "outputs_master" / "Paper3_Processed.csv"


def load_joined() -> pd.DataFrame:
    """Predictions joined to their CDHW exposure class, by position.

    predictions.csv carries a county NAME and no state, and names repeat across
    the six states ("Adams" exists in several), so the name is not a key. Joining
    on name + year + rounded yield matched only 95.7% of rows, which is not good
    enough to build a risk statement on.

    Instead each method block is aligned positionally with the rebuilt TEST
    partition -- the alignment p8_review_evidence.py established -- and the
    alignment is then VERIFIED by comparing the observed yields row by row. If
    they disagree, this refuses to produce numbers.
    """
    lab_dir = CODE_DIR / "diagnostics_loso"
    if str(lab_dir) not in sys.path:
        sys.path.insert(0, str(lab_dir))
    import _fold_lab as lab

    pred = pd.read_csv(PRED)
    test = lab.build_temporal()["frames"]["test"].reset_index(drop=True)
    print(f"rebuilt TEST partition: {len(test)} rows, {test.Year.min()}-{test.Year.max()}")

    blocks = []
    for method, g in pred.groupby("Conformal_Method", sort=False):
        g = g.reset_index(drop=True)
        if len(g) != len(test):
            raise SystemExit(f"{method}: {len(g)} prediction rows vs {len(test)} partition rows")
        delta = float(np.abs(g["Observed_Yield"].values
                             - test[cfg.PRIMARY_TARGET].values).max())
        if delta > 5e-4:                      # predictions.csv stores 4 decimals
            raise SystemExit(f"{method}: positional alignment failed, max |Δy| = {delta:.6f}")
        blocks.append(g.assign(
            GEOID=test["GEOID"].values,
            State=test["State"].values,
            CDHW_Event_Count=test.get("CDHW_Event_Count", pd.Series(np.zeros(len(test)))).values,
            Year_Type=test["Year_Type"].values,
            _align_max_delta=delta,
        ))
    out = pd.concat(blocks, ignore_index=True)
    print(f"alignment verified for {out.Conformal_Method.nunique()} methods "
          f"(max |Δy| = {out._align_max_delta.max():.2e})")
    return out.drop(columns="_align_max_delta")


def impact(g: pd.DataFrame, alpha: float) -> Dict[str, float]:
    y = g["Observed_Yield"].values
    lo = g["Lower_Interval"].values
    hi = g["Upper_Interval"].values

    unsafe = y < lo
    over = y > hi
    shortfall = np.maximum(lo - y, 0.0)

    return {
        "n": int(len(y)),
        "picp": float(((y >= lo) & (y <= hi)).mean()),
        "unsafe_miss_rate": float(unsafe.mean()),
        "unsafe_target": alpha / 2.0,
        "n_unsafe": int(unsafe.sum()),
        "mean_shortfall_t_ha": float(shortfall[unsafe].mean()) if unsafe.any() else 0.0,
        "worst_shortfall_t_ha": float(shortfall.max()),
        "expected_shortfall_t_ha": float(shortfall.mean()),
        "mean_shortfall_pct_of_yield": (float((shortfall[unsafe] / y[unsafe]).mean() * 100)
                                        if unsafe.any() else 0.0),
        "over_rate": float(over.mean()),
        "mpiw": float((hi - lo).mean()),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    alpha = 1.0 - cfg.NOMINAL_COVERAGE
    df = load_joined()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tables = {}
    for name, keys in (("by_method", ["Conformal_Method"]),
                       ("by_stratum", ["Conformal_Method", "Year_Type"]),
                       ("by_year", ["Conformal_Method", "Year"])):
        rows = [{**dict(zip(keys, k if isinstance(k, tuple) else (k,))), **impact(g, alpha)}
                for k, g in df.groupby(keys)]
        t = pd.DataFrame(rows)
        t.to_csv(OUT_DIR / f"decision_impact_{name}.csv", index=False)
        tables[name] = t

    print(f"\nNominal coverage {cfg.NOMINAL_COVERAGE:.0%}; a calibrated interval fails "
          f"downward {alpha / 2:.0%} of the time.\n")

    m = tables["by_method"].sort_values("unsafe_miss_rate")
    print("ALL TEST ROWS (2019-2023)")
    print(f"{'method':<26}{'unsafe':>8}{'n':>7}{'mean short':>12}{'worst':>8}{'E[short]':>10}")
    for _, r in m.iterrows():
        print(f"{r['Conformal_Method']:<26}{r['unsafe_miss_rate']:>8.3f}{int(r['n_unsafe']):>7}"
              f"{r['mean_shortfall_t_ha']:>12.3f}{r['worst_shortfall_t_ha']:>8.2f}"
              f"{r['expected_shortfall_t_ha']:>10.3f}")

    s = tables["by_stratum"]
    print("\nBY CDHW EXPOSURE CLASS")
    print(f"{'method':<26}{'class':<10}{'n':>6}{'unsafe':>8}{'mean short':>12}{'worst':>8}")
    for _, r in s.sort_values(["Year_Type", "unsafe_miss_rate"]).iterrows():
        print(f"{r['Conformal_Method']:<26}{str(r['Year_Type']):<10}{int(r['n']):>6}"
              f"{r['unsafe_miss_rate']:>8.3f}{r['mean_shortfall_t_ha']:>12.3f}"
              f"{r['worst_shortfall_t_ha']:>8.2f}")

    # The sentence the paper can actually use.
    ext = s[(s.Year_Type == "Extreme")].sort_values("unsafe_miss_rate", ascending=False)
    if len(ext):
        w = ext.iloc[0]
        print(f"\nHeadline: in the extreme-exposure class, {w['Conformal_Method']} left the true "
              f"yield below its stated lower bound for {w['unsafe_miss_rate']:.0%} of county-years "
              f"({int(w['n_unsafe'])} of {int(w['n'])}), understating the loss by "
              f"{w['mean_shortfall_t_ha']:.2f} t/ha on average and up to "
              f"{w['worst_shortfall_t_ha']:.2f} t/ha.")

    yr = tables["by_year"]
    per_year = yr.groupby("Year")[["unsafe_miss_rate", "mean_shortfall_t_ha",
                                   "expected_shortfall_t_ha"]].mean()
    print("\nBY TEST YEAR (mean over the six methods)")
    print(f"{'year':<6}{'unsafe':>8}{'mean short':>12}{'E[short]':>10}")
    for y, r in per_year.iterrows():
        print(f"{int(y):<6}{r['unsafe_miss_rate']:>8.3f}{r['mean_shortfall_t_ha']:>12.3f}"
              f"{r['expected_shortfall_t_ha']:>10.3f}")
    worst = per_year["unsafe_miss_rate"].idxmax()
    print(f"worst test year: {int(worst)} "
          f"({per_year.loc[worst, 'unsafe_miss_rate']:.1%} unsafe misses)")
    print(f"written: {OUT_DIR / 'decision_impact_by_stratum.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
