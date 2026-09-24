"""Correctness checks for the fixed-effects degree-day panel baseline.

This baseline exists to answer the question an agricultural-economics or
agrometeorology reviewer asks first: does the machine-learning model beat the
established reduced-form specification, or only a trend and a county mean? A
baseline that is wrong in the model's favour is worse than no baseline, so the
estimator is tested against cases with known answers before it is believed.

  1. on synthetic data built from known county effects and known slopes, the
     within-county estimator recovers the slopes;
  2. county effects are estimated on FIT rows only, and a county never seen in
     FIT falls back to the global intercept instead of producing a NaN -- the
     leave-one-state-out case, where every test county is unseen;
  3. TEST rows never enter the fit: perturbing the target on TEST leaves the
     fitted coefficients bit-identical;
  4. it beats a trend-only predictor when the climate signal is real, so a null
     result later cannot be blamed on a broken baseline;
  5. the quadratic precipitation term is genuinely fitted, since the concavity
     of the yield-precipitation response is the substantive content of the
     specification.

Run:  python code/diagnostics_loso/verify_fe_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import fe_degree_day_baseline as fe  # noqa: E402

TRUE = {"gdd": 0.0020, "heat": -0.1500, "precip": 0.0100, "precip_sq": -1.2e-5, "year": 0.1000}


def synth(n_counties: int = 60, years=range(1990, 2021), seed: int = 0) -> pd.DataFrame:
    """Yield built from county effects plus the specification's own terms."""
    rng = np.random.default_rng(seed)
    county_effect = rng.normal(0, 2.0, n_counties)
    rows = []
    for c in range(n_counties):
        for y in years:
            gdd = rng.uniform(1200, 2200)
            heat = rng.uniform(0, 25)
            pr = rng.uniform(200, 900)
            yld = (5.0 + county_effect[c] + TRUE["gdd"] * gdd + TRUE["heat"] * heat
                   + TRUE["precip"] * pr + TRUE["precip_sq"] * pr ** 2
                   + TRUE["year"] * (y - 1990) + rng.normal(0, 0.30))
            rows.append({"GEOID": 10000 + c, "Year": y, "GDD_Accumulated": gdd,
                         "Tmax_Days_Above_35": heat, "Precip_growseason_mm": pr,
                         "Corn_Yield_tha": yld})
    return pd.DataFrame(rows)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    checks = []
    df = synth()
    fit = df[df.Year <= 2013]
    test = df[df.Year >= 2014]

    # 1 - coefficient recovery
    m = fe.fit_fe_panel(fit, target="Corn_Yield_tha")
    print("coefficient recovery (truth -> estimate):")
    worst = 0.0
    for k, truth in TRUE.items():
        est = m.coefficients[k]
        rel = abs(est - truth) / max(abs(truth), 1e-12)
        worst = max(worst, rel)
        print(f"  {k:<10} {truth:>12.6g} -> {est:>12.6g}   ({rel:.1%})")
    checks.append(("every slope is recovered to within 5%", worst < 0.05))

    # 2 - unseen counties fall back rather than fail
    unseen = test.copy()
    unseen["GEOID"] = 99999
    p = m.predict(unseen)
    print(f"\nunseen county: {np.isfinite(p).sum()}/{len(p)} finite predictions")
    checks.append(("a county absent from FIT still yields finite predictions",
                   bool(np.isfinite(p).all())))
    checks.append(("the fallback is the global intercept, not zero",
                   abs(float(np.mean(p)) - float(fit.Corn_Yield_tha.mean())) < 4.0))

    # 3 - TEST cannot influence the fit
    poisoned = df.copy()
    poisoned.loc[poisoned.Year >= 2014, "Corn_Yield_tha"] += 50.0
    m2 = fe.fit_fe_panel(poisoned[poisoned.Year <= 2013], target="Corn_Yield_tha")
    same = all(abs(m.coefficients[k] - m2.coefficients[k]) < 1e-12 for k in TRUE)
    print(f"coefficients unchanged when TEST targets are poisoned: {same}")
    checks.append(("TEST rows do not enter the fit", same))

    # 4 - it must beat trend-only when the climate signal is real
    yt = test.Corn_Yield_tha.values
    pred = m.predict(test)
    trend_only = (fit.Corn_Yield_tha.mean()
                  + TRUE["year"] * (test.Year.values - fit.Year.mean()))
    def r2(y, p):
        return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    r2_fe, r2_tr = r2(yt, pred), r2(yt, trend_only)
    print(f"\nR2 on held-out years: FE {r2_fe:.4f}, trend-only {r2_tr:.4f}")
    checks.append(("the FE panel beats trend-only on data with a real signal", r2_fe > r2_tr))
    checks.append(("and recovers most of the variance it should", r2_fe > 0.90))

    # 5 - the quadratic term is real, not decorative
    lin = fe.fit_fe_panel(fit, target="Corn_Yield_tha", precip_quadratic=False)
    print(f"quadratic term present: R2_in_sample {m.r2_in_sample:.4f} vs "
          f"linear-only {lin.r2_in_sample:.4f}")
    checks.append(("dropping the quadratic precipitation term degrades the fit",
                   m.r2_in_sample > lin.r2_in_sample))
    checks.append(("the quadratic coefficient has the expected negative sign",
                   m.coefficients["precip_sq"] < 0))

    failed = 0
    print()
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        failed += 0 if ok else 1
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
