"""Verify the C5 fix: SA-ACI must widen, not narrow, under severity.

The committed implementation divided the conformal threshold by a severity
weight >= 1, so rows under the heaviest compound drought-heat stress received
the NARROWEST intervals. This script trains nothing: it feeds the committed and
the corrected calibrator the same synthetic calibration/test arrays and compares
interval width across severity strata.

Expected:
  legacy   width(severe) <  width(normal)   -- the defect
  fixed    width(severe) >  width(normal)   -- severity widens
  disabled width(severe) == width(normal)   -- switch really switches off

Run:  python code/diagnostics_loso/verify_c5_saaci.py
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import config as cfg  # noqa: E402


def load_legacy():
    src = subprocess.run(["git", "show", "HEAD:code/aci_calibrator.py"],
                         cwd=REPO_DIR, capture_output=True, text=True, check=True).stdout
    d = Path(tempfile.mkdtemp(prefix="c5_legacy_"))
    (d / "aci_calibrator_legacy.py").write_text(src, encoding="utf-8")
    sys.path.insert(0, str(d))
    return importlib.import_module("aci_calibrator_legacy")


def make_case(seed: int = 0):
    """Calibration and test arrays that behave like the real pipeline.

    Two properties matter and both are taken from the actual run:

    * The raw CQR heads UNDER-cover before calibration (DEV PICP ~0.24), so the
      conformity scores are positive and the ACI threshold is positive. Scaling
      a negative threshold would invert the effect, so a test built on
      over-covering intervals measures the wrong thing.
    * Interval width grows with severity, because the quantile heads are
      heteroscedastic. That is the local-scale model the normalised score
      construction assumes.
    """
    rng = np.random.default_rng(seed)
    n_cal, per_year = 400, 200
    years = np.array([2019, 2020, 2021, 2022, 2023])

    def block(sev, n):
        sigma = 1.6 + 0.30 * sev                    # heteroscedastic by severity
        pred = 9.0 - 0.25 * sev
        y = rng.normal(pred, sigma)
        half = 0.9 * sigma                          # ~63% coverage -> positive scores
        return y, pred - half, pred + half

    sev_cal = rng.gamma(1.5, 1.0, n_cal)
    y_cal, q_lo_cal, q_hi_cal = block(sev_cal, n_cal)

    y_t, lo_t, hi_t, yr_t, sev_t = [], [], [], [], []
    for i, yr in enumerate(years):
        sev = rng.gamma(1.0 + 1.5 * i, 1.2, per_year)   # severity grows with year
        y, lo, hi = block(sev, per_year)
        y_t.append(y); lo_t.append(lo); hi_t.append(hi)
        yr_t.append(np.full(per_year, yr)); sev_t.append(sev)

    return dict(
        y_cal=y_cal, q_lo_cal=q_lo_cal, q_hi_cal=q_hi_cal, cdhw_severity_cal=sev_cal,
        y_test=np.concatenate(y_t), q_lo_test=np.concatenate(lo_t), q_hi_test=np.concatenate(hi_t),
        years_test=np.concatenate(yr_t), cdhw_severity_test=np.concatenate(sev_t),
        point_preds_test=np.concatenate([np.full(per_year, np.nan)] * len(years)),
    )


def added_width_by_severity(res, raw_width, sev, years):
    """How much width the calibrator ADDS, high- vs low-severity, within a year.

    The raw quantile heads are already heteroscedastic, so total width rises
    with severity under every variant. What C5 is about is the calibrator's own
    contribution: added_i = calibrated_width_i - raw_width_i, which equals
    2 * threshold / w_i in the committed code and 2 * threshold * w_i after the
    fix. Comparing that isolates the mechanism from the model's own spread.
    """
    added = (res.q_hi - res.q_lo) - raw_width
    lows, highs, ratios = [], [], []
    for yr in np.unique(years):
        m = years == yr
        s_y, a_y = sev[m], added[m]
        hi = s_y >= np.quantile(s_y, 0.75)
        lo = s_y <= np.quantile(s_y, 0.25)
        lows.append(a_y[lo].mean()); highs.append(a_y[hi].mean())
        ratios.append(a_y[hi].mean() / a_y[lo].mean())
    picp = float(np.mean((res.y_true >= res.q_lo) & (res.y_true <= res.q_hi)))
    return float(np.mean(lows)), float(np.mean(highs)), float(np.mean(ratios)), picp


def main() -> int:
    case = make_case()
    sev = case["cdhw_severity_test"]
    raw_w = case["q_hi_test"] - case["q_lo_test"]

    import aci_calibrator as fixed
    legacy = load_legacy()

    rows = []
    r = legacy.severity_aware_adaptive_conformal(**case)
    rows.append(("legacy (divide)",) + added_width_by_severity(r, raw_w, sev, case["years_test"]))

    cfg.ACI_SEVERITY_WEIGHTING = True
    importlib.reload(fixed)
    r = fixed.severity_aware_adaptive_conformal(**case)
    rows.append(("fixed (normalised)",) + added_width_by_severity(r, raw_w, sev, case["years_test"]))
    fixed_meta = r.metadata

    cfg.ACI_SEVERITY_WEIGHTING = False
    importlib.reload(fixed)
    r = fixed.severity_aware_adaptive_conformal(**case)
    rows.append(("severity off",) + added_width_by_severity(r, raw_w, sev, case["years_test"]))
    off_meta = r.metadata
    cfg.ACI_SEVERITY_WEIGHTING = True
    importlib.reload(fixed)

    print("\n" + "=" * 78)
    print(f"{'variant':<20}{'added low-sev':>15}{'added high-sev':>16}{'ratio':>9}{'PICP':>9}")
    for name, w_lo, w_hi, ratio, picp in rows:
        print(f"{name:<20}{w_lo:>15.3f}{w_hi:>16.3f}{ratio:>9.3f}{picp:>9.3f}")
    print("=" * 78)

    thresholds = [h["threshold"] for h in fixed_meta["year_history"]]
    l_ratio, f_ratio, o_ratio = rows[0][3], rows[1][3], rows[2][3]
    print(f"year thresholds (fixed): {[round(t, 3) for t in thresholds]}")
    checks = [
        ("thresholds are positive (scaling widens, as in the real run)",
         all(t > 0 for t in thresholds)),
        ("legacy narrows under severity (the defect)", l_ratio < 1.0),
        ("fixed widens under severity", f_ratio > 1.0),
        ("switch off adds the same width regardless of severity", abs(o_ratio - 1.0) < 0.02),
        ("metadata records the direction", fixed_meta["severity_direction"] == "widen"
         and off_meta["severity_direction"] == "none"),
    ]
    failed = 0
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        failed += 0 if ok else 1
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
