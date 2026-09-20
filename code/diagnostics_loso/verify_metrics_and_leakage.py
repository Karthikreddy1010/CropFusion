"""Close two gaps: the consequence-metric arithmetic, and a real leakage probe.

Part 1 — metric arithmetic (RO4 rests on it).
    `coverage_metrics` and `decision_impact.impact` are simple formulas, which is
    exactly where a sign error survives inspection. Both are checked against a
    four-row fixture whose answers are computed by hand in the comments below.

Part 2 — target-permutation leakage probe (moves the fit-only discipline from
    "read the code" to "tested").
    The features, scaler, selection and imputation are all fitted on FIT. If any
    DEV/CAL/TEST information reached them, a model trained on FIT with the FIT
    labels PERMUTED could still track the test outcome. With an honest pipeline
    that R² must collapse to approximately zero. The real pipeline reaches about
    0.70 on the same split, so the two numbers are far apart if all is well.

Run:  python code/diagnostics_loso/verify_metrics_and_leakage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
for p in (str(CODE_DIR), str(CODE_DIR / "diagnostics_loso")):
    if p not in sys.path:
        sys.path.insert(0, p)

import config as cfg  # noqa: E402


# ─────────────────────────────────────────────────────────────
# Part 1: metric arithmetic against a hand-computed fixture
# ─────────────────────────────────────────────────────────────

def fixture():
    """Four rows, alpha = 0.10, worked out by hand.

        row  y     lo    hi    outcome
        1    10.0   9.0  11.0  covered
        2     9.0   9.5  11.0  BELOW  -> shortfall 0.5
        3     8.0   7.0   9.0  covered
        4    12.0   9.0  11.0  above  -> no shortfall

    picp                = 2/4                      = 0.50
    unsafe_miss_rate    = 1/4                      = 0.25
    mean_shortfall      = 0.5/1                    = 0.50
    worst_shortfall     = 0.5
    expected_shortfall  = (0 + 0.5 + 0 + 0)/4      = 0.125
    over_rate           = 1/4                      = 0.25
    mpiw                = (2.0 + 1.5 + 2.0 + 2.0)/4 = 1.875
    winkler: covered rows contribute their width; row 2 adds (2/0.1)*0.5 = 10;
             row 4 adds (2/0.1)*1.0 = 20
             = (2.0 + (1.5 + 10) + 2.0 + (2.0 + 20)) / 4 = 37.5/4 = 9.375
    """
    y = np.array([10.0, 9.0, 8.0, 12.0])
    lo = np.array([9.0, 9.5, 7.0, 9.0])
    hi = np.array([11.0, 11.0, 9.0, 11.0])
    expected = {
        "picp": 0.50, "unsafe_miss_rate": 0.25, "mean_shortfall_t_ha": 0.50,
        "worst_shortfall_t_ha": 0.50, "expected_shortfall_t_ha": 0.125,
        "mpiw": 1.875, "winkler": 9.375, "n_unsafe": 1,
    }
    return y, lo, hi, expected


def check_metrics() -> list:
    import rolling_origin as ro
    import decision_impact as di
    import pandas as pd

    y, lo, hi, exp = fixture()
    alpha = 0.10
    got = ro.coverage_metrics(y, lo, hi, alpha)

    out = []
    for k, want in exp.items():
        have = got.get(k)
        ok = have is not None and abs(float(have) - want) < 1e-9
        out.append((f"rolling_origin.coverage_metrics[{k}] == {want}", ok,
                    f"got {have}"))

    # decision_impact.impact works on a frame and must agree where they overlap
    frame = pd.DataFrame({"Observed_Yield": y, "Lower_Interval": lo, "Upper_Interval": hi})
    d = di.impact(frame, alpha)
    for k in ("picp", "unsafe_miss_rate", "mean_shortfall_t_ha",
              "worst_shortfall_t_ha", "expected_shortfall_t_ha", "mpiw"):
        ok = abs(float(d[k]) - exp[k]) < 1e-9
        out.append((f"decision_impact.impact[{k}] agrees", ok, f"got {d[k]}"))
    out.append(("decision_impact reports the over-bound rate", abs(d["over_rate"] - 0.25) < 1e-9,
                f"got {d['over_rate']}"))
    out.append(("unsafe target is alpha/2", abs(d["unsafe_target"] - 0.05) < 1e-9,
                f"got {d['unsafe_target']}"))
    return out


# ─────────────────────────────────────────────────────────────
# Part 2: target-permutation leakage probe
# ─────────────────────────────────────────────────────────────

def check_leakage() -> list:
    import lightgbm as lgb
    import _fold_lab as lab

    data = lab.build_temporal()
    X_fit, X_test = data["X"]["fit"], data["X"]["test"]
    y_fit, y_test = data["y_detrended"]["fit"], data["y_detrended"]["test"]

    def r2(y_true, pred):
        ss_res = float(np.sum((y_true - pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        return 1.0 - ss_res / max(1e-9, ss_tot)

    honest = lgb.LGBMRegressor(**cfg.LGBM_PARAMS).fit(X_fit, y_fit)
    r2_honest = r2(y_test, honest.predict(X_test))

    rng = np.random.default_rng(0)
    r2_perm = []
    for _ in range(3):
        y_shuf = rng.permutation(y_fit)
        m = lgb.LGBMRegressor(**cfg.LGBM_PARAMS).fit(X_fit, y_shuf)
        r2_perm.append(r2(y_test, m.predict(X_test)))
    r2_perm_mean = float(np.mean(r2_perm))

    print(f"\nTEST R2 with honest FIT labels   : {r2_honest:+.4f}")
    print(f"TEST R2 with permuted FIT labels : {r2_perm_mean:+.4f} "
          f"(runs: {', '.join(f'{v:+.4f}' for v in r2_perm)})")

    return [
        ("permuted-label model has no predictive power (R2 < 0.05)", r2_perm_mean < 0.05,
         f"{r2_perm_mean:+.4f}"),
        ("honest model is far better than permuted (gap > 0.5)",
         (r2_honest - r2_perm_mean) > 0.5, f"gap {r2_honest - r2_perm_mean:+.4f}"),
    ]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    checks = check_metrics()
    print("\nPart 1 — consequence-metric arithmetic")
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<58}{detail}")

    print("\nPart 2 — target-permutation leakage probe")
    leak = check_leakage()
    for name, ok, detail in leak:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<58}{detail}")

    all_checks = checks + leak
    failed = sum(0 if ok else 1 for _, ok, _ in all_checks)
    print(f"\n{len(all_checks) - failed}/{len(all_checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
