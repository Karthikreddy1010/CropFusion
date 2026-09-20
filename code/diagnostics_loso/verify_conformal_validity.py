"""Validity check for the conformal baselines (RO2 rests on these).

`group_conditional_cp` and `rolling_static_cp` were written for this project and
so far have only been shown to behave plausibly on real data. Plausible is not
valid. Under exchangeability a split-conformal interval at level 1-alpha must
cover at least 1-alpha of future points, and a Mondrian (group-conditional)
interval must do so *within each group*. That is a property with a known answer,
so it can be tested rather than assumed.

Design: exchangeable synthetic data, heteroscedastic by group so the groups
genuinely need different thresholds, deliberately miscalibrated base intervals
(the conformal step must repair them), repeated over many draws.

Run:  python code/diagnostics_loso/verify_conformal_validity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import baselines as bl  # noqa: E402

ALPHA = 0.10
N_REPS = 400
N_CAL, N_TEST = 600, 600
GROUP_SCALE = {"A": 1.0, "B": 2.5, "C": 5.0}   # B and C need much wider intervals


def draw(rng, n):
    """Exchangeable draw: group label, outcome, and a deliberately poor interval."""
    groups = rng.choice(list(GROUP_SCALE), size=n, p=[0.6, 0.3, 0.1])
    scale = np.array([GROUP_SCALE[g] for g in groups])
    y = rng.normal(10.0, scale)
    # Base "model" interval: same width everywhere and far too narrow. Conformal
    # calibration has to fix both the level and (for Mondrian) the group spread.
    q_lo, q_hi = np.full(n, 10.0 - 0.5), np.full(n, 10.0 + 0.5)
    return groups, y, q_lo, q_hi


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rng = np.random.default_rng(0)

    marg_static, marg_group = [], []
    per_group = {g: [] for g in GROUP_SCALE}

    for _ in range(N_REPS):
        g_cal, y_cal, lo_cal, hi_cal = draw(rng, N_CAL)
        g_te, y_te, lo_te, hi_te = draw(rng, N_TEST)

        lo_s, hi_s, _ = bl.rolling_static_cp(y_cal, lo_cal, hi_cal, lo_te, hi_te, alpha=ALPHA)
        marg_static.append(float(((y_te >= lo_s) & (y_te <= hi_s)).mean()))

        lo_g, hi_g, meta = bl.group_conditional_cp(
            y_cal, lo_cal, hi_cal, g_cal, lo_te, hi_te, g_te, alpha=ALPHA)
        cov_g = (y_te >= lo_g) & (y_te <= hi_g)
        marg_group.append(float(cov_g.mean()))
        for g in GROUP_SCALE:
            m = g_te == g
            if m.any():
                per_group[g].append(float(cov_g[m].mean()))

    target = 1.0 - ALPHA
    checks = []

    ms, mg = float(np.mean(marg_static)), float(np.mean(marg_group))
    print(f"\ntarget coverage {target:.2f}, {N_REPS} replications\n")
    print(f"rolling_static_cp    marginal {ms:.4f}")
    print(f"group_conditional_cp marginal {mg:.4f}")
    checks.append(("split conformal is marginally valid", ms >= target - 0.01))
    checks.append(("group-conditional is marginally valid", mg >= target - 0.01))

    print("\ncoverage WITHIN each group (this is what Mondrian buys):")
    print(f"{'group':<8}{'scale':>7}{'static':>10}{'groupCP':>10}")
    for g in GROUP_SCALE:
        # static coverage per group, recomputed on the last replication's scale
        gc = float(np.mean(per_group[g]))
        print(f"{g:<8}{GROUP_SCALE[g]:>7.1f}{'-':>10}{gc:>10.4f}")
        checks.append((f"group {g} reaches target within 2 points", gc >= target - 0.02))

    # The fallback path must trigger when a group is too small to calibrate.
    g_cal, y_cal, lo_cal, hi_cal = draw(rng, 60)
    g_cal[:] = "A"
    g_cal[:5] = "C"                                   # only 5 calibration rows for C
    g_te, y_te, lo_te, hi_te = draw(rng, 100)
    _, _, meta = bl.group_conditional_cp(y_cal, lo_cal, hi_cal, g_cal,
                                         lo_te, hi_te, g_te, alpha=ALPHA, min_group_n=20)
    fell_back = meta["groups"].get("C", {}).get("fell_back_to_pooled")
    print(f"\nsmall group (n_cal=5) fell back to pooled: {fell_back}")
    checks.append(("undersized group falls back to the pooled threshold", fell_back is True))

    failed = 0
    print()
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        failed += 0 if ok else 1
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
