"""What the random seed can and cannot move in the rolling-origin evaluation.

The paper reports rolling-origin coverage from a single seed. That is normally a
weakness, and the usual answer is to average over several. Here it is not needed,
for a reason that is structural rather than lucky: the intervals are built from
LightGBM quantile learners whose parameters set no ``subsample`` and no
``colsample_bytree``, and LightGBM disables bagging entirely unless
``subsample_freq`` is set. Those learners are therefore deterministic functions
of their training data, so every interval and every coverage figure is
seed-invariant by construction.

That is a claim the paper makes, so it is tested here rather than asserted:

  1. the quantile learners give bit-identical predictions across seeds;
  2. the point learner does NOT -- it draws 80% of features per tree, so the
     --seed flag is live where it is supposed to be, and a run with a different
     seed is a genuinely different point model;
  3. the quantile parameter block in rolling_origin.py still contains no
     stochastic keys. This is the regression guard. If someone adds
     ``colsample_bytree`` to it for a good modelling reason, the determinism
     claim in the results section silently becomes false, and this check fails
     instead.

Interval metrics being seed-invariant cuts both ways and the paper says so: it
means they cannot be stabilised by seed averaging either. If they are wrong,
they are reproducibly wrong.

Run:  python code/diagnostics_loso/verify_seed_sensitivity.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import config as cfg  # noqa: E402

STOCHASTIC_KEYS = ("subsample", "subsample_freq", "colsample_bytree", "colsample_bynode",
                   "colsample_bylevel", "bagging_fraction", "bagging_freq", "feature_fraction")
SEEDS = (42, 7, 123)


def _data(n: int = 2500, p: int = 94):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, p))
    y = X[:, :5].sum(1) + rng.normal(scale=0.5, size=n)
    return X, y, rng.normal(size=(400, p))


def quantile_params_in_source() -> dict:
    """Read the qp dict out of rolling_origin.py without running the pipeline."""
    src = (CODE_DIR / "rolling_origin.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "dict":
            continue
        keys = {kw.arg for kw in node.keywords}
        if "objective" in keys:                      # the quantile learner block
            return {kw.arg: kw for kw in node.keywords}
    raise AssertionError("could not find the quantile parameter dict in rolling_origin.py")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import lightgbm as lgb
    checks = []
    X, y, Xt = _data()

    # 1 - the quantile learners, exactly as rolling_origin.py builds them
    qp = dict(objective="quantile", n_estimators=200,
              learning_rate=cfg.LGBM_PARAMS["learning_rate"], verbose=-1)
    lo = {s: lgb.LGBMRegressor(alpha=0.05, random_state=s, **qp).fit(X, y).predict(Xt)
          for s in SEEDS}
    hi = {s: lgb.LGBMRegressor(alpha=0.95, random_state=s, **qp).fit(X, y).predict(Xt)
          for s in SEEDS}
    d_lo = max(np.abs(lo[SEEDS[0]] - lo[s]).max() for s in SEEDS[1:])
    d_hi = max(np.abs(hi[SEEDS[0]] - hi[s]).max() for s in SEEDS[1:])
    print(f"quantile learners, max |delta| across seeds {SEEDS}: lo {d_lo:.10f}  hi {d_hi:.10f}")
    checks.append(("the lower-quantile learner is seed-invariant", d_lo == 0.0))
    checks.append(("the upper-quantile learner is seed-invariant", d_hi == 0.0))

    # 2 - the point learner must NOT be, or --seed is decorative
    pt = {s: lgb.LGBMRegressor(**{**cfg.LGBM_PARAMS, "random_state": s,
                                  "n_estimators": 200}).fit(X, y).predict(Xt) for s in SEEDS}
    d_pt = max(np.abs(pt[SEEDS[0]] - pt[s]).max() for s in SEEDS[1:])
    print(f"point learner,     max |delta| across seeds {SEEDS}: {d_pt:.10f}")
    checks.append(("the point learner does respond to the seed", d_pt > 1e-6))

    # 3 - the regression guard on the source itself
    found = quantile_params_in_source()
    present = [k for k in STOCHASTIC_KEYS if k in found]
    print(f"quantile parameter block keys: {sorted(found)}")
    print(f"stochastic keys present: {present or 'none'}")
    checks.append(("the quantile parameter block declares no stochastic keys", not present))

    # and the same for the point block, which inherits LGBM_PARAMS
    live = [k for k in STOCHASTIC_KEYS if k in cfg.LGBM_PARAMS]
    print(f"LGBM_PARAMS stochastic keys: {live or 'none'}")
    checks.append(("LGBM_PARAMS still carries the feature subsampling the point model needs",
                   "colsample_bytree" in cfg.LGBM_PARAMS))
    # subsample without subsample_freq is inert in LightGBM; flag it so nobody
    # reads 0.8 in the config and believes the point model is bagging.
    inert = "subsample" in cfg.LGBM_PARAMS and "subsample_freq" not in cfg.LGBM_PARAMS
    if inert:
        print("NOTE  LGBM_PARAMS sets subsample but not subsample_freq, so bagging is OFF; "
              "the point model's seed sensitivity comes from colsample_bytree alone")

    failed = 0
    print()
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        failed += 0 if ok else 1
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
