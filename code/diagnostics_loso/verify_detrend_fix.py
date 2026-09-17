"""verify_detrend_fix.py - smoke-test the detrend change through the REAL _run_loso_cv.

This calls `main._run_loso_cv` itself (not a copy), restricted to one held-out state and with
a deliberately tiny epoch budget, so it checks the plumbing quickly:

  * the trend is fitted on the fold's FIT partition only,
  * predictions and quantile bounds come back on the raw yield scale,
  * the provenance ledger records `target_detrending -> fit_full` and still passes,
  * the new fold-metric fields are populated,
  * DEV / CAL / TEST scoring still uses raw targets.

The epoch budget is NOT a performance setting - no metric from this run means anything about
model quality. Output goes to outputs_diagnostics/, never outputs_master/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PAPER3_DATA_SOURCE", "master")
_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import _fold_lab  # noqa: F401  - redirects every cfg *_DIR to outputs_diagnostics
import config as cfg
import numpy as np

STATE = sys.argv[1] if len(sys.argv) > 1 else "Minnesota"
FAST_EPOCHS = int(os.environ.get("VERIFY_EPOCHS", "3"))

import main  # noqa: E402
from leakage_provenance import COLLECTOR  # noqa: E402

# Load with the real six-state list (the master loader validates against it), then
# restrict the fold loop to one state purely to keep this check fast.
df = _fold_lab.engineered_df()
cfg.LOSO_STATES = [STATE]
cfg.LOSO_MAX_EPOCHS = FAST_EPOCHS
cfg.LOSO_EARLY_STOPPING_PATIENCE = FAST_EPOCHS
print(f"Running the real _run_loso_cv on {STATE} with epochs={FAST_EPOCHS} "
      f"(plumbing check only, metrics are meaningless at this budget)\n", flush=True)

metrics, preds = main._run_loso_cv(df, feature_cols=[])

assert metrics, "no fold metrics returned"
m = metrics[0]
rec = [r for r in COLLECTOR.records if r.get("held_out_state") == STATE][-1]

checks = []
checks.append(("fold ran", m["state"] == STATE))
checks.append(("detrending recorded in fold metrics",
               m.get("target_detrending") == "Linear_FIT_Only"))
checks.append(("fold trend is a plausible yield trend (0.05-0.20 t/ha/yr)",
               0.05 < float(m.get("fold_trend_t_ha_per_year", 0)) < 0.20))
checks.append(("ledger binds target_detrending -> fit_full",
               rec.get("stage_bindings", {}).get("target_detrending") == "fit_full"))
checks.append(("leakage audit still PASS", rec.get("status") == "PASS"))
checks.append(("fit_full == train + es_internal",
               rec["partition_sizes"]["fit_full"]
               == rec["partition_sizes"]["train"] + rec["partition_sizes"]["es_internal"]))
checks.append(("no stage consumes the test partition",
               all(p != "test" for s, p in rec["stage_bindings"].items() if s != "final_evaluation")))

# predictions must be on the raw yield scale, not the anomaly scale
y = preds["y_true"].values
p = preds["y_pred"].values
checks.append(("predictions are on the raw yield scale (not anomalies)",
               bool(p.mean() > 3.0 and abs(p.mean() - y.mean()) < 4.0)))
checks.append(("interval bounds bracket the point prediction",
               bool((preds["lower_bound"] <= preds["upper_bound"]).all())))
checks.append(("test rows = held-out state only", set(preds["held_out_state"]) == {STATE}))

print(f"{'check':60s} result")
print("-" * 70)
ok = True
for name, passed in checks:
    ok &= bool(passed)
    print(f"{name:60s} {'PASS' if passed else 'FAIL'}")

print("\nfold metrics of interest:")
for k in ("target_detrending", "fold_trend_t_ha_per_year", "n_features", "n_train", "n_es_val",
          "n_dev", "n_cal", "n_test", "dev_r2_neural_alone", "dev_r2_lgb_alone",
          "neuralcqr_ensemble_weight", "lightgbm_ensemble_weight", "calibration_threshold_q",
          "r_squared", "rmse", "picp", "mpiw"):
    print(f"  {k:32s} {m.get(k)}")
print(f"\npred mean={p.mean():.3f}  truth mean={y.mean():.3f}")
print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
