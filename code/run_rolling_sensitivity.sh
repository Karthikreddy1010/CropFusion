#!/usr/bin/env bash
# Rolling-origin sensitivity runs declared in manuscript/SENSITIVITY_PLAN_2026-09-26.md.
# R0 re-runs the primary LightGBM design with the added calibrators (its four primary
# calibrators must reproduce outputs_diagnostics/reports/rolling_origin_rows.csv exactly);
# R1/R2 swap the point model; R3 uses four calibration seasons. Outputs go to
# outputs_diagnostics/reports/<tag>_{raw,rows,summary,loo_sensitivity}.csv; logs beside them.
# Run from the repository root:  bash code/run_rolling_sensitivity.sh
set -u
cd "$(dirname "$0")"
OUT=../outputs_diagnostics/reports
export PYTHONUTF8=1
run () {
  local tag=$1; shift
  echo "=== $tag  $(date '+%F %T') ==="
  python rolling_origin.py --first 2008 --last 2023 --sensitivity-calibrators --tag "$tag" "$@" \
    > "$OUT/${tag}.log" 2>&1
  echo "    exit $?  $(date '+%F %T')"
}
run ro_sens_lgbm     --model lgbm
run ro_sens_catboost --model catboost
run ro_sens_xgboost  --model xgboost
run ro_sens_cal4     --model lgbm --cal-seasons 4
echo "=== all done $(date '+%F %T') ==="
