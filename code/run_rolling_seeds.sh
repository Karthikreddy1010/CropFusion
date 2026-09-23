#!/usr/bin/env bash
# Rolling origin across seeds. Interval metrics are invariant to the seed by
# construction (the quantile learners specify no subsample or colsample and
# LightGBM's bagging is off without subsample_freq); this exists for the POINT
# metrics, where the model-vs-baseline margin at the 2015 origin is 0.0064 and
# the seed moves R2 by about 0.024.
set -u
cd "$(dirname "$0")/.."
for s in 7 123 2024 3407; do
  echo "=== seed $s : $(date '+%H:%M:%S') ==="
  python code/rolling_origin.py --first 2008 --last 2023 --seed "$s" \
      --tag "rolling_origin_s${s}" 2>&1 | tail -2
done
echo "=== all seeds done: $(date '+%H:%M:%S') ==="
