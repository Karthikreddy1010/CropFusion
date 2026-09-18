# Supplement: LOSO hyperparameter development history

Moved out of `code/config.py` on 2026-09-18 (audit blocker C7).

**Why this matters.** Each round below was judged by whether leave-one-state-out
R2 went up. The held-out states therefore informed the hyperparameters, so the
published LOSO numbers are optimistic by an unknown amount. The history is kept
verbatim for disclosure; the values are to be re-derived using each fold's own
DEV rows before the final run, and the frozen protocol
(`code/frozen_protocol.py`) marks them PENDING until that happens.

## Verbatim history as it stood in config.py

```
# LOSO-SPECIFIC HYPERPARAMETERS
#
# History (each validated against a real full-pipeline run, not just
# sandbox tests):
#   1. Original (lr=1e-3, batch=64, epochs=40, patience=20, es="pinball"),
#      county_baseline leaking a constant 0.0 into every held-out-state row
#      -> LOSO R2 = 0.429
#   2. Reused the temporal-split fix's hyperparameters as-is for LOSO too,
#      county_baseline still leaking -> LOSO R2 dropped to 0.268
#   3. Tried a moderate in-between regime based on sandbox testing,
#      county_baseline still leaking -> LOSO R2 dropped further to 0.223
#      (worse; sandbox testing did NOT predict this correctly)
#
# Root cause found after two failed hyperparameter-only attempts:
# `county_baseline` (a per-county historical-mean feature, genuinely
# informative for temporal/random-row splits where the same counties
# appear in train and test) is structurally broken for LOSO -- it's built
# ONLY from the development states' GEOIDs, so for every held-out-state
# test row (a GEOID that was NEVER in training) it resolves to a constant
# placeholder (0.0) for the entire test set. It's not just noisy, it
# carries zero information for the one task that most needs genuine
# spatial signal -- and the more thoroughly the model trains, the more it
# leans on this feature, which is exactly why "better" training (rounds 2
# and 3) made LOSO worse, not better.
#
#   4. county_baseline excluded from the LOSO feature list entirely (see
#      `_run_loso_cv` in main.py), keeping the temporal-split-tuned
#      hyperparameters -> LOSO R2 recovered to 0.289, no more negative
#      folds (Minnesota -0.033 -> +0.060), but still below the original
#      0.429 -- 2-fold sandbox testing at the time suggested the original
#      hyperparameters might do even better than the tuned ones once the
#      leaky feature was actually gone (Minnesota: 0.53 with original hp
#      vs 0.44 with tuned hp), so that combination gets tested here.
#
#   5. (current) county_baseline excluded (fix from round 4 kept) +
#      original hyperparameters restored for LOSO specifically. Basis:
#      the round-4 sandbox comparison above, run on 2 rebuilt folds
#      (Illinois, Minnesota) with the leaky feature actually removed --
#      original hyperparameters were competitive-to-better, especially on
#      the harder fold. Validated on the real six-fold run -- see
#      `reports/loso_summary.csv`.
```

```
#   6. (merge fix) Root-caused why round 5 (above) still only reached
#      LOSO R2=0.35 on the real six-fold run despite the county_baseline
#      fix: train_neural_cqr()'s `hidden_dims` argument was never
#      overridden for LOSO, so every fold silently trained the small
#      (64, 32) net that was tuned specifically for the temporal-split
#      regime (lr=3e-5, patience=60, 200 epochs, es_mode="rmse"). Under
#      the much shorter/faster LOSO schedule (lr=1e-3, 40 epochs), that
#      small net is capacity-starved and underfits each state's fold.
#      A prior sandbox run (see PATCH_NOTES.md / final_clean lineage)
#      showed that reusing the larger (256, 128, 64, 32) architecture
#      with this exact same short LOSO schedule (epochs=40, batch=64,
#      lr=1e-3) recovered LOSO R2 to ~0.55 without touching the
#      temporal-split config at all. Kept as a separate LOSO-only
#      architecture so the tuned main-pipeline net is untouched.
```
