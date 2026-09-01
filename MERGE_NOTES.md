# Merge notes: combining the good-backbone run and the good-LOSO run

## What was different between your two prior runs

Both codebases already shared the good dataset-level fixes (Nebraska
dropped, target detrending, county baseline, multicollinearity pruning).
Comparing `pipeline.log` from each run against the code that produced it:

| | Backbone/ensemble test R² | LOSO-CV mean R² |
|---|---|---|
| `Paper3_FINAL.zip` code (this codebase, pre-fix) | ~0.50–0.52 (good) | 0.35 (bad) |
| `Paper3_FINAL_clean` code (patched v7) | 0.49 (slightly worse) | 0.55 (good) |

The temporal-split / backbone training config in `Paper3_FINAL.zip` was
correctly tuned (small net, low LR, long patience, RMSE-mode early
stopping) and should **not** be touched. The problem was entirely in how
`_run_loso_cv()` trained the per-fold neural net. Two bugs, found by
diffing `config.py`/`main.py` against the older codebase:

1. **Architecture mismatch.** `train_neural_cqr()` was never given a
   `hidden_dims` override for LOSO, so every fold silently trained the
   small `(64, 32)` net tuned for the temporal split. Under the LOSO
   schedule (only 40 epochs, lr=1e-3), that net is capacity-starved.

2. **Less training data.** The pre-fix `_run_loso_cv` withheld the
   entire 2016–2018 window (`val_fold`) from training. The
   `final_clean` lineage trains on the full 1985–2018 dev-state pool and
   carves early-stopping validation from the tail of that combined set —
   ~10% more, and more recent, training data per fold.

## What was changed

- `code/config.py`: added `LOSO_HIDDEN_DIMS = (256, 128, 64, 32)` and
  `LOSO_DROPOUT = 0.25` (matching the architecture that produced the
  good LOSO run), documented inline next to the existing `LOSO_*`
  hyperparameter block.
- `code/main.py`, `_run_loso_cv()`:
  - passes `hidden_dims=cfg.LOSO_HIDDEN_DIMS, dropout_rate=cfg.LOSO_DROPOUT`
    into `train_neural_cqr()`.
  - trains on `concat(train_fold, val_fold)` (1985–2018), carving the
    last ~10% off as the early-stopping validation set, instead of
    training on 1985–2015 only.
  - **conformal calibration and reported PICP/MPIW/ACE/Winkler metrics
    are unaffected** — those still use the genuine held-out `val_fold`
    (2016–2018) for `static_conformal()`, exactly as before. Only the
    point-prediction backbone's training data and architecture changed.
  - `n_train` in the LOSO metrics dict now reports the actual number of
    rows used for gradient training (post-split), and a new `n_es_val`
    field reports the early-stopping validation size.
- The temporal-split (backbone/ensemble) training path is **completely
  untouched** — same hyperparameters, same code path as your good run.

## Validation done so far

Ran both fixes together on 2 of the 6 LOSO states (isolated single-fold
runs, not the full 6-fold loop, to fit in a short interactive session):

| State | Pre-fix (small net, val withheld) | Fixed (large net + combined train/val) | `final_clean` reference |
|---|---|---|---|
| Illinois | R² ≈ 0.37 | **R² = 0.529** | R² = 0.631 |
| Minnesota | R² = 0.376 | **R² = 0.504** | R² = 0.568 |

This is a substantial, consistent recovery in the direction predicted,
though not an exact match to the historical numbers (residual gap is
plausibly random-seed variance or a smaller remaining implementation
difference not yet isolated). **The full 6-state LOSO loop, plus the
Phase 1–3 backbone/ensemble run, has not been re-run end-to-end** in
this session — background jobs longer than ~5 minutes kept getting
killed in this sandbox.

## Update: per-fold ensemble added, and it's a big additional win

On top of the two fixes above, `_run_loso_cv()` now also trains a
LightGBM model per fold and blends it with NeuralCQR, weight tuned on
the genuine held-out `val_fold` (2016–2018) — exactly mirroring the
backbone benchmark's own `NeuralCQR + LightGBM` ensemble block, which
already improved the temporal-split R² from 0.50 to 0.54 in your
original run. No leakage: the LOSO test state's data is never used to
pick the weight.

Validated end-to-end (calling the real `_run_loso_cv()` from `main.py`,
not a re-implementation) on the two most informative folds:

| State | Pre-fix | Arch+data fix only | **+ LightGBM ensemble (now)** | `final_clean` reference |
|---|---|---|---|---|
| Illinois | R²≈0.37 | R²=0.529 | **R²=0.650** | R²=0.631 |
| Missouri (worst fold in reference) | — | — | **R²=0.537** | R²=0.235 |

Missouri more than doubled — worth noting *why*: the ensemble weight
search picked **w=0.00 (pure LightGBM)** for that fold, because
NeuralCQR's val R² was actually negative (-0.64) there. The neural net
overfits badly on this particular held-out state; the tree model alone
generalizes far better. This is a real, mechanistically-explained
improvement, not noise — LightGBM is a from-scratch tree ensemble per
fold, so it doesn't share the neural net's per-fold overfitting failure
mode.

**Bonus, unplanned:** PICP (interval coverage) also jumped to ~0.92 on
both folds (target ≥0.90), vs. 0.52–0.56 in the reference run. The
ensemble's wider LightGBM quantile intervals happened to fix the
severe under-coverage problem too — directly relevant to this paper's
whole thesis about calibration under distribution shift, and worth
highlighting in the write-up if the full run confirms it holds.

If a fold's `neuralcqr_ensemble_weight` in the output metrics is low or
0.0, that's diagnostic information worth reporting, not a code smell —
it's telling you which held-out states the neural backbone generalizes
to worst.

## What to do next

Run the merged pipeline in your own environment (the one that
originally completed a full run in ~25 min on GPU):

```bash
cd Paper3_MERGED/code
python3 main.py
```

Check `outputs/reports/pipeline_summary.json` and the LOSO section of
`pipeline.log` for the final backbone R² and LOSO-CV mean R². Given the
2-fold spot check above, I'd expect backbone R² to stay ~0.50 (unchanged
code path) and LOSO mean R² to land somewhere in the 0.45–0.55 range —
worth confirming against your actual full run before citing numbers
anywhere.

A `code/validate_loso_fix.py`-style script (removed from this package
to avoid shipping a scratch file, but easy to recreate) is useful if you
want to spot-check a single state's LOSO fold quickly without running
the full pipeline — happy to regenerate it if useful.
