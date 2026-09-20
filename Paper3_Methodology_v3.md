# Paper 3 — Methodology (v3)

**Supersedes `Paper3_Methodology_Updated_v2.md`.** That version described a study
that was not run: its numbers came from the ERA5-era dataset, it referred to
evidence files that do not exist, and it presented an architecture (PhenoFormer),
inputs (NDVI, DSSAT) and a proposed calibrator that are absent from the code. This
version is written from the code as it stands, and every number resolves to a file
produced by the run of 2026-09-18 (frozen protocol `1.1.0-2026-09-18`, hash
`d07d31512391047c`).

Draft status: sections marked **[C9]** await static-layer provenance; section 12
lists what is still outstanding.

---

## 1. Objectives

| # | Objective |
|---|---|
| RO1 | Quantify regime-conditional coverage of conformal prediction intervals for county maize yield under temporal and spatial distribution shift. |
| RO2 | Test whether pre-specified group-conditional (Mondrian) calibration closes the regime gap. |
| RO3 | Determine whether coverage failure is driven by interval width or by prediction bias. |
| RO4 | Express interval failure as decision-relevant loss rather than coverage alone. |
| RO5 | Establish whether the regime effect is confounded with yield level, state, or a single year. |

**Disclosure.** These objectives were finalised after the evaluation was run and
after the audit corrections described in §11. They replace an earlier
architecture-centred framing. No model, calibrator, split, regime definition or
metric was selected on test-set outcomes; `code/frozen_protocol.py` records every
such choice, is hashed, and is written into each output folder.

The study is an **evaluation** of interval validity under compound climate stress.
It does not propose a new model or a new calibration method.

## 2. Study area and period

Six rainfed US Corn Belt states, 1985–2023, county level.

| | |
|---|---|
| States | Illinois, Indiana, Iowa, Minnesota, Missouri, Ohio |
| Counties | 579 (IL 102, IN 92, IA 99, MN 85, MO 114, OH 87) |
| County-year records | 22 737, of which **20 507** carry an observed maize yield |
| Mean yield | 8.79 t/ha (sd 2.34) |

Nebraska is excluded: its irrigated share makes rainfed compound-stress response
non-comparable. The exclusion predates this analysis and is unchanged.

Maize only. The master dataset carries no soybean target, so the `SECONDARY_TARGET`
constant in `config.py` resolves to a column that does not exist; nothing in this
paper is computed on soybean.

## 3. Data sources

| Role | Source | Notes |
|---|---|---|
| Daily temperature, precipitation | PRISM AN81d | county-aggregated, `code/pipelines/prism/` |
| Radiation, humidity, wind, reference ET | NLDAS-2 | FAO-56 ET0, `code/pipelines/nldas/` |
| Storm events | NOAA Storm Events Database | county-year counts, `code/pipelines/noaa_storm/` |
| Yield | USDA NASS QuickStats | bu/acre → t/ha at 0.0627 |
| Soil, land cover, terrain | **[C9]** product, depth interval, units, NLCD epoch and slope units to be documented |

Build pipeline: `code/pipelines/master_dataset/01`–`11`.

## 4. Target and trend

The modelling target is the maize yield **anomaly**: a linear year trend is fitted
on FIT rows only and subtracted, and the trend is added back before every reported
metric. Over 1985–2013 the fitted trend is **0.1094 t/ha/yr**; the detrended target
has sd **1.785 t/ha** and skew **−0.79**.

Detrending is refitted inside every partition scheme — for leave-one-state-out,
on each fold's own FIT rows, never pooled. This was a defect corrected on
2026-09-15: LOSO folds had been trained on the raw trending target while the
temporal path was detrended, which cost roughly 0.34 in macro R².

## 5. Features

Engineered per county-year and grouped by role: temperature, precipitation and
water balance, drought indices, compound stress, phenology-stage aggregates,
terrain and soil, land cover, storm exposure, calendar encodings, and county
baseline. After FIT-only selection and a multicollinearity prune, **94 features**
enter the models (`feature_selection_report.json`, `multicollinearity_report.json`).

**Drought indices.** SPI and SPEI are standardised on a **fixed 1985–2013
reference period** (`code/pipelines/master_dataset/07_spi_spei_cdhw.py`), i.e. the
locked FIT years, so no DEV, CAL or TEST information enters the standardisation.
See §11 for the one place this creates a complication.

**Compound drought–heat–wet (CDHW) events** use absolute, pre-specified thresholds,
not sample quantiles: Tmax ≥ 35 °C, SPEI ≤ −1.0, with phenology windows set by
growing degree days (vegetative < 700, silking 700–1400, grain fill > 1400).

**Exposure class**, the regime variable for every conditional analysis, is derived
from the county-year CDHW event count and fixed in advance:

| Class | Definition | Records (all years) |
|---|---|---|
| Normal | 0 events | 15 055 |
| Moderate | ≥ 1 event | 2 407 |
| Extreme | ≥ 3 events | 3 045 |

The name `Year_Type` in the code is a misnomer — the class is county-year, not
year — and is reported as "CDHW exposure class" throughout.

## 6. Partitions

### 6.1 Four-way temporal split (primary, locked)

| Partition | Years | n | Role |
|---|---|---|---|
| FIT | 1985–2013 | 15 751 | model fitting, detrending, feature selection, scaler, county baseline |
| DEV | 2014–2015 | 978 | early stopping and ensemble weight only |
| CAL | 2016–2018 | 1 433 | conformal calibration only |
| TEST | 2019–2023 | 2 345 | locked final evaluation |

### 6.2 Leave-one-state-out (spatial transfer)

Six folds. The held-out state supplies test rows for all years; the remaining five
states are split by the same FIT/DEV/CAL year boundaries. `county_baseline` is
excluded in LOSO because it resolves to a constant for counties never seen in
training.

### 6.3 Rolling origin (replication, RO1/RO5)

For each test year Y from 2008 to 2023: FIT ≤ Y−5, DEV Y−4…Y−3, CAL Y−2…Y−1,
TEST Y. Sixteen origins, 7 844 test rows. The engineered frame is trimmed to
Year ≤ Y before splitting, so later years are not visible at any origin; the master
loader enforces this. `code/rolling_origin.py`.

## 7. Models

**NeuralCQR** — residual MLP backbone with four heads (mean, q0.05, q0.50, q0.95),
hidden dims (64, 32), dropout 0.2, AdamW at lr 3e-5, batch 128, up to 200 epochs
with early stopping on DEV RMSE (patience 60). Composite loss: Huber on the mean
head plus pinball on the three quantiles, a crossing penalty (λ = 10) and a width
penalty (λ = 0.005). LOSO uses a separate, larger configuration.

**LightGBM** — 1000 trees, lr 0.05, depth 7, 63 leaves, with separate quantile
regressors for the interval bounds.

**Ensemble** — a single weight chosen on DEV RMSE and then frozen.

The paper does not claim the architecture as a contribution. LightGBM alone reaches
temporal R² 0.731 against 0.703 for the ensemble, and that is reported.

## 8. Calibration methods compared

Static split conformal; rolling-refresh split conformal; **group-conditional
(Mondrian) conformal** with the pre-specified exposure classes (Vovk 2013; Gibbs,
Cherian & Candès 2025); standard ACI (Gibbs & Candès 2021); severity-aware ACI;
weighted conformal; locally adaptive conformal; phenology-stratified CQR. Nominal
coverage 0.90 throughout; α is fixed and never tuned on TEST.

Two points of honesty:

- **Severity-aware ACI was defective.** The published implementation multiplied the
  calibration scores by a severity weight and *divided* the test threshold by it,
  so the counties under heaviest compound stress received the narrowest intervals.
  It is corrected to the normalised-score form (divide the scores, multiply the
  threshold), verified in `code/diagnostics_loso/verify_c5_saaci.py`. Standard ACI
  is the headline adaptive method so that no finding rests on the corrected code.
- **Phenology-stratified CQR barely differs from static conformal** on this data:
  PICP 0.9493 vs 0.9488, MPIW 3.6506 vs 3.6493, Winkler 4.1475 vs 4.1473. The
  stratification changes almost nothing, and it is reported as a variant of static
  conformal rather than as a distinct method. (An earlier audit note called the two
  identical; on the 2026-09-18 run they differ in the fourth decimal.)

## 9. Metrics

Point: RMSE, MAE, R², bias. Interval: PICP, ACE, MPIW, Winkler score.

**Consequence metrics (RO4).** Two-sided coverage hides the direction that matters:
an observation above the upper bound is a pleasant surprise, one below the lower
bound means the loss exceeded what the interval admitted. For a two-sided interval
at 0.90 the calibrated downward failure rate is 0.05, so we report the one-sided
`unsafe_miss_rate` against that target, the mean and worst shortfall in t/ha, and
the expected shortfall E[max(lower − y, 0)], which combines frequency and
magnitude. `code/decision_impact.py`.

All metrics are reported marginally and conditionally on exposure class, year and
state. Pooled figures are **row-weighted**; per-origin means are reported
separately because extreme-class sizes range from 1 to 448 rows per origin.

## 10. Statistical inference

The unit of independence is the **year**. County-year rows within a year share
weather and are not independent, so row-level tests over ~2 300 rows are
anti-conservative — they returned p = 0.0 against every comparator. Headline
inference uses paired tests on year-level means, year-block cluster bootstraps, and
sign tests across origins; Holm–Bonferroni and Benjamini–Hochberg corrections are
applied to the **year-level** p-values. Row-level tests are retained as labelled
diagnostics only.

With five test years in the locked window, non-significance is weak evidence of no
difference rather than evidence of equivalence. This is the reason the rolling-origin
replication exists.

## 11. Leakage controls, corrections and known limitations

**Fit-only discipline**, verified for: RobustScaler, feature selection,
multicollinearity prune, imputation (`TrainFittedPreprocessor.fit`), county
baseline, detrending, target lags and rolling features (split-aware, backward
looking per county). A row-identity provenance ledger binds each stage to its
partition and asserts disjointness; `leakage_provenance_audit.json`.

**Corrections applied during the audit**, all disclosed rather than silently fixed:
LOSO detrending (§4); SA-ACI severity direction (§8); multiple-comparison
corrections moved to year-level p-values (§10); the LOSO ensemble-weight table read
key names the pipeline never wrote, so it was empty; an ablation distinctness check
failed on correct post-hoc behaviour; and the artefact-presence audit was labelled
as methodological compliance.

**Held-out-informed tuning (unresolved).** The LOSO hyperparameters descend from six
rounds selected by watching LOSO R² rise — that is tuning on held-out states. The
history is disclosed verbatim in `docs/supplement/development_history.md`, the
frozen protocol marks the values `PENDING` re-derivation on each fold's own DEV
rows, and until that run the LOSO estimates should be read as optimistic by an
unknown amount.

**SPEI reference period in the rolling-origin extension.** SPI/SPEI standardisation
is fixed to 1985–2013. Rolling origins move, so for the six origins with test years
2008–2013 the index reference period includes the origin's own test year. The
quantity involved is a per-county distributional constant that carries no yield
information, and re-referencing drought indices per origin is not standard practice
either — but the locked temporal analysis is unaffected (its FIT period *is* the
reference period) and the rolling-origin results should be read with this in mind.

**Other limitations.** Single seed for the reported run; LightGBM only in the
rolling-origin analysis; one test year per origin, so ACI's adaptivity is not
exercised there and the static-family methods coincide; irrigated share is not
controlled; phenology windows use fixed GDD thresholds rather than observed
silking dates.

## 12. Reproducibility

Frozen protocol `1.1.0-2026-09-18`, hash `d07d31512391047c`, written to
`outputs_master/reports/frozen_protocol.json` with a check that the live
configuration matches it. Seed 42. Model weights, scalers, per-fold metrics,
predictions and every report are written into the run's output folder. 53
protocol-compliance and pipeline tests accompany the code.

**Outstanding before submission:** [C9] static-layer provenance; the DEV-only
re-derivation of LOSO hyperparameters; a fixed-effects degree-day panel baseline;
and DtACI/AgACI comparators.
