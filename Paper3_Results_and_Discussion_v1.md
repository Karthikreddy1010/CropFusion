# Paper 3 — Results and Discussion (draft v1, 2026-09-22)

Companion to `Paper3_Methodology_v3.md` (protocol 1.4.0-2026-09-23, hash
`9ea6b15d86c6ebef`). Every figure below is quoted from an artefact on disk; the
source file is named at the head of each subsection. Pooled rolling-origin
figures carry 95% intervals that resample whole origins, because the origin is
the unit of independence (§10 of the methodology).

Nothing in this draft is a claim the artefacts do not support. Where a result is
descriptive rather than tested, it says so.

---

## 1. Results

### 1.1 The difficulty ladder

*Sources: `outputs_master/comparisons/temporal_vs_random_split_comparison.csv`,
`outputs_master/reports/loso_summary.csv`, `outputs_master/splits/`.*

The four-way temporal split assigns 20 507 county-years: FIT 1985–2013
(n = 15 751), DEV 2014–15 (n = 978), CAL 2016–18 (n = 1 433) and TEST 2019–23
(n = 2 345). 94 features enter the model.

How a model is asked to generalise dominates every other design choice:

| evaluation | role | R² | RMSE (t/ha) |
|---|---|---|---|
| random row-level 70/10/20 | interpolation benchmark | **0.938** | 0.581 |
| temporal, 2019–23 held out | out-of-time forecast | **0.703** | 0.924 |
| leave-one-state-out, macro | out-of-space transfer | **0.719** | 1.168 |

The random split is not a forecasting result and is reported only to size the
gap: it flatters the model by 0.235 R² over the temporal split, because
neighbouring county-years of the same field enter training and test together. A
literature that reports random-split accuracy for a forecasting task is
reporting interpolation.

LOSO macro R² (0.719, 95% CI across the six folds [0.643, 0.794]) is not
directly comparable with the temporal figure — it withholds space rather than
time — but it establishes that spatial transfer is no easier than temporal
transfer for this model.

### 1.2 Point accuracy and the ensemble

*Source: `outputs_master/reports/backbone_benchmark.csv`.*

| model | R² | RMSE | MAE |
|---|---|---|---|
| LightGBM | **0.7313** | 0.8788 | 0.6732 |
| CatBoost | 0.7238 | 0.8908 | 0.6918 |
| XGBoost | 0.7163 | 0.9029 | 0.6924 |
| NeuralCQR | 0.6840 | 0.9529 | 0.7328 |
| NeuralCQR + LightGBM ensemble | 0.7031 | 0.9237 | 0.7091 |

Two things should be said plainly. **LightGBM alone is the most accurate point
model**, and the ensemble the pipeline reports (0.7031) is the second-worst of
the five. The neural component is retained because it supplies conditional
quantiles the tree baselines do not, not because it improves accuracy; the
interval results in §1.3 onwards are the reason the architecture exists. On the
random split the ensemble does win (0.938 against NeuralCQR's 0.880), which is
consistent with the ensemble helping where interpolation is the task and not
where extrapolation is.

The coverage column of that table is **raw model coverage, before conformal
calibration** (NeuralCQR 0.755, LightGBM 0.661) and must not be read alongside
the calibrated figures below. This mixing is a known defect in
`comprehensive_metrics.csv` and is flagged in §3.

### 1.3 Baselines

*Sources: `rolling_origin_raw.csv`, `rolling_origin_seed_sweep.json` (5 seeds).*

Across the 16 rolling origins the model beats a trend + county-mean baseline in
**16 of 16** origins, mean R² 0.585 against 0.283. The margin is widest in 2012,
where the baseline is actively harmful (R² −1.190 against the model's 0.430) —
a county mean plus a linear trend cannot represent a drought year at all.

Unlike the interval results (§2.5), point accuracy does depend on the seed, so
this count was checked rather than assumed. Repeating all 16 origins at five
seeds (42, 7, 123, 2024, 3407) returns **16 of 16 under every seed**; the
across-origin mean R² is 0.5852 with a seed standard deviation of 0.0054. The
claim is nonetheless narrower than the headline count suggests: at the 2015
origin the worst-case margin over seeds is **0.0064** (model 0.4131 against
baseline 0.4067), so a sixth seed could plausibly overturn that one origin. We
report it as 16 of 16 across five seeds with the tightest margin named, rather
than as a clean sweep.

Per-origin seed dispersion is largest where the year is hardest — sd 0.063 at
2009, the lowest-R² origin — and below 0.023 everywhere else.

The locked window's 0.703 sits at the optimistic end of the rolling range
(0.248–0.769), which is worth stating when comparing against published
single-window results.

### 1.4 Marginal interval calibration: every method looks adequate

*Source: `outputs_master/reports/conformal_comparison.csv`, TEST 2019–23,
nominal coverage 0.90.*

| method | PICP | ACE | MPIW | Winkler |
|---|---|---|---|---|
| SA-ACI | 0.9241 | **0.0241** | **3.3203** | **4.0396** |
| Standard ACI | 0.9296 | 0.0296 | 3.4151 | 4.0788 |
| Weighted conformal | 0.9471 | 0.0471 | 3.6367 | 4.1413 |
| Static conformal | 0.9488 | 0.0488 | 3.6493 | 4.1473 |
| Locally adaptive | 0.9488 | 0.0488 | 3.6987 | 4.1670 |
| Phenology-stratified CQR | 0.9493 | 0.0493 | 3.6506 | 4.1475 |

Read marginally, this is a success story with nothing to report: every method
attains or exceeds nominal coverage, all six land within 0.05 of target, and the
adaptive methods do so with the narrowest intervals and the best Winkler scores.
All methods **over**-cover. A paper that stopped here would conclude that
conformal prediction is solved for county maize yield.

Phenology-stratified CQR differs from static conformal only in the fourth
decimal (PICP 0.9493 against 0.9488). The stratification changes almost nothing
and is reported as a variant rather than a distinct method.

### 1.5 RO1 — conditional coverage by compound drought–heat exposure

*Source: `outputs_diagnostics/reports/rolling_origin_summary.csv`,
`objectives_RO_report.json`. 16 origins, test years 2008–2023, LightGBM,
nominal 0.90, row-weighted, 95% intervals over origins.*

| stratum | rows | origins | PICP | 95% CI | MPIW |
|---|---|---|---|---|---|
| Normal | 5 777 | 16 | 0.9057 | [0.867, 0.944] | 3.543 |
| Moderate | 933 | 16 | 0.9143 | [0.832, 0.975] | 3.601 |
| Extreme | 1 134 | 13 | **0.7584** | [0.651, 0.912] | 4.066 |

The point estimates fall by 14.7 coverage points from the Normal to the Extreme
class. **That contrast is not established at 95%**: the Normal-minus-Extreme
difference is 0.147 with a year-clustered interval of [−0.010, 0.257]
(p = 0.087), and a per-origin sign test over the nine origins with at least 20
extreme rows gives 7 of 9 (p = 0.090). The two-sided coverage interval for the
Extreme class also contains the nominal 0.90.

The reason is concentration rather than absence of effect: 2012 supplies 39.5%
of all extreme-exposure rows, so the resampling distribution is heavy-tailed and
the interval is wide despite a large point estimate. We report RO1 as the
descriptive stratification and RO4 as the test, because a two-sided coverage
statistic spends half its power on the upper tail, where an observation above
the interval is a good harvest rather than a failure.

Four of the 16 origins had too few extreme rows in the two-year calibration
window to form a group threshold, and five had fewer than 20 moderate rows;
both are reported rather than pooled away.

### 1.6 RO2 — group-conditional calibration recovers part of the deficit

*Source: as above.*

| stratum | static CP | group-conditional CP | Δ |
|---|---|---|---|
| Extreme | 0.7584 | **0.8228** | +0.064 |
| Moderate | 0.9143 | 0.9143 | 0.000 |
| Normal | 0.9057 | 0.8866 | −0.019 |

The extreme-class gain is **+0.0644, 95% CI [0.0314, 0.1053]**, p = 0.001 — and
unlike RO1 it survives every subset we tested:

| subset | gain | 95% CI |
|---|---|---|
| all 16 origins | +0.0644 | [0.0314, 0.1053] |
| excluding 2012 | +0.0758 | [0.0134, 0.1155] |
| clean origins 2014–23 | +0.0939 | [0.0120, 0.1408] |

The unsafe-miss rate falls correspondingly, from 0.231 to 0.170, a reduction of
0.0608 [0.0294, 0.0974]. A per-origin sign test agrees: group CP is better in
7 of the 7 origins where the two methods differ (p = 0.0156).

**The remedy reallocates width; it does not create it.** Extreme-class intervals
widen from 4.07 to 5.10 t/ha while normal-class intervals narrow from 3.54 to
3.16, and normal-class coverage falls by 0.019 [−0.042, −0.003] to 0.8866 —
below nominal. That cost is itself statistically established and is not
presented as a free improvement.

### 1.7 RO3 — the failure is a level error, not a width error

*Sources: `outputs_master/predictions/loso_predictions.csv`,
`rolling_origin_raw.csv`.*

Across 234 LOSO state-years, coverage tracks the absolute bias of the point
prediction (**r = −0.752**) and barely responds to interval width
(**r = −0.093**) — a correlation eight times smaller, and of a sign that means
wider intervals do not buy coverage here: the state-years with the widest
intervals are the hard ones, and widening did not rescue them.

The 2012 origin makes the mechanism concrete. Model bias is −0.987 t/ha, and
every calibrator fails together in the extreme stratum:

| method | PICP, 2012 extreme (n = 448) |
|---|---|
| rolling static CP | 0.5982 |
| standard ACI | 0.5982 |
| SA-ACI | 0.6138 |
| group-conditional CP | 0.6451 |

Group-conditional calibration helps by 4.7 points and still misses a third of
the stratum. No reallocation of interval width can repair a conditional mean
that is wrong by a metric tonne per hectare; the interval is centred in the
wrong place, and widening it symmetrically is the wrong instrument.

### 1.8 RO4 — the consequence, and the inversion

*Sources: `rolling_origin_summary.csv`, `decision_impact_by_stratum.csv`.
Target downward failure rate α/2 = 0.05.*

Rolling origin, static CP:

| stratum | unsafe miss rate | 95% CI | vs 0.05 | mean shortfall | E[shortfall] |
|---|---|---|---|---|---|
| Normal | 0.0386 | [0.014, 0.072] | — | 0.697 | 0.027 |
| Moderate | 0.0364 | [0.010, 0.071] | — | 0.690 | 0.025 |
| Extreme | **0.2310** | [0.073, 0.340] | **exceeds** | 1.113 | 0.257 |

In the extreme class the lower bound is breached for 23.1% of county-years
against a 5% target, and when it is breached the realised yield falls 1.11 t/ha
below the stated bound. Expected shortfall is roughly ten times the normal-class
value. Subsets: clean origins 0.155 [0.065, 0.225], **exceeds**; excluding 2012
0.122 [0.049, 0.186], which does **not** exclude the target (p = 0.053). The
strength of this claim therefore depends on admitting the most severe year in
the record as evidence, and we report it both ways.

**The inversion.** On the locked window, across all six conformal methods, the
ranking by marginal calibration is the reverse of the ranking by extreme-class
consequence:

| method | marginal ACE | extreme PICP | extreme unsafe | extreme shortfall |
|---|---|---|---|---|
| SA-ACI | **0.0241** (best) | 0.8366 | **0.1337** (worst) | 0.816 |
| Standard ACI | 0.0296 | 0.8515 | 0.1238 | 0.855 |
| Weighted conformal | 0.0471 | 0.8713 | 0.1188 | 0.732 |
| Static conformal | 0.0488 | 0.8762 | 0.1139 | 0.757 |
| Locally adaptive | 0.0488 | 0.8960 | **0.0941** (best) | **0.712** |
| Phenology-stratified CQR | 0.0493 (worst) | 0.8762 | 0.1139 | 0.757 |

The ordering is monotone: ranked by marginal ACE, the extreme-class unsafe rate
falls from 0.134 to 0.094 in almost the reverse order. Spearman ρ = −0.868
(p = 0.025). Selecting on marginal ACE would pick SA-ACI, which has the highest
miss rate and the largest shortfall of the six.

**We do not rest the claim on that p-value, and neither should a reader.** The
extreme stratum holds 202 rows, so one county-year is worth 0.005 in the miss
rate — about the distance between adjacent methods in the table. Flipping a
single row per method, chosen adversarially across all 64 sign combinations,
moves ρ to −0.431 (p = 0.393). The rank correlation is therefore a description
of this table, not an inference about methods in general: six methods, two of
them near-duplicates (phenology-stratified and static agree to four decimals),
and not six independent samples.

What does not depend on any of that arithmetic is the finding underneath it:
**every one of the six methods meets the nominal target marginally and breaches
the 0.05 safety target in the extreme class**, by factors of 1.9 to 2.7. That
statement needs no rank correlation, survives any single-row perturbation, and
replicates in the rolling-origin design over 1 134 extreme-class rows (§1.5,
§1.8 above), where the interval on the excess excludes the target. The direction
of the ranking is a further observation, offered as a hypothesis worth testing on
a larger extreme stratum rather than as an established relationship.

### 1.9 RO5 — is it exposure, or something exposure stands for?

*Source: `rolling_origin_rows.csv`.*

**Yield level.** Ranking counties within each origin, the exposure effect
persists inside the lowest quintile: unsafe rates 0.464 (Extreme) against 0.153
(Normal). The contrast is 0.311 with a year-clustered interval of [−0.006,
0.603] (p = 0.059) — suggestive, not established. Using quintiles cut across the
whole pool instead gives 0.385 against 0.185, but those quintiles are partly a
year control, since a bad year lands in the low bins by construction; the
within-origin version is the weaker and the more honest of the two.

**State.** The extreme-to-normal unsafe ratio exceeds 1.5 in **5 of 6** states:
Illinois 9.99, Indiana 6.45, Missouri 4.79, Ohio 4.60, Minnesota 3.87, and Iowa
1.32. These are descriptive — as few as 51 extreme rows per state, too thin for
per-state intervals — and are shown to demonstrate the direction is not carried
by one state.

**One year.** Leave-one-origin-out identifies 2012 as most influential: dropping
it moves extreme coverage from 0.758 to 0.863 while normal coverage is unchanged
at 0.905. The effect attenuates but does not reverse.

### 1.10 Compound drought–heat as predictor versus as stratifier

*Source: `outputs_diagnostics/reports/cdhw_contribution_summary.csv`,
`cdhw_contribution.json`.*

Refitting with and without the six-variable CDHW block, across five model
families × 3 seeds, scored on TEST with a year-level paired test:

| model | ΔR² mean | sd | min | max | year-level p |
|---|---|---|---|---|---|
| LightGBM | +0.0009 | 0.0104 | −0.0095 | +0.0114 | 0.716 |
| RandomForest | +0.0001 | 0.0004 | −0.0002 | +0.0006 | 0.761 |
| Ridge | +0.0017 | 0.000 | +0.0017 | +0.0017 | 0.505 |
| CatBoost | −0.0032 | 0.0013 | −0.0044 | −0.0019 | 0.199 |
| XGBoost | −0.0057 | 0.000 | −0.0057 | −0.0057 | 0.202 |

**The CDHW block does not improve point accuracy in any model family.** Signs
disagree, magnitudes are within seed noise, and no year-level test approaches
significance. This is reported as found; every model and seed that ran is in the
output file.

The null result is about the variables as **predictors** and leaves RO1–RO5
untouched, because the paper uses the CDHW event count as a pre-specified
**stratification** variable. The construct check confirms the classes separate
outcomes: on TEST, mean yield is 10.02 t/ha in the Extreme class (n = 202)
against 11.39 (Moderate, n = 426) and 11.41 (Normal, n = 1 717) — a 1.4 t/ha
deficit, with the Extreme class also the most dispersed (sd 2.29 against 1.54).
The classes identify where yield is low and variable, which is what a
stratification variable is for, whether or not it adds predictive signal on top
of 88 other features.

---

## 2. Discussion

### 2.1 What this study establishes

Marginal coverage is not evidence of conditional validity, and on this data it
is actively misleading. Six conformal methods all meet or exceed a 0.90 nominal
target on a five-year held-out window — and all six breach their 0.05 downward
failure target in the compound drought–heat stratum, by factors of 1.9 to 2.7.
Extending to 16 rolling origins, the extreme-class unsafe rate is 0.231
[0.073, 0.340].

The failure has a mechanism, and it is not the one width-adaptive methods
address. Coverage tracks |bias| (r = −0.752) and barely responds to width
(r = −0.093). In 2012 the conditional mean is wrong by −0.99 t/ha and every
calibrator fails together at 0.60–0.65 coverage. Conformal prediction inherits
the level error of the model it wraps; it can only decide how much mass to place
around a centre it is given.

A pre-specified group-conditional (Mondrian) partition recovers a significant
part of the deficit — +0.064 [0.031, 0.105], holding without 2012 and on the
clean origins — at a measured cost to the normal class of −0.019
[−0.042, −0.003]. That is the most robust finding in the study, and it is a
remedy a practitioner can apply today with no change to the underlying model.

### 2.2 The marginal-calibration trap

The pattern in §1.8 is the result with the clearest implication for practice.
If a method is chosen by marginal ACE — the standard reported quantity — the
selection lands on SA-ACI, which has the best marginal calibration (0.0241) and
the worst extreme-class miss rate (0.134) and largest shortfall (0.816 t/ha) of
the six. The mechanism is not mysterious: adaptive methods buy their tight
marginal calibration by narrowing intervals on average, and average behaviour is
dominated by the 74% of county-years in the normal class. The narrowing is
withdrawn from precisely the stratum that needed it.

The strong form of this — that marginal calibration quality *predicts* conditional
failure — is not something a 202-row stratum can establish, and §1.8 shows the
rank correlation collapsing under a one-row perturbation. The weak form is enough
for the practical conclusion and is not in doubt: marginal adequacy carries no
information about conditional adequacy, because all six methods have the former
and none has the latter. A practitioner who reads only PICP cannot distinguish
the best method here from the worst.

Recent applied work reports marginal coverage and stops. Two 2026 examples:
a conformalised graph-ensemble yield model reports "80.72% empirical coverage"
against an 80% target, and a conformal nitrogen-prescription framework reports
0.912 against 0.90. Both are correct marginal statements. Neither can tell a
user whether the interval holds in a drought year, which is the only year in
which an interval on crop yield changes a decision.

### 2.3 Why the obvious fixes do not work

Three natural responses fail here, and the reasons generalise.

*Widen everything.* Uniform inflation degrades the 74% of county-years that were
already calibrated, buys extreme-class coverage at the worst possible exchange
rate, and does not address a centre that is misplaced.

*Adapt width to local difficulty.* Locally adaptive conformal is the best of the
six on extreme-class consequence, which is real and worth noting — but it still
misses at 0.094, nearly double target, because difficulty-scaling estimated from
calibration data cannot anticipate a year unlike any in that data.

*Adapt online.* ACI's adjustment operates on realised coverage, which arrives
after the season it was needed for. In our rolling-origin design each origin has
a single test year, so ACI's adaptivity is not exercised and the static-family
methods coincide exactly — a limitation of the design, but also a fair
representation of the operational case, where a forecast is issued once and the
outcome is learned at harvest.

What does work is conditioning the calibration on a variable known *before* the
outcome. The CDHW event count is available from in-season weather, so a
Mondrian partition on exposure class is deployable; and it helps for a reason
that survives the 2012 stress test only partly, which is honest about its limit.

### 2.4 Implications for practice

A crop-yield prediction interval used for credit, insurance or procurement is
consumed asymmetrically: the lower bound carries the risk. We therefore
recommend reporting, alongside PICP and MPIW, the **one-sided unsafe miss rate
against α/2** and the **conditional shortfall in physical units**, stratified by
a pre-specified exposure variable. On this data those quantities reorder the
method ranking; on any data they are cheap to compute.

The stratification must be pre-specified. Ours was fixed from the CDHW event
count before any coverage result was seen (Normal 0, Moderate ≥ 1, Extreme ≥ 3),
and the frozen protocol records it. A stratum chosen after inspecting coverage
would be a different and much weaker claim.

### 2.5 Limitations

*The central contrast is not individually significant.* RO1's Normal-minus-
Extreme difference has an interval containing zero, as does RO5's within-quintile
contrast and RO4's excluding-2012 subset. The claims that do clear their nulls
are the extreme-class excess over target on the full pool and clean origins, and
the Mondrian gain in every subset. We have stated which is which rather than
quoting the point estimates alone.

*One year does much of the work.* 2012 supplies 39.5% of extreme-exposure rows.
Dropping it attenuates the coverage gap from 0.758 to 0.863. We consider its
inclusion correct — it is the event the method is supposed to survive — but a
reader who disagrees can read every sensitivity in the report.

*Design and scope.* One test year per origin, so ACI adaptivity is untested;
LightGBM only in the rolling-origin analysis; irrigated share is not controlled;
phenology windows use fixed GDD thresholds rather than observed silking dates;
SPI/SPEI standardisation is fixed to 1985–2013, which overlaps the test year for
origins 2008–2013, so the clean-origin sensitivity is reported throughout. Six
Corn Belt states and one crop; the exposure thresholds are not claimed to
transfer unchanged.

*Seed dependence — not a limitation for the interval results, and not a virtue
either.* The rolling-origin intervals are reported from one seed, and averaging
over more would change nothing: the quantile learners that generate them specify
no `subsample` and no `colsample_bytree`, and LightGBM disables bagging without
`subsample_freq`, so they are deterministic functions of their training data.
Repeating all 16 origins at five seeds (42, 7, 123, 2024, 3407) reproduces PICP,
unsafe miss rate, MPIW and Winkler exactly — max |Δ| = 0.0 over 244
origin × method × stratum cells, for all four calibrators including the two ACI
variants that receive the point predictions. The property is asserted as a
regression test in
`code/diagnostics_loso/verify_seed_sensitivity.py`, which also fails if a
stochastic parameter is ever added to that learner. The single-seed objection
therefore does not apply to RO1, RO2, RO4 or RO5.

The same fact denies us a remedy. Seed averaging cannot stabilise these
intervals, so if the calibration is wrong it is reproducibly wrong, and
reproducibility here is not evidence of correctness. The point predictions are a
separate matter: they use the 80%-feature-subsampling configuration and do move
with the seed (R² by 0.024 and bias by 0.099 t/ha at origin 2012), so the
point-accuracy statements in §1.3 carry seed uncertainty that the coverage
statements do not. Those statements were therefore re-run over five seeds and
are reported with that dispersion; the sweep is in
`rolling_origin_seed_sweep.csv` and `_by_origin.csv`.

*The CDHW predictor null is a null.* Three seeds and five model families cannot
establish the absence of a small effect, only bound it as smaller than seed
noise on this feature set.

### 2.6 Reproducibility

The protocol is hashed and verified against the live configuration on every run
(1.4.0-2026-09-23, `9ea6b15d86c6ebef`). The full pipeline was executed
independently on two platforms; every point metric, coverage figure and interval
reproduced exactly, with differences confined to inference latency. The
hyperparameter selection resolving audit item C7 is a committed artefact rather
than a remembered procedure, and the development history that made that
correction necessary is disclosed verbatim in
`docs/supplement/development_history.md`.

---

## 3. Known reporting defects to fix before submission

1. `comprehensive_metrics.csv` and `backbone_benchmark.csv` place raw model
   coverage and conformally calibrated coverage in one column. Split them.
2. `predictions.csv` stores values at 4 dp. We checked what this can affect: of
   14 070 rows, six lie within 5e-4 of an interval bound and **all six are at the
   upper bound**, so no unsafe-miss flag is ambiguous and no consequence metric
   in this draft is at risk. Widen it anyway before any future recomputation, and
   do not assume the margin holds after a rerun.
3. `loso_summary.csv` reports a macro 95% CI across six folds; with six clusters
   this is a wide interval and should be labelled as such wherever quoted.
