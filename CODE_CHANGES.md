# Code Changes — Audit Remediation

Every change below removes a demonstrated implementation error. The locked
experimental protocol is unchanged: six LOSO states (`Illinois, Indiana, Iowa,
Minnesota, Missouri, Ohio`), temporal FIT 1985–2013 / DEV 2014–2015 / CAL
2016–2018 / TEST 2019–2023, same dataset, same feature definitions, same model
families, same evaluation metrics.

---

## 1. `code/main.py` — `_run_loso_cv()`

**FILE:** `code/main.py` (function `_run_loso_cv`, fully rewritten)

**PROBLEM:** Three independent contaminations, all stemming from one line:

```python
X_tr_full = np.concatenate([X_tr, X_va], axis=0)
y_tr_full = np.concatenate([y_tr, y_va], axis=0)
```

* The LOSO validation fold was concatenated into the training matrix, and the
  internal early-stopping split was carved from the *tail* of the combined array
  — which is the tail of `X_va`. So ~90% of the validation fold was used for
  model fitting and the remaining ~10% for early stopping.
* The ensemble weight was then selected by scoring `preds_val` against `y_va` —
  data the models had just trained on.
* `static_conformal(y_va, q_lo_val, q_hi_val, ...)` used those same rows again
  for conformal calibration. There was no calibration partition at all.
* Feature selection was inherited from the main temporal pipeline, which had
  fitted it on **all** states including the held-out one.

**CHANGE:** The fold now has four disjoint roles:

| Partition | Years (non-held-out states) | Allowed use |
|---|---|---|
| `fit` | 1985–2013 | model fitting; the chronological last 10% is carved off *within fit* as the internal early-stopping split |
| `dev` | 2014–2015 | ensemble-weight selection only |
| `cal` | 2016–2023 | conformal calibration only, after the weight is frozen |
| `test` | held-out state, all years | final evaluation only |

Each fold re-runs consensus feature selection on its own `fit` partition, builds
a `PartitionLedger` of real county-year IDs, and calls `assert_disjoint()` before
a single model is fitted — so a violation aborts the run rather than surfacing in
a report. Every candidate ensemble weight is scored on `dev` and written to
`loso_ensemble_weight_search.csv` with an `is_selected` flag; the winner is
frozen before calibration touches anything.

**WHY:** The held-out state and the validation fold must not influence any
learned component. Under the old code every LOSO metric was optimistic for three
separate reasons at once, and there was no way to tell how much of the reported
R² was genuine spatial generalization.

---

## 2. `code/splitting.py` — `loso_cv_folds()`

**FILE:** `code/splitting.py`

**PROBLEM:** The generator produced only train/val/test; `val` was everything from
2016 onward, leaving no calibration partition. The module docstring said "7
spatial folds". `Optional` was used in an annotation but never imported.

**CHANGE:** Added `return_cal=True` yielding `(state, fit, dev, cal, test)` using
the project's own locked `FIT_YEARS` / `DEV_YEARS` / `CAL_YEARS` boundaries
within the five non-held-out states. The legacy `return_val` 4-tuple still works
(`val = dev + cal`) so existing audit and test callers are unaffected. Added an
assertion that the three development partitions exhaustively partition the
development-state rows. Added `add_obs_ids()` / `obs_ids()` helpers that attach
`obs_id = "<GEOID>_<Year>"`. Corrected the docstring to six folds and imported
`Optional`.

**WHY:** Calibration observations must be separate from model-selection
observations. Reusing the project's existing locked year boundaries keeps this a
role separation rather than an invented split, and leaves the six-state list
untouched.

---

## 3. `code/leakage_provenance.py` — new module

**FILE:** `code/leakage_provenance.py` (new)

**PROBLEM:** Leakage was "verified" with checks like `id(res.q_lo) != id(res.q_hi)`.
Object identity says nothing about which observations influenced a learned
parameter: two arrays can be distinct objects derived from identical rows. These
checks could never have detected the LOSO contamination that was present.

**CHANGE:** `PartitionLedger` registers the observation-ID set each stage
consumed and binds stages to partitions (`model_fitting -> train`,
`conformal_calibration -> cal`, …). `audit_ledger()` computes every required
pairwise overlap on real county-year IDs; `assert_disjoint()` raises on any
violation. `LeakageAuditCollector` accumulates per-experiment records and exports
`leakage_provenance_audit.json` / `.md`.

**WHY:** A leakage check must test the statistical property (which rows reached
which stage), not a memory-address coincidence.

---

## 4. `code/model_training.py` — genuine joint vs post-hoc paths

**FILE:** `code/model_training.py`

**PROBLEM:** `joint_training` was consumed by exactly one statement:

```python
mode_str = "Joint End-to-End" if joint_training else "Post-Hoc"
```

Both settings then ran the identical `cqr_composite_loss` through the identical
optimizer. The two "conditions" were the same computation with different labels,
which made ablation rows 4 and 5 and the whole of O4 vacuous.

**CHANGE:** Two separate code paths.

* **Joint (Condition B)** — one optimization stage over the full composite
  objective; gradients from the quantile heads flow into the shared backbone.
* **Post-hoc (Condition A)** — `_train_neural_cqr_posthoc()`: stage 1 trains the
  backbone and mean head on Huber **alone** with the quantile heads excluded from
  the optimizer (`requires_grad_(False)`); stage 2 freezes the trunk
  (`requires_grad_(False)` plus `eval()` so BatchNorm running statistics stop
  moving) and fits only the three quantile heads on pinball + crossing loss over
  the fixed representation.

Each returns `training_paradigm` and a `paradigm_fingerprint` recording stage
count, loss terms per stage, and whether the backbone received quantile
gradients.

**WHY:** §4.2 of the methodology defines the contribution as training the
interval head jointly with the backbone "rather than calibrating post-hoc on
fixed backbone outputs". The two arms must differ in optimizer parameter groups,
loss, and resulting predictions — not in a log string.

---

## 5. `code/main.py` — `_run_ablation()`

**FILE:** `code/main.py` (function `_run_ablation`)

**PROBLEM:** Rows 4 and 5 differed only by the inert `is_joint` flag, so they were
mathematically identical (the prior run recorded byte-identical
1.2873 / 0.4232 / 0.9207 / 4.9354 for both). Rows 1 and 2 claimed to be
"point-prediction-only" but were trained with the full composite loss including
pinball, crossing and width terms, so row 2 → row 3 was also not a real step.

**CHANGE:** `use_cqr=False` now trains with `lambda_pinball=lambda_crossing=
lambda_width=0.0` — a genuine point-only model. `use_cqr=True` dispatches to the
real joint or post-hoc path. Each row records `training_paradigm`,
`n_optimization_stages`, per-stage loss terms and a `prediction_checksum`, and
the report ends with `configuration_distinctness_checks` that FAIL if two rows
claiming different methodologies produced identical predictions.

**WHY:** An ablation whose rungs are the same computation measures nothing.
Identical results between 4 and 5 had to be provable as a bug rather than
interpreted as "no effect".

---

## 6. `code/o4_experiment.py` — new module; `code/evaluation.py` — `evaluate_objective_o4()`

**FILE:** `code/o4_experiment.py` (new), `code/evaluation.py`

**PROBLEM:** O4 was called as

```python
evaluate_objective_o4(eval_report["methods"]["sa_aci"],
                      {"static": ..., "phenology": ...})
```

— one *calibration method* against other *calibration methods*. All of them
share the same point predictions by construction, so the reported "RMSE
improvement" was structurally zero and the experiment could not detect a
training-paradigm effect in either direction. Conformal calibration changes
intervals, not point predictions.

**CHANGE:** `run_o4_joint_vs_posthoc()` runs a controlled paired experiment:
identical FIT/DEV/CAL/TEST partitions, identical feature set and fitted scaler,
identical architecture, optimizer, learning rate, weight decay, batch size,
epoch budget, early-stopping patience, random seed, and identical conformal
calibration applied to both arms. The only difference is the training paradigm.
It reports RMSE / MAE / R² and PICP / MPIW / ACE / Winkler under both static
conformal and ACI, as absolute and relative differences, with year-block cluster
bootstrap and year-level paired tests, plus a distinctness check that FAILs if
the arms produced identical predictions. `evaluate_objective_o4()` consumes this
and emits a verdict that follows the measured result; it can return "Not
supported".

**WHY:** This is the actual research question — whether genuine joint end-to-end
training beats the intended post-hoc approach — and it needs an instrument that
can answer it in either direction.

---

## 7. `code/dependence_aware_stats.py` — new module

**FILE:** `code/dependence_aware_stats.py` (new)

**PROBLEM:** Every significance claim treated ~2,300 county-year test rows as
independent samples, despite substantial lag-1 autocorrelation and strong
within-year spatial correlation. p-values were anticonservative by orders of
magnitude and slope confidence intervals far too narrow.

**CHANGE:** `cluster_bootstrap_mean`, `cluster_bootstrap_paired_difference`,
`cluster_robust_ols` (CR1 small-sample correction, t-distribution with
`n_clusters − 1` df), `year_level_paired_test`, `effective_sample_size` (ICC and
design effect), `nemenyi_critical_difference`, and `friedman_on_blocks`. Each
reports its unit of independence, and refuses to return a p-value when there are
too few independent clusters rather than producing one anyway.

**WHY:** Item 12 of the audit — inference must match the dependence structure,
and the reader must be able to see the unit of independence behind every claim.

---

## 8. `code/evaluation.py` — `evaluate_objective_o5()`

**FILE:** `code/evaluation.py`

**PROBLEM:** Used `pearsonr` / `spearmanr` / `linregress` / ANOVA over correlated
rows. `Year_Type` is a year-level label, so the ANOVA across year types compared
thousands of correlated rows as if each were an independent draw.

**CHANGE:** Primary inference is cluster-robust OLS clustered by year,
corroborated by a year-block cluster bootstrap of the slope, with the effective
sample size (ICC, design effect) reported alongside. The year-type comparison is
done on year-level means. Naive p-values are retained but explicitly labelled
invalid, to document the size of the distortion. The report states statistical
detectability and effect magnitude as two separate conclusions, with the
magnitude label derived from R² against stated bands.

**WHY:** Item 13 — a slope can be reliably non-zero while explaining almost none
of the variance, and the report must not let one imply the other.

---

## 9. `code/evaluation.py` — `evaluate_objective_o6()`

**FILE:** `code/evaluation.py`

**PROBLEM:** Four separate defects:
* `q_alpha = 2.728 if k == 5 else 2.569` — 2.569 is the constant for **k = 4**,
  so the six-method comparison used a critical difference that was far too small.
* Friedman blocked on individual observations, treating correlated county-year
  rows as independent blocks.
* The test **never actually executed**: it read
  `data["uncertainty"]["per_sample_winkler"]`, a key `evaluate_calibration()`
  never wrote. The JSON therefore emitted `"statistic": null, "p_value": null`.
* The Markdown printed "differences across conformal calibration methods are
  statistically significant (p < 0.05)" whenever the test merely ran, regardless
  of the p-value, and computed a CD without performing any pairwise comparison.

**CHANGE:** `evaluate_calibration()` now stores `per_sample_winkler` and
`run_full_evaluation()` stores `test_years`, so the test can be constructed.
Blocking is by year. The critical value is looked up by the actual `k` (2.850 for
k = 6) and returns `None` for an untabulated `k` rather than guessing. Nemenyi
pairwise comparisons are performed only when the omnibus test rejects. The
conclusion text follows the p-value, and when the test cannot be justified the
ranking is labelled "descriptive ranking only". Practical spread (Winkler, PICP,
MPIW ranges) is reported so the reader can judge whether differences matter.

**WHY:** Items 14 and 15 — no null statistic may be presented as a passed test,
and the critical value must follow the real number of methods.

---

## 10. `code/visualization.py` — `export_shap_consistency_report()`

**FILE:** `code/visualization.py`

**PROBLEM:** The function was not computing SHAP at all. Tree models used
`feature_importances_` (split-gain importance); the NeuralCQR backbone has no
such attribute and fell through to:

```python
imp = np.abs(np.random.RandomState(cfg.RANDOM_SEED).randn(d))
```

— **random numbers reported as feature attributions**. The `X_sample` argument
was accepted and never used. One of four backbones in the cross-model agreement
statistic was pure noise, which largely explains the low similarity. The
Markdown additionally hard-coded "(High cross-backbone attribution alignment)"
regardless of the computed value.

**CHANGE:** Tree backbones use `shap.TreeExplainer` on real data; the neural
backbone uses `shap.GradientExplainer` against the mean head via a wrapper that
exposes a single output. XGBoost's `base_score` serialisation incompatibility is
normalised and retried, then falls back to `shap.PermutationExplainer` — still a
genuine SHAP estimator. A backbone whose attributions cannot be computed is
recorded as `unavailable` and **excluded** from the agreement statistics; nothing
is fabricated. The interpretation text is derived from the computed similarity
against stated rank-correlation bands.

**WHY:** Item 16, and basic research integrity — a published attribution figure
must not contain random numbers.

---

## 11. `code/evaluation.py` — `benchmark_computational_complexity()`

**FILE:** `code/evaluation.py`

**PROBLEM:** No warm-up iterations; no `torch.no_grad()` (so every timed pass
built an autograd graph); no CUDA synchronisation (CUDA launches are
asynchronous, so a GPU timing would measure queueing); batch size implicit; a
single throughput number derived from one whole-test-set batch; preprocessing,
inference and end-to-end latency conflated; and `peak_gpu_memory_mb` reported
without recording whether a GPU existed at all.

**CHANGE:** Warm-up iterations run and are discarded; all timed regions use
`torch.no_grad()` and are bracketed by `torch.cuda.synchronize()` when a GPU is
present; several explicit batch sizes are reported; model-inference latency,
end-to-end latency and preprocessing/transfer overhead are reported separately;
device, device name, `cuda_available`, torch version, parameter count and the
memory measurement methodology are all recorded.

**WHY:** Item 18. On this machine `torch.cuda.is_available()` is `False` (torch
2.8.0+cpu), so any GPU throughput figure would have been unfounded — the device
is now recorded explicitly so that cannot recur.

---

## 12. `code/utils.py` — dataset checksum

**FILE:** `code/utils.py`

**PROBLEM:** `hasher.update(f.read(1024 * 1024))` — the checksum covered only the
first 1 MB of a 24 MB dataset (about 4% of the bytes), while being reported as
`dataset_md5_header` in the reproducibility record. Any change after the first
megabyte — including the entire test period — produced an identical "checksum".
Artifact hashes had the same defect.

**CHANGE:** Added `file_sha256()`, which streams the whole file, and
`file_partial_md5()`, whose name states what it is. The reproducibility report
now records `dataset_sha256_full_file`, `dataset_size_bytes` and
`dataset_checksum_scope`. Artifact audit columns renamed to `SHA256_Full_File`.

**WHY:** Item 19 — a partial checksum cannot support a reproducibility claim.

---

## 13. `code/methodology_validator.py` — new module (replaces the fake validator)

**FILE:** `code/methodology_validator.py` (new); `code/main.py`
(`_generate_methodology_validation_report` reduced to a redirect)

**PROBLEM:** The old validator was a hand-written list of 15 dictionaries with
`"Status": "Pass"` typed into every row. `all_pass = all(e["Status"] == "Pass" …)`
was therefore trivially `True`, and it reported
`FULL_METHODOLOGY_COMPLIANCE_100%`, "15/15 passed" — while the LOSO folds were
training on their own validation data, the ablation was comparing a model against
itself, and one SHAP backbone was random noise. It also stated "7 state folds".
It verified nothing.

**CHANGE:** 18 property-based checks that each read a real artefact and test a
property of it: six locked LOSO states actually executed; zero overlap on real
IDs for every partition pair in every experiment; early-stopping split drawn only
from training data; temporal boundaries with no year in two partitions; feature
selection and scaler bound to the FIT partition; ensemble weight is the DEV
argmin/argmax (verified against the recorded candidate grid); joint and post-hoc
demonstrably distinct; ablation rows distinct; no stage other than
`final_evaluation` bound to `test`; no stale seven-fold reference anywhere;
dependence-aware inference in use with the correct `k`; SHAP genuinely computed;
full-file checksum; benchmark methodology sound. Statuses are `PASS` / `FAIL` /
`WARNING` / `NOT_APPLICABLE`, and **a check whose evidence is missing is FAIL,
never PASS**.

**WHY:** Item 9 — a validator that cannot fail is worse than no validator,
because it launders unverified claims as verified ones.

---

## 14. `code/main.py` — compliance report, traceability matrix, final summary

**FILE:** `code/main.py`

**PROBLEM:** Three more fully hard-coded reports:
* `_generate_methodology_compliance_report()` was a fixed Markdown string ending
  "The codebase has been verified to achieve **100% Methodology Compliance** …
  with **Zero Data Leakage**", listing the superseded 1985-2015 / 2016-2018
  temporal boundaries and "Leave-One-State-Out CV (7 Folds)".
* `_generate_methodology_traceability_matrix()` wrote `"Verified": "Yes"` on
  every row without checking anything.
* `_generate_paper3_final_summary()` printed "Methodology Compliance Status: 100%
  Fully Compliant" plus four fixed conclusions — that joint training improves
  RMSE and MPIW, that interval width is "significantly positively correlated"
  with severity, that ACI "achieves top rank", and that the framework
  "demonstrates superior coverage guarantee (PICP ≥ 0.90)" — none of which were
  read from the results, and several of which contradicted them.

**CHANGE:** All three now render from computed artefacts. The compliance report
renders the validator's real check table. The traceability matrix keeps the
section→code mapping (legitimate documentation) but derives an
`Artifact Status` (PRESENT / MISSING / EMPTY) from disk and inherits
`Validator Status` from the real checks. The final summary reads O1/O4/O5/O6 and
SHAP verdicts from their JSON reports and prints a measured PICP-vs-nominal table
instead of asserting coverage.

**WHY:** Item 20 — every claim in a generated artefact must be traceable to a
computed result.

---

## 15. `code/claim_consistency_audit.py` — new module

**FILE:** `code/claim_consistency_audit.py` (new)

**PROBLEM:** Nothing checked the generated outputs for mutually contradictory or
unsupported statements.

**CHANGE:** Scans every generated `.json` / `.csv` / `.md` / `.txt` / `.log` for
the claim patterns named in the audit brief ("100% compliance", "zero leakage",
"7 folds", "high SHAP alignment", "statistically superior", "strong
relationship", "joint training improvement") and checks each against the artefact
that would have to support it, emitting `claim_consistency_audit.{json,md}` with
per-hit file, line, claim type and supporting evidence.

**WHY:** Item 20 — and it enforces the rule that reports get fixed at the source
that generates them, not by editing the artefact.

---

## 16. `code/final_audit_report.py` — new module

**FILE:** `code/final_audit_report.py` (new)

**PROBLEM:** `FINAL_METHODOLOGY_AUDIT.md` did not exist, and a hand-written one
would drift from the results exactly as the previous reports did.

**CHANGE:** Generates `FINAL_METHODOLOGY_AUDIT.md` entirely from run artefacts —
validator checks, leakage table, LOSO results, ablation, O4/O5/O6, SHAP, feature
funnel, benchmark, reproducibility, and outstanding issues.

**WHY:** Item 24.I.

---

## 17. `code/feature_engineering.py` — `build_split_aware_rolling_features()`

**FILE:** `code/feature_engineering.py`

**PROBLEM:** The `split_type == "loso"` branch handled only `tr`, `va` and `te`
and silently ignored `ca`. With a calibration fold introduced, the calibration
partition came back without any `Rolling*_mean` columns and scaling raised
`KeyError`.

**CHANGE:** The branch now builds the rolling windows over all non-held-out-state
partitions together (`fit + dev + cal`, which is exactly the old `tr + va` row
set) and maps the result back to each. The held-out state is still computed
separately and never contributes.

**WHY:** Required to support the new calibration partition; preserves the
previous computation for the rows that already had it.

---

## 18. `code/feature_selection.py` — `select_features()`

**FILE:** `code/feature_selection.py`

**PROBLEM:** Per-fold LOSO selection would overwrite the main temporal pipeline's
`feature_selection_report.*`. The funnel counts (95 candidates → 83 consensus →
73 after pruning → 74 final) were not recorded anywhere machine-readable, which
is why different reports quoted different numbers.

**CHANGE:** Added `report_suffix` so each LOSO fold writes its own artefacts, and
`n_candidates` / `n_consensus_selected` to the returned report. `main.py` now
writes `feature_selection_pipeline.json` recording all four stages and the exact
sequence.

**WHY:** Item 17 — resolve the 83-vs-74 discrepancy by documenting the real
four-stage pipeline rather than changing the methodology to make numbers match.

---

## 19. `code/config.py`

**FILE:** `code/config.py`

**PROBLEM:** Two comments referred to "the real 7-fold run".

**CHANGE:** Corrected to "six-fold". Added `LOSO_PER_FOLD_FEATURE_SELECTION` with
a comment explaining why the held-out state must not influence feature selection.
`LOSO_STATES` is untouched.

**WHY:** Item 10 — documentation must consistently state six folds.
`LOSO_STATES` was explicitly not to be modified.

---

## 20. `code/test_protocol_compliance.py` — new test module

**FILE:** `code/test_protocol_compliance.py` (new, 28 tests)

**PROBLEM:** No automated test could have caught any of the above.

**CHANGE:** Tests for exactly six LOSO states and six executed folds; zero
overlap for every LOSO partition pair; the early-stopping split being a subset of
FIT and disjoint from dev/cal/test; the locked temporal year windows with no year
in two partitions; calibration ∩ test = ∅; the scaler being invariant to
perturbations outside FIT (with a control test proving it does respond to FIT);
feature selection receiving only FIT rows; the ensemble weight being the DEV
argmin on a fixture where the test set implies a different weight; the joint and
post-hoc paths differing in paradigm, stage count, loss terms, history and
predictions; `k = 6` for Nemenyi with `q_alpha = 2.850` and a test that k=6 does
not reuse the k=4 constant; Friedman blocking on years and refusing with too few
blocks; and a provenance test proving row-ID checks catch overlap between
*distinct objects* holding the same rows — the exact case `id()` comparison
misses.

**WHY:** Item 22.

---

## 2026-09-23 — the git directory now lives outside OneDrive

**WHAT:** `.git` has been moved to `C:\Users\dukar\git-repos\Paper3.git`. The
project keeps a 44-byte `.git` *file* in its place containing
`gitdir: C:/Users/dukar/git-repos/Paper3.git`, which is git's own redirect
mechanism — the same one `git worktree` and submodules use. `core.worktree` in
the moved config points back at the project. Nothing about how you use git
changes: run the same commands from the same directory.

**WHY:** Three incidents in two days, all with the same signature — a Colab sync
writing into the project directory while OneDrive was syncing it.

1. `outputs_diagnostics/` deleted (51 tracked files), recovered from git.
2. `outputs_master/` half-replaced, leaving it 43 files short.
3. `.git` itself reduced to `objects/` and `refs/` with no HEAD, config or
   index, and an object store missing a tree — the repository could not be read
   at all. `.git/objects` was modified at 11:11 while the sync ran 11:12–11:20.

The third was recoverable only because the remote was current. A repository is
not a place to find out whether your backup strategy works, and a sync client
has no business inside `.git`.

OneDrive cannot exclude an arbitrary subfolder, and a directory junction depends
on OneDrive continuing to skip reparse points. The `gitdir:` file needs neither:
OneDrive now syncs 44 bytes of text and cannot touch the object store.

**Still exposed:** `outputs_master/` and `outputs_diagnostics/` remain inside the
synced tree, so incidents 1 and 2 can recur. Both are reproducible from a
pipeline run and both are committed, so the cost is time rather than data. When
syncing results down from Colab, pull `outputs_master/` only and never
`outputs_diagnostics/`, which Colab creates as an empty scaffold.
