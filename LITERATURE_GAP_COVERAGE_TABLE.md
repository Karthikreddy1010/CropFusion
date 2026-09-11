# Literature Gap Coverage Table

> ## ⚠️ READ `LITERATURE_CITATION_VERIFICATION.md` FIRST
>
> **This table was written before the citations themselves were verified.**
> Subsequent verification found that **15 of the 36 papers in the literature
> review cannot be found as real publications** — including Papers 18 and 21,
> the two most load-bearing citations in the entire methodology.
>
> Consequence for this table: several rows below cite band-B (unverifiable)
> papers as the *evidence from literature* establishing a gap. **A gap
> "identified by Paper 18" is not an established gap if Paper 18 cannot be
> produced.** Specifically affected rows below: **#1 (Paper 18), #3 (Papers
> 13, 29), #4 (Papers 21, 24), #5 (Paper 23), #6 (Paper 26), #7 (Paper 19),
> #8 (Paper 27), #12 (Paper 18), #13 (Papers 13, 29)**.
>
> The *code-side* columns of this table (Is It Implemented? / Where in Code? /
> Evidence from Results) remain valid — those were verified against the actual
> implementation. It is the *literature-side* columns that must be
> re-established against real sources before any of this is used in writing.
>
> Note also: several affected gaps are genuinely real phenomena that *can* be
> re-grounded in verified papers — see the "Recommended actions" section of
> `LITERATURE_CITATION_VERIFICATION.md`.

Requested column format, built from `LITERATURE_GAP_MAPPING.md` (Parts 3/15
of the original audit) plus everything found in Rounds 2–4 (the CDHW
regression, the O4/O5/O6 bugs, the O1 consistency fix). Classified strictly:
**FULLY ADDRESSED / PARTIALLY ADDRESSED / NOT ADDRESSED / CLAIMED BUT NOT
ACTUALLY TESTED / NOT RELEVANT**. A component existing in code is never by
itself treated as "addressed" — only a real, executed experiment producing
real evidence counts.

| # | Literature Gap | Evidence from Literature | What the Paper Claims to Address | Required Methodological Component | Is It Implemented? | Where in Code? | Evidence from Results | Problem/Weakness | Required Fix | **Status** |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Static conformal collapses during anomalous/extreme climate years | Paper 18: PICP drops 91%→73% in compound-event years | ACI maintains ≥0.90 coverage in anomalous years | Online/adaptive recalibration | Yes | `aci_calibrator.py::adaptive_conformal_inference`, genuine 3-yr rolling window, verified leak-free | Real run: PICP 0.933 (normal)→0.826 (moderate)→0.820 (extreme); best of 5 methods, but below 0.90 in anomalous years | Coverage degrades, doesn't fully close the Paper-18 gap | Soften the claim to "ACI mitigates but does not eliminate" coverage collapse; report the stratified numbers, not just the aggregate | **PARTIALLY ADDRESSED** |
| 2 | No validated CQR-based UQ for ag yield specifically | Papers 3, 6, 16, 18 | CQR produces sharper, valid intervals vs. fixed-width | Pinball-loss quantile heads + conformal correction | Yes | `model_training.py::NeuralCQRNet`, `aci_calibrator.py::static_conformal` | Ablation row `3_plus_cqr` produces valid non-crossing intervals where none existed before | Not tested: CQR vs. an *uncalibrated* raw quantile-regression baseline specifically | Low priority — the tree-model quantile baselines (LightGBM/CatBoost/XGBoost) already exist and could be wired into this specific comparison if wanted | **FULLY ADDRESSED** (for the tested claim); narrower CQR-vs-raw-QR claim **NOT ADDRESSED** |
| 3 | Compound drought-heat (CDHW) as a joint predictive signal | Papers 13, 29, 32, 33, 37, 38 | CDHW improves point-prediction accuracy | CDHW feature construction + controlled ablation | Yes | `feature_engineering.py`, `main.py::_run_ablation` (rows `1_backbone_only`/`2_plus_cdhw`/`2b_plus_cdhw_improved`) | Real Round-1 run: current CDHW *reduces* R² by 0.015; a tested improved representation reduces it by a further 0.101 | The construction is sound and the test is rigorous, but the specific "improves prediction" claim is **not supported** by it | Rewrite the manuscript claim: CDHW is predictively informative in the interaction analysis and useful for calibration stratification, but does not improve this backbone's aggregate accuracy — report honestly, do not force a positive framing | **PARTIALLY ADDRESSED** (genuinely tested; predictive-improvement claim specifically **NOT ADDRESSED**) |
| 4 | Phenology-stage-dependent stress timing matters for yield/uncertainty | Papers 21, 24 | Phenology-stratified calibration improves interval quality | Stage-weighted CDHW severity + phenology-stratified CQR baseline | Yes | Real upstream stage-severity columns (`CDHW_Veg/Silking/GrainFill_Severity`); `aci_calibrator.py::phenology_stratified_cqr` | Real run: phenology-stratified CQR ties/slightly underperforms static conformal and ACI; O5 stage correlations weak (Silking r=0.12, n=35 test rows) | `Silking_R1` sample too thin for confident inference; the advantage over static conformal is not demonstrated | State the sample-size limitation explicitly; do not claim phenology-stratification is superior without a significance test showing it | **PARTIALLY ADDRESSED** |
| 5 | Shift-robust conformal alternatives beyond static conformal | Papers 4 (ACI), 9 (weighted), 17 (locally-adaptive), 23 | ACI beats the *field*, not just static conformal | Head-to-head comparison, same test set | Yes | `aci_calibrator.py` (all 4 methods), `evaluation.py::evaluate_objective_o6` | Real Round-1 run: ACI best average rank of 5; Wilcoxon-significant vs. all 4. **Friedman/Nemenyi omnibus test was broken (always `null`) until the Round-3 fix — pending re-verification on a fresh run** | Numbers are provisional until re-run with the fixed Friedman test | Re-run; then confirm the omnibus test agrees with the pairwise Wilcoxon picture | **FULLY ADDRESSED methodologically; numbers PENDING RE-VERIFICATION** |
| 6 | Spatial generalization untested in ag conformal work | Paper 26 | Genuine geographic generalization | Leave-one-state-out CV | Yes | `main.py::_run_loso_cv`, 6 states | Real run: R² 0.538–0.729 across 6 states, PICP 0.856–0.973; Missouri/Minnesota weakest, consistent with their independently-measured volatility | Scope is 6 Corn Belt states, not national — correctly stated in the corrected methodology | None — already correctly scoped | **FULLY ADDRESSED** (within stated 6-state scope) |
| 7 | Temporal autocorrelation / distribution-shift robustness | Papers 19, 4, 5 | Model/calibration robust to temporal dependence | Block bootstrap, multi-seed check, exchangeability diagnostics | Yes | `evaluation.py::block_bootstrap_county/_year`, `run_backbone_robustness_check`, `compute_exchangeability_diagnostics` | Real run: lag-1 residual ACF=0.72 (`exchangeable_temporally: false`, honestly reported); 3-seed backbone R² 0.410±0.034 | Temporal exchangeability is confirmed violated — a real, disclosed limitation, not resolved | State this as an explicit limitation (already done in methodology §4.4); a spatially/temporally-aware ACI variant is future work | **FULLY ADDRESSED** (as an honest diagnostic; the underlying violation is disclosed, not fixed) |
| 8 | Tail-risk / extreme-value modeling for agricultural losses | Paper 27 (GEV/GPD, P5 coverage 72%→89%) | — (no explicit claim currently made) | EVT fit on residuals or an equivalent tail-focused metric | **No** | Nowhere — confirmed absent | N/A | Year-type stratification (Part 9 of the earlier audit) is a related but distinct, coarser mechanism | Either implement a tail-focused analysis or explicitly state this is out of scope/future work — do not let the year-type stratification imply EVT-equivalent rigor | **NOT ADDRESSED** |
| 9 | Joint end-to-end point+interval training vs. post-hoc calibration | Implicit novelty claim (no single source paper) | Joint training improves accuracy and sharpness (O4) | Genuinely distinct training procedures | **Was NOT genuinely implemented until this session** (`joint_training` flag was a no-op) | `model_training.py::train_neural_cqr` / `_train_neural_cqr_posthoc` (fixed Round 3) | All O4 numbers from before Round 3 are invalid (both procedures were identical). Fixed and verified genuinely different (smoke test); real numbers **pending fresh run** | The entire prior evidentiary basis for this claim was void | Re-run and re-evaluate O4 from scratch; do not carry forward any pre-Round-3 O4 number | **Was CLAIMED BUT NOT ACTUALLY TESTED → now genuinely testable, PENDING RE-VERIFICATION** |
| 10 | Severity-conditioned/adaptive calibration mechanism | Methodology §4.3's own described "severity-weighted" ACI | ACI adapts calibration sharpness to compound-stress severity | Severity-weighted nonconformity scoring, evaluated against the unweighted variant | **Was implemented in code but never invoked with real data until this session** | `aci_calibrator.py::adaptive_conformal_inference` (`cdhw_severity_cal/_test` args); now wired in `main.py`, evaluated in `evaluation.py::evaluate_objective_o5b_severity_aware_aci` (new, Round 3) | Smoke test confirms the mechanism is now genuinely active (different MPIW/PICP than the unweighted variant); real numbers **pending fresh run** | No real evidence existed for this mechanism until this session | Re-run and report the real O5-B comparison; do not describe the mechanism as "evaluated" in any writing based on pre-Round-3 material | **Was CLAIMED BUT NOT ACTUALLY TESTED → now genuinely testable, PENDING RE-VERIFICATION** |
| 11 | No existing framework combines adaptive conformal + CDHW + phenology + ag spatial generalization | Scan of all 36 reviewed papers | Positions this work as a novel combination | — (literature-differentiation claim, not a code claim) | N/A | N/A | Confirmed: no single reviewed paper combines all four elements; closest precedents are Paper 21 (phenology CQR, static) and Paper 4 (ACI, generic) | Novelty is narrow (a combination), not fundamental (no single component is individually new) | State the novelty claim narrowly: "combining X and Y for this domain," not "inventing X or Y" | **FULLY ADDRESSED** (as a literature-scoping claim, verified by inspection) |
| 12 | Uncertainty/interval quality specifically during extremes, not just in aggregate | Central thesis of the paper itself | Intervals remain informative/well-calibrated during extreme years | Year-type/ENSO-stratified PICP/MPIW/Winkler | Yes | `evaluation.py::evaluate_winkler_by_year_type_and_enso` | Real run: honestly shows degradation (see gap #1) | The *experiment* is real and rigorous; the *result* is a partial, not clean, success | Report the stratified table itself as the contribution, not a cherry-picked aggregate number | **FULLY ADDRESSED as an evaluation**; underlying claim **PARTIALLY SUPPORTED** |
| 13 | Nonlinear/super-additive heat-drought interaction | Papers 13, 29, 32, 33, 37, 38 | CDHW captures compound, non-additive yield damage | Interaction regression with confound controls | Yes | `evaluation.py::run_cdhw_interaction_analysis` (state+year FE, cluster-robust SE, 12-combination sensitivity grid, continuous check) | Real run: binary flag not significant (p=0.090) once controlled; continuous specification significant (p=6.8e-5); 3/12 sensitivity combinations significant | Result is specification-dependent, not a clean confirmation | Report both specifications and the sensitivity grid; never cite only the significant continuous result as if it were unconditional | **PARTIALLY ADDRESSED** |
| 14 | Interpretability of yield-prediction drivers | General ML-in-ag concern, no single source | Identify which variables drive predictions | SHAP / permutation / consensus feature importance | Yes | `visualization.py::export_shap_consistency_report`, `feature_selection.py` (4-method consensus) | Real run: consensus-selected feature list with cross-method rank agreement | Already correctly avoids causal language in generated reports | None needed | **FULLY ADDRESSED** |

## Notes on classification discipline

- Gaps #9 and #10 are the most important entries in this table: they show
  the audit finding real, previously-invisible bugs that made two claimed
  contributions **CLAIMED BUT NOT ACTUALLY TESTED** for the entire history
  of this project until this session. Any number citing O4 or O5-B from
  before Round 3 must be discarded, not merely re-labeled.
- Gap #3 (CDHW improves prediction) is the clearest example of the audit's
  "feature inclusion → predictive usefulness → mechanism evidence" ladder
  you asked to keep distinct: CDHW is correctly *constructed* (rung 1), was
  *tested* for predictive usefulness (rung 2) and found not to help, and
  *does* show up as a real mechanism in the separate interaction analysis
  (rung 3, but only for the continuous specification) — these are three
  different findings and must not be collapsed into a single "CDHW works"
  or "CDHW doesn't work" statement.
