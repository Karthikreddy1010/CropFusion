# LITERATURE_GAP_MAPPING.md — Parts 3 & 15

> ## ⚠️ SUPERSEDED IN PART — READ `LITERATURE_CITATION_VERIFICATION.md` FIRST
>
> This mapping was built on the assumption that the 36 papers in
> `Literature_Review_Paper3_ConformalInference.xlsx` are real. Subsequent
> verification found **15 of them cannot be found as real publications**,
> including Papers 18 and 21 — the two this document repeatedly treats as
> the anchor citations for the paper's central gap and its novelty claim.
>
> Every reference below to Papers **13, 14, 15, 18, 19, 20, 21, 22, 23, 24,
> 26, 27, 28, 29, 30** is therefore citing an unverifiable source. The
> topic-by-topic analysis of *what this codebase actually implements and
> measures* remains valid; the claim that each corresponds to a documented
> gap in prior literature does not, wherever it rests on those rows.
>
> Papers **1–12, 16, 17, 25, 31, 32, 33, 36, 37, 38** were all verified real,
> so gaps grounded in those (conformal theory, exchangeability, compound
> drought-heat climate science) stand.

Source: `Literature_Review_Paper3_ConformalInference.xlsx` (36 papers reviewed,
numbered 1–33, 36–38). This mapping is deliberately skeptical: a gap is marked
"Addressed" only when a real, executed experiment in this codebase actually
tests the claim — not merely because the methodology document mentions it.
Every "Evidence in code" cell below was independently verified against the
actual `code/` files and, where marked, this session's real pipeline run —
not assumed from the methodology's own prose.

---

## Topic-by-topic mapping (Part 3's 15 focus areas)

### 1. Conformal prediction for agricultural yield prediction
- **Literature:** rows 14 (Bayesian NN yield UQ), 15 (UQ deep learning yield),
  16 (Quantile Regression Forests), 18 (static conformal failure in ag —
  the paper's own central motivating problem), 20 (ag risk assessment
  ensemble), 21 (phenology-aware CQR), 22 (ENSO mediation).
- **What the methodology does:** positions this paper as directly following
  Paper 18's documented failure mode and extending Paper 21's static
  phenology-aware CQR into an online-adaptive setting.
- **Evidence in code:** `aci_calibrator.py` implements 5 real calibration
  methods including `phenology_stratified_cqr` (Paper 21's exact protocol)
  and `static_conformal` (Paper 18's protocol), both actually run on real
  yield predictions in every pipeline run, not just described.
- **Experiment that tests it:** the primary temporal-split evaluation, all
  5 calibration methods applied to the same backbone's predictions.
- **Metric:** PICP, MPIW, ACE, Winkler score, per calibration method.
- **Addressed?** **Yes.**
- **Strength: Strong.** This is the paper's central, actually-executed
  comparison, not a claim resting on citation alone.

### 2. CQR versus conventional prediction intervals
- **Literature:** row 3 (Romano et al., Conformalized Quantile Regression),
  row 6 (classical quantile regression, no coverage guarantee), row 16 (QRF).
- **What the methodology does:** builds the CQR head per Romano et al.'s exact
  formulation (pinball loss at τ=0.05/0.95, conformal correction on residuals).
- **Evidence in code:** `model_training.py::NeuralCQRNet` (q05/q95 heads +
  pinball loss), `aci_calibrator.py::static_conformal` (the conformal
  correction step).
- **Experiment:** ablation row `3_plus_cqr` vs `1_backbone_only`/`2_plus_cdhw`
  isolates the CQR head's effect on interval metrics (PICP/MPIW go from n/a to
  real numbers only once this head exists).
- **Metric:** PICP, MPIW.
- **Addressed?** **Yes**, as "does adding a CQR head produce valid,
  measurable intervals" — genuinely tested via the ablation.
- **Strength: Strong** for "CQR produces valid intervals here." **Not
  tested** — and the methodology does not claim — a head-to-head CQR vs.
  a non-conformal quantile-regression baseline's *coverage validity*
  specifically (e.g., raw quantile regression without conformal correction);
  the tree-baseline quantile models (LightGBM/CatBoost/XGBoost quantile
  regressors) exist in the code but are not run through the same 5-method
  calibration/coverage evaluation as the primary NeuralCQR backbone in the
  observed pipeline flow. **This narrower comparison is Not Addressed.**

### 3. Conditional / adaptive uncertainty
- **Literature:** row 4 (ACI, Gibbs & Candès), row 9 (weighted/covariate-shift
  conformal), row 10 (local conformal bands), row 17 (locally adaptive
  conformal), row 23 (distributionally robust conformal).
- **What the methodology does:** implements ACI as the primary contribution,
  and explicitly positions O6 as a head-to-head test against Paper 9's and
  Paper 17's methods specifically (not just against static conformal).
- **Evidence in code:** all three — `adaptive_conformal_inference`,
  `weighted_conformal`, `locally_adaptive_conformal` — implemented and (per
  the code read in this session) all actually invoked in the primary
  evaluation's calibration block, not just static conformal vs. ACI.
- **Experiment:** `evaluate_objective_o6`, called on the full `eval_report`
  containing all 5 methods' metrics.
- **Metric:** PICP/MPIW comparison across methods, with Wilcoxon +
  Holm-Bonferroni correction (§5.5) for the pairwise significance claims.
- **Addressed?** **Yes** — this is a genuinely executed, multi-method
  comparison, not a single ACI-vs-static claim dressed up as "beats the
  field."
- **Strength: Strong, and confirmed** (real run): ACI achieves the best
  Winkler/ACE/MPIW of all 5 methods, beating every alternative at
  Wilcoxon p < 10⁻¹⁴², Holm-Bonferroni corrected.

### 4. Extreme heat / compound heatwave effects
- **Literature:** rows 11–13 (typology, attribution, global prevalence), 29
  (North America CDHW maize study — closest direct precedent), 31–33
  (WUE, cascading droughts, CMIP6 projection), 37–38 (data-gaps review,
  global compound heat-moisture).
- **What the methodology does:** builds CDHW as a joint SPEI+Tmax indicator,
  phenology-aligned, following Paper 37's explicit recommendation.
- **Evidence in code:** confirmed in `CDHW_DESIGN_REPORT.md` — real,
  upstream-computed phenology-stage severity columns are used, not a
  degenerate fallback.
- **Experiment:** the CDHW ablation (current and improved rows) plus the
  drought×heat interaction analysis.
- **Metric:** ΔR²/ΔRMSE (ablation), interaction coefficient + p-value
  (interaction analysis).
- **Addressed?** **Partially, with an honest negative component.**
  Predictive-contribution is genuinely tested, and the real result is that
  CDHW **reduces** this backbone's aggregate R² (current: −0.015; improved,
  tested in this audit: −0.101 further) — CDHW does not improve point
  prediction here. The interaction analysis finds real, if
  specification-dependent, statistical association (significant for the
  continuous specification, not for the primary binary one). The
  literature's strongest claims (Paper 29: compound CDHW causes 2.3×
  drought-alone and 1.8× heat-alone damage; Paper 32: ~50% vs ~25% yield loss)
  are **magnitude/causal, mechanistic crop-model findings** (APSIM, WOFOST)
  — this paper's associational regression on observational county-year data
  cannot and does not attempt to reproduce those specific multipliers, and
  should not cite them as if this paper's own result matches them.
- **Strength: Moderate-to-Weak** — real tests were run, but the predictive-
  contribution result is negative (not merely "not yet confirmed"), and the
  quantitative superadditivity claims from the crop-modeling literature are
  not, and cannot be, directly replicated by this paper's design. The paper
  should frame CDHW's value around interpretability and the interaction
  finding, not as an accuracy-improving input feature.

### 5. Phenology-aligned climate stress
- **Literature:** row 21 (phenology-aware CQR), row 24 (drought-timing
  meta-analysis — pollination stress ~4× vegetative-stage stress).
- **What the methodology does:** weights CDHW severity by growth stage
  (silking/R1 weighted highest, `SILKING_WEIGHT_PRIMARY=1.5` vs.
  `VEG_WEIGHT_PRIMARY=1.0`/`GRAIN_WEIGHT_PRIMARY=0.8`), consistent with
  Paper 24's finding.
- **Evidence in code:** real upstream stage-severity columns, confirmed
  non-degenerate (§1 of `CDHW_DESIGN_REPORT.md`).
- **Experiment:** phenology-stratified CQR (Paper 21's protocol, directly
  run as one of the 5 calibration baselines) and O5 (interval width by
  phenological window).
- **Metric:** PICP/MPIW by phenology window; O5's width-severity Spearman
  correlation.
- **Addressed?** **Yes**, for "does the model use phenology-stage-specific
  information" — genuinely wired in and genuinely tested against a
  phenology-only baseline (Paper 21).
- **Strength: Strong.**

### 6. Spatial generalization
- **Literature:** row 26 (spatially aware conformal / leave-one-zone-out).
- **What the methodology does:** LOSO-CV across 6 states (§5.2), explicitly
  framed as the spatial-generalization analog of Paper 26's protocol.
- **Evidence in code:** `_run_loso_cv`, confirmed working end-to-end
  (6/6 folds) in this session's runs.
- **Experiment:** LOSO-CV.
- **Metric:** per-fold RMSE/MAE/R²/PICP/MPIW.
- **Addressed?** **Yes**, scoped correctly (see Part 4 / §5.2 of the
  methodology) — genuine spatial generalization *within the six-state Corn
  Belt panel*, not nationwide.
- **Strength: Strong**, within the stated scope; would be **Not Addressed**
  for any claim of national/cross-region generalization (correctly not
  claimed after this audit's methodology correction).

### 7. Temporal generalization
- **Literature:** row 19 (temporal autocorrelation correction), row 4 (ACI
  under distribution shift).
- **What the methodology does:** the entire temporal-split design (train
  1985–2015, test 2019–2023) is itself a temporal-generalization test; §4.4
  documents temporal autocorrelation as a known, only-partially-mitigated
  limitation.
- **Evidence in code:** the backbone robustness check — genuinely run across
  3 seeds — found validation R² and test R² are **not reliably correlated**
  across seeds, meaning the 2016–2018 validation window does not reliably
  rank models for 2019–2023 test performance. This is a real, unplanned,
  executed finding.
- **Experiment:** `run_backbone_robustness_check`.
- **Metric:** Spearman correlation between per-seed val R² and test R².
- **Addressed?** **Yes** — and this is, ironically, some of the paper's
  strongest evidence *for* its own thesis (static validation-based
  model/hyperparameter selection doesn't reliably transfer under the exact
  kind of distribution shift the paper is about), found via a robustness
  diagnostic rather than a dedicated "temporal generalization" experiment.
- **Strength: Strong**, and worth foregrounding in the paper, not burying as
  a robustness footnote.

### 8. State-level generalization
- Same evidence as Topic 6 (LOSO-CV). **Addressed, Strong**, same scope caveat.

### 9. Distribution shift
- **Literature:** row 4 (ACI), row 5 (conformal beyond exchangeability), row 9
  (covariate shift), row 23 (distributionally robust conformal).
- **Evidence in code:** ACI's whole design is a distribution-shift response;
  the temporal-split anomalous-year evaluation (§4.5) and the backbone
  robustness check (Topic 7) both directly probe shift sensitivity.
- **Addressed?** **Yes.**
- **Strength: Strong.**

### 10. Calibration under extreme events
- **Literature:** row 18 (the paper's central motivating finding: static
  conformal collapses to 73% PICP in compound CDHW years, from 91% overall).
- **Evidence in code:** `evaluate_winkler_by_year_type_and_enso`,
  Year-Type stratification (Normal/Moderate/Extreme via CDHW event count).
- **Experiment:** genuinely executed, stratified PICP/MPIW/Winkler by
  year-type.
- **Addressed?** **Yes, and the honest result is a partial shortfall**: real
  PICP is 0.933 (normal), 0.826 (moderate), 0.820 (extreme) — ACI's coverage
  measurably degrades in exactly the conditions this paper is about, and does
  not reach the methodology's own ≥0.90 target in anomalous years.
- **Strength: Moderate.** This is the paper's central claim, tested with real
  data, with an honest, non-trivial finding: ACI clearly outperforms every
  alternative calibration method (Topic 3), but does not fully eliminate the
  coverage-collapse failure mode documented in Paper 18 — it mitigates it.
  Report both facts.

### 11–12. Prediction interval coverage / width
- Directly measured (PICP, MPIW) for all 5 methods, all year-types, all
  ablation rows. **Addressed, Strong** — these are the pipeline's most
  thoroughly instrumented metrics.

### 13. Tail / extreme-event performance
- **Literature:** row 27 (Extreme Value Theory — GEV/GPD tail modeling,
  P5 coverage improving from 72%→89% via EVT).
- **Evidence in code:** **none.** No GEV/GPD fitting, no explicit tail-focused
  metric (e.g., P5/P95 coverage specifically, as distinct from the P05/P95
  CQR heads' ordinary coverage) exists anywhere in this codebase.
- **Addressed?** **Not addressed.** The year-type stratified PICP/MPIW
  (Topic 10) is a related but distinct concept (stratifying by a
  CDHW-severity year-label, not by fitting a tail distribution to residuals).
  **Do not claim this paper applies or benchmarks against EVT-style tail
  modeling** — it does not, and should not cite Paper 27's specific
  improvement numbers as if replicated here.

### 14. Interaction / nonlinear effects of heat and drought
- See `CDHW_DESIGN_REPORT.md` §5 for the full, already-executed result:
  **specification-dependent** — significant for the continuous interaction,
  not significant for the binary flag once state+year fixed effects are
  added. **Addressed, but the honest finding is mixed, not a clean
  confirmation** — see `CLAIM_AUDIT.md`.
- **Strength: Moderate** (real test run, real result, but the result itself
  is only partially confirmatory).

### 15. Whether existing literature combines these components in one framework
- Scanning all 36 reviewed papers: no single paper combines (a) an online/
  adaptive conformal method, (b) a compound drought-heat (CDHW) predictor,
  (c) phenology-stage alignment, and (d) agricultural yield prediction with
  spatial (LOSO) generalization, together, in one framework. The closest
  individual precedents are Paper 21 (phenology-aware CQR, but static, not
  adaptive) and Paper 4 (ACI, but generic, not ag/CDHW-specific). **This
  supports a genuine, but narrow, novelty claim**: combining online ACI with
  a CDHW/phenology-aware backbone for agricultural yield is not demonstrated
  elsewhere in this review. It does **not** support a claim that any
  individual component (CQR, ACI, phenology-conditioning, compound-event
  indices) is itself novel — each exists individually in the reviewed
  literature, sometimes in more sophisticated forms (e.g., Paper 27's EVT
  tail modeling, which this paper does not implement).

---

## Final gap-mapping table (Part 15 format)

| Literature Gap | My Method | Experiment | Evidence | Addressed? | Strength |
|---|---|---|---|---|---|
| Static conformal fails in ag during anomalous years (Paper 18) | ACI + 4 alternative calibration baselines | Primary temporal-split eval, all 5 methods | `aci_calibrator.py`, real run PICP/MPIW/Winkler by year-type | Yes | **Strong** |
| No unified CQR tutorial/benchmark for ag (Paper 3, 6, 16) | CQR head w/ pinball loss + conformal correction | Ablation row `3_plus_cqr` | `model_training.py`, `aci_calibrator.py::static_conformal` | Yes (valid intervals produced) | **Strong** |
| Conditional/adaptive UQ under shift (Paper 4, 9, 17, 23) | ACI vs. weighted vs. locally-adaptive conformal | O6 | `evaluate_objective_o6`, all 3 methods actually run | Yes | **Strong** |
| Compound drought-heat superadditivity (Paper 13, 29, 32, 33, 37, 38) | CDHW feature (current + improved), interaction regression | Ablation + `run_cdhw_interaction_analysis` | `CDHW_DESIGN_REPORT.md`; real ablation shows CDHW *reduces* R² (current: −0.015; improved: −0.101 further) | Partially (negative predictive result; specification-dependent interaction) | **Moderate-to-Weak** |
| Phenology-stage-dependent stress sensitivity (Paper 21, 24) | Stage-weighted CDHW severity, phenology-stratified CQR | O5, phenology-stratified calibration baseline | Real upstream stage columns, Paper 21 baseline implemented | Yes | **Strong** |
| Spatial generalization untested in ag conformal work (Paper 26) | LOSO-CV, 6 states | `_run_loso_cv` | 6/6 folds run, per-fold metrics | Yes (scoped to 6 Corn Belt states) | **Strong** |
| Temporal autocorrelation / distribution shift (Paper 19, 4, 5) | Temporal split, backbone robustness check | `run_backbone_robustness_check` | Val/test R² correlation across seeds | Yes | **Strong** |
| Coverage collapse specifically during compound/extreme years (Paper 18) | Year-type stratified evaluation | `evaluate_winkler_by_year_type_and_enso` | Real run: PICP 0.933→0.826→0.820 (normal→moderate→extreme) — ACI mitigates but does not eliminate the collapse (≥0.90 target not met in anomalous years) | Yes, with an honest partial-shortfall finding | **Moderate** (real, central result; genuinely tested, but the outcome is a partial, not full, success) |
| Tail-risk / EVT-style extreme modeling (Paper 27) | — | — | No GEV/GPD fitting anywhere in codebase | **No** | **Not addressed** |
| No framework combines ACI + CDHW + phenology + ag spatial generalization | Full pipeline | End-to-end pipeline | Confirmed absent elsewhere in the 36-paper review | Yes, narrowly | **Moderate** (narrow, not sweeping, novelty claim) |

**Do not manufacture novelty beyond what this table supports.** In particular,
do not claim: (a) this paper is the first to use CQR or ACI (both are cited,
adopted methods, correctly attributed already in §1 of the methodology); (b)
this paper matches or exceeds the crop-model literature's (Papers 29, 32)
specific superadditivity magnitudes — it does not attempt to, being an
associational statistical model, not a mechanistic crop simulator; (c) this
paper performs tail/EVT-style extreme-value analysis — it does not.
