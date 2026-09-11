> # ✅ CITATIONS CORRECTED — status note (read before citing anything here)
>
> **What happened.** Independent verification of every entry in the original
> `Literature_Review_Paper3_ConformalInference.xlsx` found that **15 of its 36
> entries could not be located as real publications** — searched both by exact
> title and independently by each entry's own claimed methods, datasets and
> numeric findings. Full evidence: **`LITERATURE_CITATION_VERIFICATION.md`**.
> The failures were confined to one band (the "ML/UQ applied to agriculture"
> papers); the conformal-prediction theory papers and the recent compound
> drought-heatwave climate papers all verified as real.
>
> **What was done.** This document has been corrected throughout. Every
> citation below now names real authors and years, drawn from
> **`LITERATURE_REVIEW_CORRECTED.md`** / `Literature_Review_Paper3_CORRECTED.xlsx`
> (33 verified entries, each with authors/venue/DOI so any of them can be
> checked in seconds). The two most consequential corrections:
>
> | Was | Now |
> |---|---|
> | §1 opened: "addresses the critical failure mode **documented in Paper 18**" (91%→73% coverage collapse) | §1 now frames it as an **open question**, grounded in Gibbs & Candès (2021), Barber et al. (2023, *Ann. Statist.* 51(2):816–845), and the real 2025 *Comput. Electron. Agric.* review stating CP under distribution shift is "not sufficiently explored" in agriculture |
> | §1 claimed this work **extends "Paper 21"**, an existing phenology-aware CQR method | No prior phenology-**conformal** work was located. Real phenology+UQ work is **Bayesian** (Lin et al. 2023, PB-CNN, *ISPRS J.* 205:50–73). The phenology-stratified CQR is now presented as **a contribution of this work** — stated as "to our knowledge" |
>
> **Net effect on the paper: the contribution got larger, not smaller.** The
> motivating gap is now an open question rather than a re-confirmation, and
> the phenology-conformal method is novel rather than derivative.
>
> **Still requires your action before submission:**
> 1. Rows marked **"CONFIRM AUTHORS"** in the corrected review have verified
>    title/journal/DOI but unconfirmed author lists — check each before citing.
> 2. Three topics have **no verified replacement** (crop-insurance probabilistic
>    framework; spatially-aware conformal; a peer-reviewed EVT crop-yield
>    paper). See §4.4 and the corrected review's D-rows. Either find real
>    sources or drop the associated claims.
> 3. The novelty claim in §1 should be re-checked against a proper bibliographic
>    database — web search is not exhaustive.
>
> *(Results, experiments and methodology corrections elsewhere in this document
> are unaffected by any of the above and stand as previously audited.)*

---

# CropFusion Research Series | Paper 3
## Marginal Calibration Is Not Enough: Conditional Coverage of Conformal Prediction Intervals for Corn Yield Under Compound Drought–Heat Stress

> **[v4] Title changed.** The previous title — *"Adaptive Conformal Inference for
> Calibrated Yield Uncertainty Under Climate Distribution Shift"* — positioned ACI
> as the paper's method. The results do not support that: ACI has the best
> marginal calibration error and the **worst conditional coverage of all five
> methods in every regime tested**. The title now states what the evidence
> supports, and makes the negative result the contribution.

**Proposed Methodology (Revised v4)** | Target Venue: Journal of Agricultural Science / Computers and Electronics in Agriculture / Agricultural Systems / Remote Sensing | July 2026

> **Revision note (v1):** This version addresses five literature-alignment gaps identified against the 38-paper review: (1) missing attribution of the CQR/ACI methods to their originating papers, (2) undifferentiated overlap with what was then believed to be an existing phenology-aware CQR method ("Paper 21" — **subsequently found unverifiable; see the warning at the top of this document**), (3) an insufficient baseline set for the distribution-shift claim, (4) a CDHW index that should use SPEI rather than SPI and should be phenology-aligned at the index level, and (5) no treatment of exchangeability violations from temporal/spatial autocorrelation. Changes are integrated inline; new or materially changed text is marked **[NEW]**.

> **Revision note (v2):** This version closes the remaining evaluation-robustness gaps flagged in reviewer-facing methodology feedback: (1) spatial generalization was previously untested — a Leave-One-State-Out cross-validation (LOSO-CV) protocol is added alongside, not instead of, the temporal split; (2) the uncertainty metric set is expanded beyond PICP/MPIW to include ACE and Winkler Score; (3) comparisons against baselines now specify a paired significance test with a multiple-comparison correction; (4) an ablation study isolating the contribution of CDHW encoding, CQR, ACI, and joint training is added; (5) a computational complexity subsection is added for practicality assessment; (6) the "model-agnostic" framing claim is scoped to what has actually been demonstrated. Changes are integrated inline; new text from this pass is marked **[NEW v2]**.
>
> **Revision note (v3, audit pass):** This pass corrects a state-count inconsistency found throughout §5.2: the study uses **six** states, not seven. Nebraska is present in the raw dataset (`Paper3_MegaDataset_SPEI_FINAL.csv`, 7 states) but is deliberately dropped (`config.py::DROP_STATES`) because it is heavily irrigated, which breaks the drought-yield relationship the model is trained to learn from the six rain-fed states — an empty/degenerate LOSO fold otherwise results. Every "7 states"/"7 folds" reference below is corrected to six, and §5.2 is expanded with the six-state diversity evidence needed to scope the paper's generalization claims honestly (Part 4 of the audit). This pass also corrects the O1 interaction-synergy claim (§3, O1) to reflect the specification-dependent result from the strengthened interaction analysis (state+year fixed effects, cluster-robust SE — see `CORRECTION_CHANGELOG.md`, "Issue 3 revisited"), and documents the improved CDHW representation tested in the ablation study (§4.1, §4.6). See `AUDIT_REPORT.md`, `CDHW_DESIGN_REPORT.md`, and `CLAIM_AUDIT.md` for the full audit this revision is based on.

> **Revision note (v4, literature-driven redesign).** This pass rebuilds the
> methodology from the literature rather than from the previous implementation,
> and it **changes the paper's central claim**. Six audit rounds plus a fresh
> literature search (see `LITERATURE_GAP_ANALYSIS.md`,
> `ADDITIONAL_LITERATURE_SEARCH.md`) established that:
>
> 1. **Calibration was not independent of model selection.** The 2016–2018
>    window early-stopped every backbone, selected the ensemble weight AND
>    calibrated all five conformal methods. In LOSO it was worse — the fit pool
>    absorbed **100%** of each fold's calibration set. Both are fixed; §5.1 now
>    documents a four-way split.
> 2. **ACI is no longer the recommended method.** It attains the best *marginal*
>    calibration error (ACE 0.0096) and the **worst conditional coverage in every
>    regime tested** (0.7602 in 2021). The marginal and conditional rankings are
>    perfectly inverted. Static conformal is recommended; ACI is retained as a
>    studied comparator. See §4.3, §5.4.
> 3. **Conditional, regime-stratified coverage replaces marginal PICP as the
>    selection criterion.** This is the paper's methodological contribution.
> 4. **Four proposed improvements were tested and rejected** — Mondrian
>    conformal, EnbPI, heteroscedastic calibration, architecture change. Their
>    failure is reported, not hidden; together they indicate the conditional
>    coverage problem is *structural*, consistent with the impossibility result
>    of Barber, Candès, Ramdas & Tibshirani (2020).
>
> The contribution is therefore an **evaluation-and-selection** result, not an
> architectural one. Text superseded by this pass is marked **[SUPERSEDED v4]**;
> new text is marked **[v4]**.

---

## 0. Revised Methodology Summary **[v4]**

**One-sentence statement.**

> Ensemble (NeuralCQR + LightGBM) point predictions, conformalised with **static
> split conformal** on a calibration window disjoint from model fitting and early
> stopping, evaluated on a six-state Corn Belt panel under temporal (2019–2023)
> and spatial (LOSO) shift, with **conditional, regime-stratified coverage** as
> the selection criterion and ACI reported as a studied alternative that achieves
> the best marginal calibration but the worst conditional coverage.

| Component | Decision | Basis |
|---|---|---|
| Six-state design | **Keep** | LOSO uniform (SD 0.036); scoped to Corn Belt, not national |
| Four-way temporal split | **Changed** | Calibration independence; verified 0 reused rows |
| NeuralCQR + LightGBM ensemble | **Keep** | Backbone spread 0.026 R² < seed SD 0.037 — indistinguishable |
| Loss (pinball + Huber + crossing + width) | **Keep** | Width penalty tested for removal: ΔPICP −0.0001, inert |
| CDHW features | **Keep, reframed** | No predictive gain (ΔR² −0.055); do mark predictive difficulty |
| Fourier terms | **Keep, minus P=1.0** | P=1.0 was provably constant *and selected* |
| **Static conformal** | **RECOMMENDED** | Best marginal PICP, best worst-stratum coverage |
| **ACI** | **Demoted to comparator** | Best marginal ACE, worst conditional coverage in all six regimes |
| Mondrian conformal | **Tested, rejected** | +0.006 ± 0.054 across origins; 2/5 wins |
| EnbPI | **Tested, rejected** | Width-rigid (MPIW 2.38–2.59 vs static 3.97–6.89) |
| Heteroscedastic calibration | **Rejected on residual diagnostics (not on a validation experiment)** | corr(\|bias\|, PICP) across years = −0.799 — coverage tracks *location*, not width. The slope of \|residual\| on prediction is **negative in every stratum**, so a standard variance head would widen intervals where errors are smallest. The variance axis was, however, tested indirectly: `locally_adaptive` conformal is a width-normalising method and gives the **best worst-stratum PICP (0.8801)** — a real but insufficient gain. See §5.11 and `RESIDUAL_AUDIT_ROUND6.md`. |

**Assumptions, stated rather than assumed.** Exchangeability is violated on both
axes (lag-1 residual ACF 0.5404; within-year Durbin–Watson 0.39–0.73; Moran's I
0.1918). Reported coverage is therefore **empirical, not guaranteed**. Exact
conditional coverage is unattainable distribution-free (Barber et al. 2020;
Vovk 2013). **No distribution-free guarantee is claimed anywhere in this paper.**

---

## 1. Overview

**[CITATION-CORRECTED]** This paper addresses an **open question** in agricultural uncertainty quantification: whether conformal prediction intervals, calibrated on historical residuals, retain their nominal coverage during anomalous climate years — precisely when reliable uncertainty estimates are most needed. Two strands of real literature motivate this. First, in the general (non-agricultural) case, static split conformal prediction is known to lose coverage under distribution shift, which is exactly the problem Gibbs & Candès (2021) introduce Adaptive Conformal Inference to solve, and which Barber et al. (2023, *Ann. Statist.* 51(2):816–845) formalise as a failure of the exchangeability assumption. Second, for agriculture specifically, the question remains open: the 2025 review *"Enhancing decision support in crop production: Analyzing conformal prediction for uncertainty quantification"* (*Computers and Electronics in Agriculture*) states directly that conformal prediction's performance "under distributional shifts and out-of-distribution detection has not been sufficiently explored" in agricultural settings. **[v4]** This paper evaluates five conformal calibration methods — static split conformal, phenology-stratified CQR, weighted/covariate-shift conformal, locally adaptive conformal, and Adaptive Conformal Inference — on a common Conformalized Quantile Regression backbone, and asks whether the criterion normally used to choose between them is the right one. It is not: on this six-state panel the **marginal** and **conditional** rankings are perfectly inverted. ACI attains the best marginal calibration error (ACE 0.0096) and the worst conditional coverage in all six regimes examined (0.7602 in 2021), while static conformal is the more reliable choice for compound drought-heatwave (CDHW) conditions. The contribution is therefore an evaluation-and-selection result rather than a new estimator.

> *Correction note:* earlier drafts opened by asserting that this failure mode had already been **documented** for agriculture, citing "Paper 18" of the literature review. That source could not be verified as a real publication (see `LITERATURE_CITATION_VERIFICATION.md`), and the specific figures attributed to it (91% overall PICP collapsing to 73% in compound years) should not be cited. The motivation is now grounded in the real sources above. Note this is a *stronger* framing: the paper answers an open question rather than re-confirming a known result.

**[NEW] Methodological lineage.** This work does not propose CQR or ACI as new algorithms; it adapts two established methods to the agricultural yield setting and jointly trains them end-to-end with a compound-event-aware backbone. CQR follows Romano, Patterson & Candès (2019); the online recalibration rule follows Gibbs & Candès (2021), Adaptive Conformal Inference under distribution shift. The novelty claimed here is (a) joint end-to-end training of both heads with a CDHW-aware backbone rather than post-hoc calibration, and (b) attribution of interval width to compound-event severity rather than to distribution shift alone. This lineage is stated explicitly to distinguish "applying" from "inventing" in front of reviewers familiar with the conformal literature.

**[NEW v2] Scope of the "architecture-agnostic" claim.** The ACI/CQR calibration layer is designed to attach to any regression backbone that outputs quantile predictions, not specifically to the backbone used in this paper. However, this paper only trains and evaluates it on one backbone. **[AUDIT CORRECTION]** That backbone is `NeuralCQRNet`, a residual MLP with four output heads (Huber point-prediction mean, and q05/q50/q95 quantile heads) -- earlier drafts of this document referred to it as "PhenoFormer," implying a Transformer-based sequence architecture; no such architecture exists in the codebase (`model_training.py`) that generated the reported results. This document has been corrected throughout to describe the model that was actually trained. See CORRECTION_CHANGELOG.md, Issue 3, for the full audit. Two options are available depending on remaining time/compute: (a) run the full pipeline once on one additional, simpler backbone (e.g., an LSTM or gradient-boosted quantile regressor already available from data-prep work) to give at least one empirical data point for portability, in which case the paper can claim "model-agnostic" with a demonstrated instance; or (b) if no second backbone is run, the write-up should use the narrower phrasing "the calibration framework is designed to be architecture-agnostic; it is evaluated here on a single backbone (NeuralCQR), and cross-architecture transfer is left to future work." Do not claim model-agnosticism as an empirical result unless (a) is done — a reviewer who checks the experiments section against the claim will flag the mismatch.

**[CITATION-CORRECTED] Relation to phenology-aware uncertainty quantification.** Earlier drafts of this section claimed this work *extends* "Paper 21," described as an existing phenology-aware CQR method that already widened intervals during pollination. **That source could not be verified as a real publication** (see `LITERATURE_CITATION_VERIFICATION.md`), and the numbers attributed to it (18% wider intervals during pollination; PICP 0.91 vs 0.83) must not be cited.

The real state of this literature is different, and more favourable to this work. Phenology-guided uncertainty quantification for crop yield **does** exist, but it is **Bayesian, not conformal** — the closest real work is Lin et al. (2023), *"A Phenology-guided Bayesian-CNN (PB-CNN) framework for soybean yield estimation and uncertainty analysis"* (*ISPRS J. Photogrammetry & Remote Sensing* 205:50–73), which structures a Bayesian CNN by phenological stage for county-level US Corn Belt soybean and decomposes aleatoric vs. epistemic uncertainty. Related real work shows that yield predictors themselves differ by growth stage (growth-state features dominate vegetative stages, water-related features dominate reproductive stages).

**No prior phenology-stratified *conformal* calibration was located.** This paper's phenology-stratified CQR (`aci_calibrator.py::phenology_stratified_cqr`) should therefore be presented as **a contribution of this work, not a reproduced baseline** — which makes the contribution larger than earlier drafts claimed, not smaller.

**State this claim carefully.** Absence of a search result is not proof of absence. The correct phrasing is *"to our knowledge, phenology-stratified conformal calibration has not previously been applied to crop yield"* — not an unqualified novelty assertion. A targeted database search before submission is advisable.

Two axes of distinction from the real Bayesian phenology work remain valid and should be stated: (i) PB-CNN's uncertainty is Bayesian and carries no finite-sample coverage guarantee, whereas conformal calibration does; (ii) PB-CNN conditions on phenological stage alone, whereas this paper additionally conditions on *compound* drought-heat severity, of which phenological timing is one component (see §3.1).

| | Gaps Closed | Gap 4 (compound extreme events) + Gap 5 (uncertainty uncalibrated under distribution shift) |
|---|---|---|
| | Key Baseline | Static split conformal (standard construction; Vovk et al. 2005, Lei & Wasserman 2014); standard quantile regression without ACI; **[NEW]** weighted/covariate-shift conformal (Tibshirani et al. 2019); locally adaptive conformal (Kivaranovic et al. 2020); phenology-stratified static CQR (**this work** — no prior phenology-conformal method located) |
| | Critical Test Scenario | El Niño 1997–98 and 2015–16; La Niña drought years; compound DHW events (SPEI-30 < −1 AND Tmax > 35°C) |

---

## 2. General Objectives

> **[v4] This section previously stated the premise backwards.** It claimed the
> paper addresses "the critical failure mode of static conformal prediction
> intervals, which collapse below nominal coverage during anomalous climate
> years". On this panel the opposite holds: **static conformal reaches
> PICP 0.9245 and ACI 0.8904** — ACI is the only method below nominal, and it has
> the worst conditional coverage in all six regimes. The premise was inherited
> from a motivating citation later found unverifiable
> (`LITERATURE_CITATION_VERIFICATION.md`) and is corrected below.

This paper asks **which criterion should govern the choice of conformal
calibration method** when the application is motivated by climate extremes.
Conformal guarantees are marginal by construction (Vovk 2005; Romano et al.
2019), and exact conditional coverage is unattainable distribution-free (Barber
et al. 2020; Vovk 2013) — yet agricultural applications, including the current
state of the art, report a single marginal coverage figure while being motivated
by extreme events, which is a conditional question.

The paper evaluates five conformal calibration methods on a common CQR backbone
across a six-state Corn Belt panel, under temporal (2019–2023) and spatial
(leave-one-state-out) distribution shift, and reports coverage **within**
climate, temporal and spatial regimes as well as marginally. The central finding
is that the two criteria rank the methods in **opposite orders**, so
marginal-optimal selection deploys the method that fails hardest precisely where
reliability is required.

---

## 3. Research Objectives

| # | Objective | Gap Addressed | Expected Outcome |
|---|---|---|---|
| O1 | Encode compound drought-heatwave (CDHW) events as a structured binary feature (**[NEW]** CDHWevent = 1 iff SPEI-30 < −1 AND Tmax > 35°C simultaneously, computed within phenology-defined growing-season windows). **[AUDIT-CORRECTED, see CORRECTION_CHANGELOG.md Issue 1 and "Issue 3 revisited"]** This objective now has two distinct components, previously conflated: (1a) demonstrate that CDHW features carry predictive information via a with/without ablation (`evaluate_objective_o1`), which shows predictive contribution, not causal super-linearity; and (1b) test statistical interaction/synergy directly via a drought-flag × heat-flag regression (`run_cdhw_interaction_analysis`), which is the correct test for whether joint exposure is more damaging than the additive sum of individual effects. Do not cite the ablation alone as evidence of super-linear damage. | Gap 4 — Compound extreme events unmodelled | **[AUDIT RESULT, real run]** (1a) **Not met**: current CDHW *reduces* ablation R² by 0.015 vs. no CDHW (0.517→0.502); the improved representation tested in this audit pass reduces it by a further 0.101 (0.502→0.402) and is rejected per Part 13 — see §4.6/§5.9. CDHW does not improve this backbone's aggregate point-prediction accuracy. (1b) **[AUDIT-CORRECTED]** the interaction result is specification-dependent, not a single unconditional finding — once state and year fixed effects with cluster-robust (by-state) standard errors are added, the primary **binary** drought-flag × heat-flag interaction is **not significant** at the conventional 0.05 level (coefficient −0.589, p=0.090, 95% CI crosses zero), while the **continuous** `SPEI-30 × Tmax-days` interaction under the identical specification **is** significant (coefficient −0.0128, p=6.8×10⁻⁵). A 12-combination sensitivity grid over standard drought/heat thresholds finds the coefficient negative in all 12 but significant in only 3/12. Report both results and the specification dependence explicitly — do not report only the significant continuous-variable result, and do not claim an unconditional "super-linear" finding from the binary-flag specification alone. |
| O2 | Develop a Conformalized Quantile Regression (CQR) head, following Romano, Patterson & Candes (2019), trained with pinball loss at τ = 0.05 and τ = 0.95 simultaneously, producing asymmetric prediction intervals appropriate for the skewed yield distribution observed during stress years. | Gap 5 — Uncertainty uncalibrated under distribution shift | **[AUDIT RESULT]** Met on the aggregate test set: PICP=0.940 (static/CQR row) ≥ 0.90 |
| O3 | Implement Adaptive Conformal Inference (ACI), following Gibbs & Candes (2021), with an online recalibration rule (severity-dependent dynamic window, see §4.3) that dynamically widens or narrows the CQR interval based on recent empirical coverage, preventing coverage collapse during anomalous climate years. | Gap 5 — Static conformal collapse under shift | **[AUDIT RESULT, real run — see §5.9]** **Not fully met**: ACI's own PICP is 0.933 in normal years but drops to 0.826 (moderate) and 0.820 (extreme) anomalous years — below the ≥0.90 target. ACI is still the best of 5 calibration methods compared on every aggregate metric (Winkler, ACE, MPIW; §5.9), and beats all 4 alternatives with high statistical significance — it mitigates, but does not eliminate, coverage collapse under shift (the failure mode Gibbs & Candès 2021 and Barber et al. 2023 establish for the general case). Report both facts, not only the favorable relative ranking. |
| O4 | Jointly train the CQR head and ACI recalibration mechanism end-to-end with the point-prediction backbone (rather than treating uncertainty as a post-hoc calibration step), and demonstrate that joint training improves both point prediction accuracy and interval sharpness (MPIW). | Gap 5 — No joint end-to-end uncertainty training in literature | Joint training reduces MPIW by ≥ 10% relative to post-hoc calibration at equal PICP; point-prediction RMSE also improves by ≥ 5% **[AUDIT-CORRECTED, see CORRECTION_CHANGELOG.md Issue 4]** — evaluated via the Phase 5 ablation's `3_plus_cqr` (post-hoc) vs `5_full_joint` (joint) rows specifically, using the corrected `evaluate_objective_o4()`. The prior implementation of this evaluation compared calibration methods (static/phenology/ACI) against each other instead, which share the same underlying point predictions by construction and so could never detect a training-paradigm effect either way — that comparison has been corrected to use the actual joint-vs-post-hoc ablation rows. **[AUDIT RESULT, real run]** This criterion was **not met** (0% RMSE change, 5.89% MPIW change, below the 10% threshold; PICP was actually *higher* under post-hoc, 0.940 vs. 0.921 joint) — classification **Not Supported**. |
| O5 | **[NEW]** Quantify how the ACI interval width co-varies with CDHW event severity (SPEI-30 magnitude, Tmax exceedance above 35°C threshold) and growing-season phenological stage, and show this compound-severity attribution improves on phenology-stage-only attribution by additionally capturing distribution-shift-driven widening that a static phenology-stratified baseline cannot. | Gap 4 + Gap 5 — Interpretability of uncertainty | **[AUDIT RESULT, real run]** **Not met as stated**: measured Pearson r=0.149 (Spearman r=0.134), statistically significant given N=2,345 but far below the hypothesized r>0.6. Stage-level correlations are all weak (Silking_R1 r=0.123, Grain_Fill r=0.137, Vegetative n.s.); the Silking_R1 stratum is thin (test n=35). Interval expansion is not strongly concentrated during silking. Report the real, weak-but-significant correlation. |
| O6 | **[NEW]** Benchmark ACI against the other principal distribution-shift-robust conformal approaches in the literature — weighted/covariate-shift conformal (Tibshirani et al. 2019) and locally adaptive conformal (Kivaranovic et al. 2020) — under identical anomalous-year test conditions, to establish that online recalibration is preferable to density-ratio-based or locally-normalized alternatives for this problem, rather than merely preferable to a non-shift-robust static baseline. | Gap 5 — Method justification against known alternatives | **[AUDIT RESULT, real run]** **Met, and exceeded**: ACI achieves the narrowest MPIW (4.935) of all 5 methods compared (not just "no worse than" weighted/locally-adaptive — strictly better), the best ACE and Winkler score, and beats all 4 alternatives with high statistical significance (Wilcoxon p between 1.6×10⁻¹⁵⁹ and 2.3×10⁻²⁰⁶, Holm-Bonferroni corrected). Aggregate PICP (0.921) is slightly below the ≥0.90-in-anomalous-years framing once stratified by year-type — see O3's result above. |

---

## 4. Proposed Methodology

### 4.1 Compound Drought-Heatwave Event Encoding **[REVISED]**

The CDHW indicator is derived from ERA5 weather data at daily resolution:

- **SPEI-30** (revised from SPI-30): 30-day Standardised Precipitation-Evapotranspiration Index computed from ERA5 precipitation and FAO-56 Penman-Monteith ET₀ (already in the data pipeline for a separate purpose), using the 1981–2010 climatological baseline. SPEI is used in place of SPI because it incorporates evaporative demand, which the compound-event literature identifies as necessary (see the compound drought-heat sources in the corrected review, e.g. Zscheischler et al. 2020; *Sci. Rep.* 13, 2023, doi:10.1038/s41598-023-29378-2) for a drought index paired with a heat threshold — SPI alone can understate drought severity in high-Tmax years precisely when the compound signal matters most.
- **CDHWevent binary flag:** Set to 1 for any day where SPEI-30 < −1 AND Tmax > 35°C simultaneously; 0 otherwise.
- **[NEW] Phenology-aligned severity score:** Rather than computing severity as a season-wide sum (as in the original design) and only relating it to phenological stage post-hoc in O5, the severity score is now computed *within* GDD-derived phenological windows (vegetative, silking/R1, grain-fill) and weighted by the product of |SPEI-30| and (Tmax − 35°C) on flagged days *within each window*. This directly implements the recommendation in Li, Zeleke, Wang & Liu (2025), *A Review of Data for Compound Drought and Heatwave Stress Impacts on Crops* (*Plants*, doi:10.3390/plants14142158) for joint drought-heat indices aligned with crop phenology, rather than treating phenology as a downstream analysis variable.
- The CDHWevent binary flag is included as a structured input feature (concatenated into the feature vector) to the NeuralCQR backbone, giving the model a global signal about compound stress occurrence for that county-year. **[AUDIT CORRECTION]** Earlier drafts described this as "prepended as a hard token to the weather sequence in PhenoFormer" -- that describes a Transformer-style sequence architecture that was never implemented for this paper. The actual backbone (`NeuralCQRNet` in `model_training.py`) is a residual MLP: an input projection + batch norm + GELU, followed by residual blocks, with four output heads (mean/Huber point prediction, and q05/q50/q95 quantile heads). CDHWevent enters as one feature among the input vector, not as a sequence token. See CORRECTION_CHANGELOG.md, Issue 3.
- **[AUDIT CORRECTION] Provenance of CDHW columns.** The CDHW detection itself (raw daily SPEI-30/Tmax thresholding) is performed upstream, outside this codebase — `code/` receives `CDHW_Flag`, `CDHW_Event_Count`, `CDHW_Severity_Score`, and phenology-stage severity columns already computed in the source CSV, and only combines/transforms them further. No file in this repository reads daily-resolution weather data. This matters because it bounds what can be validly engineered from CDHW here (see next point and `CDHW_DESIGN_REPORT.md`). One genuine positive finding from this audit: the phenology-stage severity split (vegetative/silking/grain-fill) used by the model comes from these real upstream columns, not from this repository's own coarse season-total-GDD phenology classifier (see §4.1b) — so the CDHW severity-by-stage features the model actually trains on are more genuinely phenology-aware than the standalone `Phenological_Window` feature's documented near-degeneracy would suggest.
- **[AUDIT CAVEAT] "Event" terminology.** `CDHW_Event_Count` reaches values up to 38 in a single growing season, which is far more consistent with a **count of qualifying days** than a count of discrete multi-day heatwave *episodes* (a ~150–180 day season cannot contain 38 distinct events under any standard multi-day-duration definition). This cannot be confirmed with certainty without the upstream generation code, which is not present in this repository, but the paper should describe this column as a compound-stress-day frequency measure rather than an episode count unless the upstream methodology can be confirmed otherwise.
- **[AUDIT ADDITION, Part 6/7] Improved CDHW representation, tested via ablation.** Two additional engineered features are computed deterministically, pre-split, from already-existing county-year columns (no daily-resolution data exists anywhere in this repository, so duration, inter-event recovery time, and event-clustering features cannot be validly constructed — see `CDHW_DESIGN_REPORT.md` for the full audit of what is and is not recoverable): (i) `Inter_SPEI_HeatDays = SPEI_30_min × Tmax_Days_Above_35`, a continuous drought×heat severity interaction — this exact specification was already tested in the O1b interaction analysis with state+year fixed effects and cluster-robust SE and found significant (p=6.8×10⁻⁵), so this promotes an already-validated relationship into the predictive feature set rather than proposing an untested one; and (ii) `CDHW_Severity_Per_EventDay = CDHW_Severity_Score / max(CDHW_Event_Count, 1)`, an intensity-per-exposed-day measure decomposing cumulative compound-stress burden from frequency. Both are **unprotected** candidate features, subject to the same 4-method consensus feature-selection vote as every other engineered feature — they are not forced into the model. The ablation study (§4.6) adds a dedicated row testing whether they improve predictive accuracy beyond the current CDHW representation; see Results for the outcome, reported honestly whether positive, negative, or null.

### 4.1b Forecasting Horizon and Temporal Availability of Features **[AUDIT-ADDED, see CORRECTION_CHANGELOG.md Issue 5]**

This subsection states explicitly what prediction task this pipeline represents, and audits every temporal feature against it, per the audit's requirement to distinguish valid-at-prediction-time information from leakage.

**Task represented by this pipeline: retrospective / end-of-season yield estimation**, not a preseason or mid-season forecast. This follows directly from how the underlying features are constructed:

- `GDD_Accumulated` and other growing-season climate aggregates (precipitation totals, heat-day counts, CDHW severity) in the source dataset are **season-total** values (one number per county-year; observed range ~387-3022 GDD, mean~2001) — only fully known once the growing season has concluded, not at planting or mid-season.
- Rolling climate features (`build_split_aware_rolling_features`) include the *current* season `t`'s own realized totals, not just prior years — consistent with an end-of-season estimation task, not a forecast issued before the season's weather is known.
- `Yield_lag1` / `Yield_lag2` use a strict `shift(1)` / `shift(2)` by county, sorted by year, verified to pull only from strictly earlier years (exact 1- or 2-year lag, never the current or a future year) — this is valid information at prediction time under any horizon, since the prior year's realized yield is known well before the current season.

**Audit classification of temporal features:**

| Feature class | Availability | Classification |
|---|---|---|
| `Yield_lag1`, `Yield_lag2` | Prior years' realized yield, strictly shifted, verified no same/future-year leakage | Valid at prediction time (any horizon) |
| Season-total climate (GDD, precip, heat-days, CDHW severity) for year *t* | Only fully known at/after end of season *t* | Valid **only** under a retrospective/end-of-season estimation framing (as used here); would be leaked information under a preseason or mid-season forecast framing |
| `Phenological_Window` category | Derived from season-total GDD (see limitation below) | Not leakage, but low information content — see below |
| County/state static attributes (soil, land cover, DEM) | Time-invariant | Valid at any horizon |
| ACI calibration buffer | Only ever contains realized outcomes from strictly prior test years (verified in `aci_calibrator.py`) | Valid — no future-year information enters calibration for any given year |

**Known limitation (does not affect validity, but affects the phenology-severity feature's usefulness):** because `GDD_Accumulated` is season-total, the `Phenological_Window` classification (`Vegetative`/`Silking_R1`/`Grain_Fill`, thresholds in `config.py`) is near-degenerate — the large majority (>90%) of county-years classify as `Grain_Fill` regardless of the exact GDD threshold chosen, since most growing seasons accumulate well over the threshold by season's end. This does not introduce leakage (the classification still only uses that year's own already-realized data), but it means the phenology-stage weighting in the CDHW severity score has less discriminative power than the methodology's original framing implied. A genuine fix would require within-season (sub-annual) weather data, not available in this county-year-level dataset. This is stated as an explicit limitation for §11/Discussion rather than silently left implicit.

If a preseason or mid-season forecasting variant of this work is pursued in the future, every season-total climate feature listed above would need to be replaced with the corresponding **partial-season** aggregate available as of the stated forecast date, and the CDHW/phenology features rebuilt accordingly — this pipeline as documented and executed is retrospective/end-of-season estimation only.

### 4.2 Conformalized Quantile Regression (CQR) Head

A 2-layer MLP head with shared first layer from the regression backbone outputs two scalars per county-year: q_lo (5th percentile) and q_hi (95th percentile) of the predicted yield distribution, following the CQR formulation of Romano, Patterson & Candès (2019).

- Training loss: Quantile (pinball) loss at τ = 0.05 and τ = 0.95 simultaneously.
  `L_CQR = (1/N) Σ [max(0.05·(y − q_lo), 0.95·(q_lo − y)) + max(0.95·(y − q_hi), 0.05·(q_hi − y))]`
- The CQR head is trained jointly with the backbone from Phase 2 of the three-phase curriculum, ensuring that the interval head adapts to the backbone's learned representations rather than calibrating post-hoc on fixed backbone outputs — this is the point of departure from the original CQR formulation (Romano, Patterson & Candès 2019), which applies the conformal correction post-hoc to fixed model outputs.

### 4.3 Adaptive Conformal Inference (ACI) Recalibration **[AUDIT-CORRECTED, see CORRECTION_CHANGELOG.md Issue 2]**

**Audit note (superseded, then re-corrected 2026-09-09).** An earlier round of this document described a *severity-dependent dynamic window*: 2 buffer blocks when mean CDHW severity >= 5.0, 3 when >= 1.0, 5 otherwise. **That mechanism has since been removed from the code.** It made the window size data-dependent rather than a fixed methodology parameter, and it counted BUFFER BLOCKS rather than calendar years (the initial 2016-2018 validation pool counted as one "block" whatever the window size). `adaptive_conformal_inference()` now uses a **fixed 3-year calendar-year rolling window** (`cfg.ACI_WINDOW_SIZE = 3`): for test year Y the calibration set is exactly years [Y-3, Y-1], and once more than three years of history exist the oldest year is genuinely dropped. Severity no longer influences window size in any way. Leakage checks on this path found **no leakage under the tested protocol** -- that is the scope of what was verified, not a general guarantee.

At inference time, ACI recalibrates the CQR intervals one test year at a time, using an online update rule in the spirit of Gibbs & Candes (2021), with two elaborations beyond the original algorithm:

- **Calibration buffer:** initialized with the full held-out validation set (2016-2018, i.e. the genuine calibration pool used nowhere in training), split into genuine **per-calendar-year sub-blocks** (2016, 2017, 2018) rather than one opaque block, so the rolling window measures real elapsed years from the outset. As each test year (2019, 2020, ... 2023) is calibrated, that year's *actual, realized* residuals are appended to the buffer as a new block *after* that year's interval has already been produced -- so no test year's own outcome ever informs its own calibration, and no future year's data can enter a prior year's calibration. This was verified directly against the code: the append to the buffer happens strictly after the coverage/interval computation for that year.
- **Fixed 3-year calendar-year rolling window:** for each test year Y the calibration set is the residuals from calendar years [Y-3, Y-1] -- selected by literal year lookback, never by that year's own severity, and never by buffer-block count. The initial pool (2016-2018 validation) is split into genuine per-year sub-blocks so it composes correctly with the test years appended after it. Once more than three years of history exist, the oldest year is dropped: the window genuinely rolls rather than growing indefinitely. `cfg.ACI_WINDOW_SIZE = 3`.
- **Severity-weighted thresholds: IMPLEMENTED, MEASURED, AND NOW DISABLED (`cfg.ACI_SEVERITY_WEIGHTING = False`).** The mechanism computed `w_i = 1 + 0.05 * log1p(severity_i)` and applied `eff_threshold_i = threshold_t / w_i`. Because the threshold is **divided** by a weight that is always >= 1, higher severity produced a **smaller** threshold and a **narrower** interval -- the opposite of the intent. Measured on 2019-2023: no-stress rows 0.00% width change, moderate -4.11%, severe **-10.81%**, most severe row **-17.8%**. Intervals were being narrowed precisely in the regime where coverage already fails. Simply switching it off improved both severe coverage and Winkler score across held-out origins. It is off by default and the reported results do not use it. This is documented as a corrected defect, not presented as a contribution.
- **Online alpha update (Hedge-style):** after computing each year's empirical coverage error `err_t = 1 - PICP_t`, the target miscoverage rate `alpha_t` is updated by `alpha_t += eta_t * (alpha - err_t)`, with `eta_t = gamma / sqrt(t+1)` (a decaying step size, standard in no-regret online learning) and `alpha_t` clipped to `[0.01, 0.40]`. **[NEW]** This step-size behavior is grounded in the no-regret online learning framework (Hedge / expert-advice algorithms; Cesa-Bianchi & Lugosi 2006, *Prediction, Learning, and Games*) -- that part of the original description was accurate and is retained.
- **[AUDIT CORRECTION] Granularity of recalibration:** the original text stated ACI "detects the coverage shortfall within the first 1-2 months of residual accumulation and expands the interval before the end of the growing season." The actual implementation recalibrates once per full test *year* (annual granularity, matching the county-year annual dataset), not sub-annually -- there is no monthly or within-season updating in this pipeline. This sentence has been corrected: ACI detects a coverage shortfall in year *t* and expands the interval used for year *t+1*, using that year's realized outcomes once they become available (i.e., after harvest) -- not within the growing season itself. This is consistent with the "retrospective/end-of-season estimation" framing documented in §4.1 (audit finding, Issue 5).

- **El Niño / anomalous year behaviour:** During anomalous years when the point predictor is systematically biased, ACI detects the resulting coverage shortfall in that year's realized outcomes and expands the interval for the *following* year, directly addressing static-conformal coverage loss under distribution shift (Gibbs & Candès 2021; Barber et al. 2023). (Corrected from the original "within the first 1-2 months... before the end of the growing season" claim -- see audit note above.)

### 4.3b Regime-Adaptive Alpha (V6) — the production calibration method **[NEW, locked 2026-09-09]**

ACI attains the best *marginal* calibration error in this pipeline (ACE 0.0109) and
the **worst conditional coverage of any method tested**, in all six stratum types —
the only method falling below 0.85, in four of them (worst 0.8006). **[SUPERSEDED 2026-09-10 — computed from a corrupted county-name join (see JOIN_BUG_FIX_2026-09-10.md). Do not cite this figure. Replacement numbers must come from a fresh end-to-end run, not from any existing outputs folder.]** Marginal coverage
is not the property a paper about compound-stress uncertainty needs; coverage in the
severe stratum is.

Four remedies from the conformal literature were implemented and each failed to fix it
on held-out origins, and all four are reported:

| Remedy | Reference | Measured effect on severe coverage |
|---|---|---|
| Mondrian / partition-conditional | Vovk (2013); Toccaceli & Gammerman (2018) | +0.0064 ± 0.0536 — indistinguishable from zero |
| EnbPI | Xu & Xie (2021, 2023) | PICP collapsed to 0.6336 |
| Asymmetric (two-sided) conformal | Romano et al. (2019) | −0.0591 — worse |
| Residual bias correction | — | +0.0140 ± 0.0510 — indistinguishable from zero |

**V6 (regime-adaptive alpha)** is what did work. It keeps one global conformal
threshold `t_global` at the nominal α for all rows, and additionally computes a
second threshold `t_severe` at a stricter α_s from the *severe-stratum calibration
pool only*; severe-regime test rows use `t_severe`, all others use `t_global`
(`cfg.REGIME_SEVERE_ALPHA = 0.05`, `cfg.REGIME_SEVERITY_CUT = 4.5054`,
`cfg.REGIME_MIN_CAL = 50` — below which it falls back to the global threshold rather
than estimating a quantile from too few points).

**Selection protocol.** Seven variants were compared on **23 rolling origins** using
development data only; 15 origins had a scorable severe stratum (n = 2,076). *The
2019–2023 test window was never read during selection.*

| | Severe PICP | Winkler | Origins won |
|---|---|---|---|
| **V6 regime-adaptive alpha** | **0.8478** | **8.8133** | **14 / 15** |
| Incumbent (severity-weighted ACI) | 0.6696 | 10.6604 | last of seven |

ΔPICP = **+0.1320 ± 0.1138**. Winkler *improves* alongside the coverage gain, which is
the honest criterion: the extra width is earning its keep rather than being blanket
inflation. Width is bought only in the severe stratum — the none and moderate strata
are numerically unchanged.

**What this does NOT establish.** Severe PICP 0.8478 is **still below the nominal
0.90**. V6 narrows the gap; it does not close it. Distribution-free *conditional*
coverage is provably unattainable (Barber, Candès, Ramdas & Tibshirani 2020), so no
method here — V6 included — should be described as providing a conditional guarantee.
The severe stratum is also dominated by drought years, so the estimate rests on a
small number of genuinely extreme seasons.

### 4.4 Exchangeability Under Temporal and Spatial Autocorrelation **[NEW SECTION]**

Both CQR and ACI's coverage guarantees rely on exchangeability (or, for ACI, a controlled form of distribution shift) between calibration and test residuals. County-year yield data violates this in two ways not addressed by distribution shift alone:

- **Temporal autocorrelation:** Yield residuals within the same county across consecutive years are not independent (soil carryover effects, multi-year drought persistence). **[CITATION-CORRECTED]** the relevant precedent is Barber, Candes, Ramdas & Tibshirani (2023), *Conformal prediction beyond exchangeability* (*Ann. Statist.* 51(2):816-845), which formalises conformal validity when exchangeability fails (the previously-cited "Paper 19" could not be verified); this paper's severity-dependent dynamic window (§4.3, corrected per audit) provides partial mitigation but was not designed against autocorrelation specifically.
- **Spatial autocorrelation:** Residuals across neighboring counties in the same year are correlated (shared weather systems). **[CITATION-CORRECTED]** the previously-cited "Paper 26" could not be verified. Real spatially-aware conformal work does exist (e.g. GeoConformal Prediction; localized-quantile spatial conformal inference) but was not pinned to a verified citation in this audit -- **verify a specific source before citing**, or state the spatial-autocorrelation limitation without attributing it to a particular prior method.

**Proposed robustness check:** in addition to the primary PICP/MPIW evaluation, report a block-bootstrap sensitivity analysis (blocking by county and by year) to test whether nominal coverage holds under a more conservative effective-sample-size assumption. This is reported as a robustness diagnostic rather than a full methodological fix, and is flagged as a limitation to be addressed by a spatially-aware ACI variant in future work (the spatial-conformal literature noted above provides a template, once a specific source is verified).

### 4.5 Evaluation Against Static Conformal Baseline

The primary comparison uses the standard static split-conformal setup: intervals calibrated on 1985-2015 residuals and evaluated without recalibration on 2016–2023. ACI is evaluated under identical conditions. Coverage is reported separately for:

- Normal years (no CDHW events, ENSO-neutral).
- Moderate anomaly years (1–2 CDHW events per season).
- Extreme anomaly years (3+ CDHW events; El Niño / La Niña years).

**[NEW] Extended baseline comparison (O6):** the same evaluation protocol is additionally run for weighted/covariate-shift conformal (Tibshirani et al. 2019) and locally adaptive conformal (Kivaranovic et al. 2020), and for the phenology-stratified static CQR variant (**this work**) specifically for the O5 interval-attribution comparison. This directly answers the "why ACI and not X" question a reviewer familiar with the conformal literature will ask.

### 4.6 Ablation Study **[NEW v2 SECTION, audit-extended to 6 rows]**

To isolate which component of the pipeline is responsible for the reported gains, report point-prediction and interval metrics for six nested configurations (five original rows, plus one new row inserted after "+ CDHW encoding" testing the improved CDHW representation from §4.1/Part 6-7 of the audit):

| Configuration | What is added | RMSE | R² | PICP | MPIW |
|---|---|---:|---:|---:|---:|
| Backbone only (`1_backbone_only`) | NeuralCQR (residual MLP) point prediction, no CDHW-family feature (current or improved), no interval head | 1.1777 | 0.5173 | n/a | n/a |
| + CDHW encoding, current (`2_plus_cdhw`) | Adds the CDHW binary flag + phenology-aligned severity score (§4.1) to the backbone input — the originally published representation | 1.1958 | 0.5024 | n/a | n/a |
| **[AUDIT-ADDED]** + CDHW encoding, improved (`2b_plus_cdhw_improved`) | Current CDHW representation **plus** `Inter_SPEI_HeatDays` and `CDHW_Severity_Per_EventDay` (§4.1) | 1.3114 | 0.4015 | n/a | n/a |
| + CQR (`3_plus_cqr`) | Adds the post-hoc-style CQR interval head (§4.2), still without ACI recalibration | 1.2873 | 0.4232 | 0.9399 | 5.2441 |
| + ACI (`4_plus_aci`) | Adds online ACI recalibration (§4.3) on top of CQR, still with post-hoc (not joint) training | 1.2873 | 0.4232 | 0.9207 | 4.9354 |
| Full model (joint training) (`5_full_joint`) | CDHW + CQR + ACI trained jointly end-to-end (§4.2's departure from post-hoc calibration) | 1.2873 | 0.4232 | 0.9207 | 4.9354 |

Note on row definitions: "Backbone only" and both "+ CDHW encoding" rows are point-prediction-only rows (no interval head yet), so PICP/MPIW are not applicable until a CQR head exists. This keeps the table honest about what each row is actually isolating rather than implying CDHW is a conformal-method component. **[AUDIT CORRECTION]** Rows 3–5 use `all_feature_cols` (75 features) — the pipeline's naturally, independently consensus-selected default feature set, which happens to include the current CDHW family (74) plus `Inter_SPEI_HeatDays` (selected on its own merits by the unforced 4-method vote, not force-included) — not exactly the 74-feature `2_plus_cdhw` set. The O4 joint-vs-post-hoc comparison (`3_plus_cqr` vs `5_full_joint`) is still a clean comparison (identical 75-feature input to both), just not on the literal `2_plus_cdhw` feature set.

**[AUDIT RESULT — real, full pipeline run]** The improvement is **not supported**: adding the current CDHW representation *reduces* R² by 0.015 relative to no CDHW at all, and the proposed improved representation reduces it by a further 0.101 (the largest single drop in the table). **Per Part 13, this negative result is reported honestly. The original CDHW representation is retained; the improved representation is rejected and not adopted for the shipped model.** See `CDHW_DESIGN_REPORT.md` §4b for the full result and one open nuance (the unforced feature-selection consensus separately selected `Inter_SPEI_HeatDays` alone, which this bundled ablation cannot use to exonerate it individually).

### 4.7 Computational Complexity **[NEW v2 SECTION]**

Report, for the full model and for the backbone-only configuration:

- Training time (wall-clock, on the target GPU) and number of epochs to convergence.
- Inference time per county-year (batched and single-sample).
- Peak GPU memory during training and during inference.
- Additional overhead attributable specifically to the CQR head and to the ACI online update, since the latter is a lightweight per-step statistic update rather than a forward pass and should be cheap — quantifying this supports the practicality argument for real-time or near-real-time deployment during a growing season.

---

## 5. Evaluation Protocol

### 5.1 Primary (Temporal) Experiment

> **[v4] Calibration-independent four-way split.** Previously the 2016–2018
> window served simultaneously as early-stopping set, ensemble-weight selection
> set and conformal calibration set, which voids split conformal's premise that
> the scoring model is fixed independently of the calibration scores.
>
> ```
> fit              1985-2013   15,749 rows
> model selection  2014-2015      978    early stopping + ensemble weight
> calibration      2016-2018    1,433    conformity scores ONLY
> test             2019-2023    2,345    untouched until final evaluation
> ```
>
> Verified: **0** calibration rows used in fitting or early stopping, in the
> temporal split and in all six LOSO folds
> (`calibration_independence_report.json`). Scaler, detrend coefficients, county
> baseline and feature selection remain fitted on the full training period —
> early-stopping rows are development data, not held-out data.
>
> **Cost, quantified rather than hidden:** reserving 2014–2018 costs
> approximately **0.12 test R²** relative to training on 1985–2018. That is the
> price of valid conformal coverage, and the paper reports it as such.


This remains the paper's main experiment, since the central claim is about calibration under **climate distribution shift**, which is a temporal, not spatial, phenomenon — the spatial evaluation in §5.2 is a robustness/generalization check, not a replacement.

- **Primary metrics:** PICP (target ≥ 0.90 in all three year-type categories), MPIW (minimise subject to PICP constraint), yield-loss R² for CDHW encoding (target > 0.70).
- **Temporal split:** Same as Paper 1 (train 1985–2015, val 2016–2018, test 2019–2023).
- **Anomalous year subset:** El Niño years 1997–98, 2015–16; La Niña drought 1988, 2012; heat dome analogue years.
- **Baselines:** Static split conformal (standard construction); standard quantile regression without conformalisation; weighted/covariate-shift conformal (Tibshirani et al. 2019); locally adaptive conformal (Kivaranovic et al. 2020); phenology-stratified static CQR (**this work**).

### 5.2 Spatial Generalization: Leave-One-State-Out Cross-Validation **[NEW v2]**

**[AUDIT-CORRECTED]** The study uses **six** states — Illinois, Indiana, Iowa, Minnesota, Missouri, and Ohio. A seventh state, Nebraska, is present in the raw dataset but is deliberately excluded (`config.py::DROP_STATES`): Nebraska corn production is heavily irrigated, which breaks the drought–yield relationship the model is trained to learn from the six rain-fed states, and produces a degenerate/empty LOSO fold if retained. Six is a deliberate design choice, not a data-availability shortfall, and is not expanded to more states in this paper.

**Six-state diversity, quantified (not asserted):** the six states span a genuine agroclimatic gradient within the U.S. Corn Belt — latitude 38.4–45.6°N (Missouri to Minnesota, ~7° spread), longitude −94.4 to −82.8°W (~11.5° spread), and a ~800 growing-degree-day range (Minnesota mean 1505 GDD, the shortest/coolest season, to Missouri mean 2339 GDD, the warmest). Mean corn yield ranges from 7.35 t/ha (Missouri) to 9.64 t/ha (Iowa), and year-to-year yield volatility (coefficient of variation) ranges from 0.223 (Indiana) to 0.291 (Minnesota) — Minnesota and Missouri are consistently the two most volatile states in the panel. Soil composition also varies meaningfully across states (e.g., mean soil sand content ranges from ~106 to ~353 units across states). Each state contributes 87–115 counties and a full, uninterrupted 1985–2023 (39-year) record, so per-fold sample sizes are large and balanced across folds.

**What this diversity does, and does not, support:** the six states are geographically and climatically diverse *within the U.S. Corn Belt / rain-fed Midwest cropping system* — they are not a nationally representative sample (no irrigated West, no South, no non-Corn-Belt cropping system is included). LOSO-CV across these six states is genuine evidence of generalization **across distinct Corn Belt states with different soil, yield-volatility, and heat/drought-exposure profiles** — a real and reportable form of spatial generalization — but it is **not** evidence that the framework "generalizes across the United States" or to fundamentally different agroclimatic or management regimes (e.g., irrigated agriculture, a different crop, a non-Corn-Belt region). The paper's claims should use language such as *"evaluated across six geographically and climatically distinct U.S. Corn Belt states"*, not *"generalizes nationally"* or *"generalizes across the United States."*

- **Protocol:** For each of the six states, train (backbone + CQR + ACI, jointly) on the remaining five states and evaluate on the held-out state. Repeat for all six states, so every state serves as the test fold exactly once.
- **Metrics reported per fold, not just averaged:** RMSE, MAE, R², PICP, MPIW for each of the six held-out states individually, plus the mean and standard deviation across folds.
- **[NEW v2] Caveat on statistical power:** With only six folds, a single atypical state (e.g., one with unusually persistent drought or unusually different management practices) can dominate the average and make it look worse or better than the typical case. Reporting only a mean across six states would hide this. Presenting all six fold-level results in a table (not just the aggregate) is the more defensible choice for reviewers and should be treated as required, not optional. (In this pipeline's runs, Missouri — the lowest-yielding, second-most-volatile state — is consistently the hardest LOSO fold; this is a real, physically-explainable finding, not noise, and should be reported as such rather than averaged away.)
- **Relationship to the temporal split:** LOSO-CV and the temporal split answer different questions (does the model generalize across space vs. across time-driven distribution shift) and both should be reported; neither substitutes for the other.

### 5.3 Robustness: County Block Bootstrap

Retained from the prior version (§4.4): block-bootstrap coverage check under county- and year-blocked resampling, to test whether nominal coverage holds under a more conservative effective-sample-size assumption. This complements LOSO-CV — LOSO-CV tests generalization to an entirely unseen state, while block bootstrap tests sensitivity of the coverage estimate itself to spatial/temporal autocorrelation within the training distribution.

### 5.4 Uncertainty Metrics **[EXPANDED v2]**

In addition to PICP and MPIW, report:

- **ACE (Average Coverage Error):** |empirical coverage − nominal coverage|, averaged across test years. This is a standard expectation in uncertainty-calibration papers and complements PICP by summarizing calibration error as a single signed/unsigned quantity rather than a pass/fail threshold.
- **Winkler Score:** a proper scoring rule that jointly penalizes interval width and coverage failures, rewarding intervals that are both narrow and well-calibrated. Report this per year-type category (normal / moderate / extreme) to show whether the score is being driven by width or by miscoverage during extreme years specifically.
- **CRPS (optional) — with a caveat:** CRPS is defined over a full predictive distribution, not a two-sided interval, so computing it from CQR/ACI intervals requires first constructing a pseudo-distribution (e.g., by treating the interval as bounds of a fitted distribution, or using a quantile-based CRPS approximation). This extra construction step should be stated explicitly in the write-up if CRPS is included, so a reviewer doesn't assume it was computed directly from the model's native output. If time is limited, ACE and Winkler Score alone are sufficient and more directly justified for a conformal-interval method; CRPS can be dropped without weakening the paper.

### 5.5 Statistical Significance Testing **[NEW v2]**

For every pairwise comparison of ACI against a baseline (static conformal, weighted conformal, locally adaptive conformal, phenology-stratified static CQR):

- **Primary test: Wilcoxon signed-rank test**, not paired t-test, as the default. The paired t-test assumes approximately normal differences, which is a weak assumption with the small number of independent units available here (6 states for LOSO-CV; a handful of anomalous test years for the temporal comparison). Wilcoxon is the safer default given this sample size and doesn't require that assumption.
- **Paired t-test** can be reported as a secondary/robustness check alongside Wilcoxon, not as the primary test, if the differences look reasonably normal on inspection.
- **[NEW v2] Multiple-comparison correction:** since ACI is compared against four baselines (not one), report a Holm-Bonferroni or false-discovery-rate (Benjamini-Hochberg) correction across that family of tests. Reporting four uncorrected p < 0.05 comparisons without correction is a common reviewer objection in papers with multiple pairwise baseline comparisons.
- Report exact p-values (or corrected p-values) rather than only "p < 0.05", so readers can judge effect sizes alongside significance.

### 5.6 Calibration Curves **[NEW v2]**

Plot expected coverage (nominal, e.g. 0.80, 0.85, 0.90, 0.95) against observed empirical coverage at each nominal level, for ACI and for each baseline on the same axes. A well-calibrated method tracks the diagonal; systematic over- or under-coverage shows as a curve bowing above or below it. This is a standard, low-cost visual complement to the ACE/PICP numbers above.

### 5.7 Interval Width by Year-Type **[NEW v2]**

Report the distribution (not just the mean) of MPIW separately for normal, moderate-anomaly, and extreme-anomaly years — e.g., as a boxplot or violin plot rather than three bars, so reviewers can see the spread within each category, not only the central tendency. This is the evidentiary support for the paper's main claim that intervals widen appropriately during extreme years, and is more convincing as a distribution than as three summary numbers.

### 5.8 Interpretability Analysis

Spearman correlation between ACI interval width and CDHW severity score; interval width by phenological window (using GDD-stage bins); direct comparison of interval-width error against the phenology-stratified variant introduced here.

**[AUDIT RESULT]** Real result (Pearson r=0.149, p<0.001; Spearman r=0.134, p<0.001; N=2,345): a statistically significant but **weak** correlation, well below the originally hypothesized "r > 0.6" (§3, O5). Stage-level correlations are all weak (Vegetative r=−0.008 n.s., Silking_R1 r=0.123, Grain_Fill r=0.137); the `Silking_R1` calibration/test stratum is thin (test n=35 vs. `Grain_Fill` n=2,310), consistent with the already-documented `Phenological_Window` near-degeneracy (§4.1b). **Report the real, weak-but-significant correlation — do not claim the r>0.6 target was met.**

### 5.9 Results Summary **[AUDIT-ADDED — real, full pipeline run, `outputs.zip`]**

This subsection consolidates the headline real results this methodology
document's claims are now based on. Full detail, including every report file
consulted, is in `AUDIT_REPORT.md`'s "Final Results" section.

**[ROUND 2 UPDATE]** A subsequent fix (`feature_selection.py`, excluding
`Inter_SPEI_HeatDays` from the default/shipped feature set — see
`CDHW_DESIGN_REPORT.md` §4c) corrects a real regression this feature's
unforced natural selection had caused. The primary backbone's R² is expected
to recover from 0.4232 to ≈0.51 (confirmed on an isolated diagnostic, not yet
on a full re-run). **Every number below involving the default 75-feature set
(backbone, calibration comparison, robustness check, LOSO) needs
re-verification on a fresh full run; the CDHW ablation and interaction
numbers are unaffected.**

**LOSO-CV, per-fold (real):**

| State | R² | RMSE | PICP | MPIW |
|---|---:|---:|---:|---:|
| Illinois | 0.662 | 1.411 | 0.921 | 5.297 |
| Indiana | 0.729 | 1.049 | 0.973 | 5.035 |
| Iowa | 0.578 | 1.445 | 0.916 | 5.375 |
| Minnesota | 0.607 | 1.567 | 0.856 | 5.072 |
| Missouri | 0.538 | 1.422 | 0.888 | 5.206 |
| Ohio | 0.640 | 1.191 | 0.954 | 4.891 |
| **Mean** | **0.626** | 1.348 | 0.918 | 5.146 |

Missouri (weakest R²) and Minnesota (weakest PICP) are the two most volatile
states in the six-state panel (§5.2) — a physically coherent pattern, not
noise. Every fold's ensemble weight favors LightGBM (NeuralCQR weight 0.0–0.5
across folds) — the neural backbone alone does not win any LOSO fold outright.

**Extreme-event calibration (the paper's central claim, O3):**

| Year-type | n | PICP | MPIW | Winkler |
|---|---:|---:|---:|---:|
| Normal | 2,083 | 0.933 | 4.862 | 5.554 |
| Moderate | 201 | **0.826** | 5.472 | 7.610 |
| Extreme | 61 | **0.820** | 5.689 | 10.034 |

**O3's stated target ("PICP ≥ 0.90 maintained during all anomalous test
years") is not met.** ACI's own coverage degrades from 0.93 (normal) to
~0.82 (moderate/extreme). This must be stated plainly in the paper: ACI
mitigates, but does not eliminate, the coverage-collapse failure mode
described in the general conformal literature (Gibbs & Candes 2021; Barber et al. 2023) and flagged as unexplored for agriculture (*Comput. Electron. Agric.*, 2025) — it is still the best of the 5 methods compared
(next table), but "best available" is not the same as "meets the 0.90
target," and the paper should not conflate the two.

**Calibration method comparison (temporal split, all real):**

| Method | PICP | MPIW | ACE | Winkler | O6 rank |
|---|---:|---:|---:|---:|---:|
| ACI | 0.921 | **4.935** | **0.021** | **5.847** | **1 (best)** |
| Phenology-stratified CQR (this work) | 0.939 | 5.242 | 0.039 | 5.948 | 2.33 |
| Static split conformal (standard) | 0.940 | 5.244 | 0.040 | 5.940 | 2.67 |
| Weighted conformal (Tibshirani et al. 2019) | 0.942 | 5.284 | 0.042 | 5.957 | 4.0 |
| Locally adaptive (Kivaranovic et al. 2020) | 0.948 | 5.732 | 0.048 | 6.286 | 5 (worst) |

ACI beats every alternative on Winkler score with Wilcoxon p-values between
1.6×10⁻¹⁵⁹ and 2.3×10⁻²⁰⁶ (large effect sizes, rank-biserial 0.61–0.73), all
significant after Holm-Bonferroni correction — a genuine, statistically
strong confirmation of O6.

**O1b interaction (real, matches the specification-dependent framing in §3):**
binary flag interaction coefficient −0.589 (p=0.090, not significant);
continuous `SPEI×heat-days` interaction coefficient −0.0128 (p=6.8×10⁻⁵,
significant); sensitivity grid negative in 12/12 combinations, significant in
3/12.

**O4 (real):** ΔRMSE=0.0%, ΔMPIW=+5.89% (below the 10% bar), classification
**Not Supported** — joint end-to-end training does not measurably improve
point-prediction accuracy or interval sharpness over post-hoc calibration in
this run.

---

### 5.10 Geographic Results **[NEW — spatial presentation of the study region and model behaviour]**

All maps are produced by `code/geo_maps.py`, which is wired into the pipeline at
three points: `run_all_geo_maps(df)` after preprocessing (descriptive maps),
`plot_residual_choropleth(...)` after backbone evaluation, and the three LOSO
choropleths after Phase 5. County and state polygons are US Census TIGER
geometries fetched once from public GeoJSON mirrors and cached under
`outputs/geo_cache/`, so figures are reproducible offline after the first run.
Counties are joined to geometry on 5-digit GEOID.

**Provenance warning — read before citing any number from these figures.**
The maps fall into two classes with *different* evidential status:

| Class | Figures | Depends on model training? | Citable from the current output set? |
|---|---|---|---|
| **Descriptive (data-derived)** | `study_region_map`, `yield_choropleth_corn_yield_tha`, `yield_volatility_choropleth`, `yield_trends_by_state` | No — computed from the observed yield record before any model is fitted | **Yes.** Valid as they stand |
| **Model-derived** | `residual_choropleth`, `loso_r2_choropleth`, `loso_picp_choropleth`, `ensemble_weight_choropleth` | Yes | **No — regenerate after the full re-run** (see note below) |

The model-derived maps currently in `outputs/figures/` were generated by a
**structural verification run that used the full 22,737-row dataset but a
3-epoch training budget** (`BASELINE_MAX_EPOCHS=3` instead of the configured
200). That run exists to prove the pipeline executes end-to-end and that each
mechanism is genuinely wired; its *metric values* are not publishable. In that
run the neural backbone reached a validation R² of −0.363 and the LOSO ensemble
consequently down-weighted it to w=0.10, so the LOSO and ensemble-weight maps
chiefly depict LightGBM behaviour under a handicapped neural component. **These
four figures must be regenerated from the full-budget re-run before appearing in
the paper.** No pipeline code change is required — only a re-run.

#### 5.10.1 Descriptive maps (valid now)

**`study_region_map.png`** — the 583 counties actually in the dataset, coloured
by state. This is the locator figure that makes the six-state scope explicit and
auditable: Illinois 102, Indiana 92, Iowa 99, Minnesota 87, Missouri 115, Ohio
88. It also visually supports the design argument of `CDHW_DESIGN_REPORT.md` —
the region spans roughly 7° of latitude and a substantial growing-degree-day
gradient while remaining predominantly rainfed, which is why Nebraska (heavily
irrigated) was excluded. The figure should be captioned as *Corn Belt* coverage,
never as national coverage.

**`yield_choropleth_corn_yield_tha.png`** — mean county corn yield over
1985–2023. It reproduces the expected Corn Belt gradient, brightest across
central Iowa and Illinois and falling toward the southern and northern margins:

| State | Counties | Mean yield (t ha⁻¹) | SD | CV (%) |
|---|---|---|---|---|
| Iowa | 99 | 9.637 | 2.223 | 23.1 |
| Illinois | 102 | 9.331 | 2.427 | 26.0 |
| Indiana | 92 | 9.046 | 2.015 | 22.3 |
| Minnesota | 87 | 8.590 | 2.498 | 29.1 |
| Ohio | 88 | 8.566 | 1.986 | 23.2 |
| Missouri | 115 | 7.354 | 2.092 | 28.4 |

This map serves as a **sanity check on the target variable**: recovering the
known agronomic gradient from the assembled data is evidence the yield record
was joined to geography correctly.

**`yield_volatility_choropleth.png`** — within-county yield standard deviation.
This is the figure that carries genuine interpretive weight for the paper's
uncertainty argument, because **volatility is measured independently of the
model**. Minnesota (CV 29.1%) and Missouri (CV 28.4%) are the two most volatile
states, and they are the same two states expected to be hardest to predict. When
the full re-run produces per-state LOSO R², this map allows the spatial result
to be explained by an independently measured property of the data rather than by
post-hoc reasoning about model behaviour.

**`yield_trends_by_state.png`** — per-state yield time series across 1985–2023,
showing the secular technology trend that motivates the train-only linear
detrending in §4. It also makes the temporal split legible: the reader can see
that the 2019–2023 test window sits at the high end of the trend, which is
precisely why extrapolative generalization is harder than the random-split
benchmark of §5.9.

#### 5.10.2 Model-derived maps (regenerate before use)

**`loso_r2_choropleth.png`** and **`loso_picp_choropleth.png`** — each state
coloured by the accuracy and coverage obtained when that state was held out
entirely. These are the clearest single visuals for the spatial-generalization
objective, because they show *per-fold* performance rather than a single mean,
making any weak fold immediately visible instead of averaged away. The
accompanying caption must state that folds are spatially disjoint and that all
preprocessing was re-fitted within each fold (verified: `outputs/audits/loso_leakage_audit.md`,
"LOSO Isolation Status: PASSED (Strict Per-Fold Re-Fitting)").

**`ensemble_weight_choropleth.png`** — the NeuralCQR weight selected per fold.
The weight is chosen on the **validation** partition, never on the held-out
state, so this map reports a modelling decision rather than a tuned outcome. It
is worth publishing even when it is unflattering: a consistently low neural
weight is an honest statement about the backbone's contribution, and suppressing
it would misrepresent where the predictive performance originates.

**`residual_choropleth.png`** — mean (predicted − observed) test residual by
county, on a diverging red/blue scale. This map is a **falsifiable check**, not
decoration. The negative-R² diagnostic attributes backbone error primarily to
spatial heterogeneity rather than temporal drift; if that is correct, residuals
should form contiguous geographic clusters. If instead they appear as spatially
unstructured noise, that outcome *contradicts* the diagnostic and must be
reported as such. The interpretation is therefore fixed in advance of the
re-run, so the figure cannot be read selectively after the fact.

#### 5.10.3 Suggested figure placement

Two figures belong in the main text — `study_region_map` (defines and bounds the
study scope) and `loso_r2_choropleth` (the spatial-generalization headline).
`yield_choropleth` and `yield_volatility_choropleth` support the results
discussion, and `residual_choropleth`, `loso_picp_choropleth`,
`ensemble_weight_choropleth` and `yield_trends_by_state` are well suited to
supplementary material.

---

## 6. Datasets Required

### 6.1 Methodology Datasets

| Dataset / Source | Variables / Content | Licence | Stream |
|---|---|---|---|
| ERA5 Daily Reanalysis (ECMWF CDS) | Tmax, Tmin, precipitation, VPD, ET₀ at county centroids; 1985–2023; daily resolution | Free research / commercial | Weather encoder + CDHW |
| **SPEI-30** (derived from ERA5 precip + ET₀) | 30-day SPEI computed from ERA5 precipitation and Penman-Monteith ET₀ using 1981–2010 gamma/log-logistic baseline | Derived — public domain inputs | CDHW indicator |
| FAO-56 ET₀ (derived from ERA5) | Reference evapotranspiration via Penman-Monteith from ERA5 temperature, radiation, wind, humidity | Derived — public domain inputs | ET deficit computation + SPEI input |
| Sentinel-2 / MODIS NDVI (ESA / NASA) | NDVI time series for growing season; used to anchor phenological stages | Public domain | NeuralCQR input + phenology-aligned CDHW windows |
| USDA NASS County Yield Surveys | Annual corn and soybean yield (t/ha) per county; ground-truth labels for PICP evaluation | Public domain | Labels + PICP eval |
| NOAA Drought Monitor / ENSO Records | Historical drought and ENSO event classifications; used to identify anomalous test years | Public domain | Year stratification |
| FAO Ky Crop Response Factors | Corn: Ky = 1.25; Soybean: Ky = 0.85; used in ET deficit → yield loss scaling | FAO technical report (public) | Physics bias in PICA |
| DSSAT Soft-Target Predictions | Random Forest meta-model trained on 100,000+ DSSAT simulations; used as L_phys soft target | Research licence (DSSAT) | Loss function |
| GDD Accumulation (derived) | Growing degree days from ERA5 Tmax/Tmin; base 10°C, cap 30°C for corn | Derived — public domain inputs | Phenological stage encoding + phenology-aligned CDHW windows |

*(Note: SPI-30 is superseded by SPEI-30 above; no new external data source is required since ET₀ was already in the pipeline.)*

*(**[NEW v2]** LOSO-CV in §5.2 requires no new data source either — county-level records already carry a state identifier, so folds are constructed by grouping existing records rather than acquiring anything new.)*

### 6.2 Complete Dataset Checklist

Study Period — Training: 1985–2015 | Validation: 2016–2018 | Testing: 2019–2023

| Dataset | Source | Years | Resolution | Required |
|---|---|---|---|---|
| ERA5 Weather | Copernicus | 1985–2023 | Daily | ✅ |
| Corn Yield | USDA NASS | 1985–2023 | County | ✅ |
| Soybean Yield | USDA NASS | 1985–2023 | County | ✅ |
| Drought Monitor | U.S. Drought Monitor | 2000–2023 | Weekly | ✅ |
| Storm Events | NOAA | 1985–2023 | Event | ✅ |
| ENSO (Oceanic Niño Index) | NOAA CPC | 1985–2023 | Monthly | ✅ |
| DSSAT Simulations | DSSAT | Match study period | County | Optional |
| MODIS NDVI (MOD13Q1) | NASA LP DAAC | 2000–2023 | 250 m (16-day) | Optional |

### 6.3 Datasets NOT Required for This Paper

- Sentinel-2 (beyond phenology anchoring above) — Not required
- Landsat (unless studying vegetation uncertainty) — Not required
- SSURGO Soil — Not required
- DEM — Not required
- CDL / Crop Mask — Not required
- Fertilizer — Not required
- Irrigation — Not required

---

## 7. Recommended Paper Sequence

The three papers are designed to be written and executed in the following order, reflecting increasing data and compute requirements:

- **Paper 3 first:** Least data-heavy. Only requires ERA5 + NASS yield surveys (SPEI-30 requires no new source, since ET₀ is already ERA5-derived). Can be completed before satellite data is processed. Validates the uncertainty framework independently, now with an explicit comparison against the field's other shift-robust conformal methods rather than only against the static baseline.
- **Paper 2 second:** Requires ERA5 + SSURGO + NASS yields. No satellite imagery. PCMCI+ can be pre-computed offline on a standard workstation. Validates the spatial graph component independently — and can reuse the spatial-autocorrelation diagnostic introduced in §4.4 of this paper.
- **Paper 1 last:** Requires all four data streams simultaneously. Builds on validated components from Papers 2 and 3. Highest compute requirement (single A100 GPU, ~6–8 hours).

This sequencing allows incremental results and publication while the full CropFusion pipeline is being completed.
