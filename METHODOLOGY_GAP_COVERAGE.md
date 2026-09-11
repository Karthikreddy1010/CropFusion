# Methodology ↔ Literature Gap Coverage Audit

Built from `LITERATURE_REVIEW_CORRECTED.md` (32 verified entries),
`ADDITIONAL_LITERATURE_SEARCH.md` (6 additional verified papers, N1–N6) and the
validation experiments already run on development data only.

A gap is listed only where the literature supports it. Paper limitations that no
other work treats as unresolved are **not** counted as gaps.

---

## 1. Identified gaps

| ID | Area | Supporting papers | Already solved | Still unresolved | Genuinely open? |
|---|---|---|---|---|---|
| **G1** | Conditional vs marginal coverage in agricultural UQ | N1, N2, C3, N6, A4 | Marginal validity (A1, A3); marginal adaptation (A4) | Whether marginal-optimal selection **misranks** methods for extreme regimes | **YES — insufficiently evaluated** |
| **G2** | Conformal validity under temporal dependence | A5, N3, N4 | Validity bounds (A5); EnbPI under strong mixing (N3) | Whether either survives *non-stationarity* (trend), as opposed to mixing | **PARTIAL** |
| **G3** | Partition-conditional calibration for crop yield | N1, N5 | Mechanism fully solved (Mondrian) | Whether it is *implementable* when extremes are temporally clustered | **YES — feasibility, not method** |
| G4 | Distribution-free **conditional guarantees** | N2, N1 | — | — | **NO — proven impossible** |
| G5 | CDHW representation improving point prediction | B1–B11 | Phenomenon established (B5, B8, B11) | — | **NO — tested, negative ×4** |
| G6 | Conformal under spatial dependence | band D row 26 | Partially (work exists, unpinned) | Verified citation | **PARTIAL — low priority here** |
| G7 | Architecture for county yield prediction | C1, C2, C4, N6 | Extensively solved | — | **NO** |
| G8 | EVT / tail risk for crop loss | C6 (preprint only) | — | Peer-reviewed treatment | **PARTIAL — out of scope** |
| G9 | Crop-insurance probabilistic framework | none found | — | — | **NO — drop the claim** |

---

## 2. Gap-coverage matrix

| Gap | Evidence | Covered? | Current component | Coverage quality | Missing component | Action |
|---|---|---|---|---|---|---|
| **G1** | ACI best marginal ACE 0.0096 / worst conditional PICP 0.7602; rank inversion across **all six** regime types | **PARTIALLY COVERED** | `conditional_coverage_report.py` (added this audit) | **Strong empirically, absent from the methodology doc until v4** | Nothing — evidence exists; the paper must *lead* with it | **Make it the contribution** |
| **G2** | ACF 0.5404; DW 0.39–0.73 within year; `exchangeable_temporally: false` | **PARTIALLY COVERED** | Exchangeability diagnostics + block bootstrap (§4.4, §5.3) | Diagnosed honestly, **not solved** | A method valid under this dependence — EnbPI tested, **failed** | Report as a stated limitation |
| **G3** | Binary 0 fallbacks / ternary 3 fallbacks across 5 origins; severe stratum n=10 | **NOT COVERED → now tested** | Mondrian added, evaluated, **rejected** | Validation-grade | None | Report the feasibility finding |
| G4 | N2 impossibility | **NOT RELEVANT** | — | — | — | Never claim it |
| G5 | ΔR² −0.055 (post-hoc), −0.040 (joint); max r 0.872 with `Tmax_Days_Above_35` | **FULLY COVERED** | 6-row ablation + interaction analysis | Rigorous, four independent tests | None | Report the negative |
| G6 | Moran's I 0.1918, `spatial_dependence_low: true` | **PARTIALLY COVERED** | LOSO = spatial block validation | Adequate for this data | Spatial conformal | Defer — temporal dominates |
| G7 | Backbone spread 0.026 R² vs seed SD 0.0299 | **FULLY COVERED** | 4-backbone benchmark + ensemble | Rigorous | None | Report indistinguishability |
| G8 | C6 preprint only | **NOT COVERED** | — | — | EVT model | Drop the claim |
| G9 | No source located | **NOT RELEVANT** | — | — | — | Drop the claim |

---

## 3. Per-gap methodology audit (A–F)

### G1 — Conditional vs marginal coverage **[the selected gap]**

**A. Addressed?** Only after this audit. The original methodology reported
marginal PICP/MPIW/ACE/Winkler and selected ACI on that basis.

**B. Rigorously?** Now yes — five methods × six regime types (year, state, CDHW
severity, drought, heat, yield), each scored from **its own** interval arrays,
never reusing another method's coverage vector.

**C. Implementation consistent with methodology?** It is now. Until v4 the
methodology named ACI as the method while the code's own O6 report ranked static
conformal first — a live contradiction, since fixed.

**D. Evaluation sufficient?** Yes. The rank inversion is perfect and holds in
every regime, on n=2,345 test rows.

**E. Missing experiment?** No.

**F. Literature-supported?** Yes — N1, N2 supply the theory; N6 (2026) and C3
(2025) demonstrate the field still reports one marginal number.

### G2 — Temporal dependence

**A/B.** Diagnosed thoroughly (ACF, DW, Moran's I, block bootstrap) but **not
solved**. EnbPI was implemented specifically to address it and failed
(PICP 0.6336; width rigid at 2.38–2.59 vs static 3.97–6.89).

**C.** Consistent — §4.4 already states exchangeability is violated.

**D.** Sufficient to state the limitation; **not** sufficient to claim a remedy.

**E. Missing experiment:** a fair EnbPI test with a residual pool matched in
recency to the other methods. Currently its pool is the whole fit set.

**F.** Yes — A5, N3, N4.

### G3 — Partition-conditional calibration

**A.** Not in the original methodology; added and tested this audit.

**B.** Rolling-origin over 5 development origins, both binary and ternary strata,
with an explicit small-stratum guard.

**D.** Sufficient — and the answer is negative: +0.006 ± 0.054, 2/5 origins.

**E.** None. **F.** Yes — N1, N5.

### G5 — CDHW predictive contribution

**A–D.** Fully covered and rigorously: ablation (post-hoc −0.055, joint −0.040),
top-k, feature-family ablation, and a redundancy analysis showing every CDHW
feature's strongest correlate is `Tmax_Days_Above_35` (max r 0.872).

**E. Missing experiment:** none for prediction. **The upstream CDHW generation
code remains unavailable**, so drought/heat thresholds, climatological baseline
and event detection could not be audited at all.

**F.** Yes — B1–B11 establish the phenomenon; none claims it must improve a
statistical yield model that already contains heat-day counts.

---

## 4. Where the current methodology actually fails

Ranked by consequence:

1. **It selected the calibration method on the wrong criterion.** ACI was chosen
   and headlined on marginal ACE. That is G1, and it is the paper.
2. **It stated a premise its own data contradicts.** §2 claimed static conformal
   "collapses below nominal coverage during anomalous climate years". Static
   reaches 0.9245; **ACI** is the only method below nominal at 0.8904. The
   premise came from an unverifiable citation. Corrected in v4.
3. **Its reported coverage assumed exchangeability that the data violates.** Now
   stated as empirical rather than guaranteed.
4. **Its point predictions are mis-centred, and that — not interval width —
   drives the coverage failures.** `corr(|bias|, PICP)` across test years is
   **−0.799**. This *is* a modelling weakness, and no component tested so far
   addresses it. See `FINAL_METHODOLOGY_AUDIT.md` §I.4.

Items 1–3 are evaluation and reporting failures. **Item 4 is not** — so the
answer to §15 of the brief needs qualifying: the *contribution* is an
evaluation result, and no new **architecture** is warranted, but there is a real
modelling weakness underneath it. The candidate fix (asymmetric conformal
scores) is cheap and untested, and it may not work at all: year-level bias
**changes sign** (−0.487 … +0.691), so it cannot be predicted in advance from
a calibration window. If it fails, that is itself a reportable finding — it
would mean the 2021 coverage collapse is irreducible with this feature set,
which strengthens the structural reading of G1 rather than weakening it.
