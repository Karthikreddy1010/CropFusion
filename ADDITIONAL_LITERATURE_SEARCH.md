# Additional Literature Search

## Was the existing literature study sufficient?

**No.** `LITERATURE_REVIEW_CORRECTED.md` holds 32 verified entries across four
bands and is strong on conformal foundations (A1–A15), compound drought–heat
climatology (B1–B11) and crop-yield UQ (C1–C6).

Assessed against the 24 topic areas the brief requires, **eight were absent or
thin**, and one of those absences is decisive: the review contains **no paper on
conditional versus marginal coverage** — the exact theory underlying this
project's strongest empirical result.

| Required area | Existing coverage | Verdict |
|---|---|---|
| Conformal foundations, CQR, ACI, covariate shift | A1–A15 | sufficient |
| Compound drought–heat, drought impacts, ENSO | B1–B11 | sufficient |
| Crop yield + UQ | C1–C6 | sufficient |
| **Conditional coverage / its impossibility** | **absent** | **searched** |
| **Conformal under temporal dependence** | only A5 (general) | **searched** |
| **Online conformal beyond ACI** | absent | **searched** |
| **Group / Mondrian / stratified calibration** | absent | **searched** |
| Conformal under spatial dependence | flagged unresolved (band D, row 26) | searched, unresolved |
| Heteroscedastic / distributional regression | absent | not searched — see note |
| Deep ensembles | absent | not searched — see note |

*Note on the last two:* the residual audit established that this project's
problem is a **year-level location shift**, not variance misspecification
(corr(|bias|, PICP) ≈ −0.8 to −0.9 across methods; a 16% change in raw interval
width moved coverage by 0.0001). Searching variance-modelling literature would
have been searching for a solution to a problem the data says does not exist.
Recorded as a deliberate omission rather than an oversight.

---

## Papers found

All verified against Crossref by DOI or exact-title match. No author list is
reconstructed from memory.

### N1 — Vovk, V. (2013)
**Conditional validity of inductive conformal predictors**
*Machine Learning* 92:349–376, doi:10.1007/s10994-013-5355-6

*Why searched:* the project's central result is a marginal/conditional
divergence, and the review had nothing on the distinction.

*What it contributes:* the formal statement that inductive conformal predictors
are **valid marginally but not conditionally**, and the construction of
Mondrian/label-conditional predictors as the partial remedy.

*Changed the methodology decision:* **Yes.** It reframes ACI's behaviour from a
defect into an instance of a known theoretical limit, and it names the class of
remedy (partition-conditional calibration).

### N2 — Barber, R.F., Candès, E.J., Ramdas, A., & Tibshirani, R.J. (2020)
**The limits of distribution-free conditional predictive inference**
*Information and Inference: A Journal of the IMA* 10:455–482, doi:10.1093/imaiai/iaaa017

*Why searched:* to establish whether conditional coverage is achievable at all
before proposing a method to achieve it.

*What it contributes:* an impossibility result — distribution-free
**conditional** coverage cannot be attained in finite samples without additional
assumptions. Any method promising it is either assuming more than
distribution-free, or producing uninformative intervals.

*Changed the methodology decision:* **Yes, decisively.** It rules out an entire
class of candidate approaches. The contribution cannot be "a conformal method
with conditional guarantees"; it can only be evaluation, partitioning, or an
explicit assumption.

### N3 — Xu, C., & Xie, Y. (2021)
**Conformal prediction interval for dynamic time-series**
*ICML*, PMLR 139 — journal version: **Conformal Prediction for Time Series**,
*IEEE TPAMI* 45:11575–11587 (2023), doi:10.1109/tpami.2023.3272339

*Why searched:* this project's residuals are demonstrably non-exchangeable
(lag-1 ACF 0.5404, `exchangeable_temporally: false`), which voids the assumption
behind every conformal method currently implemented.

*What it contributes:* EnbPI — sequential intervals with approximately valid
**marginal** coverage under *strongly mixing* errors, requiring **no
exchangeability** and no data splitting.

*Changed the methodology decision:* **Yes.** It is the first method found whose
stated assumptions this dataset actually satisfies. Note carefully: its
guarantee is still **marginal**, so it addresses the dependence problem, not the
conditional-coverage problem. The two are separate.

### N4 — Zaffran, M., Dieuleveut, A., Féron, O., Goude, Y., & Josse, J. (2022)
**Adaptive Conformal Predictions for Time Series** — arXiv:2202.07282

*Why searched:* to find work correcting ACI specifically for temporal dependence.

*What it contributes:* AgACI, an aggregated ACI for non-exchangeable series, with
theoretical analysis of the autoregressive case. **Evaluates marginal coverage
only.**

*Changed the methodology decision:* **Indirectly — it strengthens the gap.** The
paper devoted to fixing ACI for dependent time series still selects and reports
on marginal coverage. Crossref returned no published version; treat as a
preprint.

### N5 — Toccaceli, P., & Gammerman, A. (2018)
**Combination of inductive Mondrian conformal predictors**
*Machine Learning* 108:489–510, doi:10.1007/s10994-018-5754-9

*Why searched:* Mondrian conformal is the established route to
partition-conditional validity, following from N1.

*What it contributes:* practical construction and combination of Mondrian
predictors — calibrate within predefined strata to obtain validity *within* each.

*Changed the methodology decision:* **Yes.** It is the concrete, literature-backed
mechanism for the regime-conditional problem, and it is far simpler than any
architectural change.

### N6 — Mahmood, S., Hasan, R., & Ahmad, S. (2026)
**HSE-GNN-CP: Spatiotemporal Teleconnection Modeling and Conformalized
Uncertainty Quantification for Global Crop Yield Forecasting**
*Information* 17(2):141, doi:10.3390/info17020141

*Why searched:* to find the most recent competing crop-yield + conformal work
and test whether the gap is already closed.

*What it contributes:* the current state of the art in this exact intersection —
stacked ensembles + GNN + conformal prediction, R² 0.9594, reporting **"80.72%
empirical coverage"** against an 80% nominal.

*Changed the methodology decision:* **Yes — it confirms the gap is open.** The
most recent and most sophisticated paper in this intersection reports **a single
marginal coverage number**, with no regime-stratified evaluation and no
comparison among conformal calibration methods. (Full text is behind an MDPI
fetch block; assessment is from the Crossref abstract and should be confirmed
against the PDF before citing as a contrast.)

---

## Still unresolved after searching

**Conformal prediction under spatial dependence.** Band D row 26 of the existing
review flagged this, and targeted Crossref queries returned no clean match.
Real work exists (GeoConformal; localized-quantile spatial conformal) but was
not pinned to a verified citation here.

This matters less than it appears: the spatial audit measured Moran's I = 0.1918
on residuals with `spatial_dependence_low: true`, against a lag-1 temporal ACF
of 0.5404. **Temporal dependence is the binding constraint**, and N3 addresses
it. Recorded as an open item, not a blocker.

---

## Net effect on the methodology decision

Three of the six new papers changed the decision, and all three push in the same
direction — **away from architectural complexity**:

1. **N2 forbids** the obvious framing. No method can deliver distribution-free
   conditional guarantees, so "a conformal method that fixes conditional
   coverage" is not available as a contribution.
2. **N1 + N5 supply** the only literature-backed remedy: partition-conditional
   (Mondrian) calibration — a change to *how calibration is stratified*, not to
   the model.
3. **N3 supplies** the method whose assumptions this data actually meets, for
   the separate dependence problem.

None of the three requires a new architecture. Two of them make the pipeline
*simpler*. That is the outcome the evidence supports, and it is recorded here
before any model was designed.
