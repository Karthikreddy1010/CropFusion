# Literature Gap Analysis

Written before any model design, as the brief requires. The gap determines the
model; the existing implementation had no vote.

---

## 1. Existing literature study — summary

`LITERATURE_REVIEW_CORRECTED.md`: 32 verified entries.

- **A (15)** conformal & quantile foundations — Vovk 2005, Romano 2019 (CQR),
  Gibbs & Candès 2021 (ACI), Barber 2023 (beyond exchangeability), Tibshirani
  2019 (covariate shift), Koenker & Bassett 1978, Meinshausen 2006, Lei &
  Wasserman 2014, Kivaranovic 2020, Gneiting 2007.
- **B (12)** compound drought–heat climatology — Zscheischler 2020, Lesk 2022,
  Heino 2023, He 2022, Tripathy 2023, Daryanto 2016/2017, Yan 2025, Sutanto 2024.
- **C (6)** crop yield + UQ — Ma 2021 (BNN), Zhang & Diao 2023 (PB-CNN),
  Farag 2025 (CP for crop decision support), Ren 2023, Jabed 2024, Onogi 2025.
- **D (4)** topics with no verified replacement.

## 2–3. Additional search and papers added

See `ADDITIONAL_LITERATURE_SEARCH.md`. Six papers (N1–N6); three changed the
methodology decision.

## 4. Evidence matrix — methodological patterns, not summaries

| Pattern | Papers | What it establishes |
|---|---|---|
| Conformal guarantees are **marginal by construction** | A1, A3, A8, N1 | Coverage is averaged over the joint distribution — not per-subpopulation |
| **Conditional coverage is impossible** distribution-free | **N2**, N1 | Cannot be attained in finite samples without extra assumptions |
| Adaptivity targets the **marginal** rate | A4 (ACI), N4 (AgACI) | `alpha_t` tracks aggregate miscoverage; no per-regime promise |
| Exchangeability fails on dependent data | A5, N3 | Validity holds only approximately, bounded by non-exchangeability |
| Methods exist for **temporal dependence** | **N3** (EnbPI) | Strongly-mixing errors, no exchangeability — but still marginal |
| **Partition-conditional** calibration is the remedy | N1, **N5** | Calibrate within strata → validity *within* stratum |
| Compound drought–heat is super-additive on maize | B1, B5, B8, B11 | Establishes the phenomenon, not its predictability |
| Crop-yield UQ papers report **marginal coverage only** | C1, C2, C3, **N6** | Including the 2026 state of the art (one number: 80.72%) |
| Richer inputs, not richer models, drive crop-yield gains | C1, C2, C5 | Their advantage is satellite time series, not architecture |

**The cross-cutting pattern:** every methodological paper is explicit that
guarantees are marginal, and every *application* paper reports a single marginal
coverage number — while the applications are motivated by **extreme events**,
which are precisely a conditional question.

## 5. Established solutions — NOT gaps

- Distribution-free marginal coverage (A1, A3)
- Marginal adaptation under distribution shift (A4)
- Covariate-shift reweighting (A9)
- Validity bounds under non-exchangeability (A5)
- Marginal validity under temporal dependence (N3)
- Compound drought–heat yield damage as a phenomenon (B1–B11)
- Partition-conditional calibration as a mechanism (N1, N5)

## 6. Partially solved

- **Conformal under spatial dependence** — real work exists, not pinned here
- **EVT / tail risk for crop loss** — only a preprint (C6)
- **Phenology-aware UQ** — C2 is Bayesian, not conformal

## 7. Genuine unresolved gaps

**Nothing in the "genuinely unresolved" category survives scrutiny for this
project.** Conditional coverage is not unresolved — it is *proven impossible*
(N2). Stating otherwise would be the kind of manufactured novelty this brief
forbids.

## 8–9. Candidate gaps, ranked

| # | Candidate gap | Classification | Novelty | Feasible here | Rank |
|---|---|---|---|---|---|
| **1** | **Marginal calibration metrics misrank conformal methods for extreme-regime agricultural use** | **Insufficiently evaluated** | Moderate | **Yes — evidence in hand** | **1** |
| 2 | Conformal validity under temporal dependence in crop yield | Partially solved (N3 exists, untested in this domain) | Low–moderate | Yes | 2 |
| 3 | Partition-conditional calibration by climate regime for crop yield | Mechanism solved (N1/N5), untested in this domain | Low–moderate | Yes | 3 |
| 4 | A conformal method with conditional guarantees | **Already proven impossible (N2)** | — | No | reject |
| 5 | Better CDHW representation improving prediction | Tested — negative four ways | — | Feasible, unpromising | reject |
| 6 | Spatial conformal for county yield | Partially solved | Low | Weak — Moran's I 0.19 | reject |
| 7 | Novel architecture for yield prediction | Solved; gains come from inputs (C1, C2) | None | Backbones differ by 0.026 R² < seed SD 0.037 | reject |

## 10. Selected gap and why it is real

> **Conformal methods in agricultural yield UQ are selected and reported on
> marginal calibration. Where the application is motivated by extreme events —
> as it always is — the operationally relevant quantity is conditional coverage
> within those regimes. Whether marginal-optimal selection misranks methods for
> that purpose has not been systematically evaluated in this domain.**

**Honest classification: "insufficiently evaluated", not "genuinely
unresolved".** The theory (N1, N2) is settled. What is missing is the
domain-specific demonstration that it *changes the answer*.

Three independent reasons it is real:

1. **The most recent competing work reports one marginal number.** N6 (2026)
   — the state of the art in exactly this intersection — reports "80.72%
   empirical coverage" with no regime stratification and no comparison among
   conformal methods. C3 (2025) and C1/C2 likewise.
2. **Even the methodological correction evaluates marginally.** N4 fixes ACI for
   dependent time series and still reports marginal coverage.
3. **It changes the answer, decisively.** In this project's own data the
   marginal and conditional rankings are **perfectly inverted**:

| Method | Marginal ACE | O6 rank | Worst conditional PICP |
|---|---|---|---|
| locally_adaptive | 0.0412 (worst) | 5.000 (worst) | **0.8801** (best) |
| static_conformal | 0.0245 | 3.333 | 0.8621 |
| **aci** | **0.0096** (best) | **2.000** (best) | **0.7602** (worst) |

ACI is worst in **all six** regime types tested — year, state, CDHW severity,
drought, heat, yield regime — reaching 0.7602 in 2021 (24% miss rate at nominal
10%). A practitioner selecting on marginal ACE would deploy the method that
fails hardest exactly where the application demands reliability.

## 11. Closest existing papers

| Paper | How close | What it does not do |
|---|---|---|
| **N6** Mahmood 2026 | Same intersection, most recent | Single marginal coverage number; no regimes; no method comparison |
| **C3** Farag 2025 | CP for crop decision support | Analyses CP as a tool; no extreme-regime conditional evaluation |
| **N4** Zaffran 2022 | Fixes ACI for dependence | Marginal evaluation only |
| **N2** Barber 2020 | The governing theory | Theoretical; no agricultural instantiation |
| **A4** Gibbs & Candès 2021 | The method under test | Marginal by design; not a failure of the paper |

## 12. How the proposed approach differs

Not by proposing a new estimator. By **changing the selection criterion** and
testing the consequence:

- evaluate all five calibration methods **within** climate/temporal/spatial
  regimes, not only marginally;
- report the rank inversion where it occurs;
- apply the literature-backed remedy (Mondrian/partition-conditional calibration,
  N1/N5) and the assumption-appropriate method (EnbPI, N3);
- state the impossibility result (N2) rather than implying conditional
  guarantees.

## 13. Research hypothesis

> **H1.** For county-level corn yield under compound drought–heat stress,
> conformal calibration methods ranked best on **marginal** calibration error
> are **not** those ranked best on **conditional** coverage within
> extreme-climate regimes, and the divergence is large enough to reverse method
> selection.
>
> **H2.** Partition-conditional (Mondrian) calibration over climate-regime strata
> improves worst-regime coverage relative to marginal calibration, at a cost in
> marginal sharpness.
>
> **H3.** Because residuals are non-exchangeable (lag-1 ACF 0.5404), a method
> assuming only strong mixing (EnbPI) will be better calibrated than methods
> assuming exchangeability.

H1 is already supported by evidence in hand. H2 and H3 are untested and are the
work to do.

## 14–15. Candidate designs, and the selected one

| Design | From | Status |
|---|---|---|
| **D1. Regime-stratified evaluation protocol** | N1, N2 | **SELECTED** — the gap is an evaluation gap |
| **D2. Mondrian conformal over climate-regime strata** | N1, N5 | **SELECTED** — tests H2 |
| **D3. EnbPI** | N3 | **SELECTED** — tests H3 |
| D4. Conditional conformal with guarantees | — | **REJECTED** — impossible (N2) |
| D5. Heteroscedastic / variance-aware quantile net | — | **REJECTED** — problem is location, not variance |
| D6. New architecture (transformer, GNN) | N6, C2 | **REJECTED** — gains there come from satellite inputs; backbone spread here is inside seed noise |
| D7. Spatial conformal | band D | **DEFERRED** — Moran's I 0.19 vs ACF 0.54 |

## 16. Why the rejected designs were rejected

- **D4** — N2 proves it cannot be done distribution-free. Claiming it would be
  the manufactured novelty this brief prohibits.
- **D5** — the residual audit shows coverage tracks year-level **bias**
  (corr(|bias|, PICP) ≈ −0.8 to −0.9), and a 16% change in raw interval width
  moved final coverage by 0.0001. Variance methods target the wrong axis.
- **D6** — CatBoost 0.4597, XGBoost 0.4585, NeuralCQR 0.4462, LightGBM 0.4335:
  spread 0.026 against a seed SD of 0.037. The backbones are indistinguishable;
  more architecture cannot help. C1/C2's advantage is remote-sensing inputs, a
  data project rather than a model one.
- **D7** — deferred on measured evidence, not preference.

---

## The uncomfortable implication, stated plainly

**The literature-driven answer is a simpler pipeline, not a more advanced one.**
The selected designs change *how calibration is stratified* and *which
assumptions are invoked*. None adds a network, a loss term, or a feature family.

The brief said not to force the proposed model to outperform, and not to call a
combination novel merely for being combined. Applied honestly, the evidence says
this project's contribution is an **evaluation and selection** result, not an
architectural one — and that ACI, its current headline, should be demoted to a
studied method rather than the recommended one.
