# Literature Citation Verification — COMPLETE (all 36 entries)

**This is the single most important finding of the entire audit.** Every one
of the 36 entries in `Literature_Review_Paper3_ConformalInference.xlsx` was
checked against live web search — by exact title, and where that failed, by
the entry's own distinctive content (specific methods, datasets, and the
exact numeric findings claimed in its "Results / Key Findings" column). A
real paper is findable by at least one of those routes.

**Result: 21 of 36 verified as real. 15 of 36 could not be verified.**

A structural red flag worth stating up front: the spreadsheet lists **no
author names, no journal/venue, and no DOI** for any entry — only
Title/Year/Gap/Methods/Findings/Dataset/Category. That is not how a real
literature review is normally compiled, and it is precisely the format in
which a fabricated entry is indistinguishable from a real one at a glance.

---

## The pattern

The failures are not scattered — they fall in one contiguous band:

| Band | Rows | Character | Verification |
|---|---|---|---|
| **A** | 1–12, 16, 17, 25 | Foundational conformal-prediction / statistics / climate-science theory | **15 / 15 REAL** |
| **B** | 13–15, 18–24, 26–30 | "ML / uncertainty quantification applied to agriculture" + 3 climate papers | **0 / 15 REAL** |
| **C** | 31–33, 36–38 | Recent (2022–2025) compound drought-heatwave literature | **6 / 6 REAL** |

Every entry outside band B verifies. Every entry inside band B fails. Band B
is precisely the set of citations needed to bridge general conformal-prediction
theory (band A, real) to this paper's specific agricultural CDHW/phenology/ACI
claims — i.e. exactly the gap a fabricated citation would be invented to fill.

---

## Band A — CONFIRMED REAL (15/15)

| # | Title | Verification |
|---|---|---|
| 1 | Algorithmic Learning in a Random World | Vovk, Gammerman & Shafer — Springer, 1st ed. 2005 ✓ |
| 2 | A Gentle Introduction to Conformal Prediction and Distribution-Free UQ | Angelopoulos & Bates — arXiv:2107.07511 ✓ (review says 2023; arXiv 2021, FnT version 2023 — defensible) |
| 3 | Conformalized Quantile Regression | Romano, Patterson & Candès — NeurIPS 2019 ✓ |
| 4 | Adaptive Conformal Inference Under Distribution Shift | Gibbs & Candès — 2021 ✓ |
| 5 | Conformal prediction beyond exchangeability | Barber, Candès, Ramdas & Tibshirani — *Ann. Statist.* 51(2), 2023 ✓ exact year |
| 6 | Regression Quantiles | Koenker & Bassett — *Econometrica* 46(1):33–50, 1978 ✓ exact |
| 7 | Strictly Proper Scoring Rules, Prediction, and Estimation | Gneiting & Raftery — *JASA* 102(477):359–378, 2007 ✓ exact |
| 8 | A Tutorial on Conformal Prediction | Shafer & Vovk — *JMLR* 9:371–421, 2008 ✓ exact |
| 9 | Conformal Prediction Under Covariate Shift | Tibshirani et al. — 2019 ✓ |
| 10 | Distribution-Free Prediction Bands for Non-Parametric Regression | Lei & Wasserman — 2014 ✓ |
| 11 | A typology of compound weather and climate events | Zscheischler et al. — *Nat. Rev. Earth Environ.* 1:333–347, 2020 ✓ exact |
| 12 | Anthropogenic contribution to global occurrence of heavy-precipitation and high-temperature extremes | Fischer & Knutti — *Nat. Clim. Change*, 2015 ✓ exact |
| 16 | Quantile Regression Forests | Meinshausen — 2006 ✓ |
| 17 | Adaptive, Distribution-Free Prediction Intervals for Deep Neural Networks | Kivaranovic, Johnson & Leeb — arXiv:1905.10634 (2019), AISTATS 2020 ✓ (canonical title says "Deep Networks"; the "Deep Neural Networks" variant is used on ResearchGate/SlidesLive, so the review's title is acceptable) |
| 25 | Online Learning and Prediction with Expert Advice | Matches the real Hedge / expert-advice literature (Cesa-Bianchi & Lugosi) ✓ as a field reference |

## Band C — CONFIRMED REAL (6/6)

| # | Title | Verification |
|---|---|---|
| 31 | Intensifying Impacts of Compound Drought and Heatwave Events on WUE in US Corn and Soybean | *Agric. For. Meteorol.*, 2025 ✓ exact title |
| 32 | Compound and Cascading Droughts and Heatwaves Decrease Maize Yields by Nearly Half | *npj Natural Hazards*, 2024, Wageningen — Sinaloa, Mexico ✓ exact title |
| 33 | Climate Change Will Accelerate the High-End Risk of Compound Drought and Heatwave Events | *PNAS* ✓ real |
| 36 | Enhancing Decision Support in Crop Production: Conformal Prediction for UQ | Real — actual title includes "**Analyzing** Conformal Prediction"; *ScienceDirect*, 2025. Minor one-word title drift |
| 37 | A Review of Data for Compound Drought and Heatwave Stress Impacts on Crops | Li, Zeleke, Wang & Liu — *Plants*, **2025** (review says 2024) ✓ real, year off by one |
| 38 | Compound Heat and Moisture Extreme Impacts on Global Crop Yields under Climate Change | *Nat. Rev. Earth Environ.* 3(12):872–889, **2022** (review says 2021) ✓ real, year off by one |

## Band B — COULD NOT BE VERIFIED (0/15 real)

| # | Title as given | What searching actually found instead |
|---|---|---|
| 13 | Compound drought and heat stress: global prevalence and climate change impacts | Real CDHE papers using SPEI/STI/SSMI + vine-copula, and CMIP6 Tmax+sc-PDSI frameworks — none with this title or these claims |
| 14 | Probabilistic Crop Yield Forecasting Using Bayesian Neural Networks | Ma et al. 2021 (real, different title, R²=0.77 not the claimed PICP 88.3%) |
| 15 | Uncertainty Quantification in Deep Learning for Yield Prediction | General MC-Dropout / deep-ensemble / conformal UQ literature — no such MODIS+ERA5+NASS paper with the claimed 0.91-vs-nominal-0.90 result |
| **18** | **Limitations of Static Conformal Prediction for Agricultural Yield Forecasting** | **Nothing. Searched by title, by the claimed 91%→79%→73% coverage-collapse finding, and by the ENSO/CDHW framing. This is the paper the entire Introduction is built on.** |
| 19 | Time Series Conformal Prediction with Temporal Autocorrelation Correction | Real but different: AcMCP, Temporal Conformal Prediction (TCP), rolling-origin conformal — none matching the claimed 89.7% AR(1) result |
| 20 | Agricultural Risk Assessment Under Extreme Weather: A Probabilistic Framework | Real crop-insurance/quantile/copula literature — no QRF+BNN+CQR paper with the claimed 0.82 P10-vs-claims correlation |
| **21** | **Phenology-aware uncertainty quantification for corn yield prediction** | **Nothing. Closest real work (Phenology-guided Bayesian-CNN) is soybean, different method, different numbers. This is the paper §1 claims this work extends.** |
| 22 | El Niño impacts on US crop production: Quantifying the role of precipitation and temperature | Real ENSO-yield literature (e.g. Iizumi et al.) — no SEM+DSSAT Indiana mediation study as described |
| 23 | Coverage-Optimal Prediction Sets Under Distribution Shift | Real underlying concept exists as **Cauchois et al., "Robust Validation: Confident Predictions Even When Distributions Shift"** (f-divergence ball) — but not under this title |
| 24 | Drought stress timing and crop yield loss: A meta-analysis | The underlying agronomy is real and well-established (pollination-stage drought is most damaging, ~9%/day). No 2,100-experiment meta-analysis with the claimed 32%-vs-8% figures |
| 26 | Spatially aware uncertainty quantification for geographic distribution shift | Real but different: GeoConformal Prediction (GeoCP), GeoSIMCP, Localized Spatial CP — none with the claimed leave-one-USDA-zone-out 0.89-vs-0.78 result |
| 27 | Improving Predictions of Extreme Agricultural Losses with Extreme Value Theory | Real GEV/GPD crop-yield literature exists — none with the claimed P5 coverage 72%→89% improvement |
| 28 | Calibration of Probabilistic Forecasts of Financial Returns and Economic Variables | Real underlying work is Diebold et al. (1998) PIT histograms and Gneiting, Balabdaoui & Raftery (2007) — not this title |
| 29 | Compound drought-heatwave events in North America and implications for maize production | Real finding exists that compound dry-hot conditions coincide with US/France maize loss in ~61% of cases — but no APSIM paper with the claimed "2.3× drought alone, 1.8× heat alone" multipliers |
| 30 | Machine Learning for Agricultural Forecasting: A Review of Recent Advances | Real 2024 review exists ("Crop yield prediction in agriculture: A comprehensive review...") — not this 200-paper survey with the claimed "GNNs add 5–15%" figure |

---

## Why band B matters more than any code finding

**Papers 18 and 21 are not peripheral.** The methodology's opening sentence
(§1) states this work "addresses the critical failure mode documented in
Paper 18," and §1's "Relation to Paper 21" paragraph defines the paper's
entire novelty claim as an extension of Paper 21 along two named axes. If
neither paper exists, then the paper's stated **motivation** and its stated
**contribution relative to prior work** both rest on sources that cannot be
produced on request.

This is categorically more serious than the code bugs found earlier in this
audit. Those were about whether real experiments measured what they claimed.
This is about whether the premise is real.

Secondary consequence, stated plainly: **several "research gaps" in my own
earlier audit deliverables were themselves grounded in band-B citations.**
`LITERATURE_GAP_MAPPING.md` and `LITERATURE_GAP_COVERAGE_TABLE.md` treat
Papers 18, 19, 21, 24, 26, 27, 29 as real literature establishing real gaps.
Those gap rows must now be re-read with this document in hand — a gap
"identified by Paper 18" is not an established gap if Paper 18 cannot be
found. Both documents have been annotated accordingly.

## Calibration on this finding

Absence of a search result is not absolute proof of non-existence. A paper
could exist in a poorly-indexed venue, as a thesis chapter, or under a
substantially different title. Two things make that unlikely as a blanket
explanation here:

1. Each band-B entry was searched **twice** — by title, and independently by
   its own distinctive claimed findings/methods/datasets. Real papers
   surface by the second route even when retitled.
2. The failure is perfectly correlated with band. 15/15 fail inside band B;
   21/21 succeed outside it. Indexing gaps do not cluster that cleanly.

## What I will not do

I will not invent replacement citations, and I will not nominate real papers
as "close enough" substitutes for the specific claims currently attributed to
band-B entries. Either would repeat the exact failure this document exists to
catch.

## Recommended actions

1. **Do not submit or circulate any manuscript text that cites Papers 18 or
   21.** These are the two a reviewer is most likely to look up first.
2. If you have PDFs, reference-manager entries, or DOIs for any band-B row,
   send them — I will re-verify directly and correct this document.
3. For any that cannot be produced, choose per claim: (a) find a genuine
   citation that actually supports it, or (b) rewrite the claim to rest on
   something real. Note that several band-B claims **can** be re-grounded in
   band-A/C papers that genuinely say something similar — e.g. static
   conformal losing coverage under distribution shift is genuinely
   established by Gibbs & Candès (row 4, real) and Barber et al. (row 5,
   real); compound drought-heat yield damage is genuinely established by
   rows 31/32/33/38 (all real). The *phenomena* your paper is about are
   real and well-supported. It is specifically the agriculture-applied
   conformal-prediction citations that are not.
4. Re-examine how this literature review was produced. The band structure —
   real theory papers, real recent climate papers, fabricated bridge papers,
   no authors/venues/DOIs anywhere — is consistent with entries having been
   generated rather than collected.
