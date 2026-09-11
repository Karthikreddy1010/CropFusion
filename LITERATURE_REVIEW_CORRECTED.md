# Corrected Literature Review — Corn Yield under Compound Drought–Heat Stress with Conformal Uncertainty

**Replaces** `Literature_Review_Paper3_ConformalInference.xlsx`, which contained
15 unverifiable entries (see `LITERATURE_CITATION_VERIFICATION.md`).

**Every entry below was located via live search during this audit.** Each row
carries the metadata the original review lacked — **authors, venue, year, and
DOI/identifier** — precisely so that you, a co-author, or a reviewer can check
any of them in seconds. The `Verified` column states exactly what was
confirmed, and where I could only confirm part of the record I say so rather
than filling the gap.

**Two rules I held to while building this:**
1. Nothing here was invented. Where I could not find a real paper for a topic,
   the row says **"NO REPLACEMENT FOUND"** instead of a plausible-sounding
   substitute.
2. Numeric findings are reported only where the search result actually stated
   them. Where I did not see a number, the Findings cell describes the paper's
   contribution qualitatively rather than inventing a statistic.

---

## A. Conformal prediction & quantile regression foundations
*(all were already real in the original review — kept unchanged)*

| # | Authors | Year | Title | Venue / ID | Relevance to this work | Verified |
|---|---|---|---|---|---|---|
| A1 | Vovk, V., Gammerman, A., & Shafer, G. | 2005 (2nd ed. 2022) | Algorithmic Learning in a Random World | Springer | Origin of conformal prediction; validity under exchangeability | Title, authors, publisher ✓ |
| A2 | Angelopoulos, A.N., & Bates, S. | 2021 (arXiv) | A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification | arXiv:2107.07511 | Practitioner reference for split conformal & CQR | Title, authors, arXiv ID ✓ |
| A3 | Romano, Y., Patterson, E., & Candès, E.J. | 2019 | Conformalized Quantile Regression | NeurIPS 32 | **The CQR method this paper's quantile heads implement** | Title, authors ✓ |
| A4 | Gibbs, I., & Candès, E.J. | 2021 | Adaptive Conformal Inference Under Distribution Shift | NeurIPS 34 | **The ACI method this paper implements**; also the real evidence that static conformal degrades under shift | Title, authors ✓ |
| A5 | Barber, R.F., Candès, E.J., Ramdas, A., & Tibshirani, R.J. | 2023 | Conformal prediction beyond exchangeability | Annals of Statistics 51(2):816–845, doi:10.1214/23-AOS2276 | **Real grounding for the exchangeability-violation limitation** (§4.4) under temporal/spatial autocorrelation | Title, authors, journal, vol/pages, DOI ✓ |
| A6 | Koenker, R., & Bassett, G. | 1978 | Regression Quantiles | Econometrica 46(1):33–50 | Origin of quantile/pinball loss | Title, authors, journal, vol/pages ✓ |
| A7 | Gneiting, T., & Raftery, A.E. | 2007 | Strictly Proper Scoring Rules, Prediction, and Estimation | JASA 102(477):359–378 | Proper scoring rules; basis for Winkler score | Title, authors, journal, vol/pages ✓ |
| A8 | Shafer, G., & Vovk, V. | 2008 | A Tutorial on Conformal Prediction | JMLR 9:371–421 | Accessible conformal treatment | Title, authors, journal, vol/pages ✓ |
| A9 | Tibshirani, R.J., Barber, R.F., Candès, E.J., & Ramdas, A. | 2019 | Conformal Prediction Under Covariate Shift | NeurIPS 32 | **The weighted-conformal baseline implemented in `aci_calibrator.py`** | Title, authors ✓ |
| A10 | Lei, J., & Wasserman, L. | 2014 | Distribution-Free Prediction Bands for Non-Parametric Regression | JRSS-B 76(1):71–96 | Local/adaptive conformal bands | Title, authors ✓ |
| A11 | Meinshausen, N. | 2006 | Quantile Regression Forests | JMLR 7:983–999 | Tree-based quantile baseline (LightGBM/CatBoost/XGBoost quantile models) | Title, authors ✓ |
| A12 | Kivaranovic, D., Johnson, K.D., & Leeb, H. | 2019 (arXiv) / 2020 (AISTATS) | Adaptive, Distribution-Free Prediction Intervals for Deep Networks | AISTATS, PMLR 108:4346–4356; arXiv:1905.10634 | **The locally-adaptive conformal baseline implemented in `aci_calibrator.py`** | Title, authors, venue, pages, arXiv ID ✓ |
| A13 | Cesa-Bianchi, N., & Lugosi, G. | 2006 | Prediction, Learning, and Games | Cambridge University Press | Hedge / no-regret online learning underlying ACI's α-update. **Note:** the original review listed this as a topic ("Online Learning and Prediction with Expert Advice") rather than a citable work; this is the canonical reference | Standard canonical reference; **edition VERIFIED**: 1st edition, Cambridge University Press 2006, DOI 10.1017/cbo9780511546921 (monograph) |
| **A14 (NEW)** | Cauchois, M., Gupta, S., Ali, A., & Duchi, J.C. | 2024 | Robust Validation: Confident Predictions Even When Distributions Shift | JASA 119(548); arXiv:2008.04267 | **Replaces fabricated row 23.** Prediction sets valid for any test distribution in an *f*-divergence ball — exactly the method the fabricated entry described | Title, authors, journal, vol/issue, arXiv ID ✓ |
| **A15 (NEW)** | Gneiting, T., Balabdaoui, F., & Raftery, A.E. | 2007 | Probabilistic forecasts, calibration and sharpness | JRSS-B 69(2):243–268 | **Replaces fabricated row 28.** The real, canonical calibration-and-sharpness framework (PIT-based) | Title, authors, journal, vol/pages ✓ |

## B. Compound drought–heat extremes and crop impact
*(rows 11, 12, 31, 32, 33, 37, 38 were already real — kept; 13, 22, 24, 29 replaced)*

| # | Authors | Year | Title | Venue / ID | Relevance | Verified |
|---|---|---|---|---|---|---|
| B1 | Zscheischler, J., Martius, O., Westra, S., Bevacqua, E., Raymond, C., Horton, R.M., et al. | 2020 | A typology of compound weather and climate events | Nature Reviews Earth & Environment 1:333–347, doi:10.1038/s43017-020-0060-z | Definitional framework for "compound event" — grounds what CDHW *is* | Title, authors, journal, vol/pages, DOI ✓ |
| B2 | Fischer, E.M., & Knutti, R. | 2015 | Anthropogenic contribution to global occurrence of heavy-precipitation and high-temperature extremes | Nature Climate Change 5:560–564 | Attribution: ~75% of moderate hot extremes over land attributable to warming | Title, authors, journal, year ✓ |
| **B3 (NEW)** | Heino, M., Kinnunen, P., Anderson, W., Ray, D. K., Puma, M. J., Varis, O., Siebert, S., & Kummu, M. | 2023 | Increased probability of hot and dry weather extremes during the growing season threatens global crop yields | Scientific Reports 13, doi:10.1038/s41598-023-29378-2 | **Replaces fabricated row 29.** Provides the *real* compound-vs-individual evidence: for maize, compound drought+heatwave → **31% yield decrease**, vs heatwave alone **4%** and drought alone **7%** — genuine super-additivity, replacing the fabricated "2.3×/1.8×" multipliers | Title, journal, DOI ✓; ****authors VERIFIED** (Crossref DOI 10.1038/s41598-023-29378-2; *Sci. Rep.* **13**:3583) |
| **B4 (NEW)** | He, Y., Hu, X., Xu, W., Fang, J., & Shi, P. | 2022 | Increased probability and severity of compound dry and hot growing seasons over world's major croplands | Science of the Total Environment | **Replaces fabricated row 13.** Real global prevalence/trend evidence for compound dry-hot growing seasons | Title, journal ✓; ****authors VERIFIED** (Crossref DOI 10.1016/j.scitotenv.2022.153885; *Sci. Total Environ.* **824**:153885) |
| **B5 (NEW)** | Daryanto, S., Wang, L., & Jacinthe, P.-A. | 2016 | Global Synthesis of Drought Effects on Maize and Wheat Production | PLOS ONE 11(5):e0156362 | **Replaces fabricated row 24 — and is a better fit.** Real, maize-specific, stage-resolved: reproductive-stage drought **41.6–46.6%** yield reduction vs vegetative **18.6–26.2%**. **This is the citation that now justifies the phenology stage weighting in `config.py`** | Title, authors, journal, vol/article ✓ |
| B5b | Daryanto, S., Wang, L., & Jacinthe, P.-A. | 2017 | Global synthesis of drought effects on cereal, legume, tuber and root crops production: A review | Agricultural Water Management 179:18–33 | Companion synthesis; reproductive-phase sensitivity across crop types | Title, authors, journal ✓ |
| **B6 (NEW)** | Iizumi, T., et al. | 2014 | Impacts of El Niño Southern Oscillation on the global yields of major crops | Nature Communications 5:3712 | **Replaces fabricated row 22.** Real ENSO–yield teleconnection evidence, grounding the ENSO-stratified evaluation | Title, authors (lead), journal, vol ✓ |
| B7 | Yan, H., You, Y., Jiao, W., & Pan, N. | 2025 | Intensifying impacts of compound drought and heatwave events on water use efficiency in U.S. corn and soybean | Agricultural and Forest Meteorology | Compound events reduced WUE 14.7% (corn) / 11.3% (soybean); compound > single stressor | Title, journal, year ✓; **authors VERIFIED** (Crossref DOI 10.1016/j.agrformet.2025.110873; *Agric. For. Meteorol.* **375**:110873) |
| B8 | Sutanto, S. J., Zarzoza Mora, S. B., Supit, I., & Wang, M. | 2024 | Compound and cascading droughts and heatwaves decrease maize yields by nearly half in Sinaloa, Mexico | npj Natural Hazards, doi:10.1038/s44304-024-00026-7 | WOFOST-based: drought alone ~25% loss; compound/cascading amplify to ~44% | Title, journal, DOI ✓; **authors VERIFIED** (Crossref DOI 10.1038/s44304-024-00026-7) |
| B9 | Tripathy, K. P., Mukherjee, S., Mishra, A. K., Mann, M. E., & Williams, A. P. | 2023 | Climate change will accelerate the high-end risk of compound drought and heatwave events | PNAS, doi:10.1073/pnas.2219825120 | Projected nonlinear acceleration of CDHW risk | Title, journal, DOI ✓; **authors VERIFIED** (Crossref DOI 10.1073/pnas.2219825120; *PNAS* **120**(28)) |
| B10 | Li, Y., Zeleke, K., Wang, B., & Liu, D.-L. | 2025 | A Review of Data for Compound Drought and Heatwave Stress Impacts on Crops: Current Progress, Knowledge Gaps, and Future Pathways | Plants, doi:10.3390/plants14142158; PMC12298496 | **Data-gap review — a genuine source for the "what's missing" framing.** (Original review dated this 2024; it is 2025) | Title, authors, journal, DOI ✓ |
| B11 | Lesk, C., Anderson, W., Rigden, A., Coast, O., Jagermeyr, J., McDermid, S., Davis, K. F., & Konar, M. | 2022 | Compound heat and moisture extreme impacts on global crop yields under climate change | Nature Reviews Earth & Environment 3(12):872–889, doi:10.1038/s43017-022-00368-8 | Compound extremes → up to 30% yield losses in breadbaskets; crop-physiological compounding mechanisms. (Original review dated this 2021; it is 2022) | Title, journal, vol/pages, DOI ✓; **authors VERIFIED** (Crossref DOI 10.1038/s43017-022-00368-8; *Nat. Rev. Earth Environ.* **3**(12):872-889) |

## C. Crop-yield prediction with uncertainty quantification
*(this is the band where **all** original entries were unverifiable — every row here is a real replacement)*

| # | Authors | Year | Title | Venue / ID | Relevance | Verified |
|---|---|---|---|---|---|---|
| **C1 (NEW)** | Ma, Y., Zhang, Z., Kang, Y., & Özdoğan, M. | 2021 | Corn yield prediction and uncertainty analysis based on remotely sensed variables using a Bayesian neural network approach | Remote Sensing of Environment 259:112408 | **Replaces fabricated rows 14 & 15 — and is a strictly better fit (corn, US, uncertainty-aware).** BNN for corn yield + predictive uncertainty; uncertainty rises under environmental stress — directly comparable framing to this paper | Title, authors, journal, vol/article ✓ |
| **C2 (NEW)** | Zhang, C., & Diao, C. | 2023 | A Phenology-guided Bayesian-CNN (PB-CNN) framework for soybean yield estimation and uncertainty analysis | ISPRS J. Photogrammetry & Remote Sensing 205:50–73 | **Replaces fabricated row 21 — the closest real work to this paper's phenology+UQ premise.** County-level US Corn Belt, phenology-stage-structured, aleatoric + epistemic uncertainty. **Crucially: it is Bayesian, NOT conformal** — see the novelty note below | Title, journal, vol/pages, code repo (github.com/rssiuiuc/PBCNN) ✓; **authors CORRECTED and VERIFIED** (Crossref DOI 10.1016/j.isprsjprs.2023.09.025 -- the previous entry "Lin, Y., et al. (rssiuiuc group)" was WRONG; the paper is by Chishan Zhang & Chunyuan Diao) |
| **C3 (NEW)** | Farag, M., Emam, A., Leonhardt, J., & Roscher, R. | 2025 | Enhancing decision support in crop production: Analyzing conformal prediction for uncertainty quantification | Computers and Electronics in Agriculture, ScienceDirect S0168169925006659 | **Replaces fabricated row 18 — the load-bearing motivating citation.** Real, agriculture-specific conformal-prediction evaluation that explicitly identifies the open problem: CP's "performance under distributional shifts and out-of-distribution detection has not been sufficiently explored" in agriculture. *(This paper was already row 36 of the original review, verified real)* | Title, journal, article ID ✓; **authors VERIFIED** (Crossref DOI 10.1016/j.compag.2025.110559; *Comput. Electron. Agric.* **237**:110559) |
| **C4 (NEW)** | Jabed, M. A., & Azmi Murad, M. A. | 2024 | Crop yield prediction in agriculture: A comprehensive review of machine learning and deep learning approaches, with insights for future research and sustainability | Heliyon / ScienceDirect S2405844024168673; PMC11667600 | **Replaces fabricated row 30.** Real ML/DL-for-yield survey (97 papers, 2017–2024) | Title, journal, PMC ID ✓; **authors VERIFIED** (PMC11667600; DOI 10.1016/j.heliyon.2024.e40836; *Heliyon* **10**(24):e40836) |
| **C5 (NEW)** | Ren, Y., Li, Q., Du, X., Zhang, Y., Wang, H., Shi, G., & Wei, M. | 2023 | Analysis of Corn Yield Prediction Potential at Various Growth Phases Using a Process-Based Model and Deep Learning | PMC9920366 | Real corn, growth-phase-resolved prediction: predictors differ by stage (growth-state features dominate vegetative; water features dominate reproductive) — supports stage-aware feature design | Title, PMC ID ✓; **authors VERIFIED** (PMC9920366; DOI 10.3390/plants12030446; *Plants* **12**(3):446)/journal |
| **C6 (NEW)** | Onogi, A. | 2025 | Estimating the changing risks of low crop yield using non-stationary generalized Pareto distributions | bioRxiv 2025.05.10.653234 | **Partial replacement for fabricated row 27 (EVT tail risk).** ⚠️ **PREPRINT — not peer reviewed.** Cite with that caveat or find a published equivalent | Title, bioRxiv ID ✓; **preprint status flagged** | **author VERIFIED** (bioRxiv API, DOI 10.1101/2025.05.10.653234, posted 2025-05-11; single author)

## D. Topics where NO adequate real replacement was found

**I did not invent substitutes for these.** Each represents either a genuine
gap in the literature (which *helps* your novelty claim) or a topic where you
should search further with database access I don't have.

| Original (fabricated) row | Topic | Status | What to do |
|---|---|---|---|
| 19 | Time-series conformal prediction with temporal autocorrelation correction | **Partly covered** by A5 (Barber et al. 2023, real) which is the canonical treatment of conformal beyond exchangeability. Real newer work exists (online/multi-step conformal, e.g. arXiv:2410.13115) but I did not verify it closely enough to list | Cite A5 as primary; optionally add a verified time-series-conformal paper |
| 20 | Agricultural risk assessment / crop insurance probabilistic framework | **NO REPLACEMENT FOUND** that matches the described QRF+BNN+CQR insurance framework | Either drop this citation and the claim it supports, or search AgEcon/actuarial literature directly |
| 26 | Spatially-aware conformal prediction for geographic shift | **Real work exists** (GeoConformal Prediction; "Spatial Conformal Inference through Localized Quantile Regression", OpenReview) but I could not pin down full publication metadata | Worth pursuing — this is directly relevant to your LOSO design. Verify one and add it |
| 27 | EVT for extreme agricultural losses (peer-reviewed) | Only a **preprint** (C6) found | Find a published GEV/GPD crop-yield paper, or scope the claim to acknowledge it's not implemented here anyway (this pipeline does no EVT — see `LITERATURE_GAP_COVERAGE_TABLE.md` gap #8) |

---

## What the corrected literature changes about your contribution

This is the important part, and it is **not** bad news.

**1. Your phenology-conformal method looks more novel, not less.**
The fabricated row 21 claimed phenology-conditioned CQR already existed, so
the methodology positioned this work as merely *extending* it. In the real
literature, phenology-guided uncertainty work exists but is **Bayesian**
(C2, PB-CNN) — I found no phenology-stratified **conformal** calibration.
`phenology_stratified_cqr()` in `aci_calibrator.py` should therefore be
presented as a contribution, not a reproduced baseline. **Claim it as "to our
knowledge" — my searches are not exhaustive.** The code has been relabelled
accordingly.

**2. Your motivating gap survives, better grounded.**
The fabricated row 18 asserted that static conformal's agricultural failure
was already *documented* (91%→73%). The real position (C3, 2025) is that this
is an **open, insufficiently-explored question** in agriculture. That is a
stronger motivation for your study, not a weaker one — you are answering an
open question rather than re-confirming a known result. Your real measured
finding (ACI PICP 0.933 normal → 0.826/0.820 moderate/extreme) becomes a
genuine contribution to that open question.

**3. Compound > individual effects are real and quantified — better numbers than the fabricated ones.**
B3 gives maize-specific **31% (compound) vs 4% (heat alone) vs 7% (drought
alone)**. That is real, published super-additivity evidence, and it is
stronger than the fabricated "2.3×/1.8×". It motivates CDHW as a *construct*
— while your own ablation honestly shows it doesn't improve this backbone's
point accuracy. Those two facts coexist and should both be reported.

**4. The stage-weighting in your code now has a real source.**
B5 (Daryanto et al. 2016) supplies real maize numbers (reproductive 41.6–46.6%
vs vegetative 18.6–26.2%), which revealed that the previous weights had
grain-fill ranked *below* vegetative — backwards relative to the literature.
`config.py` has been corrected (see the code-change list).

---

## Author-verification record (2026-09-07)

Every previously-outstanding author list has been resolved against a
machine-readable bibliographic source. **No author name below was inferred,
reconstructed from memory, or copied from a secondary description** — each was
read from Crossref, PubMed Central, or the bioRxiv API, by DOI where one
existed and by exact-title match otherwise.

| Row | Verified authors | Source of truth |
|---|---|---|
| A13 | Cesa-Bianchi, N., & Lugosi, G. | Crossref `10.1017/cbo9780511546921` — 1st ed., CUP 2006, monograph |
| B3 | Heino, Kinnunen, Anderson, Ray, Puma, Varis, Siebert, Kummu | Crossref `10.1038/s41598-023-29378-2` — *Sci. Rep.* 13:3583 |
| B4 | He, Hu, Xu, Fang, Shi | Crossref `10.1016/j.scitotenv.2022.153885` — *STOTEN* 824:153885 |
| B7 | Yan, You, Jiao, Pan | Crossref `10.1016/j.agrformet.2025.110873` — *Agric. For. Meteorol.* 375:110873 |
| B8 | Sutanto, Zarzoza Mora, Supit, Wang | Crossref `10.1038/s44304-024-00026-7` |
| B9 | Tripathy, Mukherjee, Mishra, Mann, Williams | Crossref `10.1073/pnas.2219825120` — *PNAS* 120(28) |
| B11 | Lesk, Anderson, Rigden, Coast, Jägermeyr, McDermid, Davis, Konar | Crossref `10.1038/s43017-022-00368-8` — *Nat. Rev. Earth Environ.* 3(12):872–889 |
| **C2** | **Zhang, C., & Diao, C.** | Crossref `10.1016/j.isprsjprs.2023.09.025` — *ISPRS J.* 205:50–73 |
| C3 | Farag, Emam, Leonhardt, Roscher | Crossref `10.1016/j.compag.2025.110559` — *Comput. Electron. Agric.* 237:110559 |
| C4 | Jabed, M. A., & Azmi Murad, M. A. | PMC11667600 / `10.1016/j.heliyon.2024.e40836` — *Heliyon* 10(24):e40836 |
| C5 | Ren, Li, Du, Zhang, Wang, Shi, Wei | PMC9920366 / `10.3390/plants12030446` — *Plants* 12(3):446 |
| C6 | Onogi, A. (single author) | bioRxiv API `10.1101/2025.05.10.653234`, posted 2025-05-11 — **preprint, not peer reviewed** |

### One correction, not just a fill-in

**C2's author list in the previous version of this document was wrong.** It
read *"Lin, Y., et al. (rssiuiuc group, UIUC)"*. The paper is by **Chishan
Zhang and Chunyuan Diao**. The title, journal, volume and page range were all
correct — only the authors were wrong, which is the hardest kind of citation
error for a reader to notice and the easiest for a reviewer to catch.

C2 matters more than most rows: it is the closest real work to this paper's
phenology + uncertainty premise, and the novelty claim ("no phenology-stratified
*conformal* method located") is stated relative to it. Citing it under the wrong
authors would have undercut exactly the comparison it is there to support.

### Status

All 12 previously-outstanding rows are now resolved. **No row in this document
is still marked "confirm authors".**

Two caveats that remain, and are not author problems:

1. **C6 is a preprint** (bioRxiv, not peer reviewed). It is the only support for
   the EVT / tail-risk topic. Either cite it explicitly as a preprint or drop
   the claim — do not present it as peer-reviewed literature.
2. Verification here covers **authorship, venue and identifiers**. It does not
   re-assess whether each paper actually supports the claim it is cited for;
   that judgement is recorded in the "Relevance" column and should be re-read
   independently.
