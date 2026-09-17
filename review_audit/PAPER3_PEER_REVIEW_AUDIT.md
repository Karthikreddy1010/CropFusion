# Paper 3 — Peer-Review-Level Audit
### Literature · Gaps · Novelty · Methodology · Validation · Publishability

**Date:** 2026-09-16 · **Reviewer stance:** skeptical, evidence-first · **Scope:** the Paper 3 project as it exists on disk today

---

## 0. How to read this audit

### 0.1 What was examined

| Input | File(s) | Status |
|---|---|---|
| Original proposal | `Paper3_ConformalInference.docx` | read in full |
| Current methodology | `Paper3_Methodology_Updated_v2.md` (layered revisions v1 to v4) | read in full |
| Literature study | `LITERATURE_REVIEW_CORRECTED.md`, `CDHW_LITERATURE_REVIEW.md`, `ADDITIONAL_LITERATURE_SEARCH.md`, `LITERATURE_GAP_ANALYSIS.md`, `METHODOLOGY_GAP_COVERAGE.md`, `LITERATURE_CITATION_VERIFICATION.md`, both `.xlsx` reviews | read |
| Code | `code/*.py` (main pipeline, splitting, preprocessing, feature selection, models, calibration, audits) | inspected at source level |
| Data | `master_dataset/` (FULL 22,737 x 220; MODEL-READY 20,507 x 95), catalog, QC, merge audit, final data report | inspected |
| Results | `outputs_master/` — the Colab run of 2026-09-16 with the LOSO detrend fix, 155 files | inspected; key numbers recomputed |
| Diagnostics | `outputs_diagnostics/LOSO_FORENSIC_DIAGNOSTIC.md`, `code/diagnostics_loso/p8_review_evidence.py` | produced for this audit |

### 0.2 Evidence labels used throughout

- **[M]** measured by this audit on the current data or current run (script and output named)
- **[L]** supported by a verified publication (DOI or registry record checked against Crossref, OpenAlex, Semantic Scholar or arXiv on 2026-09-16)
- **[I]** reviewer interpretation or judgement, which is stated as such

No citation in this audit was taken from memory. Each new reference was resolved against a bibliographic registry, and the full verification log is in `LITERATURE_REVIEW_v3.md` §7. Where a claim could only be checked against an abstract, this is said.

### 0.3 Headline verdict

**The project is not ready for submission. It is publishable after major revision as an evaluation study of conformal calibration under climate and cross-state distribution shift, but not as a methods or architecture paper.**

The strongest asset is an empirical result that this audit re-derived on the new PRISM/NLDAS data: **conformal methods ranked best on marginal calibration rank worst on coverage in the weakest climate-exposure stratum**. In cross-state transfer, **coverage failures track prediction bias, not interval width**. **[M]**

The blockers are fixable, but they are real:

1. **There is no single coherent manuscript.** The proposal, the v4 methodology and the current code describe three different studies. v4's central numbers come from superseded ERA5 runs, and nine files cited as evidence do not exist in the project.
2. **The central claim rests on one five-year test window.** Its "Extreme" stratum is 87 % from 2022–2023 and 73 % from Missouri and Iowa, and the significance tests treat 2,345 dependent county-years as independent.
3. **Essential baselines are missing.** These are a naive trend baseline, a fixed-effects degree-day panel, a group-conditional conformal method, and a modern online conformal method. A 2025 *JRSS-B* paper gives finite-sample group-conditional guarantees, which directly bounds the novelty claim as it is currently written.
4. **The literature omits the US maize yield–weather canon, the US Corn Belt ML literature and 2023–2026 conformal work.** It also mischaracterises two of its own anchor citations.
5. **Model and design choices were refined across repeated full-pipeline runs that exposed held-out results.** This needs disclosure and a frozen final protocol.

---

## 1. Literature audit

### 1.1 Coverage (Phase 1-A)

| # | Research area the paper depends on | In current review? | Verdict | Consequence if left as is |
|---|---|---|---|---|
| 1 | Conformal foundations: split CP, CQR, weighted CP, locally adaptive CP, ACI | A1–A15 | **Sufficient** | — |
| 2 | Marginal vs conditional coverage theory | N1 (Vovk 2013), N2 (Barber et al. 2021) | **Incomplete** — has the impossibility result, lacks the positive results | Reviewers from statistics will cite group-conditional and multivalid guarantees against the "evaluation is the only possible contribution" framing |
| 3 | Online and time-series conformal beyond ACI | N3 (EnbPI), N4 (AgACI) | **Thin** | Missing DtACI, conformal PID, SAOCP and SPCI, the current state of the art for the ACI comparator |
| 4 | Spatial conformal prediction | flagged "unresolved" (row D-26) | **Absent** — but now verifiable | Mao et al. 2024 (*JASA*) and Lou et al. 2025 (GeoCP) resolve the open row |
| 5 | Conformal prediction in environmental and Earth-observation science | none | **Absent** | No evidence the field already uses CP outside agriculture |
| 6 | UQ for crop yield (Bayesian, ensembles, CP) | C1, C2, C3, N6 | **Thin and partly mischaracterised** (§1.4) | Novelty is stated against too small a neighbourhood |
| 7 | Compound drought–heat climatology (global) | B1–B11, L1–L11 | **Sufficient** | — |
| 8 | **US maize yield–weather statistics and compound extremes in the US** | none | **Absent — major** | The "compound extremes are unmodelled" gap is contradicted by work the review does not cite |
| 9 | **Machine-learning yield prediction in the US Corn Belt** | effectively none (two generic reviews) | **Absent — major** | No comparable skill benchmarks; architecture choices unjustified against the literature |
| 10 | Spatial transfer and domain adaptation for yield models | none | **Absent** | LOSO results cannot be positioned |
| 11 | Validation design for structured data; leakage in ML-based science | none | **Absent** | The strongest methodological asset (leakage control, LOSO) has no scholarly anchor |
| 12 | Detrending of yield series | none | **Absent** | Detrending decides LOSO skill (0.357 to 0.698) and needs justification |
| 13 | Inference with dependent data and few clusters; forecast comparison | none | **Absent** | Current significance tests are not defensible without it |
| 14 | Primary references for data products (PRISM, NLDAS-2, SPEI, USDM) | none | **Absent** | Standard reviewer request |
| 15 | Explainability (SHAP) | none | **Absent** | SHAP results are reported but uncited |
| 16 | Causal inference | not applicable | **Not needed** — but O1's "super-linear damage" is a causal-sounding claim that belongs to the econometric interaction literature (area 8) | Either drop the claim or cite and meet that literature's standard |
| 17 | Fairness and equity | not applicable | **Not needed as such**; the relevant concept is equalised coverage across groups (area 2) | — |

### 1.2 Quality of the existing groups (Phase 1-B, group level)

Paper-level detail is in the comparison matrix, §3.

| Group | Problem solved | Data / scale | Method | Validation / metrics | What it establishes | Limitation for this paper |
|---|---|---|---|---|---|---|
| A — conformal theory | Distribution-free predictive inference | Theory; generic benchmarks | Split CP, CQR, ACI, weighted and local CP | Finite-sample marginal coverage theorems | The calibration toolbox and its assumptions | Silent on crop yield; marginal by construction |
| B — compound drought–heat | Frequency, trends and yield impact of hot-dry extremes | Global or regional climate and yield statistics | Indices, copulas, statistical attribution, crop models | Descriptive and attribution statistics | Hot-dry co-occurrence is increasing and damaging to maize | Almost none is US-county-specific; none is about prediction intervals |
| C — crop yield UQ | Uncertainty in yield estimates | US county (C1, C2), agricultural images (C3), global country-level (N6) | Bayesian NN, Bayesian CNN, inductive CP, CQR + GNN | Marginal coverage or predictive variance | UQ is emerging in yield work | Marginal only; no method comparison; no regime or state conditioning |
| N — added theory | Conditional validity, dependence, Mondrian CP | Theory | Mondrian CP, EnbPI, AgACI | Marginal coverage under mixing | Limits and partial remedies | Lacks the 2023–2025 positive conditional results |

### 1.3 Recency (Phase 1-C)

The review is **not** mainly old. Its weakness is **selective recency**: it cites 2022–2025 climate-impact papers but misses 2023–2026 conformal methods and the 2009–2021 domain canon.

- **Foundational — keep:** Vovk et al. 2005; Shafer & Vovk 2008; Koenker & Bassett 1978; Gneiting & Raftery 2007; Romano et al. 2019; Tibshirani et al. 2019; Gibbs & Candès 2021; Vovk 2013; Barber et al. 2021 and 2023; Zscheischler et al. 2020. Add Schlenker & Roberts 2009, Lei et al. 2018 and Vicente-Serrano et al. 2010.
- **No longer sufficient on their own:** Lei & Wasserman 2014, since Lei et al. 2018 (*JASA*) is the standard split-CP regression reference. Also Cesa-Bianchi & Lugosi 2006 as the only online-learning anchor, which DtACI (Gibbs & Candès 2024) supersedes for this use.
- **Weak, low-quality or off-target — remove or replace:**
  - Fischer & Knutti 2015 is about attribution of heavy-precipitation and hot extremes, not compound crop stress; replace with Mazdiyasni & AghaKouchak 2015 for US concurrent droughts and heatwaves.
  - Poudel & Poudel 2023 (L10) appeared in a technical magazine, not a peer-reviewed journal.
  - Onogi 2025 (C6) is a bioRxiv preprint on EVT, which the project does not implement.
  - Jabed & Azmi Murad 2024 is a generic review; van Klompenburg et al. 2020 and CY-Bench 2026 serve better.
- **Emerging 2024–2026 work that affects the novelty claim:**
  - Gibbs, Cherian & Candès 2025 (*JRSS-B*): finite-sample coverage over pre-specified subgroups and covariate-shift classes.
  - Gibbs & Candès 2024 (*JMLR*): DtACI.
  - Mao, Martin & Reich 2024 (*JASA*) and Lou, Luo & Meng 2025 (*Ann. AAG*): spatial CP.
  - Kallenberg et al. 2026 (*ESSD*): CY-Bench.
  - Mahmood et al. 2026 (*Information*): CQR for crop yield.
  - Jia et al. 2026 (*WRR*): weighted CP under shift in hydrology.
  - Namdeo & Baraskar 2026 (Zenodo, **not peer reviewed**): split CP + ACI for crop yield under inter-annual climate variability. It overlaps the original framing and must be disclosed as grey literature.

### 1.4 Errors found in the existing literature documents

These are verified errors, not matters of taste.

| ID | Where | Error | Correct record / action |
|---|---|---|---|
| E1 | `Paper3_Methodology_Updated_v2.md` §1 (twice) and the status box | PB-CNN is attributed to "Lin et al. (2023)" | **Zhang, C. & Diao, C. (2023)**, *ISPRS J.* 205:50–73, doi:10.1016/j.isprsjprs.2023.09.025. The corrected review fixed this; the methodology file did not. |
| E2 | `ADDITIONAL_LITERATURE_SEARCH.md` N4 | Zaffran et al. is labelled a preprint with "no published version" | Published at **ICML 2022** (PMLR 162:25834–25866; arXiv:2202.07282) |
| E3 | N2 | Barber et al. dated 2020 | **2021**, *Information and Inference* 10(2):455–482 |
| E4 | N5 | Toccaceli & Gammerman dated 2018 | **2019**, *Machine Learning* 108(3):489–510 |
| E5 | C3, and methodology §1 and §2 | Farag et al. 2025 is described as a "review" stating that CP under shift is "not sufficiently explored" | The abstract shows an **empirical evaluation of inductive CP for image models (ResNet-18, ViT-B/16) on agricultural tasks, including covariate-shift and OOD experiments**. It is not a yield-regression study. The quoted sentence could not be checked against the full text; verify it or do not quote it. |
| E6 | N6 | Mahmood et al. 2026 is called "the state of the art in this exact intersection" | Country-level, 15 countries and 6 crops, in *Information* (MDPI). Its R² 0.9594 is not comparable to county-year corn. Describe it as "a recent CQR application to crop yield that reports marginal coverage only". |
| E7 | Methodology §4.3b, §5.9, §0 table | Headline numbers (ACI ACE 0.0096, worst PICP 0.7602, V6 severe PICP 0.8478, LOSO mean 0.626, backbone R² 0.44–0.52) | They come from legacy ERA5 runs, some explicitly flagged as computed on a corrupted county-name join. **None may be cited**; current values are in §15. |
| E8 | Methodology (throughout) | Evidence is cited from `AUDIT_REPORT.md`, `CORRECTION_CHANGELOG.md`, `CDHW_DESIGN_REPORT.md`, `CLAIM_AUDIT.md`, `FINAL_METHODOLOGY_AUDIT.md`, `RESIDUAL_AUDIT_ROUND6.md`, `JOIN_BUG_FIX_2026-09-10.md`, `CDHW_FINAL_AUDIT.md`, `calibration_independence_report.json` | **None of these files exists in the project [M].** `conditional_coverage_report.py`, Mondrian, EnbPI and "V6 regime-adaptive alpha" are also **absent from the current code [M]**. |
| E9 | A2 | arXiv version cited | Cite the archival version: *Foundations and Trends in ML* 16(4):494–591 (2023), doi:10.1561/2200000101 |
| E10 | §6 datasets table of the methodology | ERA5, MODIS NDVI, DSSAT soft targets, FAO Ky, soybean yield are listed as required; SSURGO, DEM and land cover as not required | This is the inverse of the implemented study: PRISM + NLDAS-2 are used, soil, DEM and NLCD are in the model, and NDVI and DSSAT are absent |

---

## 2. Missing literature list (verified)

Priority: **P1** = a reviewer will ask for it; **P2** = strengthens positioning; **P3** = useful context. Every entry below was resolved against a registry on 2026-09-16 (log in `LITERATURE_REVIEW_v3.md` §7).

### 2.1 US maize yield–weather statistics and compound extremes (area 8)

| P | Reference | Why it is needed |
|---|---|---|
| P1 | Schlenker, W. & Roberts, M.J. (2009). Nonlinear temperature effects indicate severe damages to U.S. crop yields under climate change. *PNAS* 106(37):15594–15598. doi:10.1073/pnas.0906865106 | Canonical county-panel evidence: corn yields rise to 29 °C, then decline steeply. It justifies KDD and Tmax thresholds and supplies the standard statistical baseline. |
| P1 | Lobell, D.B. et al. (2013). The critical role of extreme heat for maize production in the United States. *Nature Climate Change* 3(5):497–501. doi:10.1038/nclimate1832 | Extreme-heat mechanism for US maize |
| P1 | Lobell, D.B. et al. (2014). Greater sensitivity to drought accompanies maize yield increase in the U.S. Midwest. *Science* 344(6183):516–519. doi:10.1126/science.1251423 | Drought (VPD) sensitivity *increased* 1995–2012. Non-stationary yield–weather response bears on trend handling and temporal shift. |
| P1 | Haqiqi, I., Grogan, D.S., Hertel, T.W. & Schlenker, W. (2021). Quantifying the impacts of compound extremes on agriculture. *HESS* 25(2):551–564. doi:10.5194/hess-25-551-2021 | **US corn, 1981–2015: compound metrics predict yield better than individual extremes; heat damage up to 4x worse with water stress.** It directly contradicts "compound extremes unmodelled". |
| P1 | Hamed, R., Van Loon, A.F., Aerts, J. & Coumou, D. (2021). Impacts of compound hot–dry extremes on US soybean yields. *Earth System Dynamics* 12(4):1371–1391. doi:10.5194/esd-12-1371-2021 | US interaction-effect modelling with out-of-sample evaluation |
| P2 | Ortiz-Bobea, A., Wang, H., Carrillo, C.M. & Ault, T.R. (2019). Unpacking the climatic drivers of US agricultural yields. *ERL* 14(6):064003. doi:10.1088/1748-9326/ab1e75 | Water stress vs heat stress attribution with land-surface-model soil moisture |
| P2 | Lobell, D.B. & Burke, M.B. (2010). On the use of statistical models to predict crop yield responses to climate change. *Agric. For. Meteorol.* 150(11):1443–1452. doi:10.1016/j.agrformet.2010.07.008 | Methodological basis for statistical yield models |
| P2 | Troy, T.J., Kipgen, C. & Pal, I. (2015). The impact of climate extremes and irrigation on US crop yields. *ERL* 10(5):054013. doi:10.1088/1748-9326/10/5/054013 | Irrigation decouples yields from climate. It justifies excluding Nebraska and exposes uncontrolled irrigation within the six states. |
| P2 | Butler, E.E. & Huybers, P. (2013). Adaptation of US maize to temperature variations. *Nature Climate Change* 3(1):68–72. doi:10.1038/nclimate1585 | Spatial differences in heat sensitivity bear on the LOSO transfer assumption |
| P2 | Butler, E.E. & Huybers, P. (2015). Variations in the sensitivity of US maize yield to extreme temperatures by region and growth phase. *ERL* 10(3):034009. doi:10.1088/1748-9326/10/3/034009 | Region- and stage-specific sensitivity; supports stage windows and questions uniform transfer |
| P2 | Mazdiyasni, O. & AghaKouchak, A. (2015). Substantial increase in concurrent droughts and heatwaves in the United States. *PNAS* 112(37):11484–11489. doi:10.1073/pnas.1422945112 | US-specific compound-event trend (replaces Fischer & Knutti) |
| P3 | Zscheischler, J. & Seneviratne, S.I. (2017). Dependence of drivers affects risks associated with compound events. *Science Advances* 3(6):e1700263. doi:10.1126/sciadv.1700263 | Why joint (not univariate) risk matters |
| P3 | Ribeiro, A.F.S. et al. (2020). Risk of crop failure due to compound dry and hot extremes estimated with nested copulas. *Biogeosciences* 17(19):4815–4830. doi:10.5194/bg-17-4815-2020 | Probabilistic compound-risk precedent |
| P3 | Lesk, C., Rowhani, P. & Ramankutty, N. (2016). Influence of extreme weather disasters on global crop production. *Nature* 529(7584):84–87. doi:10.1038/nature16467 | Drought and extreme heat effects on production |
| P3 | Vogel, E. et al. (2019). The effects of climate extremes on global agricultural yields. *ERL* 14(5):054010. doi:10.1088/1748-9326/ab154b | ML attribution of extremes to yield anomalies |
| P3 | Anderson, W.B. et al. (2019). Synchronous crop failures and climate-forced production variability. *Science Advances* 5(7):eaaw1976. doi:10.1126/sciadv.aaw1976 | ENSO-forced synchronous failures (ENSO stratification) |
| P3 | Xiao, K. et al. (2025). Drought and extreme heat reduce wheat and maize production in the United States by lowering both crop yields and harvestable fraction. *Earth's Future* 13(12):e2024EF005557. doi:10.1029/2024ef005557 | Recent US county evidence (KDD, drought days) |

### 2.2 Machine-learning yield prediction, US Corn Belt (area 9)

| P | Reference | Why it is needed |
|---|---|---|
| P1 | Jiang, H. et al. (2020). A deep learning approach to conflating heterogeneous geospatial data for corn yield estimation: a case study of the US Corn Belt at the county level. *Global Change Biology* 26(3):1754–1766. doi:10.1111/gcb.14885 | County corn, end-of-season; LSTM explains 76 % of variation; skill in the 2012 extreme year. The closest point-prediction comparator. |
| P1 | Kang, Y. et al. (2020). Comparative assessment of environmental variables and machine learning algorithms for maize yield prediction in the US Midwest. *ERL* 15(6):064005. doi:10.1088/1748-9326/ab7df9 | **XGBoost best; LSTM and CNN not advantageous.** Literature support for not adding architecture. |
| P1 | Shahhosseini, M., Hu, G. & Archontoulis, S.V. (2020). Forecasting corn yield with machine learning ensembles. *Frontiers in Plant Science* 11:1120. doi:10.3389/fpls.2020.01120 | Ensembles, blocked sequential CV; IL, IN, IA overlap your states |
| P1 | Khaki, S., Wang, L. & Archontoulis, S.V. (2020). A CNN-RNN framework for crop yield prediction. *Frontiers in Plant Science* 10:1750. doi:10.3389/fpls.2019.01750 | 13 Corn Belt states, temporal hold-out 2016–2018, point only |
| P1 | Crane-Droesch, A. (2018). Machine learning methods for crop yield prediction and climate change impact assessment in agriculture. *ERL* 13(11):114003. doi:10.1088/1748-9326/aae159 | Semiparametric NN that keeps parametric trend and fixed-effect structure; withheld-year evaluation |
| P2 | Shahhosseini, M. et al. (2021). Coupling machine learning and crop modeling improves crop yield prediction in the US Corn Belt. *Scientific Reports* 11:1606. doi:10.1038/s41598-020-80820-1 | Hydrological inputs (soil moisture, water table) most influential; bears on missing soil-moisture features |
| P2 | Leng, G. & Hall, J.W. (2020). Predicting spatial and temporal variability in crop yields: an inter-comparison of machine learning, regression and process-based models. *ERL* 15(4):044027. doi:10.1088/1748-9326/ab7b24 | ML vs regression vs process models for averages, variability and extremes |
| P2 | Lischeid, G. et al. (2022). Machine learning in crop yield modelling: a powerful tool, but no surrogate for science. *Agric. For. Meteorol.* 312:108698. doi:10.1016/j.agrformet.2021.108698 | Smooth non-linear detrending; **equally good models with different predictor sets**, a caution for SHAP-based claims |
| P2 | Kallenberg, M., Paudel, D. et al. (2026). CY-Bench: a comprehensive benchmark dataset for sub-national crop yield forecasting. *ESSD* 18(6):3997–4018. doi:10.5194/essd-18-3997-2026 | Current benchmark standard for yield protocols |
| P2 | Paudel, D. et al. (2021). Machine learning for large-scale crop yield forecasting. *Agricultural Systems* 187:103016. doi:10.1016/j.agsy.2020.103016 | Leakage-free ML baseline workflow for yield |
| P2 | van Klompenburg, T., Kassahun, A. & Catal, C. (2020). Crop yield prediction using machine learning: a systematic literature review. *Comput. Electron. Agric.* 177:105709. doi:10.1016/j.compag.2020.105709 | Stronger review than Jabed 2024 |
| P3 | You, J. et al. (2017). Deep Gaussian process for crop yield prediction based on remote sensing data. *Proc. AAAI* 31(1). doi:10.1609/aaai.v31i1.11172 | Early probabilistic deep yield model |
| P3 | Khaki, S. & Wang, L. (2019). Crop yield prediction using deep neural networks. *Frontiers in Plant Science* 10:621. doi:10.3389/fpls.2019.00621 | Deep-learning baseline lineage |
| P3 | Ansarifar, J., Wang, L. & Archontoulis, S.V. (2021). An interaction regression model for crop yield prediction. *Scientific Reports* 11:17754. doi:10.1038/s41598-021-97221-7 | Interpretable interaction model in IL, IN, IA |
| P3 | Storm, H., Baylis, K. & Heckelei, T. (2020). Machine learning in agricultural and applied economics. *Eur. Rev. Agric. Econ.* 47(3):849–892. doi:10.1093/erae/jbz033 | ML vs econometrics positioning |

### 2.3 Spatial transfer and validation design (areas 10, 11, 12)

| P | Reference | Why it is needed |
|---|---|---|
| P1 | Roberts, D.R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40(8):913–929. doi:10.1111/ecog.02881 | Scholarly basis for the temporal and LOSO blocking |
| P1 | Ploton, P. et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications* 11:4540. doi:10.1038/s41467-020-18321-y | Random CV inflates skill (compare random split R² 0.938 with temporal 0.703) |
| P1 | Kapoor, S. & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns* 4(9):100804. doi:10.1016/j.patter.2023.100804 | Leakage taxonomy that the provenance ledger answers |
| P1 | Lu, J., Carbone, G.J. & Gao, P. (2017). Detrending crop yield data for spatial visualization of drought impacts in the United States, 1895–2014. *Agric. For. Meteorol.* 237–238:196–208. doi:10.1016/j.agrformet.2017.02.001 | Detrending choices for US yield series |
| P2 | Meyer, H. & Pebesma, E. (2021). Predicting into unknown space? Estimating the area of applicability of spatial prediction models. *Methods Ecol. Evol.* 12(9):1620–1633. doi:10.1111/2041-210x.13650 | Transfer to a new domain (held-out state) |
| P2 | Meyer, H., Reudenbach, C., Wöllauer, S. & Nauss, T. (2019). Importance of spatial predictor variable selection in machine learning applications — moving from data reproduction to spatial prediction. *Ecol. Model.* 411:108815. doi:10.1016/j.ecolmodel.2019.108815 | Location-identifying predictors hurt spatial transfer, as with `county_baseline` and the lags |
| P2 | Wadoux, A.M.J.-C., Heuvelink, G.B.M., de Bruin, S. & Brus, D.J. (2021). Spatial cross-validation is not the right way to evaluate map accuracy. *Ecol. Model.* 457:109692. doi:10.1016/j.ecolmodel.2021.109692 | Counter-view a reviewer may raise; use it to scope LOSO as a transfer test, not map accuracy |
| P2 | Kapoor, S. et al. (2024). REFORMS: consensus-based recommendations for machine-learning-based science. *Science Advances* 10(18):eadk3452. doi:10.1126/sciadv.adk3452 | Reporting checklist |
| P2 | Ma, Y. & Zhang, Z. (2022). A Bayesian domain adversarial neural network for corn yield prediction. *IEEE GRSL* 19:1–5. doi:10.1109/lgrs.2022.3211444 | County corn transfer across two ecoregions, using unlabelled target data (a different protocol from strict LOSO) |
| P3 | Wang, A.X., Tran, C., Desai, N., Lobell, D. & Ermon, S. (2018). Deep transfer learning for crop yield prediction with remote sensing data. *Proc. ACM COMPASS*:1–5. doi:10.1145/3209811.3212707 | Cross-region transfer for yield |
| P3 | Linnenbrink, J., Milà, C., Ludwig, M. & Meyer, H. (2024). kNNDM CV: k-fold nearest-neighbour distance matching cross-validation for map accuracy estimation. *GMD* 17(15):5897–5912. doi:10.5194/gmd-17-5897-2024 | Recent structured CV |
| P3 | Guarda, T. (2026). Limits to the spatial transferability of machine-learning models for farm-record-level maize yield prediction in Manabí, Ecuador. *Agriculture* 16(17):1932. doi:10.3390/agriculture16171932 | Recent evidence that grouped validation can erase ML skill over a mean benchmark; supports adding naive baselines |
| P3 | Gulrajani, I. & Lopez-Paz, D. (2021). In search of lost domain generalization. *ICLR*. arXiv:2007.01434 | Domain-generalization evaluation rigour |

### 2.4 Conformal prediction since 2020 (areas 2–5)

| P | Reference | Why it is needed |
|---|---|---|
| P1 | **Gibbs, I., Cherian, J.J. & Candès, E.J. (2025). Conformal prediction with conditional guarantees. *JRSS-B* 87(4):1100–1126. doi:10.1093/jrsssb/qkaf008** | **Exact finite-sample coverage over a pre-specified finite set of subgroups or covariate shifts.** It is the principled group-conditional baseline for climate-regime strata, and it bounds the novelty claim. |
| P1 | Gibbs, I. & Candès, E.J. (2024). Conformal inference for online prediction with arbitrary distribution shifts. *JMLR* 25(162):1–36. arXiv:2208.08401 | DtACI removes ACI's step-size sensitivity; the fair modern online comparator |
| P1 | Zaffran, M., Dieuleveut, A., Féron, O., Goude, Y. & Josse, J. (2022). Adaptive conformal predictions for time series. *ICML*, PMLR 162:25834–25866. arXiv:2202.07282 | AgACI; the published version corrects E2 |
| P1 | Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R.J. & Wasserman, L. (2018). Distribution-free predictive inference for regression. *JASA* 113(523):1094–1111. doi:10.1080/01621459.2017.1307116 | Standard split-CP regression reference |
| P1 | Romano, Y., Barber, R.F., Sabatti, C. & Candès, E.J. (2020). With malice toward none: assessing uncertainty via equalized coverage. *Harvard Data Science Review*. doi:10.1162/99608f92.03f00592 | Group-conditional (equalised) coverage as an evaluation target — the conceptual core of the v4 claim |
| P2 | Angelopoulos, A.N., Candès, E.J. & Tibshirani, R.J. (2023). Conformal PID control for time series prediction. *NeurIPS*. arXiv:2307.16895 | Online CP with trend and bias correction; directly relevant to the location-bias mechanism |
| P2 | Bhatnagar, A., Wang, H., Xiong, C. & Bai, Y. (2023). Improved online conformal prediction via strongly adaptive online learning. *ICML*. arXiv:2302.07869 | Online CP comparator |
| P2 | Jung, C., Noarov, G., Ramalingam, R. & Roth, A. (2023). Batch multivalid conformal prediction. *ICLR*. arXiv:2209.15145 | Coverage over overlapping groups |
| P2 | Xu, C. & Xie, Y. (2023). Sequential predictive conformal inference for time series. *ICML*. arXiv:2212.03463 | Time-series CP comparator |
| P2 | Mao, H., Martin, R. & Reich, B.J. (2024). Valid model-free spatial prediction. *JASA* 119(546):904–914. doi:10.1080/01621459.2022.2147531 | Spatial CP; resolves row D-26 |
| P2 | Lou, X., Luo, P. & Meng, L. (2025). GeoConformal prediction: a model-agnostic framework for measuring the uncertainty of spatial prediction. *Ann. Am. Assoc. Geogr.* 115(8):1971–1998. doi:10.1080/24694452.2025.2516091 | Geographic CP; resolves row D-26 |
| P2 | Angelopoulos, A.N. & Bates, S. (2023). Conformal prediction: a gentle introduction. *Found. Trends Mach. Learn.* 16(4):494–591. doi:10.1561/2200000101 | Archival version of A2 |
| P3 | Guan, L. (2023). Localized conformal prediction: a generalized inference framework for conformal prediction. *Biometrika* 110(1):33–50. doi:10.1093/biomet/asac040 | Localised CP theory |
| P3 | Sesia, M. & Candès, E.J. (2020). A comparison of some conformal quantile regression methods. *Stat* 9(1):e261. doi:10.1002/sta4.261 | CQR variants; bears on the joint vs post-hoc claim |
| P3 | Xu, C. & Xie, Y. (2023). Conformal prediction for time series. *IEEE TPAMI* 45(10):11575–11587. doi:10.1109/tpami.2023.3272339 | Journal version of EnbPI |

### 2.5 Conformal and UQ applications in environmental science and agriculture (areas 5, 6)

| P | Reference | Why it is needed |
|---|---|---|
| P1 | Mahmood, S., Hasan, R. & Ahmad, S. (2026). HSE-GNN-CP: spatiotemporal teleconnection modeling and conformalized uncertainty quantification for global crop yield forecasting. *Information* 17(2):141. doi:10.3390/info17020141 | Closest peer-reviewed CQR + crop yield work; reports a single marginal coverage (80.72 % at 80 %) |
| P1 | Farag, M., Emam, A., Leonhardt, J. & Roscher, R. (2025). Enhancing decision support in crop production: analyzing conformal prediction for uncertainty quantification. *Comput. Electron. Agric.* 237:110559. doi:10.1016/j.compag.2025.110559 | Keep, re-described correctly (E5) |
| P1 | Namdeo, D. & Baraskar, R. (2026). Climate-adaptive transformer for crop yield prediction with conformal uncertainty quantification. Zenodo. doi:10.5281/zenodo.19429121 — **grey literature, not peer reviewed** | Split CP + ACI for crop yield under inter-annual climate variability (India). It must be acknowledged so the paper does not appear unaware of an overlapping framing. |
| P2 | Singh, G., Moncrieff, G.R., Venter, Z.S. & Cawse-Nicholson, K. (2024). Uncertainty quantification for probabilistic machine learning in earth observation using conformal prediction. *Scientific Reports* 14. doi:10.1038/s41598-024-65954-w | CP adoption in Earth observation |
| P2 | Jia, Y., Su, X., Singh, V.P. & Zhao, B. (2026). A novel hybrid predictive model based on mixture density networks with weighted conformal inference strategy for runoff interval prediction across Australia. *Water Resour. Res.* doi:10.1029/2024wr039807 | Weighted CP under distribution shift in hydrology |
| P3 | Kakhani, N. et al. (2024). Uncertainty quantification of soil organic carbon estimation from remote sensing data with conformal prediction. *Remote Sensing* 16(3):438. doi:10.3390/rs16030438 | Environmental CP application |
| P3 | Khan, A. et al. (2026). Smart sensing-enabled risk-aware nitrogen prescriptions via conformal profit bounds for precision agriculture. *Frontiers in Plant Science*. doi:10.3389/fpls.2026.1821003 | Agricultural CP for decisions |

### 2.6 Inference, forecasting, UQ baselines, explainability (areas 13, 15)

| P | Reference | Why it is needed |
|---|---|---|
| P1 | Cameron, A.C., Gelbach, J.B. & Miller, D.L. (2008). Bootstrap-based improvements for inference with clustered errors. *Rev. Econ. Stat.* 90(3):414–427. doi:10.1162/rest.90.3.414 | **Year-clustered inference with five clusters is unreliable** — the current tests need this |
| P1 | Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy. *J. Bus. Econ. Stat.* 13(3):253–263. doi:10.1080/07350015.1995.10524599 | Comparing forecast losses under dependence |
| P2 | Gneiting, T. & Katzfuss, M. (2014). Probabilistic forecasting. *Annu. Rev. Stat. Appl.* 1:125–151. doi:10.1146/annurev-statistics-062713-085831 | Calibration and sharpness framing of interval evaluation |
| P2 | Lundberg, S.M. & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*. arXiv:1705.07874 | SHAP is used but uncited |
| P3 | Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*. arXiv:1612.01474 | Non-conformal UQ baseline, if one is wanted |

### 2.7 Data-product primary references (area 14)

| P | Reference | Used for |
|---|---|---|
| P1 | Daly, C. et al. (2008). Physiographically sensitive mapping of climatological temperature and precipitation across the conterminous United States. *Int. J. Climatol.* 28(15):2031–2064. doi:10.1002/joc.1688 | PRISM |
| P1 | Xia, Y. et al. (2012). Continental-scale water and energy flux analysis and validation for the North American Land Data Assimilation System project phase 2 (NLDAS-2): 1. *J. Geophys. Res. Atmos.* 117(D3). doi:10.1029/2011jd016048 | NLDAS-2 |
| P1 | Vicente-Serrano, S.M., Beguería, S. & López-Moreno, J.I. (2010). A multiscalar drought index sensitive to global warming: the SPEI. *J. Climate* 23(7):1696–1718. doi:10.1175/2009jcli2909.1 | SPEI |
| P1 | Beguería, S. et al. (2014). Standardized precipitation evapotranspiration index (SPEI) revisited. *Int. J. Climatol.* 34(10):3001–3023. doi:10.1002/joc.3887 | GLO/PWM fitting used here |
| P2 | Stagge, J.H. et al. (2015). Candidate distributions for climatological drought indices (SPI and SPEI). *Int. J. Climatol.* 35(13):4027–4040. doi:10.1002/joc.4267 | Distribution choice |
| P2 | Svoboda, M. et al. (2002). The Drought Monitor. *Bull. Am. Meteorol. Soc.* 83(8):1181–1190. doi:10.1175/1520-0477-83.8.1181 | USDM DSCI validation of SPEI |
| P3 | Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Q. J. R. Meteorol. Soc.* 146(730):1999–2049. doi:10.1002/qj.3803 | Only if the ERA5 vs PRISM comparison is reported |

Also needed, with no DOI: the FAO-56 report (Allen et al. 1998), and the product documentation for NOAA NCEI Storm Events, NLCD (with the epoch actually used), the soil source (SSURGO/gSSURGO or SoilGrids — **unverified**, see §10.2) and the USDA NASS Quick Stats county yield.

---

## 3. Updated literature comparison tables

### 3.1 Paper-level comparison matrix

"n.s." means not stated in the abstract; those cells were not filled from memory.

| Paper | Year | Problem | Study area | Dataset | Spatial scale | Temporal scale | Method | Baseline | Validation | Metrics | Key finding | Limitation | Gap relevant to this study |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Schlenker & Roberts | 2009 | Yield–temperature nonlinearity | US | County yields (corn, soy, cotton) + daily temperature distribution | County | Annual panel | Panel regression on degree days | Time-series vs cross-section consistency | In-sample structural | Response curves | Corn gains to 29 °C, steep loss above | No predictive intervals | Supplies the statistical baseline and heat thresholds |
| Lobell et al. | 2014 | Changing drought sensitivity | Central US | Field-level maize and soy | Field | 1995–2012 | Statistical sensitivity | n.s. | n.s. | Sensitivity | Maize drought (VPD) sensitivity increased | Not predictive UQ | Non-stationary response supports temporal shift and trend handling |
| Haqiqi et al. | 2021 | Compound vs individual extremes | US | Fine-scale weather + hydrological model | County | 1981–2015 | Yield response function with compound metrics | Individual-extreme metrics | Statistical evidence (n.s.) | n.s. | Compound metrics predict better; heat damage up to 4x with water stress | Econometric; no intervals | **Compound extremes are modelled for US corn**, so Gap 4 is not a gap |
| Hamed et al. | 2021 | Hot–dry drivers of soybean yield | US | Reanalysis + root-zone soil moisture | Sub-national | n.s. | Statistical models with interactions | n.s. | Out-of-sample | Explained variance | ~70 % (60 % out-of-sample); hot-dry lowers yields ~2 SD | Soybean; no intervals | Interaction modelling precedent |
| Troy et al. | 2015 | Extremes and irrigation | US | County yields, daily P and T | County | n.s. | Conditional density analysis | n.s. | Descriptive | n.s. | Irrigation shifts or decouples climate effects | Descriptive | Irrigation must be controlled |
| Crane-Droesch | 2018 | ML for yield and climate impact | US Midwest | Corn yield + weather | County | Multi-decade | Semiparametric deep NN | Classical statistics; fully nonparametric NN | Withheld years | Out-of-sample error | Beats both baselines | Point only | NN with parametric trend and fixed effects: a hybrid baseline idea |
| Jiang et al. | 2020 | County corn yield estimation | US Corn Belt | Phenology, meteorology, remote sensing | County | Multi-year incl. 2012 | LSTM | LASSO, RF | n.s. | R², RMSE | 76 % of variation; 2012 RMSE 1.47 vs 1.93 / 2.43 Mg/ha | Point only; remote-sensing inputs | End-of-season skill benchmark; extreme-year point skill |
| Kang et al. | 2020 | Variables and algorithms for maize yield | US Midwest | Satellite, weather, land-surface model, soil, crop progress | County | n.s. | Lasso, SVR, RF, XGBoost, LSTM, CNN | Lasso | n.s. | Accuracy, stability | XGBoost best; DL not advantageous; forecasts from early June | Point only | Tree ensembles are enough; argues against architecture novelty |
| Khaki et al. | 2020 | Corn and soy yield prediction | 13 Corn Belt states | Environment + management | County | Test 2016–2018 | CNN-RNN | RF, DFNN, LASSO | Temporal hold-out | RMSE | RMSE 9 % (corn) of mean yield | Point only | Temporal hold-out without intervals |
| Shahhosseini et al. | 2020 | Corn yield forecasting | IL, IN, IA | Weather, soil, management | County | n.s. | ML ensembles | Individual learners | Blocked sequential | RRMSE, MBE | Ensemble RRMSE 7.8 % | Point only | Ensemble and blocked-CV precedent in your states |
| Shahhosseini et al. | 2021 | ML + crop model | US Corn Belt | Weather, soil + APSIM outputs | County | n.s. | LightGBM, XGBoost, RF, LASSO, ensembles | ML without APSIM | n.s. | RMSE | APSIM features cut RMSE 7–20 %; soil-moisture variables most influential | Point only | Missing hydrology inputs are a known limitation |
| Leng & Hall | 2020 | Model classes for average, variability, extremes | US | Maize yield | n.s. | n.s. | ML vs regression vs process models | Each other | n.s. | Variability explained | ML 93 % national variability | Point / distribution only | Extremes of the yield distribution |
| Lischeid et al. | 2022 | ML drivers of yield | Germany, 351 counties | Soil + meteorology | County | 40 years | RF, SVM after smooth detrending | Each other | n.s. | Variance explained | 50–70 %; different predictor sets equally good | Not US | Detrending approach; non-unique predictors undercut SHAP claims |
| Kallenberg et al. (CY-Bench) | 2026 | Benchmark for sub-national yield forecasting | Global (maize, wheat) | Harmonised yield, weather, soil, remote sensing | Sub-national | Multi-year | Benchmark + baselines | Provided | Standardised | Standardised | Common protocol | Global, not county-US-specific | External benchmark and protocol reference |
| Ma et al. | 2021 | Corn yield UQ | US | Remote-sensing variables | County | n.s. | Bayesian NN | n.s. | n.s. | Predictive uncertainty | Uncertainty rises under stress (existing verified record) | No coverage guarantee | Bayesian UQ comparator |
| Zhang & Diao | 2023 | Soybean yield UQ | US Corn Belt | Phenology-guided inputs | County | n.s. | Bayesian CNN | n.s. | n.s. | Aleatoric / epistemic | Phenology-structured uncertainty | No coverage guarantee | Phenology + UQ comparator |
| Ma & Zhang | 2022 | Cross-region corn yield | Two US Corn Belt ecoregions | Remote sensing | County | n.s. | Bayesian domain-adversarial NN | State-of-the-art methods (n.s.) | Source→target domain | n.s. | Better transfer on small training sets | Uses unlabelled target data | Spatial transfer for yield (different protocol) |
| Farag et al. | 2025 | CP for agricultural decision support | Agricultural image tasks | Image datasets | — | — | Inductive CP on ResNet-18 / ViT-B/16 | Other UQ methods | Ablation; covariate shift; OOD | Coverage | ICP meets nominal error levels | **Classification images, not yield regression** | CP in agriculture (positioning only) |
| Mahmood et al. | 2026 | Crop yield forecast with CQR | 15 countries, 6 crops | Global yields | Country | 1990–2023 | Stacked ensembles + GNN + CQR | Uncalibrated intervals | n.s. | R², RMSE, coverage | R² 0.9594; 80.72 % coverage at 80 % | Marginal coverage only; not county; no method comparison | **Closest peer-reviewed CQR-yield work; marginal only** |
| Namdeo & Baraskar (grey) | 2026 | Crop yield + CP under inter-annual climate variability | 10 Indian states | 7,182 farm-years, ERA5, MODIS/S2 | Farm | 2010–2024 | Transformer + split CP + ACI | RF, XGBoost, LSTM, BiLSTM | n.s. | R², RMSE, PICP | PICP 0.940 at 95 % | Not peer reviewed; marginal PICP | Overlapping framing; must be disclosed |
| Gibbs & Candès | 2021 | Coverage under arbitrary shift | — | — | — | Online | ACI | Split CP | Theory + experiments | Long-run coverage | Long-run marginal coverage without exchangeability | Marginal; step-size sensitive | The method under test |
| Gibbs, Cherian & Candès | 2025 | Conditional guarantees | — | — | — | — | Coverage over covariate-shift classes | Marginal CP | Theory | Group coverage | **Exact finite-sample coverage over pre-specified subgroups** | Needs pre-specified groups or function class | **Required baseline for regime-conditional coverage** |
| Zaffran et al. | 2022 | ACI for time series | — | Electricity price data (n.s.) | — | Time series | AgACI | ACI | Theory + experiments | Coverage, width | Aggregation removes γ tuning | Marginal | Online comparator |
| Mao, Martin & Reich | 2024 | Spatial CP | — | Real + simulated | Spatial | — | Local spatial CP | Model-based intervals | Theory + numerics | Validity, efficiency | Valid, efficient without stationarity | Infill asymptotics | Spatial dependence comparator |
| Lou, Luo & Meng | 2025 | Geographic CP | — | Housing prices, interpolation | Spatial | — | GeoCP | Bootstrap, kriging variance | Simulation + cases | Coverage | 93.67 % vs bootstrap 81.00 % | Not agriculture | Spatial CP comparator |
| Roberts et al.; Ploton et al. | 2017; 2020 | Structured validation | Ecology | Various | Various | Various | Blocked CV | Random CV | Structured vs random | Skill | Random CV inflates skill | Not yield | Supports temporal + LOSO design |
| Guarda | 2026 | Spatial transferability of yield ML | Manabí, Ecuador | 518 farm records | Parish | Seasonal | Elastic Net, RF, gradient boosting | Training-fold mean | Parish-grouped nested CV | MAE, R² | No model beat the mean benchmark; pooled R² negative; random CV optimistic | Small, farm-level | **Naive baselines are necessary under grouped validation** |

### 3.2 Research-dimension table — is the claimed contribution supported?

Status of "what this study does" is as of the 2026-09-16 run.

| Research dimension | Existing literature | Limitation | What this study does | Evidence of difference |
|---|---|---|---|---|
| Target and scale | County corn yield, US Corn Belt (Jiang 2020; Kang 2020; Khaki 2020; Shahhosseini 2020) | — | County corn yield, six rainfed Corn Belt states, 1985–2023 | **None**: same target and scale |
| Weather inputs | Fine-scale weather + hydrological or land-surface models (Haqiqi 2021; Ortiz-Bobea 2019; Kang 2020) | — | PRISM daily + NLDAS-2 FAO-56 ET0; county-area-weighted | **Engineering quality, not scientific novelty** |
| Compound drought–heat | Compound metrics already outperform univariate ones for US corn (Haqiqi 2021); interaction models (Hamed 2021) | Rarely stage-windowed inside ML predictors | Daily SPEI-30 < −1 AND Tmax > 35 °C, windowed by GDD stage | **Weak**: O1 gain +0.0116 R², single seed; the ablation gives the opposite sign **[M]** |
| Point model | Tree ensembles are competitive with or beat DL (Kang 2020) | — | Residual MLP + LightGBM ensemble | **None**: LightGBM alone (R² 0.731) beats the ensemble (0.703) on test **[M]** |
| Uncertainty method | Bayesian NN/CNN (Ma 2021; Zhang & Diao 2023); CQR (Mahmood 2026) | No coverage guarantee (Bayesian) or marginal only (CQR) | CQR backbone + 6 conformal calibrators | **Moderate**: breadth of the calibrator comparison |
| Coverage evaluation | Marginal coverage only in yield applications (Mahmood 2026; Namdeo 2026 grey) | Extreme regimes, where the decision value lies, are not evaluated | Marginal **and** coverage within year, state, CDHW-exposure, drought, heat and ENSO strata | **Moderate–strong, the defensible core** |
| Conditional-coverage methods | Group-conditional guarantees exist (Romano 2020; Jung 2023; Gibbs 2025) | Not applied to crop yield in the verified literature | None implemented in current code | **Gap in this study**, not in the literature: required baseline |
| Temporal shift | Temporal hold-out standard (Khaki 2020; Crane-Droesch 2018) | Single hold-out common | Four-way split with calibration independent of model selection; one 5-year test window | **Better leakage control; weaker temporal replication** (one window) |
| Spatial shift | Domain adaptation with target data (Ma & Zhang 2022; Wang 2018) | Coverage under cross-state transfer not reported | Strict six-state LOSO with per-fold selection, scaling, detrending and calibration | **Moderate**: coverage under LOSO + bias–coverage decomposition is new in this literature neighbourhood |
| Mechanism of coverage failure | Theory: ACI adapts to the recent past (Gibbs & Candès 2021); PID corrects bias (Angelopoulos 2023) | No yield-domain demonstration | LOSO: coverage tracks \|bias\| (r = −0.77), not width (r = 0.00); all methods fail most in 2021 **[M]** | **Moderate** |
| Inference | Cluster-robust and few-cluster bootstrap (Cameron 2008); forecast comparison (Diebold & Mariano 1995) | — | Row-level Wilcoxon (p = 0.0); 5-cluster bootstrap | **Deficient** |
| Leakage control | Taxonomy and checklists (Kapoor & Narayanan 2023; REFORMS 2024) | Rarely audited at row level in yield ML | Row-identity provenance ledger, asserted before fitting, 7 experiments PASS | **Strong practice**; modest publishable novelty |
| Baselines | Statistical panel models (Schlenker & Roberts 2009); naive mean benchmark (Guarda 2026) | — | Tree and neural backbones only | **Deficient**: naive and panel baselines missing |

**Conclusion of §3.2.** Only two dimensions carry a defensible difference: **regime-conditional coverage evaluation of conformal calibrators** and **coverage under strict cross-state transfer with its bias mechanism**. Everything else is established practice, engineering quality, or currently deficient.

---

## 4. Research gap audit

Each gap below was claimed in at least one project document. Classification scale: **1** strongly supported · **2** partially supported · **3** weakly supported · **4** not actually a gap · **5** potentially outdated · **6** unsupported by literature.

### 4.1 Summary

| ID | Claimed gap (source) | Classification | Publishable as a gap? |
|---|---|---|---|
| G1 | Compound drought–heat events are unmodelled in yield models (proposal Gap 4, O1) | **4 + 5** — not a gap, and outdated | No |
| G2 | Yield uncertainty is uncalibrated under climate distribution shift (proposal Gap 5) | **2** | Only in the narrowed form in §4.2 |
| G3 | Static conformal intervals "collapse" in anomalous years (proposal, "Paper 18") | **6** — unsupported, and contradicted by the data | No |
| G4 | No joint end-to-end uncertainty training in the literature (O4) | **3** | No, as a gap; possibly as a secondary result |
| G5 | Interval width should track compound-stress severity (O5) | **3** | No, since the result is weak |
| G6 | Phenology-aware *conformal* calibration does not exist (methodology §1) | **3** | No |
| G7 | Marginal calibration metrics misrank conformal methods for extreme-regime use (v4 G1) | **2**, upgradable to **1** | **Yes — the core gap** |
| G8 | Conformal validity under temporal dependence in crop yield (v4 G2) | **2** | Not standalone |
| G9 | Partition-conditional calibration is infeasible with clustered extremes (v4 G3) | **2**, but must now be tested against Gibbs et al. 2025 | Only together with G7 |
| G10 | Coverage under cross-state (LOSO) transfer for county yield (methodology §5.2) | **2** | **Yes — second gap** |
| G11 | Distribution-free conditional guarantees (v4 G4) | **4** — correctly rejected, but the rejection is over-stated | No |
| G12 | Spatial conformal for county yield (v4 G6) | **2** — methods exist (Mao 2024; Lou 2025) | Optional comparator |
| G13 | Novel architecture for county yield (v4 G7) | **4** | No |
| G14 | EVT tail risk / crop-insurance framework (v4 G8, G9) | **4 / 6** | No — drop |

### 4.2 Detail

**G1 — Compound drought–heat unmodelled.** *Classification: 4 (not a gap) and 5 (outdated).*
- **Existing evidence [L]:**
  - Haqiqi et al. (2021): US corn, 1981–2015; compound metrics predict yield variation better than individual extremes, and heat damage is up to 4x worse with water stress.
  - Hamed et al. (2021): interaction models for US soybean with out-of-sample evaluation.
  - Lobell et al. (2013, 2014); Lesk et al. (2022); Heino et al. (2023); Ribeiro et al. (2020); Xiao et al. (2025).
- **Already solved:** the phenomenon, its US-corn yield impact, and its superior explanatory power over univariate metrics.
- **Still unresolved:** whether phenology-windowed compound-exposure *features* add out-of-sample skill to ML models that already contain heat-day counts, KDD and drought indices.
- **Your evidence [M]:** O1 gives ΔR² +0.0116 (RMSE 0.9701 → 0.9529), single seed. The pipeline's own 5-row ablation gives the **opposite sign** (RMSE 0.9511 without CDHW → 0.9629 with). The CDHW thresholds and the 1 April GDD phenology start were never subjected to a sensitivity test.
- **Publication-worthy?** No. **Conservative rewrite:** "Compound drought–heat stress is a documented driver of US maize yield loss (Haqiqi et al., 2021). We test whether phenology-windowed compound-exposure features change predictive skill or interval behaviour beyond univariate heat and drought covariates." Treat it as a secondary question with a possibly null answer.

**G2 — Uncertainty uncalibrated under distribution shift.** *Classification: 2.*
- **Existing evidence [L]:**
  - In general statistics this is solved or actively solved: ACI (Gibbs & Candès 2021), DtACI (Gibbs & Candès 2024), AgACI (Zaffran et al. 2022), conformal PID (Angelopoulos et al. 2023), weighted CP (Tibshirani et al. 2019), and validity bounds beyond exchangeability (Barber et al. 2023).
  - In crop yield: Bayesian UQ without coverage guarantees (Ma et al. 2021; Zhang & Diao 2023); CQR with one marginal coverage figure (Mahmood et al. 2026); split CP + ACI in non-peer-reviewed work (Namdeo & Baraskar 2026).
- **Still unresolved:** peer-reviewed, systematic evaluation of *several* conformal calibrators on county yields under both temporal and cross-state shift.
- **Why it matters:** yield intervals are used exactly in anomalous years.
- **How the methodology addresses it:** six calibrators, temporal and LOSO evaluation. In LOSO, however, only static CP is run.
- **Conservative rewrite:** "Conformal methods for distribution shift are well developed, but their comparative behaviour on county-level crop yields under temporal and cross-state shift has not been systematically reported in peer-reviewed work."

**G3 — "Static conformal collapses in anomalous years".** *Classification: 6 (unsupported).*
- The motivating citation ("Paper 18": 91 % → 73 %) was fabricated (`LITERATURE_CITATION_VERIFICATION.md`).
- **Current data contradicts the premise [M]:** static CP has the second-best worst-stratum coverage (0.876). The ACI family has the worst (standard ACI 0.852, SA-ACI 0.807).
- **Action:** delete from every document. The real, narrower phenomenon is covered by G7.

**G4 — No joint end-to-end uncertainty training.** *Classification: 3.*
- **Existing evidence [L]:** jointly trained multi-quantile networks are standard (the CQR paper itself fits quantile networks; Sesia & Candès 2020 compare CQR variants). The claim "not in the literature" is not supported by any cited source.
- **Your evidence [M], single seed:**
  - Joint vs post-hoc RMSE is identical (0.9630 vs 0.9629).
  - Joint training narrows static-CP MPIW by 19.4 % and improves Winkler by 15.6 %, with PICP 0.954 vs 0.965 (both above nominal).
  - Winkler improves in all 5 test years (year-level paired t p = 0.0004; Wilcoxon p = 0.0625, the minimum attainable with n = 5).
- **Publication-worthy?** Not as a gap. At most a secondary, multi-seed-replicated result: "joint training yields sharper conformalised intervals without changing point accuracy".

**G5 — Interval width tracks CDHW severity (O5).** *Classification: 3.*
- **Existing evidence [L]:** locally adaptive and conditional CP (Lei et al. 2018; Guan 2023; Gibbs et al. 2025) make width covariate-dependent by design.
- **Your evidence [M]:** Pearson r = 0.166, R² = 0.028. The slope p is 0.116 with year clusters (5) and ≈ 0 with county clusters (555), so the inference depends on the clustering unit, and neither choice is reliable with five years.
- **Rewrite:** report as a weak, inconclusive association; do not frame as a gap.

**G6 — Phenology-stratified conformal as a new method.** *Classification: 3.*
- **Existing evidence [L]:** stratifying calibration by a partition *is* Mondrian conformal prediction (Vovk 2013; Toccaceli & Gammerman 2019).
- **Your evidence [M]:** phenology-stratified CQR is **numerically indistinguishable** from static CP (PICP 0.9493 vs 0.9488; MPIW 3.6506 vs 3.6493; identical worst stratum 0.8762). The phenology partition is near-degenerate because season-total GDD puts most county-years in one window.
- **Action:** drop as a contribution. Keep it only as a Mondrian instance, if at all.

**G7 — Marginal metrics misrank conformal methods for extreme-regime use (core).** *Classification: 2, upgradable to 1.*
- **Existing evidence [L]:**
  - CP guarantees are marginal (Vovk 2013).
  - Distribution-free conditional coverage is impossible in general (Barber et al. 2021).
  - Group-conditional coverage is attainable for pre-specified groups (Romano et al. 2020; Jung et al. 2023; Gibbs et al. 2025).
  - Crop-yield CP papers report a single marginal figure (Mahmood et al. 2026; Namdeo & Baraskar 2026, grey).
- **Already solved:** the theory, and methods for group-conditional coverage.
- **Still unresolved:** whether the choice among *commonly used* calibrators changes when judged within climate-exposure regimes on real county-yield data, and why.
- **Your evidence on the current run [M]** (`p8_conditional_coverage_current_run.csv`):

| Method | Marginal PICP | Marginal ACE (rank) | Marginal Winkler (rank) | Worst-stratum PICP (rank) | Marginal − worst |
|---|---|---|---|---|---|
| SA-ACI | 0.9194 | 0.0194 (**1**) | 4.0751 (**1**) | 0.8069, Extreme (**6**) | 0.1125 |
| Standard ACI | 0.9296 | 0.0296 (2) | 4.0788 (2) | 0.8515, Extreme (5) | 0.0781 |
| Weighted CP | 0.9471 | 0.0471 (3) | 4.1413 (3) | 0.8713, Extreme (4) | 0.0758 |
| Static CP | 0.9488 | 0.0488 (4) | 4.1473 (4) | 0.8762, Extreme (2) | 0.0726 |
| Phenology-stratified CQR | 0.9493 | 0.0493 (6) | 4.1475 (5) | 0.8762, Extreme (2) | 0.0731 |
| Locally adaptive | 0.9488 | 0.0488 (4) | 4.1670 (**6**) | 0.8911, 2021 (**1**) | 0.0577 |

  SA-ACI has the lowest coverage of all six methods in **every one of the seven regime axes** tested: year, state, CDHW class, drought, heat, ENSO and "Year_Type".

- **Confounds a reviewer will raise [M]:**
  - **(a) Stratum composition.** "Extreme" (n = 202) is 175/202 from 2022–2023 and 146/202 from Missouri and Iowa (Ohio 1, Indiana 5). Regime is confounded with year and state.
  - **(b) Level vs shape.** Methods start from different marginal coverage (0.919–0.949), so part of the inversion is slack. The *drop* (marginal − worst) still differs (SA-ACI 0.113 vs 0.058–0.078), so it is not only slack.
  - **(c) One test window of five years.** All methods fail most in 2021.
  - **(d) Information asymmetry.** ACI and SA-ACI update on realised test-year outcomes; static CP does not.
  - **(e) No group-conditional baseline.** The obvious remedy is untested.
- **Why it matters:** the practical choice of a UQ method for drought and heat years would be wrong if made on the standard marginal criterion.
- **Sufficiently important?** Yes, once (a)–(e) are addressed with rolling-origin replication, dependence-aware inference and a group-conditional baseline (§17 C2–C5).
- **Conservative rewrite:** "In crop-yield applications, conformal calibration methods are compared on marginal coverage. We test whether that ranking holds within pre-specified climate-exposure, year and state strata, where decision value concentrates, and whether a group-conditional procedure changes the conclusion."

**G8 — Temporal dependence.** *Classification: 2.*
- **Existing evidence [L]:** methods exist (EnbPI, SPCI, AgACI, DtACI).
- **Your evidence [M]:** the current exchangeability report computes a lag-1 ACF on **five yearly residual means** (−0.158), which is uninformative. `residual_diagnostic_report.json` reports "spatial autocorrelation detected: true" while `exchangeability_diagnostics_report.json` reports "spatial dependence low: true". The two diagnostics disagree.
- **Action:** fold into the methodology as a stated assumption plus one online comparator (DtACI or AgACI); not a standalone gap.

**G9 — Partition-conditional feasibility.** *Classification: 2.*
- The feasibility concern (few calibration points per stratum) is legitimate. But Gibbs et al. (2025) handles overlapping, pre-specified groups with finite-sample guarantees, and the legacy Mondrian test is **not in the current code** (E8).
- **Action:** implement and test; the result (it works, or it fails for a documented reason) belongs inside G7.

**G10 — Coverage under cross-state transfer.** *Classification: 2.*
- **Existing evidence [L]:**
  - Spatial CP exists (Mao et al. 2024; Lou et al. 2025).
  - Yield-model spatial transfer is studied (Ma & Zhang 2022; Wang et al. 2018), and grouped validation can erase ML skill (Guarda 2026).
  - Coverage of conformal intervals under leave-one-state-out transfer for county yields is not reported in the verified literature.
- **Your evidence [M]:**
  - After the detrend fix, macro LOSO R² is 0.698 and PICP 0.916, with Minnesota 0.845 and Illinois 0.886 below nominal.
  - Across 234 state-years, coverage correlates with |bias| at r = −0.77 and with interval width at r = 0.00.
  - The residual yearly-bias slope tracks the mismatch between the five-state and own-state trend at r = 0.96.
  - Before the fix, 72 % of MPIW was the conformal correction for that bias.
- **Why it matters:** trend transfer, not spatial covariate shift, is the dominant coverage risk in cross-state yield UQ. This is a reusable lesson for LOSO designs.
- **Addressed?** Partially: only static CP is evaluated in LOSO, and there is a single seed.
- **Conservative rewrite:** "We evaluate whether conformal coverage transfers across states under strict leave-one-state-out validation and diagnose whether failures stem from interval width or from level bias."

**G11 — Conditional guarantees impossible.** *Classification: 4, correctly rejected.*
- The current wording ("exact conditional coverage is unattainable… no method can…") **over-generalises** Barber et al. (2021). Their impossibility concerns full conditional coverage on continuous covariates. Coverage over a finite set of pre-specified regimes **is** attainable (Gibbs et al. 2025).
- **Action:** correct the sentence; it currently exposes the paper to an easy rebuttal.

**G12 — Spatial conformal.** *Classification: 2.* Methods now verified. Optional as a LOSO comparator.

**G13 — Architecture.** *Classification: 4.*
- **[L]** Kang et al. (2020): XGBoost beats LSTM and CNN.
- **[M]** LightGBM alone beats the ensemble on test.
- Agree with the existing rejection.

**G14 — EVT and insurance.** *Classification: 4 / 6.* No peer-reviewed support was located and nothing is implemented. Drop.

---

## 5. Novelty audit

### 5.1 By category

| Category | Genuinely novel? | Or an engineering combination? | Similar published work | Distance from closest work | Evidence |
|---|---|---|---|---|---|
| 1. Problem | **Low–moderate.** Yield UQ under climate extremes is an established problem; the narrower question (does calibrator choice survive regime-conditional and cross-state evaluation) is less studied | — | Mahmood 2026; Namdeo 2026 (grey); Ma 2021; Zhang & Diao 2023 | Moderate for the narrow question | §3.2 |
| 2. Data / fusion | **Low as science; high as engineering** | Standard products and methods: PRISM, NLDAS-2 FAO-56 ET0, SPEI via GLO/PWM, area-weighted county aggregation | Haqiqi 2021 (fine-scale weather + hydrological model); Ortiz-Bobea 2019 (land-surface model + weather); Kang 2020 (land-surface model, weather, soil, crop progress) | Small | Daly 2008; Xia 2012; Vicente-Serrano 2010; Beguería 2014 |
| 3. Methodological | **Low.** CQR, split, weighted, local and online CP all exist; SA-ACI is an ACI variant that ranks worst conditionally | Combination | Romano 2019; Gibbs & Candès 2021, 2024; Tibshirani 2019; Vovk 2013; Gibbs 2025 | Small | **[M]** SA-ACI worst in 7/7 axes |
| 4. Modelling | **None** | Residual MLP + LightGBM ensemble | Kang 2020; Shahhosseini 2020; Jiang 2020 | None | **[M]** LightGBM alone R² 0.731 > ensemble 0.703 |
| 5. Evaluation / validation | **Moderate.** Regime-stratified coverage of six calibrators; calibration independent of model selection; strict LOSO with per-fold selection and detrending; row-identity leakage ledger | Protocol design | Group coverage evaluation (Romano 2020); structured CV (Roberts 2017) | Moderate in the crop-yield literature | §3.2 |
| 6. Application | **Low–moderate** | — | County corn yield ML (many) | Small | — |
| 7. Practical | **Moderate, if replicated**: do not choose a yield-UQ method on marginal metrics when the use case is extreme years; detrending and trend transfer decide LOSO coverage | — | None located | Moderate | **[M]** §4.2 G7, G10 |
| 8. Scientific | **Moderate**: coverage failures are driven by location bias, not width; ACI's lagged adaptation after calm years | Mechanistic analysis | Theory implies it (Gibbs & Candès 2021; Angelopoulos 2023) | Moderate as domain evidence | **[M]** r(\|bias\|, coverage) = −0.77, r(width, coverage) = 0.00 |

### 5.2 Closest existing studies

| Closest existing study | Their approach | This study's approach | Key difference | Genuine novelty? | Evidence |
|---|---|---|---|---|---|
| Mahmood et al. 2026, *Information* | Stacked ensembles + GNN + CQR; 15 countries, 6 crops; one coverage figure (80.72 % at 80 %) | County corn; 6 calibrators; marginal + regime-conditional; temporal + LOSO | Scale, conditional evaluation, calibrator comparison, cross-state | **Yes, on evaluation** | Abstract **[L]** |
| Namdeo & Baraskar 2026, Zenodo (not peer reviewed) | Transformer + split CP + ACI for yield under inter-annual variability; PICP 0.940 marginal | Same families of calibrators; conditional + LOSO evaluation | Conditional and cross-state evaluation; peer-reviewable rigour | **Partial.** The framing "ACI for yield under climate variability" is **not** novel. | Abstract **[L]** |
| Farag et al. 2025, *Comput. Electron. Agric.* | Inductive CP on image classifiers; covariate shift and OOD experiments | Regression CP on county yields | Task (classification vs regression), data, regime conditioning | Yes, as distinct task | Abstract **[L]** |
| Ma et al. 2021, *RSE* | Bayesian NN corn yield uncertainty | Conformal intervals with empirical coverage evaluation | Coverage-evaluated vs predictive-variance UQ | Moderate | Existing verified record |
| Zhang & Diao 2023, *ISPRS J.* | Phenology-guided Bayesian CNN, soybean, county | Phenology-windowed CDHW features + CP | UQ family; crop | Low–moderate; phenology stratification adds nothing here **[M]** | Existing verified record |
| Gibbs & Candès 2021, *NeurIPS* | ACI: long-run marginal coverage under shift | Applies ACI and a severity-aware variant | Domain application; shows conditional failure | No, methodologically; yes, as a domain finding | **[M]** |
| **Gibbs, Cherian & Candès 2025, *JRSS-B*** | Finite-sample coverage over pre-specified groups | No group-conditional method yet | — | **Threatens the claim** unless implemented and compared | Abstract **[L]** |
| Zaffran et al. 2022, *ICML* | AgACI for dependent series | ACI and SA-ACI only | Missing comparator | No | **[L]** |
| Haqiqi et al. 2021, *HESS* | Compound-extreme metrics for US corn yield response | CDHW features in ML | ML vs econometric; no UQ there | **No** for compound-event novelty | Abstract **[L]** |
| Jiang et al. 2020, *GCB* | LSTM county corn; extreme-year point skill | Tabular ML + intervals | Intervals and coverage | Yes, for UQ; no, for point skill | Abstract **[L]** |
| Mao 2024, *JASA*; Lou 2025, *Ann. AAG* | Spatial conformal prediction | Strict LOSO with global calibrators | Transfer across states vs local spatial exchangeability | Moderate; a spatial-CP comparator would strengthen it | Abstracts **[L]** |

---

## 6. Defensible novelty statements

### A. Conservative (survives any review)

> This study evaluates six conformal calibration methods — split, weighted, locally adaptive, phenology-stratified, adaptive (ACI) and a severity-aware ACI variant — for county-level corn yield in six U.S. Corn Belt states under temporal (2019–2023) and leave-one-state-out distribution shift. Coverage is assessed both marginally and within pre-specified climate-exposure, year and state strata. The methods that are best on marginal calibration error have the lowest coverage in the most compound-drought–heat-exposed stratum. Under cross-state transfer, coverage failures track year-level prediction bias rather than interval width.

### B. Strong but defensible

Only after critical changes C2–C5 (§17) are complete and the result replicates:

> Crop-yield uncertainty studies typically report a single marginal coverage figure (e.g., Mahmood et al., 2026), or quantify uncertainty without coverage guarantees (Ma et al., 2021; Zhang & Diao, 2023). We show on a leakage-audited county panel that the marginal ranking of conformal calibrators reverses within compound drought–heat regimes across N rolling test years. We show that a group-conditional conformal procedure (Gibbs et al., 2025) [restores / does not restore] within-regime coverage at a quantified width cost. Under leave-one-state-out transfer, the dominant coverage risk is transfer of the technology trend — not covariate shift — so detrending and trend-transfer assumptions, rather than the calibrator, determine whether intervals remain valid in an unseen state.

### C. Claims that must NOT be made

| Claim | Why it fails |
|---|---|
| "Novel architecture", "PhenoFormer", "Transformer" | No such model exists; backbones are a residual MLP and LightGBM |
| "State-of-the-art accuracy" | LightGBM alone beats the proposed ensemble; no comparison to Jiang 2020 or Kang 2020 under a common protocol |
| "First to model compound drought–heat in yield prediction" | Haqiqi 2021, Hamed 2021 and others |
| "CDHW captures super-linear yield damage" (O1 wording) | Ablation sign reversal; single seed; predictive ≠ causal |
| "Joint training improves point accuracy" | RMSE unchanged (0.9630 vs 0.9629) |
| "ACI prevents coverage collapse in anomalous years" | ACI family has the worst extreme-stratum coverage |
| "Static conformal collapses in anomalous years" | Fabricated source; contradicted by data |
| "Guaranteed" or "distribution-free" coverage under shift | Exchangeability violated; coverage is empirical |
| "Conditional coverage is impossible, hence evaluation is the only possible contribution" | Group-conditional coverage is attainable (Gibbs 2025) |
| "Phenology-stratified conformal calibration is a new method" | It is Mondrian CP, and numerically identical to static here |
| "SA-ACI is significantly superior (p = 0.0)" | Row-level tests ignore dependence; Friedman p = 0.52 over 5 year-blocks; Winkler spread 2.2 % |
| "Model-agnostic" | Calibration comparison is run on one backbone |
| "Generalises across the United States" | Six rainfed Corn Belt states |
| "Early warning" / "in-season decision support" | Features are season totals: end-of-season estimation only |
| "100 % methodology compliance", "zero leakage" | Validator is NON_COMPLIANT; say "no leakage detected by the row-identity audit" |
| "Strong SHAP agreement across backbones" | Similarity 0.60, dominated by `county_baseline` |

---

## 7. Revised research objectives

Built to follow from Literature → Gap (G7, G10) → Research question. The core aim — calibrated yield uncertainty under climate distribution shift — is unchanged.

| # | Objective | Serves |
|---|---|---|
| **RO1** | Construct a leakage-controlled county-year corn yield panel (six rainfed U.S. Corn Belt states, 1985–2023) with PRISM/NLDAS-2 weather and phenology-windowed compound drought–heat exposure, with all learned transformations fitted on training years only. | Enabler (not claimed as a scientific contribution) |
| **RO2** | Quantify point-prediction skill under temporal and leave-one-state-out shift relative to naive trend and fixed-effects degree-day baselines. | Establishes that intervals are built on a model worth calibrating |
| **RO3** | Compare conformal calibration methods on marginal and on pre-specified regime-conditional coverage, under temporal (rolling-origin) and cross-state shift. The methods are split, rolling split, weighted, locally adaptive, ACI, one modern online variant, and group-conditional. | G7, G9, G10 |
| **RO4** | Diagnose the mechanism of coverage failure: location bias vs interval width, adaptation lag, and trend transfer across states. | G7, G10 |
| **RO5** (secondary) | Test, with multiple seeds, whether joint point-quantile training or compound-exposure features change interval sharpness or regime-conditional coverage. | G1 and G4 narrowed; null results reportable |

---

## 8. Revised research questions

Every question can be answered with the existing data plus the changes in §17; none needs new data.

- **RQ1.** Under temporal and leave-one-state-out shift, how much predictive skill do tree and neural backbones add over (a) a training-period trend plus county-mean baseline and (b) a fixed-effects degree-day panel regression?
- **RQ2.** Across rolling test years, are the conformal methods that minimise marginal calibration error also those with the highest coverage in the weakest pre-specified climate-exposure, year and state strata?
- **RQ3.** Is regime-conditional under-coverage explained mainly by prediction bias or by interval width? Under cross-state transfer, does mismatch between the training-states' and the held-out state's yield trend explain coverage failures?
- **RQ4.** Does a group-conditional conformal procedure with pre-specified regime groups restore within-regime coverage, and at what cost in interval width?
- **RQ5.** Across ≥ 5 seeds, do joint point-quantile training or phenology-windowed compound drought–heat features change interval sharpness or regime-conditional coverage, relative to post-hoc training and univariate stress features?

---

## 9. Revised contributions

Only claims the current or required evidence can carry. Items marked † depend on the critical changes replicating the current result.

### 9.1 Scientific contributions

1. **† Evidence that marginal and regime-conditional rankings of conformal calibrators diverge for crop yield.** On the current run, the best marginal method (SA-ACI) has the lowest coverage in all seven regime axes (worst stratum 0.807), and the worst marginal-Winkler method (locally adaptive) has the best worst-stratum coverage (0.891).
2. **A mechanism.** Under cross-state transfer, coverage failures track year-level bias (r = −0.77 over 234 state-years), not interval width (r = 0.00). All calibrators fail most in the same year (2021), consistent with a location shift that width-adaptive methods cannot anticipate.
3. **A validation lesson for leave-one-state-out yield studies.** Target detrending and trend transfer dominate both skill and coverage. Macro LOSO R² was 0.357 without detrending and 0.698 with it; MPIW was 8.11 and 4.25. Residual failures track the state-trend mismatch (r = 0.96).

### 9.2 Methodological contribution

Modest; claim as protocol, not as a method.
- A reporting and validation protocol for yield UQ: calibration data disjoint from model selection, strict LOSO with per-fold selection, scaling and detrending, a row-identity provenance ledger, and regime-stratified coverage reporting with stratum composition.

### 9.3 Practical contributions

Scoped to end-of-season estimation.
- **†** Guidance: select yield-UQ methods on regime-conditional coverage when the use case is extreme years.
- Quantified interval widths (t/ha) and their cost under temporal and cross-state use.

---

## 10. Methodology audit

Audited independently of the literature, against the code and the 2026-09-16 run. Status: **OK** adequate · **NI** needs improvement · **MAJ** major issue.

### 10.1 The 35 checks

| # | Item | Status | Evidence | Required action |
|---|---|---|---|---|
| 1 | Research objectives | **MAJ** | Three incompatible objective sets. The proposal's O1–O5 carry numeric targets that were not met: ΔR² ≥ 0.10, PICP ≥ 0.90 in all anomalous years, MPIW −10 % *and* RMSE −5 %, r > 0.6. v4 retitles the paper around marginal-vs-conditional coverage but keeps O1–O6. The code's O1–O6 evaluate component efficacy, not the v4 claim. | Adopt RO1–RO5 (§7) and retire the rest |
| 2 | Research questions and hypotheses | **MAJ** | v4 hypotheses H2 (Mondrian) and H3 (EnbPI) cite experiments whose code is absent (E8) | Adopt RQ1–RQ5 (§8); pre-specify regime groups and success criteria before the final run |
| 3 | Data sources | **NI** | Weather, drought, USDA and NOAA sources are documented and validated. Soil source and depth unverified (SSURGO vs SoilGrids-like ~250 m cells); NLCD epoch undocumented; slope units undocumented; ArcGIS extraction manual. | Document or re-derive static layers (§10.2) |
| 4 | Data acquisition | **OK / NI** | PRISM, NLDAS-2 (14,244/14,244 days, 3,998 backfilled hours with manifests), NOAA and NASS are scripted. ArcGIS static layers are not scripted. | Script or archive the ArcGIS workflow with parameters |
| 5 | Preprocessing | **OK** | NLDAS 24/24-hour rule; FAO-56 ET0 validated; soil dilution corrected by texture closure | Describe the soil correction as a data-quality fix with its validation |
| 6 | Spatial alignment | **OK** | Fractional-area county weights x cos(lat); DEM validated against NASA GSFC (r = 0.995) | — |
| 7 | Temporal alignment | **NI** | PRISM day ends 12 UTC; NLDAS day is 00–23 UTC — a documented half-day offset inside the SPEI water balance, never sensitivity-tested. Growing season fixed at 1 Apr–31 Oct; GDD phenology starts 1 Apr regardless of planting date. | Offset sensitivity (shift NLDAS by 12 h) or a stated limitation; validate stage windows with NASS Crop Progress (state level) |
| 8 | Missing-value handling | **NI** | Target never imputed (good). Lag fallback makes `Yield_lag1/2` a single constant on 100 % of LOSO CAL and TEST rows and on DEV-2015. `Hist_Normal_*` falls back to a global constant for unseen counties. | Drop lags in LOSO or declare them structurally unavailable; report the constant-feature count per partition |
| 9 | Outlier handling | **OK** | Analysed and reported; no deletion | — |
| 10 | Feature engineering | **NI** | `Fourier_sin_P1_0` / `Fourier_cos_P1_0` are mathematically constant yet selected. "Year_Type" is a **county-level** CDHW-count class (0 / 1–2 / ≥3), not a year type. The phenology window is near-degenerate. | Remove P = 1; rename "Year_Type" to "CDHW exposure class"; report class composition |
| 11 | Feature selection | **NI** | FIT-only 4-method consensus (good), but `Fourier_*`, `Yield_lag*`, the CDHW family and weather columns are **hard-protected** and bypass the vote. That is how constants survive. | Protect only a documented minimal set; drop zero-variance columns before selection |
| 12 | Data fusion | **OK** | 13 joins, one-to-one; merge audit and 31/31 QC pass | — |
| 13 | Target construction | **NI** | Linear trend fitted on 1985–2013 (0.109 t/ha/yr) is below the 1985–2023 trend (0.135). State trends range 0.069–0.157. In LOSO the residual bias slope tracks the trend mismatch (r = 0.96) **[M]**. | Trend-form sensitivity (linear vs quadratic or smooth, fitted on FIT); in the temporal split, state-level FIT trends; in LOSO, report as the transfer assumption |
| 14 | Train / validation / test separation | **OK** | FIT 1985–2013 / DEV 2014–15 / CAL 2016–18 / TEST 2019–23 with zero row overlap; LOSO fit / dev / cal / test roles | — |
| 15 | Spatial leakage | **OK** | Held-out state absent from every learned component (asserted) | — |
| 16 | Temporal leakage | **NI** | Index calibration on FIT years (good). But: static NLCD (epoch ≥ 2001) applied to 1985–2000; rolling features include the current season (valid only as end-of-season estimation); ACI updates on realised test outcomes (valid only as an online protocol). | Disclose all three; frame the task as end-of-season estimation |
| 17 | Group leakage | **OK** | County-level quantities fitted on FIT only; LOSO groups by state | — |
| 18 | Feature leakage | **OK** | No same-year yield-derived inputs; USDA acreage excluded | — |
| 19 | Scaling leakage | **OK** | RobustScaler fitted on FIT, per fold | — |
| 20 | Hyperparameter-tuning leakage | **MAJ** | `config.py` lines 196–257 record six rounds of LOSO hyperparameter and architecture changes, "each validated against a real full-pipeline run", chosen by LOSO R² (0.429 → 0.268 → 0.223 → 0.289 → 0.35 → "~0.55"). The "R² improvement patch" block made design changes after observing test R². This is selection on held-out outcomes across the project history. | Disclose the development history; re-derive every tuned setting on DEV only; freeze the protocol in writing before one final multi-seed run |
| 21 | Calibration leakage | **OK** | Temporal CAL is disjoint from fitting and selection. LOSO CAL (2016–2023, development states) is disjoint from the held-out state — but see §12 on same-year dependence. | — |
| 22 | Model-selection leakage | **NI** | Within a run, the ensemble weight is DEV-only (good). Across runs, the headline method and feature exclusions were chosen after seeing test results (garden of forking paths). | As #20; report every method evaluated, not only the winner |
| 23 | Cross-validation strategy | **MAJ** | One temporal origin (5 test years) carries the central claim | Rolling-origin evaluation (§17 C2) |
| 24 | LOSO evaluation | **OK / NI** | Correct after the detrend fix; single seed; static CP only; eras not separated | §12 |
| 25 | Random-split evaluation | **OK**, if labelled | R² 0.938 vs temporal 0.703 — an optimistic reference, as expected (Roberts 2017; Ploton 2020) | Label as non-independent upper bound |
| 26 | Baseline models | **MAJ** | No naive or fixed-effects panel baseline in the pipeline; no group-conditional, rolling-split or modern online CP baseline | §13 |
| 27 | Ablation studies | **NI** | Single seed; CDHW sign conflicts with O1; confounded ladder | §14 |
| 28 | Statistical significance | **MAJ** | Wilcoxon on 2,345 dependent rows (p = 0.0); d ≈ −0.07; cluster bootstrap with 5 year-clusters; O5 inference flips with cluster choice | Year as the unit, with ≥ 12 years from rolling origin; few-cluster-robust methods (Cameron et al. 2008); no row-level p-values |
| 29 | Error analysis | **NI** | Residual choropleth exists; bias vs width decomposition is not in the pipeline (done in this audit) | Add per-year and per-state bias, width and coverage tables |
| 30 | Robustness analysis | **NI** | Seeds exist for the temporal backbone only; LOSO robustness flag off; no trend-form, threshold, nominal-level or window sensitivity | §18 I1, I2, I5 |
| 31 | Generalisation | **NI** | Six rainfed Corn Belt states; irrigated pockets within them uncontrolled (Troy 2015) | Irrigation-share sensitivity; scope language |
| 32 | Explainability | **NI** | SHAP dominated by `county_baseline` (0.215, 4x the next feature); "strong agreement" at similarity 0.60; SHAP on the detrended anomaly; non-uniqueness of predictor sets (Lischeid 2022) | SHAP without `county_baseline`; seed stability; drop the label |
| 33 | Reproducibility | **MAJ** | `models/` empty; registry has no LOSO rows; methodology cites nine non-existent files; ArcGIS manual; FINAL_METHODOLOGY_AUDIT written outside the output folder. Positive: temporal results reproduce to 4 d.p. across runs **[M]**. | §17 C1, C8, C9 |
| 34 | Computational requirements | **OK** | ~55 min on a Colab T4; benchmark stored in `evaluation_report.json` | Write the benchmark CSV the traceability matrix expects |
| 35 | Sensitivity analysis | **MAJ** | None for CDHW thresholds (Tmax 35 °C, SPEI −1), stage GDD cut-offs (700 / 1400), stratum cut-points, nominal level, calibration window, ACI γ, trend form | §18 I5 plus the trend and window items |

### 10.2 Critical implementation finding — SA-ACI narrows intervals where stress is highest

`aci_calibrator.py::severity_aware_adaptive_conformal` (lines 389–430):
- selects the calibration window from the **test year's** mean CDHW severity (2, 3 or 5 years);
- multiplies calibration scores by `1 + 0.05·log1p(severity)`;
- then **divides** each test row's threshold by its own severity weight: `eff_threshold = current_threshold / this_weights`.

Higher severity therefore produces a **smaller** conformal margin. The docstring presents this as "sharpness preservation".

**Measured on the current run [M]:**
- For rows with CDHW severity > 0, the weight has median 1.060, 90th percentile 1.125 and maximum 1.234. The margin is up to ~19 % smaller.
- SA-ACI intervals are only **5.7 %** wider in the Extreme class than in Normal. The other methods widen by 12.4 % (standard ACI), 14.6 % (static) and 26.4 % (locally adaptive).

**Consequences:**
1. SA-ACI's worst-regime coverage (0.807) is **partly by construction**, not only evidence about adaptive conformal inference.
2. `Paper3_Methodology_Updated_v2.md` §4.3 states that severity weighting was found to narrow intervals, was disabled (`cfg.ACI_SEVERITY_WEIGHTING = False`), and that the dynamic window was removed. **Neither the flag nor the removal exists in the current code [M].** Standard ACI contains no severity terms.
3. O6 currently ranks SA-ACI first. The paper's "best" marginal method is the one engineered to shrink intervals under stress.

**Required:** remove SA-ACI from the headline, or reverse the direction and re-evaluate. Build the G7 claim on standard ACI vs static vs a group-conditional method. Standard ACI still shows the inversion (marginal rank 2, worst-stratum rank 5), so the finding survives, but on sound ground.

### 10.3 Data-source audit (Phase 11)

| Dataset | Purpose | Spatial resolution | Temporal resolution | Years | Processing | Potential leakage / validity risk | Necessary? | Justification |
|---|---|---|---|---|---|---|---|---|
| USDA NASS county corn yield | Target | County | Annual | 1985–2023 | bu/acre x 0.0627 → t/ha; 20,507 observed; never imputed | Survey revisions; non-random county omissions | **Yes** | Target |
| PRISM daily tmax, tmin, ppt | Temperature, precipitation, GDD, KDD30, heat days, SPI, CDHW | 4 km | Daily | 1985–2023 | Area-weighted county means | Station-network changes affect long-term homogeneity; confirm which PRISM series (time-series vs long-term) was used | **Yes** | Best US gridded daily temperature and precipitation (Daly 2008) |
| NLDAS-2 forcing (FORA0125_H v2.0) | Radiation, humidity, wind, pressure → FAO-56 ET0; VPD | 0.125° | Hourly → daily (UTC) | 1985–2023 | 24/24-hour rule; backfilled corrupt granules | Half-day offset vs PRISM day | **Yes** | ET0 needed for SPEI (Xia 2012) |
| SPI-1/3/6/12 (PRISM) | Drought features | County | Daily → growing season | 1985–2023 | Gamma with zero mass, calibrated on 1985–2013 | Weather only; in LOSO each county's parameters use its own climate (not target) — disclose | **Partly** | Highly collinear with SPEI and precipitation; keep one or two scales |
| SPEI-30/90 (PRISM − NLDAS ET0) | Drought; CDHW component | County | Daily → season / stage | 1985–2023 | GLO via PWM (Beguería 2014), calibrated on 1985–2013; validated vs USDM DSCI (r = −0.758) | As SPI | **Yes** | Required by the CDHW definition |
| CDHW (SPEI-30 < −1 AND Tmax > 35 °C) | Compound exposure; regime strata | County | Daily → stage | 1985–2023 | GDD 700/1400 windows from 1 Apr | Thresholds fixed a priori (good); phenology fixed-date | **Yes** (core to the regime definition) | Literature-supported construct (Zscheischler 2020; Haqiqi 2021) |
| NOAA Storm Events (hail, tornado, thunderstorm wind) | Severe-weather counts | County / event | Event → growing season | 1985–2023 | Recounted and validated | **Reporting density rises over time** (documented in the master report), so the counts carry a reporting trend that can proxy the yield trend | **Questionable** | Justify with a with/without sensitivity, or drop |
| Soil (AWC, BD, clay, silt, OC; ArcGIS) | Static soil context | ~250 m cells (source unverified) | Static | — | Texture-closure dilution correction; 5 counties set missing | Provenance, depth and units undocumented | **Yes**, if documented | Soil explains spatial yield potential; essential for LOSO transfer |
| DEM elevation / slope | Static terrain | n.s. | Static | — | Elevation validated (r = 0.995); slope units undocumented | Low | **Marginal** | Keep elevation; document slope or drop it |
| NLCD grouped land-cover fractions (7) | Static land use | 30 m | Static, epoch ≥ 2001 undocumented | — | Class grouping | **Anachronistic for 1985–2000**; compositional (sums to 1) | **Weak** | Keep cropland fraction at most, with epoch documented |
| ONI (NOAA CPC) | ENSO stratification (evaluation only) | Global index | Monthly | 1985–2023 | Evaluation only | None | **Yes** | Stratification (Iizumi 2014; Anderson 2019) |
| USDM DSCI | SPEI validation only | County | Weekly | 2000s–2023 | FULL only | None | **Yes** (validation) | External check of drought index |
| ERA5 (legacy dataset) | Superseded | ~31 km | Hourly | 1985–2023 | — | — | **No** | Validation comparison only (Hersbach 2020) |

**Redundant:** SPI at four scales + SPEI-30/90 + precipitation + water balance (a collinear drought block); seven compositional NLCD fractions; 2- and 3-year rolling means of the same variables; Fourier P = 1.

**Missing but important:**
- **Irrigation share** (USDA Census of Agriculture, carry the last census forward) — irrigation decouples yield from climate (Troy 2015), and Nebraska was excluded for exactly this reason.
- **Planting and silking dates** (NASS Crop Progress, state level) — to validate or anchor the fixed 1 April phenology.

**Not needed:** NDVI (it changes the paper into a remote-sensing study), fertilizer (the USGS county series ends in 2017), DSSAT.

---

## 11. Data-leakage audit

| Leakage type | Status | Evidence | Risk to conclusions |
|---|---|---|---|
| Train–test row overlap | **None detected** | Row-identity ledger asserted before fitting; 7 experiments PASS **[M]** | — |
| Preprocessing, scaling, feature selection | **None detected** | All fitted on FIT (temporal) or the fold's FIT (LOSO) | — |
| Target detrending | **None detected** | FIT-only trend; Year known at prediction time; LOSO ledger binds `target_detrending → fit_full` | — |
| Yield-derived features | **None detected** | Lags use strictly earlier years; `county_baseline` from FIT; excluded from LOSO | — |
| Calibration reuse | **None detected (temporal)** | CAL 2016–18 disjoint from fitting, early stopping and ensemble weighting | — |
| Drought index calibration | **Low** | Parameters from 1985–2013 only; per-county fitting uses the county's own climate (weather, not target) | Disclose for LOSO |
| Anachronistic static layers | **Low–moderate** | NLCD epoch ≥ 2001 applied to 1985–2000 | Time-invariant; could leak post-2001 land-use structure into early years |
| Reporting-trend proxies | **Moderate** | NOAA storm counts rise with reporting density over time | Can act as a trend proxy in the temporal split |
| Online use of test outcomes (ACI, SA-ACI) | **By design; must be disclosed** | Year-t interval uses realised outcomes of years < t | Unequal information vs static CP |
| Test-year covariates in calibration (SA-ACI) | **Design issue** | Window size and per-row margin depend on test-year severity | Not target leakage; distorts the comparison (§10.2) |
| **Hyperparameter / design selection on held-out results** | **Present in project history** | `config.py` LOSO rounds 1–6 chosen by LOSO R²; "R² improvement patch" | **Main residual leakage risk.** Held-out estimates may be optimistic. |
| Same-year cross-state dependence in LOSO calibration | **Present** | LOSO CAL = 2016–2023 of neighbouring states; held-out rows of the same years share weather shocks | Coverage for 2016–2023 held-out rows may be optimistic (§12) |
| Random split | **Structural (by design)** | Future years and same counties in training; R² 0.938 | Report only as an optimistic reference |

---

## 12. LOSO audit

| Requirement | Verdict | Evidence |
|---|---|---|
| Exact states | **Correct** | Illinois, Indiana, Iowa, Minnesota, Missouri, Ohio. The master dataset contains exactly these (583 counties); `DROP_STATES = ["Nebraska"]` is inert for this data. |
| Training uses only the remaining states | **Correct** | Asserted per fold before fitting |
| Held-out state excluded from preprocessing, selection, scaling, detrending, model and ensemble selection, calibration | **Correct** | Per-fold `TrainFittedPreprocessor`, `select_features`, `fit_scaler`, FIT-only trend; weight chosen on DEV (development states); CAL from development states |
| Scalers, encoders, imputers fitted on training data | **Correct** | Verified at source and by reconstruction (Minnesota DEV R² reproduced exactly) |
| Temporal information leaking across states | **Partially present** | CAL (2016–2023) shares calendar years with 2016–2023 held-out rows. Weather shocks are spatially correlated, so calibration residuals and test residuals are dependent. |
| Spatial interpolation or kriging leakage | **None of concern** | PRISM and NLDAS are gridded from multi-state station networks, but they are covariates, not targets |
| External datasets built with held-out-state information | **Covariates only** | Gridded weather and static layers; no yield information |
| Truly out-of-sample test results | **Yes, within a run** | — but see the tuning history (§10.1 #20) |
| LOSO correct, so do not change it for score | **Agreed** | The detrend fix aligned LOSO with the pre-registered temporal protocol (`DETREND_TARGET = True`) and was justified on DEV evidence, not on test R² |

**Required additions (reporting, not redesign):**
1. **Report held-out results by era:** 1985–2013 (in-period, spatial shift only), 2014–2015, and 2016–2023 (spatial + temporal shift; same-year calibration dependence). Current per-era coverage **[M]**: Illinois 2014–2023 = 0.736, Iowa 2014–2023 = 0.782, Minnesota 1985–1999 = 0.786.
2. **Evaluate every calibrator in LOSO**, not only static CP. The conditional-coverage claim is currently untested under spatial shift for five of six methods.
3. **Multiple seeds** (`ENABLE_LOSO_ROBUSTNESS_CHECK`).
4. **State the transfer assumption** (five-state trend applied to the held-out state) and report the trend-mismatch diagnostic (r = 0.96).
5. **Do not report the per-fold ensemble weight as a finding** without the tuning-history disclosure.

---

## 13. Baseline audit

| Baseline | Currently present? | Scientifically necessary? | Why | Note |
|---|---|---|---|---|
| **Trend + county-mean anomaly** (temporal); **five-state trend** (LOSO) | **No** | **Essential** | Shows the ML adds skill beyond trend and location. **Measured [M]:** temporal R² 0.415 (RMSE 1.296) vs ensemble 0.703; LOSO trend-only macro 0.276 vs model 0.698. **The pre-fix LOSO model (0.357) was barely above trend-only.** | Cheap; closed form |
| **Fixed-effects degree-day panel** (GDD, KDD > 29 °C, precipitation and its square, county FE, FIT-only trend) | **No** | **Essential** for agricultural and climate journals | Canonical US yield–weather model (Schlenker & Roberts 2009; Lobell & Burke 2010). Conformalised, it also tests whether the calibration findings depend on the backbone. | Linear; seconds to fit |
| LightGBM (point + quantile) | Yes | **Yes** | **Best point model on test (R² 0.731)** — must be headlined honestly | — |
| NeuralCQR (residual MLP) | Yes | **Yes, as the joint-training vehicle** | Required for RQ5, not for accuracy | Do not present as the best model |
| CatBoost, XGBoost | Yes | Supplementary | Redundant with LightGBM for tabular county-year data (Kang 2020) | Move to supplement |
| Split conformal (static) | Yes | **Yes** | Reference calibrator (Lei et al. 2018) | — |
| **Rolling-window split conformal** (refreshed yearly) | **No** | **Essential** | Removes the information asymmetry against ACI; isolates adaptivity from recency | Trivial to add |
| ACI (standard) | Yes | **Yes** | Method under test (Gibbs & Candès 2021) | — |
| **DtACI or AgACI** | **No** | **Essential** for a conformal-methods comparison | Current online variants (Gibbs & Candès 2024; Zaffran et al. 2022); removes γ sensitivity | Choose one |
| **Group-conditional CP** (Mondrian by pre-specified regimes; or Gibbs et al. 2025) | **No** | **Essential** | The direct remedy for the gap the paper claims | Pre-specify groups on FIT/CAL; report group sizes |
| Weighted CP | Yes | Optional | Density-ratio weights need justification; ≈ static here | Keep only with a documented weight model |
| Locally adaptive CP | Yes | **Yes** | Best worst-stratum coverage | — |
| Phenology-stratified CQR | Yes | **No** | Numerically identical to static | Drop, or report as a degenerate Mondrian case |
| SA-ACI | Yes | **No, in current form** | Narrows intervals under stress by construction (§10.2) | Remove or fix and re-justify |
| LSTM / Transformer / GNN | No | **No** | Tabular county-year inputs; DL not advantageous (Kang 2020); backbone choice is not the question | Do not add |
| Bayesian NN / deep ensemble | No | Optional | Non-conformal UQ contrast (Ma 2021; Lakshminarayanan 2017) | Only if space allows |
| Spatial CP (GeoCP / local spatial CP) | No | Optional | LOSO comparator (Lou 2025; Mao 2024) | — |

---

## 14. Ablation audit

### 14.1 What the current 5-row ladder isolates

| Step | Change | Confounded? | Problem |
|---|---|---|---|
| 1 → 2 | + CDHW family | **Yes** | Correlated univariate heat and drought features remain (the CDHW family's strongest correlate is heat-day count; legacy analysis r = 0.87), so this measures *incremental* value given near-duplicates. Single seed. Sign conflicts with O1 (RMSE 0.9511 → 0.9629 here; O1 has 0.9701 → 0.9529). |
| 2 → 3 | + post-hoc quantile head, static CP | No | Point predictions identical by design. The distinctness validator wrongly expects them to differ (M12 FAIL). |
| 3 → 4 | Static → ACI | No | Uses standard ACI while the main results headline SA-ACI |
| 4 → 5 | Post-hoc → joint training | **Partly** | Changes the training paradigm while ACI is on; O4 does it cleanly with both calibrators |
| All | — | — | Single seed; single test window; the ablation trains on FIT without the FIT+DEV refit used by the main pipeline, so its numbers are not the headline numbers |

**Missing ablations:** detrending on/off and trend form; yield lags on/off (critical for LOSO); static layers on/off; NOAA storm features on/off; calibration window length; CDHW thresholds; stage cut-offs.

**Redundant:** phenology-stratified CQR (≡ static); rows 3 and 4 reported as architecture steps when they are calibration choices.

### 14.2 Minimal publication-quality ablation matrix

Each block changes one factor, uses ≥ 5 seeds, and is reported on the temporal rolling origin and on LOSO with year- or state-level uncertainty.

| Block | Question | Arms (everything else fixed) | Primary metrics |
|---|---|---|---|
| **A — point skill** | RQ1 | Trend + county mean · FE degree-day panel · LightGBM · NeuralCQR · ensemble | R², RMSE, MAE; skill over trend baseline |
| **B — feature families** | RQ5 | Full · − CDHW family · − CDHW and − univariate heat / drought · − static layers · − yield lags · − storm counts | ΔRMSE and Δ worst-stratum coverage vs full |
| **C — calibrators** | RQ2, RQ4 | Split · rolling split · locally adaptive · ACI · DtACI/AgACI · group-conditional — on the best point model **and** the FE panel | Marginal PICP / ACE / MPIW / Winkler **and** worst-stratum PICP with composition |
| **D — training paradigm** | RQ5 | Post-hoc vs joint (NeuralCQR) | MPIW, Winkler, conditional coverage; RMSE as a check |
| **E — sensitivity** (not ablation) | Robustness | Trend form; CDHW thresholds (Tmax 32/35/38 °C; SPEI −1/−1.5); stratum cut-points; nominal level 0.80/0.90/0.95; calibration window | Whether the ranking in C is stable |

---

## 15. Results and evaluation audit

All numbers are from the 2026-09-16 run or recomputed from it **[M]**. No legacy numbers are used.

| Result | Value | Appropriate metric? | Fair comparison? | Meaningful? | Reviewer reading |
|---|---|---|---|---|---|
| Temporal ensemble (2019–2023) | R² 0.703, RMSE 0.924 t/ha; county-block CI 0.677–0.727; year-block CI 0.648–0.746 | Yes | Yes | Yes vs naive (0.415) | Solid |
| Best point model | **LightGBM alone R² 0.731, RMSE 0.879** > ensemble | Yes | Yes | Ensemble adds nothing on test | Must be stated; weakens the NeuralCQR framing |
| Random split | LightGBM R² 0.938 | As a reference only | By design no | Shows optimism of random CV | Report only as an upper bound |
| LOSO (after fix) | Macro R² 0.698 [0.613, 0.782]; RMSE 1.208; PICP 0.916; MPIW 4.25 | Yes | Yes | Yes vs trend-only 0.276 | Good; **Minnesota 0.615 / Missouri 0.608; PICP Minnesota 0.845 and Illinois 0.886 below nominal** — show per fold |
| LOSO before fix | Macro R² 0.357; MPIW 8.11 (72 % conformal correction) | — | — | — | Report as the detrending lesson (§9.1 #3) |
| Conformal, marginal | PICP 0.919–0.949; Winkler spread 2.2 %; Friedman p = 0.52 (5 year-blocks) | Yes | Unequal information (ACI family online) | **Differences not significant** | "Descriptive ranking only" is correct |
| Conformal, conditional | Worst-stratum PICP 0.807 (SA-ACI) to 0.891 (locally adaptive); SA-ACI lowest in 7/7 axes | Yes | Confounded (stratum = 2022–23 × Missouri/Iowa); SA-ACI defect | Potentially | Replicate with rolling origin; exclude or fix SA-ACI |
| Worst year | 2021 for **all** six methods (0.849–0.893) | Yes | Yes | Yes | Location shock, not width |
| Bias vs width (LOSO, 234 state-years) | r(\|bias\|, coverage) = −0.77; r(width, coverage) = 0.00 | Yes (descriptive) | Yes | Yes | Strong mechanistic evidence; state-years are not independent, so no p-value |
| O1 (CDHW) | ΔR² +0.0116, single seed; ablation sign reversed | Yes | Not replicated | **No** | Inconclusive |
| O4 (joint vs post-hoc) | RMSE equal; MPIW −19.4 %; Winkler −15.6 %; better in 5/5 years; PICP 0.954 vs 0.965 | Yes | Yes | Plausibly | Needs ≥ 5 seeds |
| O5 (width vs severity) | r = 0.166; R² 0.028; p = 0.116 (year-clustered) | Yes | — | **No** | Weak or inconclusive |
| SHAP | Similarity 0.60; `county_baseline` 0.215 (top) | Partially | — | Limited | Recompute without `county_baseline` |
| Significance tests | Row-level Wilcoxon p = 0.0 | **No** | — | — | Remove; use year-level inference |
| Exchangeability diagnostics | Lag-1 ACF from 5 yearly means; conflicting spatial flags | **No** | — | — | Recompute on county series and a residual panel |

**Hidden or under-reported weak results:**
- LightGBM beats the ensemble on test.
- Phenology-stratified CQR ≡ static.
- SA-ACI's design defect.
- Two LOSO folds below nominal coverage.
- Extreme-class composition (Ohio 1 row, Indiana 5).
- Invalid row-level p-values.
- O1 contradiction.

**Overfitting:** no sign of it — temporal DEV R² (0.61) is *below* TEST (0.70). The random vs temporal gap reflects distribution shift and structural leakage in random CV, not overfitting.

**Does the claimed contribution survive strict out-of-domain evaluation?** The point-skill result does (LOSO 0.698 vs trend-only 0.276). The conditional-coverage claim is **untested under LOSO** for five of six calibrators, and it rests on one five-year temporal window.

---

## 16. Top reviewer concerns

Ordered by how likely they are to drive a reject or major-revision decision.

1. **"The manuscript does not describe the study that was run."** The methodology cites ERA5, PhenoFormer-era datasets and nine non-existent evidence files, and reports numbers from superseded runs. Code, data and document disagree (E7, E8, E10).
2. **"The novelty is a combination of existing conformal methods."** CQR, ACI and weighted/local CP are standard. Group-conditional guarantees already exist (Gibbs et al. 2025). A non-peer-reviewed 2026 preprint already applies split CP + ACI to crop yield under climate variability.
3. **"One five-year test window cannot support a method-ranking claim."** The Extreme class is two years and two states; all methods fail in the same year.
4. **"The statistics are invalid."** Wilcoxon on 2,345 dependent rows gives p = 0.0; the cluster bootstrap uses five clusters; the O5 conclusion flips with the clustering unit.
5. **"The headline method is built to fail in extremes."** SA-ACI divides its margin by a severity weight (§10.2), and the methodology text says this was disabled.
6. **"Where are the obvious baselines?"** No trend-only, no fixed-effects degree-day panel, no rolling split CP, no DtACI/AgACI, no group-conditional CP.
7. **"The inversion is partly mechanical."** Methods are compared at different marginal coverage levels, and the ACI family receives online test feedback that static CP does not.
8. **"Hyperparameters and design were tuned on held-out results."** The config history shows six LOSO rounds selected by LOSO R².
9. **"The literature ignores the US maize yield–weather canon and US Corn Belt ML."** It also mis-describes Farag 2025 and over-states Mahmood 2026.
10. **"Compound drought–heat is not a gap"** (Haqiqi 2021), and O1 is contradicted by the project's own ablation.
11. **"Detrending and trend transfer drive LOSO."** A linear 1985–2013 trend under-predicts later years, and state trends differ from 0.069 to 0.157 t/ha/yr. This is unaddressed beyond a code comment.
12. **"Data provenance is incomplete."** Soil source and depth unknown, NLCD epoch unknown, slope units unknown, manual ArcGIS extraction, NOAA reporting inhomogeneity.
13. **"Irrigation is uncontrolled within the six 'rainfed' states,"** while Nebraska was excluded for irrigation.
14. **"Feature pathologies"**: constant Fourier P = 1 features and constant yield lags in LOSO test, both protected into the model; the phenology stratification is degenerate.
15. **"Scope and terminology."** "Year_Type" is a county-level class; season-total features mean end-of-season estimation, yet the framing speaks of decision support during extremes; "generalisation" rests on six states.

---

## 17. Critical required changes (must fix before submission)

| ID | Problem | Why it matters | Exact action | Expected scientific benefit | New data? | Retraining? | Changes core methodology? | Priority |
|---|---|---|---|---|---|---|---|---|
| **C1** | Three incompatible study descriptions; legacy numbers; phantom evidence files | Automatic major revision or desk reject; irreproducible | Write one methodology from the current code. Delete or archive every legacy number and every reference to missing files. Replace PhenoFormer, ERA5, NDVI and DSSAT text. Adopt RO1–RO5 and RQ1–RQ5. | A coherent, auditable paper | No | No | No (documentation) | 1 |
| **C2** | Central claim rests on one 5-year test window | Ranking claims need replication across many years | **Rolling-origin temporal evaluation**: for each test year Y from ~2008 to 2023, fit on years ≤ Y−5, DEV Y−4…Y−3, CAL Y−2…Y−1 (or equivalent), test Y. Report per-year and pooled marginal and conditional coverage. Keep the current 2019–2023 split as the primary locked test. | 12–16 independent year-level replicates; fixes 2022–23 × Missouri/Iowa confounding | No | **Yes** (compute-heavy; LightGBM + static-family CP are cheap) | No (extends evaluation) | 1 |
| **C3** | Invalid inference | Reviewers reject p = 0.0 on dependent rows | Use **year as the unit of analysis**, with C2 years. Use paired permutation or sign tests on year-level metrics; few-cluster-robust intervals (Cameron et al. 2008); drop row-level Wilcoxon and t-tests. Report LOSO fold results as descriptive (n = 6). | Defensible uncertainty on every comparison | No | No (recompute from predictions) | No | 1 |
| **C4** | Missing essential baselines | Cannot claim ML or calibration conclusions without them | Add (a) trend + county mean and five-state trend; (b) FE degree-day panel, conformalised; (c) rolling-window split CP; (d) DtACI or AgACI; (e) group-conditional CP with **pre-specified** regime groups (Mondrian or Gibbs et al. 2025). | Answers RQ1 and RQ4; neutralises concerns 2, 6, 7 | No | Minor | Adds comparators; the core stays | 1 |
| **C5** | SA-ACI defect; unequal comparison conditions | The inversion could be an artefact | Remove SA-ACI from the headline, or reverse the severity direction and re-validate. Give static CP a rolling-refresh counterpart. Report coverage–width frontiers and the **marginal-minus-worst gap**, plus comparisons at matched marginal coverage (α tuned on CAL, never on test). | Makes the G7 finding robust to "it is just slack or a bug" | No | No (calibration only) | No | 1 |
| **C6** | Literature incomplete and partly wrong | Novelty and gaps are judged against the literature | Integrate §2 (P1 items at minimum). Fix E1–E10. Rewrite gaps as in §4 and novelty as in §6 (versions A/B). Disclose Namdeo & Baraskar 2026 as grey literature. Narrow the Barber et al. 2021 impossibility sentence. | Survives a statistics and an agronomy reviewer | No | No | No | 1 |
| **C7** | Tuning and design choices informed by held-out results | Held-out estimates may be optimistic | Write a **frozen protocol** (features, trend form, hyperparameters, calibrators, regime groups, metrics, tests) before the final run. Re-derive tuned settings on DEV only. Run once with ≥ 5 seeds. Disclose the development history in the supplement. | Credible out-of-sample claims | No | **Yes** (one final run) | No | 1 |
| **C8** | Reporting defects | Visible errors undermine trust | Fix the NaN LOSO weight table (`evaluation.py:915–921` key names); register LOSO experiments; rename the "100 % compliance" label (`utils.py:383`); fix M12 (`main.py:2039`) and M14 (`main.py:21`); save model weights and run configuration; write FINAL_METHODOLOGY_AUDIT inside the output folder. | Reproducible, self-consistent artefacts | No | No | No | 2 |
| **C9** | Static-layer provenance | Reviewers and replicators need sources, depths, epochs, units | Document the soil product, depth interval and units; NLCD epoch; slope units; PRISM series; script or archive the ArcGIS steps. Add a NOAA reporting-inhomogeneity statement and a with/without storm sensitivity. | Reproducibility; removes a trend proxy | No (documentation); possibly re-extraction | Only if storms are dropped | No | 2 |

---

## 18. Important recommended changes

| ID | Problem | Why it matters | Exact action | Benefit | New data? | Retraining? | Changes core? | Priority |
|---|---|---|---|---|---|---|---|---|
| I1 | Single seed (LOSO, O4, ablations) | Seed variance can exceed method differences | ≥ 5 seeds; report mean ± SD; enable `ENABLE_LOSO_ROBUSTNESS_CHECK` | Stable conclusions | No | Yes | No | High |
| I2 | Trend misspecification and transfer | Drives LOSO skill and coverage; late-period under-prediction | FIT-only linear vs quadratic or smooth trend sensitivity; state-level trends in the temporal split; LOSO results by era | Quantifies the main error source | No | Yes | No | High |
| I3 | Irrigation uncontrolled | Confounds drought and heat response | County irrigated share (Census of Agriculture, last census carried forward): covariate or exclusion sensitivity | Answers concern 13 | **Yes** (public) | Yes | No | High |
| I4 | Fixed-date phenology; degenerate phenology stratification | CDHW stage windows may be misplaced | Validate GDD 700/1400 against NASS Crop Progress silking (state level), or anchor GDD at state-year planting dates; drop phenology-stratified CQR | Credible stage attribution | **Yes** (public) | Yes | Minor | Medium |
| I5 | No threshold or definition sensitivity | Regime membership depends on arbitrary cut-offs | Tmax 32/35/38 °C; SPEI −1/−1.5; stratum cut-points; nominal 0.80/0.90/0.95 | Robust regime conclusions | No | Calibration only | No | Medium |
| I6 | Constant and redundant features | Wasted protection; misleading feature counts | Drop Fourier P = 1; handle lags in LOSO explicitly; thin the collinear drought block | Cleaner model; honest counts | No | Yes | No | Medium |
| I7 | "Year_Type" misnomer; stratum composition hidden | Misleads readers about what "extreme years" are | Rename to "CDHW exposure class"; add year-level regimes; always tabulate stratum × year × state | Transparent conditional results | No | No | No | Medium |
| I8 | SHAP dominated by `county_baseline`; "strong agreement" label | Over-interpretation | SHAP without `county_baseline`; seed stability; neutral wording | Honest explainability | No | No | No | Medium |
| I9 | LOSO evaluates static CP only | Spatial part of the claim untested | Run every calibrator in LOSO; report by era; note same-year calibration dependence | Tests G7 under spatial shift | No | Calibration only | No | High |
| I10 | Scope language | Over-claiming invites rejection | "End-of-season estimation"; "six rainfed U.S. Corn Belt states"; no "early warning" or "national generalisation" | Defensible framing | No | No | No | High |

**Optional (useful, not necessary):**
- Spatial CP comparator for LOSO (Lou et al. 2025; Mao et al. 2024).
- NLDAS-2 Noah soil-moisture features (Ortiz-Bobea 2019; Shahhosseini 2021).
- Check on the CY-Bench US maize subset (Kallenberg et al. 2026).
- Multi-α interval scores or a quantile-based CRPS.
- County-level coverage maps.

---

## 19. Updated literature study

Delivered as **`review_audit/LITERATURE_REVIEW_v3.md`**. It:
- adds the verified P1/P2 literature of §2;
- removes or downgrades weak entries (Fischer & Knutti 2015; L10 Poudel & Poudel 2023; C6 Onogi 2025; generic reviews);
- corrects E1–E6 and E9;
- builds each gap from cited limitations;
- connects the gaps to RO3 and RO4;
- states novelty at version A of §6;
- ends with the verified reference list and the registry verification log.

It was written **after** this audit, as instructed.

---

## 20. Final research-readiness assessment

### 20.1 Area classification (Phase 12)

| Area | Classification | Evidence |
|---|---|---|
| A. Literature completeness | **Major revision required** | 13 of the 15 applicable areas are absent, thin or incomplete (§1.1); errors E1–E10 (§1.4) |
| B. Research-gap strength | **Needs improvement** | Two gaps survive as partially supported (G7, G10); proposal gaps G1 and G3 are not gaps (§4) |
| C. Novelty | **Major revision required** | Claims exceed evidence; the defensible novelty is evaluation-level (§5, §6) |
| D. Methodological rigour | **Needs improvement** | Excellent leakage control and four-way split; weak inference, one test window, tuning history, SA-ACI defect |
| E. Data quality | **Needs improvement** | Weather and drought processing strong and validated; static-layer provenance and NOAA homogeneity gaps |
| F. Validation rigour | **Needs improvement** | Temporal + strict LOSO (good); single window; single seed; LOSO calibrators incomplete |
| G. Baseline quality | **Major revision required** | Naive, FE-panel, rolling-split, modern online and group-conditional baselines absent |
| H. Ablation quality | **Needs improvement** | Single seed, confounded CDHW step, missing detrend / lag / static ablations |
| I. Reproducibility | **Major revision required** | Phantom evidence files, no saved models, manual ArcGIS, incomplete registry; positive: 4-d.p. reproducibility across runs |
| J. Results strength | **Needs improvement** | Point skill vs naive is strong; conditional-coverage claim fragile until replicated |
| K. Scientific contribution | **Needs improvement** | Real but modest: ranking divergence + bias mechanism + LOSO detrending lesson |
| L. Practical contribution | **Needs improvement** | Plausible guidance, scoped to end-of-season estimation |

### 20.2 Chain-of-logic audit (Phase 16)

| Link | Status | Break found | Fix |
|---|---|---|---|
| Existing literature → known limitations | **Broken, now repairable** | Limitations were asserted from a fabricated source ("Paper 18") and from misread sources (Farag 2025) | §2, §4 |
| Known limitations → research gap | **Partly broken** | G1 is outdated (Haqiqi 2021); G2 is general-solved; G3 is unsupported | Use G7 and G10 |
| Research gap → research question | **Broken** | O1–O6 test component efficacy (CDHW, joint training, severity correlation, method rank), not the gap about calibrator selection under regimes and transfer | RQ1–RQ5 |
| Research question → objective | **Broken** | The v4 title and claim do not match the O1–O6 objectives | RO1–RO5 |
| Objective → methodology | **Partly broken** | Unnecessary components for the claim (SA-ACI as designed, phenology-stratified CQR, CatBoost/XGBoost in the main text); necessary components absent (group-conditional CP, rolling origin, naive and panel baselines) | C4, C5, §13 |
| Methodology → experiment | **Partly broken** | The pipeline computes conditional coverage for three methods on one axis; the full evidence came from this audit's script | Move `p8_review_evidence.py` logic into the pipeline |
| Experiment → evaluation | **Broken** | Invalid significance tests; LOSO calibrators incomplete | C3, I9 |
| Evaluation → result | **Partly supported** | Conditional ranking divergence replicated on new data; confounded composition; SA-ACI artefact | C2, C5 |
| Result → contribution | **Over-stated** | "Novel method", "ACI prevents collapse", "CDHW super-linear", "joint improves accuracy" are unsupported | §6 C list |
| Novelty claim vs evidence | **Stronger than evidence** | Gibbs 2025 and grey-literature overlap not acknowledged | §6 version A |
| Evaluation vs claimed generalisation | **Mismatch** | Spatial part of the conditional claim not tested for 5/6 calibrators | I9 |

### 20.3 Verdict

**Not publishable in the current form. Publishable after major revision as an evaluation study.** The blockers are specific:

1. The document state (C1).
2. Single-window, dependence-ignoring evidence for the central claim (C2, C3).
3. Missing baselines, especially group-conditional and rolling split CP (C4).
4. The SA-ACI design defect and unequal comparison conditions (C5).
5. An incomplete and partly incorrect literature base (C6).
6. Undisclosed held-out-informed tuning (C7).

**Minimum revision set for a credible submission:** C1–C7, plus I1 (seeds), I9 (all calibrators in LOSO) and I10 (scope language). C8, C9 and I2–I8 substantially reduce reviewer risk and are strongly advised.

### 20.4 What should remain unchanged

These parts are methodologically sound. Changing them to improve metrics would weaken the study.

- The **four-way temporal split** with calibration disjoint from model fitting, early stopping and ensemble weighting.
- **FIT-only fitting** of every learned transformation (imputation, feature selection, scaling, detrending, index calibration).
- The **row-identity provenance ledger** asserted before any model is trained.
- The **six-state LOSO protocol** with per-fold feature selection, scaling, FIT-only detrending and development-state calibration, including the documented Nebraska rationale. Add reporting (by era, all calibrators, seeds); do not change the protocol.
- The **PRISM / NLDAS-2 / SPEI construction** (24/24-hour rule, FAO-56 ET0, GLO-PWM SPEI calibrated on 1985–2013, USDM validation).
- **No new architectures.** The literature (Kang 2020) and this study's own results (LightGBM ≥ ensemble) both argue against it.
- **ENSO as evaluation-only stratification.**
- **End-of-season framing** of the prediction task.
- The honest negative-result reporting culture already visible in the claim-consistency and validator reports.

---

*Evidence scripts and outputs for this audit:*
- `code/diagnostics_loso/p8_review_evidence.py`
- `outputs_diagnostics/reports/p8_review_evidence.json`
- `outputs_diagnostics/reports/p8_conditional_coverage_current_run.csv`
- `outputs_diagnostics/LOSO_FORENSIC_DIAGNOSTIC.md`

*Literature verification log:* `review_audit/LITERATURE_REVIEW_v3.md` §7.
