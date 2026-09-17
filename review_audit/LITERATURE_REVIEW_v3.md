# Literature Review v3 (audited)
## Conformal calibration of county-level corn yield predictions under compound drought–heat and cross-state distribution shift

**Status.** This replaces `LITERATURE_REVIEW_CORRECTED.md` as the working review. It was written *after* the peer-review audit (`PAPER3_PEER_REVIEW_AUDIT.md`), following that audit's gap and novelty conclusions.

**Verification.** Every reference in §8 was resolved on 2026-09-16 against Crossref, OpenAlex, Semantic Scholar or the arXiv API, or was carried over from earlier registry-verified documents (marked). Statements attributed to papers are limited to what their abstracts or established method definitions support. Where only an abstract was checked, this is noted in the verification log.

---

## 1. Scope and approach

The review serves one research aim: **to evaluate whether conformal prediction intervals for county-level corn yield remain valid under climate-driven and cross-state distribution shift, both marginally and within compound drought–heat regimes, and to diagnose why they fail when they do.**

Five literatures bear on that aim:
1. yield–weather relationships and compound drought–heat stress in the U.S. Corn Belt (§2);
2. machine-learning prediction of county yields and how it is validated (§3);
3. uncertainty quantification for crop yield (§4);
4. conformal prediction under shift, dependence and conditional targets (§5);
5. the evaluation of probabilistic predictions (§6).

§7 synthesises the state of the art, the limitations and the gaps. Papers were included when they establish a phenomenon, a method or a validation practice this study depends on, or are the closest comparators to its contribution. Breadth for its own sake was not a criterion.

---

## 2. Yield–weather relationships and compound drought–heat stress

### 2.1 Nonlinear temperature and water-stress effects on U.S. maize

**Temperature response.** County-panel evidence shows a strongly nonlinear temperature response for U.S. corn: yields increase with temperature up to about 29 °C and decline steeply above it (Schlenker & Roberts, 2009). Extreme heat is a central mechanism of U.S. maize yield loss (Lobell et al., 2013).

**Changing sensitivity over time.** Maize sensitivity to drought associated with high vapour-pressure deficit increased in the central United States between 1995 and 2012, despite rising average yields (Lobell et al., 2014). The yield–weather relationship is therefore not stationary over the period this study covers.

**Variation across space and season.** Sensitivity to extreme temperatures varies by region and growth phase (Butler & Huybers, 2015). Maize response to temperature variation also differs spatially, consistent with adaptation (Butler & Huybers, 2013). Linking land-surface-model soil moisture and fine-scale weather to county yields, Ortiz-Bobea et al. (2019) found an important historical role of water stress, while temperature-driven heat stress dominates projected impacts.

**Why statistical models remain a reference.** Statistical yield models remain the reference tool for these relationships, with known strengths and failure modes (Lobell & Burke, 2010).

**Implication for this study.** Degree-day heat metrics and drought indices are the established predictors. A fixed-effects degree-day panel is the natural statistical baseline. Non-stationary sensitivity and spatially varying response are two reasons to expect temporal and cross-state distribution shift.

### 2.2 Compound drought–heat extremes

**Definition and risk.** Compound events are combinations of drivers or hazards whose joint occurrence produces impacts different from those of each driver alone (Zscheischler et al., 2020). Dependence between drivers changes the associated risk (Zscheischler & Seneviratne, 2017). Concurrent droughts and heatwaves increased substantially in the United States (Mazdiyasni & AghaKouchak, 2015). Their high-end risk is projected to accelerate (Tripathy et al., 2023). Compound dry and hot growing seasons have become more probable and severe over major croplands (He et al., 2022; Heino et al., 2023).

**Crop impacts.** Compound heat and moisture extremes are a leading pathway of crop yield loss (Lesk et al., 2022). Drought and extreme heat reduced national cereal production (Lesk et al., 2016). Nested-copula estimates quantify the added crop-failure risk from compound dry–hot conditions (Ribeiro et al., 2020). Climate extremes explain a substantial share of global yield anomalies (Vogel et al., 2019).

**U.S.-specific evidence — decisive for how this study is positioned.**
- For U.S. corn in 1981–2015, **metrics of compound hydroclimatic extremes are better predictors of yield variation than metrics of individual extremes, and heat-stress damage was up to four times more severe when combined with water stress** (Haqiqi et al., 2021).
- For U.S. soybean, interaction-aware statistical models attribute the largest negative effects to high temperature combined with low soil moisture during reproductive stages, with out-of-sample evaluation (Hamed et al., 2021).
- County-level killing degree days and drought days reduce both maize yield and harvestable fraction (Xiao et al., 2025).
- Compound drought–heat events have reduced water-use efficiency in U.S. corn and soybean (Yan et al., 2025).

**Stage dependence.** Reproductive-stage drought reduces maize yield more than vegetative-stage drought (Daryanto et al., 2016). Soil-moisture stress around silk emergence is particularly damaging (Vennam et al., 2023). This supports stage-windowed exposure. There is no consensus index, threshold or aggregation for compound drought–heat events; recent data reviews call for phenology-aligned joint indices (Li et al., 2025).

**Implication.** That compound drought–heat stress damages U.S. maize is **established**, not a gap. What this study can legitimately ask is narrower: whether phenology-windowed compound-exposure features change predictive skill or interval behaviour beyond univariate heat and drought covariates, and whether prediction intervals remain valid in compound-exposed conditions.

### 2.3 Irrigation, management and trend

- **Irrigation** shifts or decouples the relationship between climate extremes and U.S. crop yields (Troy et al., 2015). This justifies restricting the study to predominantly rainfed states, and it implies that irrigated counties *within* those states should be controlled for or tested.
- **ENSO.** ENSO modulates the global yields of major crops (Iizumi et al., 2014) and can force synchronous crop failures (Anderson et al., 2019). This motivates ENSO-stratified evaluation, used here for evaluation only and not as a predictor.
- **Technology trend.** Long-term trends from technology and management must be separated from weather-driven variation. The choice of detrending method affects the apparent spatial pattern of drought impacts in U.S. yield records (Lu et al., 2017). Smooth non-linear trends are one accepted alternative to linear detrending (Lischeid et al., 2022).

### 2.4 Drought indices and weather data products

- **SPEI** extends precipitation-based drought indices by accounting for evaporative demand (Vicente-Serrano et al., 2010). Its parameter fitting and evapotranspiration choices are documented by Beguería et al. (2014). The choice of candidate distribution matters for both SPI and SPEI (Stagge et al., 2015).
- **Data products.** Temperature and precipitation come from PRISM (Daly et al., 2008). Radiation, humidity and wind for FAO-56 reference evapotranspiration come from NLDAS-2 (Xia et al., 2012). Drought-index validity is checked against the U.S. Drought Monitor (Svoboda et al., 2002).
- **Relevance of ERA5.** ERA5 (Hersbach et al., 2020) is relevant only as the source of the superseded legacy dataset.

---

## 3. Machine-learning prediction of county-level yield

### 3.1 Predictive skill

**Neural models.**
- An LSTM combining phenology-based remote-sensing and meteorological indices explained 76 % of county corn yield variation across the Corn Belt, compared with 39 % from meteorological indices alone, and was robust in the 2012 extreme year (Jiang et al., 2020).
- A CNN-RNN framework forecast corn and soybean yields across 13 Corn Belt states for 2016–2018 (Khaki et al., 2020), following earlier deep-network work (Khaki & Wang, 2019).
- Probabilistic deep models appeared early for yield from remote sensing (You et al., 2017).

**Tree ensembles and algorithm comparisons.**
- A comparison of six algorithms for county maize in the U.S. Midwest found **XGBoost best in accuracy and stability, while LSTM and CNN were not advantageous** (Kang et al., 2020).
- Weighted ML ensembles gave the best county corn forecasts in Illinois, Indiana and Iowa under blocked sequential validation (Shahhosseini et al., 2020).
- Adding crop-model outputs — especially soil-moisture and water-table variables — improved ML predictions by 7–20 % in RMSE (Shahhosseini et al., 2021).
- An interaction regression model combined accuracy with interpretable environment-by-management effects in the same three states (Ansarifar et al., 2021).

**Hybrid and comparative studies.**
- A semiparametric neural network that retains parametric structure and cross-sectional heterogeneity outperformed both classical statistical models and fully nonparametric networks on withheld years (Crane-Droesch, 2018).
- Across model classes, machine learning best reproduced the distribution of U.S. maize yields, including variability (Leng & Hall, 2020).
- Machine learning can capture much of yield variability, but equally good models can rest on different predictor sets, which limits mechanistic interpretation of feature importance (Lischeid et al., 2022).

**Reviews and benchmarks.** Systematic reviews and large-scale baselines document the field (van Klompenburg et al., 2020; Paudel et al., 2021). CY-Bench now provides a harmonised sub-national benchmark for maize and wheat forecasting (Kallenberg et al., 2026).

**Implication.** For tabular county-year inputs, well-tuned tree ensembles are at least as skilful as deep architectures. Architecture is not a credible source of novelty. Point-prediction skill must be shown against naive and statistical baselines.

### 3.2 Validation, leakage and generalisation

**Structured validation.** Data with temporal, spatial or hierarchical structure require blocked cross-validation (Roberts et al., 2017). Random validation can greatly overstate large-scale predictive performance (Ploton et al., 2020). Location-identifying predictors can reproduce training data rather than enable spatial prediction (Meyer et al., 2019). Transfer to new domains should be qualified by an area of applicability (Meyer & Pebesma, 2021). There is also a counter-view: spatial cross-validation is not the right tool for estimating *map accuracy* (Wadoux et al., 2021). The distinction supports treating leave-one-state-out validation as a **transfer test** rather than a map-accuracy estimate. Newer distance-matching schemes refine structured cross-validation (Linnenbrink et al., 2024).

**Leakage.** Leakage is a widespread cause of irreproducible ML-based science (Kapoor & Narayanan, 2023), and consensus reporting standards now exist (Kapoor et al., 2024).

**Spatial transfer of yield models.**
- Transfer of yield models across regions has been approached with deep transfer learning (Wang et al., 2018) and with Bayesian domain-adversarial networks that use unlabelled target-region data (Ma & Zhang, 2022). The latter is a different protocol from strict leave-one-state-out.
- Grouped (location-held-out) validation can remove apparent ML skill entirely: in a recent maize study, no model beat a training-mean benchmark under parish-grouped cross-validation, while random validation was optimistic (Guarda, 2026).
- Domain-generalisation benchmarks in machine learning likewise show that careful evaluation often erases claimed gains (Gulrajani & Lopez-Paz, 2021).

**Implication.** Temporal hold-out and strict leave-one-state-out are the appropriate tests. Naive baselines are necessary. Random-split results are an optimistic reference only.

---

## 4. Uncertainty quantification for crop yield

**Bayesian approaches.** Bayesian neural networks have quantified uncertainty for U.S. corn yield from remotely sensed variables (Ma et al., 2021). A phenology-guided Bayesian CNN decomposed aleatoric and epistemic uncertainty for county-level soybean yield in the U.S. Corn Belt (Zhang & Diao, 2023). **These approaches do not provide finite-sample coverage guarantees.**

**Conformal approaches in agriculture.**
- A recent crop-yield framework combining stacked ensembles, graph neural networks and conformalised quantile regression across 15 countries and six crops reported calibrated 80 % intervals with 80.72 % empirical coverage — **a single marginal coverage figure** (Mahmood et al., 2026).
- In agricultural image tasks, inductive conformal prediction on ResNet-18 and ViT-B/16 met pre-defined error levels, with experiments including covariate shift and out-of-distribution detection (Farag et al., 2025). That study concerns classification, not yield regression.
- A non-peer-reviewed 2026 preprint applies split conformal prediction and adaptive conformal inference to crop yield under inter-annual climate variability in India and reports marginal PICP (Namdeo & Baraskar, 2026). It is cited here so that the overlap in framing is acknowledged, not as established evidence.

**Conformal approaches in environmental science.** Conformal prediction is spreading in environmental science:
- Google Earth Engine–native tools for Earth observation (Singh et al., 2024);
- soil organic carbon mapping (Kakhani et al., 2024);
- weighted conformal inference to handle distribution shift in runoff interval prediction across 222 Australian basins (Jia et al., 2026).

**Implication.** Yield UQ is either Bayesian without coverage guarantees, or conformal with marginal coverage reported as a single number. None of the verified crop-yield studies compares several conformal calibrators, evaluates coverage within climate regimes, or evaluates coverage under cross-state transfer.

---

## 5. Conformal prediction

### 5.1 Foundations

Conformal prediction produces prediction sets with finite-sample **marginal** coverage under exchangeability (Vovk et al., 2022; Shafer & Vovk, 2008). Split conformal prediction is the standard distribution-free procedure for regression (Lei et al., 2018). Local and adaptive variants make interval width depend on the covariates (Lei & Wasserman, 2014; Kivaranovic et al., 2020; Guan, 2023).

Conformalised quantile regression combines quantile regression (Koenker & Bassett, 1978) with conformal calibration to obtain adaptive intervals with marginal coverage (Romano et al., 2019). Its variants have been compared (Sesia & Candès, 2020). Quantile regression forests (Meinshausen, 2006) are a tree-based quantile alternative. Accessible syntheses are available (Angelopoulos & Bates, 2023).

### 5.2 Distribution shift and non-exchangeability

**Shift in the covariates.** Weighted conformal prediction restores validity under covariate shift when likelihood ratios are known or estimated (Tibshirani et al., 2019). Robust validation targets coverage over a neighbourhood of distributions (Cauchois et al., 2024). When exchangeability fails, coverage degrades by an amount that can be bounded (Barber et al., 2023).

**Online methods for arbitrary shift.**
- Adaptive conformal inference updates the miscoverage level online and attains long-run **marginal** coverage under arbitrary distribution shift (Gibbs & Candès, 2021).
- Its step-size sensitivity is addressed by dynamically-tuned ACI (Gibbs & Candès, 2024) and by aggregated ACI for dependent time series (Zaffran et al., 2022).
- Related online procedures use strongly adaptive online learning (Bhatnagar et al., 2023) or control-theoretic updates that can also correct systematic bias (Angelopoulos et al., 2023).
- For dependent series, ensemble batch prediction intervals give approximately valid marginal coverage under strongly mixing errors (Xu & Xie, 2023a), and sequential predictive conformal inference extends this (Xu & Xie, 2023b).

**Implication.** Online conformal methods guarantee *long-run marginal* coverage. They adapt to the recent past, so they are not designed to anticipate a location shift in a single anomalous year.

### 5.3 Conditional and group-conditional coverage

**Limits.** Inductive conformal predictors are valid marginally but not conditionally (Vovk, 2013). Distribution-free conditional coverage on continuous covariates cannot be achieved in finite samples without further assumptions (Barber et al., 2021).

**What is achievable.** Coverage over pre-specified groups **is** achievable:
- Mondrian conformal predictors calibrate within partitions (Vovk, 2013; Toccaceli & Gammerman, 2019).
- Equalised coverage across groups has been proposed as an evaluation and design target (Romano et al., 2020).
- Batch multivalid conformal prediction covers overlapping groups (Jung et al., 2023).
- Recent work gives **exact finite-sample coverage over every group in a pre-specified finite collection, and over finite-dimensional classes of covariate shift**, with quantified error for richer classes (Gibbs et al., 2025).

**Implication.** The statement "conditional coverage is impossible" is correct only for full conditional coverage. For pre-specified climate regimes, group-conditional methods are the principled remedy and must be compared, not dismissed.

### 5.4 Spatial conformal prediction

Spatial data can be locally approximately exchangeable, which enables valid model-free spatial prediction intervals (Mao et al., 2024). GeoConformal prediction provides a model-agnostic geographic framework: its coverage exceeded bootstrap intervals, and it attributes uncertainty to local dependence (Lou et al., 2025). Both address spatial interpolation and dependence, **not** transfer to an entirely unseen administrative region.

---

## 6. Evaluating probabilistic predictions

- **Calibration and sharpness.** Probabilistic forecasts should maximise sharpness subject to calibration (Gneiting et al., 2007). Proper scoring rules — including interval scores such as the Winkler score — reward both (Gneiting & Raftery, 2007). An overview is given by Gneiting & Katzfuss (2014).
- **Comparing predictors under dependence.** Losses from two predictors evaluated on dependent data should be compared with tests that account for that dependence (Diebold & Mariano, 1995).
- **Few clusters.** Cluster-robust inference with few clusters — for example five test years — is unreliable, and bootstrap refinements are needed (Cameron et al., 2008).
- **Explanations.** SHAP values (Lundberg & Lee, 2017) explain individual predictions. Their stability across equally good models should not be assumed (Lischeid et al., 2022).

---

## 7. Synthesis: state of the art, limitations, gaps and positioning

### 7.1 State of the art

| Strand | Established | Representative work |
|---|---|---|
| U.S. maize yield–weather | Nonlinear heat damage; changing drought sensitivity; compound hot–dry impacts exceed individual stressors | Schlenker & Roberts 2009; Lobell et al. 2014; Haqiqi et al. 2021 |
| County-yield ML | Tree ensembles ≥ deep architectures on tabular inputs; structured validation required | Kang et al. 2020; Jiang et al. 2020; Roberts et al. 2017 |
| Yield UQ | Bayesian predictive uncertainty; conformal intervals with marginal coverage | Ma et al. 2021; Zhang & Diao 2023; Mahmood et al. 2026 |
| Conformal under shift | Long-run marginal coverage online; weighted CP for covariate shift; validity bounds | Gibbs & Candès 2021, 2024; Tibshirani et al. 2019; Barber et al. 2023 |
| Conditional coverage | Impossible in general; attainable for pre-specified groups | Barber et al. 2021; Gibbs et al. 2025 |

### 7.2 Limitations of existing research — each tied to evidence

1. **Coverage is reported marginally in crop-yield applications.** A single coverage figure is reported (Mahmood et al., 2026), even though the motivation is extreme years, where coverage within regimes is the operational question (Romano et al., 2020).
2. **No comparison among conformal calibrators for yield.** Crop-yield conformal work applies one procedure (Mahmood et al., 2026) or studies classification (Farag et al., 2025). The trade-offs documented in the methods literature (Gibbs & Candès, 2021, 2024; Tibshirani et al., 2019; Gibbs et al., 2025) are untested on yield data.
3. **Cross-state transfer of interval validity is unreported.** Spatial conformal methods target local dependence (Mao et al., 2024; Lou et al., 2025). Yield transfer studies evaluate point accuracy (Ma & Zhang, 2022; Guarda, 2026).
4. **Trend and non-stationarity are handled implicitly.** Detrending choices change inferred drought impacts (Lu et al., 2017), and sensitivities change over time (Lobell et al., 2014). How trend transfer affects interval validity across states is not examined.

### 7.3 Research gaps (evidence-based and conservatively stated)

- **Gap A (primary).** Conformal calibration methods for crop yield are compared, when compared at all, on marginal coverage. Whether their ranking holds within pre-specified compound drought–heat, year and state strata — where decision value concentrates — and whether a group-conditional procedure changes the conclusion, has not been evaluated on county-level yield data. *Supported by limitations 1–2; theory in Romano et al. 2020, Barber et al. 2021 and Gibbs et al. 2025.*
- **Gap B (primary).** The validity of conformal yield intervals under strict leave-one-state-out transfer, and whether failures stem from interval width or from level (trend) bias, has not been reported. *Supported by limitations 3–4.*
- **Gap C (secondary; a possibly null result).** Whether phenology-windowed compound drought–heat features, or joint point-quantile training, change interval sharpness or regime-conditional coverage beyond univariate stress covariates and post-hoc training. Compound impacts themselves are established (Haqiqi et al., 2021).

### 7.4 Connection to the objectives

| Gap | Research question | Objective |
|---|---|---|
| A | RQ2, RQ4 | RO3 (calibrator comparison: marginal and regime-conditional; rolling-origin temporal; LOSO) |
| B | RQ3 | RO4 (bias vs width; trend transfer) |
| C | RQ5 | RO5 (multi-seed feature and training-paradigm tests) |
| Enabling | RQ1 | RO1–RO2 (leakage-controlled panel; skill over naive and panel baselines) |

### 7.5 Positioning (novelty stated without exaggeration)

> This study evaluates several established conformal calibration methods for county-level corn yield in six rainfed U.S. Corn Belt states under temporal and leave-one-state-out distribution shift. It assesses coverage both marginally and within pre-specified climate-exposure, year and state strata, and diagnoses whether coverage failures arise from interval width or from prediction bias. The contribution is an evaluation and diagnosis, not a new estimator or architecture.

It differs from:
- Bayesian yield UQ (Ma et al., 2021; Zhang & Diao, 2023) — coverage-evaluated intervals;
- recent conformal yield work (Mahmood et al., 2026) — a multi-method comparison with regime-conditional and cross-state evaluation;
- the conformal methods literature — domain evidence on real county panels, including a group-conditional comparator (Gibbs et al., 2025).

---

## 8. References (verified)

"Via" = registry used on 2026-09-16: CR Crossref · OA OpenAlex · S2 Semantic Scholar · AX arXiv API · PR carried over from a prior registry-verified project document (dated).

### Yield–weather and compound extremes
- Anderson, W.B., Seager, R., Baethgen, W., Cane, M., & You, L. (2019). Synchronous crop failures and climate-forced production variability. *Science Advances*, 5(7), eaaw1976. https://doi.org/10.1126/sciadv.aaw1976 (CR)
- Butler, E.E., & Huybers, P. (2013). Adaptation of US maize to temperature variations. *Nature Climate Change*, 3(1), 68–72. https://doi.org/10.1038/nclimate1585 (CR)
- Butler, E.E., & Huybers, P. (2015). Variations in the sensitivity of US maize yield to extreme temperatures by region and growth phase. *Environmental Research Letters*, 10(3), 034009. https://doi.org/10.1088/1748-9326/10/3/034009 (CR)
- Daryanto, S., Wang, L., & Jacinthe, P.-A. (2016). Global synthesis of drought effects on maize and wheat production. *PLOS ONE*, 11(5), e0156362. https://doi.org/10.1371/journal.pone.0156362 (CR)
- Hamed, R., Van Loon, A.F., Aerts, J., & Coumou, D. (2021). Impacts of compound hot–dry extremes on US soybean yields. *Earth System Dynamics*, 12(4), 1371–1391. https://doi.org/10.5194/esd-12-1371-2021 (CR, OA abstract)
- Haqiqi, I., Grogan, D.S., Hertel, T.W., & Schlenker, W. (2021). Quantifying the impacts of compound extremes on agriculture. *Hydrology and Earth System Sciences*, 25(2), 551–564. https://doi.org/10.5194/hess-25-551-2021 (CR, OA abstract)
- He, Y., Hu, X., Xu, W., Fang, J., & Shi, P. (2022). Increased probability and severity of compound dry and hot growing seasons over world's major croplands. *Science of the Total Environment*, 824, 153885. https://doi.org/10.1016/j.scitotenv.2022.153885 (CR)
- Heino, M., Kinnunen, P., Anderson, W., Ray, D.K., Puma, M.J., Varis, O., Siebert, S., & Kummu, M. (2023). Increased probability of hot and dry weather extremes during the growing season threatens global crop yields. *Scientific Reports*, 13, 3583. https://doi.org/10.1038/s41598-023-29378-2 (CR)
- Iizumi, T., et al. (2014). Impacts of El Niño Southern Oscillation on the global yields of major crops. *Nature Communications*, 5, 3712. https://doi.org/10.1038/ncomms4712 (CR)
- Lesk, C., Rowhani, P., & Ramankutty, N. (2016). Influence of extreme weather disasters on global crop production. *Nature*, 529(7584), 84–87. https://doi.org/10.1038/nature16467 (CR)
- Lesk, C., Anderson, W., Rigden, A., Coast, O., Jägermeyr, J., McDermid, S., Davis, K.F., & Konar, M. (2022). Compound heat and moisture extreme impacts on global crop yields under climate change. *Nature Reviews Earth & Environment*, 3(12), 872–889. https://doi.org/10.1038/s43017-022-00368-8 (CR)
- Li, Y., Zeleke, K., Wang, B., & Liu, D.-L. (2025). A review of data for compound drought and heatwave stress impacts on crops: current progress, knowledge gaps, and future pathways. *Plants*, 14(14), 2158. https://doi.org/10.3390/plants14142158 (CR)
- Lobell, D.B., & Burke, M.B. (2010). On the use of statistical models to predict crop yield responses to climate change. *Agricultural and Forest Meteorology*, 150(11), 1443–1452. https://doi.org/10.1016/j.agrformet.2010.07.008 (CR)
- Lobell, D.B., Hammer, G.L., McLean, G., Messina, C., Roberts, M.J., & Schlenker, W. (2013). The critical role of extreme heat for maize production in the United States. *Nature Climate Change*, 3(5), 497–501. https://doi.org/10.1038/nclimate1832 (CR)
- Lobell, D.B., Roberts, M.J., Schlenker, W., Braun, N., Little, B.B., Rejesus, R.M., & Hammer, G.L. (2014). Greater sensitivity to drought accompanies maize yield increase in the U.S. Midwest. *Science*, 344(6183), 516–519. https://doi.org/10.1126/science.1251423 (CR, OA abstract)
- Lu, J., Carbone, G.J., & Gao, P. (2017). Detrending crop yield data for spatial visualization of drought impacts in the United States, 1895–2014. *Agricultural and Forest Meteorology*, 237–238, 196–208. https://doi.org/10.1016/j.agrformet.2017.02.001 (CR)
- Mazdiyasni, O., & AghaKouchak, A. (2015). Substantial increase in concurrent droughts and heatwaves in the United States. *PNAS*, 112(37), 11484–11489. https://doi.org/10.1073/pnas.1422945112 (CR)
- Ortiz-Bobea, A., Wang, H., Carrillo, C.M., & Ault, T.R. (2019). Unpacking the climatic drivers of US agricultural yields. *Environmental Research Letters*, 14(6), 064003. https://doi.org/10.1088/1748-9326/ab1e75 (CR, OA abstract)
- Ribeiro, A.F.S., Russo, A., Gouveia, C.M., Páscoa, P., & Zscheischler, J. (2020). Risk of crop failure due to compound dry and hot extremes estimated with nested copulas. *Biogeosciences*, 17(19), 4815–4830. https://doi.org/10.5194/bg-17-4815-2020 (CR)
- Schlenker, W., & Roberts, M.J. (2009). Nonlinear temperature effects indicate severe damages to U.S. crop yields under climate change. *PNAS*, 106(37), 15594–15598. https://doi.org/10.1073/pnas.0906865106 (CR, OA abstract)
- Tripathy, K.P., Mukherjee, S., Mishra, A.K., Mann, M.E., & Williams, A.P. (2023). Climate change will accelerate the high-end risk of compound drought and heatwave events. *PNAS*, 120(28), e2219825120. https://doi.org/10.1073/pnas.2219825120 (CR)
- Troy, T.J., Kipgen, C., & Pal, I. (2015). The impact of climate extremes and irrigation on US crop yields. *Environmental Research Letters*, 10(5), 054013. https://doi.org/10.1088/1748-9326/10/5/054013 (CR, OA abstract)
- Vennam, R.R., Poudel, S., Ramamoorthy, P., Samiappan, S., Reddy, K.R., & Bheemanahalli, R. (2023). Impact of soil moisture stress during the silk emergence and grain-filling in maize. *Physiologia Plantarum*, 175(5), e14029. https://doi.org/10.1111/ppl.14029 (CR)
- Vogel, E., et al. (2019). The effects of climate extremes on global agricultural yields. *Environmental Research Letters*, 14(5), 054010. https://doi.org/10.1088/1748-9326/ab154b (CR)
- Xiao, K., Zhou, X., Gui, H., Tian, Y., Chen, X., Li, Y., Matomela, N., & Xin, Q. (2025). Drought and extreme heat reduce wheat and maize production in the United States by lowering both crop yields and harvestable fraction. *Earth's Future*, 13(12), e2024EF005557. https://doi.org/10.1029/2024ef005557 (CR, OA abstract)
- Yan, H., You, Y., Jiao, W., & Pan, N. (2025). Intensifying impacts of compound drought and heatwave events on water use efficiency in U.S. corn and soybean. *Agricultural and Forest Meteorology*, 375, 110873. https://doi.org/10.1016/j.agrformet.2025.110873 (CR)
- Zscheischler, J., et al. (2020). A typology of compound weather and climate events. *Nature Reviews Earth & Environment*, 1(7), 333–347. https://doi.org/10.1038/s43017-020-0060-z (CR)
- Zscheischler, J., & Seneviratne, S.I. (2017). Dependence of drivers affects risks associated with compound events. *Science Advances*, 3(6), e1700263. https://doi.org/10.1126/sciadv.1700263 (CR)

### Indices and data products
- Beguería, S., Vicente-Serrano, S.M., Reig, F., & Latorre, B. (2014). Standardized precipitation evapotranspiration index (SPEI) revisited. *International Journal of Climatology*, 34(10), 3001–3023. https://doi.org/10.1002/joc.3887 (CR)
- Daly, C., et al. (2008). Physiographically sensitive mapping of climatological temperature and precipitation across the conterminous United States. *International Journal of Climatology*, 28(15), 2031–2064. https://doi.org/10.1002/joc.1688 (CR)
- Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999–2049. https://doi.org/10.1002/qj.3803 (CR)
- Stagge, J.H., Tallaksen, L.M., Gudmundsson, L., Van Loon, A.F., & Stahl, K. (2015). Candidate distributions for climatological drought indices (SPI and SPEI). *International Journal of Climatology*, 35(13), 4027–4040. https://doi.org/10.1002/joc.4267 (CR)
- Svoboda, M., et al. (2002). The Drought Monitor. *Bulletin of the American Meteorological Society*, 83(8), 1181–1190. https://doi.org/10.1175/1520-0477-83.8.1181 (CR)
- Vicente-Serrano, S.M., Beguería, S., & López-Moreno, J.I. (2010). A multiscalar drought index sensitive to global warming: the standardized precipitation evapotranspiration index. *Journal of Climate*, 23(7), 1696–1718. https://doi.org/10.1175/2009jcli2909.1 (CR)
- Xia, Y., et al. (2012). Continental-scale water and energy flux analysis and validation for the North American Land Data Assimilation System project phase 2 (NLDAS-2): 1. Intercomparison and application of model products. *Journal of Geophysical Research: Atmospheres*, 117(D3). https://doi.org/10.1029/2011jd016048 (CR)
- *Documentation to cite without DOI:* Allen et al. (1998) FAO-56; NOAA NCEI Storm Events Database; NLCD (state the epoch used); the soil product (state source and depth); USDA NASS Quick Stats.

### Yield machine learning and validation
- Ansarifar, J., Wang, L., & Archontoulis, S.V. (2021). An interaction regression model for crop yield prediction. *Scientific Reports*, 11, 17754. https://doi.org/10.1038/s41598-021-97221-7 (CR, OA abstract)
- Crane-Droesch, A. (2018). Machine learning methods for crop yield prediction and climate change impact assessment in agriculture. *Environmental Research Letters*, 13(11), 114003. https://doi.org/10.1088/1748-9326/aae159 (CR, OA abstract)
- Guarda, T. (2026). Limits to the spatial transferability of machine-learning models for farm-record-level maize yield prediction in Manabí, Ecuador. *Agriculture*, 16(17), 1932. https://doi.org/10.3390/agriculture16171932 (OA abstract)
- Gulrajani, I., & Lopez-Paz, D. (2021). In search of lost domain generalization. *International Conference on Learning Representations*. arXiv:2007.01434 (S2)
- Jiang, H., et al. (2020). A deep learning approach to conflating heterogeneous geospatial data for corn yield estimation: a case study of the US Corn Belt at the county level. *Global Change Biology*, 26(3), 1754–1766. https://doi.org/10.1111/gcb.14885 (CR, OA abstract)
- Kallenberg, M., Paudel, D., et al. (2026). CY-Bench: a comprehensive benchmark dataset for sub-national crop yield forecasting. *Earth System Science Data*, 18(6), 3997–4018. https://doi.org/10.5194/essd-18-3997-2026 (CR, OA abstract)
- Kang, Y., Ozdogan, M., Zhu, X., Ye, Z., Hain, C., & Anderson, M. (2020). Comparative assessment of environmental variables and machine learning algorithms for maize yield prediction in the US Midwest. *Environmental Research Letters*, 15(6), 064005. https://doi.org/10.1088/1748-9326/ab7df9 (CR, OA abstract)
- Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804. https://doi.org/10.1016/j.patter.2023.100804 (CR)
- Kapoor, S., et al. (2024). REFORMS: consensus-based recommendations for machine-learning-based science. *Science Advances*, 10(18), eadk3452. https://doi.org/10.1126/sciadv.adk3452 (CR)
- Khaki, S., & Wang, L. (2019). Crop yield prediction using deep neural networks. *Frontiers in Plant Science*, 10, 621. https://doi.org/10.3389/fpls.2019.00621 (CR)
- Khaki, S., Wang, L., & Archontoulis, S.V. (2020). A CNN-RNN framework for crop yield prediction. *Frontiers in Plant Science*, 10, 1750. https://doi.org/10.3389/fpls.2019.01750 (CR, OA abstract)
- Leng, G., & Hall, J.W. (2020). Predicting spatial and temporal variability in crop yields: an inter-comparison of machine learning, regression and process-based models. *Environmental Research Letters*, 15(4), 044027. https://doi.org/10.1088/1748-9326/ab7b24 (CR, OA abstract)
- Linnenbrink, J., Milà, C., Ludwig, M., & Meyer, H. (2024). kNNDM CV: k-fold nearest-neighbour distance matching cross-validation for map accuracy estimation. *Geoscientific Model Development*, 17(15), 5897–5912. https://doi.org/10.5194/gmd-17-5897-2024 (CR)
- Lischeid, G., Webber, H., Sommer, M., Nendel, C., & Ewert, F. (2022). Machine learning in crop yield modelling: a powerful tool, but no surrogate for science. *Agricultural and Forest Meteorology*, 312, 108698. https://doi.org/10.1016/j.agrformet.2021.108698 (CR, OA abstract)
- Ma, Y., & Zhang, Z. (2022). A Bayesian domain adversarial neural network for corn yield prediction. *IEEE Geoscience and Remote Sensing Letters*, 19, 1–5. https://doi.org/10.1109/lgrs.2022.3211444 (CR, OA abstract)
- Meyer, H., & Pebesma, E. (2021). Predicting into unknown space? Estimating the area of applicability of spatial prediction models. *Methods in Ecology and Evolution*, 12(9), 1620–1633. https://doi.org/10.1111/2041-210x.13650 (CR)
- Meyer, H., Reudenbach, C., Wöllauer, S., & Nauss, T. (2019). Importance of spatial predictor variable selection in machine learning applications — moving from data reproduction to spatial prediction. *Ecological Modelling*, 411, 108815. https://doi.org/10.1016/j.ecolmodel.2019.108815 (CR)
- Paudel, D., et al. (2021). Machine learning for large-scale crop yield forecasting. *Agricultural Systems*, 187, 103016. https://doi.org/10.1016/j.agsy.2020.103016 (CR, OA abstract)
- Ploton, P., et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications*, 11, 4540. https://doi.org/10.1038/s41467-020-18321-y (CR)
- Roberts, D.R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. https://doi.org/10.1111/ecog.02881 (CR)
- Shahhosseini, M., Hu, G., & Archontoulis, S.V. (2020). Forecasting corn yield with machine learning ensembles. *Frontiers in Plant Science*, 11, 1120. https://doi.org/10.3389/fpls.2020.01120 (CR, OA abstract)
- Shahhosseini, M., Hu, G., Huber, I., & Archontoulis, S.V. (2021). Coupling machine learning and crop modeling improves crop yield prediction in the US Corn Belt. *Scientific Reports*, 11, 1606. https://doi.org/10.1038/s41598-020-80820-1 (CR, OA abstract)
- van Klompenburg, T., Kassahun, A., & Catal, C. (2020). Crop yield prediction using machine learning: a systematic literature review. *Computers and Electronics in Agriculture*, 177, 105709. https://doi.org/10.1016/j.compag.2020.105709 (CR)
- Wadoux, A.M.J.-C., Heuvelink, G.B.M., de Bruin, S., & Brus, D.J. (2021). Spatial cross-validation is not the right way to evaluate map accuracy. *Ecological Modelling*, 457, 109692. https://doi.org/10.1016/j.ecolmodel.2021.109692 (CR)
- Wang, A.X., Tran, C., Desai, N., Lobell, D., & Ermon, S. (2018). Deep transfer learning for crop yield prediction with remote sensing data. *Proceedings of the 1st ACM SIGCAS Conference on Computing and Sustainable Societies*, 1–5. https://doi.org/10.1145/3209811.3212707 (CR)
- You, J., Li, X., Low, M., Lobell, D., & Ermon, S. (2017). Deep Gaussian process for crop yield prediction based on remote sensing data. *Proceedings of the AAAI Conference on Artificial Intelligence*, 31(1). https://doi.org/10.1609/aaai.v31i1.11172 (CR)

### Uncertainty quantification in yield and environmental science
- Farag, M., Emam, A., Leonhardt, J., & Roscher, R. (2025). Enhancing decision support in crop production: analyzing conformal prediction for uncertainty quantification. *Computers and Electronics in Agriculture*, 237, 110559. https://doi.org/10.1016/j.compag.2025.110559 (CR, OA abstract)
- Jia, Y., Su, X., Singh, V.P., & Zhao, B. (2026). A novel hybrid predictive model based on mixture density networks with weighted conformal inference strategy for runoff interval prediction across Australia. *Water Resources Research*. https://doi.org/10.1029/2024wr039807 (OA abstract)
- Kakhani, N., Alamdar, S., Kebonye, N.M., Amani, M., & Scholten, T. (2024). Uncertainty quantification of soil organic carbon estimation from remote sensing data with conformal prediction. *Remote Sensing*, 16(3), 438. https://doi.org/10.3390/rs16030438 (CR)
- Ma, Y., Zhang, Z., Kang, Y., & Özdoğan, M. (2021). Corn yield prediction and uncertainty analysis based on remotely sensed variables using a Bayesian neural network approach. *Remote Sensing of Environment*, 259, 112408. https://doi.org/10.1016/j.rse.2021.112408 (OA; PR 2026-09-07)
- Mahmood, S., Hasan, R., & Ahmad, S. (2026). HSE-GNN-CP: spatiotemporal teleconnection modeling and conformalized uncertainty quantification for global crop yield forecasting. *Information*, 17(2), 141. https://doi.org/10.3390/info17020141 (CR, OA abstract)
- Namdeo, D., & Baraskar, R. (2026). Climate-adaptive transformer for crop yield prediction with conformal uncertainty quantification [Preprint, not peer reviewed]. Zenodo. https://doi.org/10.5281/zenodo.19429121 (OA abstract)
- Singh, G., Moncrieff, G.R., Venter, Z.S., & Cawse-Nicholson, K. (2024). Uncertainty quantification for probabilistic machine learning in earth observation using conformal prediction. *Scientific Reports*, 14. https://doi.org/10.1038/s41598-024-65954-w (OA abstract)
- Zhang, C., & Diao, C. (2023). A phenology-guided Bayesian-CNN (PB-CNN) framework for soybean yield estimation and uncertainty analysis. *ISPRS Journal of Photogrammetry and Remote Sensing*, 205, 50–73. https://doi.org/10.1016/j.isprsjprs.2023.09.025 (OA; PR 2026-09-07)

### Conformal prediction
- Angelopoulos, A.N., & Bates, S. (2023). Conformal prediction: a gentle introduction. *Foundations and Trends in Machine Learning*, 16(4), 494–591. https://doi.org/10.1561/2200000101 (CR)
- Angelopoulos, A.N., Candès, E.J., & Tibshirani, R.J. (2023). Conformal PID control for time series prediction. *Advances in Neural Information Processing Systems*. arXiv:2307.16895 (S2)
- Barber, R.F., Candès, E.J., Ramdas, A., & Tibshirani, R.J. (2021). The limits of distribution-free conditional predictive inference. *Information and Inference: A Journal of the IMA*, 10(2), 455–482. https://doi.org/10.1093/imaiai/iaaa017 (CR)
- Barber, R.F., Candès, E.J., Ramdas, A., & Tibshirani, R.J. (2023). Conformal prediction beyond exchangeability. *The Annals of Statistics*, 51(2), 816–845. https://doi.org/10.1214/23-AOS2276 (CR)
- Bhatnagar, A., Wang, H., Xiong, C., & Bai, Y. (2023). Improved online conformal prediction via strongly adaptive online learning. *International Conference on Machine Learning*. arXiv:2302.07869 (S2)
- Cauchois, M., Gupta, S., Ali, A., & Duchi, J.C. (2024). Robust validation: confident predictions even when distributions shift. *Journal of the American Statistical Association*, 119(548), 3033–3044. https://doi.org/10.1080/01621459.2023.2298037 (CR)
- Gibbs, I., & Candès, E.J. (2021). Adaptive conformal inference under distribution shift. *Advances in Neural Information Processing Systems*, 34. (PR 2026-09-06)
- Gibbs, I., & Candès, E.J. (2024). Conformal inference for online prediction with arbitrary distribution shifts. *Journal of Machine Learning Research*, 25(162), 1–36. arXiv:2208.08401 (S2)
- Gibbs, I., Cherian, J.J., & Candès, E.J. (2025). Conformal prediction with conditional guarantees. *Journal of the Royal Statistical Society Series B*, 87(4), 1100–1126. https://doi.org/10.1093/jrsssb/qkaf008 (CR, OA abstract)
- Guan, L. (2023). Localized conformal prediction: a generalized inference framework for conformal prediction. *Biometrika*, 110(1), 33–50. https://doi.org/10.1093/biomet/asac040 (CR)
- Jung, C., Noarov, G., Ramalingam, R., & Roth, A. (2023). Batch multivalid conformal prediction. *International Conference on Learning Representations*. arXiv:2209.15145 (S2)
- Kivaranovic, D., Johnson, K.D., & Leeb, H. (2020). Adaptive, distribution-free prediction intervals for deep networks. *Proceedings of AISTATS*, PMLR 108, 4346–4356. (PR 2026-09-06)
- Koenker, R., & Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33–50. https://doi.org/10.2307/1913643 (CR)
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R.J., & Wasserman, L. (2018). Distribution-free predictive inference for regression. *Journal of the American Statistical Association*, 113(523), 1094–1111. https://doi.org/10.1080/01621459.2017.1307116 (CR)
- Lei, J., & Wasserman, L. (2014). Distribution-free prediction bands for non-parametric regression. *Journal of the Royal Statistical Society Series B*, 76(1), 71–96. https://doi.org/10.1111/rssb.12021 (CR)
- Lou, X., Luo, P., & Meng, L. (2025). GeoConformal prediction: a model-agnostic framework for measuring the uncertainty of spatial prediction. *Annals of the American Association of Geographers*, 115(8), 1971–1998. https://doi.org/10.1080/24694452.2025.2516091 (CR, OA abstract)
- Mao, H., Martin, R., & Reich, B.J. (2024). Valid model-free spatial prediction. *Journal of the American Statistical Association*, 119(546), 904–914. https://doi.org/10.1080/01621459.2022.2147531 (CR, OA abstract)
- Meinshausen, N. (2006). Quantile regression forests. *Journal of Machine Learning Research*, 7, 983–999. (PR 2026-09-06)
- Romano, Y., Barber, R.F., Sabatti, C., & Candès, E.J. (2020). With malice toward none: assessing uncertainty via equalized coverage. *Harvard Data Science Review*. https://doi.org/10.1162/99608f92.03f00592 (S2)
- Romano, Y., Patterson, E., & Candès, E.J. (2019). Conformalized quantile regression. *Advances in Neural Information Processing Systems*, 32. arXiv:1905.03222 (OA; PR 2026-09-06)
- Sesia, M., & Candès, E.J. (2020). A comparison of some conformal quantile regression methods. *Stat*, 9(1), e261. https://doi.org/10.1002/sta4.261 (CR)
- Shafer, G., & Vovk, V. (2008). A tutorial on conformal prediction. *Journal of Machine Learning Research*, 9, 371–421. (PR 2026-09-06)
- Tibshirani, R.J., Barber, R.F., Candès, E.J., & Ramdas, A. (2019). Conformal prediction under covariate shift. *Advances in Neural Information Processing Systems*, 32. (PR 2026-09-06)
- Toccaceli, P., & Gammerman, A. (2019). Combination of inductive Mondrian conformal predictors. *Machine Learning*, 108(3), 489–510. https://doi.org/10.1007/s10994-018-5754-9 (CR)
- Vovk, V. (2013). Conditional validity of inductive conformal predictors. *Machine Learning*, 92(2–3), 349–376. https://doi.org/10.1007/s10994-013-5355-6 (CR)
- Vovk, V., Gammerman, A., & Shafer, G. (2022). *Algorithmic Learning in a Random World* (2nd ed.). Springer. https://doi.org/10.1007/978-3-031-06649-8 (CR)
- Xu, C., & Xie, Y. (2023a). Conformal prediction for time series. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 45(10), 11575–11587. https://doi.org/10.1109/tpami.2023.3272339 (CR)
- Xu, C., & Xie, Y. (2023b). Sequential predictive conformal inference for time series. *International Conference on Machine Learning*. arXiv:2212.03463 (S2)
- Zaffran, M., Dieuleveut, A., Féron, O., Goude, Y., & Josse, J. (2022). Adaptive conformal predictions for time series. *International Conference on Machine Learning*, PMLR 162, 25834–25866. arXiv:2202.07282 (S2)

### Evaluation, inference and explanation
- Cameron, A.C., Gelbach, J.B., & Miller, D.L. (2008). Bootstrap-based improvements for inference with clustered errors. *Review of Economics and Statistics*, 90(3), 414–427. https://doi.org/10.1162/rest.90.3.414 (CR)
- Diebold, F.X., & Mariano, R.S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263. https://doi.org/10.1080/07350015.1995.10524599 (CR)
- Gneiting, T., Balabdaoui, F., & Raftery, A.E. (2007). Probabilistic forecasts, calibration and sharpness. *Journal of the Royal Statistical Society Series B*, 69(2), 243–268. https://doi.org/10.1111/j.1467-9868.2007.00587.x (CR)
- Gneiting, T., & Katzfuss, M. (2014). Probabilistic forecasting. *Annual Review of Statistics and Its Application*, 1, 125–151. https://doi.org/10.1146/annurev-statistics-062713-085831 (CR)
- Gneiting, T., & Raftery, A.E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378. https://doi.org/10.1198/016214506000001437 (CR)
- Lundberg, S.M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30. arXiv:1705.07874 (AX)

---

## 9. Verification log and changes from the previous review

**Registries queried (2026-09-16):**
- `api.crossref.org/works` — by DOI and by bibliographic query;
- `api.openalex.org` — records and abstracts;
- `api.semanticscholar.org/graph/v1/paper/search/match` — title matching;
- `export.arxiv.org/api/query` — by identifier.

Discovery searches (OpenAlex) covered: conformal prediction with crop yield, agriculture, drought, climate and spatial terms; crop-yield uncertainty; spatial cross-validation for crop yield; conditional coverage in climate applications.

**Limits:** full texts were not read. Statements rely on abstracts or on established method definitions. Semantic Scholar keyword search and arXiv keyword search were rate-limited or refused during the session, so discovery relied on OpenAlex and Crossref. The literature search is extensive but not exhaustive.

**Removed from the previous review, with reason:**
- Fischer & Knutti (2015): attribution of heavy-precipitation and hot extremes; off-target. Replaced by Mazdiyasni & AghaKouchak (2015).
- Poudel & Poudel (2023), *InWascon Technical Magazine*: not a peer-reviewed journal. Stage sensitivity is covered by Daryanto et al. (2016) and Vennam et al. (2023).
- Onogi (2025), bioRxiv: preprint on EVT, which the study does not use.
- Jabed & Azmi Murad (2024) and Ren et al. (2023): generic or tangential. Superseded by van Klompenburg et al. (2020), Kang et al. (2020) and Kallenberg et al. (2026).
- Sutanto et al. (2024, Mexico) and Daryanto et al. (2017): optional; not needed for the U.S. argument.
- Cesa-Bianchi & Lugosi (2006): optional background; online conformal is now anchored by Gibbs & Candès (2021, 2024).

**Corrected records:**
- PB-CNN authors → Zhang & Diao (not "Lin et al.").
- Barber et al. → 2021.
- Toccaceli & Gammerman → 2019.
- Zaffran et al. → ICML 2022 (not a preprint).
- Angelopoulos & Bates → FnT 2023 version.
- Vovk et al. → 2nd edition 2022.
- Farag et al. (2025) re-described as an empirical image-classification CP study.
- Mahmood et al. (2026) re-described as country-level with marginal coverage only.
