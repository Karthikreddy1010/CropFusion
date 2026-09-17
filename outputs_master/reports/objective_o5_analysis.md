# Objective O5 Report: Uncertainty Scaling vs CDHW Severity (§3 O5 & §5.8)

## Unit of independence
County-year rows are **not** independent: counties share year-level weather shocks and
each county is an autocorrelated series. All inference below therefore clusters on
**year** (5 clusters over 2345 rows).

- **Effective sample size**: `25` (from `2345` rows;
  ICC = `0.1945`, design effect = `92.04`)

## Primary inference — OLS with cluster-robust (CR1) SE, clustered by year
- **Slope**: `0.014295` per unit CDHW severity
- **95% CI**: `[-0.005525, 0.034115]`
- **p-value**: `0.11578` (t = `2.0025`, df = `4`)
- **R²**: `0.0276`

## Corroborating year-block cluster bootstrap
- **Slope**: `0.022668`, **95% CI**: `[0.004272, 0.067332]`, **p**: `0.023`
- Resampling unit: `year` (5 blocks)

## Secondary inference — clustered by county
- **Slope**: `0.014295`, **95% CI**: `[0.009903, 0.018687]`, **p**: `0.0`
  (555 county clusters)

## Descriptive association (NOT used for inference)
- Pearson r = `0.1661`, Spearman ρ = `0.0972`
- Naive OLS p-value = `0.0` — **invalid**, shown only to
  document how much the independence assumption distorts the result.

## Phenology stage-specific associations
- **Vegetative**: cluster-robust slope = `0.067581` (p = `0.036872`, 95% CI `[0.006693, 0.128469]`), R² = `0.027771`, descriptive Pearson r = `0.1666`
- **Silking_R1**: cluster-robust slope = `0.011995` (p = `0.087912`, 95% CI `[-0.002825, 0.026815]`), R² = `0.015046`, descriptive Pearson r = `0.1227`
- **Grain_Fill**: cluster-robust slope = `0.040403` (p = `0.039373`, 95% CI `[0.003191, 0.077614]`), R² = `0.0107`, descriptive Pearson r = `0.1034`

## Group differences across year types
- Unit of independence: `year`
- Test: `Kruskal-Wallis on year-level means`, statistic = `0.7829`, p = `0.67609`
- Inference supported: `True`

## Interpretation
Prediction interval width shows a positive association with CDHW severity that is not statistically distinguishable from zero once dependence is accounted for (cluster-robust slope = 0.0143, 95% CI [-0.005525, 0.034115], p = 0.11578). The association is weak in magnitude: CDHW severity explains R² = 0.0276 (2.8%) of the variation in interval width, so severity alone accounts for only a small fraction of the observed variation. Statistical detectability and effect magnitude are distinct conclusions and are reported separately here.
