# Objective O5 Report: Uncertainty Scaling vs CDHW Severity (§3 O5 & §5.8)

## Unit of independence
County-year rows are **not** independent: counties share year-level weather shocks and
each county is an autocorrelated series. All inference below therefore clusters on
**year** (5 clusters over 2345 rows).

- **Effective sample size**: `35` (from `2345` rows;
  ICC = `0.1391`, design effect = `66.083`)

## Primary inference — OLS with cluster-robust (CR1) SE, clustered by year
- **Slope**: `0.024459` per unit CDHW severity
- **95% CI**: `[-0.001738, 0.050655]`
- **p-value**: `0.060539` (t = `2.5923`, df = `4`)
- **R²**: `0.0759`

## Corroborating year-block cluster bootstrap
- **Slope**: `0.036396`, **95% CI**: `[0.013414, 0.104425]`, **p**: `0.02`
- Resampling unit: `year` (5 blocks)

## Secondary inference — clustered by county
- **Slope**: `0.024459`, **95% CI**: `[0.018988, 0.029929]`, **p**: `0.0`
  (555 county clusters)

## Descriptive association (NOT used for inference)
- Pearson r = `0.2755`, Spearman ρ = `0.2436`
- Naive OLS p-value = `0.0` — **invalid**, shown only to
  document how much the independence assumption distorts the result.

## Phenology stage-specific associations
- **Vegetative**: cluster-robust slope = `0.102162` (p = `0.006882`, 95% CI `[0.046771, 0.157553]`), R² = `0.059625`, descriptive Pearson r = `0.2442`
- **Silking_R1**: cluster-robust slope = `0.021742` (p = `0.042131`, 95% CI `[0.001252, 0.042232]`), R² = `0.046445`, descriptive Pearson r = `0.2155`
- **Grain_Fill**: cluster-robust slope = `0.062166` (p = `0.003537`, 95% CI `[0.034119, 0.090213]`), R² = `0.023801`, descriptive Pearson r = `0.1543`

## Group differences across year types
- Unit of independence: `year`
- Test: `Kruskal-Wallis on year-level means`, statistic = `3.9057`, p = `0.141868`
- Inference supported: `True`

## Interpretation
Prediction interval width shows a positive association with CDHW severity that is not statistically distinguishable from zero once dependence is accounted for (cluster-robust slope = 0.0245, 95% CI [-0.001738, 0.050655], p = 0.060539). The association is weak in magnitude: CDHW severity explains R² = 0.0759 (7.6%) of the variation in interval width, so severity alone accounts for only a small fraction of the observed variation. Statistical detectability and effect magnitude are distinct conclusions and are reported separately here.
