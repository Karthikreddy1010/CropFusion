# Statistical Significance Testing Report (§5.5)

The unit of independence is the **year**. County-year rows within a year share weather and are not independent, so row-level tests over ~2300 rows are anti-conservative — they produced p = 0.0 against every comparator. Those are kept in the appendix as diagnostics and must not be cited.

## 1. Primary Pairwise Tests: year-level paired comparison

- **sa_aci_vs_static_conformal**: mean difference = `-0.103295` over `5` years, paired t p = `0.154338`, 95% CI = `[-0.266826, 0.060236]`, Cohen's dz = `-0.7843`; year-block bootstrap p = `0.033`, CI = `[-0.202477, -0.006489]`
- **sa_aci_vs_standard_aci**: mean difference = `-0.039243` over `5` years, paired t p = `0.082163`, 95% CI = `[-0.086439, 0.007954]`, Cohen's dz = `-1.0324`; year-block bootstrap p = `0.0005`, CI = `[-0.068166, -0.010879]`
- **sa_aci_vs_phenology_stratified_cqr**: mean difference = `-0.103429` over `5` years, paired t p = `0.154059`, 95% CI = `[-0.267028, 0.060169]`, Cohen's dz = `-0.785`; year-block bootstrap p = `0.033`, CI = `[-0.202447, -0.005799]`
- **sa_aci_vs_weighted_conformal**: mean difference = `-0.097323` over `5` years, paired t p = `0.167372`, 95% CI = `[-0.257734, 0.063089]`, Cohen's dz = `-0.7533`; year-block bootstrap p = `0.033`, CI = `[-0.195006, -0.003684]`
- **sa_aci_vs_locally_adaptive**: mean difference = `-0.126907` over `5` years, paired t p = `0.168262`, 95% CI = `[-0.336644, 0.08283]`, Cohen's dz = `-0.7513`; year-block bootstrap p = `0.033`, CI = `[-0.252345, -0.005557]`

With 5 test years, a non-significant result is weak evidence of no difference, not evidence of equivalence.

## 2. Row-level paired t-tests and Cohen's d (diagnostic only)

*Each county-year row is treated as independent, which it is not. Reported for completeness; not valid for inference.*

- **sa_aci_vs_static_conformal**: t = `-5.9268`, p-value = `0.0`, Cohen's d = `-0.1224`, 95% CI Diff = `[np.float64(-0.1433), np.float64(-0.0721)]`
- **sa_aci_vs_standard_aci**: t = `-4.9849`, p-value = `1e-06`, Cohen's d = `-0.1029`, 95% CI Diff = `[np.float64(-0.0546), np.float64(-0.0238)]`
- **sa_aci_vs_phenology_stratified_cqr**: t = `-5.904`, p-value = `0.0`, Cohen's d = `-0.1219`, 95% CI Diff = `[np.float64(-0.1436), np.float64(-0.072)]`
- **sa_aci_vs_weighted_conformal**: t = `-5.7374`, p-value = `0.0`, Cohen's d = `-0.1185`, 95% CI Diff = `[np.float64(-0.1364), np.float64(-0.0669)]`
- **sa_aci_vs_locally_adaptive**: t = `-4.4366`, p-value = `1e-05`, Cohen's d = `-0.0916`, 95% CI Diff = `[np.float64(-0.1837), np.float64(-0.0711)]`

## 3. Multi-Method Significance: Friedman & Nemenyi Post-Hoc Test

- *Omnibus Friedman/Nemenyi rank test not executed; pairwise Wilcoxon signed-rank and block bootstraps serve as authoritative statistical inference.*

## 4. Multiple Comparison Corrections

Applied to **year (paired t on year-level means)** p-values.

### Holm-Bonferroni Correction
- **static_conformal**: adjusted p = `0.463014` (significant at α=0.05: `False`)
- **standard_aci**: adjusted p = `0.410815` (significant at α=0.05: `False`)
- **phenology_stratified_cqr**: adjusted p = `0.616236` (significant at α=0.05: `False`)
- **weighted_conformal**: adjusted p = `0.334744` (significant at α=0.05: `False`)
- **locally_adaptive**: adjusted p = `0.168262` (significant at α=0.05: `False`)

### Benjamini-Hochberg (FDR) Correction
- **static_conformal**: BH adjusted p = `0.168262` (significant at FDR=0.05: `False`)
- **standard_aci**: BH adjusted p = `0.168262` (significant at FDR=0.05: `False`)
- **phenology_stratified_cqr**: BH adjusted p = `0.168262` (significant at FDR=0.05: `False`)
- **weighted_conformal**: BH adjusted p = `0.168262` (significant at FDR=0.05: `False`)
- **locally_adaptive**: BH adjusted p = `0.168262` (significant at FDR=0.05: `False`)

## 5. Primary Robustness Inference: Spatial & Temporal Clustered Block Bootstraps

### County-Block Bootstrap (2000 iterations, 555 counties)
- **R2**: Mean = `0.7026`, 95% CI = `[0.677, 0.7269]`
- **RMSE**: Mean = `0.9234`, 95% CI = `[0.8887, 0.9592]`
- **PICP**: Mean = `0.9241`, 95% CI = `[0.9128, 0.935]`
- **MPIW**: Mean = `3.3204`, 95% CI = `[3.2938, 3.3479]`
- **ACE**: Mean = `0.0241`, 95% CI = `[0.0128, 0.035]`
- **WINKLER**: Mean = `4.0388`, 95% CI = `[3.89, 4.1998]`

### Year-Block Bootstrap (2000 iterations, 5 years)
- **R2**: Mean = `0.6994`, 95% CI = `[0.6483, 0.746]`
- **RMSE**: Mean = `0.9226`, 95% CI = `[0.8448, 1.0071]`
- **PICP**: Mean = `0.9242`, 95% CI = `[0.89, 0.9544]`
- **MPIW**: Mean = `3.3242`, 95% CI = `[3.2066, 3.4671]`
- **ACE**: Mean = `0.0254`, 95% CI = `[0.0012, 0.0544]`
- **WINKLER**: Mean = `4.0415`, 95% CI = `[3.8558, 4.255]`

## Appendix. Row-level tests (anti-conservative, do not cite)

Retained so the contrast with the year-level results above is visible.

- **sa_aci_vs_static_conformal**: row-level Wilcoxon W = `372545.0`, p-value = `0.0`
- **sa_aci_vs_standard_aci**: row-level Wilcoxon W = `446433.0`, p-value = `0.0`
- **sa_aci_vs_phenology_stratified_cqr**: row-level Wilcoxon W = `372817.0`, p-value = `0.0`
- **sa_aci_vs_weighted_conformal**: row-level Wilcoxon W = `366984.0`, p-value = `0.0`
- **sa_aci_vs_locally_adaptive**: row-level Wilcoxon W = `554491.0`, p-value = `0.0`

### Holm-Bonferroni on row-level p-values (invalid)
- **static_conformal**: adjusted p = `0.0`
- **standard_aci**: adjusted p = `0.0`
- **phenology_stratified_cqr**: adjusted p = `0.0`
- **weighted_conformal**: adjusted p = `0.0`
- **locally_adaptive**: adjusted p = `0.0`

### Benjamini-Hochberg on row-level p-values (invalid)
- **static_conformal**: adjusted p = `0.0`
- **standard_aci**: adjusted p = `0.0`
- **phenology_stratified_cqr**: adjusted p = `0.0`
- **weighted_conformal**: adjusted p = `0.0`
- **locally_adaptive**: adjusted p = `0.0`
