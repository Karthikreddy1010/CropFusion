# Statistical Significance Testing Report (§5.5)

## 1. Primary Pairwise Tests: Wilcoxon Signed-Rank

- **sa_aci_vs_static_conformal**: Wilcoxon W = `448075.0`, p-value = `0.0`
- **sa_aci_vs_standard_aci**: Wilcoxon W = `920196.0`, p-value = `0.0`
- **sa_aci_vs_phenology_stratified_cqr**: Wilcoxon W = `440042.0`, p-value = `0.0`
- **sa_aci_vs_weighted_conformal**: Wilcoxon W = `449834.0`, p-value = `0.0`
- **sa_aci_vs_locally_adaptive**: Wilcoxon W = `593806.0`, p-value = `0.0`

## 2. Secondary Robustness Check: Paired t-Tests & Effect Sizes (Cohen's d)

- **sa_aci_vs_static_conformal**: t = `-3.4187`, p-value = `0.00064`, Cohen's d = `-0.0706`, 95% CI Diff = `[np.float64(-0.1135), np.float64(-0.0308)]`
- **sa_aci_vs_standard_aci**: t = `-0.3569`, p-value = `0.721172`, Cohen's d = `-0.0074`, 95% CI Diff = `[np.float64(-0.0236), np.float64(0.0163)]`
- **sa_aci_vs_phenology_stratified_cqr**: t = `-3.4122`, p-value = `0.000655`, Cohen's d = `-0.0705`, 95% CI Diff = `[np.float64(-0.1139), np.float64(-0.0308)]`
- **sa_aci_vs_weighted_conformal**: t = `-3.2001`, p-value = `0.001392`, Cohen's d = `-0.0661`, 95% CI Diff = `[np.float64(-0.1067), np.float64(-0.0256)]`
- **sa_aci_vs_locally_adaptive**: t = `-2.848`, p-value = `0.004438`, Cohen's d = `-0.0588`, 95% CI Diff = `[np.float64(-0.1551), np.float64(-0.0287)]`

## 3. Multi-Method Significance: Friedman & Nemenyi Post-Hoc Test

- *Omnibus Friedman/Nemenyi rank test not executed; pairwise Wilcoxon signed-rank and block bootstraps serve as authoritative statistical inference.*

## 4. Multiple Comparison Corrections

### Holm-Bonferroni Correction
- **static_conformal**: adjusted p = `0.0` (significant at α=0.05: `True`)
- **standard_aci**: adjusted p = `0.0` (significant at α=0.05: `True`)
- **phenology_stratified_cqr**: adjusted p = `0.0` (significant at α=0.05: `True`)
- **weighted_conformal**: adjusted p = `0.0` (significant at α=0.05: `True`)
- **locally_adaptive**: adjusted p = `0.0` (significant at α=0.05: `True`)

### Benjamini-Hochberg (FDR) Correction
- **static_conformal**: BH adjusted p = `0.0` (significant at FDR=0.05: `True`)
- **standard_aci**: BH adjusted p = `0.0` (significant at FDR=0.05: `True`)
- **phenology_stratified_cqr**: BH adjusted p = `0.0` (significant at FDR=0.05: `True`)
- **weighted_conformal**: BH adjusted p = `0.0` (significant at FDR=0.05: `True`)
- **locally_adaptive**: BH adjusted p = `0.0` (significant at FDR=0.05: `True`)

## 5. Primary Robustness Inference: Spatial & Temporal Clustered Block Bootstraps

### County-Block Bootstrap (2000 iterations, 555 counties)
- **R2**: Mean = `0.7026`, 95% CI = `[0.677, 0.7269]`
- **RMSE**: Mean = `0.9234`, 95% CI = `[0.8887, 0.9592]`
- **PICP**: Mean = `0.9194`, 95% CI = `[0.9075, 0.9305]`
- **MPIW**: Mean = `3.3178`, 95% CI = `[3.2926, 3.3441]`
- **ACE**: Mean = `0.0194`, 95% CI = `[0.0075, 0.0305]`
- **WINKLER**: Mean = `4.0741`, 95% CI = `[3.9209, 4.2408]`

### Year-Block Bootstrap (2000 iterations, 5 years)
- **R2**: Mean = `0.6994`, 95% CI = `[0.6483, 0.746]`
- **RMSE**: Mean = `0.9226`, 95% CI = `[0.8448, 1.0071]`
- **PICP**: Mean = `0.9196`, 95% CI = `[0.8824, 0.954]`
- **MPIW**: Mean = `3.322`, 95% CI = `[3.1825, 3.4744]`
- **ACE**: Mean = `0.0225`, 95% CI = `[0.0019, 0.054]`
- **WINKLER**: Mean = `4.077`, 95% CI = `[3.875, 4.297]`
