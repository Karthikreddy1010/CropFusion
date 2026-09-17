# Paper 3 Evaluation Summary (§5)

## Calibration Methods Benchmarking

| Method                   |   RMSE |    MAE |     R2 |   PICP |   MPIW |    ACE |   Winkler_Score |
|:-------------------------|-------:|-------:|-------:|-------:|-------:|-------:|----------------:|
| static_conformal         | 0.9237 | 0.7091 | 0.7031 | 0.9488 | 3.6493 | 0.0488 |          4.1473 |
| standard_aci             | 0.9237 | 0.7091 | 0.7031 | 0.9296 | 3.4151 | 0.0296 |          4.0788 |
| sa_aci                   | 0.9237 | 0.7091 | 0.7031 | 0.9194 | 3.3177 | 0.0194 |          4.0751 |
| phenology_stratified_cqr | 0.9237 | 0.7091 | 0.7031 | 0.9493 | 3.6506 | 0.0493 |          4.1475 |
| weighted_conformal       | 0.9237 | 0.7091 | 0.7031 | 0.9471 | 3.6367 | 0.0471 |          4.1413 |
| locally_adaptive         | 0.9237 | 0.7091 | 0.7031 | 0.9488 | 3.6987 | 0.0488 |          4.167  |

## County Block Bootstrap Diagnostics

- **PICP 95% CI**: [0.9075, 0.9305] (mean: 0.9194)
- **MPIW 95% CI**: [3.2926, 3.3441] (mean: 3.3178)
- **ACE 95% CI**: [0.0075, 0.0305] (mean: 0.0194)

## Year Block Bootstrap Diagnostics

- **PICP 95% CI**: [0.8824, 0.954] (mean: 0.9196)
- **MPIW 95% CI**: [3.1825, 3.4744] (mean: 3.322)
- **ACE 95% CI**: [0.0019, 0.054] (mean: 0.0225)

## Secondary Paired t-Test Robustness

| Comparison                         |   t_statistic |   p_value |   Cohen_d | CI_95_Diff                                 |
|:-----------------------------------|--------------:|----------:|----------:|:-------------------------------------------|
| sa_aci_vs_static_conformal         |       -3.4187 |  0.00064  |   -0.0706 | [np.float64(-0.1135), np.float64(-0.0308)] |
| sa_aci_vs_standard_aci             |       -0.3569 |  0.721172 |   -0.0074 | [np.float64(-0.0236), np.float64(0.0163)]  |
| sa_aci_vs_phenology_stratified_cqr |       -3.4122 |  0.000655 |   -0.0705 | [np.float64(-0.1139), np.float64(-0.0308)] |
| sa_aci_vs_weighted_conformal       |       -3.2001 |  0.001392 |   -0.0661 | [np.float64(-0.1067), np.float64(-0.0256)] |
| sa_aci_vs_locally_adaptive         |       -2.848  |  0.004438 |   -0.0588 | [np.float64(-0.1551), np.float64(-0.0287)] |

