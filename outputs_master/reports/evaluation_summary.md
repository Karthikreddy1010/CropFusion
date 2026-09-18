# Paper 3 Evaluation Summary (§5)

## Calibration Methods Benchmarking

| Method                   |   RMSE |    MAE |     R2 |   PICP |   MPIW |    ACE |   Winkler_Score |
|:-------------------------|-------:|-------:|-------:|-------:|-------:|-------:|----------------:|
| static_conformal         | 0.9237 | 0.7091 | 0.7031 | 0.9488 | 3.6493 | 0.0488 |          4.1473 |
| standard_aci             | 0.9237 | 0.7091 | 0.7031 | 0.9296 | 3.4151 | 0.0296 |          4.0788 |
| sa_aci                   | 0.9237 | 0.7091 | 0.7031 | 0.9241 | 3.3203 | 0.0241 |          4.0396 |
| phenology_stratified_cqr | 0.9237 | 0.7091 | 0.7031 | 0.9493 | 3.6506 | 0.0493 |          4.1475 |
| weighted_conformal       | 0.9237 | 0.7091 | 0.7031 | 0.9471 | 3.6367 | 0.0471 |          4.1413 |
| locally_adaptive         | 0.9237 | 0.7091 | 0.7031 | 0.9488 | 3.6987 | 0.0488 |          4.167  |

## County Block Bootstrap Diagnostics

- **PICP 95% CI**: [0.9128, 0.935] (mean: 0.9241)
- **MPIW 95% CI**: [3.2938, 3.3479] (mean: 3.3204)
- **ACE 95% CI**: [0.0128, 0.035] (mean: 0.0241)

## Year Block Bootstrap Diagnostics

- **PICP 95% CI**: [0.89, 0.9544] (mean: 0.9242)
- **MPIW 95% CI**: [3.2066, 3.4671] (mean: 3.3242)
- **ACE 95% CI**: [0.0012, 0.0544] (mean: 0.0254)

## Secondary Paired t-Test Robustness

| Comparison                         |   t_statistic |   p_value |   Cohen_d | CI_95_Diff                                 |
|:-----------------------------------|--------------:|----------:|----------:|:-------------------------------------------|
| sa_aci_vs_static_conformal         |       -5.9268 |     0     |   -0.1224 | [np.float64(-0.1433), np.float64(-0.0721)] |
| sa_aci_vs_standard_aci             |       -4.9849 |     1e-06 |   -0.1029 | [np.float64(-0.0546), np.float64(-0.0238)] |
| sa_aci_vs_phenology_stratified_cqr |       -5.904  |     0     |   -0.1219 | [np.float64(-0.1436), np.float64(-0.072)]  |
| sa_aci_vs_weighted_conformal       |       -5.7374 |     0     |   -0.1185 | [np.float64(-0.1364), np.float64(-0.0669)] |
| sa_aci_vs_locally_adaptive         |       -4.4366 |     1e-05 |   -0.0916 | [np.float64(-0.1837), np.float64(-0.0711)] |

