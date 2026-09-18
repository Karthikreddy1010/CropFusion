# Objective O6 Report: Distribution-Shift-Robust Conformal Benchmarking (§3 O6 & §4.5 & §5.5)

**Methods compared (k = 6)**: sa_aci, standard_aci, weighted_conformal, static_conformal, phenology_stratified_cqr, locally_adaptive

| Method                   |   PICP |   MPIW |    ACE |   Winkler_Score |   RMSE |    MAE |   Rank_Winkler |   Rank_ACE |   Rank_MPIW |   Average_Rank |
|:-------------------------|-------:|-------:|-------:|----------------:|-------:|-------:|---------------:|-----------:|------------:|---------------:|
| sa_aci                   | 0.9241 | 3.3203 | 0.0241 |          4.0396 | 0.9237 | 0.7091 |              1 |        1   |           1 |        1       |
| standard_aci             | 0.9296 | 3.4151 | 0.0296 |          4.0788 | 0.9237 | 0.7091 |              2 |        2   |           2 |        2       |
| weighted_conformal       | 0.9471 | 3.6367 | 0.0471 |          4.1413 | 0.9237 | 0.7091 |              3 |        3   |           3 |        3       |
| static_conformal         | 0.9488 | 3.6493 | 0.0488 |          4.1473 | 0.9237 | 0.7091 |              4 |        4.5 |           4 |        4.16667 |
| phenology_stratified_cqr | 0.9493 | 3.6506 | 0.0493 |          4.1475 | 0.9237 | 0.7091 |              5 |        6   |           5 |        5.33333 |
| locally_adaptive         | 0.9488 | 3.6987 | 0.0488 |          4.167  | 0.9237 | 0.7091 |              6 |        4.5 |           6 |        5.5     |

**Lowest average rank**: `sa_aci` (mean rank over Winkler, ACE, MPIW).

**Status of this ranking**: descriptive ranking only.

## Statistical testing
- **Blocking unit**: `year` (5 blocks) — per-observation blocking would treat correlated county-year rows as independent.
- **Friedman**: Q = `7.2989`, p = `0.199346`
- **Average ranks**: `{'static_conformal': 4.3, 'standard_aci': 3.1, 'sa_aci': 1.8, 'phenology_stratified_cqr': 4.2, 'weighted_conformal': 3.2, 'locally_adaptive': 4.4}`
- **Nemenyi CD** (k = 6, N = 5, α = 0.05): `3.3722` (q_α = `2.85`)
- **Conclusion**: the omnibus test does **not** reject at α = 0.05 (p = `0.199346`). No post-hoc comparison is performed and no method is claimed to be statistically superior. The ordering above is descriptive.

## Practical significance
- Winkler score spans `4.0396` to `4.167` (3.06% relative spread)
- PICP spans `0.9241` to `0.9493` (nominal 0.9)
- MPIW relative spread: `10.23%`
