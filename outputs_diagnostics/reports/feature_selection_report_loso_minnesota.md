# Feature Selection Consensus Report (§7)

## Method Consensus Summary
- **Total Candidate Features Evaluated**: `117`
- **Consensus Selected Features**: `101`
- **Features Dropped**: `16`
- **Methodology Protected Features Preserved**: `43`

## Top 20 Selected Features by Consensus Rank
| feature                      |   votes | selected   | protected   |   avg_rank |   rank_mi |   rank_lgbm |   rank_rf |   rank_perm |
|:-----------------------------|--------:|:-----------|:------------|-----------:|----------:|------------:|----------:|------------:|
| Yield_lag2                   |       4 | True       | True        |       0.75 |         2 |           2 |         1 |           2 |
| Yield_lag1                   |       4 | True       | True        |       1.75 |         4 |           1 |         3 |           3 |
| spei30_prism_nldas_mean_silk |       4 | True       | False       |       6    |        14 |           6 |         4 |           4 |
| prism_tmax_mean_silk         |       4 | True       | False       |      12.75 |         3 |          49 |         2 |           1 |
| nlcd_cropland_frac           |       4 | True       | True        |      13    |        11 |          35 |         5 |           5 |
| nldas_vpd_mean_veg           |       4 | True       | False       |      14.25 |        36 |          11 |         7 |           7 |
| nlcd_pasture_hay_frac        |       4 | True       | True        |      15.5  |        20 |          23 |        11 |          12 |
| prism_kdd30_gs               |       4 | True       | False       |      15.5  |         6 |          48 |         6 |           6 |
| nldas_et0_total_silk         |       4 | True       | False       |      16    |        43 |           7 |         9 |           9 |
| spi1_prism_min_silk          |       4 | True       | False       |      19.5  |        39 |          27 |         8 |           8 |
| prism_ppt_total_silk         |       4 | True       | False       |      20.75 |        45 |           3 |        20 |          19 |
| Fourier_sin_P19_0            |       4 | True       | True        |      25.75 |         1 |          73 |        17 |          16 |
| Fourier_sin_P11_0            |       3 | True       | True        |      26    |         5 |          78 |        14 |          11 |
| Anom_GDD_Accumulated         |       4 | True       | False       |      27    |        65 |           9 |        18 |          20 |
| Fourier_cos_P19_0            |       3 | True       | True        |      27.25 |        12 |          81 |        10 |          10 |
| nldas_rs_mean_gs             |       4 | True       | False       |      28    |        46 |          21 |        26 |          23 |
| spi3_prism_min_gs            |       4 | True       | False       |      30.25 |        58 |          26 |        19 |          22 |
| spei30_prism_nldas_min_silk  |       4 | True       | False       |      30.5  |        35 |          45 |        25 |          21 |
| prism_tmax_max_gs            |       4 | True       | True        |      32.25 |        18 |          37 |        40 |          38 |
| elevation_mean_m             |       4 | True       | True        |      33.25 |        67 |          16 |        28 |          26 |
