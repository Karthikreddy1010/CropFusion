# Feature Selection Consensus Report (§7)

## Method Consensus Summary
- **Total Candidate Features Evaluated**: `117`
- **Consensus Selected Features**: `97`
- **Features Dropped**: `20`
- **Methodology Protected Features Preserved**: `43`

## Top 20 Selected Features by Consensus Rank
| feature                      |   votes | selected   | protected   |   avg_rank |   rank_mi |   rank_lgbm |   rank_rf |   rank_perm |
|:-----------------------------|--------:|:-----------|:------------|-----------:|----------:|------------:|----------:|------------:|
| Yield_lag2                   |       4 | True       | True        |       0.25 |         1 |           2 |         1 |           1 |
| Yield_lag1                   |       4 | True       | True        |       1.25 |         3 |           1 |         3 |           2 |
| prism_tmax_mean_silk         |       4 | True       | False       |      12.75 |         4 |          46 |         2 |           3 |
| spei30_prism_nldas_mean_silk |       4 | True       | False       |      15    |        39 |          13 |         6 |           6 |
| prism_kdd30_gs               |       4 | True       | False       |      16.5  |         6 |          48 |         7 |           9 |
| nlcd_cropland_frac           |       4 | True       | True        |      16.5  |         8 |          37 |        13 |          12 |
| nldas_vpd_mean_veg           |       4 | True       | False       |      19.5  |        47 |          11 |        14 |          10 |
| nlcd_pasture_hay_frac        |       4 | True       | True        |      20.5  |        27 |          15 |        23 |          21 |
| spi1_prism_min_silk          |       4 | True       | False       |      24.25 |        66 |           5 |        17 |          13 |
| prism_tmax_max_gs            |       4 | True       | True        |      24.75 |        17 |          36 |        25 |          25 |
| nldas_et0_total_silk         |       4 | True       | False       |      24.75 |        67 |           7 |        15 |          14 |
| soil_bulk_density            |       4 | True       | True        |      24.75 |        41 |          28 |        19 |          15 |
| Fourier_sin_P11_0            |       3 | True       | True        |      25    |         5 |          79 |        12 |           8 |
| prism_tmax_days_gt35_silk    |       3 | True       | False       |      26    |        10 |          90 |         4 |           4 |
| Fourier_cos_P19_0            |       3 | True       | True        |      26.75 |        12 |          84 |         8 |           7 |
| prism_tmin_mean_gs           |       4 | True       | True        |      27.5  |        48 |          56 |         5 |           5 |
| nldas_rs_mean_gs             |       4 | True       | False       |      28.5  |        46 |          19 |        26 |          27 |
| soil_clay                    |       4 | True       | True        |      29    |        33 |          18 |        36 |          33 |
| Fourier_sin_P19_0            |       4 | True       | True        |      29.5  |         2 |          73 |        27 |          20 |
| Anom_GDD_Accumulated         |       3 | True       | False       |      30.25 |        78 |          10 |        18 |          19 |
