# Feature Selection Consensus Report (§7)

## Method Consensus Summary
- **Total Candidate Features Evaluated**: `117`
- **Consensus Selected Features**: `99`
- **Features Dropped**: `18`
- **Methodology Protected Features Preserved**: `43`

## Top 20 Selected Features by Consensus Rank
| feature                      |   votes | selected   | protected   |   avg_rank |   rank_mi |   rank_lgbm |   rank_rf |   rank_perm |
|:-----------------------------|--------:|:-----------|:------------|-----------:|----------:|------------:|----------:|------------:|
| Yield_lag2                   |       4 | True       | True        |       0.5  |         2 |           2 |         1 |           1 |
| Yield_lag1                   |       4 | True       | True        |       1    |         3 |           1 |         2 |           2 |
| spei30_prism_nldas_mean_silk |       4 | True       | False       |       7.75 |        19 |           7 |         5 |           4 |
| nldas_et0_total_silk         |       4 | True       | False       |      11.25 |        24 |           5 |        12 |           8 |
| spi3_prism_min_gs            |       4 | True       | False       |      12.25 |        21 |          11 |        11 |          10 |
| prism_tmax_mean_silk         |       4 | True       | False       |      12.5  |         4 |          44 |         3 |           3 |
| soil_bulk_density            |       4 | True       | True        |      17.5  |        32 |          20 |        13 |           9 |
| nldas_rs_mean_gs             |       4 | True       | False       |      18.75 |        25 |          14 |        22 |          18 |
| nldas_vpd_mean_veg           |       4 | True       | False       |      19.25 |        27 |          22 |        19 |          13 |
| spei30_prism_nldas_min_silk  |       4 | True       | False       |      19.75 |        39 |          35 |         4 |           5 |
| prism_kdd30_gs               |       4 | True       | False       |      23.25 |         9 |          48 |        16 |          24 |
| Fourier_cos_P19_0            |       3 | True       | True        |      23.5  |         6 |          80 |         6 |           6 |
| spei30_prism_nldas_mean_veg  |       4 | True       | False       |      24.25 |        26 |          17 |        29 |          29 |
| Fourier_sin_P11_0            |       3 | True       | True        |      26    |         5 |          79 |        17 |           7 |
| prism_ppt_total_veg          |       4 | True       | False       |      26.75 |        67 |          23 |        10 |          11 |
| spi1_prism_min_silk          |       4 | True       | False       |      27.25 |        50 |           8 |        28 |          27 |
| Anom_GDD_Accumulated         |       4 | True       | False       |      28    |        70 |          12 |        20 |          14 |
| Fourier_sin_P19_0            |       4 | True       | True        |      28.25 |         1 |          71 |        26 |          19 |
| nldas_et0_total_veg          |       4 | True       | False       |      29.75 |        34 |          50 |        23 |          16 |
| nldas_et0_total_gs           |       4 | True       | False       |      30    |        38 |          49 |        25 |          12 |
