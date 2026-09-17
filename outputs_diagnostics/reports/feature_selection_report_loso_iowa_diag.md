# Feature Selection Consensus Report (§7)

## Method Consensus Summary
- **Total Candidate Features Evaluated**: `117`
- **Consensus Selected Features**: `100`
- **Features Dropped**: `17`
- **Methodology Protected Features Preserved**: `43`

## Top 20 Selected Features by Consensus Rank
| feature                      |   votes | selected   | protected   |   avg_rank |   rank_mi |   rank_lgbm |   rank_rf |   rank_perm |
|:-----------------------------|--------:|:-----------|:------------|-----------:|----------:|------------:|----------:|------------:|
| Yield_lag1                   |       4 | True       | True        |       0.5  |         3 |           1 |         1 |           1 |
| Yield_lag2                   |       4 | True       | True        |       1    |         2 |           2 |         2 |           2 |
| spei30_prism_nldas_mean_silk |       4 | True       | False       |      10.5  |        26 |          10 |         5 |           5 |
| nlcd_pasture_hay_frac        |       4 | True       | True        |      14    |        15 |          17 |        17 |          11 |
| prism_tmax_mean_silk         |       4 | True       | False       |      14.5  |         4 |          51 |         4 |           3 |
| soil_bulk_density            |       4 | True       | True        |      15.25 |        36 |          16 |         7 |           6 |
| prism_kdd30_gs               |       4 | True       | False       |      17.75 |        11 |          47 |         9 |           8 |
| spi1_prism_min_silk          |       4 | True       | False       |      21    |        54 |           6 |        14 |          14 |
| nldas_vpd_mean_veg           |       4 | True       | False       |      22.25 |        61 |          11 |        12 |           9 |
| nlcd_cropland_frac           |       4 | True       | True        |      23.5  |        10 |          39 |        25 |          24 |
| nldas_et0_total_silk         |       4 | True       | False       |      24.25 |        62 |           9 |        15 |          15 |
| Fourier_sin_P11_0            |       4 | True       | True        |      27.25 |         5 |          74 |        18 |          16 |
| Anom_GDD_Accumulated         |       4 | True       | False       |      27.75 |        75 |           4 |        19 |          17 |
| Fourier_cos_P19_0            |       3 | True       | True        |      28    |         8 |          79 |        16 |          13 |
| prism_tmax_days_gt35_silk    |       3 | True       | False       |      28.5  |         7 |          93 |         6 |          12 |
| Fourier_sin_P19_0            |       4 | True       | True        |      29    |         1 |          70 |        26 |          23 |
| Inter_CDHW_GDD               |       3 | True       | False       |      29.5  |        19 |          96 |         3 |           4 |
| Fourier_cos_P11_0            |       3 | True       | True        |      33    |        14 |          84 |        20 |          18 |
| nldas_rs_mean_gs             |       4 | True       | False       |      33.5  |        55 |          32 |        23 |          28 |
| spi3_prism_min_gs            |       4 | True       | False       |      34.75 |        69 |          19 |        28 |          27 |
