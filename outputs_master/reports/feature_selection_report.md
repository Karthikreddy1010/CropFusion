# Feature Selection Consensus Report (§7)

## Method Consensus Summary
- **Total Candidate Features Evaluated**: `117`
- **Consensus Selected Features**: `100`
- **Features Dropped**: `17`
- **Methodology Protected Features Preserved**: `43`

## Top 20 Selected Features by Consensus Rank
| feature                      |   votes | selected   | protected   |   avg_rank |   rank_mi |   rank_lgbm |   rank_rf |   rank_perm |
|:-----------------------------|--------:|:-----------|:------------|-----------:|----------:|------------:|----------:|------------:|
| Yield_lag2                   |       4 | True       | True        |       0.5  |         2 |           2 |         1 |           1 |
| Yield_lag1                   |       4 | True       | True        |       1.25 |         3 |           1 |         3 |           2 |
| spei30_prism_nldas_mean_silk |       4 | True       | False       |       9.25 |        24 |           5 |         7 |           5 |
| prism_tmax_mean_silk         |       4 | True       | False       |      13.5  |         4 |          49 |         2 |           3 |
| soil_bulk_density            |       4 | True       | True        |      14.25 |        38 |          13 |         6 |           4 |
| nlcd_cropland_frac           |       4 | True       | True        |      14.75 |         6 |          29 |        14 |          14 |
| prism_kdd30_gs               |       4 | True       | False       |      16.25 |         7 |          45 |         8 |           9 |
| nlcd_pasture_hay_frac        |       4 | True       | True        |      19    |        16 |          15 |        25 |          24 |
| nldas_vpd_mean_veg           |       4 | True       | False       |      20    |        51 |          10 |        12 |          11 |
| nldas_et0_total_silk         |       4 | True       | False       |      23.25 |        62 |           7 |        16 |          12 |
| spi1_prism_min_silk          |       4 | True       | False       |      23.5  |        57 |           8 |        17 |          16 |
| Fourier_cos_P19_0            |       3 | True       | True        |      25.75 |         9 |          83 |         9 |           6 |
| Fourier_sin_P19_0            |       4 | True       | True        |      26    |         1 |          69 |        21 |          17 |
| Anom_GDD_Accumulated         |       3 | True       | False       |      26.5  |        78 |           6 |        13 |          13 |
| Fourier_sin_P11_0            |       3 | True       | True        |      27    |         5 |          77 |        15 |          15 |
| prism_tmax_days_gt35_silk    |       3 | True       | False       |      27.25 |        11 |          89 |         5 |           8 |
| nldas_rs_mean_gs             |       4 | True       | False       |      28.5  |        49 |          20 |        22 |          27 |
| Inter_CDHW_GDD               |       3 | True       | False       |      31.25 |        21 |          97 |         4 |           7 |
| spi3_prism_min_gs            |       4 | True       | False       |      31.5  |        69 |          21 |        19 |          21 |
| prism_ppt_total_silk         |       3 | True       | False       |      33    |        84 |           4 |        23 |          25 |
