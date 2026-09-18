# Feature Selection Consensus Report (§7)

## Method Consensus Summary
- **Total Candidate Features Evaluated**: `117`
- **Consensus Selected Features**: `97`
- **Features Dropped**: `20`
- **Methodology Protected Features Preserved**: `43`

## Top 20 Selected Features by Consensus Rank
| feature                      |   votes | selected   | protected   |   avg_rank |   rank_mi |   rank_lgbm |   rank_rf |   rank_perm |
|:-----------------------------|--------:|:-----------|:------------|-----------:|----------:|------------:|----------:|------------:|
| Yield_lag1                   |       4 | True       | True        |       0    |         1 |           1 |         1 |           1 |
| Yield_lag2                   |       4 | True       | True        |       1.25 |         2 |           2 |         3 |           2 |
| nldas_vpd_mean_veg           |       4 | True       | False       |       8    |        15 |           9 |         6 |           6 |
| prism_tmax_mean_silk         |       4 | True       | False       |      13.5  |         4 |          49 |         2 |           3 |
| spei30_prism_nldas_mean_silk |       4 | True       | False       |      14.75 |        39 |           6 |         8 |          10 |
| nldas_et0_total_silk         |       4 | True       | False       |      17.25 |        32 |          18 |        12 |          11 |
| Fourier_sin_P19_0            |       4 | True       | True        |      18    |         3 |          63 |         5 |           5 |
| nlcd_pasture_hay_frac        |       4 | True       | True        |      19.25 |        33 |          13 |        21 |          14 |
| soil_bulk_density            |       4 | True       | True        |      20    |        52 |          17 |         7 |           8 |
| nldas_rs_mean_gs             |       4 | True       | False       |      20.25 |        46 |          10 |        14 |          15 |
| nlcd_cropland_frac           |       4 | True       | True        |      21    |        12 |          28 |        26 |          22 |
| Fourier_sin_P11_0            |       4 | True       | True        |      22    |         5 |          71 |         9 |           7 |
| spi1_prism_min_silk          |       4 | True       | False       |      22    |        55 |           5 |        16 |          16 |
| prism_ppt_total_silk         |       4 | True       | False       |      26.25 |        69 |           3 |        19 |          18 |
| prism_kdd30_gs               |       4 | True       | False       |      27.25 |         8 |          45 |        33 |          27 |
| spi3_prism_min_gs            |       4 | True       | False       |      28.25 |        66 |          21 |        18 |          12 |
| Fourier_cos_P19_0            |       4 | True       | True        |      30.5  |         6 |          76 |        25 |          19 |
| prism_tmax_days_gt35_silk    |       3 | True       | False       |      31    |         9 |          91 |        11 |          17 |
| nldas_et0_total_gs           |       4 | True       | False       |      31.75 |        50 |          51 |        17 |          13 |
| Inter_CDHW_GDD               |       3 | True       | False       |      32    |        27 |          97 |         4 |           4 |
