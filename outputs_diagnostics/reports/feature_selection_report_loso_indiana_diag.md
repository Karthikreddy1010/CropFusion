# Feature Selection Consensus Report (§7)

## Method Consensus Summary
- **Total Candidate Features Evaluated**: `117`
- **Consensus Selected Features**: `101`
- **Features Dropped**: `16`
- **Methodology Protected Features Preserved**: `43`

## Top 20 Selected Features by Consensus Rank
| feature                      |   votes | selected   | protected   |   avg_rank |   rank_mi |   rank_lgbm |   rank_rf |   rank_perm |
|:-----------------------------|--------:|:-----------|:------------|-----------:|----------:|------------:|----------:|------------:|
| Yield_lag2                   |       4 | True       | True        |       0.25 |         1 |           2 |         1 |           1 |
| Yield_lag1                   |       4 | True       | True        |       1    |         2 |           1 |         3 |           2 |
| nlcd_cropland_frac           |       4 | True       | True        |      12.25 |         5 |          27 |        10 |          11 |
| prism_tmax_mean_silk         |       4 | True       | False       |      13.25 |         4 |          48 |         2 |           3 |
| spei30_prism_nldas_mean_silk |       4 | True       | False       |      13.5  |        41 |           7 |         6 |           4 |
| prism_kdd30_gs               |       4 | True       | False       |      16.25 |         7 |          49 |         5 |           8 |
| nlcd_pasture_hay_frac        |       4 | True       | True        |      16.5  |        12 |          17 |        23 |          18 |
| soil_bulk_density            |       4 | True       | True        |      18.25 |        21 |          36 |        11 |           9 |
| nldas_vpd_mean_veg           |       4 | True       | False       |      20.75 |        51 |          13 |        13 |          10 |
| nldas_rs_mean_gs             |       4 | True       | False       |      21    |        28 |          14 |        21 |          25 |
| Fourier_sin_P19_0            |       4 | True       | True        |      25.25 |         3 |          69 |        19 |          14 |
| Anom_GDD_Accumulated         |       4 | True       | False       |      26.5  |        76 |           8 |        14 |          12 |
| Fourier_sin_P11_0            |       4 | True       | True        |      27    |         8 |          76 |        15 |          13 |
| Fourier_cos_P19_0            |       3 | True       | True        |      30.75 |        29 |          85 |         8 |           5 |
| spi1_prism_min_silk          |       3 | True       | False       |      31.25 |        81 |           6 |        22 |          20 |
| prism_tmin_mean_gs           |       4 | True       | True        |      31.75 |        55 |          62 |         7 |           7 |
| nldas_et0_total_silk         |       4 | True       | False       |      32.5  |        64 |          10 |        29 |          31 |
| nldas_et0_total_veg          |       4 | True       | False       |      32.75 |        35 |          47 |        32 |          21 |
| spi3_prism_min_gs            |       3 | True       | False       |      33    |        86 |          18 |        16 |          16 |
| Inter_CDHW_GDD               |       3 | True       | False       |      33.25 |        32 |          95 |         4 |           6 |
