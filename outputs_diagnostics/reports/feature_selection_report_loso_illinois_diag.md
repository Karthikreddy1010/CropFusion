# Feature Selection Consensus Report (§7)

## Method Consensus Summary
- **Total Candidate Features Evaluated**: `117`
- **Consensus Selected Features**: `99`
- **Features Dropped**: `18`
- **Methodology Protected Features Preserved**: `43`

## Top 20 Selected Features by Consensus Rank
| feature                         |   votes | selected   | protected   |   avg_rank |   rank_mi |   rank_lgbm |   rank_rf |   rank_perm |
|:--------------------------------|--------:|:-----------|:------------|-----------:|----------:|------------:|----------:|------------:|
| Yield_lag2                      |       4 | True       | True        |       0.5  |         2 |           2 |         1 |           1 |
| Yield_lag1                      |       4 | True       | True        |       1    |         3 |           1 |         2 |           2 |
| nlcd_cropland_frac              |       4 | True       | True        |      11    |         8 |          23 |         9 |           8 |
| prism_tmax_mean_silk            |       4 | True       | False       |      13    |         4 |          46 |         3 |           3 |
| spei30_prism_nldas_mean_silk    |       4 | True       | False       |      14.25 |        38 |           8 |         8 |           7 |
| nlcd_pasture_hay_frac           |       4 | True       | True        |      15.5  |        23 |           5 |        21 |          17 |
| prism_kdd30_gs                  |       4 | True       | False       |      16.25 |         9 |          50 |         6 |           4 |
| soil_bulk_density               |       4 | True       | True        |      21.75 |        54 |          17 |        10 |          10 |
| Fourier_sin_P19_0               |       4 | True       | True        |      22.25 |         1 |          64 |        15 |          13 |
| Fourier_cos_P19_0               |       3 | True       | True        |      26.5  |         6 |          82 |        11 |          11 |
| nldas_vpd_mean_veg              |       4 | True       | False       |      27    |        72 |          12 |        16 |          12 |
| Rolling2yr_GDD_Accumulated_mean |       4 | True       | False       |      27.5  |        21 |          56 |        13 |          24 |
| nldas_et0_total_silk            |       4 | True       | False       |      27.75 |        53 |          19 |        23 |          20 |
| Fourier_sin_P11_0               |       3 | True       | True        |      27.75 |         5 |          78 |        17 |          15 |
| prism_tmax_days_gt35_silk       |       3 | True       | False       |      28.25 |        16 |          90 |         5 |           6 |
| spi3_prism_min_gs               |       4 | True       | False       |      28.5  |        75 |          15 |        12 |          16 |
| spi1_prism_min_silk             |       4 | True       | False       |      29.75 |        70 |          10 |        24 |          19 |
| Fourier_cos_P11_0               |       3 | True       | True        |      30.25 |        10 |          81 |        20 |          14 |
| nldas_rs_mean_gs                |       4 | True       | False       |      30.5  |        68 |           6 |        27 |          25 |
| Anom_GDD_Accumulated            |       3 | True       | False       |      31    |        81 |          11 |        18 |          18 |
