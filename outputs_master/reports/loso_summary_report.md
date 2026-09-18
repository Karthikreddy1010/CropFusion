# Leave-One-State-Out (LOSO) Cross-Validation Report (§5.2)

## Master LOSO Results Table

| Held-out State     | RMSE             | MAE              | R²               | PICP             | ACE              | MPIW             | Winkler          |
|:-------------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| Illinois           | 1.1932           | 0.9521           | 0.7583           | 0.8862           | 0.0138           | 3.8859           | 5.1021           |
| Indiana            | 0.8713           | 0.6607           | 0.8130           | 0.9781           | 0.0781           | 4.1093           | 4.3824           |
| Iowa               | 1.2634           | 1.0296           | 0.6770           | 0.9130           | 0.0130           | 4.3109           | 4.9896           |
| Minnesota          | 1.5502           | 1.1894           | 0.6148           | 0.8453           | 0.0547           | 4.2204           | 7.0875           |
| Missouri           | 1.3087           | 0.9958           | 0.6084           | 0.9269           | 0.0269           | 4.7521           | 5.7013           |
| Ohio               | 1.0612           | 0.8073           | 0.7143           | 0.9467           | 0.0467           | 4.2245           | 4.8392           |
| Macro Mean         | 1.2080           | 0.9392           | 0.6976           | 0.9160           | 0.0389           | 4.2505           | 5.3503           |
| Macro 95% CI       | [0.9662, 1.4498] | [0.7462, 1.1321] | [0.6130, 0.7823] | [0.8672, 0.9648] | [0.0119, 0.0658] | [3.9501, 4.5509] | [4.3514, 6.3493] |
| Pooled Performance | 1.2197           | 0.9371           | 0.7275           | 0.9165           | 0.0165           | 4.2441           | 5.3049           |

> **Note on Uncertainty Summaries**: Confidence intervals summarize variation across the six held-out states and should not be interpreted as population-level uncertainty estimates.

## Fold-Level Calibration Diagnostics

| held_out_state   |   n_test |   nominal_coverage |   PICP |    ACE |   MPIW |   Winkler |   lower_violation_rate |   upper_violation_rate |   n_covered |   n_lower_violations |   n_upper_violations |   calibration_threshold_q |
|:-----------------|---------:|-------------------:|-------:|-------:|-------:|----------:|-----------------------:|-----------------------:|------------:|---------------------:|---------------------:|--------------------------:|
| Illinois         |     3821 |                0.9 | 0.8862 | 0.0138 | 3.8859 |    5.1021 |                 0.0183 |                 0.0955 |        3386 |                   70 |                  365 |                    1.0521 |
| Indiana          |     3377 |                0.9 | 0.9781 | 0.0781 | 4.1093 |    4.3824 |                 0.0145 |                 0.0074 |        3303 |                   49 |                   25 |                    1.1164 |
| Iowa             |     3803 |                0.9 | 0.913  | 0.013  | 4.3109 |    4.9896 |                 0.0053 |                 0.0818 |        3472 |                   20 |                  311 |                    1.2451 |
| Minnesota        |     2947 |                0.9 | 0.8453 | 0.0547 | 4.2204 |    7.0875 |                 0.1446 |                 0.0102 |        2491 |                  426 |                   30 |                    1.3191 |
| Missouri         |     3353 |                0.9 | 0.9269 | 0.0269 | 4.7521 |    5.7013 |                 0.0591 |                 0.014  |        3108 |                  198 |                   47 |                    1.1865 |
| Ohio             |     3206 |                0.9 | 0.9467 | 0.0467 | 4.2245 |    4.8392 |                 0.0508 |                 0.0025 |        3035 |                  163 |                    8 |                    1.2482 |

## Ensemble Weight Audit (Tuned on Dev Validation Only)

| held_out_state   |   NeuralCQR_weight |   LightGBM_weight |   dev_r2_neural_alone |   dev_r2_lgb_alone |   dev_r2_selected_blend |   dev_RMSE_selected_blend |   dev_MAE_selected_blend | weight_selection_partition   | selected_objective      |
|:-----------------|-------------------:|------------------:|----------------------:|-------------------:|------------------------:|--------------------------:|-------------------------:|:-----------------------------|:------------------------|
| Illinois         |                0.6 |               0.4 |                0.4363 |             0.4173 |                  0.4517 |                    1.1986 |                   0.9657 | LOSO_DEV_2014_2015           | Maximize R2 on LOSO DEV |
| Indiana          |                0.4 |               0.6 |                0.48   |             0.5142 |                  0.5319 |                    1.1434 |                   0.9155 | LOSO_DEV_2014_2015           | Maximize R2 on LOSO DEV |
| Iowa             |                0.9 |               0.1 |                0.5738 |             0.4189 |                  0.5753 |                    1.1257 |                   0.8675 | LOSO_DEV_2014_2015           | Maximize R2 on LOSO DEV |
| Minnesota        |                0.8 |               0.2 |                0.4697 |             0.3949 |                  0.4739 |                    1.1557 |                   0.9283 | LOSO_DEV_2014_2015           | Maximize R2 on LOSO DEV |
| Missouri         |                0.8 |               0.2 |                0.526  |             0.4623 |                  0.5288 |                    1.0873 |                   0.8648 | LOSO_DEV_2014_2015           | Maximize R2 on LOSO DEV |
| Ohio             |                0.7 |               0.3 |                0.5399 |             0.4665 |                  0.5534 |                    1.102  |                   0.8767 | LOSO_DEV_2014_2015           | Maximize R2 on LOSO DEV |

