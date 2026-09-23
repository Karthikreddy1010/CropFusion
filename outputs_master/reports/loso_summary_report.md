# Leave-One-State-Out (LOSO) Cross-Validation Report (§5.2)

## Master LOSO Results Table

| Held-out State     | RMSE             | MAE              | R²               | PICP             | ACE              | MPIW             | Winkler          |
|:-------------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| Illinois           | 1.2075           | 0.9548           | 0.7524           | 0.9058           | 0.0058           | 4.2688           | 5.3656           |
| Indiana            | 0.8713           | 0.6607           | 0.8130           | 0.9781           | 0.0781           | 4.1093           | 4.3824           |
| Iowa               | 1.1714           | 0.9450           | 0.7223           | 0.9361           | 0.0361           | 4.2643           | 4.7203           |
| Minnesota          | 1.5294           | 1.1697           | 0.6251           | 0.8544           | 0.0456           | 4.3684           | 7.0671           |
| Missouri           | 1.2504           | 0.9580           | 0.6425           | 0.9231           | 0.0231           | 4.6023           | 5.6358           |
| Ohio               | 0.9799           | 0.7513           | 0.7564           | 0.9626           | 0.0626           | 4.2142           | 4.5723           |
| Macro Mean         | 1.1683           | 0.9066           | 0.7186           | 0.9267           | 0.0419           | 4.3045           | 5.2906           |
| Macro 95% CI       | [0.9279, 1.4087] | [0.7187, 1.0945] | [0.6429, 0.7943] | [0.8805, 0.9729] | [0.0143, 0.0694] | [4.1277, 4.4814] | [4.2467, 6.3344] |
| Pooled Performance | 1.1803           | 0.9041           | 0.7448           | 0.9276           | 0.0276           | 4.3020           | 5.2487           |

> **Note on Uncertainty Summaries**: Confidence intervals summarize variation across the six held-out states and should not be interpreted as population-level uncertainty estimates.

## Fold-Level Calibration Diagnostics

| held_out_state   |   n_test |   nominal_coverage |   PICP |    ACE |   MPIW |   Winkler |   lower_violation_rate |   upper_violation_rate |   n_covered |   n_lower_violations |   n_upper_violations |   calibration_threshold_q |
|:-----------------|---------:|-------------------:|-------:|-------:|-------:|----------:|-----------------------:|-----------------------:|------------:|---------------------:|---------------------:|--------------------------:|
| Illinois         |     3821 |                0.9 | 0.9058 | 0.0058 | 4.2688 |    5.3656 |                 0.0173 |                 0.0769 |        3461 |                   66 |                  294 |                    1.0923 |
| Indiana          |     3377 |                0.9 | 0.9781 | 0.0781 | 4.1093 |    4.3824 |                 0.0145 |                 0.0074 |        3303 |                   49 |                   25 |                    1.1164 |
| Iowa             |     3803 |                0.9 | 0.9361 | 0.0361 | 4.2643 |    4.7203 |                 0.0042 |                 0.0597 |        3560 |                   16 |                  227 |                    1.1887 |
| Minnesota        |     2947 |                0.9 | 0.8544 | 0.0456 | 4.3684 |    7.0671 |                 0.1408 |                 0.0048 |        2518 |                  415 |                   14 |                    1.4181 |
| Missouri         |     3353 |                0.9 | 0.9231 | 0.0231 | 4.6023 |    5.6358 |                 0.0707 |                 0.0063 |        3095 |                  237 |                   21 |                    1.0618 |
| Ohio             |     3206 |                0.9 | 0.9626 | 0.0626 | 4.2142 |    4.5723 |                 0.0352 |                 0.0022 |        3086 |                  113 |                    7 |                    1.1796 |

## Ensemble Weight Audit (Tuned on Dev Validation Only)

| held_out_state   |   NeuralCQR_weight |   LightGBM_weight |   dev_r2_neural_alone |   dev_r2_lgb_alone |   dev_r2_selected_blend |   dev_RMSE_selected_blend |   dev_MAE_selected_blend | weight_selection_partition   | selected_objective      |
|:-----------------|-------------------:|------------------:|----------------------:|-------------------:|------------------------:|--------------------------:|-------------------------:|:-----------------------------|:------------------------|
| Illinois         |                1   |               0   |                0.5137 |             0.4173 |                  0.5137 |                    1.1288 |                   0.8999 | LOSO_DEV_2014_2015           | Maximize R2 on LOSO DEV |
| Indiana          |                0.4 |               0.6 |                0.48   |             0.5142 |                  0.5319 |                    1.1434 |                   0.9155 | LOSO_DEV_2014_2015           | Maximize R2 on LOSO DEV |
| Iowa             |                0.5 |               0.5 |                0.4388 |             0.4189 |                  0.481  |                    1.2444 |                   0.9969 | LOSO_DEV_2014_2015           | Maximize R2 on LOSO DEV |
| Minnesota        |                1   |               0   |                0.5148 |             0.3949 |                  0.5148 |                    1.1099 |                   0.8702 | LOSO_DEV_2014_2015           | Maximize R2 on LOSO DEV |
| Missouri         |                0.6 |               0.4 |                0.4863 |             0.4623 |                  0.5007 |                    1.1192 |                   0.8919 | LOSO_DEV_2014_2015           | Maximize R2 on LOSO DEV |
| Ohio             |                0.5 |               0.5 |                0.4807 |             0.4665 |                  0.5095 |                    1.1549 |                   0.918  | LOSO_DEV_2014_2015           | Maximize R2 on LOSO DEV |

