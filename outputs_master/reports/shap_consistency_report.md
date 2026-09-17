# Cross-Backbone SHAP Attribution Consistency Report (§5.8)

## Provenance
- **Attribution method**: mean |SHAP value| per feature, normalized to sum to 1
- **Backbones with computed SHAP**: NeuralCQR, LightGBM, CatBoost, XGBoost
- **Explained samples**: 500 (background: 200)
- Per-model explainer: `NeuralCQR` = shap.GradientExplainer (mean head), `LightGBM` = shap.TreeExplainer, `CatBoost` = shap.TreeExplainer, `XGBoost` = shap.TreeExplainer

## Cross-model agreement
- **Mean Spearman ρ**: `0.6752`
- **Mean Kendall τ**: `0.5341`
- **Similarity index** (mean of the two): `0.6046` — **strong agreement**

### Pairwise agreement
| model_a   | model_b   |   spearman |   kendall_tau |   top10_overlap |   top10_jaccard |
|:----------|:----------|-----------:|--------------:|----------------:|----------------:|
| NeuralCQR | LightGBM  |     0.4522 |        0.3168 |               3 |          0.1765 |
| NeuralCQR | CatBoost  |     0.5077 |        0.3624 |               3 |          0.1765 |
| NeuralCQR | XGBoost   |     0.449  |        0.3199 |               2 |          0.1111 |
| LightGBM  | CatBoost  |     0.8365 |        0.6735 |               9 |          0.8182 |
| LightGBM  | XGBoost   |     0.9345 |        0.8109 |               9 |          0.8182 |
| CatBoost  | XGBoost   |     0.8716 |        0.7207 |               9 |          0.8182 |

## Interpretation
Cross-backbone attribution agreement is strong agreement (similarity index = 0.6046, mean Spearman = 0.6752, mean Kendall tau = 0.5341). Rank-correlation bands used: |value| < 0.20 negligible, < 0.40 limited, < 0.60 moderate, otherwise strong. Different model families can fit similarly well while distributing credit across correlated predictors differently, so limited agreement does not by itself invalidate any single model's attributions -- but it does mean the attribution ranking should not be presented as a model-independent finding.

## Top 15 consensus features
|   Rank | Feature                      |   Mean_Attribution |   Attribution_NeuralCQR |   Attribution_LightGBM |   Attribution_CatBoost |   Attribution_XGBoost |
|-------:|:-----------------------------|-------------------:|------------------------:|-----------------------:|-----------------------:|----------------------:|
|      1 | county_baseline              |           0.214905 |                0.10087  |               0.231895 |               0.254499 |              0.272354 |
|      2 | spei30_prism_nldas_mean_silk |           0.050266 |                0.02939  |               0.055843 |               0.056445 |              0.059386 |
|      3 | Yield_lag1                   |           0.035746 |                0.019405 |               0.041134 |               0.046994 |              0.03545  |
|      4 | spi1_prism_min_silk          |           0.028767 |                0.007579 |               0.02802  |               0.044303 |              0.035167 |
|      5 | prism_kdd30_gs               |           0.028493 |                0.034246 |               0.026176 |               0.032173 |              0.021378 |
|      6 | prism_ppt_total_silk         |           0.024923 |                0.011235 |               0.023947 |               0.034558 |              0.029952 |
|      7 | nldas_et0_total_silk         |           0.023176 |                0.037484 |               0.017073 |               0.017804 |              0.020345 |
|      8 | prism_tmax_mean_silk         |           0.02143  |                0.00861  |               0.026435 |               0.023988 |              0.026688 |
|      9 | spei30_prism_nldas_mean_veg  |           0.021242 |                0.005561 |               0.029485 |               0.023616 |              0.026304 |
|     10 | nldas_vpd_mean_veg           |           0.020497 |                0.01511  |               0.021232 |               0.024026 |              0.021621 |
|     11 | prism_ppt_total_veg          |           0.0202   |                0.006683 |               0.024493 |               0.022762 |              0.026862 |
|     12 | Fourier_sin_P5_0             |           0.019905 |                0.021634 |               0.019539 |               0.018045 |              0.020403 |
|     13 | Fourier_cos_P19_0            |           0.018459 |                0.007063 |               0.022616 |               0.019478 |              0.024677 |
|     14 | CDHW_Silking_Severity        |           0.013876 |                0.009495 |               0.014506 |               0.012414 |              0.019089 |
|     15 | spei30_prism_nldas_min_silk  |           0.013117 |                0.021668 |               0.011036 |               0.008082 |              0.011681 |

