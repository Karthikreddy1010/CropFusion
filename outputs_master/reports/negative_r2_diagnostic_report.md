# Negative R² Diagnostic Report (§1)

## Performance Metrics & Variance Decomposition
- **Training Set Target Mean**: `8.0758` t/ha
- **Test Set Target Mean**: `11.2861` t/ha
- **Target Mean Shift**: `3.2103` t/ha
- **Residual Sum of Squares ($SS_{res}$)**: `2000.7854`
- **Total Sum of Squares ($SS_{tot}$)**: `6738.0576`
- **Out-of-Sample $R^2$**: `0.7031`
- **$R^2$ Relative to Train Mean Baseline**: `0.9353`

## Drift Comparison
- **Temporal Yield Variance**: `0.1768`
- **Spatial Yield Variance**: `1.0328`

## Dominant Source of Error Conclusion
**Spatial Heterogeneity**: Spatial variance across Midwestern states dominates out-of-sample error.

## Action Plan
1. Dedicated Multi-Task Huber Mean Head ($\hat{y}_{\text{mean}}$) exclusively evaluated for RMSE/MAE/$R^2$.
2. Fourier Temporal Encodings to capture multi-year climate oscillations.
3. Static Environmental Descriptors for spatial context without state IDs.
4. Composite Loss combining Huber Loss + Pinball Loss + Quantile Crossing Penalty.
