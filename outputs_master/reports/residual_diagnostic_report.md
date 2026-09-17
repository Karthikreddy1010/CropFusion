# Residual Diagnostic Report (§11)

## Summary of Diagnostic Tests
- **Residual Mean**: `0.1011`
- **Residual Std**: `0.9181`
- **Normality Test (KS p-value)**: `0.232706` (Normally Distributed: `True`)
- **Durbin-Watson Autocorrelation Statistic**: `1.7606` (Autocorrelation Detected: `False`)
- **Breusch-Pagan Heteroscedasticity p-value**: `0.0` (Heteroscedasticity Detected: `True`)
- **Spatial Autocorrelation Detected**: `True`

## Interpretation & Next Steps
- **Heteroscedasticity**: Present — Adaptive conformal inference (ACI) interval scaling handles input-dependent variance.
- **Temporal Autocorrelation**: Minimal.
