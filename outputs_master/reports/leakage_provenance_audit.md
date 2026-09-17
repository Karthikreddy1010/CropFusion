# Leakage & Provenance Audit (row-identity based)

**Overall status**: PASS (7 experiments audited)

Overlaps below are intersections of real county-year observation IDs (`obs_id = <GEOID>_<Year>`). Object identity / memory-address checks are not used anywhere in this audit.

| Experiment | Train/Test overlap | Val/Test overlap | Cal/Test overlap | Train/Val overlap | Status |
|---|---:|---:|---:|---:|---|
| Temporal | 0 | 0 | 0 | 0 | PASS |
| LOSO/Illinois | 0 | 0 | 0 | 0 | PASS |
| LOSO/Indiana | 0 | 0 | 0 | 0 | PASS |
| LOSO/Iowa | 0 | 0 | 0 | 0 | PASS |
| LOSO/Minnesota | 0 | 0 | 0 | 0 | PASS |
| LOSO/Missouri | 0 | 0 | 0 | 0 | PASS |
| LOSO/Ohio | 0 | 0 | 0 | 0 | PASS |

## Stage bindings per experiment

### Temporal

- Partition sizes: `{"train": 15751, "dev": 978, "cal": 1433, "test": 2345}`
- `feature_selection` consumed partition **train**
- `multicollinearity_pruning` consumed partition **train**
- `county_baseline` consumed partition **train**
- `target_detrending` consumed partition **train**
- `preprocessor_fitting` consumed partition **train**
- `scaler_fitting` consumed partition **train**
- `model_fitting` consumed partition **train**
- `early_stopping` consumed partition **dev**
- `ensemble_weight_selection` consumed partition **dev**
- `conformal_calibration` consumed partition **cal**
- `final_evaluation` consumed partition **test**

### LOSO/Illinois

- Partition sizes: `{"train": 11549, "es_internal": 1283, "fit_full": 12832, "dev": 791, "cal": 3063, "test": 3821}`
- `feature_selection` consumed partition **train**
- `scaler_fitting` consumed partition **train**
- `preprocessor_fitting` consumed partition **train**
- `target_detrending` consumed partition **fit_full**
- `model_fitting` consumed partition **train**
- `early_stopping` consumed partition **es_internal**
- `ensemble_weight_selection` consumed partition **dev**
- `conformal_calibration` consumed partition **cal**
- `final_evaluation` consumed partition **test**

### LOSO/Indiana

- Partition sizes: `{"train": 11861, "es_internal": 1317, "fit_full": 13178, "dev": 806, "cal": 3146, "test": 3377}`
- `feature_selection` consumed partition **train**
- `scaler_fitting` consumed partition **train**
- `preprocessor_fitting` consumed partition **train**
- `target_detrending` consumed partition **fit_full**
- `model_fitting` consumed partition **train**
- `early_stopping` consumed partition **es_internal**
- `ensemble_weight_selection` consumed partition **dev**
- `conformal_calibration` consumed partition **cal**
- `final_evaluation` consumed partition **test**

### LOSO/Iowa

- Partition sizes: `{"train": 11592, "es_internal": 1288, "fit_full": 12880, "dev": 784, "cal": 3040, "test": 3803}`
- `feature_selection` consumed partition **train**
- `scaler_fitting` consumed partition **train**
- `preprocessor_fitting` consumed partition **train**
- `target_detrending` consumed partition **fit_full**
- `model_fitting` consumed partition **train**
- `early_stopping` consumed partition **es_internal**
- `ensemble_weight_selection` consumed partition **dev**
- `conformal_calibration` consumed partition **cal**
- `final_evaluation` consumed partition **test**

### LOSO/Minnesota

- Partition sizes: `{"train": 12148, "es_internal": 1349, "fit_full": 13497, "dev": 835, "cal": 3228, "test": 2947}`
- `feature_selection` consumed partition **train**
- `scaler_fitting` consumed partition **train**
- `preprocessor_fitting` consumed partition **train**
- `target_detrending` consumed partition **fit_full**
- `model_fitting` consumed partition **train**
- `early_stopping` consumed partition **es_internal**
- `ensemble_weight_selection` consumed partition **dev**
- `conformal_calibration` consumed partition **cal**
- `final_evaluation` consumed partition **test**

### LOSO/Missouri

- Partition sizes: `{"train": 11741, "es_internal": 1304, "fit_full": 13045, "dev": 861, "cal": 3248, "test": 3353}`
- `feature_selection` consumed partition **train**
- `scaler_fitting` consumed partition **train**
- `preprocessor_fitting` consumed partition **train**
- `target_detrending` consumed partition **fit_full**
- `model_fitting` consumed partition **train**
- `early_stopping` consumed partition **es_internal**
- `ensemble_weight_selection` consumed partition **dev**
- `conformal_calibration` consumed partition **cal**
- `final_evaluation` consumed partition **test**

### LOSO/Ohio

- Partition sizes: `{"train": 11991, "es_internal": 1332, "fit_full": 13323, "dev": 813, "cal": 3165, "test": 3206}`
- `feature_selection` consumed partition **train**
- `scaler_fitting` consumed partition **train**
- `preprocessor_fitting` consumed partition **train**
- `target_detrending` consumed partition **fit_full**
- `model_fitting` consumed partition **train**
- `early_stopping` consumed partition **es_internal**
- `ensemble_weight_selection` consumed partition **dev**
- `conformal_calibration` consumed partition **cal**
- `final_evaluation` consumed partition **test**

