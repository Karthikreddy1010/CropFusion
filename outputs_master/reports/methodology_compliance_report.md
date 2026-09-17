# Paper 3 Methodology Compliance Report (§1–§7)

## Overall status: `NON_COMPLIANT`

| Result | Count |
|---|---:|
| PASS | 16 |
| FAIL | 2 |
| WARNING | 0 |
| NOT_APPLICABLE | 0 |

This status is computed by `methodology_validator.run_methodology_validation()`,
which reads the artefacts this run produced and tests properties of them. It is
not a fixed statement: a check whose evidence is missing counts as FAIL.

## Locked protocol as implemented
- **Temporal split**: FIT 1985–2013, DEV 2014–2015, CAL 2016–2018, TEST 2019–2023
- **LOSO-CV**: 6 state folds — Illinois, Indiana, Iowa, Minnesota, Missouri, Ohio
- **Leakage audit**: zero observation overlap across 7 experiments, checked on county-year row IDs

## Check results
| ID | Requirement | Status | Evidence |
|---|---|---|---|
| M1 | LOSO uses exactly the six locked states | **PASS** | config and executed folds both contain exactly 6 states: Illinois, Indiana, Iowa, Minnesota, Missouri, Ohio |
| M2 | No train/validation/test observation overlap in any experiment | **PASS** | all 7 experiments show zero overlap on real county-year IDs |
| M3 | Calibration observations never overlap test observations | **PASS** | calibration/test overlap = 0 in all 7 experiments with a calibration partition |
| M4 | Internal early-stopping split is drawn only from training data | **PASS** | early-stopping split is disjoint from dev, cal and test in all 6 experiments that use one |
| M5 | Test labels never used for fitting, selection or calibration | **PASS** | every stage is bound to a partition disjoint from test: Temporal[feature_selection->train, multicollinearity_pruning->train, county_baseline->train, target_detrending->train, preprocessor_fitting->train, scaler_fitting->train, model_fitting->train, early_stopping->dev, ensemble_weight_selection->dev, conformal_calibration->cal, final_evaluation->test]; LOSO/Illinois[feature_selection->train, scaler_fitting->train, preprocessor_fitting->train, target_detrending->fit_full, model_fitting->train, early_stopping->es_internal, ensemble_weight_selection->dev, conformal_calibration->cal, final_evaluation->test] ... |
| M6 | Temporal partitions are FIT 1985-2013 / DEV 2014-2015 / CAL 2016-2018 / TEST 2019-2023 with no year in two partitions | **PASS** | observed year ranges {'fit': (1985, 2013), 'dev': (2014, 2015), 'cal': (2016, 2018), 'test': (2019, 2023)}; each year belongs to exactly one partition |
| M7 | Feature selection fitted on the FIT partition only | **PASS** | selection funnel 117 candidates -> 100 consensus -> 93 after multicollinearity pruning -> 94 final; all stages fitted on FIT partition (1985-2013) only, and the provenance ledger binds feature_selection to the train partition |
| M8 | Scaler fitted on the FIT partition only | **PASS** | scaler_fitting bound to the train partition in all 7 experiments (Temporal, LOSO/Illinois, LOSO/Indiana, LOSO/Iowa, LOSO/Minnesota, LOSO/Missouri, LOSO/Ohio); scaling_report.json records scaler='n/a' |
| M9 | Temporal ensemble weight selected on DEV only, and is the DEV argmin | **PASS** | selection metric = DEV_2014_2015_RMSE; frozen weight 0.85 equals the argmin over 21 candidate weights scored on DEV |
| M10 | LOSO ensemble weight selected on LOSO DEV only, and is the DEV argmax | **PASS** | all 66 candidate weights scored on LOSO_DEV_2014_2015; each state's frozen weight is the DEV argmax |
| M11 | Joint and post-hoc training paths are genuinely different | **PASS** | joint = 1 stage(s) with quantile gradients reaching the backbone; post-hoc = 2 stages with a frozen backbone; max absolute difference in test point predictions = 1.215924 |
| M12 | Ablation configurations 3, 4 and 5 are mutually distinct | **FAIL** | 2_plus_cdhw vs 3_plus_cqr: FAIL (interval head added); 3_plus_cqr vs 4_plus_aci: PASS (static conformal -> ACI (calibration only)); 4_plus_aci vs 5_full_joint: PASS (post-hoc -> joint end-to-end training) |
| M13 | TEST partition untouched until final evaluation | **PASS** | no pipeline stage other than final_evaluation is bound to the test partition in any experiment; decisions were frozen before the final refit (model_decisions_lock.json status=LOCKED_FINAL_DECISIONS, early_stopping_on_refit=False) |
| M14 | No artefact claims seven LOSO folds (the protocol has six) | **FAIL** | stale references found at: claim_consistency_audit.py:10, main.py:21, main.py:2287 |
| M15 | Significance testing accounts for temporal/spatial dependence | **PASS** | O5 primary inference = OLS with cluster-robust (CR1) standard errors clustered by year (5 clusters, effective n = 25 from 2345 rows); O6 Friedman executed = True, block unit = year, k = 6 with q_alpha = 2.85 |
| M16 | SHAP attributions are genuinely computed (no synthetic fallback) | **PASS** | 4 backbone(s) explained with real SHAP explainers (NeuralCQR: shap.GradientExplainer (mean head), LightGBM: shap.TreeExplainer, CatBoost: shap.TreeExplainer, XGBoost: shap.TreeExplainer); excluded (not fabricated): none; agreement label = strong agreement |
| M17 | Dataset checksum covers the complete file | **PASS** | SHA-256 = 86f02b734d8719ebce41af03ccc7a736fd0396db2c667def450851f9dfce2f5f over 12539554 bytes (complete file (all bytes)) |
| M18 | Computational benchmark uses warm-up, no-grad, explicit batch sizes and separates inference from end-to-end latency | **PASS** | device = cuda (Tesla T4), cuda_available = True, warm-up = 10, timed repeats = 50, gradients disabled = True, batch sizes = [1, 32, 256, 2345]; GPU memory method = torch.cuda.max_memory_allocated |

## Verification statement
2 check(s) FAILED and 0 raised WARNING. The implementation is not fully compliant; see the rows above.
- **M12** (FAIL): Ablation configurations 3, 4 and 5 are mutually distinct — 2_plus_cdhw vs 3_plus_cqr: FAIL (interval head added); 3_plus_cqr vs 4_plus_aci: PASS (static conformal -> ACI (calibration only)); 4_plus_aci vs 5_full_joint: PASS (post-hoc -> joint end-to-end training)
- **M14** (FAIL): No artefact claims seven LOSO folds (the protocol has six) — stale references found at: claim_consistency_audit.py:10, main.py:21, main.py:2287
