# Methodology Validation Report (property-based implementation audit)

## Overall status: `NON_COMPLIANT`

| Result | Count |
|---|---:|
| PASS | 16 |
| FAIL | 2 |
| WARNING | 0 |
| NOT_APPLICABLE | 0 |

Each check reads a real artefact produced by this run and tests a property of it. A check whose evidence is missing is FAIL, never PASS.

This replaces the previous validator, which was a hard-coded table with
`"Status": "Pass"` typed into every row and reported "100% compliance, 15/15"
without testing anything.

| ID | Requirement | Status | Evidence | Artifact |
|---|---|---|---|---|
| M1 | LOSO uses exactly the six locked states | **PASS** | config and executed folds both contain exactly 6 states: Illinois, Indiana, Iowa, Minnesota, Missouri, Ohio | `config.py + loso_fold_metrics.csv` |
| M2 | No train/validation/test observation overlap in any experiment | **PASS** | all 7 experiments show zero overlap on real county-year IDs | `leakage_provenance_audit.json` |
| M3 | Calibration observations never overlap test observations | **PASS** | calibration/test overlap = 0 in all 7 experiments with a calibration partition | `leakage_provenance_audit.json` |
| M4 | Internal early-stopping split is drawn only from training data | **PASS** | early-stopping split is disjoint from dev, cal and test in all 6 experiments that use one | `leakage_provenance_audit.json` |
| M5 | Test labels never used for fitting, selection or calibration | **PASS** | every stage is bound to a partition disjoint from test: Temporal[feature_selection->train, multicollinearity_pruning->train, county_baseline->train, target_detrending->train, preprocessor_fitting->train, scaler_fitting->train, model_fitting->train, early_stopping->dev, ensemble_weight_selection->dev, conformal_calibration->cal, final_evaluation->test]; LOSO/Illinois[feature_selection->train, scaler_fitting->train, preprocessor_fitting->train, target_detrending->fit_full, model_fitting->train, early_stopping->es_internal, ensemble_weight_selection->dev, conformal_calibration->cal, final_evaluation->test] ... | `leakage_provenance_audit.json` |
| M6 | Temporal partitions are FIT 1985-2013 / DEV 2014-2015 / CAL 2016-2018 / TEST 2019-2023 with no year in two partitions | **PASS** | observed year ranges {'fit': (1985, 2013), 'dev': (2014, 2015), 'cal': (2016, 2018), 'test': (2019, 2023)}; each year belongs to exactly one partition | `splits/temporal_split_assignments.csv` |
| M7 | Feature selection fitted on the FIT partition only | **PASS** | selection funnel 117 candidates -> 100 consensus -> 93 after multicollinearity pruning -> 94 final; all stages fitted on FIT partition (1985-2013) only, and the provenance ledger binds feature_selection to the train partition | `feature_selection_pipeline.json` |
| M8 | Scaler fitted on the FIT partition only | **PASS** | scaler_fitting bound to the train partition in all 7 experiments (Temporal, LOSO/Illinois, LOSO/Indiana, LOSO/Iowa, LOSO/Minnesota, LOSO/Missouri, LOSO/Ohio); scaling_report.json records scaler='n/a' | `leakage_provenance_audit.json` |
| M9 | Temporal ensemble weight selected on DEV only, and is the DEV argmin | **PASS** | selection metric = DEV_2014_2015_RMSE; frozen weight 0.85 equals the argmin over 21 candidate weights scored on DEV | `model_decisions_lock.json + model_selection_audit.csv` |
| M10 | LOSO ensemble weight selected on LOSO DEV only, and is the DEV argmax | **PASS** | all 66 candidate weights scored on LOSO_DEV_2014_2015; each state's frozen weight is the DEV argmax | `loso_ensemble_weight_search.csv` |
| M11 | Joint and post-hoc training paths are genuinely different | **PASS** | joint = 1 stage(s) with quantile gradients reaching the backbone; post-hoc = 2 stages with a frozen backbone; max absolute difference in test point predictions = 1.215924 | `objective_O4_experiment_raw.json` |
| M12 | Ablation configurations 3, 4 and 5 are mutually distinct | **FAIL** | 2_plus_cdhw vs 3_plus_cqr: FAIL (interval head added); 3_plus_cqr vs 4_plus_aci: PASS (static conformal -> ACI (calibration only)); 4_plus_aci vs 5_full_joint: PASS (post-hoc -> joint end-to-end training) | `ablation_report.json` |
| M13 | TEST partition untouched until final evaluation | **PASS** | no pipeline stage other than final_evaluation is bound to the test partition in any experiment; decisions were frozen before the final refit (model_decisions_lock.json status=LOCKED_FINAL_DECISIONS, early_stopping_on_refit=False) | `leakage_provenance_audit.json + model_decisions_lock.json` |
| M14 | No artefact claims seven LOSO folds (the protocol has six) | **FAIL** | stale references found at: claim_consistency_audit.py:10, main.py:21, main.py:2287 | `code/*.py + outputs/reports/*` |
| M15 | Significance testing accounts for temporal/spatial dependence | **PASS** | O5 primary inference = OLS with cluster-robust (CR1) standard errors clustered by year (5 clusters, effective n = 25 from 2345 rows); O6 Friedman executed = True, block unit = year, k = 6 with q_alpha = 2.85 | `objective_o5_report.json + objective_O6_report.json` |
| M16 | SHAP attributions are genuinely computed (no synthetic fallback) | **PASS** | 4 backbone(s) explained with real SHAP explainers (NeuralCQR: shap.GradientExplainer (mean head), LightGBM: shap.TreeExplainer, CatBoost: shap.TreeExplainer, XGBoost: shap.TreeExplainer); excluded (not fabricated): none; agreement label = strong agreement | `shap_consistency_report.json` |
| M17 | Dataset checksum covers the complete file | **PASS** | SHA-256 = 86f02b734d8719ebce41af03ccc7a736fd0396db2c667def450851f9dfce2f5f over 12539554 bytes (complete file (all bytes)) | `reproducibility_report.json` |
| M18 | Computational benchmark uses warm-up, no-grad, explicit batch sizes and separates inference from end-to-end latency | **PASS** | device = cuda (Tesla T4), cuda_available = True, warm-up = 10, timed repeats = 50, gradients disabled = True, batch sizes = [1, 32, 256, 2345]; GPU memory method = torch.cuda.max_memory_allocated | `evaluation_report.json` |

## Failing checks
- **M12** — Ablation configurations 3, 4 and 5 are mutually distinct: 2_plus_cdhw vs 3_plus_cqr: FAIL (interval head added); 3_plus_cqr vs 4_plus_aci: PASS (static conformal -> ACI (calibration only)); 4_plus_aci vs 5_full_joint: PASS (post-hoc -> joint end-to-end training)
- **M14** — No artefact claims seven LOSO folds (the protocol has six): stale references found at: claim_consistency_audit.py:10, main.py:21, main.py:2287
