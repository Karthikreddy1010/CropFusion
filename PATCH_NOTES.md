# Paper 3 Pipeline Patch — Notes (v7)

All changes are leakage-free (every fit on the TRAIN split only) and preserve the
existing architecture. Toggles live in config.py.

## Files changed
- config.py, data_loader.py, main.py

## Fixes in this version
1. **Nebraska removed** (DROP_STATES=["Nebraska"]) at load time — irrigated, breaks
   the drought–yield relationship the model learns from rain-fed states.
2. **Detrend target** (DETREND_TARGET=True): model the technology-adjusted yield
   anomaly (trend fit on TRAIN years only), add the trend back to predictions and
   intervals before evaluation. Main R² fix (-0.75 → ~0.43).
3. **Multicollinearity removal** (correlation |r|>0.95 + iterative VIF>10) with core
   CDHW/SPEI features protected. Real reduction (~63 → ~47 features).
4. **County baseline** feature (mean TRAIN detrended anomaly per county; leakage-free).
5. **Benchmark bug FIXED**: the multi-backbone benchmark now adds the trend back
   before scoring (previously compared detrended predictions vs raw yields, giving the
   spurious NeuralCQR R² = -40.89). backbone_benchmark.csv is now valid.
6. **LOSO / Nebraska FIXED**: Nebraska removed from LOSO_STATES → clean 6-state
   LOSO-CV (previously an empty Nebraska fold crashed the run). An empty-fold guard is
   also added as a safety net.
7. **Epoch metadata FIXED**: model metadata now records epochs_configured and the
   ACTUAL epochs_trained (from early stopping), instead of a hard-coded 150.

## How to run
1. Paper3_MegaDataset_SPEI_FINAL.csv present (already in the folder).
2. pip install statsmodels lightgbm xgboost catboost torch scikit-learn seaborn.
3. python main.py
4. Key outputs:
   - outputs/reports/evaluation_summary.md  (main result, R² ~0.43, 5 calibration methods)
   - outputs/reports/backbone_benchmark.csv (now valid, NOT -40)
   - outputs/figures/  (PICP/MPIW/backbone/LOSO/residual plots for all 5 methods)

## To revert
config.py: DROP_STATES=[], DETREND_TARGET=False, ADD_COUNTY_BASELINE=False.

## Honest note on R²
Current honest R² ≈ 0.43 (validation ~0.50). Pushing to 0.60 (per request) is not
guaranteed — the remaining gap is real train/test distribution shift. Candidate
techniques: Fourier feature encoding, deeper residual MLP, DRO. These require
additional GPU runs and are not yet applied here.

## v5 addition
8. **Tree-baseline benchmark FIXED**: LightGBM/CatBoost/XGBoost now train on the
   detrended target like NeuralCQR, so the benchmark's trend add-back applies
   consistently. Previously they trained on raw yield but got the trend added again
   (double-counted), giving spurious R² around -25. Now all four backbones are on the
   same scale and comparable.

## v6 addition
9. **Ablation study detrend FIXED**: the 5-row ablation now detrends the target
   (train-only) and adds the trend back before scoring, consistent with the main
   pipeline. Previously the ablation ran on raw yield without detrending, giving
   negative R² (~-0.5 to -0.65) that misrepresented each configuration. Ablation R²
   values are now on the same scale as the main results.

## v7 addition (R² tuning attempt)
10. **Network capacity + training tuning**: widened hidden dims (256,128,64,32) ->
    (384,256,128,64) and raised early-stopping patience 20 -> 35, to give the model
    more capacity and training time. This is an honest tuning attempt; gains are not
    guaranteed since the model already implements residual blocks, a mean-head, the
    quantile-crossing loss, Fourier encodings, and lag features (the diagnostic's main
    recommendations are already present). The remaining gap to higher R² is genuine
    train/test distribution shift, quantified by the 0.88 R² relative to the train-mean
    baseline vs 0.49 standard R².
