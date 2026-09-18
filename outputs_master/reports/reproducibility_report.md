# Reproducibility Report (§7 & Appendix)

## System & Hardware Specifications
- **Operating System**: Linux 6.6.122+ (#1 SMP Thu Apr 30 18:17:14 UTC 2026)
- **Python Version**: 3.13.15
- **CPU Architecture**: x86_64
- **Total System RAM**: 12.67 GB
- **CUDA Available**: True (GPU: `Tesla T4`, CUDA `12.8`)

## Software Dependencies
- **PyTorch**: `2.11.0+cu128`
- **LightGBM**: `4.6.0`
- **CatBoost**: `1.2.10`
- **XGBoost**: `3.4.1`
- **Scikit-Learn**: `1.6.1`
- **NumPy**: `2.1.3`
- **Pandas**: `2.2.3`

## Experiment Settings & Hashes
- **Global Random Seed**: `42`
- **Dataset File**: `Paper3_MasterDataset_PRISM_NLDAS_NOAA_USDA_FULL_1985_2023.parquet` (MD5 Checksum: `86f02b734d8719ebce41af03ccc7a736fd0396db2c667def450851f9dfce2f5f`)
- **Temporal Splits**: Train (1985, 2015), Val (2016, 2018), Test (2019, 2023)
- **Target Quantiles**: (0.05, 0.95)
- **Total Pipeline Execution Time**: `3338.1` seconds
