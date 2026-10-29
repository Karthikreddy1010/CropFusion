# Legacy scripts

These scripts are not used by any analysis in the paper. They are kept as a record of how the project developed
and are not maintained; some import modules from `code/` and will not run from this folder.

| Script | What it was for |
|---|---|
| `download_era5_daily.py`, `audit_era5_daily.py`, `test_cds_quick.py`, `pipelines/cancel_cds_era5.py`, `pipelines/purge_all_cds.py`, `pipelines/scan_era5_data.py` | An ERA5 reanalysis download that was abandoned; the paper uses no reanalysis product |
| `noaa_cdo_client.py`, `noaa_station_selector.py`, `noaa_weather_processor.py`, `noaa_weather_validator.py`, `noaa_config.yaml` | Station weather from NOAA GHCN-Daily, replaced by PRISM and NLDAS-2 |
| `pipelines/test_nldas_auth.py` | One-off check of the NASA Earthdata login |
| `pipelines/merge_with_mega_dataset.py` | Merged storm counts into an earlier dataset, replaced by `code/pipelines/master_dataset/` |
| `audit_pipeline.py` | Pre-tuning leakage audit of an earlier output layout, replaced by `tests/` and `code/smoke_test_master_loader.py` |
| `root_main_wrapper.py`, `run_final_pipeline.py` | Old top-level entry points; run `python code/main.py` instead |
