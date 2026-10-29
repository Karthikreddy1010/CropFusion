# County-year panel

| File | Rows | Columns | Content |
|---|---|---|---|
| `model_ready/Paper3_MasterDataset_PRISM_NLDAS_CDHWH_CORN_MODEL_1985_2023.parquet` | 20,507 | 95 | County-years with an observed maize yield; the input to every model |
| `full/Paper3_MasterDataset_PRISM_NLDAS_NOAA_USDA_FULL_1985_2023.parquet` | 22,737 | 220 | All county-years of the 583 counties, 1985–2023, including those without a published yield, with annual as well as growing-season variables |
| `metadata/master_variable_catalog.csv` | 220 | 20 | Every variable: source, description, unit, temporal and spatial resolution, calculation, aggregation |
| `metadata/master_data_provenance.json` | | | Source files, processing steps and checks behind each variable |
| `metadata/legacy_column_mapping.csv` | | | Names in an earlier version of the dataset mapped to the current names |

One row is one county in one growing season (keys: `state_fips`, `county_fips`, `year`). The target is
`corn_yield_bu_acre` (USDA NASS county survey yield, bushels per acre; the paper converts it to t ha⁻¹ by
multiplying by 0.0627). The growing season is 1 April to 31 October.

## Sources

| Source | Variables | Provider |
|---|---|---|
| PRISM AN81d, 4 km daily | Maximum and minimum temperature, precipitation | PRISM Climate Group, Oregon State University |
| NLDAS-2 primary forcing (NLDAS_FORA0125_H), 0.125° hourly | Radiation, humidity, wind, vapor-pressure deficit, FAO-56 reference evapotranspiration | NASA GES DISC |
| NOAA Storm Events database | Hail, thunderstorm-wind and tornado counts; maximum hail size | NOAA National Centers for Environmental Information |
| USDA NASS Quick Stats | County maize grain yield | USDA National Agricultural Statistics Service |
| Soil, land cover (NLCD), elevation | Static county descriptors | From an earlier workflow; product, epoch and resolution only partly recorded |

The panel is built by `code/pipelines/` (downloads and county aggregation) and
`code/pipelines/master_dataset/01_audit_sources.py` to `11_final_report.py`. The raw downloads and intermediate
daily files (about 4 GB) are not stored in this repository. Please cite the original data providers when using
the panel.
