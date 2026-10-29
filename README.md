# CropFusion: Reproducibility and Analysis Code

**Study:** *Marginal coverage hides conditional failure: conformal
prediction intervals for county maize yield under compound drought and
heat*

**Repository:** https://github.com/Karthikreddy1010/CropFusion\
**Archive:** https://doi.org/10.5281/zenodo.23600496\
**Language:** Python 3.13\
**Study period:** 1985--2023\
**Study area:** 583 counties across Illinois, Indiana, Iowa, Minnesota,
Missouri, and Ohio, USA

This repository contains the code, the derived county-year data, and the
result files supporting the study of conformal prediction intervals for
county-level maize yield. The analysis evaluates whether prediction
intervals that achieve acceptable overall (marginal) coverage remain
reliable during compound drought--heat conditions.

> **Reproducibility note:** The paper's main runs used protocol version
> `1.3.0` (configuration hash `d7089a2e7e1db551`, recorded in
> `outputs_master/reports/frozen_protocol.json`). The later version
> `1.4.0` (configuration hash `9ea6b15d86c6ebef`) changes only the
> protocol's wording. These are hashes of the locked configuration, not
> Git commit identifiers. For exact reproduction, use the code,
> configuration, data and dependencies of the archived release.

## Research objectives

The study investigates three questions:

1.  Do conformal prediction intervals that perform adequately on average
    remain reliable in county-years exposed to compound drought and
    heat?
2.  Are interval failures more closely associated with prediction bias
    or interval width, and which weather exposures and approximate
    crop-growth stages are associated with failure?
3.  Does group-conditional calibration reduce failures in the
    high-exposure group, and what trade-offs does it create for other
    groups?

The study is an evaluation of existing models and calibration methods;
it does **not** propose a new prediction model or a new conformal
calibration method.

## Study design

The workflow constructs a county-year panel, trains point-prediction
models, calibrates prediction intervals on separate calibration data,
and evaluates the frozen models under predefined evaluation designs.

### Study data

The study uses the following input sources:

-   **PRISM:** daily maximum and minimum temperature and precipitation,
    using the 4 km daily product and county-area-weighted aggregation.
-   **NLDAS-2:** hourly meteorological forcing used to derive variables
    including radiation, humidity, wind, vapor-pressure deficit, and
    reference evapotranspiration.
-   **NOAA Storm Events:** storm-event counts and maximum hail size.
-   **USDA NASS Quick Stats:** county-level maize grain yield, converted
    to tonnes per hectare (`t ha⁻¹`).
-   **National Land Cover Database (NLCD):** county land-cover
    fractions.
-   **Soil and terrain descriptors:** soil properties, mean elevation,
    and slope. The provenance of some static descriptors is incomplete;
    consult Supplementary Section S1 and Table S1.

The final study panel covers 583 counties in six Corn Belt states over
39 growing seasons (1985--2023). The modelling unit is the county-year.
County-years without published yield observations were excluded rather
than imputed.

The derived county-year panel is included in this repository
(`master_dataset/`); the raw source downloads are not (see
[Data availability](#data-availability-and-limitations)).

### Feature engineering

Growing-season summaries cover 1 April--31 October, together with
degree-day windows and features derived from drought indices, compound
drought--heat exposure, storm events, yield lags, calendar year, county
climate normals, and static county descriptors. Feature selection and
collinearity pruning left 94 features.

A compound drought--heat day is defined as a day with maximum
temperature above 35 °C and 30-day SPEI below −1. The Extreme exposure
class comprises county-years with at least three such days. Approximate
maize growth stages are defined using growing degree days accumulated
from 1 April; the paper's "silking window" is the 700--1400 GDD
interval. These are degree-day proxies, not observed planting or silking
dates.

### Prediction models

The study compares five machine-learning point models:

-   LightGBM
-   CatBoost
-   XGBoost
-   NeuralCQR
-   An ensemble combining NeuralCQR and LightGBM

Two statistical baselines are also evaluated for point accuracy:

-   A linear yield-trend plus county-mean baseline
-   A county fixed-effects degree-day panel

The conformal calibration methods adjust prediction intervals produced
by a frozen point model; they are not separate point-prediction models.

### Conformal calibration methods

The locked temporal comparison evaluates six calibration approaches:

1.  Static split conformal calibration
2.  Adaptive conformal inference (ACI)
3.  Severity-aware ACI (SA-ACI)
4.  Width-scaled conformalized quantile regression (CQR)
5.  Weighted conformal prediction
6.  Phenology-stratified CQR

The rolling-origin analysis compares static split calibration,
group-conditional calibration, standard ACI, and SA-ACI. The paper
describes the implementation and qualifications of each method in its
Methods section and Supplementary Section S5.

### Evaluation designs

-   **Locked temporal split:** fitting (1985--2013), development
    (2014--2015), calibration (2016--2018), and test (2019--2023).
-   **Leave-one-state-out (LOSO):** each of the six states is held out
    in turn to evaluate transfer to an unseen state.
-   **Rolling-origin evaluation:** 16 test origins, spanning 2008--2023,
    to assess performance across multiple seasons.
-   **Random 70/10/20 split:** an interpolation benchmark for point
    accuracy only; it is not treated as the primary forecasting
    evaluation.

The protocol aims to prevent test-set use in model selection and
calibration. The paper also notes specific limitations and exceptions,
including the drought-index reference period used for early rolling
origins. See the manuscript and supplementary material for full details.

## Repository contents

```
code/                        analysis code (paths cited in Supplementary Section S12)
  main.py                    locked temporal, leave-one-state-out and random-split evaluations
  rolling_origin.py          rolling-origin evaluation (16 test years, 2008–2023)
  run_rolling_sensitivity.sh, sensitivity_rolling.py   pre-declared sensitivity runs and their analysis
  run_rolling_seeds.sh       seed runs of the rolling origin
  dependence_aware_stats.py  year-clustered bootstrap
  phenology_coverage.py      exposure and growth-stage analysis
  objectives_ro.py           research-objective tests (bias versus width, the 2012 case, ...)
  fe_degree_day_baseline.py  fixed-effects degree-day baseline
  config.py, frozen_protocol.py   settings and the locked-protocol check
  make_manuscript_figures.py        every figure in the paper, drawn from the result files
  make_graphical_abstract_pptx.py   the graphical abstract (PowerPoint, PNG, TIFF), drawn from the result files
  pipelines/                 building the county-year panel from the raw sources
    prism/, nldas/, noaa_storm/   download and county aggregation of each source
    master_dataset/          steps 01–11 that assemble, check and document the panel
  diagnostics_loso/          verification scripts and development diagnostics
  legacy/                    scripts not used for the paper, kept for the record (see code/legacy/README.md)
master_dataset/              county-year panel and its metadata (see master_dataset/README.md)
outputs_master/              results of the main run (see outputs_master/README.md)
outputs_diagnostics/         results of the rolling-origin, sensitivity and diagnostic runs
tests/                       unit tests of the split, leakage and protocol rules
requirements.txt             pinned package versions used for the reported runs
```

Every figure in the paper, and the graphical abstract, is produced by
code from the result files in this repository; no figure is drawn by
hand or by an image generator.

## Requirements

Python 3.13 with the package versions in `requirements.txt` (NumPy,
pandas, scikit-learn, LightGBM, CatBoost, XGBoost, PyTorch, GeoPandas and
others). A CUDA-compatible GPU accelerates the neural-network components
when available but is not required. Exporting the graphical abstract to
PNG and TIFF also needs `pywin32` and Microsoft PowerPoint on Windows;
the PowerPoint file itself is built without them.

## Getting started

### 1. Clone the repository

Using GitHub CLI:

``` bash
gh repo clone Karthikreddy1010/CropFusion
cd CropFusion
```

Or using Git:

``` bash
git clone https://github.com/Karthikreddy1010/CropFusion.git
cd CropFusion
```

The repository is public.

### 2. Set up Python

Create and activate a Python 3.13 environment, then install the pinned
dependencies:

``` bash
python -m pip install -r requirements.txt
```

### 3. Run

From the repository root:

``` bash
python -m pytest tests                       # protocol and leakage tests
python code/make_manuscript_figures.py       # all figures, from the result files
python code/make_graphical_abstract_pptx.py  # graphical abstract
python code/main.py                          # locked temporal, LOSO and random-split evaluations
python code/rolling_origin.py                # rolling origin, LightGBM, two calibration seasons
bash code/run_rolling_sensitivity.sh         # the four pre-declared sensitivity runs
```

The tests and the figure scripts were checked on a fresh clone of this
repository: the tests pass and the figures are identical to those in the
paper. Because the result files of the reported runs are included, the
figures can be rebuilt without rerunning the models. Rerunning the
models reproduces the reported point metrics, coverage figures and
intervals (Supplementary Section S10); run times depend on the hardware.

## Outputs and reported artifacts

The repository includes:

-   The county-year panel in Apache Parquet format
    (`master_dataset/model_ready/`, about 6 MB; 20,507 county-years with
    an observed yield, 95 columns), the full panel of all 22,737
    county-years (`master_dataset/full/`), a catalog of every variable
    and a provenance record (`master_dataset/metadata/`).
-   The evaluation splits and county geometry
    (`outputs_master/splits/`, `outputs_master/geo_cache/`).
-   Result files in CSV and JSON formats behind every table and figure:
    point accuracy, prediction-interval coverage, interval width and
    interval scores, unsafe lower-bound miss rates and yield shortfalls by
    compound drought--heat exposure, temporal, LOSO and rolling-origin
    results, and year-clustered bootstrap intervals
    (`outputs_master/`, `outputs_diagnostics/`).

Each data folder has a README that lists its files and links them to the
tables and figures of the paper.

## Reproducibility and interpretation

Please follow these principles when reproducing the study:

1.  Use the intended temporal and spatial splits; do not randomly mix
    county-years for the primary forecasting claims.
2.  Fit learned preprocessing and feature-selection steps using the
    appropriate fitting partition only.
3.  Keep model selection, conformal calibration, and final testing
    separated as specified in the manuscript.
4.  Do not tune the final method using the test outcomes.
5.  Preserve the predefined compound-exposure classes and evaluation
    metrics.
6.  Treat years, rather than all county-years, as the resampling unit
    for the paper's year-clustered bootstrap.
7.  Report the code version, configuration, data version, package
    versions, and any deviations from the paper's protocol.

## Data availability and limitations

All inputs are public: PRISM, NLDAS-2, NOAA Storm Events, USDA NASS
Quick Stats and NLCD. Their availability and terms are governed by their
providers; please cite them when using the panel. The derived
county-year panel and all result files are included here. The raw
downloads and intermediate daily files (about 4 GB) are not; the scripts
in `code/pipelines/` download and process them. The provenance of some
static soil and terrain descriptors is only partly documented (see
Supplementary Section S1).

## Citation

If you use this code or data, please cite the associated study:

> Duddukunta, K. R., Chung, K., & Lal, P. *Marginal coverage hides
> conditional failure: conformal prediction intervals for county maize
> yield under compound drought and heat.* Manuscript citation details
> (journal, year, volume, pages, and DOI) will be added when available.

and the archived release: CropFusion, https://doi.org/10.5281/zenodo.23600496.

## Version and licence

-   **Reported experimental protocol:** `1.3.0` (configuration hash
    `d7089a2e7e1db551`).
-   **Later protocol version:** `1.4.0` (configuration hash
    `9ea6b15d86c6ebef`; wording only).
-   **Licence:** MIT (see `LICENSE`). The licence covers the code and the
    derived files produced by it; the underlying PRISM, NLDAS-2, NOAA,
    USDA NASS and NLCD data remain under their providers' terms.
-   **Archived release / DOI:** https://doi.org/10.5281/zenodo.23600496

## Contact

**Corresponding author:** Karthik Reddy Duddukunta\
**Email:** duddukuntak1@montclair.edu

For questions about the implementation or reproducibility materials,
contact the corresponding author.
