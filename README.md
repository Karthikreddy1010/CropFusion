[README(20261023-154518).md](https://github.com/user-attachments/files/33880482/README.20261023-154518.md)
# CropFusion: Reproducibility and Analysis Code

**Study:** *Marginal coverage hides conditional failure: conformal
prediction intervals for county maize yield under compound drought and
heat*

**Repository:** https://github.com/Karthikreddy1010/CropFusion\
**Language:** Python\
**Study period:** 1985--2023\
**Study area:** 583 counties across Illinois, Indiana, Iowa, Minnesota,
Missouri, and Ohio, USA

This repository contains code and computational workflows supporting the
study of conformal prediction intervals for county-level maize yield.
The analysis evaluates whether prediction intervals that achieve
acceptable overall (marginal) coverage remain reliable during compound
drought--heat conditions.

> **Reproducibility note:** The paper reports that its main runs used
> protocol version `1.3.0` (commit `d7089a2e7e1db551`). The repository's
> later version `1.4.0` (commit `9ea6b15d86c6ebef`) is reported to
> change protocol wording only. Verify these identifiers against the Git
> history before citing them as permanent identifiers. For exact
> reproduction, use the code, configuration, data-processing inputs, and
> dependencies corresponding to the reported run.

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

The manuscript describes the following input sources:

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
    and slope. The manuscript notes that the provenance of some static
    descriptors is incomplete; consult Supplementary Section S1 and
    Table S1.

The final study panel covers 583 counties in six Corn Belt states over
39 growing seasons (1985--2023). The reported modelling unit is the
county-year. County-years without published yield observations were
excluded rather than imputed.

The repository should be used alongside the source datasets and the
paper's supplementary information. Do not assume that all raw source
data are included in this GitHub repository unless the repository
contents confirm that.

### Feature engineering

The manuscript describes growing-season summaries for 1 April--31
October, as well as degree-day windows and features derived from drought
indices, compound drought--heat exposure, storm events, yield lags,
calendar year, county climate normals, and static county descriptors.
Feature selection and collinearity pruning left 94 features.

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

The repository currently includes Python code and Google Colab
notebooks. The visible repository listing includes:

-   `code/` --- analysis and panel-building code.
-   `main.py` --- Python entry point; inspect the file before using it
    to determine the supported command-line interface.
-   `Run_Paper3_Colab.ipynb` --- Colab notebook associated with the
    study.
-   `Run_Paper3_Colab_Master.ipynb` --- Colab notebook associated with
    the master-data pipeline and LOSO diagnostics.
-   `Run_Paper3_Remaining_Colab.ipynb` --- additional Colab workflow.
-   `Run_W3_DEV_Ladder_Colab.ipynb` --- development/evaluation workflow.

Notebook names and repository contents may change. Inspect the current
repository before relying on this list. Supplementary Section S12
describes the scripts associated with reported results.

## Requirements

The manuscript specifies Python 3.13 and the following main packages:

-   NumPy
-   pandas
-   scikit-learn
-   LightGBM
-   CatBoost
-   XGBoost
-   PyTorch
-   GeoPandas

A CUDA-compatible GPU can accelerate neural-network components when
available. A GPU is not stated to be mandatory for all parts of the
workflow.

The manuscript does not provide a complete, pinned dependency
specification in the information summarized here. Before a
reproducibility release, add a tested `requirements.txt`,
`environment.yml`, or equivalent file with the actual package versions
used. Do not treat an unpinned installation as an exact reproduction
environment.

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

The repository is currently private according to the repository
screenshot supplied by the author. Users must have permission to access
it unless its visibility is changed.

### 2. Set up Python

Create and activate a Python 3.13 environment using your preferred
environment manager. Install the project's tested dependencies after the
repository's dependency file is available.

For example, if a verified `requirements.txt` is provided:

``` bash
python -m pip install -r requirements.txt
```

Do not run the full analysis until you have checked the notebook
instructions, configuration files, required data paths, and any
credentials or external-data download steps.

### 3. Identify the workflow and data paths

Review `README.md`, `main.py`, the `code/` directory, and the relevant
notebook before running an experiment. Confirm:

-   where the raw source data are expected to reside;
-   whether the derived county-year panel is included or must be
    generated;
-   where output artifacts are written;
-   how configuration and random seeds are set;
-   which notebook or script corresponds to each table and figure in the
    manuscript.

The exact run commands and path configuration should be documented after
they have been tested against the current repository. No unverified
command is provided here.

## Outputs and reported artifacts

The manuscript describes these derived outputs:

-   A county-year panel stored in Apache Parquet format, approximately 6
    MB.
-   Result artifacts in CSV and JSON formats, approximately 90 MB in
    total.
-   Point-accuracy results, prediction-interval coverage, interval-width
    and interval-score metrics.
-   Unsafe lower-bound miss rates and yield shortfalls, including
    analyses by compound drought--heat exposure.
-   Results from temporal, LOSO, and rolling-origin evaluations.
-   Year-clustered bootstrap intervals for pooled rolling-origin
    results.

These sizes are those reported by the author and may vary by release.
Confirm that each artifact is present in the repository or archive
before stating that it is publicly downloadable.

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

The reported results should not be assumed to reproduce simply by
running the current branch. Reproduction requires the corresponding
data, configuration, software environment, and protocol version.

## Data availability and limitations

The manuscript identifies PRISM, NLDAS-2, NOAA Storm Events, USDA NASS
Quick Stats, and NLCD as data sources. Their availability and terms are
governed by their respective providers. The provenance of some static
soil and terrain descriptors is only partly documented in the
manuscript; this should be resolved in the supplementary material and
data documentation to improve reproducibility.

Before describing the repository as a complete public data archive,
verify which raw inputs, processed files, and result artifacts are
actually included. Do not commit credentials, restricted data, or files
whose redistribution terms have not been checked.

## Citation

If you use this code, please cite the associated study:

> Duddukunta, K. R., Chung, K., & Lal, P. *Marginal coverage hides
> conditional failure: conformal prediction intervals for county maize
> yield under compound drought and heat.* Manuscript citation details
> (journal, year, volume, pages, and DOI) should be added when
> available.

For software citation, cite the exact archived release and its
persistent identifier once available:

> CropFusion, version \[VERSION\], \[ARCHIVE DOI\], \[YEAR\].

## Version and licence

-   **Reported experimental protocol:** `1.3.0` (reported commit
    `d7089a2e7e1db551`; verify against Git history).
-   **Later protocol version:** `1.4.0` (reported commit
    `9ea6b15d86c6ebef`; verify against Git history).
-   **Licence:** Not specified here. Add the repository's actual licence
    and include a `LICENSE` file before presenting the software as
    openly licensed.
-   **Archived release / DOI:** To be added after archiving the exact
    code and data used for the reported results.

## Contact

**Corresponding author:** Karthik Reddy Duddukunta\
**Email:** duddukuntak1@montclair.edu

For questions about the implementation or reproducibility materials,
contact the corresponding author.
