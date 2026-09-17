# NOAA Historical Weather Integration: 583 Corn Belt Counties (1985–2023)

## 1. Executive Summary & Provenance

This directory contains the complete, reproducible pipeline and dataset artifacts integrating authoritative daily historical weather observations from the **National Oceanic and Atmospheric Administration (NOAA)** and the **National Centers for Environmental Information (NCEI)** into the existing `CropFusion` / `NeuralCQR` crop yield modeling pipeline.

- **Primary Source**: NOAA / NCEI Global Historical Climatology Network Daily (GHCND) archive.
- **Access Endpoints**:
  - NOAA Climate Data Online (CDO) API v2: `https://www.ncei.noaa.gov/cdo-web/api/v2/`
  - NOAA NCEI Direct Archive Access: `https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/`
- **Study Scope**: 583 locked corn counties across six U.S. Corn Belt states:
  - Illinois (FIPS 17, 102 counties)
  - Indiana (FIPS 18, 92 counties)
  - Iowa (FIPS 19, 99 counties)
  - Minnesota (FIPS 27, 87 counties)
  - Missouri (FIPS 29, 115 counties)
  - Ohio (FIPS 39, 88 counties)
- **Temporal Horizon**: 1985–2023 (39 consecutive years, 14,245 daily records per county).
- **Evaluation Design Compatibility**: Fully aligned with locked temporal splits:
  - `FIT`: 1985–2013 (29 years)
  - `DEV`: 2014–2015 (2 years)
  - `CAL`: 2016–2018 (3 years)
  - `TEST`: 2019–2023 (5 years, completely locked)
  - `LOSO`: 6-state Leave-One-State-Out cross-validation.

---

## 2. Methodology & Station Selection

### 2.1 Station Selection Criteria
To provide uninterrupted, high-fidelity ground observations for all 583 counties, a multi-tiered spatial mapping was conducted:
1. **Authoritative Centroids**: County geographical centroids (latitude, longitude) were extracted from the study dataset (`Paper3_MegaDataset_SPEI_FINAL.csv`).
2. **Candidate Discovery**: NOAA CDO API was queried for all GHCND stations within each state having temperature (`TMAX`, `TMIN`) and precipitation (`PRCP`) observations active between 1985 and 2023. A total of 1,946 candidate stations were cataloged.
3. **Spatial Search & Distance Calculation**: Great-circle distances between all county centroids and candidate stations were computed using the vectorized Haversine formulation ($R = 6371.0\text{ km}$).
4. **Long-Record Prioritization**:
   - Stations active continuously from $\le 1985\text{-}01\text{-}01$ through $\ge 2023\text{-}12\text{-}31$ with historical coverage $\ge 80\%$ were designated as `is_long_record = True`.
   - Ranking criterion: $\text{Rank Score} = (\neg \text{is\_long\_record}) \times 1000.0 + \text{distance\_km}$.
   - Result: **578 out of 583 counties (99.14%)** are served by a long-record primary station (Rank 1).
   - Mean distance to primary station across all 583 counties is **16.52 km** (median: 13.92 km, max: 49.32 km).
5. **Fallback Network**: The top 3 ranked stations within 50.0 km were selected per county (totaling 634 unique weather stations) to enable multi-station Inverse Distance Weighting (IDW) interpolation.

### 2.2 County-Level Spatial Aggregation
County centroids represent the target location. For each calendar day from 1985-01-01 to 2023-12-31 (14,245 days):
1. **Primary Station**: If the Rank 1 primary station has non-missing, quality-verified observations on date $d$, its values are used directly (`aggregation_method = "primary_station"`).
2. **Secondary IDW Interpolation**: If the primary station is missing or invalid on date $d$, available stations from the top-3 network within 50.0 km are combined using inverse distance squared weighting ($p = 2$):
   $$w_i = \frac{1}{d_i^2}, \quad \bar{X} = \frac{\sum_{i=1}^k w_i X_i}{\sum_{i=1}^k w_i}$$
   where $d_i$ is the distance from the county centroid to station $i$.
3. **Short-Gap Linear Interpolation**: For isolated gaps $\le 3$ days where all local stations experienced regional data dropouts, linear time interpolation was applied.
4. **Day-of-Year Climatology**: For any remaining extended historical gaps, the 39-year local day-of-year (DOY) median climatology was applied, guaranteeing 100% complete daily time series for downstream modeling without artificial discontinuities.

---

## 3. Variables, Units, and Transformations

### 3.1 Raw GHCND Units vs Converted Units
NOAA GHCND stores daily temperature and precipitation in integer tenths:
| Variable | GHCND Raw Unit | Target Unit | Transformation Formula | Valid Bounds |
|:---|:---|:---|:---|:---|
| `TMAX` | Tenths of degrees Celsius | °C | `TMAX_degC = TMAX_raw / 10.0` | $[-50.0, +55.0]$ °C |
| `TMIN` | Tenths of degrees Celsius | °C | `TMIN_degC = TMIN_raw / 10.0` | $[-50.0, +55.0]$ °C |
| `PRCP` | Tenths of millimeters | mm/day | `PRCP_mm = PRCP_raw / 10.0` | $[0.0, 400.0]$ mm |

### 3.2 Agronomic Indicators & Formulas
1. **Corn Growing Degree Days (86/50 cutoff method)**:
   $$T_{\text{max,adj}} = \min(30.0, \max(10.0, T_{\text{max}}))$$
   $$T_{\text{min,adj}} = \min(30.0, \max(10.0, T_{\text{min}}))$$
   $$T_{\text{mean,adj}} = \frac{T_{\text{max,adj}} + T_{\text{min,adj}}}{2.0}$$
   $$\text{GDD}_{\text{corn}} = \max(0.0, T_{\text{mean,adj}} - 10.0)$$
   *Units*: °C-days (base 10°C, upper ceiling 30°C).
2. **Extreme Heat Stress Indicators**:
   - `heat_stress_30`: $1 \text{ if } T_{\text{max}} \ge 30.0^\circ\text{C} \text{ else } 0$ (initial heat stress threshold).
   - `heat_stress_32`: $1 \text{ if } T_{\text{max}} \ge 32.0^\circ\text{C} \text{ else } 0$ (pollen viability reduction threshold).
   - `heat_stress_35`: $1 \text{ if } T_{\text{max}} \ge 35.0^\circ\text{C} \text{ else } 0$ (severe reproductive damage & silk desiccation).
3. **Growing Season Cumulative Metrics**:
   - May 1 to September 30 (Months 5–9, 153 days): `GDD_accum_may_sep`, `Precip_total_growseason_mm`, `TMAX_mean_growseason_C`, `TMIN_mean_growseason_C`, `TMAX_max_growseason_C`, `Extreme_heat_days_30_growseason`, `Extreme_heat_days_32_growseason`, `Extreme_heat_days_35_growseason`, `Dry_days_growseason` ($\text{PRCP} < 1\text{ mm}$), `Wet_days_growseason` ($\text{PRCP} \ge 1\text{ mm}$).
   - April 1 to October 31 (Months 4–10, 214 days): `GDD_accum_apr_oct` (full season thermal accumulation).
   - `Drought_index_proxy`: $\frac{\text{Precip\_total\_growseason\_mm}}{\text{GDD\_accum\_may\_sep} + 10^{-4}}$ (hydrothermal ratio).

---

## 4. Deliverable File Schema & Data Dictionary

All output artifacts are located in `outputs/noaa_weather/`:

### 1. `noaa_station_inventory.csv`
Station metadata and proximity mapping to target counties:
- `station_id`: GHCND station identifier (e.g., `GHCND:USW00093989`).
- `state`: U.S. State name.
- `county`: County name.
- `GEOID`: 5-digit county FIPS code (e.g., `17001`).
- `latitude`, `longitude`: Station geographic coordinates.
- `elevation`: Elevation in meters above sea level.
- `first_date`, `last_date`: Historical observation range.
- `distance_to_primary_county_km`: Distance to mapped county centroid in kilometers.
- `is_long_record`: Boolean flag indicating continuous $\ge 80\%$ record across 1985–2023.
- `datacoverage`, `coverage_TMAX`, `coverage_TMIN`, `coverage_PRCP`: Historical completeness scores.

### 2. `noaa_daily_station_weather.csv`
Station-level daily observations (1985–2023) for all primary network stations:
- `station_id`: GHCND station identifier.
- `date`: Calendar date (`YYYY-MM-DD`).
- `TMAX_raw`, `TMIN_raw`, `PRCP_raw`: Raw integer values from GHCND archive.
- `TMAX_degC`, `TMIN_degC`: Cleaned maximum and minimum temperature in °C.
- `PRCP_mm`: Cleaned precipitation in mm/day.
- `GDD_corn`: Calculated corn GDD in °C-days.
- `qflag_TMAX`, `qflag_TMIN`, `qflag_PRCP`: Official NOAA GHCND quality flags.
- `is_interpolated`: Boolean flag indicating if station observation was imputed.

### 3. `noaa_daily_county_weather.csv`
County-level daily aggregated weather time series (1985–2023, 8.3M records):
- `GEOID`: 5-digit county FIPS code.
- `state`: State name.
- `county`: County name.
- `date`: Calendar date (`YYYY-MM-DD`).
- `TMAX_degC`, `TMIN_degC`, `TMEAN_degC`: Spatially aggregated temperatures in °C.
- `PRCP_mm`: Daily precipitation in mm.
- `GDD_corn`: Corn growing degree days in °C-days.
- `heat_stress_30`, `heat_stress_32`, `heat_stress_35`: Binary heat stress indicators.
- `n_contributing_stations`: Number of stations used for daily aggregation.
- `aggregation_method`: Method label (`primary_station`, `idw_interpolation`, `linear_interp`, `climatology`).

### 4. `noaa_county_weather_annual.csv`
Annual county-level growing season summaries (22,737 records, 583 counties $\times$ 39 years):
- `GEOID`: 5-digit county FIPS code.
- `state`: State name.
- `county`: County name.
- `Year`: Calendar year (1985–2023).
- `GDD_accum_may_sep`: Cumulative GDD from May 1 to September 30.
- `GDD_accum_apr_oct`: Cumulative GDD from April 1 to October 31.
- `Precip_total_growseason_mm`: Total growing season precipitation (May–Sep) in mm.
- `TMAX_mean_growseason_C`, `TMIN_mean_growseason_C`: Mean temperatures (May–Sep) in °C.
- `TMAX_max_growseason_C`: Highest single-day maximum temperature (May–Sep) in °C.
- `Extreme_heat_days_30_growseason`, `Extreme_heat_days_32_growseason`, `Extreme_heat_days_35_growseason`: Counts of days exceeding 30°C, 32°C, and 35°C during May–Sep.
- `Dry_days_growseason`, `Wet_days_growseason`: Counts of dry (<1 mm) and wet (≥1 mm) days.
- `Drought_index_proxy`: Hydrothermal ratio ($\text{Precip} / \text{GDD}$).
- `Completeness_pct`: Final post-QC observation completeness (100.0%).

### 5. `noaa_data_quality_report.csv`
Per-county quality audit summary:
- `GEOID`, `state`, `county`: County identification.
- `primary_station_id`, `primary_station_distance_km`: Station provenance.
- `total_study_days`: 14,245 days (1985–2023).
- `raw_missing_days`, `raw_missing_pct`: Missing observations in raw primary station record.
- `interpolated_days`, `interpolated_pct`: Days resolved via secondary IDW interpolation.
- `final_valid_pct`: 100.0%.
- `mean_contributing_stations`: Average station support count.

---

## 5. Reproduction & Execution

### 5.1 Prerequisites
Python 3.10+ with standard scientific libraries:
```bash
pip install numpy pandas scipy pyyaml matplotlib
```

### 5.2 Setting NOAA CDO API Token
The NOAA CDO API token must be set as an environment variable before execution:
```powershell
$env:NOAA_CDO_TOKEN = "YOUR_NOAA_TOKEN_HERE"
```

### 5.3 Execution Commands
1. **Compile Station Inventory**:
   ```powershell
   python code/noaa_station_selector.py
   ```
2. **Execute Full Weather Ingestion & Aggregation**:
   ```powershell
   python code/noaa_weather_processor.py
   ```
3. **Run Statistical Validation & Comparisons**:
   ```powershell
   python code/noaa_weather_validator.py
   ```

---

## 6. Integration Protocol for Crop Yield Pipeline

### 6.1 Safe Merge Instructions
To incorporate the NOAA weather features into `CropFusion` / `NeuralCQR`:
1. **Join Key**: Inner join on `["GEOID", "Year"]` with the primary modeling table (`Paper3_MegaDataset_SPEI_FINAL.csv`).
2. **Strict Split Boundary Enforcement**:
   - Fit all scaling transformers (`StandardScaler`, `QuantileTransformer`) **STRICTLY on FIT years (1985–2013)**.
   - Never compute means, standard deviations, or quantiles across DEV, CAL, or TEST splits.
3. **LOSO Cross-Validation Integrity**:
   - When evaluating held-out states (e.g. Illinois), train all spatial encoders and scalers strictly on the remaining 5 states.
   - Do not allow state-level weather normalization across LOSO fold boundaries.
