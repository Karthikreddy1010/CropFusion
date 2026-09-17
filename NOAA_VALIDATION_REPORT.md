# NOAA/NCEI Weather Data Integration & Validation Report

**Study Period**: 1985–2023 (39 Years, 14,245 Days)  
**Study Scope**: 583 Locked Corn Counties across Illinois, Indiana, Iowa, Minnesota, Missouri, and Ohio  
**Evaluation Protocol**: FIT (1985–2013), DEV (2014–2015), CAL (2016–2018), TEST (2019–2023), 6-State LOSO  

---

## Executive Summary

To provide official, empirical ground truth for historical weather and climate extremes in the `CropFusion` / `NeuralCQR` crop yield prediction project, we integrated station-level observations from the **National Oceanic and Atmospheric Administration (NOAA)** and the **National Centers for Environmental Information (NCEI)** Global Historical Climatology Network Daily (GHCND) archive. 

All 583 target counties across the six locked Corn Belt states (IL, IN, IA, MN, MO, OH) were mapped to official GHCND weather stations using authoritative geographical centroids. A multi-tiered network of 626 distinct weather stations was selected based on proximity and historical record continuity. Station data were quality-controlled, converted to physical agronomic units, aggregated to county level via Inverse Distance Weighting (IDW), and synthesized into annual growing-season agroclimatic features.

All deliverables were generated without modifying ML model code, loss functions, temporal split boundaries, or held-out LOSO folds.

---

## Answers to Section 20 Questions

### 1. Did you obtain true station-level or gridded weather data from an official NOAA/NCEI service? Which one?
**Yes.** We obtained **true station-level daily ground observations** from the official **NOAA / NCEI Global Historical Climatology Network Daily (GHCND)** archive. 
- These are direct physical measurements recorded by weather stations belonging to the U.S. Cooperative Observer Network (COOP) and the National Weather Service (NWS) Automated Surface Observing System (ASOS / WBAN).
- No gridded reanalysis, synthetic interpolation, or model proxy was used as source data.

### 2. What are the exact NOAA endpoints and dataset IDs used?
We utilized two complementary official NOAA/NCEI access services:
1. **NOAA Climate Data Online (CDO) API v2**:
   - Base URL: `https://www.ncei.noaa.gov/cdo-web/api/v2/`
   - Endpoints:
     - `/stations`: Used for state-level station discovery, coordinate indexing, date range verification, and coverage scoring.
     - Query Parameters: `datasetid=GHCND`, `locationid=FIPS:<STATE_FIPS>`, `datacategoryid=TEMP`, `datatypeid=TMAX`, `limit=1000`, `offset=...`.
     - Authentication: Read securely via `NOAA_CDO_TOKEN` environment variable.
2. **NOAA NCEI Direct Archive Access**:
   - Base URL: `https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/`
   - Resource: `{STATION_ID}.csv` (direct download of complete, un-truncated daily historical records with native measurement, quality, source, and time-of-observation flags).
   - Dataset ID: `GHCND` (Global Historical Climatology Network Daily).

### 3. How many weather stations were considered, selected, and mapped to the 583 target counties?
- **Total Stations Considered**: 1,946 candidate GHCND stations across the 6 states cataloged via the NOAA CDO API.
- **Active Study-Period Stations**: 1,306 stations having active observations during 1985–2023.
- **County-Station Spatial Pairs Evaluated**: 6,598 pairs within a 50.0 km radius of county centroids.
- **Selected Multi-Station Network**: **626 unique GHCND weather stations** selected to serve the 583 counties as primary (Rank 1) and secondary/backup stations (Rank 2 and Rank 3).
- **Primary Stations**: 421 unique primary stations (each county has exactly 1 primary station; adjacent counties occasionally share a high-continuity long-record station).

### 4. What is the spatial distribution of selected stations across IL, IN, IA, MN, MO, and OH?
The primary network distribution across the six states is detailed below:

| State | Counties | Primary Stations (Rank 1) | Long-Record Primaries (≥80% 1985–2023) | Mean Dist to Centroid (km) | Max Dist to Centroid (km) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Illinois** | 102 | 102 | 102 (100.0%) | 16.15 km | 44.60 km |
| **Indiana** | 92 | 92 | 90 (97.8%) | 16.18 km | 43.57 km |
| **Iowa** | 99 | 99 | 99 (100.0%) | 12.19 km | 36.31 km |
| **Minnesota** | 87 | 87 | 84 (96.6%) | 18.96 km | 49.32 km |
| **Missouri** | 115 | 115 | 115 (100.0%) | 19.40 km | 45.93 km |
| **Ohio** | 88 | 88 | 88 (100.0%) | 16.00 km | 48.65 km |
| **Total / Overall** | **583** | **583** | **578 (99.14%)** | **16.52 km** | **49.32 km** |

### 5. What fraction of counties have at least one long-record station active from 1985 through 2023?
- **99.14% (578 out of 583 counties)** have a designated primary station with continuous long-record status (`first_date <= 1985-01-01`, `last_date >= 2023-01-01`, and historical coverage $\ge 80\%$).
- When secondary stations within 50 km are included, **100.0% of all 583 counties** have active station support spanning the entire 1985–2023 period.

### 6. What is the average distance from county centroids to the nearest selected weather station?
- **Mean Distance**: **16.52 km**
- **Median Distance (50th percentile)**: **13.92 km**
- **25th Percentile**: **6.40 km**
- **75th Percentile**: **25.28 km**
- **Minimum Distance**: **0.15 km** (station situated immediately adjacent to county centroid)
- **Maximum Distance**: **49.32 km** (all 583 counties reside within the 50.0 km search envelope).

### 7. What are the primary weather variables downloaded, their original units, and your conversion formulas?
NOAA GHCND archives daily weather variables in integer tenths:
1. **Maximum Temperature (`TMAX`)**:
   - Original Unit: Tenths of degrees Celsius ($0.1^\circ\text{C}$).
   - Converted Unit: Degrees Celsius ($^\circ\text{C}$).
   - Formula: $T_{\text{max}} (^\circ\text{C}) = \text{TMAX\_raw} / 10.0$
2. **Minimum Temperature (`TMIN`)**:
   - Original Unit: Tenths of degrees Celsius ($0.1^\circ\text{C}$).
   - Converted Unit: Degrees Celsius ($^\circ\text{C}$).
   - Formula: $T_{\text{min}} (^\circ\text{C}) = \text{TMIN\_raw} / 10.0$
3. **Precipitation (`PRCP`)**:
   - Original Unit: Tenths of millimeters ($0.1\text{ mm}$).
   - Converted Unit: Millimeters per day ($\text{mm/day}$).
   - Formula: $\text{PRCP} (\text{mm}) = \text{PRCP\_raw} / 10.0$

Both raw integer values (`TMAX_raw`, `TMIN_raw`, `PRCP_raw`) and converted physical quantities (`TMAX_degC`, `TMIN_degC`, `PRCP_mm`) are preserved in Deliverables 2 and 3.

### 8. What quality-control checks were performed on raw weather data, and what anomalies were detected?
Four levels of rigorous quality control were enforced:
1. **Official NOAA Quality Flag Filtering**:
   - GHCND attributes format: `(measurement_flag, quality_flag, source_flag, time_of_obs)`.
   - Any observation with an active quality flag (`D`, `G`, `I`, `K`, `L`, `M`, `N`, `O`, `R`, `S`, `T`, `W`, `X`, `Z`) indicating spatial inconsistency, failed internal consistency checks, or unrealistic z-scores was discarded and replaced with `NaN`.
2. **Physical Bound Verification**:
   - Temperature: Filtered to $[-50.0^\circ\text{C}, +55.0^\circ\text{C}]$. Values outside were rejected.
   - Precipitation: Filtered to $[0.0\text{ mm}, 400.0\text{ mm/day}]$. Negative values or values $>400\text{ mm}$ were rejected.
3. **Thermodynamic Consistency**:
   - Enforced $T_{\text{max}} \ge T_{\text{min}}$. Any record where $T_{\text{max}} < T_{\text{min}}$ was rejected.
4. **Frozen Sensor & Zero-Variance Checks**:
   - Time series were inspected for identical non-zero values repeated across $>14$ consecutive days (none detected in selected stations).

### 9. How did you aggregate station data to county level, and how did you handle missing values or station transitions?
1. **Spatial Aggregation**:
   - For each county and each date $d$, the primary station observation was used if valid (`aggregation_method = "primary_station"`).
   - If the primary station had missing or QC-rejected data, secondary stations (Rank 2 and Rank 3) within 50 km were aggregated using **Inverse Distance Weighting (IDW)** with power $p=2$:
     $$w_i = \frac{1}{d_i^2}, \quad \bar{X} = \frac{\sum_{i=1}^k w_i X_i}{\sum_{i=1}^k w_i}$$
     where $d_i$ is distance from county centroid in km (`aggregation_method = "idw_interpolation"`).
2. **Temporal Gap Imputation**:
   - For short isolated missing periods ($\le 3$ days), linear time interpolation was applied (`aggregation_method = "linear_interp"`).
   - For any residual regional outages, 39-year day-of-year (DOY) median climatology was used (`aggregation_method = "climatology"`).
3. **Provenance Logging**:
   - Every single daily record in `noaa_daily_county_weather.csv` logs `n_contributing_stations` and `aggregation_method`.

### 10. What agronomic weather indicators were computed, and what formulas were used?
1. **Corn Growing Degree Days (86/50 cutoff method)**:
   $$T_{\text{max,adj}} = \min(30.0, \max(10.0, T_{\text{max}}))$$
   $$T_{\text{min,adj}} = \min(30.0, \max(10.0, T_{\text{min}}))$$
   $$T_{\text{mean,adj}} = \frac{T_{\text{max,adj}} + T_{\text{min,adj}}}{2.0}$$
   $$\text{GDD}_{\text{corn}} = \max(0.0, T_{\text{mean,adj}} - 10.0)$$
   *Units*: °C-days.
2. **Heat Stress Indicator Days**:
   - `heat_stress_30`: Binary flag for $T_{\text{max}} \ge 30^\circ\text{C}$ (photosynthetic slowdown).
   - `heat_stress_32`: Binary flag for $T_{\text{max}} \ge 32^\circ\text{C}$ (pollen viability reduction).
   - `heat_stress_35`: Binary flag for $T_{\text{max}} \ge 35^\circ\text{C}$ (silk desiccation & reproductive failure).
3. **Seasonal Aggregations**:
   - `GDD_accum_may_sep`: Cumulative GDD from May 1 to September 30 (core reproductive season).
   - `GDD_accum_apr_oct`: Cumulative GDD from April 1 to October 31 (full growing season).
   - `Precip_total_growseason_mm`: Total precipitation May 1 to September 30.
   - `TMAX_mean_growseason_C`, `TMIN_mean_growseason_C`: Growing season mean temperatures.
   - `TMAX_max_growseason_C`: Highest single-day maximum temperature (May–Sep).
   - `Extreme_heat_days_30_growseason`, `Extreme_heat_days_32_growseason`, `Extreme_heat_days_35_growseason`: Cumulative heat day counts.
   - `Dry_days_growseason` ($\text{PRCP} < 1\text{ mm}$), `Wet_days_growseason` ($\text{PRCP} \ge 1\text{ mm}$).
   - `Drought_index_proxy`: Hydrothermal ratio $\frac{\text{Precip}}{\text{GDD} + 10^{-4}}$.

### 11. How does the temporal coverage and completeness compare across FIT (1985-2013), DEV (2014-2015), CAL (2016-2018), and TEST (2019-2023)?
Across all temporal splits, the final processed county-level weather series achieves **100.0% completeness** without missing days:

| Split | Years | Span (Years) | Total Days / County | Primary Station Raw Coverage | Final Post-Imputation Completeness |
|:---|:---:|:---:|:---:|:---:|:---:|
| **FIT** | 1985–2013 | 29 | 10,592 | 96.8% | **100.0%** |
| **DEV** | 2014–2015 | 2 | 730 | 97.4% | **100.0%** |
| **CAL** | 2016–2018 | 3 | 1,096 | 97.6% | **100.0%** |
| **TEST** | 2019–2023 | 5 | 1,826 | 97.9% | **100.0%** |
| **Total / Full** | **1985–2023** | **39** | **14,245** | **97.1%** | **100.0%** |

The raw primary station completeness remains remarkably uniform (96.8% to 97.9%), demonstrating that station network quality does not degrade in earlier decades (1980s) or recent test years.

### 12. How do the NOAA-derived weather variables compare statistically (mean, std, correlation) to existing climate features in the dataset?
We conducted an exact statistical comparison across all 22,737 merged county-years (583 counties $\times$ 39 years):

| Feature Comparison | Metric | NOAA GHCND (Mean ± Std) | Existing Repo Dataset (Mean ± Std) | Pearson $r$ | RMSE | Mean Bias (NOAA - Repo) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Peak Temperature** | $T_{\text{max}}$ Max (°C) | $35.48 \pm 2.08$ | $33.69 \pm 2.21$ | **0.7790** | 2.31 °C | +1.79 °C |
| **Corn GDD (Apr–Oct)** | Cumulative GDD (°C-d) | $1892.84 \pm 222.1$ | $2001.28 \pm 250.4$ | **0.9483** | 157.17 °C-d | -108.43 °C-d |
| **Corn GDD (May–Sep)** | Cumulative GDD (°C-d) | $1629.86 \pm 185.3$ | $2001.28 \pm 250.4$ | **0.9309** | 403.87 °C-d | -371.42 °C-d |
| **Precipitation** | Total Grow Season (mm) | $520.63 \pm 124.6$ | $694.20 \pm 146.5$ | **0.6259** | 217.62 mm | -173.57 mm |

**Split-by-Split Pearson Correlation ($r$)**:
- **FIT (1985–2013)**: $T_{\text{max}}$ Max $r = 0.7742$, GDD (Apr–Oct) $r = 0.9461$, Precip $r = 0.6214$
- **DEV (2014–2015)**: $T_{\text{max}}$ Max $r = 0.7812$, GDD (Apr–Oct) $r = 0.9510$, Precip $r = 0.6401$
- **CAL (2016–2018)**: $T_{\text{max}}$ Max $r = 0.7925$, GDD (Apr–Oct) $r = 0.9542$, Precip $r = 0.6358$
- **TEST (2019–2023)**: $T_{\text{max}}$ Max $r = 0.7831$, GDD (Apr–Oct) $r = 0.9521$, Precip $r = 0.6312$

### 13. Did you detect any discrepancies between existing climate features and official NOAA records?
**Yes, two critical scientific discrepancies were uncovered**:
1. **Reanalysis Spatial Smoothing Bias in Peak Heat**:
   - NOAA ground stations recorded a seasonal peak $T_{\text{max}}$ average of **35.48°C**, whereas ERA5 recorded **33.69°C**—a persistent **+1.79°C ground heat bias**.
   - *Explanation*: ERA5 averages temperatures over a $\sim 31\text{ km} \times 31\text{ km}$ grid cell, blending agricultural canopy microclimates with surrounding terrain. True physical weather stations capture surface extreme heat spikes that cause reproductive corn failure (pollen desiccation at $\ge 35^\circ\text{C}$). Relying solely on ERA5 underestimates the frequency and intensity of compound dry-hot wind events (CDHW).
2. **Growing Season Definition Window Discrepancy**:
   - The repository feature `Precip_growseason_mm` and `GDD_Accumulated` were accumulated over a 7-month window (**April 1 through October 31**, 214 days, mean 694.2 mm).
   - In contrast, the standard agronomic core corn reproductive window is May 1 to September 30 (153 days, mean 520.6 mm).
   - When NOAA GDD is accumulated over the matching April–October window (`GDD_accum_apr_oct`), the Pearson correlation with the existing dataset reaches **$r = 0.9483$** with a difference of only 5.4%, proving that thermal dynamics are exceptionally well aligned once window definitions are reconciled.

### 14. How should the processed NOAA weather features be merged into the existing pipeline without violating the locked evaluation design?
To preserve strict scientific validity:
1. **Merge Key**: Inner join on `["GEOID", "Year"]` with `outputs/noaa_weather/noaa_county_weather_annual.csv`.
2. **Split Isolation**:
   - All feature scalers (`StandardScaler`, `QuantileTransformer`, `RobustScaler`) must be fitted **STRICTLY on the FIT split (1985–2013)**.
   - Means, standard deviations, and median statistics must NEVER be computed across DEV, CAL, or TEST splits.
3. **LOSO Cross-Validation Integrity**:
   - In 6-state LOSO, when state $s$ is held out (e.g., Iowa), all feature engineering and scaling must be fitted strictly on the remaining 5 training states.
   - Zero target or weather normalization parameters may leak across state boundaries.
4. **Model Architecture Protection**:
   - The processed annual NOAA features (`GDD_accum_may_sep`, `Precip_total_growseason_mm`, `TMAX_max_growseason_C`, `Extreme_heat_days_35_growseason`, `Drought_index_proxy`) should be concatenated directly to the tabular input vector $x$ entering the MLP / Transformer trunk.
   - Do NOT modify the quantile loss function, pinball loss weights, or quantile outputs ($\tau \in \{0.05, 0.50, 0.95\}$).
