"""In-memory access to the CornSight data store (built by etl/build_store.py)."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

STORE_DIR = Path(os.environ.get("CORNSIGHT_STORE", Path(__file__).resolve().parents[1] / "data_store"))

STATE_NAMES = {"IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "MN": "Minnesota", "MO": "Missouri", "OH": "Ohio"}
METHODS = {
    "group_conditional_cp": "Group-conditional conformal",
    "rolling_static_cp": "Static conformal",
    "standard_aci": "Adaptive conformal (ACI)",
    "sa_aci": "Stratum-aware ACI",
}
DEFAULT_METHOD = "group_conditional_cp"

# Variables, units and plain-language labels. Units are taken from the master variable catalog.
VARS = {
    "prism_tmax_mean_gs": ("Mean daily maximum temperature", "°C"),
    "prism_tmax_max_gs": ("Hottest day", "°C"),
    "prism_tmin_mean_gs": ("Mean daily minimum temperature", "°C"),
    "prism_ppt_total_gs": ("Season precipitation", "mm"),
    "prism_gdd_gs": ("Growing degree days (10–30 °C)", "°C·day"),
    "prism_kdd30_gs": ("Killing degree days above 30 °C", "°C·day"),
    "prism_tmax_days_gt35_gs": ("Days above 35 °C", "days"),
    "prism_heat_exceedance_total_gs": ("Heat exceedance above 35 °C", "°C·day"),
    "nldas_et0_total_gs": ("Reference evapotranspiration", "mm"),
    "nldas_vpd_mean_gs": ("Mean vapour-pressure deficit", "kPa"),
    "nldas_rs_mean_gs": ("Mean solar radiation", "MJ/m²/day"),
    "nldas_humidity_mean_gs": ("Mean actual vapour pressure", "kPa"),
    "nldas_wind_mean_gs": ("Mean wind speed", "m/s"),
    "water_balance_total_gs": ("Water balance (precipitation − ET₀)", "mm"),
    "spi1_prism_min_gs": ("Lowest SPI-1 month", "index"),
    "spi3_prism_min_gs": ("Lowest SPI-3 month", "index"),
    "spi6_prism_min_gs": ("Lowest SPI-6 month", "index"),
    "spi12_prism_min_gs": ("Lowest SPI-12 month", "index"),
    "spi1_prism_mean_gs": ("Mean SPI-1", "index"),
    "spi3_prism_mean_gs": ("Mean SPI-3", "index"),
    "spi6_prism_mean_gs": ("Mean SPI-6", "index"),
    "spi12_prism_mean_gs": ("Mean SPI-12", "index"),
    "spei30_prism_nldas_mean_gs": ("Mean SPEI-30", "index"),
    "spei30_prism_nldas_min_gs": ("Lowest SPEI-30", "index"),
    "spei30_prism_nldas_drought_days_gs": ("Drought days (SPEI-30 < −1)", "days"),
    "spei30_prism_nldas_drought_severity_gs": ("Drought severity", "SPEI·day"),
    "spei90_prism_nldas_mean_gs": ("Mean SPEI-90", "index"),
    "spei90_prism_nldas_min_gs": ("Lowest SPEI-90", "index"),
    "cdhw_count_gs": ("Compound drought–heat episodes", "episodes"),
    "cdhw_total_days_gs": ("Compound drought–heat days", "days"),
    "cdhw_max_duration_gs": ("Longest compound episode", "days"),
    "cdhw_max_severity_gs": ("Most severe compound episode", "SPEI·°C·day"),
    "cdhw_cumulative_severity_gs": ("Compound drought–heat severity", "SPEI·°C·day"),
    "noaa_hail_count_gs": ("Hail reports", "events"),
    "noaa_thunderstorm_wind_count_gs": ("Thunderstorm wind reports", "events"),
    "noaa_tornado_count_gs": ("Tornado reports", "events"),
    "noaa_max_hail_size_gs": ("Largest hail", "inches"),
}


def _clean(v):
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


class Store:
    def __init__(self, path: Path = STORE_DIR):
        if not (path / "manifest.json").exists():
            raise FileNotFoundError(f"Data store not found at {path}. Run etl/build_store.py first.")
        self.path = path
        self.manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        self.counties = pd.read_parquet(path / "counties.parquet").set_index("county_fips")
        cy = pd.read_parquet(path / "county_year.parquet")
        cy = cy.merge(self.counties[["state", "county"]], left_on="county_fips", right_index=True, how="left")
        self.cy = cy.sort_values(["county_fips", "year"]).reset_index(drop=True)
        self.preds = pd.read_parquet(path / "predictions.parquet")
        self.attribution = pd.read_csv(path / "attribution.csv")
        self.geo_counties = (path / "counties.geojson").read_text(encoding="utf-8")
        self.geo_states = (path / "states.geojson").read_text(encoding="utf-8")
        self._derive()

    # ------------------------------------------------------------------ derived columns
    def _derive(self) -> None:
        cy = self.cy
        g = cy.groupby("county_fips")
        # prior five-season average of observed yield (needs >= 3 of the 5 seasons)
        prior = g["yield_t_ha"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
        cy["prior5_avg_t_ha"] = prior
        # percentile of each season within the county's own 1985-2023 record (for proposed relative classes)
        for col, name in [("spei30_prism_nldas_drought_days_gs", "drought_pct"),
                          ("prism_tmax_days_gt35_gs", "heat_pct"),
                          ("prism_kdd30_gs", "kdd_pct"),
                          ("prism_ppt_total_gs", "rain_pct")]:
            cy[name] = g[col].rank(pct=True, method="average")
        self.cy = cy
        self.years = sorted(int(y) for y in cy["year"].unique())
        self.pred_years = sorted(int(y) for y in self.preds["year"].unique())
        self.pred_index = self.preds.set_index(["method", "county_fips", "year"]).sort_index()
        self.cy_index = self.cy.set_index(["county_fips", "year"]).sort_index()

    # ------------------------------------------------------------------ lookups
    def county_exists(self, fips: str) -> bool:
        return fips in self.counties.index

    def county_label(self, fips: str) -> str:
        r = self.counties.loc[fips]
        return f"{r['county']} County, {r['state']}"

    def row(self, fips: str, year: int) -> dict | None:
        try:
            r = self.cy_index.loc[(fips, year)]
        except KeyError:
            return None
        return {k: _clean(v) for k, v in r.items()}

    def pred(self, fips: str, year: int, method: str = DEFAULT_METHOD) -> dict | None:
        try:
            r = self.pred_index.loc[(method, fips, year)]
        except KeyError:
            return None
        return {k: _clean(v) for k, v in r.items()}

    def find_counties(self, q: str, limit: int = 10) -> list[dict]:
        ql = q.strip().lower().replace(" county", "")
        state = None
        for ab, nm in STATE_NAMES.items():
            for token in (f", {ab.lower()}", f" {nm.lower()}", f", {nm.lower()}"):
                if ql.endswith(token):
                    state, ql = ab, ql[: -len(token)].strip(" ,")
        df = self.counties
        m = df["county"].str.lower().str.startswith(ql) if ql else pd.Series(True, index=df.index)
        if state:
            m &= df["state"] == state
        hits = df[m].head(limit)
        return [{"fips": f, "county": r["county"], "state": r["state"]} for f, r in hits.iterrows()]


@lru_cache(maxsize=1)
def get_store() -> Store:
    return Store()


def clean(v):
    return _clean(v)
