#!/usr/bin/env python3
"""
fao56_et0.py

Standard FAO-56 Penman–Monteith Reference Evapotranspiration (ET0) Implementation
for Paper 3 / NeuralCQR NLDAS-2 Downstream Climate Pipeline.

Methodological Notes:
---------------------
1. Primary FAO-56 Net Longwave (Rnl) formulation:
   Rnl is calculated via FAO-56 Equation 39 using:
     - Tmax, Tmin (K)
     - ea (actual vapor pressure, kPa)
     - Rs (incoming solar radiation, MJ/m2/day)
     - Rso (clear-sky solar radiation, MJ/m2/day)
   NOTE: LWdown_mean (NLDAS downward longwave flux) is retained as a QC/diagnostic
   variable and is NOT used as the primary Rnl because 2m Tair does not represent
   effective surface radiating skin temperature.

2. Saturation Vapor Pressure (es):
   es = 0.5 * [e°(Tmax) + e°(Tmin)] (FAO-56 eq. 12)
   NEVER calculated from e°(Tmean), preserving non-linear Clausius-Clapeyron response.

3. Psychrometric Slope (Delta) & Constant (gamma):
   Delta evaluated at Tmean = (Tmax + Tmin) / 2 (FAO-56 eq. 13)
   gamma = 0.000665 * (PSurf_Pa / 1000) (FAO-56 eq. 8)

4. Net Radiation (Rn):
   Rns = 0.77 * Rs (albedo = 0.23, FAO-56 eq. 38)
   Rn = Rns - Rnl (FAO-56 eq. 40)
   G = 0 (daily time step)

5. Reference Evapotranspiration (ET0):
   ET0 = [0.408*Delta*(Rn - G) + gamma*(900/(Tmean + 273))*u2*(es - ea)] /
         [Delta + gamma*(1 + 0.34*u2)] (FAO-56 eq. 6)
"""

from __future__ import annotations

import datetime as dt
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import xarray as xr

# Physical constants
STEFAN_BOLTZMANN_MJ = 4.903e-9  # MJ K^-4 m^-2 day^-1
SOLAR_CONSTANT_MJ_MIN = 0.0820  # MJ m^-2 min^-1 (Gsc)
ALBEDO_REF_GRASS = 0.23


def calc_sat_vapor_pressure_single(t_c: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """FAO-56 Eq. 11: Saturation vapor pressure at temperature t (°C) in kPa."""
    return 0.6108 * np.exp((17.27 * t_c) / (t_c + 237.3))


def calc_es(
    tmax_c: Union[float, np.ndarray],
    tmin_c: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """FAO-56 Eq. 12: Mean saturation vapor pressure (kPa)."""
    return 0.5 * (calc_sat_vapor_pressure_single(tmax_c) + calc_sat_vapor_pressure_single(tmin_c))


def calc_delta(tmean_c: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """FAO-56 Eq. 13: Slope of saturation vapor pressure curve (kPa / °C)."""
    e_o = calc_sat_vapor_pressure_single(tmean_c)
    return (4098.0 * e_o) / ((tmean_c + 237.3) ** 2)


def calc_gamma(psurf_pa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """FAO-56 Eq. 8: Psychrometric constant (kPa / °C) from surface pressure in Pa."""
    p_kpa = psurf_pa / 1000.0
    return 0.000665 * p_kpa


def calc_extraterrestrial_radiation(
    lat_deg: Union[float, np.ndarray],
    day_of_year: int,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    FAO-56 Eqs. 21-25: Daily extraterrestrial radiation Ra (MJ m^-2 day^-1)
    and sunset hour angle omega_s (radians).
    """
    phi = np.radians(lat_deg)
    # Solar declination delta (eq. 24)
    delta = 0.409 * np.sin((2.0 * np.pi / 365.0) * day_of_year - 1.39)
    # Inverse relative distance Earth-Sun dr (eq. 23)
    dr = 1.0 + 0.033 * np.cos((2.0 * np.pi / 365.0) * day_of_year)

    # Sunset hour angle omega_s (eq. 25)
    # Clamp argument to [-1.0, 1.0] for physical stability
    arg = -np.tan(phi) * np.tan(delta)
    arg_clamped = np.clip(arg, -1.0, 1.0)
    omega_s = np.arccos(arg_clamped)

    # Extraterrestrial radiation Ra (eq. 21)
    # Constant: (24 * 60 / pi) * Gsc = (1440 / pi) * 0.0820 = 37.58603 MJ m^-2 day^-1
    c = (24.0 * 60.0 / np.pi) * SOLAR_CONSTANT_MJ_MIN
    ra = c * dr * (omega_s * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(omega_s))
    ra = np.maximum(ra, 0.0)
    return ra, omega_s


_FIXED_ELEVATION_CACHE: Optional[xr.DataArray] = None

def get_fixed_grid_elevation(elev_path: Optional[Union[str, Path]] = None) -> xr.DataArray:
    """
    Loads the official static fixed NLDAS elevation grid (116 x 196).
    Source: Official NASA GSFC NLDAS Elevation Dataset (NLDAS_elevation.nc4).
    """
    global _FIXED_ELEVATION_CACHE
    if _FIXED_ELEVATION_CACHE is not None:
        return _FIXED_ELEVATION_CACHE

    if elev_path is None:
        p = Path("data/nldas_pipeline/config/nldas_grid_elevation.nc")
    else:
        p = Path(elev_path)

    if not p.is_absolute():
        try:
            from nldas_common import get_project_root
            proj_root = get_project_root()
        except ImportError:
            try:
                from data.nldas_pipeline.scripts.nldas_common import get_project_root
                proj_root = get_project_root()
            except ImportError:
                proj_root = Path(__file__).resolve().parents[3]
        p = (proj_root / p).resolve()

    if not p.exists():
        raise FileNotFoundError(f"Authoritative NLDAS elevation artifact not found at: {p}")

    ds = xr.open_dataset(p)
    elev = ds["elevation_m"].load()
    ds.close()
    _FIXED_ELEVATION_CACHE = elev
    return elev


def calc_clear_sky_radiation(
    ra: Union[float, np.ndarray],
    elevation_m: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    FAO-56 Eq. 37: Clear-sky solar radiation Rso (MJ m^-2 day^-1).
    
    Rso = (0.75 + 2e-5 * z) * Ra
    where z is the geographically fixed terrain elevation in meters above sea level.
    NOTE: Daily meteorological surface pressure is strictly prohibited from deriving z.
    """
    elev = np.maximum(elevation_m, 0.0)
    rso = (0.75 + 2.0e-5 * elev) * ra
    return rso


def calc_rnl_fao56(
    tmax_c: Union[float, np.ndarray],
    tmin_c: Union[float, np.ndarray],
    ea_kpa: Union[float, np.ndarray],
    rs: Union[float, np.ndarray],
    rso: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    FAO-56 Eq. 39: Net outgoing longwave radiation Rnl (MJ m^-2 day^-1).
    
    Rnl = sigma * [(Tmax_K^4 + Tmin_K^4)/2] * (0.34 - 0.14 * sqrt(ea)) * [1.35*(Rs/Rso) - 0.35]
    """
    tmax_k = tmax_c + 273.15
    tmin_k = tmin_c + 273.15
    t_term = (tmax_k**4 + tmin_k**4) / 2.0

    # Humidity term (0.34 - 0.14 * sqrt(ea))
    # ea clamped >= 0 to prevent complex numbers
    ea_safe = np.maximum(ea_kpa, 0.0)
    humidity_term = np.maximum(0.34 - 0.14 * np.sqrt(ea_safe), 0.05)

    # Cloudiness term: [1.35 * (Rs / Rso) - 0.35]
    # FAO-56 specifies Rs / Rso clamped to [0.3, 1.0]
    rso_safe = np.maximum(rso, 1e-4)
    rel_solar = np.clip(rs / rso_safe, 0.3, 1.0)
    cloud_term = 1.35 * rel_solar - 0.35
    cloud_term = np.clip(cloud_term, 0.05, 1.0)

    rnl = STEFAN_BOLTZMANN_MJ * t_term * humidity_term * cloud_term
    return np.maximum(rnl, 0.0)


def calc_fao56_et0_from_dataset(
    ds: xr.Dataset,
    day_of_year: int,
    g_soil: float = 0.0,
    elevation_m: Optional[Union[float, np.ndarray, xr.DataArray]] = None,
) -> xr.Dataset:
    """
    Computes complete FAO-56 Penman–Monteith Reference Evapotranspiration (ET0)
    and intermediate radiative/thermodynamic fields directly from an NLDAS daily NetCDF dataset.
    
    Inputs in ds:
      - Rs_daily_MJ [MJ m^-2 day^-1]
      - LWdown_mean [W m^-2] (retained for diagnostic QC)
      - u2_daily_mean [m s^-1]
      - ea_daily_mean_kPa [kPa]
      - PSurf_daily_mean_Pa [Pa]
      - Tair_daily_mean_C [°C]
      - Tair_daily_min_C [°C]
      - Tair_daily_max_C [°C]
      - lat, lon coordinates
    """
    tmax = ds["Tair_daily_max_C"]
    tmin = ds["Tair_daily_min_C"]
    tmean = 0.5 * (tmax + tmin)  # Standard FAO-56 eq. 9
    ea = ds["ea_daily_mean_kPa"]
    u2 = ds["u2_daily_mean"]
    psurf = ds["PSurf_daily_mean_Pa"]
    rs = ds["Rs_daily_MJ"]

    # 2D latitude grid
    lat = ds["lat"]
    # Broadcast lat to 2D matching (lat, lon)
    lat_2d, _ = xr.broadcast(lat, ds["lon"])

    # Fixed elevation grid (geographically invariant)
    if elevation_m is None:
        elev_da = get_fixed_grid_elevation()
        elev_vals = elev_da.values
    elif isinstance(elevation_m, xr.DataArray):
        elev_vals = elevation_m.values
    else:
        elev_vals = np.asarray(elevation_m)

    # 1. Extraterrestrial radiation Ra
    ra_vals, _ = calc_extraterrestrial_radiation(lat_2d.values, day_of_year)
    ra = xr.DataArray(ra_vals.astype(np.float32), coords=ds.coords, dims=("lat", "lon"))

    # 2. Clear-sky radiation Rso using fixed elevation z (meters)
    rso_vals = calc_clear_sky_radiation(ra_vals, elevation_m=elev_vals)
    rso = xr.DataArray(rso_vals.astype(np.float32), coords=ds.coords, dims=("lat", "lon"))

    # 3. Saturation vapor pressure es
    es_vals = calc_es(tmax.values, tmin.values)
    es = xr.DataArray(es_vals.astype(np.float32), coords=ds.coords, dims=("lat", "lon"))

    # 4. Psychrometric slope Delta
    delta_vals = calc_delta(tmean.values)
    delta = xr.DataArray(delta_vals.astype(np.float32), coords=ds.coords, dims=("lat", "lon"))

    # 5. Psychrometric constant gamma
    gamma_vals = calc_gamma(psurf.values)
    gamma = xr.DataArray(gamma_vals.astype(np.float32), coords=ds.coords, dims=("lat", "lon"))

    # 6. Net shortwave radiation Rns = 0.77 * Rs
    rns = (1.0 - ALBEDO_REF_GRASS) * rs

    # 7. Standard FAO-56 Net longwave radiation Rnl (Eq. 39)
    rnl_vals = calc_rnl_fao56(tmax.values, tmin.values, ea.values, rs.values, rso.values)
    rnl = xr.DataArray(rnl_vals.astype(np.float32), coords=ds.coords, dims=("lat", "lon"))

    # Diagnostic QC longwave from NLDAS LWdown (retained for comparison only)
    # Rnl_diagnostic = [sigma * T_mean_K^4 - LWdown_mean] * 0.0864
    t_mean_k = tmean + 273.15
    rnl_diag_vals = (5.670374e-8 * (t_mean_k.values**4) - ds["LWdown_mean"].values) * 0.0864
    rnl_diagnostic = xr.DataArray(rnl_diag_vals.astype(np.float32), coords=ds.coords, dims=("lat", "lon"))

    # 8. Net Radiation Rn = Rns - Rnl
    rn = rns - rnl

    # 9. FAO-56 Penman–Monteith ET0
    t_factor = 900.0 / (tmean + 273.15)
    vpd = np.maximum(es - ea, 0.0)
    rad_term = 0.408 * delta * (rn - g_soil)
    wind_term = gamma * t_factor * u2 * vpd
    denom = delta + gamma * (1.0 + 0.34 * u2)

    et0 = (rad_term + wind_term) / denom
    # Reference ET is non-negative
    et0 = xr.where(et0 < 0.0, 0.0, et0).astype(np.float32)

    # Return enriched dataset
    out_ds = ds.copy()
    out_ds["elevation_m"] = xr.DataArray(elev_vals.astype(np.float32), coords=ds.coords, dims=("lat", "lon"))
    out_ds["Ra_daily_MJ"] = ra
    out_ds["Rso_daily_MJ"] = rso
    out_ds["es_daily_kPa"] = es
    out_ds["Delta_daily_kPa_C"] = delta
    out_ds["gamma_daily_kPa_C"] = gamma
    out_ds["Rns_daily_MJ"] = rns.astype(np.float32)
    out_ds["Rnl_fao56_daily_MJ"] = rnl
    out_ds["Rnl_diagnostic_LWdown_MJ"] = rnl_diagnostic
    out_ds["Rn_daily_MJ"] = rn.astype(np.float32)
    out_ds["ET0_fao56_daily_mm"] = et0

    out_ds.attrs["fao56_et0_version"] = "1.0-standard-fao56-eq39-fixed-elevation"
    out_ds.attrs["elevation_source"] = "Official NASA GSFC NLDAS Elevation Dataset (NLDAS_elevation.nc4)"
    out_ds.attrs["fao56_rnl_primary"] = "FAO-56 Eq. 39 empirical (Tmax, Tmin, ea, Rs, Rso)"
    out_ds.attrs["fao56_rnl_diagnostic"] = "NLDAS LWdown balance retained as QC diagnostic"

    return out_ds


def validate_sample_dates(daily_dir: Path) -> list[Dict[str, float]]:
    """
    Loads completed daily NetCDF files, computes FAO-56 ET0 and all 13 reported terms,
    and returns spatial mean summary metrics for validation.
    """
    files = sorted(daily_dir.rglob("nldas_daily_*.nc"))
    results = []

    for f in files:
        # Extract date
        datestr = f.stem.replace("nldas_daily_", "")
        d = dt.datetime.strptime(datestr, "%Y%m%d").date()
        doy = d.timetuple().tm_yday

        ds = xr.open_dataset(f)
        et0_ds = calc_fao56_et0_from_dataset(ds, day_of_year=doy)

        tmax = float(et0_ds["Tair_daily_max_C"].mean().values)
        tmin = float(et0_ds["Tair_daily_min_C"].mean().values)
        tmean = (tmax + tmin) / 2.0

        metrics = {
            "date": d.strftime("%Y-%m-%d"),
            "doy": doy,
            "Rs": float(et0_ds["Rs_daily_MJ"].mean().values),
            "Rso": float(et0_ds["Rso_daily_MJ"].mean().values),
            "Tmax": tmax,
            "Tmin": tmin,
            "Tmean": tmean,
            "ea": float(et0_ds["ea_daily_mean_kPa"].mean().values),
            "Rnl": float(et0_ds["Rnl_fao56_daily_MJ"].mean().values),
            "Rns": float(et0_ds["Rns_daily_MJ"].mean().values),
            "Rn": float(et0_ds["Rn_daily_MJ"].mean().values),
            "Delta": float(et0_ds["Delta_daily_kPa_C"].mean().values),
            "gamma": float(et0_ds["gamma_daily_kPa_C"].mean().values),
            "u2": float(et0_ds["u2_daily_mean"].mean().values),
            "ET0": float(et0_ds["ET0_fao56_daily_mm"].mean().values),
            "Rnl_diag_LWdown": float(et0_ds["Rnl_diagnostic_LWdown_MJ"].mean().values),
        }
        results.append(metrics)
        ds.close()
        et0_ds.close()

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FAO-56 ET0 Validation")
    parser.add_argument("--raw-dir", default="data/nldas_pipeline/raw")
    args = parser.parse_args()

    results = validate_sample_dates(Path(args.raw_dir))
    print(f"Validated {len(results)} dates:")
    for r in results:
        print(f"\n--- Date: {r['date']} (DOY: {r['doy']}) ---")
        print(f"1.  Rs       : {r['Rs']:.3f} MJ m^-2 day^-1")
        print(f"2.  Rso      : {r['Rso']:.3f} MJ m^-2 day^-1")
        print(f"3.  Tmax     : {r['Tmax']:.2f} °C")
        print(f"4.  Tmin     : {r['Tmin']:.2f} °C")
        print(f"5.  Tmean    : {r['Tmean']:.2f} °C")
        print(f"6.  ea       : {r['ea']:.3f} kPa")
        print(f"7.  Rnl      : {r['Rnl']:.3f} MJ m^-2 day^-1 (FAO-56 primary Eq. 39)")
        print(f"    (QC Diag): {r['Rnl_diag_LWdown']:.3f} MJ m^-2 day^-1 (NLDAS LWdown balance)")
        print(f"8.  Rns      : {r['Rns']:.3f} MJ m^-2 day^-1")
        print(f"9.  Rn       : {r['Rn']:.3f} MJ m^-2 day^-1")
        print(f"10. Delta    : {r['Delta']:.4f} kPa °C^-1")
        print(f"11. gamma    : {r['gamma']:.5f} kPa °C^-1")
        print(f"12. u2       : {r['u2']:.2f} m s^-1")
        print(f"13. ET0      : {r['ET0']:.2f} mm day^-1")
