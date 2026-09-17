import xarray as xr
import numpy as np

def inspect_data():
    ds = xr.open_dataset("data/nldas_pipeline/audit/NLDAS_FORA0125_H.A20230715.1200.020.nc")
    print("=== FULL CONUS GRID STATS (excluding NaNs) ===")
    for v in ["Tair", "Qair", "PSurf", "Wind_E", "Wind_N", "LWdown", "SWdown", "Rainf"]:
        arr = ds[v].values
        valid = arr[~np.isnan(arr)]
        units = ds[v].attrs.get("units", "")
        print(f"  {v:<8}: min={valid.min():>10.4f}, max={valid.max():>10.4f}, mean={valid.mean():>10.4f} {units}")

    # Spatial subsetting test
    sub = ds.sel(lat=slice(35.5, 50.0), lon=slice(-104.5, -80.0))
    print("\n=== SUBSETTED STUDY REGION (7 States: IL, IN, IA, MN, MO, NE, OH) ===")
    print(f"Subset shape: lat={len(sub.lat)}, lon={len(sub.lon)} (Total cells: {len(sub.lat)*len(sub.lon)})")
    for v in ["Tair", "Qair", "PSurf", "Wind_E", "Wind_N", "LWdown", "SWdown"]:
        arr = sub[v].values
        valid = arr[~np.isnan(arr)]
        units = sub[v].attrs.get("units", "")
        print(f"  {v:<8}: min={valid.min():>10.4f}, max={valid.max():>10.4f}, mean={valid.mean():>10.4f} {units}")

    # Test FAO-56 conversion functions
    # 1. Wind speed u2
    wind_10 = np.sqrt(sub["Wind_E"].values**2 + sub["Wind_N"].values**2)
    wind_2 = 0.748 * wind_10
    print(f"\nFAO-56 Derived 2m Wind Speed (u2): min={np.nanmin(wind_2):.2f}, max={np.nanmax(wind_2):.2f}, mean={np.nanmean(wind_2):.2f} m/s")

    # 2. Vapor pressure ea (kPa)
    q = sub["Qair"].values
    p = sub["PSurf"].values
    ea_kpa = (q * p / (0.622 + 0.378 * q)) / 1000.0
    print(f"FAO-56 Derived Vapor Pressure (ea): min={np.nanmin(ea_kpa):.3f}, max={np.nanmax(ea_kpa):.3f}, mean={np.nanmean(ea_kpa):.3f} kPa")

    # 3. Psychrometric constant gamma (kPa / deg C)
    gamma = 0.000665 * (p / 1000.0)
    print(f"FAO-56 Derived Psychrometric Constant (gamma): min={np.nanmin(gamma):.5f}, max={np.nanmax(gamma):.5f}, mean={np.nanmean(gamma):.5f} kPa/C")

    # 4. Air Temperature in deg C
    tair_c = sub["Tair"].values - 273.15
    print(f"Air Temperature (deg C): min={np.nanmin(tair_c):.2f}, max={np.nanmax(tair_c):.2f}, mean={np.nanmean(tair_c):.2f} C")

if __name__ == "__main__":
    inspect_data()
