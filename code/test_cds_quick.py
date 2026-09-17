"""
=============================================================================
STATUS: ABANDONED / DISABLED
=============================================================================
ERA5-Land acquisition has been officially discontinued and decommissioned by
user directive (2026-09-12).
=============================================================================
"""
import sys
print("ERROR: ERA5-Land CDS test script has been permanently DISABLED by user directive.")
sys.exit(0)

import time
import cdsapi
import xarray as xr

def main():
    print("Initializing CDS client...")
    client = cdsapi.Client()
    
    test_dir = os.path.join("data", "era5_daily", "test")
    os.makedirs(test_dir, exist_ok=True)
    target_nc = os.path.join(test_dir, "test_era5_land_20230701.nc")
    
    request = {
        "variable": [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "total_precipitation",
            "surface_solar_radiation_downwards",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "surface_pressure"
        ],
        "year": "2023",
        "month": "07",
        "day": "01",
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [50.0, -104.5, 35.5, -80.0],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }
    
    print(f"Submitting 1-day test request to reanalysis-era5-land...")
    print(f"Variables: {request['variable']}")
    print(f"Area: {request['area']}")
    print(f"Target: {target_nc}")
    
    t0 = time.time()
    client.retrieve("reanalysis-era5-land", request, target_nc)
    elapsed = time.time() - t0
    print(f"Download completed in {elapsed:.1f}s. File size: {os.path.getsize(target_nc) / 1024 / 1024:.2f} MB")
    
    print("Inspecting downloaded NetCDF file with xarray...")
    ds = xr.open_dataset(target_nc)
    print(f"Dimensions: {dict(ds.sizes)}")
    print(f"Data variables: {list(ds.data_vars.keys())}")
    print(f"Coordinates: {list(ds.coords.keys())}")
    for var_name, da in ds.data_vars.items():
        print(f"  {var_name}: shape={da.shape}, min={float(da.min()):.4f}, max={float(da.max()):.4f}, units={da.attrs.get('units', 'unknown')}")
    ds.close()
    print("Test passed successfully!")

if __name__ == "__main__":
    main()
