import os
import requests
from pathlib import Path
import xarray as xr
import netCDF4 as nc

def audit_test_granule():
    audit_dir = Path("data/nldas_pipeline/audit")
    audit_dir.mkdir(parents=True, exist_ok=True)
    target_nc = audit_dir / "NLDAS_FORA0125_H.A20230715.1200.020.nc"

    token = os.environ.get("NASA_EARTHDATA_TOKEN")
    if not token:
        with open(".env", "r") as f:
            for line in f:
                if line.strip().startswith("NASA_EARTHDATA_TOKEN="):
                    token = line.strip().split("=", 1)[1].strip()
                    break

    headers = {"Authorization": f"Bearer {token}"}
    url = "https://data.gesdisc.earthdata.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.2.0/2023/196/NLDAS_FORA0125_H.A20230715.1200.020.nc"

    print(f"Downloading test granule from {url}...")
    r = requests.get(url, headers=headers, stream=True, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP download failed: {r.status_code} {r.text[:300]}")

    with open(target_nc, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)

    print(f"Downloaded: {target_nc} ({target_nc.stat().st_size} bytes)")

    # Open with netCDF4
    print("\n--- Inspecting with netCDF4 ---")
    with nc.Dataset(target_nc, "r") as ds:
        print("Format:", ds.data_model)
        print("Global attributes:")
        for k in ds.ncattrs()[:10]:
            print(f"  {k}: {ds.getncattr(k)}")
        
        print("\nDimensions:")
        for k, v in ds.dimensions.items():
            print(f"  {k}: {len(v)}")

        print("\nVariables:")
        for k, v in ds.variables.items():
            print(f"  Variable: {k:<12} | Shape: {str(v.shape):<15} | Type: {str(v.dtype):<8} | Units: {v.getncattr('units') if hasattr(v, 'units') else 'N/A'}")

    # Open with xarray
    print("\n--- Inspecting with xarray ---")
    ds_xr = xr.open_dataset(target_nc)
    print("Coordinates:")
    for c in ds_xr.coords:
        coord = ds_xr[c]
        print(f"  {c}: shape={coord.shape}, min={coord.values.min()}, max={coord.values.max()}")

    print("\nData Variables detail:")
    for vname in ds_xr.data_vars:
        da = ds_xr[vname]
        attrs = da.attrs
        fill_val = attrs.get("_FillValue", attrs.get("missing_value", "None"))
        valid_vals = da.values[da.values != fill_val]
        val_min = valid_vals.min() if len(valid_vals) > 0 else "N/A"
        val_max = valid_vals.max() if len(valid_vals) > 0 else "N/A"
        print(f"  {vname}:")
        print(f"    long_name: {attrs.get('long_name')}")
        print(f"    units:     {attrs.get('units')}")
        print(f"    FillValue: {fill_val}")
        print(f"    valid min: {val_min}")
        print(f"    valid max: {val_max}")

if __name__ == "__main__":
    audit_test_granule()
