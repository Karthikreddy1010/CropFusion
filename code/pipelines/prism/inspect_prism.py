#!/usr/bin/env python3
"""
inspect_prism.py

Inspects a real downloaded PRISM file (ZIP or netCDF) and reports every
property the pipeline needs BEFORE any parsing code assumes it:

    - archive contents / internal filenames
    - raster format (BIL, ASCII, TIFF, netCDF...)
    - CRS
    - affine transform / spatial resolution
    - grid dimensions
    - nodata value
    - dtype
    - variable name(s)
    - units (read from PRISM's own metadata files, never assumed)
    - spatial extent (bounds)
    - min/max of actual grid values (sanity, not QC)

Run this on the actual test files BEFORE trusting download_prism.py's
default file-extension assumption (.zip) or before writing
aggregate_prism_county.py's zonal-stats logic.

Usage
-----
    python code/pipelines/prism/inspect_prism.py --file data/prism/raw/tmax/20190715.zip
    python code/pipelines/prism/inspect_prism.py --file data/prism/raw/tmax/20190715.zip \
        --extract-to /tmp/prism_inspect --out reports/prism_grid_metadata.csv
"""

from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path
from typing import Optional


def list_zip_contents(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return zf.namelist()


def extract_zip(zip_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
        names = zf.namelist()
    return [out_dir / n for n in names]


def read_prism_info_txt(extracted_files: list[Path]) -> dict:
    """
    PRISM ZIP bundles typically include an *_info.txt / *.xml companion
    file documenting units, version, and creation date. We parse whatever
    text/xml companion exists rather than assuming units.
    """
    info = {}
    for p in extracted_files:
        if p.suffix.lower() in (".txt", ".xml", ".stx", ".prj"):
            try:
                text = p.read_text(errors="ignore")
            except OSError:
                continue
            info[p.name] = text
    return info


def inspect_with_rasterio(raster_path: Path) -> dict:
    """
    Inspect a BIL/ASCII/TIFF raster with rasterio. Requires `pip install
    rasterio`. Import is deferred so this script can still report ZIP
    contents even if rasterio isn't installed yet.
    """
    try:
        import rasterio
    except ImportError:
        return {"error": "rasterio not installed — run `pip install rasterio`"}

    result = {}
    with rasterio.open(raster_path) as ds:
        result["driver"] = ds.driver
        result["crs"] = str(ds.crs)
        result["transform"] = str(ds.transform)
        result["width"] = ds.width
        result["height"] = ds.height
        result["count_bands"] = ds.count
        result["dtype"] = str(ds.dtypes[0])
        result["nodata"] = ds.nodata
        result["bounds"] = str(ds.bounds)
        result["res"] = str(ds.res)
        band1 = ds.read(1, masked=True)
        result["data_min"] = float(band1.min()) if band1.count() else None
        result["data_max"] = float(band1.max()) if band1.count() else None
        result["tags"] = dict(ds.tags())
    return result


def inspect_netcdf(nc_path: Path) -> dict:
    try:
        import xarray as xr
    except ImportError:
        return {"error": "xarray not installed — run `pip install xarray netCDF4`"}

    result = {}
    ds = xr.open_dataset(nc_path)
    result["variables"] = list(ds.data_vars.keys())
    result["dims"] = dict(ds.dims)
    result["coords"] = list(ds.coords.keys())
    result["global_attrs"] = dict(ds.attrs)
    for v in ds.data_vars:
        result[f"{v}_attrs"] = dict(ds[v].attrs)
        result[f"{v}_dtype"] = str(ds[v].dtype)
        result[f"{v}_shape"] = ds[v].shape
    ds.close()
    return result


def find_raster_file(extracted_files: list[Path]) -> Optional[Path]:
    # PRISM BIL bundles: <name>.bil is the raster, <name>.hdr is the header.
    candidates = [p for p in extracted_files if p.suffix.lower() == ".bil"]
    if candidates:
        return candidates[0]
    candidates = [p for p in extracted_files if p.suffix.lower() in (".tif", ".tiff", ".asc")]
    if candidates:
        return candidates[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Inspect a real PRISM download")
    parser.add_argument("--file", required=True, help="Path to a downloaded PRISM .zip or .nc file")
    parser.add_argument("--extract-to", default=None, help="Directory to extract ZIP contents into")
    parser.add_argument("--out", default=None, help="Optional CSV path to append a metadata row")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} does not exist. Download a real test file first "
            "(this script never fabricates metadata)."
        )

    print("=" * 70)
    print(f"INSPECTING: {file_path}")
    print(f"File size: {file_path.stat().st_size} bytes")
    print("=" * 70)

    report = {"file": str(file_path), "file_size": file_path.stat().st_size}

    if file_path.suffix.lower() == ".zip":
        contents = list_zip_contents(file_path)
        print("\nZIP contents:")
        for c in contents:
            print(f"  - {c}")
        report["zip_contents"] = ";".join(contents)

        extract_dir = Path(args.extract_to) if args.extract_to else file_path.parent / (file_path.stem + "_extracted")
        extracted = extract_zip(file_path, extract_dir)

        info = read_prism_info_txt(extracted)
        if info:
            print("\nCompanion metadata files found:")
            for name, text in info.items():
                print(f"\n--- {name} ---")
                print(text[:2000])
                report[f"meta::{name}"] = text[:2000]
        else:
            print("\nNo .txt/.xml/.stx/.prj companion metadata files found in the archive.")

        raster_file = find_raster_file(extracted)
        if raster_file:
            print(f"\nRaster file identified: {raster_file.name}")
            raster_info = inspect_with_rasterio(raster_file)
            print("\nRaster properties (rasterio):")
            for k, v in raster_info.items():
                if k == "tags":
                    print(f"  tags: {v}")
                else:
                    print(f"  {k}: {v}")
            report.update({f"raster::{k}": v for k, v in raster_info.items()})
        else:
            print("\nNo .bil/.tif/.asc raster file found — inspect the ZIP contents "
                  "list above manually; PRISM's internal naming may differ from expected.")

    elif file_path.suffix.lower() == ".nc":
        nc_info = inspect_netcdf(file_path)
        print("\nnetCDF properties:")
        for k, v in nc_info.items():
            print(f"  {k}: {v}")
        report.update({f"nc::{k}": v for k, v in nc_info.items()})

    else:
        print(f"Unrecognized extension {file_path.suffix} — inspect manually.")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not out_path.exists()
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(report.keys()))
            if is_new:
                writer.writeheader()
            writer.writerow(report)
        print(f"\nAppended metadata row to {out_path}")

    print("\n" + "=" * 70)
    print("IMPORTANT: use the values printed above — CRS, nodata, units, "
          "dimensions — to configure aggregate_prism_county.py. Do not "
          "hard-code assumed values.")
    print("=" * 70)


if __name__ == "__main__":
    main()
