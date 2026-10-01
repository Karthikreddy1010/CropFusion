"""
03_build_county_weights.py -- county-polygon fractional-area zonal weights.

For each grid (NLDAS 0.125 deg study subset; PRISM 2.5 arc-min CONUS) exactextract computes, for
each of the 583 study-county polygons, every intersecting cell and the fraction of that cell covered
by the polygon. Weight = coverage_fraction x cos(latitude of cell centre) (true area weighting on a
geographic lat/lon grid). County value on a day = sum(w * v) / sum(w) over cells with valid data.

Outputs (master_dataset/intermediate/weights/):
  nldas_county_weights.parquet   (county_fips, row, col, coverage, weight)
  prism_county_weights.parquet
  *_weights_summary.csv          (cells per county, covered area check)
Usage:  python 03_build_county_weights.py nldas|prism
"""
from __future__ import annotations

import sys
import zipfile

import numpy as np
import pandas as pd
import rasterio
from exactextract import exact_extract
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from md_common import INTER, NLDAS_ELEV, PRISM_RAW, get_logger, load_study_counties

log = get_logger("03_build_county_weights")
WDIR = INTER / "weights"
WDIR.mkdir(parents=True, exist_ok=True)


def weights_for_grid(transform, shape, crs, counties, name):
    rows, cols = shape
    cell_id_grid = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)
    profile = dict(driver="GTiff", height=rows, width=cols, count=1, dtype="float64", crs=crs, transform=transform)
    gdf = counties.to_crs(crs)
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(cell_id_grid, 1)
        with mem.open() as src:
            res = exact_extract(src, gdf, ["values", "coverage", "center_y"], include_cols=["county_fips"],
                                output="pandas")
    recs = []
    for _, r in res.iterrows():
        ids = np.asarray(r["values"], dtype=np.int64)
        cov = np.asarray(r["coverage"], dtype=np.float64)
        cy = np.asarray(r["center_y"], dtype=np.float64)
        keep = cov > 0
        recs.append(pd.DataFrame(dict(county_fips=r["county_fips"], cell=ids[keep], row=ids[keep] // cols,
                                      col=ids[keep] % cols, coverage=cov[keep],
                                      weight=cov[keep] * np.cos(np.radians(cy[keep])))))
    w = pd.concat(recs, ignore_index=True)
    summ = w.groupby("county_fips").agg(n_cells=("cell", "size"), coverage_sum=("coverage", "sum"),
                                        weight_sum=("weight", "sum")).reset_index()
    missing = sorted(set(counties.county_fips) - set(summ.county_fips))
    if missing:
        raise RuntimeError(f"{name}: counties with no intersecting cells: {missing}")
    w.to_parquet(WDIR / f"{name}_county_weights.parquet", index=False)
    summ.to_csv(WDIR / f"{name}_weights_summary.csv", index=False)
    log.info("%s weights: %d counties, %d county-cell pairs, cells/county min=%d median=%d max=%d",
             name, summ.shape[0], len(w), summ.n_cells.min(), int(summ.n_cells.median()), summ.n_cells.max())
    return w


def nldas():
    import xarray as xr
    e = xr.open_dataset(NLDAS_ELEV)
    lat = e.lat.values.astype(np.float64)
    lon = e.lon.values.astype(np.float64)
    e.close()
    res = 0.125
    assert np.allclose(np.diff(lat), res) and np.allclose(np.diff(lon), res)
    # Grid stored south->north; build a north-up raster whose row 0 = northernmost latitude.
    transform = from_origin(lon.min() - res / 2, lat.max() + res / 2, res, res)
    counties = load_study_counties()
    w = weights_for_grid(transform, (len(lat), len(lon)), "EPSG:4326", counties, "nldas")
    # Convert north-up row index back to the ascending-latitude array index used by NLDAS files.
    w["lat_idx"] = (len(lat) - 1) - w["row"]
    w["lon_idx"] = w["col"]
    w["grid"] = "NLDAS-2 0.125deg study subset (116x196), lat ascending"
    w.to_parquet(WDIR / "nldas_county_weights.parquet", index=False)


def prism():
    z = zipfile.ZipFile(PRISM_RAW / "tmax" / "20000701.zip")
    tif = [n for n in z.namelist() if n.endswith(".tif")][0]
    with MemoryFile(z.read(tif)) as m, m.open() as r:
        transform, shape, crs = r.transform, r.shape, r.crs
        meta = dict(crs=str(crs), shape=shape, transform=tuple(transform), nodata=r.nodata, dtype=r.dtypes[0])
    log.info("PRISM grid: %s", meta)
    counties = load_study_counties()
    w = weights_for_grid(transform, shape, crs, counties, "prism")
    pd.DataFrame([meta]).to_csv(WDIR / "prism_grid_metadata.csv", index=False)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "nldas"
    {"nldas": nldas, "prism": prism}[which]()
