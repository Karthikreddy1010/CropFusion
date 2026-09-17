import zipfile
import tempfile
import shutil
from pathlib import Path
import datetime as dt
import rasterio
import numpy as np

def qc_rasters():
    raw_dir = Path("data/prism_pipeline/raw")
    
    # Representative sample dates across all requested periods and leap days
    sample_dates = [
        # 1985-1989
        "1985-01-15", "1985-07-15", "1988-02-29",
        # 1990-1999
        "1992-02-29", "1995-07-15", "1996-02-29",
        # 2000-2009
        "2000-02-29", "2004-02-29", "2005-07-15", "2008-02-29",
        # 2010-2019
        "2012-02-29", "2015-07-15", "2016-02-29", "2019-07-15",
        # 2020-2023
        "2020-02-29", "2023-07-15"
    ]

    temp_dir = Path(tempfile.mkdtemp(prefix="prism_qc_"))
    print(f"=== PRISM RASTER QUALITY CONTROL (QC) AUDIT ===")
    print(f"Sampling {len(sample_dates)} dates across 1985–2023, including all 9 leap days and historical benchmarks.\n")

    qc_records = []

    try:
        for dstr in sample_dates:
            compact = dstr.replace("-", "")
            day_data = {}
            meta_info = {}

            for v in ["tmax", "tmin", "ppt"]:
                zip_path = raw_dir / v / f"{compact}.zip"
                if not zip_path.exists():
                    print(f"WARNING: File missing for {v} on {dstr}")
                    continue

                # Extract
                target_extract = temp_dir / v / compact
                target_extract.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(target_extract)
                    tif_files = [f for f in target_extract.glob("*.tif")]
                    if not tif_files:
                        tif_files = [f for f in target_extract.glob("*.bil")]
                    
                    if not tif_files:
                        print(f"ERROR: No raster found in {zip_path}")
                        continue
                    
                    rpath = tif_files[0]
                    with rasterio.open(rpath) as src:
                        arr = src.read(1)
                        nodata = src.nodata
                        day_data[v] = (arr, nodata)
                        meta_info[v] = {
                            "crs": src.crs.to_string(),
                            "res": src.res,
                            "shape": src.shape,
                            "bounds": src.bounds,
                            "nodata": nodata
                        }

            # Check raster properties consistency
            if len(day_data) == 3:
                m_tmax = meta_info["tmax"]
                m_tmin = meta_info["tmin"]
                m_ppt = meta_info["ppt"]

                crs_ok = (m_tmax["crs"] == m_tmin["crs"] == m_ppt["crs"] == "EPSG:4269")
                shape_ok = (m_tmax["shape"] == m_tmin["shape"] == m_ppt["shape"] == (621, 1405))
                res_ok = np.isclose(m_tmax["res"][0], 0.0416666666) and np.isclose(m_tmax["res"][1], 0.0416666666)

                tmax_arr, tmax_nd = day_data["tmax"]
                tmin_arr, tmin_nd = day_data["tmin"]
                ppt_arr, ppt_nd = day_data["ppt"]

                # Mask nodata
                mask_tmax = (tmax_arr != tmax_nd) & (~np.isnan(tmax_arr))
                mask_tmin = (tmin_arr != tmin_nd) & (~np.isnan(tmin_arr))
                mask_ppt = (ppt_arr != ppt_nd) & (~np.isnan(ppt_arr))

                valid_t_mask = mask_tmax & mask_tmin
                inversions = np.sum(tmin_arr[valid_t_mask] > tmax_arr[valid_t_mask])

                ppt_valid_vals = ppt_arr[mask_ppt]
                negative_ppt = np.sum(ppt_valid_vals < 0.0)

                tmax_range = (float(np.min(tmax_arr[mask_tmax])), float(np.max(tmax_arr[mask_tmax])))
                tmin_range = (float(np.min(tmin_arr[mask_tmin])), float(np.max(tmin_arr[mask_tmin])))
                ppt_range = (float(np.min(ppt_valid_vals)), float(np.max(ppt_valid_vals)))

                qc_records.append({
                    "date": dstr,
                    "crs": m_tmax["crs"],
                    "shape": m_tmax["shape"],
                    "crs_ok": crs_ok,
                    "shape_ok": shape_ok,
                    "res_ok": res_ok,
                    "t_inversions": inversions,
                    "neg_ppt": negative_ppt,
                    "tmax_min_max": tmax_range,
                    "tmin_min_max": tmin_range,
                    "ppt_min_max": ppt_range,
                })

                print(f"Date: {dstr:<10} | CRS: {m_tmax['crs']} | Shape: {m_tmax['shape']} | Res: ({m_tmax['res'][0]:.5f}, {m_tmax['res'][1]:.5f})")
                print(f"  TMAX range : [{tmax_range[0]:6.2f}, {tmax_range[1]:6.2f}] deg C")
                print(f"  TMIN range : [{tmin_range[0]:6.2f}, {tmin_range[1]:6.2f}] deg C")
                print(f"  PPT range  : [{ppt_range[0]:6.2f}, {ppt_range[1]:6.2f}] mm")
                print(f"  QC Checks  : Inversions (Tmin > Tmax): {inversions} | Negative PPT: {negative_ppt}")
                print(f"  Status     : {'PASS' if (crs_ok and shape_ok and res_ok and inversions == 0 and negative_ppt == 0) else 'FAIL'}\n")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    qc_rasters()
