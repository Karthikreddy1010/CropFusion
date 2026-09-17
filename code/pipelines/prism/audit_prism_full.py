import os
import zipfile
import hashlib
import shutil
from pathlib import Path
import datetime as dt
import pandas as pd
import rasterio

def run_audit():
    start_d = dt.date(1985, 1, 1)
    end_d = dt.date(2023, 12, 31)
    all_dates = [start_d + dt.timedelta(days=i) for i in range((end_d - start_d).days + 1)]
    print(f"Total expected calendar days: {len(all_dates)}")
    print(f"Total expected variable-day archives: {len(all_dates) * 3}")

    leap_days = [d for d in all_dates if d.month == 2 and d.day == 29]
    print(f"Leap days count: {len(leap_days)} -> {[d.isoformat() for d in leap_days]}")

    raw_dir = Path("data/prism_pipeline/raw")
    manifest_path = Path("data/prism_pipeline/manifests/prism_download_manifest.csv")
    variables = ["tmax", "tmin", "ppt"]

    part_files = list(raw_dir.glob("**/*.part"))
    print(f"Leftover .part files: {len(part_files)}")
    for pf in part_files:
        print(f"  PART: {pf}")

    # Inspect manifest
    manifest_records = {}
    if manifest_path.exists():
        m_df = pd.read_csv(manifest_path)
        print(f"\nManifest total rows: {len(m_df)}")
        for idx, row in m_df.iterrows():
            key = (str(row["date"]), str(row["variable"]))
            manifest_records[key] = row
    else:
        print("\nManifest NOT found!")

    # Check files on disk
    file_hashes = {}
    duplicates = []
    sizes = []
    total_bytes = 0
    stats = {v: {"valid": 0, "corrupt": 0, "missing": 0, "files": set()} for v in variables}

    for v in variables:
        vdir = raw_dir / v
        if not vdir.exists():
            continue
        for z in vdir.glob("*.zip"):
            sz = z.stat().st_size
            sizes.append(sz)
            total_bytes += sz
            
            # Compute hash
            h = hashlib.sha256()
            with open(z, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            digest = h.hexdigest()

            # Check duplicate hash within same variable
            if (v, digest) in file_hashes:
                duplicates.append((z, file_hashes[(v, digest)]))
            else:
                file_hashes[(v, digest)] = z

            # Test ZIP integrity
            is_valid = False
            try:
                with zipfile.ZipFile(z) as zf:
                    bad = zf.testzip()
                    names = zf.namelist()
                    if bad is None and len(names) > 0:
                        # Check raster file inside
                        tif_names = [n for n in names if n.lower().endswith((".tif", ".bil"))]
                        if tif_names:
                            is_valid = True
            except Exception as e:
                is_valid = False

            dstr = z.stem # YYYYMMDD
            try:
                f_date = dt.datetime.strptime(dstr, "%Y%m%d").date()
                f_date_str = f_date.isoformat()
            except ValueError:
                f_date_str = dstr

            if is_valid:
                stats[v]["valid"] += 1
                stats[v]["files"].add(f_date_str)
            else:
                stats[v]["corrupt"] += 1

    # Missing counts
    all_date_strs = set(d.isoformat() for d in all_dates)
    for v in variables:
        stats[v]["missing"] = len(all_date_strs - stats[v]["files"])

    print("\nFile and Manifest Audit:")
    for v in variables:
        print(f"  Variable: {v.upper():<5} | Expected: {len(all_dates):>6} | Valid: {stats[v]['valid']:>6} | Missing: {stats[v]['missing']:>6} | Corrupt: {stats[v]['corrupt']:>6}")

    print(f"\nDuplicates detected: {len(duplicates)}")
    if duplicates:
        for dup in duplicates:
            print(f"  DUP: {dup[0]} matches {dup[1]}")

    # Storage calculations
    total_valid = sum(stats[v]["valid"] for v in variables)
    total_missing = sum(stats[v]["missing"] for v in variables)
    total_corrupt = sum(stats[v]["corrupt"] for v in variables)

    print(f"\nTotal Valid Variable-Days: {total_valid} / {len(all_dates) * 3} ({total_valid / (len(all_dates) * 3) * 100:.3f}%)")
    print(f"Total Missing: {total_missing}")
    print(f"Total Corrupt: {total_corrupt}")

    if sizes:
        mean_bytes = sum(sizes) / len(sizes)
        mean_mb = mean_bytes / (1024 ** 2)
        min_mb = min(sizes) / (1024 ** 2)
        max_mb = max(sizes) / (1024 ** 2)
        total_gb = total_bytes / (1024 ** 3)
        print(f"\nStorage Metrics (from {len(sizes)} downloaded archives):")
        print(f"  Total Downloaded Size  : {total_gb:.3f} GB ({total_bytes / (1024**2):.1f} MB)")
        print(f"  Mean Archive Size      : {mean_mb:.3f} MB")
        print(f"  Min Archive Size       : {min_mb:.3f} MB")
        print(f"  Max Archive Size       : {max_mb:.3f} MB")

        remaining_needed_gb = (total_missing * mean_bytes) / (1024 ** 3)
        print(f"  Remaining Data to DL   : {remaining_needed_gb:.2f} GB (compressed archives)")
        
        # Disk check
        total_d, used_d, free_d = shutil.disk_usage("data/prism_pipeline")
        free_gb = free_d / (1024 ** 3)
        print(f"\nCurrent Drive C: Metrics:")
        print(f"  Total Capacity         : {total_d / (1024**3):.2f} GB")
        print(f"  Used Space             : {used_d / (1024**3):.2f} GB")
        print(f"  Free Space             : {free_gb:.2f} GB")
        
        safety_margin_gb = 10.0
        temp_download_req_gb = 1.0  # streaming chunk buffer
        total_required_gb = remaining_needed_gb + temp_download_req_gb + safety_margin_gb
        deficit_gb = total_required_gb - free_gb

        print(f"\nDynamic Storage Evaluation:")
        print(f"  Estimated Remaining Data : {remaining_needed_gb:.2f} GB")
        print(f"  Temp Buffer Requirement  : {temp_download_req_gb:.2f} GB")
        print(f"  Safety Margin            : {safety_margin_gb:.2f} GB")
        print(f"  Total Projected Required : {total_required_gb:.2f} GB")
        print(f"  Current Available Space  : {free_gb:.2f} GB")
        if deficit_gb > 0:
            print(f"  STORAGE STATUS           : INSUFFICIENT STORAGE — DOWNLOAD PAUSED")
            print(f"  Deficit                  : {deficit_gb:.2f} GB additional storage required")
        else:
            print(f"  STORAGE STATUS           : SUFFICIENT")

if __name__ == "__main__":
    run_audit()
