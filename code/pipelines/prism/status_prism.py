#!/usr/bin/env python3
"""
status_prism.py

Authoritative status reporter for PRISM 1985–2023 acquisition pipeline.
Formats exactly to the specification required by Prompt Section 18.
Dynamically resolves raw_dir from prism_config.yaml.
"""

import shutil
import hashlib
import zipfile
import datetime as dt
from pathlib import Path
import pandas as pd
import yaml

from prism_common import load_config, resolve_path

def get_status():
    cfg = load_config()
    raw_dir = resolve_path(cfg, "paths.raw_dir")
    manifest_path = resolve_path(cfg, "manifest.download_manifest_csv")

    start_d = dt.date(1985, 1, 1)
    end_d = dt.date(2023, 12, 31)
    all_dates = [start_d + dt.timedelta(days=i) for i in range((end_d - start_d).days + 1)]
    expected_dates = len(all_dates)
    expected_total = expected_dates * 3

    vars = ["tmax", "tmin", "ppt"]
    var_stats = {v: {"valid": 0, "missing": 0, "corrupt": 0, "files": set()} for v in vars}
    sizes = []
    total_bytes = 0

    for v in vars:
        vdir = raw_dir / v
        if vdir.exists():
            for z in vdir.glob("*.zip"):
                sz = z.stat().st_size
                sizes.append(sz)
                total_bytes += sz
                try:
                    with zipfile.ZipFile(z) as zf:
                        bad = zf.testzip()
                        names = zf.namelist()
                        if bad is None and len(names) > 0:
                            var_stats[v]["valid"] += 1
                            var_stats[v]["files"].add(z.stem)
                        else:
                            var_stats[v]["corrupt"] += 1
                except Exception:
                    var_stats[v]["corrupt"] += 1

        all_stems = {d.strftime("%Y%m%d") for d in all_dates}
        var_stats[v]["missing"] = len(all_stems - var_stats[v]["files"])

    total_valid = sum(var_stats[v]["valid"] for v in vars)
    remaining_total = expected_total - total_valid
    progress_pct = (total_valid / expected_total) * 100.0

    # Storage calculations on configured raw_dir
    total_d, used_d, free_d = shutil.disk_usage(raw_dir if raw_dir.exists() else Path("."))
    total_gb = total_d / (1024 ** 3)
    used_gb = used_d / (1024 ** 3)
    free_gb = free_d / (1024 ** 3)

    avg_archive_bytes = (sum(sizes) / len(sizes)) if sizes else (1.821 * 1024 * 1024)
    remaining_bytes = remaining_total * avg_archive_bytes
    remaining_gb = remaining_bytes / (1024 ** 3)

    safety_margin_gb = 10.0
    temp_download_req_gb = 1.0
    total_needed_gb = remaining_gb + temp_download_req_gb + safety_margin_gb
    storage_sufficient = "YES" if free_gb >= total_needed_gb else "NO"

    drive_name = raw_dir.drive if raw_dir.drive else str(raw_dir)

    print("PRISM 1985–2023 STATUS")
    print(f"\nTarget Location: {raw_dir} (Drive: {drive_name})")
    print(f"Expected dates:  {expected_dates:,}")
    print()
    for v in ["tmax", "tmin", "ppt"]:
        print(f"{v.upper()}:")
        print(f"  valid:   {var_stats[v]['valid']:,}")
        print(f"  missing: {var_stats[v]['missing']:,}")
        print(f"  corrupt: {var_stats[v]['corrupt']}")
        print()

    print(f"Total valid variable-date combinations: {total_valid:,}")
    print(f"Expected:                               {expected_total:,}")
    print(f"Remaining:                              {remaining_total:,}")
    print()
    print("Disk:")
    print(f"  total: {total_gb:.2f} GB")
    print(f"  used:  {used_gb:.2f} GB")
    print(f"  free:  {free_gb:.2f} GB")
    print()
    print(f"Estimated remaining storage:           {remaining_gb:.2f} GB (compressed)")
    print(f"Total projected with safety margin:    {total_needed_gb:.2f} GB")
    print(f"Storage sufficient:                    {storage_sufficient}")
    print()
    print(f"Download progress %:                   {progress_pct:.3f}%")
    print(f"Estimated remaining download volume:   {remaining_gb:.2f} GB")

if __name__ == "__main__":
    get_status()
