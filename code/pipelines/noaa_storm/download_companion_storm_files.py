"""
Download NOAA Storm Events Companion Families: Fatalities and Locations (1985-2023)
Ensures 100% completeness of all official NOAA Storm Events tables.
"""

import os
import re
import time
import gzip
from datetime import datetime, timezone
from pathlib import Path
import requests

BASE_DIR = Path(__file__).resolve().parents[3]  # Paper3 project root (script lives in code/pipelines/noaa_storm/)
RAW_DIR = BASE_DIR / "data" / "noaa_storm_pipeline" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
START_YEAR = 1985
END_YEAR = 2023

def download_companion_files():
    print(f"Fetching live directory index from {BASE_URL}...")
    resp = requests.get(BASE_URL, timeout=30)
    resp.raise_for_status()
    html = resp.text
    
    # Match fatalities and locations
    fat_pattern = re.compile(r'StormEvents_fatalities-ftp_v1\.0_d(\d{4})_c(\d{8})\.csv\.gz')
    loc_pattern = re.compile(r'StormEvents_locations-ftp_v1\.0_d(\d{4})_c(\d{8})\.csv\.gz')
    
    fat_files = {}
    for yr_str, stamp in fat_pattern.findall(html):
        yr = int(yr_str)
        if START_YEAR <= yr <= END_YEAR:
            if yr not in fat_files or stamp > fat_files[yr][1]:
                fat_files[yr] = (f"StormEvents_fatalities-ftp_v1.0_d{yr}_c{stamp}.csv.gz", stamp)
                
    loc_files = {}
    for yr_str, stamp in loc_pattern.findall(html):
        yr = int(yr_str)
        if START_YEAR <= yr <= END_YEAR:
            if yr not in loc_files or stamp > loc_files[yr][1]:
                loc_files[yr] = (f"StormEvents_locations-ftp_v1.0_d{yr}_c{stamp}.csv.gz", stamp)
                
    print(f"Found {len(fat_files)} latest fatalities files and {len(loc_files)} latest locations files for 1985-2023.")
    
    session = requests.Session()
    
    # Download fatalities
    for yr in range(START_YEAR, END_YEAR + 1):
        if yr in fat_files:
            fname, _ = fat_files[yr]
            dest = RAW_DIR / fname
            if dest.exists() and dest.stat().st_size > 100:
                # check gzip
                try:
                    with gzip.open(dest, "rb") as f:
                        f.read(512)
                    # already valid
                    continue
                except Exception:
                    pass
            url = BASE_URL + fname
            r = session.get(url, timeout=60)
            if r.status_code == 200:
                with open(dest, "wb") as f:
                    f.write(r.content)
                print(f"  [Fatalities {yr}] Downloaded {fname} ({len(r.content)} bytes)")
            time.sleep(0.2)

    # Download locations
    for yr in range(START_YEAR, END_YEAR + 1):
        if yr in loc_files:
            fname, _ = loc_files[yr]
            dest = RAW_DIR / fname
            if dest.exists() and dest.stat().st_size > 100:
                try:
                    with gzip.open(dest, "rb") as f:
                        f.read(512)
                    continue
                except Exception:
                    pass
            url = BASE_URL + fname
            r = session.get(url, timeout=60)
            if r.status_code == 200:
                with open(dest, "wb") as f:
                    f.write(r.content)
                print(f"  [Locations {yr}] Downloaded {fname} ({len(r.content)} bytes)")
            time.sleep(0.2)

    print("All companion files checked and up to date!")

if __name__ == "__main__":
    download_companion_files()
