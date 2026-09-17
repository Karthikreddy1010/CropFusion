"""
USDA NASS Quick Stats Extraction Script
Authoritative agricultural data extraction for Corn (1985-2023)
States: IL, IN, IA, MN, MO, OH
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import requests
import pandas as pd
from dotenv import load_dotenv, find_dotenv

# 1. Setup Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "usda_nass"
RAW_DIR = DATA_DIR / "raw"
LOGS_DIR = DATA_DIR / "logs"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

for d in [RAW_DIR, LOGS_DIR, PROCESSED_DIR, METADATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 2. Setup Logging
log_file = LOGS_DIR / "extraction.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("usda_nass_extract")

# 3. Secure API Key Acquisition
load_dotenv(find_dotenv(usecwd=True))
API_KEY = os.environ.get("USDA_NASS_API_KEY")
if not API_KEY:
    # Try direct .env path in workspace
    load_dotenv(BASE_DIR / ".env")
    API_KEY = os.environ.get("USDA_NASS_API_KEY")

if not API_KEY:
    logger.error("USDA_NASS_API_KEY environment variable not set. Aborting.")
    sys.exit(1)

API_BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"

# 4. Target Configurations
TARGET_STATES = ["IL", "IN", "IA", "MN", "MO", "OH"]
START_YEAR = 1985
END_YEAR = 2023
YEAR_CHUNK_SIZE = 5

# Generate Year Chunks: 1985-1989, 1990-1994, ..., 2020-2023
YEAR_CHUNKS = []
curr_start = START_YEAR
while curr_start <= END_YEAR:
    curr_end = min(curr_start + YEAR_CHUNK_SIZE - 1, END_YEAR)
    YEAR_CHUNKS.append((curr_start, curr_end))
    curr_start = curr_end + 1

logger.info(f"Target States: {TARGET_STATES}")
logger.info(f"Year Chunks: {YEAR_CHUNKS}")

# 5. Extraction Routine with Backoff & Retries
def fetch_batch_with_retry(params, max_retries=5, initial_backoff=2):
    """
    Fetch data from USDA NASS api_GET with exponential backoff and retry.
    Never logs or exposes the API key.
    """
    backoff = initial_backoff
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            resp = requests.get(API_BASE_URL, params=params, timeout=60)
            elapsed = time.time() - t0
            
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                return resp.status_code, data, elapsed, None
            elif resp.status_code == 400:
                # Often returned if 0 records match or bad parameter
                # Check message
                err_text = resp.text[:200]
                return resp.status_code, [], elapsed, err_text
            else:
                logger.warning(f"HTTP {resp.status_code} on attempt {attempt}/{max_retries}. Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request exception on attempt {attempt}/{max_retries}: {e}. Retrying in {backoff}s...")
            time.sleep(backoff)
            backoff *= 2

    return 500, [], 0, "Max retries exceeded"

# 6. Main Extraction Loop
def run_extraction():
    manifest_records = []
    manifest_file = LOGS_DIR / "query_manifest.csv"
    
    total_chunks = len(TARGET_STATES) * len(YEAR_CHUNKS)
    chunk_counter = 0

    logger.info(f"Starting extraction for {total_chunks} batches...")

    for state in TARGET_STATES:
        for y_start, y_end in YEAR_CHUNKS:
            chunk_counter += 1
            query_id = f"Q_{state}_{y_start}_{y_end}"
            raw_filename = f"raw_{state}_{y_start}_{y_end}.json"
            raw_filepath = RAW_DIR / raw_filename
            
            logger.info(f"[{chunk_counter}/{total_chunks}] Requesting {query_id} (State: {state}, Years: {y_start}-{y_end})...")

            # Parameters without exposing key in logs
            query_params = {
                "key": API_KEY,
                "commodity_desc": "CORN",
                "state_alpha": state,
                "year__GE": str(y_start),
                "year__LE": str(y_end),
                "agg_level_desc": "COUNTY",
                "source_desc": "SURVEY",
                "freq_desc": "ANNUAL"
            }

            # Reproducible params dict for manifest (excluding API key)
            logged_params = {k: v for k, v in query_params.items() if k != "key"}

            status_code, records, elapsed, err = fetch_batch_with_retry(query_params)
            
            num_records = len(records)
            timestamp = datetime.utcnow().isoformat() + "Z"
            
            if status_code == 200:
                # Save raw JSON
                with open(raw_filepath, "w", encoding="utf-8") as f:
                    json.dump(records, f)
                logger.info(f"  -> Success: {num_records} records in {elapsed:.2f}s saved to {raw_filename}")
                batch_status = "SUCCESS"
            else:
                logger.error(f"  -> Failed: Status {status_code}, Error: {err}")
                batch_status = f"FAILED: {status_code}"

            manifest_records.append({
                "query_id": query_id,
                "state": state,
                "year_start": y_start,
                "year_end": y_end,
                "parameters": json.dumps(logged_params),
                "timestamp": timestamp,
                "http_status": status_code,
                "record_count": num_records,
                "elapsed_seconds": round(elapsed, 2),
                "status": batch_status,
                "output_file": raw_filename
            })

            # Polite pacing between API calls
            time.sleep(0.5)

    # Save manifest
    df_manifest = pd.DataFrame(manifest_records)
    df_manifest.to_csv(manifest_file, index=False)
    logger.info(f"Extraction complete. Query manifest written to {manifest_file}")
    logger.info(f"Total batches: {len(df_manifest)}, Total records fetched: {df_manifest['record_count'].sum()}")

if __name__ == "__main__":
    run_extraction()
