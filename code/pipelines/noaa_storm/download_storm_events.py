#!/usr/bin/env python3
"""
download_storm_events.py

Downloads NOAA/NCEI Storm Events "details" yearly files.

CRITICAL, CONFIRMED-REAL BEHAVIOR: file names are NOT predictable from
the year alone. The real pattern is:

    StormEvents_details-ftp_v1.0_d<YEAR>_c<CREATION_DATE>.csv.gz

where <CREATION_DATE> changes whenever NOAA reprocesses that year (this
was directly observed: several years carry creation dates from 2026
despite being historical data years). Constructing this URL from the
year alone will silently produce a 404 or, worse, a stale hard-coded
guess. This script always re-fetches the live directory index and
regex-matches the current filename before downloading.

Usage
-----
    # Required first test (task Section 26): 4 years only
    python code/pipelines/noaa_storm/download_storm_events.py --config config/storm_config.yaml \
        --years 2010 2015 2019 2023

    # Full 1985-2023 acquisition
    python code/pipelines/noaa_storm/download_storm_events.py --config config/storm_config.yaml --mode full
"""

from __future__ import annotations

import argparse
import gzip
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from storm_common import (
    load_config, resolve_path, setup_logger,
    DownloadRecord, ManifestWriter, load_done_years, DOWNLOAD_MANIFEST_FIELDS,
)


def fetch_directory_listing(cfg: dict, session: requests.Session, logger) -> str:
    url = cfg["noaa"]["base_dir_url"]
    net = cfg["network"]
    resp = session.get(url, timeout=net["timeout_seconds"],
                        headers={"User-Agent": net.get("user_agent", "storm-downloader")})
    resp.raise_for_status()
    logger.info(f"Fetched live directory listing from {url} ({len(resp.text)} bytes)")
    return resp.text


def find_current_filename(directory_html: str, year: int, cfg: dict) -> Optional[str]:
    """
    Regex-match the CURRENT filename for a given year out of the live
    directory listing HTML. Returns None if no match (year not present
    yet, or NOAA changed the naming convention — either way, this must
    be surfaced, not silently guessed around).
    """
    pattern = cfg["noaa"]["details_filename_regex"].format(year=year)
    matches = re.findall(pattern, directory_html)
    if not matches:
        return None
    # If multiple creation-date stamps somehow appear (shouldn't happen
    # for a live index, but be defensive), prefer the most recent.
    full_pattern = re.compile(
        rf'StormEvents_details-ftp_v1\.0_d{year}_c(\d{{8}})\.csv\.gz'
    )
    all_files = full_pattern.findall(directory_html)
    if not all_files:
        return None
    latest_stamp = max(all_files)
    return f"StormEvents_details-ftp_v1.0_d{year}_c{latest_stamp}.csv.gz"


def is_valid_gzip(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as f:
            f.read(1024)  # forces decompression of the first block
        return True
    except (OSError, gzip.BadGzipFile):
        return False
def safe_replace(src: Path, dst: Path):
    import os
    for _ in range(10):
        try:
            if dst.exists():
                try:
                    dst.unlink()
                except PermissionError:
                    time.sleep(0.5)
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(0.5)
    os.replace(src, dst)


def download_year(cfg: dict, session: requests.Session, logger, year: int, filename: str) -> DownloadRecord:
    base_url = cfg["noaa"]["base_dir_url"]
    url = base_url + filename
    raw_dir = resolve_path(cfg, "paths.raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    local_path = raw_dir / filename

    net = cfg["network"]
    max_retries = net["max_retries"]
    backoff_factor = net["backoff_factor"]
    backoff_max = net["backoff_max_seconds"]
    attempt = 0
    last_error = ""
    last_http_status = None

    while attempt <= max_retries:
        try:
            resp = session.get(url, timeout=net["timeout_seconds"],
                                headers={"User-Agent": net.get("user_agent", "storm-downloader")},
                                stream=True)
            last_http_status = resp.status_code
            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}"
                raise requests.HTTPError(last_error)

            tmp_path = local_path.with_suffix(local_path.suffix + ".part")
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)

            file_size = tmp_path.stat().st_size
            if file_size < 100:
                last_error = f"File too small ({file_size} bytes)"
                tmp_path.unlink(missing_ok=True)
                raise ValueError(last_error)

            if not is_valid_gzip(tmp_path):
                safe_replace(tmp_path, local_path)
                logger.error(f"[{year}] downloaded file failed gzip integrity check")
                return DownloadRecord(
                    year=year, url=url, local_path=str(local_path), status="CORRUPT",
                    http_status=last_http_status, file_size=file_size,
                    download_timestamp=datetime.now(timezone.utc).isoformat(),
                    validation_status="FAIL", error_message="Bad gzip",
                )

            safe_replace(tmp_path, local_path)
            logger.info(f"[{year}] SUCCESS -> {filename} ({file_size} bytes)")
            return DownloadRecord(
                year=year, url=url, local_path=str(local_path), status="SUCCESS",
                http_status=last_http_status, file_size=file_size,
                download_timestamp=datetime.now(timezone.utc).isoformat(),
                validation_status="PASS", error_message="",
            )

        except (requests.RequestException, ValueError) as e:
            last_error = str(e)
            attempt += 1
            if attempt > max_retries:
                break
            sleep_s = min(backoff_factor ** attempt, backoff_max)
            logger.warning(f"[{year}] retry {attempt}/{max_retries}: {last_error} (sleep {sleep_s:.1f}s)")
            time.sleep(sleep_s)

    logger.error(f"[{year}] FAILED after {max_retries} retries: {last_error}")
    return DownloadRecord(
        year=year, url=url, local_path=str(local_path), status="FAILED",
        http_status=last_http_status, file_size=None,
        download_timestamp=datetime.now(timezone.utc).isoformat(),
        validation_status="FAIL", error_message=last_error,
    )


def main():
    parser = argparse.ArgumentParser(description="NOAA Storm Events downloader")
    parser.add_argument("--config", default="config/storm_config.yaml")
    parser.add_argument("--mode", choices=["years", "full", "schema_test"], default="years")
    parser.add_argument("--years", nargs="+", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("download_storm_events", cfg, "download_storm_events.log")

    if args.mode == "schema_test":
        years = cfg["schema_test_years"]
    elif args.mode == "full":
        years = list(range(cfg["study_period"]["start_year"], cfg["study_period"]["end_year"] + 1))
    else:
        if not args.years:
            parser.error("--years is required for mode=years")
        years = args.years

    manifest_path = resolve_path(cfg, "manifest.download_manifest_csv")
    done_years = load_done_years(manifest_path)
    logger.info(f"Requested years: {years}. Already SUCCESS+PASS: {sorted(done_years)}")

    session = requests.Session()

    logger.info("Fetching live NOAA directory listing (required — filenames are not predictable)...")
    try:
        directory_html = fetch_directory_listing(cfg, session, logger)
    except requests.RequestException as e:
        logger.error(f"Could not fetch directory listing: {e}. Cannot proceed without it.")
        return

    plan = {}
    for year in years:
        if year in done_years:
            continue
        filename = find_current_filename(directory_html, year, cfg)
        if filename is None:
            logger.error(
                f"[{year}] No matching file found in the live directory listing. "
                "This year may not be published yet, or NOAA's naming convention "
                "changed — check the directory manually before assuming failure."
            )
            plan[year] = None
        else:
            plan[year] = filename
            logger.info(f"[{year}] resolved current filename: {filename}")

    if args.dry_run:
        for year, filename in plan.items():
            logger.info(f"[DRY RUN] would download {filename or '<UNRESOLVED>'} for {year}")
        return

    delay_s = cfg["network"]["request_delay_seconds"]
    n_success = n_failed = n_corrupt = 0

    with ManifestWriter(manifest_path, DOWNLOAD_MANIFEST_FIELDS) as manifest:
        for year, filename in plan.items():
            if filename is None:
                record = DownloadRecord(
                    year=year, url="", local_path="", status="FAILED",
                    download_timestamp=datetime.now(timezone.utc).isoformat(),
                    validation_status="FAIL",
                    error_message="No matching filename in live directory listing",
                )
            else:
                record = download_year(cfg, session, logger, year, filename)

            manifest.write(record.as_row())
            if record.status == "SUCCESS":
                n_success += 1
            elif record.status == "CORRUPT":
                n_corrupt += 1
            else:
                n_failed += 1
            time.sleep(delay_s)

    skipped = len(years) - len(plan)
    logger.info(f"DONE. requested={len(years)} skipped={skipped} success={n_success} failed={n_failed} corrupt={n_corrupt}")


if __name__ == "__main__":
    main()
