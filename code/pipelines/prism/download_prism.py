#!/usr/bin/env python3
"""
download_prism.py

Resumable, polite downloader for PRISM daily CONUS 4km grids (TMAX, TMIN, PPT).
Includes dynamic storage requirement calculation, atomic writes, manifest source of truth,
conservative rate limiting, and safe pause on insufficient disk capacity.

Usage
-----
    # Verification mode (historical dates):
    python code/pipelines/prism/download_prism.py --config config/prism_config.yaml --mode validation_dates

    # Small manual test on arbitrary dates:
    python code/pipelines/prism/download_prism.py --config config/prism_config.yaml \
        --start-date 2019-07-15 --end-date 2019-07-15 --allow-partial-batch

    # Full 1985-01-01..2023-12-31 acquisition:
    python code/pipelines/prism/download_prism.py --config config/prism_config.yaml --mode full
"""

from __future__ import annotations

import argparse
import hashlib
import time
import zipfile
import shutil
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Optional, Tuple
import os
import sys

import requests

from prism_common import (
    load_config,
    resolve_path,
    setup_logger,
    daterange,
    ManifestRecord,
    ManifestWriter,
    load_manifest_keys,
)


def safe_replace(src: Path, dst: Path):
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


def build_url(cfg: dict, variable: str, d: date) -> str:
    base = cfg["prism"]["base_url"]
    region = cfg["prism"]["region"]
    res = cfg["prism"]["resolution"]
    datestr = d.strftime("%Y%m%d")
    url = f"{base}/{region}/{res}/{variable}/{datestr}"
    if cfg["prism"].get("request_netcdf"):
        url += "?format=nc"
    return url


def local_path_for(cfg: dict, variable: str, d: date) -> Path:
    raw_dir = resolve_path(cfg, "paths.raw_dir")
    ext = "nc" if cfg["prism"].get("request_netcdf") else "zip"
    return raw_dir / variable / f"{d.strftime('%Y%m%d')}.{ext}"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad is not None or len(zf.namelist()) == 0:
                return False
            # Ensure at least one raster file (.tif or .bil) exists inside
            has_raster = any(n.lower().endswith((".tif", ".bil")) for n in zf.namelist())
            return has_raster
    except Exception:
        return False


def is_valid_netcdf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            header = f.read(8)
        return header[:3] == b"CDF" or header[:4] == b"\x89HDF"
    except OSError:
        return False


def calculate_storage_metrics(raw_dir: Path, remaining_count: int, safety_margin_gb: float) -> dict:
    """
    Dynamically computes storage metrics:
      - Current free space
      - Average archive size based on currently downloaded files (or 2.3 MB fallback)
      - Estimated remaining compressed data
      - Temporary download buffer requirement (1.0 GB)
      - Safety margin
      - Total projected required space
      - Storage deficit
    """
    total_b, used_b, free_b = shutil.disk_usage(raw_dir)
    
    # Calculate average archive size from existing archives
    existing_sizes = []
    for var_dir in raw_dir.glob("*"):
        if var_dir.is_dir():
            for f in var_dir.glob("*.zip"):
                existing_sizes.append(f.stat().st_size)

    if existing_sizes:
        avg_archive_bytes = sum(existing_sizes) / len(existing_sizes)
    else:
        avg_archive_bytes = 2.3 * 1024 * 1024  # conservative 2.3 MB fallback

    est_remaining_bytes = int(remaining_count * avg_archive_bytes)
    temp_buffer_bytes = int(1.0 * (1024 ** 3))
    safety_margin_bytes = int(safety_margin_gb * (1024 ** 3))
    total_required_bytes = est_remaining_bytes + temp_buffer_bytes + safety_margin_bytes
    deficit_bytes = max(0, total_required_bytes - free_b)

    drive_letter = raw_dir.drive if raw_dir.drive else str(raw_dir)

    return {
        "drive": drive_letter,
        "total_gb": total_b / (1024 ** 3),
        "used_gb": used_b / (1024 ** 3),
        "free_gb": free_b / (1024 ** 3),
        "free_bytes": free_b,
        "avg_archive_mb": avg_archive_bytes / (1024 ** 2),
        "est_remaining_gb": est_remaining_bytes / (1024 ** 3),
        "temp_buffer_gb": temp_buffer_bytes / (1024 ** 3),
        "safety_margin_gb": safety_margin_gb,
        "total_required_gb": total_required_bytes / (1024 ** 3),
        "total_required_bytes": total_required_bytes,
        "deficit_gb": deficit_bytes / (1024 ** 3),
        "is_sufficient": free_b >= total_required_bytes,
    }


def download_one(
    cfg: dict,
    session: requests.Session,
    logger,
    variable: str,
    d: date,
) -> ManifestRecord:
    url = build_url(cfg, variable, d)
    local_path = local_path_for(cfg, variable, d)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    datestr = d.strftime("%Y-%m-%d")

    net_cfg = cfg["network"]
    max_retries = net_cfg.get("max_retries", 5)
    timeout_s = net_cfg.get("timeout_seconds", 60)
    backoff_factor = net_cfg.get("backoff_factor", 2.0)
    backoff_max = net_cfg.get("backoff_max_seconds", 300)
    min_size = cfg["validation"]["min_file_size_bytes"]
    is_nc = cfg["prism"].get("request_netcdf", False)

    attempt = 0
    last_error = ""
    last_http_status: Optional[int] = None
    t_req_start = datetime.now(timezone.utc).isoformat()

    while attempt <= max_retries:
        t0 = time.time()
        try:
            resp = session.get(
                url,
                timeout=timeout_s,
                headers={"User-Agent": net_cfg.get("user_agent", "prism-downloader")},
                stream=True,
            )
            elapsed = time.time() - t0
            last_http_status = resp.status_code

            logger.info(
                f"[{variable.upper()} {datestr}] req_time={t_req_start} "
                f"attempt={attempt} status={resp.status_code} elapsed={elapsed:.2f}s"
            )

            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}"
                logger.warning(f"[{variable.upper()} {datestr}] HTTP {resp.status_code} on attempt {attempt}")
                raise requests.HTTPError(last_error)

            tmp_path = local_path.with_suffix(local_path.suffix + ".part")
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)

            file_size = tmp_path.stat().st_size
            if file_size < min_size:
                last_error = f"File too small ({file_size} bytes) — likely error page"
                tmp_path.unlink(missing_ok=True)
                raise ValueError(last_error)

            valid = is_valid_netcdf(tmp_path) if is_nc else is_valid_zip(tmp_path)
            if not valid:
                last_error = "Failed integrity check (corrupt archive/raster)"
                safe_replace(tmp_path, local_path)
                checksum = sha256_of(local_path)
                logger.error(f"[{variable.upper()} {datestr}] {last_error}")
                return ManifestRecord(
                    date=datestr, variable=variable, url=url,
                    local_path=str(local_path), status="CORRUPT",
                    http_status=last_http_status, file_size=file_size,
                    checksum_sha256=checksum,
                    download_timestamp=datetime.now(timezone.utc).isoformat(),
                    validation_status="FAIL", error_message=last_error,
                )

            # Atomic rename from .part to final .zip
            safe_replace(tmp_path, local_path)
            checksum = sha256_of(local_path)
            logger.info(f"[{variable.upper()} {datestr}] SUCCESS ({file_size} bytes, sha256={checksum[:12]}...)")
            return ManifestRecord(
                date=datestr, variable=variable, url=url,
                local_path=str(local_path), status="SUCCESS",
                http_status=last_http_status, file_size=file_size,
                checksum_sha256=checksum,
                download_timestamp=datetime.now(timezone.utc).isoformat(),
                validation_status="PASS", error_message="",
            )

        except (requests.RequestException, ValueError) as e:
            last_error = str(e)
            attempt += 1
            # Clean up .part file on failure
            tmp_path = local_path.with_suffix(local_path.suffix + ".part")
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            if attempt > max_retries:
                break
            sleep_s = min(backoff_factor ** attempt, backoff_max)
            logger.warning(
                f"[{variable.upper()} {datestr}] retry {attempt}/{max_retries} "
                f"after error: {last_error} (sleeping {sleep_s:.1f}s)"
            )
            time.sleep(sleep_s)

    logger.error(f"[{variable.upper()} {datestr}] FAILED after {max_retries} retries: {last_error}")
    return ManifestRecord(
        date=datestr, variable=variable, url=url,
        local_path=str(local_path), status="FAILED",
        http_status=last_http_status, file_size=None,
        checksum_sha256=None,
        download_timestamp=datetime.now(timezone.utc).isoformat(),
        validation_status="FAIL", error_message=last_error,
    )


def main():
    parser = argparse.ArgumentParser(description="PRISM daily grid downloader")
    parser.add_argument("--config", default="config/prism_config.yaml")
    parser.add_argument("--mode", choices=["validation_dates", "full", "range"], default="range")
    parser.add_argument("--start-date", help="YYYY-MM-DD (mode=range)")
    parser.add_argument("--end-date", help="YYYY-MM-DD (mode=range)")
    parser.add_argument("--dates", nargs="+", default=None,
                        help="Explicit list of YYYY-MM-DD dates to download")
    parser.add_argument("--variables", nargs="+", default=None,
                        help="Subset of variables, e.g. --variables tmax")
    parser.add_argument("--safety-margin-gb", type=float, default=10.0,
                        help="Safety margin in GB (default: 10.0)")
    parser.add_argument("--allow-partial-batch", action="store_true",
                        help="Allow execution of partial batch if free space covers the batch but not the entire 1985-2023 archive")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned requests and storage calculations without downloading")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("download_prism", cfg, "download_prism.log")

    variables = args.variables or cfg["prism"]["variables"]

    # 1. Determine planned dates (Chronological order)
    if args.dates:
        dates = sorted([datetime.strptime(s.strip(), "%Y-%m-%d").date() for s in args.dates])
    elif args.mode == "validation_dates":
        dates = sorted([datetime.strptime(s, "%Y-%m-%d").date() for s in cfg["validation_dates"]])
    elif args.mode == "full":
        dates = list(daterange(cfg["study_period"]["start_date"], cfg["study_period"]["end_date"]))
    else:
        if not args.start_date or not args.end_date:
            parser.error("--start-date and --end-date are required for mode=range (or provide --dates)")
        dates = list(daterange(args.start_date, args.end_date))

    total_planned = len(dates) * len(variables)
    manifest_path = resolve_path(cfg, "manifest.download_manifest_csv")
    already_done = load_manifest_keys(manifest_path)

    # Calculate pending pairs
    pending_pairs = []
    for d in dates:
        for v in variables:
            key = (d.strftime("%Y-%m-%d"), v)
            if key not in already_done:
                pending_pairs.append((d, v))

    logger.info(
        f"Acquisition Plan: Mode={args.mode} | {len(dates)} dates x {len(variables)} variables "
        f"= {total_planned} planned requests | Already Validated: {len(already_done)} | Pending: {len(pending_pairs)}"
    )

    # 2. Dynamic Storage Calculation
    raw_dir = resolve_path(cfg, "paths.raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    metrics = calculate_storage_metrics(raw_dir, len(pending_pairs), args.safety_margin_gb)

    logger.info("=" * 72)
    logger.info("DYNAMIC STORAGE EVALUATION")
    logger.info("=" * 72)
    logger.info(f"Target Storage Location      : {raw_dir} (Drive: {metrics['drive']})")
    logger.info(f"Total Capacity               : {metrics['total_gb']:.2f} GB")
    logger.info(f"Used Space                   : {metrics['used_gb']:.2f} GB")
    logger.info(f"Free Space                   : {metrics['free_gb']:.2f} GB")
    logger.info(f"Pending Download Archives    : {len(pending_pairs):,} files")
    logger.info(f"Mean Archive Size            : {metrics['avg_archive_mb']:.3f} MB")
    logger.info(f"Estimated Remaining Data     : {metrics['est_remaining_gb']:.2f} GB (compressed)")
    logger.info(f"Temp Buffer Requirement      : {metrics['temp_buffer_gb']:.2f} GB")
    logger.info(f"Configured Safety Margin     : {metrics['safety_margin_gb']:.2f} GB")
    logger.info(f"Total Projected Required     : {metrics['total_required_gb']:.2f} GB")
    logger.info(f"Storage Deficit              : {metrics['deficit_gb']:.2f} GB")
    logger.info(f"Storage Sufficient           : {'YES' if metrics['is_sufficient'] else 'NO'}")
    logger.info("=" * 72)

    if not metrics["is_sufficient"]:
        if not getattr(args, "allow_partial_batch", False):
            err_msg = (
                f"\n{'=' * 72}\n"
                f"INSUFFICIENT STORAGE — DOWNLOAD PAUSED\n"
                f"{'=' * 72}\n"
                f"Available free space on {metrics['drive']} ({metrics['free_gb']:.2f} GB) is insufficient to hold the\n"
                f"projected acquisition requirement ({metrics['total_required_gb']:.2f} GB, including {metrics['safety_margin_gb']:.1f} GB margin).\n"
                f"Additional storage required: {metrics['deficit_gb']:.2f} GB.\n\n"
                f"The downloader has safely paused to prevent disk exhaustion.\n"
                f"To resolve:\n"
                f"  1. Redirect 'paths.raw_dir' in config/prism_config.yaml to an external\n"
                f"     drive with at least {metrics['total_required_gb']:.2f} GB free capacity.\n"
                f"  2. Or pass --allow-partial-batch if downloading an explicitly small test batch.\n"
                f"{'=' * 72}"
            )
            print(err_msg, file=sys.stderr)
            logger.error("INSUFFICIENT STORAGE — DOWNLOAD PAUSED")
            return

        # If --allow-partial-batch is given, ensure free space still covers at least safety margin + temp buffer + this batch
        batch_needed_b = int(len(pending_pairs) * (metrics["avg_archive_mb"] * 1024 * 1024))
        min_required_for_batch_b = batch_needed_b + int(args.safety_margin_gb * (1024 ** 3)) + int(1.0 * (1024 ** 3))
        if metrics["free_bytes"] < min_required_for_batch_b:
            logger.error(
                f"INSUFFICIENT STORAGE — DOWNLOAD PAUSED: Even partial batch requires "
                f"{min_required_for_batch_b / (1024**3):.2f} GB, but only {metrics['free_gb']:.2f} GB is available."
            )
            return

    if args.dry_run:
        logger.info(f"[DRY RUN COMPLETE] {len(pending_pairs)} of {total_planned} requests would be downloaded.")
        return

    # 3. Execution (Conservative: 1 worker default, chronological order)
    session = requests.Session()
    delay_s = cfg["network"].get("request_delay_seconds", 1.0)
    n_success = n_failed = n_corrupt = n_skipped = 0

    with ManifestWriter(manifest_path) as manifest:
        for d in dates:
            for variable in variables:
                key = (d.strftime("%Y-%m-%d"), variable)
                if key in already_done:
                    n_skipped += 1
                    continue

                # Pre-download disk safety check before EVERY single file
                cur_total, cur_used, cur_free = shutil.disk_usage(raw_dir)
                min_safe_b = int(args.safety_margin_gb * (1024 ** 3))
                if cur_free < min_safe_b:
                    logger.error(
                        f"CRITICAL DISK SAFETY THRESHOLD REACHED: Free space ({cur_free / (1024**3):.2f} GB) "
                        f"is below safety margin ({args.safety_margin_gb:.1f} GB). Pausing download safely."
                    )
                    print("\nINSUFFICIENT STORAGE — DOWNLOAD PAUSED", file=sys.stderr)
                    return

                record = download_one(cfg, session, logger, variable, d)
                manifest.write(record)

                if record.status == "SUCCESS":
                    n_success += 1
                    already_done.add(key)
                elif record.status == "CORRUPT":
                    n_corrupt += 1
                else:
                    n_failed += 1

                time.sleep(delay_s)

    logger.info(
        f"ACQUISITION BATCH COMPLETE. planned={total_planned} skipped={n_skipped} "
        f"success={n_success} failed={n_failed} corrupt={n_corrupt}"
    )


if __name__ == "__main__":
    main()
