"""
prism_common.py

Shared utilities for the PRISM acquisition pipeline: config loading,
logging setup, date-range generation, and manifest I/O helpers.

Nothing in this module makes network calls or touches the ML pipeline.
"""

from __future__ import annotations

import csv
import datetime as dt
import logging
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Optional

import yaml


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

def get_project_root() -> Path:
    curr = Path(__file__).resolve()
    for parent in [curr] + list(curr.parents):
        if (parent / "Paper3_MegaDataset_SPEI_FINAL.csv").exists() or (parent / ".git").exists():
            return parent
    return curr.parents[3]


def load_config(config_path: str = "config/prism_config.yaml") -> dict:
    proj_root = get_project_root()
    p = Path(config_path)
    if not p.is_absolute():
        candidates = [
            Path.cwd() / p,
            proj_root / p,
            proj_root / "data" / "prism_pipeline" / p,
            Path(__file__).resolve().parents[1] / p,
        ]
        for c in candidates:
            if c.exists():
                p = c
                break
    path = p.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["_resolved_project_root"] = str(proj_root)
    return cfg


def resolve_path(cfg: dict, key_path: str) -> Path:
    """Resolve a dotted path.paths.key against the resolved project root, e.g. 'paths.raw_dir'."""
    parts = key_path.split(".")
    node = cfg
    for p in parts:
        node = node[p]
    p = Path(node)
    if p.is_absolute():
        return p.resolve()
    root = Path(cfg["_resolved_project_root"])
    return (root / p).resolve()


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def setup_logger(name: str, cfg: dict, filename: str) -> logging.Logger:
    logs_dir = resolve_path(cfg, "paths.logs_dir")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / filename

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# --------------------------------------------------------------------------
# Date range generation (leap-year safe by construction: datetime handles it)
# --------------------------------------------------------------------------

def daterange(start_date: str, end_date: str) -> Iterator[dt.date]:
    start = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    if end < start:
        raise ValueError(f"end_date {end} is before start_date {start}")
    current = start
    one_day = dt.timedelta(days=1)
    while current <= end:
        yield current
        current += one_day


def expected_day_count(year: int) -> int:
    """365 or 366, computed rather than hard-coded, so leap years are never guessed."""
    start = dt.date(year, 1, 1)
    end = dt.date(year, 12, 31)
    return (end - start).days + 1


# --------------------------------------------------------------------------
# Manifest record
# --------------------------------------------------------------------------

MANIFEST_FIELDS = [
    "date",
    "variable",
    "url",
    "local_path",
    "status",            # SUCCESS | FAILED | CORRUPT | SKIPPED | RETRYING
    "http_status",
    "file_size",
    "checksum_sha256",
    "download_timestamp",
    "validation_status",  # PASS | FAIL | NOT_RUN
    "error_message",
]


@dataclass
class ManifestRecord:
    date: str
    variable: str
    url: str
    local_path: str
    status: str
    http_status: Optional[int] = None
    file_size: Optional[int] = None
    checksum_sha256: Optional[str] = None
    download_timestamp: Optional[str] = None
    validation_status: str = "NOT_RUN"
    error_message: str = ""

    def as_row(self) -> dict:
        return asdict(self)


class ManifestWriter:
    """
    Append-only CSV writer. Safe to reopen across resumed runs — writes
    header only if the file does not already exist.
    """

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        self._fh = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=MANIFEST_FIELDS)
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def write(self, record: ManifestRecord) -> None:
        self._writer.writerow(record.as_row())
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def load_manifest_keys(csv_path: Path) -> set:
    """
    Returns the set of (date, variable) pairs already recorded with
    status SUCCESS and validation_status PASS — i.e. safe to skip.
    Anything else (FAILED, CORRUPT, NOT_RUN) is left for retry.
    """
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "SUCCESS" and row.get("validation_status") == "PASS":
                done.add((row["date"], row["variable"]))
    return done
