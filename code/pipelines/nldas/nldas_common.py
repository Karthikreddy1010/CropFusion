"""
nldas_common.py

Shared utilities for the NLDAS-2 acquisition pipeline:
- Secure NASA Earthdata Token retrieval (never exposed or logged)
- Config loading & path resolution
- Logging setup
- Date-range generation
- Manifest I/O helpers
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

def get_project_root() -> Path:
    curr = Path(__file__).resolve()
    for parent in [curr] + list(curr.parents):
        if (parent / "Paper3_MegaDataset_SPEI_FINAL.csv").exists() or (parent / ".git").exists():
            return parent
    return curr.parents[3]

def get_earthdata_token(cfg: Optional[dict] = None) -> str:
    """
    Securely retrieves the NASA Earthdata Bearer Token from the environment
    or git-ignored .env file.
    NEVER logs, prints, or exposes the token value.
    """
    env_var = "NASA_EARTHDATA_TOKEN"
    if cfg and "nldas" in cfg and "auth_env_var" in cfg["nldas"]:
        env_var = cfg["nldas"]["auth_env_var"]

    token = os.environ.get(env_var)
    if not token:
        proj_root = get_project_root()
        env_path = proj_root / ".env"
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"{env_var}=") or line.startswith("NASA_EARTHDATA_TOKEN="):
                        token = line.split("=", 1)[1].strip()
                        os.environ[env_var] = token
                        break
    if not token:
        raise ValueError(
            f"NASA Earthdata Bearer Token not found. Please set {env_var} in your "
            "environment or git-ignored .env file."
        )
    return token

def load_config(config_path: str = "data/nldas_pipeline/config/nldas_config.yaml") -> dict:
    proj_root = get_project_root()
    p = Path(config_path)
    if not p.is_absolute():
        candidates = [
            Path.cwd() / p,
            proj_root / p,
            proj_root / "data" / "nldas_pipeline" / "config" / "nldas_config.yaml",
            Path(__file__).resolve().parents[1] / "config" / "nldas_config.yaml",
        ]
        for c in candidates:
            if c.exists():
                p = c
                break
    path = p.resolve()
    if not path.exists():
        raise FileNotFoundError(f"NLDAS config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_resolved_project_root"] = str(proj_root)
    return cfg

def resolve_path(cfg: dict, key_path: str) -> Path:
    parts = key_path.split(".")
    node = cfg
    for p in parts:
        node = node[p]
    p = Path(node)
    if p.is_absolute():
        return p.resolve()
    root = Path(cfg["_resolved_project_root"])
    return (root / p).resolve()

def setup_logger(name: str, cfg: dict, filename: str) -> logging.Logger:
    logs_dir = resolve_path(cfg, "paths.logs_dir")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / filename

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger

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

MANIFEST_FIELDS = [
    "date",
    "status",               # SUCCESS | FAILED | CORRUPT | SKIPPED
    "hours_downloaded",     # e.g., 24
    "hours_expected",       # 24
    "daily_grid_path",
    "file_size_bytes",
    "checksum_sha256",
    "rs_mean_w_m2",
    "u2_mean_m_s",
    "ea_mean_kpa",
    "psurf_mean_pa",
    "tair_mean_c",
    "timestamp_utc",
    "error_message",
]

@dataclass
class NLDASManifestRecord:
    date: str
    status: str
    hours_downloaded: int
    hours_expected: int
    daily_grid_path: str
    file_size_bytes: Optional[int] = None
    checksum_sha256: Optional[str] = None
    rs_mean_w_m2: Optional[float] = None
    u2_mean_m_s: Optional[float] = None
    ea_mean_kpa: Optional[float] = None
    psurf_mean_pa: Optional[float] = None
    tair_mean_c: Optional[float] = None
    timestamp_utc: Optional[str] = None
    error_message: str = ""

    def as_row(self) -> dict:
        return asdict(self)

class NLDASManifestWriter:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        self._fh = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=MANIFEST_FIELDS)
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def write(self, record: NLDASManifestRecord) -> None:
        self._writer.writerow(record.as_row())
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def load_nldas_manifest_dates(csv_path: Path) -> set:
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "SUCCESS" and int(row.get("hours_downloaded", 0)) == 24:
                done.add(row["date"])
    return done
