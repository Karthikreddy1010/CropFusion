"""
storm_common.py

Shared utilities for the NOAA Storm Events pipeline: config loading
(project-root-anchored, same fix as the PRISM pipeline), logging, and
manifest I/O.
"""

from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import yaml


def get_project_root() -> Path:
    # Look for Paper3 marker in parents
    curr = Path(__file__).resolve()
    for parent in [curr] + list(curr.parents):
        if (parent / "Paper3_MegaDataset_SPEI_FINAL.csv").exists() or (parent / ".git").exists():
            return parent
    return curr.parents[3]


def load_config(config_path: str = "config/storm_config.yaml") -> dict:
    # Try as direct path, or relative to current dir, or relative to storm pipeline dir, or project root
    proj_root = get_project_root()
    p = Path(config_path)
    if not p.is_absolute():
        candidates = [
            Path.cwd() / p,
            proj_root / p,
            proj_root / "data" / "noaa_storm_pipeline" / p,
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
    parts = key_path.split(".")
    node = cfg
    for p in parts:
        node = node[p]
    root = Path(cfg["_resolved_project_root"])
    return (root / node).resolve()


def setup_logger(name: str, cfg: dict, filename: str) -> logging.Logger:
    logs_dir = resolve_path(cfg, "paths.logs_dir")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / filename

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


DOWNLOAD_MANIFEST_FIELDS = [
    "year", "url", "local_path", "status", "http_status",
    "file_size", "download_timestamp", "validation_status", "error_message",
]


@dataclass
class DownloadRecord:
    year: int
    url: str
    local_path: str
    status: str
    http_status: Optional[int] = None
    file_size: Optional[int] = None
    download_timestamp: Optional[str] = None
    validation_status: str = "NOT_RUN"
    error_message: str = ""

    def as_row(self) -> dict:
        return asdict(self)


class ManifestWriter:
    def __init__(self, csv_path: Path, fieldnames):
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        self._fh = open(self.csv_path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def write(self, row: dict) -> None:
        self._writer.writerow(row)
        self._fh.flush()

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def load_done_years(csv_path: Path) -> set:
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "SUCCESS" and row.get("validation_status") == "PASS":
                done.add(int(row["year"]))
    return done
