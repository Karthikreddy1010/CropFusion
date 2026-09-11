"""
leakage_provenance.py - Row-identity based leakage / provenance auditing.

Why this module exists
----------------------
The previous audit relied on checks such as ``id(res.q_lo) != id(res.q_hi)``.
Python object identity says nothing about *which observations* influenced a
learned parameter: two arrays can be distinct objects and still be derived from
exactly the same rows, and re-using one array can be entirely harmless. Such
checks could never have detected the LOSO contamination that was actually
present in this pipeline.

This module tracks provenance at the level of real observation identifiers
(``obs_id`` = ``"<GEOID>_<Year>"``, the primary key of the county-year panel)
and asserts set-level disjointness between the observations allowed to
influence each pipeline stage.

Every stage registers the ID set it consumed; the recorded sets are then
checked pairwise and exported as machine-readable JSON.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

import pandas as pd

logger = logging.getLogger("paper3")


def as_id_set(obj: Any) -> Set[str]:
    """Coerce a DataFrame / Series / iterable of ids into a set of string ids."""
    if obj is None:
        return set()
    if isinstance(obj, pd.DataFrame):
        if "obs_id" in obj.columns:
            return set(obj["obs_id"].astype(str))
        if {"GEOID", "Year"}.issubset(obj.columns):
            return set(obj["GEOID"].astype(str) + "_" + obj["Year"].astype(int).astype(str))
        raise KeyError("DataFrame has neither 'obs_id' nor ('GEOID','Year')")
    if isinstance(obj, pd.Series):
        return set(obj.astype(str))
    return {str(x) for x in obj}


@dataclass
class PartitionLedger:
    """Records which observation ids each pipeline stage was allowed to see."""

    experiment: str
    partitions: Dict[str, Set[str]] = field(default_factory=dict)
    stages: Dict[str, str] = field(default_factory=dict)

    def register(self, name: str, obj: Any) -> Set[str]:
        ids = as_id_set(obj)
        self.partitions[name] = ids
        return ids

    def bind_stage(self, stage: str, partition: str) -> None:
        """Declare that ``stage`` (e.g. 'model_fitting') consumed ``partition``."""
        self.stages[stage] = partition

    def overlap(self, a: str, b: str) -> int:
        return len(self.partitions.get(a, set()) & self.partitions.get(b, set()))

    def size(self, a: str) -> int:
        return len(self.partitions.get(a, set()))


# Pairs that must be disjoint for the result to be defensible, in report order.
REQUIRED_DISJOINT: Sequence[tuple] = (
    ("train", "test", "train_test_overlap"),
    ("dev", "test", "validation_test_overlap"),
    ("cal", "test", "calibration_test_overlap"),
    ("train", "dev", "train_validation_overlap"),
    ("train", "cal", "train_calibration_overlap"),
    ("dev", "cal", "validation_calibration_overlap"),
    ("es_internal", "dev", "earlystop_validation_overlap"),
    ("es_internal", "cal", "earlystop_calibration_overlap"),
    ("es_internal", "test", "earlystop_test_overlap"),
)


def audit_ledger(ledger: PartitionLedger, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compute every required pairwise overlap and return a machine-readable record."""
    record: Dict[str, Any] = {"experiment": ledger.experiment}
    record["partition_sizes"] = {k: len(v) for k, v in ledger.partitions.items()}
    record["stage_bindings"] = dict(ledger.stages)

    failures: List[str] = []
    for a, b, label in REQUIRED_DISJOINT:
        if a not in ledger.partitions or b not in ledger.partitions:
            continue
        n = ledger.overlap(a, b)
        record[label] = n
        if n != 0:
            failures.append(f"{label}={n}")

    record["status"] = "PASS" if not failures else "FAIL"
    record["failures"] = failures
    if extra:
        record.update(extra)
    return record


def assert_disjoint(ledger: PartitionLedger) -> Dict[str, Any]:
    """Audit the ledger and raise if any required partition pair overlaps."""
    record = audit_ledger(ledger)
    if record["status"] != "PASS":
        raise AssertionError(
            f"LEAKAGE DETECTED in '{ledger.experiment}': {', '.join(record['failures'])}"
        )
    return record


class LeakageAuditCollector:
    """Accumulates per-experiment audit records and writes the combined report."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def add(self, record: Dict[str, Any]) -> None:
        self.records.append(record)
        summary = ", ".join(
            f"{k}={record[k]}" for _, _, k in REQUIRED_DISJOINT if k in record
        )
        logger.info(
            "LEAKAGE AUDIT [%s]: %s (%s)",
            record.get("experiment"), record.get("status"),
            summary or "no comparable partitions",
        )

    @property
    def all_passed(self) -> bool:
        return bool(self.records) and all(r["status"] == "PASS" for r in self.records)

    def to_markdown(self) -> str:
        header = (
            "| Experiment | Train/Test overlap | Val/Test overlap | Cal/Test overlap "
            "| Train/Val overlap | Status |\n"
            "|---|---:|---:|---:|---:|---|\n"
        )
        rows = ""
        for r in self.records:
            rows += (
                f"| {r['experiment']} | {r.get('train_test_overlap', 'n/a')} "
                f"| {r.get('validation_test_overlap', 'n/a')} "
                f"| {r.get('calibration_test_overlap', 'n/a')} "
                f"| {r.get('train_validation_overlap', 'n/a')} "
                f"| {r['status']} |\n"
            )
        return header + rows

    def export(self, json_name: str = "leakage_provenance_audit.json",
               md_name: str = "leakage_provenance_audit.md") -> Dict[str, Any]:
        from utils import save_report, save_report_markdown

        payload = {
            "audit_type": "row_identity_provenance",
            "identifier": "obs_id = <GEOID>_<Year>",
            "note": (
                "Overlaps are computed on real county-year observation identifiers, "
                "not on Python object identity or NumPy memory addresses."
            ),
            "all_passed": self.all_passed,
            "n_experiments": len(self.records),
            "experiments": self.records,
        }
        save_report(payload, json_name)

        md = (
            "# Leakage & Provenance Audit (row-identity based)\n\n"
            f"**Overall status**: {'PASS' if self.all_passed else 'FAIL'} "
            f"({len(self.records)} experiments audited)\n\n"
            "Overlaps below are intersections of real county-year observation IDs "
            "(`obs_id = <GEOID>_<Year>`). Object identity / memory-address checks are "
            "not used anywhere in this audit.\n\n"
            + self.to_markdown()
            + "\n## Stage bindings per experiment\n\n"
        )
        for r in self.records:
            md += f"### {r['experiment']}\n\n"
            md += f"- Partition sizes: `{json.dumps(r.get('partition_sizes', {}))}`\n"
            for stage, part in r.get("stage_bindings", {}).items():
                md += f"- `{stage}` consumed partition **{part}**\n"
            if r.get("failures"):
                md += f"- **FAILURES**: {', '.join(r['failures'])}\n"
            md += "\n"
        save_report_markdown(md, md_name)
        return payload


COLLECTOR = LeakageAuditCollector()
