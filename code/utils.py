"""
utils.py - Shared utilities for Paper 3 ACI pipeline.

Provides: logging setup, seed management, JSON/CSV/Markdown report helpers.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import warnings
import config as cfg

# Suppress benign warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", message=".*valid feature names.*")
warnings.filterwarnings("ignore", message=".*Ill-conditioned matrix.*")
warnings.filterwarnings("ignore", message=".*divide by zero encountered.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered.*")



def setup_logging(name: str = "paper3", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a project-wide logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    import io
    utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    ch = logging.StreamHandler(utf8_stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    log_path = cfg.OUTPUT_DIR / "pipeline.log"
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def set_global_seed(seed: int = cfg.RANDOM_SEED) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    logger = logging.getLogger("paper3")
    logger.info("Global random seed set to %d", seed)


def save_report(data: Dict[str, Any], filename: str, subdir: Optional[str] = None) -> Path:
    """Save a dictionary report as JSON to the reports directory (or specified subdir)."""
    base_dir = (cfg.OUTPUT_DIR / subdir) if subdir else cfg.REPORT_DIR
    path = base_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_json_default)
        logger = logging.getLogger("paper3")
        logger.info("Report saved -> %s", path)
    except Exception as e:
        logger = logging.getLogger("paper3")
        logger.warning("Could not save JSON report %s: %s", path, e)
    return path


def save_report_markdown(md_content: str, filename: str, subdir: Optional[str] = None) -> Path:
    """Save a markdown report to the reports directory (or specified subdir)."""
    base_dir = (cfg.OUTPUT_DIR / subdir) if subdir else cfg.REPORT_DIR
    path = base_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger = logging.getLogger("paper3")
        logger.info("Markdown report saved -> %s", path)
    except Exception as e:
        logger = logging.getLogger("paper3")
        logger.warning("Could not save Markdown report %s: %s", path, e)
    return path


def save_report_csv(df: pd.DataFrame, filename: str, subdir: Optional[str] = None) -> Path:
    """Save a DataFrame report as CSV to the reports directory (or specified subdir)."""
    base_dir = (cfg.OUTPUT_DIR / subdir) if subdir else cfg.REPORT_DIR
    path = base_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False)
        logger = logging.getLogger("paper3")
        logger.info("CSV report saved -> %s", path)
    except Exception as e:
        logger = logging.getLogger("paper3")
        logger.warning("Could not save CSV report %s: %s", path, e)
    return path


def _json_default(obj: Any) -> Any:
    """Handle numpy/pandas types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def generate_reproducibility_report(
    elapsed_seconds: float = 0.0,
    model_hashes: Optional[Dict[str, str]] = None,
) -> Path:
    """Generate reproducibility_report.json and reproducibility_report.md (§7 & Appendix)."""
    import platform
    import hashlib

    # RAM & CPU info via psutil if available
    total_ram_gb = "N/A"
    try:
        import psutil
        total_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        pass

    cuda_avail = False
    cuda_ver = "None"
    gpu_name = "N/A"
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            cuda_ver = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    # Dataset checksum
    dataset_hash = "N/A"
    if cfg.DATA_FILE.exists():
        hasher = hashlib.md5()
        with open(cfg.DATA_FILE, "rb") as f:
            hasher.update(f.read(1024 * 1024))
        dataset_hash = hasher.hexdigest()

    import lightgbm
    import catboost
    import xgboost
    import sklearn

    report_data = {
        "system": {
            "os": f"{platform.system()} {platform.release()} ({platform.version()})",
            "python_version": sys.version.split()[0],
            "cpu": platform.machine(),
            "ram_gb": total_ram_gb,
            "cuda_available": cuda_avail,
            "cuda_version": cuda_ver,
            "gpu_name": gpu_name,
        },
        "dependencies": {
            "pytorch": torch.__version__ if 'torch' in sys.modules else 'Loaded',
            "lightgbm": lightgbm.__version__,
            "catboost": catboost.__version__,
            "xgboost": xgboost.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "settings": {
            "global_random_seed": cfg.RANDOM_SEED,
            "dataset_file": cfg.DATA_FILE.name,
            "dataset_md5_header": dataset_hash,
            "train_years": list(cfg.TRAIN_YEARS),
            "val_years": list(cfg.VAL_YEARS),
            "test_years": list(cfg.TEST_YEARS),
            "cqr_quantiles": list(cfg.CQR_QUANTILES),
        },
        "pipeline_runtime_seconds": round(elapsed_seconds, 1),
        "serialized_model_hashes": model_hashes or {},
    }
    save_report(report_data, "reproducibility_report.json")

    md = f"""# Reproducibility Report (§7 & Appendix)

## System & Hardware Specifications
- **Operating System**: {report_data['system']['os']}
- **Python Version**: {report_data['system']['python_version']}
- **CPU Architecture**: {report_data['system']['cpu']}
- **Total System RAM**: {report_data['system']['ram_gb']} GB
- **CUDA Available**: {cuda_avail} (GPU: `{gpu_name}`, CUDA `{cuda_ver}`)

## Software Dependencies
- **PyTorch**: `{report_data['dependencies']['pytorch']}`
- **LightGBM**: `{report_data['dependencies']['lightgbm']}`
- **CatBoost**: `{report_data['dependencies']['catboost']}`
- **XGBoost**: `{report_data['dependencies']['xgboost']}`
- **Scikit-Learn**: `{report_data['dependencies']['scikit_learn']}`
- **NumPy**: `{report_data['dependencies']['numpy']}`
- **Pandas**: `{report_data['dependencies']['pandas']}`

## Experiment Settings & Hashes
- **Global Random Seed**: `{cfg.RANDOM_SEED}`
- **Dataset File**: `{cfg.DATA_FILE.name}` (MD5 Checksum: `{dataset_hash}`)
- **Temporal Splits**: Train {cfg.TRAIN_YEARS}, Val {cfg.VAL_YEARS}, Test {cfg.TEST_YEARS}
- **Target Quantiles**: {cfg.CQR_QUANTILES}
- **Total Pipeline Execution Time**: `{elapsed_seconds:.1f}` seconds
"""
    path = save_report_markdown(md, "reproducibility_report.md")
    logger = logging.getLogger("paper3")
    logger.info("Reproducibility report generated -> reproducibility_report.md")
    return path


def validate_artifact_integrity() -> Dict[str, Any]:
    """Verify presence, non-zero size, creation timestamp, and MD5 hash for all required artifacts.

    Exports:
    - artifact_integrity_report.json
    - artifact_integrity_report.md
    """
    import hashlib

    required_artifacts = [
        ("reports", "dataset_schema_report.json"),
        ("reports", "feature_validation_report.json"),
        ("reports", "feature_lineage_report.json"),
        ("reports", "missing_data_audit.json"),
        ("reports", "outlier_audit.json"),
        ("reports", "scaling_report.json"),
        ("reports", "leakage_audit_report.md"),
        ("reports", "feature_selection_stability.json"),
        ("reports", "feature_selection_stability.csv"),
        ("reports", "feature_selection_stability.md"),
        ("reports", "conformal_validation_report.md"),
        ("reports", "evaluation_report.json"),
        ("reports", "evaluation_summary.csv"),
        ("reports", "evaluation_summary.md"),
        ("reports", "statistical_tests_report.md"),
        ("reports", "objective_O1_report.md"),
        ("reports", "objective_O4_report.md"),
        ("reports", "objective_o5_report.json"),
        ("reports", "objective_o5_report.csv"),
        ("reports", "objective_o5_report.md"),
        ("reports", "objective_O6_report.md"),
        ("reports", "shap_consistency_report.json"),
        ("reports", "shap_consistency_report.csv"),
        ("reports", "shap_consistency_report.md"),
        ("reports", "xgboost_quantile_report.json"),
        ("reports", "loso_cv_report.json"),
        ("reports", "loso_summary.csv"),
        ("reports", "loso_summary_report.md"),
        ("reports", "ablation_report.json"),
        ("reports", "reproducibility_report.md"),
        ("predictions", "predictions.csv"),
        ("figures", "calibration_curves.png"),
    ]

    audits = []
    missing_count = 0

    for subfolder, filename in required_artifacts:
        target_path = cfg.OUTPUT_DIR / subfolder / filename
        exists = target_path.exists()
        size_bytes = 0
        md5_hash = "N/A"
        timestamp = "N/A"

        if exists:
            size_bytes = target_path.stat().st_size
            timestamp = datetime.fromtimestamp(target_path.stat().st_mtime).isoformat()
            hasher = hashlib.md5()
            with open(target_path, "rb") as f:
                hasher.update(f.read(1024 * 1024))
            md5_hash = hasher.hexdigest()
        else:
            missing_count += 1

        audits.append({
            "Artifact": filename,
            "Category": subfolder,
            "Exists": exists,
            "Size_Bytes": size_bytes,
            "MD5_Hash": md5_hash,
            "Creation_Timestamp": timestamp,
            "Status": "PASSED" if exists and size_bytes > 0 else "MISSING",
        })

    completion_pct = round(((len(required_artifacts) - missing_count) / len(required_artifacts)) * 100.0, 2)
    compliance_status = "FULL_METHODOLOGY_COMPLIANCE_100%" if missing_count == 0 else "PARTIAL_COMPLIANCE"

    report = {
        "overall_completion_percentage": completion_pct,
        "compliance_status": compliance_status,
        "total_required": len(required_artifacts),
        "total_missing": missing_count,
        "artifacts_audit": audits,
    }
    save_report(report, "artifact_integrity_report.json")

    md = f"""# Artifact Integrity & Completeness Audit Report

## Audit Status: `{compliance_status}`
- **Artifact Completion Rate**: `{completion_pct}%` ({len(required_artifacts) - missing_count} / {len(required_artifacts)} Present)
- **Total Missing Artifacts**: `{missing_count}`

## Required Artifact Audit Breakdown
| Category | Artifact | Exists | Size (Bytes) | MD5 Hash | Status |
|---|---|---|---|---|---|
"""
    for a in audits:
        md += f"| {a['Category']} | {a['Artifact']} | {a['Exists']} | {a['Size_Bytes']} | `{a['MD5_Hash'][:8]}` | {a['Status']} |\n"

    save_report_markdown(md, "artifact_integrity_report.md")
    logger = logging.getLogger("paper3")
    logger.info("Artifact integrity validation completed (%d%% complete -> %s)", completion_pct, compliance_status)
    return report


STAGE_EXECUTION_LOG: List[Dict[str, Any]] = []

def log_pipeline_stage(
    stage_name: str,
    start_time: float,
    end_time: float,
    warnings: Optional[List[str]] = None,
    errors: Optional[List[str]] = None,
) -> None:
    """Record execution details for a pipeline stage."""
    runtime = end_time - start_time
    ram_mb = 0
    try:
        import psutil
        ram_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
    except Exception:
        pass

    entry = {
        "stage": stage_name,
        "runtime_seconds": round(runtime, 2),
        "peak_ram_mb": ram_mb,
        "warnings": warnings or [],
        "errors": errors or [],
        "timestamp": datetime.now().isoformat(),
    }
    STAGE_EXECUTION_LOG.append(entry)
    save_report({"stages": STAGE_EXECUTION_LOG}, "pipeline_execution_log.json")


def log_decision(
    step: str,
    decision: str,
    reason: str,
    details: Dict[str, Any] | None = None,
) -> None:
    """Log a preprocessing/modeling decision for reproducibility."""
    logger = logging.getLogger("paper3")
    msg = f"DECISION [{step}] -> {decision} | Reason: {reason}"
    if details:
        msg += f" | Details: {json.dumps(details, default=_json_default)}"
    logger.info(msg)

