"""
experiment_registry.py - Clean Experiment Registry for Paper 3.

Enforces strict tracking of every experiment:
- experiment_id
- model
- features
- detrending
- training period (fit_period)
- development period (dev_period)
- calibration period (cal_period)
- test period (test_period)
- hyperparameters
- ensemble_weight
- conformal_method
- random_seed
- RMSE, MAE, R2, PICP, ACE, MPIW, Winkler
- role: DEVELOPMENT | CALIBRATION | FINAL_TEST | LOSO

Prevents accidental mixing of evaluation roles.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import config as cfg

logger = logging.getLogger("paper3")


class ExperimentRegistry:
    """Registry to record and persist all pipeline experiments."""

    def __init__(self, registry_dir: Optional[Path] = None) -> None:
        self.registry_dir = registry_dir or (cfg.OUTPUT_DIR / "experiments")
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.registry_dir / "experiment_registry.csv"
        self.json_path = self.registry_dir / "experiment_registry.json"
        self.records: List[Dict[str, Any]] = []
        self._load_existing()

    def _load_existing(self) -> None:
        if self.json_path.exists():
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
            except Exception as e:
                logger.warning("Failed to load existing experiment registry JSON: %s", e)
                self.records = []

    def register_experiment(
        self,
        experiment_id: str,
        model: str,
        role: str,  # 'DEVELOPMENT' | 'CALIBRATION' | 'FINAL_TEST' | 'LOSO'
        features: List[str] | int,
        detrending: str,
        fit_period: str = "1985-2013",
        dev_period: str = "2014-2015",
        cal_period: str = "2016-2018",
        test_period: str = "2019-2023",
        hyperparameters: Optional[Dict[str, Any]] = None,
        ensemble_weight: Optional[float] = None,
        conformal_method: Optional[str] = None,
        random_seed: int = cfg.RANDOM_SEED,
        rmse: Optional[float] = None,
        mae: Optional[float] = None,
        r2: Optional[float] = None,
        picp: Optional[float] = None,
        ace: Optional[float] = None,
        mpiw: Optional[float] = None,
        winkler: Optional[float] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Record an experiment in the registry."""
        valid_roles = ["DEVELOPMENT", "CALIBRATION", "FINAL_TEST", "LOSO"]
        assert role in valid_roles, f"Invalid experiment role '{role}'. Must be one of {valid_roles}"

        rec = {
            "experiment_id": experiment_id,
            "role": role,
            "model": model,
            "feature_count": len(features) if isinstance(features, list) else int(features),
            "detrending": detrending,
            "fit_period": fit_period,
            "dev_period": dev_period,
            "cal_period": cal_period,
            "test_period": test_period,
            "hyperparameters": hyperparameters or {},
            "ensemble_weight": ensemble_weight,
            "conformal_method": conformal_method,
            "random_seed": random_seed,
            "rmse": round(float(rmse), 4) if rmse is not None else None,
            "mae": round(float(mae), 4) if mae is not None else None,
            "r2": round(float(r2), 4) if r2 is not None else None,
            "picp": round(float(picp), 4) if picp is not None else None,
            "ace": round(float(ace), 4) if ace is not None else None,
            "mpiw": round(float(mpiw), 4) if mpiw is not None else None,
            "winkler": round(float(winkler), 4) if winkler is not None else None,
            "notes": notes,
        }

        # Update if exists, else append
        self.records = [r for r in self.records if r.get("experiment_id") != experiment_id]
        self.records.append(rec)
        self.save()
        logger.info("Registered experiment [%s] (%s, %s): R2=%s, RMSE=%s, PICP=%s, Winkler=%s",
                    experiment_id, model, role, rec["r2"], rec["rmse"], rec["picp"], rec["winkler"])
        return rec

    def save(self) -> None:
        """Persist registry to CSV and JSON."""
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

        df = pd.DataFrame(self.records)
        df.to_csv(self.csv_path, index=False)


# Global instance
registry = ExperimentRegistry()
