"""The frozen protocol (audit §17 C7, spec rule D5).

Every choice that could be steered by a result is declared here, once, before
the final run: split boundaries, the trend form, model hyperparameters, the F0
flag settings, calibrators, regime groups, metrics, statistical tests and seeds.
The dict is hashed and written into each output folder, so a reader can tell
which protocol produced a given set of numbers.

Why this exists: `config.py` carried a documented history of LOSO hyperparameters
chosen by watching leave-one-state-out R² rise (rounds 1–6). That is tuning on
held-out data. The history now lives in `docs/supplement/development_history.md`,
and the values below are the ones the paper stands behind.

Usage:
    from frozen_protocol import PROTOCOL, protocol_hash, verify_against_config, write_protocol
    mismatches = verify_against_config()   # [] when config.py agrees with the protocol
    write_protocol()                       # -> outputs_*/reports/frozen_protocol.json
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List

import config as cfg

logger = logging.getLogger("paper3")

# Bump when any value below changes, and say why in the changelog.
PROTOCOL_VERSION = "1.3.0-2026-09-22"

CHANGELOG: List[Dict[str, str]] = [
    {"version": "1.3.0-2026-09-22",
     "change": "Turn LOSO_USE_DEV_SELECTED_HP ON, resolving audit C7. Each LOSO "
               "fold now uses hyperparameters re-derived on its own DEV rows; "
               "main.ensure_loso_dev_hyperparameters() runs code/loso_dev_tuning.py "
               "when no selection file exists and caches it, so the selection is "
               "fixed across reruns rather than re-searched. This CHANGES the LOSO "
               "results: the previous macro R2 of 0.6976 was produced with settings "
               "chosen by watching held-out state performance, and removing that "
               "advantage is expected to lower it. A fall is the correct outcome. "
               "Wiring verified in code/diagnostics_loso/verify_c7_wiring.py."},
    {"version": "1.2.0-2026-09-18",
     "change": "Declare LOSO_USE_DEV_SELECTED_HP (audit C7). When on, each LOSO "
               "fold reads hyperparameters re-derived on its own DEV rows by "
               "code/loso_dev_tuning.py instead of the config defaults, which were "
               "selected on held-out R2. Off by default; results unchanged until the "
               "tuning run exists."},
    {"version": "1.1.0-2026-09-18",
     "change": "Declare LGBM_DEV_EARLY_STOPPING. train_lgbm_quantile accepted a "
               "validation set and ignored it; the flag makes the behaviour "
               "explicit and defaults to the ignoring behaviour, so results are "
               "unchanged. Hash moves from dde12630b692d07d."},
    {"version": "1.0.0-2026-09-18",
     "change": "First frozen protocol. Captures the state after the audit fixes "
               "C3, C5 and C8. LOSO hyperparameters are marked as pending "
               "re-derivation on DEV; until that lands the LOSO numbers carry "
               "the C7 caveat."},
]

PROTOCOL: Dict[str, Any] = {
    "version": PROTOCOL_VERSION,

    "partitions": {
        "fit_years": cfg.FIT_YEARS,
        "dev_years": cfg.DEV_YEARS,
        "cal_years": cfg.CAL_YEARS,
        "test_years": cfg.TEST_YEARS,
        "loso_states": list(cfg.LOSO_STATES),
        "excluded_states": ["Nebraska"],
        "rule": "TEST years and held-out LOSO states inform no choice of any kind.",
    },

    "target": {
        "column": cfg.PRIMARY_TARGET,
        "detrending": "linear, fitted on FIT rows only; trend added back before every metric",
        "detrend_enabled": bool(cfg.DETREND_TARGET),
        "loso_detrending": "fitted on each fold's own FIT partition only",
    },

    "model": {
        "backbone_decision_rule": "lower DEV RMSE decides; TEST never consulted (spec D1/D4)",
        "neural": {
            "hidden_dims": list(cfg.NEURAL_CQR_HIDDEN_DIMS),
            "dropout": cfg.NEURAL_CQR_DROPOUT,
            "learning_rate": cfg.LEARNING_RATE,
            "weight_decay": cfg.WEIGHT_DECAY,
            "batch_size": cfg.BATCH_SIZE,
            "max_epochs": cfg.BASELINE_MAX_EPOCHS,
            "early_stopping": {"mode": cfg.EARLY_STOPPING_MODE,
                               "patience": cfg.EARLY_STOPPING_PATIENCE,
                               "selection_set": "DEV"},
            "loss_weights": {"pinball": cfg.LAMBDA_PINBALL, "huber": cfg.LAMBDA_HUBER,
                             "crossing": cfg.LAMBDA_CROSSING, "width": cfg.LAMBDA_WIDTH},
        },
        "neural_loso": {
            "hidden_dims": list(cfg.LOSO_HIDDEN_DIMS),
            "dropout": cfg.LOSO_DROPOUT,
            "learning_rate": cfg.LOSO_LEARNING_RATE,
            "weight_decay": cfg.LOSO_WEIGHT_DECAY,
            "batch_size": cfg.LOSO_BATCH_SIZE,
            "max_epochs": cfg.LOSO_MAX_EPOCHS,
            "early_stopping_patience": cfg.LOSO_EARLY_STOPPING_PATIENCE,
            "early_stopping_mode": cfg.LOSO_EARLY_STOPPING_MODE,
            "PENDING": "re-derive on each fold's own DEV rows (C7); the values above "
                       "descend from rounds selected on LOSO R2",
        },
        "lightgbm": dict(cfg.LGBM_PARAMS),
        "lightgbm_dev_early_stopping": cfg.LGBM_DEV_EARLY_STOPPING,
        "loso_use_dev_selected_hp": cfg.LOSO_USE_DEV_SELECTED_HP,
        "f0_flags": {
            "STANDARDIZE_TARGET": cfg.STANDARDIZE_TARGET,
            "HUBER_DELTA": cfg.HUBER_DELTA,
            "HUBER_DELTA_SIGMA": cfg.HUBER_DELTA_SIGMA,
            "USE_EMA_WEIGHTS": cfg.USE_EMA_WEIGHTS,
            "LR_ETA_MIN": cfg.LR_ETA_MIN,
            "LR_SCHEDULE_MATCH_DEV": cfg.LR_SCHEDULE_MATCH_DEV,
            "NEURAL_MONOTONE_HEADS": cfg.NEURAL_MONOTONE_HEADS,
        },
    },

    "calibration": {
        "nominal_coverage": cfg.NOMINAL_COVERAGE,
        "headline_adaptive_method": "standard_aci",
        "methods": ["static_conformal", "rolling_static_cp", "group_conditional_cp",
                    "standard_aci", "severity_aware_aci", "weighted_conformal",
                    "locally_adaptive", "phenology_stratified_cqr"],
        "sa_aci": {
            "severity_weighting": cfg.ACI_SEVERITY_WEIGHTING,
            "severity_lambda": cfg.ACI_SEVERITY_LAMBDA,
            "direction": "severity widens; scores normalised by the same weight",
            "note": "the published version divided the threshold, narrowing intervals "
                    "under stress (C5)",
        },
        "alpha_selection": "CAL only; never tuned on TEST",
    },

    "regime_groups": {
        "definition": "CDHW exposure class from CDHW_Event_Count",
        "thresholds": {"Normal": "0 events",
                       "Moderate": f">= {cfg.MODERATE_EVENT_COUNT_THRESHOLD} events",
                       "Extreme": f">= {cfg.EXTREME_EVENT_COUNT_THRESHOLD} events"},
        "pre_specified": True,
        "used_by": "group_conditional_cp and every conditional-coverage table",
    },

    "evaluation": {
        "primary_split": "temporal TEST 2019-2023 (locked)",
        "spatial": "six-state leave-one-state-out",
        "replication": "rolling origin, test years 2008-2023 (C2)",
        "metrics": ["rmse", "mae", "r2", "bias", "picp", "ace", "mpiw", "winkler"],
        "conditional_axes": ["CDHW exposure class", "year", "state"],
    },

    "inference": {
        "unit_of_independence": "year",
        "tests": ["paired t on year-level means", "year-block cluster bootstrap",
                  "Friedman on year blocks"],
        "multiple_comparison": "Holm-Bonferroni and Benjamini-Hochberg on year-level p-values",
        "row_level_tests": "diagnostic only; anti-conservative under panel dependence (C3)",
    },

    "seeds": {
        "primary": cfg.RANDOM_SEED,
        "robustness": list(cfg.ROBUSTNESS_SEEDS),
        "loso_robustness": list(cfg.LOSO_ROBUSTNESS_SEEDS),
    },
}


def protocol_hash() -> str:
    """Stable SHA-256 over the protocol, independent of key order."""
    blob = json.dumps(PROTOCOL, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def verify_against_config() -> List[str]:
    """Return the places where the live config disagrees with the protocol.

    Catches the failure this module exists to prevent: someone edits config.py
    between runs and the declared protocol quietly stops describing the code.
    """
    checks = [
        ("FIT_YEARS", tuple(cfg.FIT_YEARS), tuple(PROTOCOL["partitions"]["fit_years"])),
        ("DEV_YEARS", tuple(cfg.DEV_YEARS), tuple(PROTOCOL["partitions"]["dev_years"])),
        ("CAL_YEARS", tuple(cfg.CAL_YEARS), tuple(PROTOCOL["partitions"]["cal_years"])),
        ("TEST_YEARS", tuple(cfg.TEST_YEARS), tuple(PROTOCOL["partitions"]["test_years"])),
        ("LOSO_STATES", sorted(cfg.LOSO_STATES), sorted(PROTOCOL["partitions"]["loso_states"])),
        ("DETREND_TARGET", bool(cfg.DETREND_TARGET), PROTOCOL["target"]["detrend_enabled"]),
        ("NOMINAL_COVERAGE", cfg.NOMINAL_COVERAGE, PROTOCOL["calibration"]["nominal_coverage"]),
        ("RANDOM_SEED", cfg.RANDOM_SEED, PROTOCOL["seeds"]["primary"]),
    ]
    for flag, declared in PROTOCOL["model"]["f0_flags"].items():
        checks.append((flag, getattr(cfg, flag), declared))
    checks.append(("LGBM_DEV_EARLY_STOPPING", cfg.LGBM_DEV_EARLY_STOPPING,
                   PROTOCOL["model"]["lightgbm_dev_early_stopping"]))
    checks.append(("LOSO_USE_DEV_SELECTED_HP", cfg.LOSO_USE_DEV_SELECTED_HP,
                   PROTOCOL["model"]["loso_use_dev_selected_hp"]))

    return [f"{name}: config={live!r} protocol={declared!r}"
            for name, live, declared in checks if live != declared]


def write_protocol(filename: str = "frozen_protocol.json") -> Dict[str, Any]:
    """Write the protocol, its hash and any config drift into the output folder."""
    from utils import save_report

    mismatches = verify_against_config()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash(),
        "config_matches_protocol": not mismatches,
        "mismatches": mismatches,
        "changelog": CHANGELOG,
        "protocol": PROTOCOL,
    }
    save_report(payload, filename)
    if mismatches:
        logger.warning("FROZEN PROTOCOL: config.py disagrees with the declared protocol in "
                       "%d place(s): %s", len(mismatches), "; ".join(mismatches))
    else:
        logger.info("Frozen protocol %s (hash %s) written; config matches.",
                    PROTOCOL_VERSION, payload["protocol_hash"])
    return payload


if __name__ == "__main__":
    import sys
    bad = verify_against_config()
    print(f"protocol {PROTOCOL_VERSION}  hash {protocol_hash()}")
    if bad:
        print(f"{len(bad)} mismatch(es) against config.py:")
        for m in bad:
            print("  -", m)
    else:
        print("config.py matches the frozen protocol.")
    sys.exit(1 if bad else 0)
