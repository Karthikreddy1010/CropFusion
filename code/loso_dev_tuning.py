"""Re-derive LOSO hyperparameters on each fold's own DEV rows (audit C7).

The values in `config.py` descend from six rounds selected by watching
leave-one-state-out R² rise — i.e. chosen with held-out information. The history
is disclosed in `docs/supplement/development_history.md`; this module replaces the
values honestly.

For each held-out state:

    fit on that fold's FIT rows  ->  score on that fold's DEV rows  ->  pick the
    lowest DEV RMSE from a grid fixed in advance

The held-out state is never touched. The search grid is declared below and must
not be extended after seeing results; add to it only with a protocol version bump.
The selection is written to `loso_dev_selected_hyperparams.json` so the provenance
of every fold's configuration is a file rather than a memory.

The LOSO estimates are expected to get WORSE, not better. The current macro R² of
0.6976 was produced with held-out-informed settings; an honest re-derivation
removes that advantage. A drop is the correct outcome, not a regression.

Run (full, GPU strongly preferred):
    python code/loso_dev_tuning.py
Run (smoke, one fold and two configurations):
    python code/loso_dev_tuning.py --states Minnesota --grid 2 --epochs 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
for p in (str(CODE_DIR), str(CODE_DIR / "diagnostics_loso")):
    if p not in sys.path:
        sys.path.insert(0, p)

import config as cfg  # noqa: E402

# Pre-specified search grid. Declared before any result is seen; the first entry
# is the incumbent configuration so the comparison always includes it.
GRID: List[Dict[str, Any]] = [
    {"name": "incumbent", "hidden_dims": tuple(cfg.LOSO_HIDDEN_DIMS),
     "lr": cfg.LOSO_LEARNING_RATE, "dropout": cfg.LOSO_DROPOUT,
     "weight_decay": cfg.LOSO_WEIGHT_DECAY, "batch_size": cfg.LOSO_BATCH_SIZE},
    {"name": "small_slow", "hidden_dims": (64, 32), "lr": 3e-4, "dropout": 0.2,
     "weight_decay": 1e-3, "batch_size": 128},
    {"name": "wide_fast", "hidden_dims": (256, 128, 64, 32), "lr": 1e-3, "dropout": 0.25,
     "weight_decay": 1e-4, "batch_size": 64},
    {"name": "mid", "hidden_dims": (128, 64), "lr": 5e-4, "dropout": 0.2,
     "weight_decay": 5e-4, "batch_size": 96},
]


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def tune_fold(state: str, grid: List[Dict[str, Any]], epochs: int,
              seeds: List[int]) -> Dict[str, Any]:
    """Score every grid entry on this fold's DEV rows. Never reads the held-out state."""
    import _fold_lab as lab
    import model_training as mt

    fold = lab.build_loso_fold(state)
    X, y = fold["X"], fold["y"]

    # Three distinct roles, and they must stay distinct:
    #   tr_fit  - training rows
    #   es_val  - a carve-out of FIT used for early stopping, exactly as
    #             _run_loso_cv does it
    #   dev     - the fold's DEV partition (2014-2015 of the five training
    #             states), used ONLY to choose between configurations
    # Selecting on es_val would pick the configuration that stopped most
    # favourably on the very rows that stopped it. The held-out state appears
    # in none of the three.
    X_tr, y_tr = X["tr_fit"], y["tr_fit"]
    X_es, y_es = X["es_val"], y["es_val"]
    X_dev, y_dev = X["dev"], y["dev"]

    results = []
    for entry in grid:
        per_seed = []
        for seed in seeds:
            t0 = time.time()
            model = mt.train_neural_cqr(
                X_tr, y_tr, X_es, y_es, feature_cols=fold.get("feature_cols"),
                epochs=epochs, batch_size=entry["batch_size"], lr=entry["lr"],
                weight_decay=entry["weight_decay"], dropout_rate=entry["dropout"],
                hidden_dims=entry["hidden_dims"],
                early_stopping_mode=cfg.LOSO_EARLY_STOPPING_MODE,
                patience=cfg.LOSO_EARLY_STOPPING_PATIENCE, seed=seed,
            )
            pred, _, _ = mt.predict_intervals(model, X_dev)   # scored on DEV, not on es_val
            per_seed.append({"seed": seed, "dev_rmse": rmse(y_dev, pred),
                             "best_epoch": int(getattr(model, "best_epoch", 0)),
                             "seconds": round(time.time() - t0, 1)})
        mean_rmse = float(np.mean([s["dev_rmse"] for s in per_seed]))
        results.append({**{k: v for k, v in entry.items() if k != "hidden_dims"},
                        "hidden_dims": list(entry["hidden_dims"]),
                        "dev_rmse_mean": mean_rmse,
                        "dev_rmse_sd": float(np.std([s["dev_rmse"] for s in per_seed])),
                        "per_seed": per_seed})
        print(f"    {entry['name']:<12} DEV RMSE {mean_rmse:.4f}", flush=True)

    best = min(results, key=lambda r: r["dev_rmse_mean"])
    print(f"  -> {state}: selected '{best['name']}' (DEV RMSE {best['dev_rmse_mean']:.4f}; "
          f"incumbent {results[0]['dev_rmse_mean']:.4f})", flush=True)
    return {"state": state, "selected": best["name"], "selected_params": {
        k: best[k] for k in ("hidden_dims", "lr", "dropout", "weight_decay", "batch_size")},
        "all_results": results,
        "selection_rule": "lowest mean DEV RMSE on this fold's own DEV rows; "
                          "the held-out state was not used"}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=str, default=",".join(cfg.LOSO_STATES))
    ap.add_argument("--grid", type=int, default=len(GRID), help="use the first N grid entries")
    ap.add_argument("--epochs", type=int, default=cfg.LOSO_MAX_EPOCHS)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    states = [s.strip() for s in args.states.split(",") if s.strip()]
    grid = GRID[: args.grid]
    seeds = [cfg.RANDOM_SEED, 7, 123, 2024, 3407][: args.seeds]
    print(f"LOSO DEV tuning: {len(states)} fold(s) x {len(grid)} configs x {len(seeds)} seed(s), "
          f"epochs={args.epochs}")

    folds = []
    for state in states:
        print(f"\n[{state}]", flush=True)
        folds.append(tune_fold(state, grid, args.epochs, seeds))

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "audit C7 - replace LOSO hyperparameters selected on held-out R2",
        "grid_declared_in": "code/loso_dev_tuning.py GRID (fixed before running)",
        "epochs": args.epochs, "seeds": seeds,
        "folds": folds,
        "per_state_selection": {f["state"]: f["selected_params"] for f in folds},
    }
    out = Path(args.out) if args.out else (Path(cfg.REPORT_DIR) / "loso_dev_selected_hyperparams.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}")
    print("Next: set cfg.LOSO_USE_DEV_SELECTED_HP = True and rerun the LOSO evaluation. "
          "Expect the macro R2 to fall relative to 0.6976 - that is the point.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
