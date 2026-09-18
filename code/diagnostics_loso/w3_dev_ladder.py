"""W3 DEV ladder — evaluate each F0/F1 change on DEV only.

Decision rules (spec §2): a change is accepted only if it improves the DEV
metric averaged over seeds (D2); one variable changes at a time (D3); TEST years
and held-out LOSO states are never touched here (D1).

Variants
--------
E0  baseline           current config, unchanged
E1  F0a                target standardised on FIT, huber delta = 1.345 sigma
E2  F0b                EMA weights in the no-validation refit
E2b F0b control        same refit, last-epoch weights
E3  F0c                lr = 1e-3 with the wider LOSO net
E3b F0c                lr = 3e-4 with the current net
E4  F1                 monotone quantile heads
E5  F0d                lambda_width = 0
LG0 LightGBM default   cfg.LGBM_PARAMS, 1000 trees, no early stopping
LG1 LightGBM DEV-tuned n_estimators by DEV early stopping (fairness, spec §5.2)

E2/E2b are scored differently from the others on purpose: EMA only affects the
no-validation refit path, so those variants train on FIT for a fixed epoch budget
without early stopping and are then scored on DEV.

**Caveat that matters for D1.** Every neural variant picks its epoch by
minimising DEV RMSE, so its DEV number is optimistic in a way LightGBM's is not.
Use this ladder to rank neural variants against each other. Settling the backbone
choice needs a comparison where neither model selects on the scoring set.

Run (smoke):    python code/diagnostics_loso/w3_dev_ladder.py --seeds 1 --epochs 10
Run (serial):   python code/diagnostics_loso/w3_dev_ladder.py --seeds 5
Run (parallel): python code/diagnostics_loso/w3_dev_ladder.py --seeds 5 --parallel 5 --threads 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import _fold_lab  # noqa: E402  (redirects cfg output dirs to outputs_diagnostics)
import config as cfg  # noqa: E402
import model_training as mt  # noqa: E402

OUT_DIR = Path(_fold_lab.__file__).resolve().parents[2] / "outputs_diagnostics" / "reports"


# ─────────────────────────────────────────────────────────────
# Metrics — always reported on the real yield scale (trend added back)
# ─────────────────────────────────────────────────────────────

def dev_metrics(y_true: np.ndarray, pred: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Dict[str, float]:
    err = y_true - pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - ss_res / max(1e-9, ss_tot)),
        "bias": float(np.mean(err)),
        "picp": float(np.mean((y_true >= lo) & (y_true <= hi))),
        "mpiw": float(np.mean(hi - lo)),
        "crossing_rate": float(np.mean(hi < lo)),
    }


class FlagScope:
    """Temporarily set config flags, restoring them afterwards."""

    def __init__(self, **flags: Any) -> None:
        self.flags = flags
        self._saved: Dict[str, Any] = {}

    def __enter__(self) -> "FlagScope":
        for k, v in self.flags.items():
            self._saved[k] = getattr(cfg, k)
            setattr(cfg, k, v)
        return self

    def __exit__(self, *exc: Any) -> None:
        for k, v in self._saved.items():
            setattr(cfg, k, v)


# ─────────────────────────────────────────────────────────────
# Runners
# ─────────────────────────────────────────────────────────────

def run_neural(data: Dict[str, Any], seed: int, epochs: int, *, refit_style: bool = False,
               lr: float | None = None, hidden: tuple | None = None,
               lambda_width: float | None = None, **flags: Any) -> Dict[str, float]:
    """Train on FIT, score on DEV. `refit_style` trains without validation."""
    X_fit, X_dev = data["X"]["fit"], data["X"]["dev"]
    y_fit = data["y_detrended"]["fit"]
    y_dev_raw, dev_trend = data["y"]["dev"], data["trend"]["dev"]

    kwargs = dict(
        feature_cols=data["feature_cols"], epochs=epochs,
        batch_size=cfg.BATCH_SIZE,
        lr=cfg.LEARNING_RATE if lr is None else lr,
        weight_decay=cfg.WEIGHT_DECAY, dropout_rate=cfg.NEURAL_CQR_DROPOUT,
        hidden_dims=cfg.NEURAL_CQR_HIDDEN_DIMS if hidden is None else hidden,
        lambda_width=cfg.LAMBDA_WIDTH if lambda_width is None else lambda_width,
        joint_training=True, seed=seed,
    )

    with FlagScope(**flags):
        t0 = time.time()
        if refit_style:
            model = mt.train_neural_cqr(X_fit, y_fit, None, None, early_stopping=False, **kwargs)
        else:
            model = mt.train_neural_cqr(
                X_fit, y_fit, X_dev, data["y_detrended"]["dev"],
                early_stopping_mode=cfg.EARLY_STOPPING_MODE,
                patience=cfg.EARLY_STOPPING_PATIENCE, **kwargs,
            )
        pred, lo, hi = mt.predict_intervals(model, X_dev)
        elapsed = time.time() - t0

    m = dev_metrics(y_dev_raw, pred + dev_trend, lo + dev_trend, hi + dev_trend)
    m["best_epoch"] = int(getattr(model, "best_epoch", 0))
    m["seconds"] = round(elapsed, 1)
    return m


def run_lgbm(data: Dict[str, Any], *, dev_tuned: bool) -> Dict[str, float]:
    import lightgbm as lgb

    X_fit, X_dev = data["X"]["fit"], data["X"]["dev"]
    y_fit = data["y_detrended"]["fit"]
    y_dev_model, y_dev_raw, dev_trend = data["y_detrended"]["dev"], data["y"]["dev"], data["trend"]["dev"]

    params = dict(cfg.LGBM_PARAMS)
    fit_kw: Dict[str, Any] = {}
    if dev_tuned:
        # Fairness (spec §5.2): LightGBM gets the same DEV-based budget selection
        # NeuralCQR gets from early stopping. Nothing here looks at TEST.
        params["n_estimators"] = 3000
        fit_kw = dict(eval_set=[(X_dev, y_dev_model)], eval_metric="l2",
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])

    point = lgb.LGBMRegressor(**params).fit(X_fit, y_fit, **fit_kw)
    q_params = dict(objective="quantile", n_estimators=params.get("n_estimators", 500),
                    learning_rate=params["learning_rate"], random_state=cfg.RANDOM_SEED, verbose=-1)
    lo_m = lgb.LGBMRegressor(alpha=0.05, **q_params).fit(X_fit, y_fit)
    hi_m = lgb.LGBMRegressor(alpha=0.95, **q_params).fit(X_fit, y_fit)

    m = dev_metrics(y_dev_raw, point.predict(X_dev) + dev_trend,
                    lo_m.predict(X_dev) + dev_trend, hi_m.predict(X_dev) + dev_trend)
    m["best_epoch"] = int(getattr(point, "best_iteration_", 0) or params.get("n_estimators", 0))
    m["seconds"] = 0.0
    return m


# ─────────────────────────────────────────────────────────────
# Variants and execution
# ─────────────────────────────────────────────────────────────

def build_specs(variants: str, epochs: int) -> List[Dict[str, Any]]:
    refit_epochs = max(10, min(epochs, 40))
    specs: List[Dict[str, Any]] = [
        {"id": "E0", "label": "baseline (current config)", "kind": "nn", "kw": {}},
        {"id": "E1", "label": "F0a target standardised, delta=1.345 sigma", "kind": "nn",
         "kw": {"STANDARDIZE_TARGET": True}},
        {"id": "E2", "label": f"F0b EMA refit ({refit_epochs} epochs, no ES)", "kind": "nn",
         "kw": {"USE_EMA_WEIGHTS": True}, "refit_style": True, "epochs": refit_epochs},
        {"id": "E2b", "label": f"F0b control: last-epoch refit ({refit_epochs} epochs)", "kind": "nn",
         "kw": {"USE_EMA_WEIGHTS": False}, "refit_style": True, "epochs": refit_epochs},
        {"id": "E3", "label": "F0c lr=1e-3, wider net", "kind": "nn",
         "kw": {}, "lr": 1e-3, "hidden": cfg.LOSO_HIDDEN_DIMS},
        {"id": "E3b", "label": "F0c lr=3e-4, current net", "kind": "nn", "kw": {}, "lr": 3e-4},
        {"id": "E4", "label": "F1 monotone quantile heads", "kind": "nn",
         "kw": {"NEURAL_MONOTONE_HEADS": True}},
        {"id": "E5", "label": "F0d lambda_width = 0", "kind": "nn", "kw": {}, "lambda_width": 0.0},
        {"id": "LG0", "label": "LightGBM default (1000 trees)", "kind": "lgb", "dev_tuned": False},
        {"id": "LG1", "label": "LightGBM DEV-tuned n_estimators", "kind": "lgb", "dev_tuned": True},
    ]
    wanted = {v.strip() for v in variants.split(",") if v.strip()}
    return [s for s in specs if s["id"] in wanted or s["id"].rstrip("b") in wanted]


def run_one_spec(spec: Dict[str, Any], data: Dict[str, Any], seeds: List[int],
                 epochs: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    print(f"\n--- {spec['id']}: {spec['label']} ---", flush=True)
    if spec["kind"] == "lgb":
        m = run_lgbm(data, dev_tuned=spec["dev_tuned"])
        rows.append({"variant": spec["id"], "label": spec["label"], "seed": -1, **m})
        print(f"  DEV RMSE={m['rmse']:.4f}  R2={m['r2']:.4f}  PICP={m['picp']:.3f}  "
              f"trees={m['best_epoch']}", flush=True)
        return rows
    for seed in seeds:
        m = run_neural(
            data, seed=seed, epochs=int(spec.get("epochs", epochs)),
            refit_style=bool(spec.get("refit_style", False)),
            lr=spec.get("lr"), hidden=spec.get("hidden"),
            lambda_width=spec.get("lambda_width"), **spec["kw"],
        )
        rows.append({"variant": spec["id"], "label": spec["label"], "seed": seed, **m})
        print(f"  seed {seed:>4}: RMSE={m['rmse']:.4f}  R2={m['r2']:.4f}  PICP={m['picp']:.3f}  "
              f"MPIW={m['mpiw']:.3f}  cross={m['crossing_rate']:.3f}  ep={m['best_epoch']}  "
              f"{m['seconds']}s", flush=True)
    return rows


def build_or_load_data(cache: Path | None) -> Dict[str, Any]:
    """Build the temporal partitions, or load them from a cache file.

    Workers MUST use the cache. `select_features` calls
    `permutation_importance(..., n_jobs=-1)`, and joblib's loky backend answers
    that with one process per core -- 13 processes of ~250 MB each. Four workers
    rebuilding independently spawned 52 of them and drove free memory to 0.8 GB,
    at which point the machine thrashed and no worker finished a single seed.
    Building once in the parent avoids the pool entirely.
    """
    import joblib

    if cache is not None and cache.exists():
        return joblib.load(cache)
    data = _fold_lab.build_temporal()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data, cache, compress=0)
    return data


def run_parallel(specs: List[Dict[str, Any]], args: argparse.Namespace) -> pd.DataFrame:
    """Run each variant in its own process.

    PyTorch cannot parallelise a (64, 32) net across threads -- measured
    utilisation of the serial run was ~1.25 of 12 logical cores -- so
    variant-level processes are where the speedup is. Each worker is capped to
    `--threads` threads so the workers do not fight over the same cores, and
    they all read one pre-built partition cache (see build_or_load_data).
    """
    import subprocess

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob(f"{args.tag}_part_*.csv"):
        stale.unlink()

    cache = OUT_DIR / f"{args.tag}_partitions.joblib"
    if cache.exists():
        cache.unlink()
    print("Building temporal partitions once for all workers...", flush=True)
    t0 = time.time()
    data = build_or_load_data(cache)
    print(f"  built in {time.time() - t0:.0f}s: FIT n={len(data['y']['fit'])} "
          f"DEV n={len(data['y']['dev'])} features={len(data['feature_cols'])} "
          f"-> {cache.name} ({cache.stat().st_size / 1e6:.0f} MB)", flush=True)
    del data

    env_base = dict(os.environ,
                    OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                    OPENBLAS_NUM_THREADS=str(args.threads), NUMEXPR_NUM_THREADS=str(args.threads),
                    # Belt and braces: cap any loky pool a worker might still create.
                    LOKY_MAX_CPU_COUNT=str(args.threads))
    queue = list(specs)
    running: List[Any] = []
    print(f"\nRunning {len(queue)} variants, {args.parallel} at a time, "
          f"{args.threads} threads each", flush=True)

    while queue or running:
        while queue and len(running) < args.parallel:
            spec = queue.pop(0)
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()),
                   "--only", spec["id"], "--seeds", str(args.seeds),
                   "--epochs", str(args.epochs), "--tag", args.tag,
                   "--threads", str(args.threads), "--data-cache", str(cache)]
            log = OUT_DIR / f"{args.tag}_part_{spec['id']}.log"
            proc = subprocess.Popen(cmd, stdout=log.open("w", encoding="utf-8"),
                                    stderr=subprocess.STDOUT, env=env_base)
            running.append((spec["id"], proc, log))
            print(f"  started {spec['id']:<5} (pid {proc.pid}) -> {log.name}", flush=True)
        time.sleep(5)
        for entry in list(running):
            vid, proc, log = entry
            if proc.poll() is not None:
                running.remove(entry)
                print(f"  {vid:<5} {'done' if proc.returncode == 0 else f'FAILED rc={proc.returncode}'}",
                      flush=True)
                if proc.returncode != 0:
                    for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]:
                        print("      " + line, flush=True)

    parts = sorted(OUT_DIR.glob(f"{args.tag}_part_*.csv"))
    if not parts:
        raise SystemExit("no worker produced results; see the *_part_*.log files")
    return pd.concat([pd.read_csv(f) for f in parts], ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=cfg.BASELINE_MAX_EPOCHS)
    ap.add_argument("--variants", type=str, default="E0,E1,E2,E3,E4,E5,LG0,LG1")
    ap.add_argument("--tag", type=str, default="w3_dev_ladder")
    ap.add_argument("--parallel", type=int, default=1,
                    help="run this many variants concurrently, each in its own process")
    ap.add_argument("--threads", type=int, default=2,
                    help="torch/BLAS threads per worker")
    ap.add_argument("--only", type=str, default=None,
                    help="internal: run a single variant and write a part file")
    ap.add_argument("--data-cache", type=str, default=None,
                    help="internal: joblib file holding the pre-built partitions")
    args = ap.parse_args()

    seeds = [cfg.RANDOM_SEED, 7, 123, 2024, 3407][: args.seeds]

    # ── worker mode ───────────────────────────────────────────────────
    if args.only:
        import torch
        torch.set_num_threads(max(1, args.threads))
        spec = next((s for s in build_specs(args.only, args.epochs) if s["id"] == args.only), None)
        if spec is None:
            raise SystemExit(f"unknown variant {args.only}")
        data = build_or_load_data(Path(args.data_cache) if args.data_cache else None)
        rows = run_one_spec(spec, data, seeds, args.epochs)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT_DIR / f"{args.tag}_part_{args.only}.csv", index=False)
        return 0

    specs = build_specs(args.variants, args.epochs)
    fit_n = dev_n = n_feat = fit_sd = None
    note = ""

    if args.parallel > 1:
        df = run_parallel(specs, args)
        note = "each worker rebuilt the partitions independently"
    else:
        print(f"Building temporal partitions (FIT 1985-2013, DEV {cfg.DEV_YEARS})...")
        data = _fold_lab.build_temporal()
        fit_n, dev_n = int(len(data["y"]["fit"])), int(len(data["y"]["dev"]))
        n_feat, fit_sd = int(len(data["feature_cols"])), float(np.std(data["y_detrended"]["fit"]))
        print(f"  FIT n={fit_n}  DEV n={dev_n}  features={n_feat}  "
              f"trend={data['trend_coef'][0]:.4f} t/ha/yr")
        print(f"  detrended FIT target: sd={fit_sd:.3f}")
        rows: List[Dict[str, Any]] = []
        for spec in specs:
            rows.extend(run_one_spec(spec, data, seeds, args.epochs))
        df = pd.DataFrame(rows)

    summary = (df[df.seed >= 0].groupby(["variant", "label"])
               .agg(n_seeds=("seed", "count"), rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
                    r2_mean=("r2", "mean"), r2_sd=("r2", "std"), picp_mean=("picp", "mean"),
                    mpiw_mean=("mpiw", "mean"), crossing_mean=("crossing_rate", "mean"))
               .reset_index())
    lgb_rows = df[df.seed < 0]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"{args.tag}_raw.csv", index=False)
    summary.to_csv(OUT_DIR / f"{args.tag}_summary.csv", index=False)
    (OUT_DIR / f"{args.tag}.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": seeds, "epochs": args.epochs, "parallel": args.parallel, "note": note,
        "fit_n": fit_n, "dev_n": dev_n, "n_features": n_feat, "fit_target_sd": fit_sd,
        "neural": summary.to_dict("records"),
        "lightgbm": lgb_rows.to_dict("records"),
        "decision_rule": "D2: accept a variant only if DEV RMSE improves on E0 averaged over seeds.",
        "caveat": ("NeuralCQR variants select their epoch on DEV via early stopping, so their DEV "
                   "metric is optimistic relative to LightGBM. Rank neural variants against each "
                   "other here; do not settle the backbone choice (D1) from this table alone."),
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 92)
    print(f"{'variant':<6}{'DEV RMSE':>12}{'sd':>9}{'DEV R2':>10}{'PICP':>8}{'MPIW':>9}  label")
    for _, r in summary.iterrows():
        print(f"{r['variant']:<6}{r['rmse_mean']:>12.4f}{(r['rmse_sd'] or 0):>9.4f}"
              f"{r['r2_mean']:>10.4f}{r['picp_mean']:>8.3f}{r['mpiw_mean']:>9.3f}  {r['label']}")
    for _, r in lgb_rows.iterrows():
        print(f"{r['variant']:<6}{r['rmse']:>12.4f}{'-':>9}{r['r2']:>10.4f}{r['picp']:>8.3f}"
              f"{r['mpiw']:>9.3f}  {r['label']}")
    print("=" * 92)
    print(f"written: {OUT_DIR / (args.tag + '_summary.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
