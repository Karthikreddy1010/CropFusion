"""p7_feature_groups.py - Phase 15 controlled feature-group experiments (LightGBM only).

Runs after the target-representation diagnosis, and in BOTH regimes so the two variables
are never confounded:
    raw        - the shipped LOSO target representation
    detrended  - FIT-only linear detrend, trend re-added before scoring

Feature configurations (the held-out state, the fold boundaries, the scaler, the
hyperparameters and the per-fold selection are all unchanged; only the column list changes):

    1 full                 every selected feature
    2 no_fourier           drop Fourier_*            (periodic time encodings)
    3 no_yield_lags        drop Yield_lag1/2         (constant on CAL and TEST)
    4 climate_only         weather + drought indices only
    5 climate_soil         + soil
    6 climate_soil_topo    + topography
    7 no_cdhw              drop the CDHW family

LightGBM only, because 7 configs x 2 regimes x 6 folds is 84 fits; the tree backbone is the
one whose anomaly is under investigation and it needs no GPU. Results are indicative for the
ensemble, not a substitute for it, and are labelled as such.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from _fold_lab import build_loso_fold, cfg, DIAG_OUT
from model_training import train_lgbm_quantile, predict_intervals
from evaluation import rmse, mae, r_squared

CDHW_PREFIXES = ("CDHW_", "cdhw_", "Inter_CDHW")
SOIL_PREFIX = "soil_"
TOPO = ("elevation_mean_m", "elevation_std_m", "slope_mean")
LANDCOVER_PREFIX = "nlcd_"


def groups(cols):
    fourier = [c for c in cols if c.startswith("Fourier_")]
    lags = [c for c in cols if c.startswith("Yield_lag")]
    cdhw = [c for c in cols if c.startswith(CDHW_PREFIXES)]
    soil = [c for c in cols if c.startswith(SOIL_PREFIX)]
    topo = [c for c in cols if c in TOPO]
    land = [c for c in cols if c.startswith(LANDCOVER_PREFIX)]
    static = set(soil + topo + land)
    climate = [c for c in cols if c not in static and c not in fourier and c not in lags]
    return dict(fourier=fourier, lags=lags, cdhw=cdhw, soil=soil, topo=topo,
                landcover=land, climate=climate)


def configs(cols):
    g = groups(cols)
    keep = lambda drop: [c for c in cols if c not in set(drop)]
    return {
        "1_full": cols,
        "2_no_fourier": keep(g["fourier"]),
        "3_no_yield_lags": keep(g["lags"]),
        "4_climate_only": g["climate"],
        "5_climate_soil": g["climate"] + g["soil"],
        "6_climate_soil_topo": g["climate"] + g["soil"] + g["topo"],
        "7_no_cdhw": keep(g["cdhw"]),
    }


def run(states, regimes=("raw", "detrended")):
    res = {}
    for st in states:
        fold = build_loso_fold(st)
        cols, X, y, yr = fold["feature_cols"], fold["X"], fold["y"], fold["years"]
        frames = fold["scaled"]
        coef = np.polyfit(yr["fit"], y["fit"], 1)
        trend = {k: np.polyval(coef, yr[k]) for k in ("tr_fit", "dev", "test")}
        res[st] = {"n_features": len(cols), "groups": {k: len(v) for k, v in groups(cols).items()}}

        for name, sub in configs(cols).items():
            idx = [cols.index(c) for c in sub]
            for regime in regimes:
                det = regime == "detrended"
                y_tr = y["tr_fit"] - trend["tr_fit"] if det else y["tr_fit"]
                m = train_lgbm_quantile(X["tr_fit"][:, idx], y_tr,
                                        X["es_val"][:, idx], y["es_val"], sub)
                p_dev, _, _ = predict_intervals(m, X["dev"][:, idx])
                p_te, _, _ = predict_intervals(m, X["test"][:, idx])
                if det:
                    p_dev, p_te = p_dev + trend["dev"], p_te + trend["test"]
                res[st].setdefault(regime, {})[name] = {
                    "n_features": len(sub),
                    "dev_r2": round(float(r_squared(y["dev"], p_dev)), 4),
                    "test_r2": round(float(r_squared(y["test"], p_te)), 4),
                    "test_rmse": round(float(rmse(y["test"], p_te)), 4),
                    "test_mae": round(float(mae(y["test"], p_te)), 4),
                    "test_bias": round(float(np.mean(p_te - y["test"])), 4),
                }
                r = res[st][regime][name]
                print(f"  {st:10s} {regime:10s} {name:22s} n={r['n_features']:3d} "
                      f"DEV R2={r['dev_r2']:>8.4f}  TEST R2={r['test_r2']:>8.4f}  "
                      f"RMSE={r['test_rmse']:.4f}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="*", default=None)
    ap.add_argument("--regimes", nargs="*", default=["raw", "detrended"])
    a = ap.parse_args()
    res = run(a.states or list(cfg.LOSO_STATES), tuple(a.regimes))

    # macro summary
    print("\nMACRO (mean over folds), LightGBM only:")
    for regime in a.regimes:
        names = list(next(iter(res.values()))[regime].keys())
        for n in names:
            vals = [res[s][regime][n] for s in res]
            print(f"  {regime:10s} {n:22s} DEV R2={np.mean([v['dev_r2'] for v in vals]):>8.4f}  "
                  f"TEST R2={np.mean([v['test_r2'] for v in vals]):>8.4f}  "
                  f"RMSE={np.mean([v['test_rmse'] for v in vals]):.4f}")
    out = DIAG_OUT / "reports" / "p7_feature_groups.json"
    out.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"Wrote {out}")
