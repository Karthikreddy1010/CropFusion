"""p5_controlled.py - Phases 11-14 controlled experiments (diagnostic only).

Every experiment changes exactly ONE variable relative to the shipped LOSO path
and leaves the protocol untouched: the same six states, the same
FIT/DEV/CAL/TEST year boundaries, the same per-fold feature selection, the same
scaler, the same hyperparameters, the same composite loss, the same
ensemble-weight search on DEV and the same conformal calibration on CAL.

  X0  shipped LOSO                       (raw target, 90% FIT head, no eval_set)
  X1  + FIT-only linear detrend          (the treatment the temporal path already applies)
  X2  + LightGBM eval_set/early stopping (the X_val/y_val the function already accepts)
  X3  NeuralCQR with the temporal-split configuration (negative control, Phase 14 N1)

Nothing in code/ outside this folder is modified, and no result here is used to
select anything. Held-out-state numbers are reported once, never optimised against.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from _fold_lab import build_loso_fold, build_temporal, describe, cfg, DIAG_OUT

from model_training import train_lgbm_quantile, train_neural_cqr, predict_intervals
from aci_calibrator import static_conformal
from evaluation import rmse, mae, r_squared, picp, mpiw, winkler_score


def stats(y, p, tag=""):
    y = np.asarray(y, float); p = np.asarray(p, float)
    b = float(np.mean(p - y))
    return {"tag": tag, "r2": round(float(r_squared(y, p)), 4), "rmse": round(float(rmse(y, p)), 4),
            "mae": round(float(mae(y, p)), 4), "bias": round(b, 4),
            "r2_debiased": round(float(r_squared(y, p - b)), 4),
            "pearson_r": round(float(np.corrcoef(p, y)[0, 1]), 4)}


def yearly_bias_slope(years, y, p):
    years = np.asarray(years, float); y = np.asarray(y, float); p = np.asarray(p, float)
    uy = np.unique(years)
    b = np.array([float(np.mean(p[years == v] - y[years == v])) for v in uy])
    return round(float(np.polyfit(uy, b, 1)[0]), 4)


def _lgb_es(X_tr, y_tr, X_val, y_val, cols):
    """X2: identical to train_lgbm_quantile except the eval_set it already accepts is used."""
    import lightgbm as lgb
    cb = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    p = lgb.LGBMRegressor(**cfg.LGBM_PARAMS).fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=cb)
    lo = lgb.LGBMRegressor(objective="quantile", alpha=0.05, n_estimators=500,
                           random_state=cfg.RANDOM_SEED, verbose=-1).fit(
        X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="quantile", callbacks=cb)
    hi = lgb.LGBMRegressor(objective="quantile", alpha=0.95, n_estimators=500,
                           random_state=cfg.RANDOM_SEED, verbose=-1).fit(
        X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="quantile", callbacks=cb)
    from model_training import QuantileModelSet
    return QuantileModelSet(point_model=p, lower_model=lo, upper_model=hi,
                            model_type="LightGBM_ES", feature_cols=cols, is_neural=False)


def _neural(fold, cfg_name):
    """cfg_name='loso' -> the shipped LOSO configuration; 'temporal' -> the main-pipeline one."""
    kw = dict(joint_training=True, scaler=fold["scaler"])
    if cfg_name == "loso":
        kw.update(epochs=cfg.LOSO_MAX_EPOCHS, batch_size=cfg.LOSO_BATCH_SIZE, lr=cfg.LOSO_LEARNING_RATE,
                  weight_decay=cfg.LOSO_WEIGHT_DECAY, early_stopping_mode=cfg.LOSO_EARLY_STOPPING_MODE,
                  patience=cfg.LOSO_EARLY_STOPPING_PATIENCE, hidden_dims=cfg.LOSO_HIDDEN_DIMS,
                  dropout_rate=cfg.LOSO_DROPOUT)
    else:
        kw.update(epochs=cfg.BASELINE_MAX_EPOCHS, batch_size=cfg.BATCH_SIZE, lr=cfg.LEARNING_RATE,
                  weight_decay=cfg.WEIGHT_DECAY, early_stopping_mode=cfg.EARLY_STOPPING_MODE,
                  patience=cfg.EARLY_STOPPING_PATIENCE, hidden_dims=cfg.NEURAL_CQR_HIDDEN_DIMS,
                  dropout_rate=cfg.NEURAL_CQR_DROPOUT)
    return kw


def run_fold(state: str, detrend: bool, lgb_early_stopping: bool, neural_cfg: str) -> dict:
    """One complete LOSO fold: both backbones, DEV weight search, CAL conformal, TEST report."""
    fold = build_loso_fold(state)
    cols, X, y, yr = fold["feature_cols"], fold["X"], fold["y"], fold["years"]

    if detrend:
        coef = np.polyfit(yr["fit"], y["fit"], 1)           # FIT partition of the five dev states only
        tr = {k: np.polyval(coef, yr[k]) for k in ("tr_fit", "es_val", "dev", "cal", "test")}
    else:
        coef = np.array([0.0, 0.0])
        tr = {k: np.zeros(len(y[k])) for k in ("tr_fit", "es_val", "dev", "cal", "test")}

    y_tr_m, y_es_m = y["tr_fit"] - tr["tr_fit"], y["es_val"] - tr["es_val"]

    nm = train_neural_cqr(X["tr_fit"], y_tr_m, X["es_val"], y_es_m, cols, **_neural(fold, neural_cfg))
    pn_dev, ql_dev_n, qh_dev_n = predict_intervals(nm, X["dev"])
    pn_cal, ql_cal_n, qh_cal_n = predict_intervals(nm, X["cal"])
    pn_te, ql_te_n, qh_te_n = predict_intervals(nm, X["test"])

    trainer = _lgb_es if lgb_early_stopping else train_lgbm_quantile
    lm = trainer(X["tr_fit"], y_tr_m, X["es_val"], y_es_m, cols)
    pl_dev, ql_dev_l, qh_dev_l = predict_intervals(lm, X["dev"])
    pl_cal, ql_cal_l, qh_cal_l = predict_intervals(lm, X["cal"])
    pl_te, ql_te_l, qh_te_l = predict_intervals(lm, X["test"])

    # back to the raw scale for every score (exactly what the temporal path does)
    add = lambda v, k: v + tr[k]
    pn_dev, pn_cal, pn_te = add(pn_dev, "dev"), add(pn_cal, "cal"), add(pn_te, "test")
    pl_dev, pl_cal, pl_te = add(pl_dev, "dev"), add(pl_cal, "cal"), add(pl_te, "test")
    ql_cal_n, qh_cal_n = add(ql_cal_n, "cal"), add(qh_cal_n, "cal")
    ql_te_n, qh_te_n = add(ql_te_n, "test"), add(qh_te_n, "test")
    ql_cal_l, qh_cal_l = add(ql_cal_l, "cal"), add(qh_cal_l, "cal")
    ql_te_l, qh_te_l = add(ql_te_l, "test"), add(qh_te_l, "test")

    # DEV ensemble-weight search -- identical grid and objective to main.py:1634
    r2n = float(r_squared(y["dev"], pn_dev))
    best_w, best = 1.0, r2n
    grid = []
    for w in np.arange(0.0, 1.01, 0.1):
        w = round(float(w), 2)
        v = float(r_squared(y["dev"], w * pn_dev + (1 - w) * pl_dev))
        grid.append({"w_neural": w, "dev_r2": round(v, 4)})
        if v > best:
            best, best_w = v, w

    p_cal = best_w * pn_cal + (1 - best_w) * pl_cal
    ql_cal = best_w * ql_cal_n + (1 - best_w) * ql_cal_l
    qh_cal = best_w * qh_cal_n + (1 - best_w) * qh_cal_l
    p_te = best_w * pn_te + (1 - best_w) * pl_te
    ql_te = best_w * ql_te_n + (1 - best_w) * ql_te_l
    qh_te = best_w * qh_te_n + (1 - best_w) * qh_te_l

    cal_res = static_conformal(y["cal"], ql_cal, qh_cal, ql_te, qh_te, p_te, y["test"])
    lo, hi = np.asarray(cal_res.q_lo), np.asarray(cal_res.q_hi)

    return {
        "state": state, "detrend": detrend, "lgb_early_stopping": lgb_early_stopping,
        "neural_cfg": neural_cfg, "n_features": len(cols),
        "fit_trend_t_ha_per_year": round(float(coef[0]), 4),
        "neural_epochs": int(getattr(nm, "epochs_trained", -1) or -1),
        "neural_best_epoch": int(getattr(nm, "best_epoch", -1) or -1),
        "lgb_trees": int(lm.point_model.booster_.num_trees()),
        "dev_neural": stats(y["dev"], pn_dev, "neural DEV"),
        "dev_lgb": stats(y["dev"], pl_dev, "lgb DEV"),
        "dev_weight_grid": grid,
        "selected_w_neural": best_w, "selected_w_lgb": round(1 - best_w, 2),
        "dev_blend": stats(y["dev"], best_w * pn_dev + (1 - best_w) * pl_dev, "blend DEV"),
        "cal_neural": stats(y["cal"], pn_cal, "neural CAL"),
        "cal_lgb": stats(y["cal"], pl_cal, "lgb CAL"),
        "test_point": stats(y["test"], p_te, "ensemble TEST"),
        "test_yearly_bias_slope": yearly_bias_slope(yr["test"], y["test"], p_te),
        "test_neural_only": stats(y["test"], pn_te, "neural TEST"),
        "test_lgb_only": stats(y["test"], pl_te, "lgb TEST"),
        "conformal_threshold": round(float(cal_res.metadata.get("threshold", 0.0)), 4),
        "test_picp": round(float(picp(y["test"], lo, hi)), 4),
        "test_mpiw": round(float(mpiw(lo, hi)), 4),
        "test_winkler": round(float(winkler_score(y["test"], lo, hi)), 4),
        "neural_loss_history_tail": (nm.training_history[-1] if getattr(nm, "training_history", None) else None),
        "neural_loss_history_head": (nm.training_history[0] if getattr(nm, "training_history", None) else None),
    }


def lgb_variant_sweep(state: str) -> dict:
    """LightGBM-only sweep on one fold's DEV partition: the two candidate variables
    (target representation, training-rows) plus the unused eval_set, one at a time."""
    fold = build_loso_fold(state)
    cols, X, y, yr = fold["feature_cols"], fold["X"], fold["y"], fold["years"]
    coef = np.polyfit(yr["fit"], y["fit"], 1)
    tr = {k: np.polyval(coef, yr[k]) for k in ("tr_fit", "fit", "es_val", "dev")}

    out = {"fit_trend": round(float(coef[0]), 4)}
    variants = [
        ("E0_shipped", False, "tr_fit", False),
        ("E1_full_fit_rows", False, "fit", False),
        ("E2_eval_set_early_stopping", False, "tr_fit", True),
        ("E3_detrended", True, "tr_fit", False),
        ("E4_detrended_plus_full_fit", True, "fit", False),
        ("E5_detrended_plus_eval_set", True, "tr_fit", True),
    ]
    for name, det, part, es in variants:
        y_tr = y[part] - tr[part] if det else y[part]
        y_es = y["es_val"] - tr["es_val"] if det else y["es_val"]
        m = (_lgb_es if es else train_lgbm_quantile)(X[part], y_tr, X["es_val"], y_es, cols)
        p, _, _ = predict_intervals(m, X["dev"])
        if det:
            p = p + tr["dev"]
        rec = stats(y["dev"], p, name)
        rec.update(detrend=det, train_partition=part, eval_set=es,
                   train_years=[int(yr[part].min()), int(yr[part].max())],
                   n_train=int(len(y[part])), n_trees=int(m.point_model.booster_.num_trees()))
        out[name] = rec
    return out


def temporal_lgb() -> dict:
    tp = build_temporal()
    X, y, ydet, trn, cols = tp["X"], tp["y"], tp["y_detrended"], tp["trend"], tp["feature_cols"]
    out = {"n_features": len(cols), "trend": round(float(tp["trend_coef"][0]), 4)}
    for name, ytr, yval, back in (("A_detrended_as_shipped", ydet["fit"], ydet["dev"], True),
                                  ("B_detrend_OFF", y["fit"], y["dev"], False)):
        m = train_lgbm_quantile(X["fit"], ytr, X["dev"], yval, cols)
        pd_, _, _ = predict_intervals(m, X["dev"])
        pt_, _, _ = predict_intervals(m, X["test"])
        if back:
            pd_, pt_ = pd_ + trn["dev"], pt_ + trn["test"]
        out[name] = {"dev": stats(y["dev"], pd_, f"temporal LGBM DEV {name}"),
                     "test": stats(y["test"], pt_, f"temporal LGBM TEST {name}")}
    out["target_means"] = {p: round(float(np.mean(y[p])), 4) for p in ("fit", "dev", "cal", "test")}
    out["lag1_nan_before_fill"] = {p: round(float(tp["frames"][p]["Yield_lag1"].isna().mean()), 4)
                                   for p in ("fit", "dev", "cal", "test")}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True,
                    choices=["X0", "X1", "X2", "X3", "temporal", "lgbsweep"])
    ap.add_argument("--states", nargs="*", default=None)
    args = ap.parse_args()

    spec = {"X0": dict(detrend=False, lgb_early_stopping=False, neural_cfg="loso"),
            "X1": dict(detrend=True, lgb_early_stopping=False, neural_cfg="loso"),
            "X2": dict(detrend=False, lgb_early_stopping=True, neural_cfg="loso"),
            "X3": dict(detrend=False, lgb_early_stopping=False, neural_cfg="temporal")}

    if args.experiment == "temporal":
        res = temporal_lgb()
        print(json.dumps(res, indent=1))
    elif args.experiment == "lgbsweep":
        res = {}
        for st in (args.states or list(cfg.LOSO_STATES)):
            res[st] = lgb_variant_sweep(st)
            print(f"--- {st} (fit trend {res[st]['fit_trend']} t/ha/yr) ---", flush=True)
            for k, v in res[st].items():
                if isinstance(v, dict):
                    print(f"    {k:30s} DEV R2={v['r2']:>8.4f}  bias={v['bias']:>7.4f}  "
                          f"rmse={v['rmse']:.4f}  trees={v['n_trees']:>4d}  years={v['train_years']}", flush=True)
    else:
        res = {}
        for st in (args.states or list(cfg.LOSO_STATES)):
            r = run_fold(st, **spec[args.experiment])
            res[st] = r
            print(f"[{args.experiment}] {st}: DEV neural R2={r['dev_neural']['r2']} "
                  f"lgb R2={r['dev_lgb']['r2']} | w_neural={r['selected_w_neural']} | "
                  f"TEST R2={r['test_point']['r2']} bias={r['test_point']['bias']} "
                  f"slope={r['test_yearly_bias_slope']} PICP={r['test_picp']} MPIW={r['test_mpiw']}",
                  flush=True)
        if res:
            r2s = [v["test_point"]["r2"] for v in res.values()]
            print(f"[{args.experiment}] MACRO TEST R2 = {np.mean(r2s):.4f}", flush=True)

    out = DIAG_OUT / "reports" / f"p5_{args.experiment}.json"
    out.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"Wrote {out}")
