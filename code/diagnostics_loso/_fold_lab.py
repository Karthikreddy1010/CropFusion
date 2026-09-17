"""_fold_lab.py - read-only reconstruction of the pipeline's own LOSO / temporal data paths.

FORENSIC USE ONLY. This module imports and calls the locked pipeline functions
(`loso_cv_folds`, `build_split_aware_*`, `TrainFittedPreprocessor`, `select_features`,
`fit_scaler`, `apply_scaling`, `get_feature_target_arrays`) exactly as `main._run_loso_cv`
and `main.main` call them, so any number measured here is a measurement of the real
pipeline, not of a re-implementation.

It changes nothing:
  * no pipeline module is edited,
  * every config *_DIR is redirected to <PROJECT_ROOT>/outputs_diagnostics so the
    Colab run in outputs_master/ is never touched,
  * no model, threshold, loss, split boundary or state list is altered.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PAPER3_DATA_SOURCE", "master")

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import numpy as np
import pandas as pd

import config as cfg

# ── Redirect every output directory away from outputs_master ─────────────────
DIAG_OUT = cfg.PROJECT_ROOT / "outputs_diagnostics"
_OLD_OUT = cfg.OUTPUT_DIR
for _name in dir(cfg):
    if not _name.endswith("_DIR"):
        continue
    _val = getattr(cfg, _name)
    if not isinstance(_val, Path):
        continue
    try:
        _rel = _val.relative_to(_OLD_OUT)
    except ValueError:
        continue
    setattr(cfg, _name, DIAG_OUT / _rel)
cfg.OUTPUT_DIR = DIAG_OUT
for _d in ("", "reports", "splits", "figures", "predictions", "audits", "metrics"):
    (DIAG_OUT / _d).mkdir(parents=True, exist_ok=True)

from data_loader import load_dataset                     # noqa: E402
from feature_engineering import (                        # noqa: E402
    engineer_features, build_split_aware_lags, build_split_aware_rolling_features,
)
from feature_selection import select_features            # noqa: E402
from preprocessor import TrainFittedPreprocessor         # noqa: E402
from scaling import fit_scaler, apply_scaling            # noqa: E402
from splitting import (                                  # noqa: E402
    loso_cv_folds, temporal_split_4way, get_feature_target_arrays, obs_ids, add_obs_ids,
)
from utils import set_global_seed                        # noqa: E402

BOOKKEEPING = {"county_baseline", "Fold", "HeldOutState", "obs_id", "Split"}
_CACHE: dict = {}


def engineered_df() -> pd.DataFrame:
    """df exactly as main.py has it when it calls _run_loso_cv (line 1130)."""
    if "df" not in _CACHE:
        set_global_seed()
        df = load_dataset()
        df, _ = engineer_features(df)
        _CACHE["df"] = df
    return _CACHE["df"]


def build_loso_fold(state: str, df: pd.DataFrame | None = None) -> dict:
    """Reproduce steps 1-4 of main._run_loso_cv for one held-out state.

    Returns every intermediate object the audit needs: the raw fold frames, the
    post-lag frames, the preprocessed frames, the fold feature list (in order),
    the fold scaler, the scaled frames and the final arrays including the
    internal early-stopping carve-out that main.py hands to the models.
    """
    df = engineered_df() if df is None else df
    out: dict = {"state": state}

    for st, fit_fold, dev_fold, cal_fold, test_fold in loso_cv_folds(df, return_cal=True):
        if st != state:
            continue

        out["raw"] = {"fit": fit_fold.copy(), "dev": dev_fold.copy(),
                      "cal": cal_fold.copy(), "test": test_fold.copy()}

        # 1. split-aware lags + rolling, then fit-fitted preprocessing
        fit_fold, dev_fold, cal_fold, test_fold = build_split_aware_lags(
            fit_fold, dev_fold, test_fold, target_col=cfg.PRIMARY_TARGET,
            split_type="loso", cal_df=cal_fold,
        )
        fit_fold, dev_fold, cal_fold, test_fold = build_split_aware_rolling_features(
            fit_fold, dev_fold, test_fold, split_type="loso", cal_df=cal_fold,
        )
        out["lagged"] = {"fit": fit_fold.copy(), "dev": dev_fold.copy(),
                         "cal": cal_fold.copy(), "test": test_fold.copy()}

        prep_fold = TrainFittedPreprocessor(target_col=cfg.PRIMARY_TARGET)
        fit_fold = prep_fold.fit_transform(fit_fold, split_name=f"loso_{state}_fit")
        dev_fold = prep_fold.transform(dev_fold, split_name=f"loso_{state}_dev")
        cal_fold = prep_fold.transform(cal_fold, split_name=f"loso_{state}_cal")
        test_fold = prep_fold.transform(test_fold, split_name=f"loso_{state}_test")
        out["prep"] = prep_fold
        out["frames"] = {"fit": fit_fold, "dev": dev_fold, "cal": cal_fold, "test": test_fold}

        # 2. per-fold feature selection on the fold's FIT partition only
        fold_feats, sel_report = select_features(
            fit_fold, target=cfg.PRIMARY_TARGET, report_suffix=f"_loso_{state.lower()}_diag",
        )
        fold_feats = [f for f in fold_feats if f in fit_fold.columns]
        loso_feature_cols = [f for f in fold_feats if f not in BOOKKEEPING]
        out["feature_cols"] = loso_feature_cols
        out["selection_report"] = sel_report

        # 3. scaler fitted on the fold's FIT partition only
        fold_scaler, _ = fit_scaler(fit_fold, loso_feature_cols, scaler_type="robust")
        out["scaler"] = fold_scaler
        scaled = {
            name: apply_scaling(part, loso_feature_cols, fold_scaler, split_name=f"loso_{state}_{name}")
            for name, part in out["frames"].items()
        }
        out["scaled"] = scaled

        arrays = {
            name: get_feature_target_arrays(part, loso_feature_cols, split_name=f"loso_{state}_{name}")
            for name, part in scaled.items()
        }
        out["X"] = {k: v[0] for k, v in arrays.items()}
        out["y"] = {k: v[1] for k, v in arrays.items()}

        # 4. internal early-stopping carve-out (chronological tail of FIT)
        X_fit, y_fit = out["X"]["fit"], out["y"]["fit"]
        n_es_val = max(1, len(X_fit) // 10)
        out["n_es_val"] = n_es_val
        out["X"]["tr_fit"], out["y"]["tr_fit"] = X_fit[:-n_es_val], y_fit[:-n_es_val]
        out["X"]["es_val"], out["y"]["es_val"] = X_fit[-n_es_val:], y_fit[-n_es_val:]
        fit_ids = obs_ids(fit_fold)
        out["ids"] = {"train": fit_ids[:-n_es_val], "es_internal": fit_ids[-n_es_val:],
                      "dev": obs_ids(dev_fold), "cal": obs_ids(cal_fold), "test": obs_ids(test_fold)}
        out["years"] = {
            "tr_fit": fit_fold["Year"].values[:-n_es_val],
            "es_val": fit_fold["Year"].values[-n_es_val:],
            "fit": fit_fold["Year"].values, "dev": dev_fold["Year"].values,
            "cal": cal_fold["Year"].values, "test": test_fold["Year"].values,
        }
        return out

    raise ValueError(f"State {state!r} not produced by loso_cv_folds (LOSO_STATES={cfg.LOSO_STATES})")


def build_temporal(df: pd.DataFrame | None = None) -> dict:
    """Reproduce main.main()'s temporal data path (lines 134-310) up to the arrays.

    Includes the multicollinearity prune and the county_baseline append exactly as
    main.py does, then the FIT-only linear detrend.
    """
    df = engineered_df() if df is None else df
    from multicollinearity import analyze_multicollinearity

    fit_df, dev_df, cal_df, test_df = temporal_split_4way(df, target=cfg.PRIMARY_TARGET)
    fit_df, dev_df, cal_df, test_df = build_split_aware_lags(
        fit_df, dev_df, test_df, target_col=cfg.PRIMARY_TARGET, split_type="temporal", cal_df=cal_df)
    fit_df, dev_df, cal_df, test_df = build_split_aware_rolling_features(
        fit_df, dev_df, test_df, split_type="temporal", cal_df=cal_df)

    prep = TrainFittedPreprocessor(target_col=cfg.PRIMARY_TARGET)
    fit_df = prep.fit_transform(fit_df, split_name="fit")
    dev_df = prep.transform(dev_df, split_name="dev")
    cal_df = prep.transform(cal_df, split_name="cal")
    test_df = prep.transform(test_df, split_name="test")

    feature_cols, _ = select_features(fit_df, target=cfg.PRIMARY_TARGET, report_suffix="_temporal_diag")
    feature_cols = _multicollinearity_prune(fit_df, feature_cols)

    if getattr(cfg, "ADD_COUNTY_BASELINE", False):
        _cb = _county_baseline_map(fit_df)
        for d in (fit_df, dev_df, cal_df, test_df):
            d["county_baseline"] = d["GEOID"].map(_cb).fillna(0.0)
        if "county_baseline" not in feature_cols:
            feature_cols = feature_cols + ["county_baseline"]

    frames = {"fit": add_obs_ids(fit_df), "dev": add_obs_ids(dev_df),
              "cal": add_obs_ids(cal_df), "test": add_obs_ids(test_df)}
    scaler, _ = fit_scaler(frames["fit"], feature_cols, scaler_type="robust")
    scaled = {n: apply_scaling(p, feature_cols, scaler, split_name=n) for n, p in frames.items()}
    arrays = {n: get_feature_target_arrays(p, feature_cols, split_name=n) for n, p in scaled.items()}

    X = {k: v[0] for k, v in arrays.items()}
    y = {k: v[1] for k, v in arrays.items()}

    coef = np.polyfit(frames["fit"]["Year"].values, y["fit"], 1)
    trend = {n: np.polyval(coef, frames[n]["Year"].values) for n in frames}
    y_detrended = {n: y[n] - trend[n] for n in frames}

    return {"frames": frames, "scaled": scaled, "feature_cols": feature_cols, "scaler": scaler,
            "X": X, "y": y, "y_detrended": y_detrended, "trend": trend,
            "trend_coef": coef, "prep": prep}


def _county_baseline_map(fit_df: pd.DataFrame) -> pd.Series:
    """Verbatim copy of main.py lines 198-208 (mean FIT detrended anomaly per county)."""
    _tr = fit_df[fit_df[cfg.PRIMARY_TARGET].notna()]
    _cf0 = np.polyfit(_tr["Year"].values, _tr[cfg.PRIMARY_TARGET].values, 1)
    _anom = _tr[cfg.PRIMARY_TARGET].values - np.polyval(_cf0, _tr["Year"].values)
    return pd.Series(_anom, index=_tr["GEOID"].values).groupby(level=0).mean()


def _multicollinearity_prune(_df, cols):
    """Verbatim copy of main.py's inner _multicollinearity_prune (lines 156-191)."""
    import numpy as _np
    _prot = getattr(cfg, "MULTICOLLINEARITY_PROTECT", [])
    _ct = getattr(cfg, "CORR_DROP_THRESHOLD", 0.95)
    _vt = getattr(cfg, "VIF_DROP_THRESHOLD", 10.0)
    _X = _df[cols]
    _tc = pd.concat([_X, _df[cfg.PRIMARY_TARGET]], axis=1).corr()[cfg.PRIMARY_TARGET].abs()
    _c = _X.corr().abs()
    _up = _c.where(_np.triu(_np.ones(_c.shape), k=1).astype(bool))
    _drop = set()
    for _a in _up.columns:
        for _b in _up.index:
            _v = _up.loc[_b, _a]
            if pd.notna(_v) and _v > _ct and _a not in _drop and _b not in _drop:
                _drop.add(_a if _tc.get(_a, 0) < _tc.get(_b, 0) else _b)
    _kept = [c for c in cols if c not in _drop]
    while len(_kept) > 6:
        _Xv = _X[_kept].values
        _Xv = (_Xv - _Xv.mean(axis=0)) / _np.maximum(_Xv.std(axis=0), 1e-6)
        _corr_mat = _np.corrcoef(_Xv, rowvar=False)
        try:
            _inv_corr = _np.linalg.pinv(_corr_mat)
            _vifs = pd.Series(_np.diag(_inv_corr), index=_kept)
        except Exception:
            break
        _cand = _vifs.drop([c for c in _prot if c in _vifs.index], errors="ignore")
        if _cand.empty or _cand.max() <= _vt:
            break
        _kept.remove(_cand.idxmax())
    _kept = list(dict.fromkeys(_kept + [c for c in _prot if c in cols]))
    return _kept


def describe(v: np.ndarray, label: str = "") -> dict:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    q = np.percentile(v, [1, 25, 50, 75, 99])
    return {"label": label, "n": int(v.size), "min": float(v.min()), "q01": float(q[0]),
            "q25": float(q[1]), "median": float(q[2]), "q75": float(q[3]), "q99": float(q[4]),
            "max": float(v.max()), "mean": float(v.mean()), "std": float(v.std(ddof=1))}
