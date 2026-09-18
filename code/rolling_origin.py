"""Rolling-origin evaluation (audit §17 C2) — the paper's go/no-go test.

The central claim rests on one five-year window (2019–2023) in which the Extreme
exposure class is 175/202 rows from 2022–23 and 146/202 from Missouri and Iowa.
This harness repeats the whole evaluation with the origin rolled forward, so the
exposure gradient can be checked against many independent test years instead of
one window.

For each test year Y:

    FIT  = 1985 .. Y-5      DEV = Y-4, Y-3
    CAL  = Y-2, Y-1         TEST = Y

Everything downstream of the split — lags, preprocessor, FIT-only feature
selection, multicollinearity prune, county baseline, RobustScaler, FIT-only
detrending — is the project's own pipeline, reached by overriding the year
boundaries in config and calling the same builder the locked run uses. Nothing
here reads a held-out state, and no choice is made on test outcomes.

Calibrators compared at each origin: rolling split CP, group-conditional
(Mondrian) CP, standard ACI, SA-ACI, plus the trend + county-mean baseline.

**A note on the adaptive methods.** With TEST = one year, ACI and SA-ACI get a
single update step, so their rolling-origin numbers reflect their initial
calibration rather than their adaptation. Use --test-window to give them a
multi-year window; those windows overlap, so treat them as a secondary analysis.

Run:  python code/rolling_origin.py --first 2008 --last 2023
      python code/rolling_origin.py --first 2008 --last 2023 --model neural
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import config as cfg  # noqa: E402


# ─────────────────────────────────────────────────────────────

def coverage_metrics(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                     alpha: float) -> Dict[str, float]:
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    nominal = 1.0 - alpha
    winkler = width.copy()
    below, above = y < lo, y > hi
    winkler[below] += (2.0 / alpha) * (lo[below] - y[below])
    winkler[above] += (2.0 / alpha) * (y[above] - hi[above])
    return {
        "n": int(len(y)),
        "picp": float(covered.mean()) if len(y) else float("nan"),
        "ace": float(covered.mean() - nominal) if len(y) else float("nan"),
        "mpiw": float(width.mean()) if len(y) else float("nan"),
        "winkler": float(winkler.mean()) if len(y) else float("nan"),
    }


def point_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    err = y - pred
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "r2": float(1.0 - np.sum(err ** 2) / max(1e-9, ss_tot)),
    }


def exposure_class(frame: pd.DataFrame) -> np.ndarray:
    """CDHW exposure class, using the project's own thresholds.

    Prefers an existing Year_Type / CDHW_Exposure_Class column; otherwise derives
    it from CDHW_Event_Count exactly as feature engineering does.
    """
    for col in ("CDHW_Exposure_Class", "Year_Type"):
        if col in frame.columns:
            return frame[col].astype(str).values
    if "CDHW_Event_Count" in frame.columns:
        n = frame["CDHW_Event_Count"].fillna(0).values
        out = np.where(n >= cfg.EXTREME_EVENT_COUNT_THRESHOLD, "Extreme",
                       np.where(n >= cfg.MODERATE_EVENT_COUNT_THRESHOLD, "Moderate", "Normal"))
        return out.astype(str)
    return np.full(len(frame), "Unknown", dtype=object)


class YearWindow:
    """Temporarily set the four-way split boundaries for one origin."""

    KEYS = ("FIT_YEARS", "DEV_YEARS", "CAL_YEARS", "TEST_YEARS")

    def __init__(self, test_year: int, test_window: int = 1, fit_start: int = 1985) -> None:
        self.windows = {
            "FIT_YEARS": (fit_start, test_year - 5),
            "DEV_YEARS": (test_year - 4, test_year - 3),
            "CAL_YEARS": (test_year - 2, test_year - 1),
            "TEST_YEARS": (test_year, test_year + test_window - 1),
        }
        self._saved: Dict[str, Any] = {}

    def __enter__(self) -> Dict[str, Any]:
        for k in self.KEYS:
            self._saved[k] = getattr(cfg, k)
            setattr(cfg, k, self.windows[k])
        return self.windows

    def __exit__(self, *exc: Any) -> None:
        for k, v in self._saved.items():
            setattr(cfg, k, v)


# ─────────────────────────────────────────────────────────────

def run_origin(test_year: int, args: argparse.Namespace,
               df_all: pd.DataFrame | None = None) -> List[Dict[str, Any]]:
    """Build one origin, fit, calibrate, and return its metric rows.

    ``df_all`` is the engineered frame, built once by the caller. It is trimmed
    to ``Year <= last test year`` before the split: at origin Y the pipeline must
    not even see later years, and the master loader enforces exactly that by
    refusing rows outside the configured window.
    """
    # The partition builder lives in the diagnostics package and reproduces
    # main.py's temporal data path exactly; importing it here keeps one
    # definition of "how a partition is built" and sends output to
    # outputs_diagnostics rather than overwriting the locked run.
    lab_dir = CODE_DIR / "diagnostics_loso"
    if str(lab_dir) not in sys.path:
        sys.path.insert(0, str(lab_dir))
    import _fold_lab as lab
    import baselines as bl
    import aci_calibrator as aci

    alpha = 1.0 - cfg.NOMINAL_COVERAGE
    rows: List[Dict[str, Any]] = []

    with YearWindow(test_year, args.test_window, args.fit_start) as win:
        t0 = time.time()
        data = lab.build_temporal()
        frames, X, y_det, trend = data["frames"], data["X"], data["y_detrended"], data["trend"]
        y_raw = data["y"]

        if min(len(y_raw["cal"]), len(y_raw["test"]), len(y_raw["dev"])) == 0:
            return [{"test_year": test_year, "status": "empty partition", **win}]

        # ── model ────────────────────────────────────────────────────
        if args.model == "lgbm":
            import lightgbm as lgb
            point = lgb.LGBMRegressor(**cfg.LGBM_PARAMS).fit(X["fit"], y_det["fit"])
            qp = dict(objective="quantile", n_estimators=500,
                      learning_rate=cfg.LGBM_PARAMS["learning_rate"],
                      random_state=cfg.RANDOM_SEED, verbose=-1)
            lo_m = lgb.LGBMRegressor(alpha=alpha / 2, **qp).fit(X["fit"], y_det["fit"])
            hi_m = lgb.LGBMRegressor(alpha=1 - alpha / 2, **qp).fit(X["fit"], y_det["fit"])
            pred = {k: point.predict(X[k]) for k in ("cal", "test")}
            qlo = {k: lo_m.predict(X[k]) for k in ("cal", "test")}
            qhi = {k: hi_m.predict(X[k]) for k in ("cal", "test")}
        else:
            import model_training as mt
            model = mt.train_neural_cqr(
                X["fit"], y_det["fit"], X["dev"], y_det["dev"],
                feature_cols=data["feature_cols"], epochs=cfg.BASELINE_MAX_EPOCHS,
                batch_size=cfg.BATCH_SIZE, lr=cfg.LEARNING_RATE,
                hidden_dims=cfg.NEURAL_CQR_HIDDEN_DIMS,
                early_stopping_mode=cfg.EARLY_STOPPING_MODE,
                patience=cfg.EARLY_STOPPING_PATIENCE, seed=cfg.RANDOM_SEED,
            )
            pred, qlo, qhi = {}, {}, {}
            for k in ("cal", "test"):
                pred[k], qlo[k], qhi[k] = mt.predict_intervals(model, X[k])

        # back to the real yield scale
        for k in ("cal", "test"):
            pred[k] = pred[k] + trend[k]
            qlo[k] = qlo[k] + trend[k]
            qhi[k] = qhi[k] + trend[k]

        y_cal, y_test = y_raw["cal"], y_raw["test"]
        grp_cal = exposure_class(frames["cal"])
        grp_test = exposure_class(frames["test"])
        years_test = frames["test"]["Year"].values

        # ── calibrators ──────────────────────────────────────────────
        variants: Dict[str, Any] = {}

        lo, hi, _ = bl.rolling_static_cp(y_cal, qlo["cal"], qhi["cal"],
                                         qlo["test"], qhi["test"], alpha=alpha)
        variants["rolling_static_cp"] = (lo, hi)

        lo, hi, gmeta = bl.group_conditional_cp(
            y_cal, qlo["cal"], qhi["cal"], grp_cal,
            qlo["test"], qhi["test"], grp_test, alpha=alpha)
        variants["group_conditional_cp"] = (lo, hi)

        sev_cal = frames["cal"].get("CDHW_Event_Count", pd.Series(np.zeros(len(y_cal)))).fillna(0).values
        sev_test = frames["test"].get("CDHW_Event_Count", pd.Series(np.zeros(len(y_test)))).fillna(0).values

        r = aci.standard_aci(y_cal, qlo["cal"], qhi["cal"], y_test, qlo["test"], qhi["test"],
                             pred["test"], years_test, alpha=alpha)
        variants["standard_aci"] = (r.q_lo, r.q_hi)

        r = aci.severity_aware_adaptive_conformal(
            y_cal, qlo["cal"], qhi["cal"], y_test, qlo["test"], qhi["test"],
            pred["test"], years_test, alpha=alpha,
            cdhw_severity_test=sev_test, cdhw_severity_cal=sev_cal)
        variants["sa_aci"] = (r.q_lo, r.q_hi)

        # ── metrics, overall and by exposure class ───────────────────
        base_pred = bl.trend_county_mean(frames["fit"], frames["test"])
        base_pm = point_metrics(y_test, base_pred)
        model_pm = point_metrics(y_test, pred["test"])

        for method, (lo_m_, hi_m_) in variants.items():
            common = {
                "test_year": test_year, "model": args.model, "method": method,
                "fit_years": f"{win['FIT_YEARS'][0]}-{win['FIT_YEARS'][1]}",
                "dev_years": f"{win['DEV_YEARS'][0]}-{win['DEV_YEARS'][1]}",
                "cal_years": f"{win['CAL_YEARS'][0]}-{win['CAL_YEARS'][1]}",
                "n_fit": int(len(y_det["fit"])), "n_cal": int(len(y_cal)),
                "model_rmse": round(model_pm["rmse"], 4), "model_r2": round(model_pm["r2"], 4),
                "model_bias": round(model_pm["bias"], 4),
                "baseline_trend_county_rmse": round(base_pm["rmse"], 4),
                "baseline_trend_county_r2": round(base_pm["r2"], 4),
                "seconds": round(time.time() - t0, 1),
            }
            rows.append({**common, "stratum": "ALL", **coverage_metrics(y_test, lo_m_, hi_m_, alpha)})
            for g in np.unique(grp_test):
                m = grp_test == g
                rows.append({**common, "stratum": str(g),
                             **coverage_metrics(y_test[m], lo_m_[m], hi_m_[m], alpha)})

        rows.append({"test_year": test_year, "model": args.model, "method": "_group_cp_meta",
                     "stratum": "ALL", "n": int(len(y_test)),
                     "group_thresholds": json.dumps({k: v["threshold"] for k, v in gmeta["groups"].items()}),
                     "group_fallbacks": json.dumps({k: v["fell_back_to_pooled"] for k, v in gmeta["groups"].items()})})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2008, help="first test year")
    ap.add_argument("--last", type=int, default=2023, help="last test year")
    ap.add_argument("--fit-start", type=int, default=1985)
    ap.add_argument("--test-window", type=int, default=1,
                    help="test years per origin; >1 gives ACI room to adapt but overlaps")
    ap.add_argument("--model", choices=("lgbm", "neural"), default="lgbm")
    ap.add_argument("--tag", type=str, default="rolling_origin")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (CODE_DIR.parent / "outputs_diagnostics" / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    part_path = out_dir / f"{args.tag}_raw.csv"

    # Engineer once. Feature engineering is per county-year and does not use the
    # split boundaries; everything that must be fitted on FIT only (lags,
    # preprocessor, feature selection, scaler, detrending) happens per origin
    # inside build_temporal.
    lab_dir = CODE_DIR / "diagnostics_loso"
    if str(lab_dir) not in sys.path:
        sys.path.insert(0, str(lab_dir))
    import _fold_lab as lab
    print("Engineering features once for all origins...", flush=True)
    t_eng = time.time()
    df_all = lab.engineered_df()
    print(f"  {len(df_all)} rows, {int(df_all['Year'].min())}-{int(df_all['Year'].max())} "
          f"in {time.time() - t_eng:.0f}s", flush=True)

    all_rows: List[Dict[str, Any]] = []
    for year in range(args.first, args.last + 1):
        print(f"\n=== origin: test year {year} ===", flush=True)
        try:
            rows = run_origin(year, args, df_all)
        except Exception as exc:  # one bad origin must not lose the rest
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            rows = [{"test_year": year, "model": args.model, "method": "_error",
                     "stratum": "ALL", "error": f"{type(exc).__name__}: {exc}"}]
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(part_path, index=False)  # checkpoint every origin
        for r in rows:
            if r.get("stratum") == "ALL" and r.get("method") not in ("_group_cp_meta", "_error"):
                print(f"  {r['method']:<22} PICP={r['picp']:.3f} MPIW={r['mpiw']:.3f} "
                      f"Winkler={r['winkler']:.3f} | model R2={r.get('model_r2')} "
                      f"baseline R2={r.get('baseline_trend_county_r2')}", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(part_path, index=False)

    real = df[~df.method.isin(["_group_cp_meta", "_error"])].copy()
    if len(real):
        summary = (real.groupby(["method", "stratum"])
                   .agg(n_origins=("test_year", "nunique"), rows=("n", "sum"),
                        picp_mean=("picp", "mean"), picp_sd=("picp", "std"),
                        mpiw_mean=("mpiw", "mean"), winkler_mean=("winkler", "mean"))
                   .reset_index())
        summary.to_csv(out_dir / f"{args.tag}_summary.csv", index=False)
        print("\n" + "=" * 96)
        print(f"{'method':<22}{'stratum':<10}{'origins':>8}{'rows':>8}{'PICP':>9}{'sd':>8}{'MPIW':>9}{'Winkler':>10}")
        for _, r in summary.iterrows():
            print(f"{r['method']:<22}{r['stratum']:<10}{r['n_origins']:>8}{int(r['rows']):>8}"
                  f"{r['picp_mean']:>9.3f}{(r['picp_sd'] or 0):>8.3f}{r['mpiw_mean']:>9.3f}"
                  f"{r['winkler_mean']:>10.3f}")
        print("=" * 96)
        print(f"written: {out_dir / (args.tag + '_summary.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
