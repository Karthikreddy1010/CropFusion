"""Does the CDHW feature block improve point accuracy? Tested across models.

Why this exists. The pipeline contains two CDHW tests and they disagree:
objective O1 (`main.py:937`) reports dR2 = +0.0116 and the ablation
(`main.py:2035`) reports dR2 = -0.0079. Both use NeuralCQR, both use one seed,
and both differences are far smaller than the 0.30-0.46 seed swing that
`config.py` documents for that model. O1 nonetheless records
`hypothesis_supported: True` from a bare point difference, with none of the
dependence-aware inference used everywhere else.

What this does. Refits every available model on the same FIT partition with and
without the CDHW block, scores on TEST, and adds a year-level paired test -- the
unit of independence used throughout the rest of the project.

**Pre-committed reporting.** Every model and every seed that runs is written to
the output file, whatever the sign. The point is to find out whether the block
carries signal, not to find a model that says it does.

**What this cannot settle.** It measures the features as PREDICTORS. The paper
uses the CDHW event count as a pre-specified STRATIFICATION variable defining
exposure classes, which is a measurement question, not a prediction one. A null
result here leaves RO1-RO5 untouched; see `--construct-check`, which asks the
question that does matter: do the exposure classes separate yield outcomes?

Nothing here changes the feature set the pipeline uses. It is a reported
analysis, not a selection step.

Run:  python code/cdhw_contribution.py                  (trees only, ~1 min)
      python code/cdhw_contribution.py --neural-seeds 5 (adds NeuralCQR)
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
for p in (str(CODE_DIR), str(CODE_DIR / "diagnostics_loso")):
    if p not in sys.path:
        sys.path.insert(0, p)

import config as cfg  # noqa: E402
from scaling import fit_scaler, apply_scaling  # noqa: E402
from splitting import get_feature_target_arrays  # noqa: E402


def metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    err = y - pred
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "r2": float(1.0 - np.sum(err ** 2) / max(1e-9, ss_tot))}


def arm_arrays(frames, cols):
    """Scale a feature subset with a scaler fitted on FIT only, as main.py does."""
    scaler, _ = fit_scaler(frames["fit"], cols, scaler_type="robust")
    out = {}
    for name in ("fit", "test"):
        X, y = get_feature_target_arrays(apply_scaling(frames[name], cols, scaler, split_name=name), cols)
        out[name] = (X, y)
    return out


def tree_models(seed: int) -> Dict[str, Any]:
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    models = {
        "LightGBM": lgb.LGBMRegressor(**{**cfg.LGBM_PARAMS, "random_state": seed}),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=12,
                                              random_state=seed, n_jobs=-1),
        "Ridge": Ridge(alpha=1.0),
    }
    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = CatBoostRegressor(iterations=500, learning_rate=0.05,
                                               verbose=0, random_seed=seed)
    except ImportError:
        pass
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                                             random_state=seed)
    except ImportError:
        pass
    return models


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--neural-seeds", type=int, default=0)
    ap.add_argument("--tree-seeds", type=int, default=3)
    ap.add_argument("--construct-check", action="store_true", default=True)
    ap.add_argument("--tag", type=str, default="cdhw_contribution")
    args = ap.parse_args()

    import _fold_lab as lab
    data = lab.build_temporal()
    frames, feature_cols = data["frames"], data["feature_cols"]

    cdhw_cols = [c for c in list(cfg.CDHW_COLS) + ["CDHW_Veg_Severity", "CDHW_Silking_Severity",
                                                   "CDHW_GrainFill_Severity"] if c in feature_cols]
    non_cdhw = [c for c in feature_cols if c not in cdhw_cols]
    print(f"features: {len(feature_cols)} total, {len(cdhw_cols)} CDHW, {len(non_cdhw)} without")
    print(f"CDHW block: {cdhw_cols}")

    with_arm = arm_arrays(frames, feature_cols)
    without_arm = arm_arrays(frames, non_cdhw)
    trend_test = data["trend"]["test"]
    y_test_raw = data["y"]["test"]
    years_test = frames["test"]["Year"].values

    rows: List[Dict[str, Any]] = []
    seeds = [cfg.RANDOM_SEED, 7, 123, 2024, 3407]

    for seed in seeds[: args.tree_seeds]:
        for name, model in tree_models(seed).items():
            preds = {}
            for arm, arrays in (("with", with_arm), ("without", without_arm)):
                Xf, yf = arrays["fit"]
                Xt, _ = arrays["test"]
                yf_det = yf - data["trend"]["fit"]
                m = model.__class__(**model.get_params())
                m.fit(Xf, yf_det)
                preds[arm] = m.predict(Xt) + trend_test
            mw, mo = metrics(y_test_raw, preds["with"]), metrics(y_test_raw, preds["without"])
            rows.append({"model": name, "seed": seed,
                         "r2_with": mw["r2"], "r2_without": mo["r2"], "delta_r2": mw["r2"] - mo["r2"],
                         "rmse_with": mw["rmse"], "rmse_without": mo["rmse"],
                         "delta_rmse": mo["rmse"] - mw["rmse"],
                         "_se_with": ((y_test_raw - preds["with"]) ** 2).tolist(),
                         "_se_without": ((y_test_raw - preds["without"]) ** 2).tolist()})
            print(f"  {name:<14} seed {seed:>4}: dR2 {mw['r2'] - mo['r2']:+.4f}  "
                  f"(with {mw['r2']:.4f}, without {mo['r2']:.4f})", flush=True)

    if args.neural_seeds:
        import model_training as mt
        for seed in seeds[: args.neural_seeds]:
            preds = {}
            for arm, arrays in (("with", with_arm), ("without", without_arm)):
                Xf, yf = arrays["fit"]
                Xt, _ = arrays["test"]
                t0 = time.time()
                mdl = mt.train_neural_cqr_final(
                    Xf, yf - data["trend"]["fit"],
                    feature_cols=(feature_cols if arm == "with" else non_cdhw),
                    epochs=cfg.BASELINE_MAX_EPOCHS, seed=seed)
                p, _, _ = mt.predict_intervals(mdl, Xt)
                preds[arm] = p + trend_test
                print(f"    NeuralCQR seed {seed} arm {arm}: {time.time() - t0:.0f}s", flush=True)
            mw, mo = metrics(y_test_raw, preds["with"]), metrics(y_test_raw, preds["without"])
            rows.append({"model": "NeuralCQR", "seed": seed,
                         "r2_with": mw["r2"], "r2_without": mo["r2"], "delta_r2": mw["r2"] - mo["r2"],
                         "rmse_with": mw["rmse"], "rmse_without": mo["rmse"],
                         "delta_rmse": mo["rmse"] - mw["rmse"],
                         "_se_with": ((y_test_raw - preds["with"]) ** 2).tolist(),
                         "_se_without": ((y_test_raw - preds["without"]) ** 2).tolist()})
            print(f"  NeuralCQR      seed {seed:>4}: dR2 {mw['r2'] - mo['r2']:+.4f}", flush=True)

    # Year-level paired test on squared errors, per model (the inference O1 lacks).
    from dependence_aware_stats import year_level_paired_test
    inference = {}
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        se_with = np.mean([r["_se_with"] for r in sub], axis=0)
        se_without = np.mean([r["_se_without"] for r in sub], axis=0)
        inference[model] = year_level_paired_test(se_with, se_without, years_test,
                                                  label_a="with_cdhw", label_b="without_cdhw")

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    summary = (df.groupby("model")
               .agg(n_seeds=("seed", "count"), delta_r2_mean=("delta_r2", "mean"),
                    delta_r2_sd=("delta_r2", "std"), delta_r2_min=("delta_r2", "min"),
                    delta_r2_max=("delta_r2", "max"), r2_with_mean=("r2_with", "mean"))
               .reset_index())
    summary["straddles_zero"] = (summary.delta_r2_min < 0) & (summary.delta_r2_max > 0)
    summary["year_level_p"] = summary.model.map(
        lambda m: inference[m].get("paired_t_p_value"))

    out_dir = CODE_DIR.parent / "outputs_diagnostics" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{args.tag}_raw.csv", index=False)
    summary.to_csv(out_dir / f"{args.tag}_summary.csv", index=False)

    construct = {}
    if args.construct_check:
        te = frames["test"]
        col = "Year_Type" if "Year_Type" in te.columns else None
        if col:
            g = te.groupby(col)[cfg.PRIMARY_TARGET].agg(["count", "mean", "std"])
            construct = {str(k): {"n": int(v["count"]), "mean_yield": round(float(v["mean"]), 3),
                                  "sd": round(float(v["std"]), 3)} for k, v in g.iterrows()}

    (out_dir / f"{args.tag}.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_features_with": len(feature_cols), "n_features_without": len(non_cdhw),
        "cdhw_block": cdhw_cols,
        "reporting_rule": "every model and seed run is reported, whatever the sign",
        "scope": "point-accuracy contribution only; the exposure class is used by the "
                 "paper as a pre-specified stratification variable, which this does not test",
        "per_seed": df.to_dict("records"),
        "summary": summary.to_dict("records"),
        "year_level_inference": inference,
        "construct_check_yield_by_exposure_class": construct,
    }, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 86)
    print(f"{'model':<16}{'seeds':>6}{'dR2 mean':>11}{'sd':>9}{'min':>9}{'max':>9}{'year-level p':>14}")
    for _, r in summary.iterrows():
        p = r["year_level_p"]
        print(f"{r['model']:<16}{int(r['n_seeds']):>6}{r['delta_r2_mean']:>+11.4f}"
              f"{(r['delta_r2_sd'] or 0):>9.4f}{r['delta_r2_min']:>+9.4f}{r['delta_r2_max']:>+9.4f}"
              f"{(f'{p:.4f}' if p is not None else 'n/a'):>14}")
    print("=" * 86)
    if construct:
        print("\nConstruct check — yield by exposure class on TEST (the question that matters):")
        for k, v in construct.items():
            print(f"  {k:<10} n={v['n']:>5}  mean yield {v['mean_yield']:>6.2f} t/ha  (sd {v['sd']})")
    print(f"\nwritten: {out_dir / (args.tag + '_summary.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
