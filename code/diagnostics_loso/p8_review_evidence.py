"""p8_review_evidence.py - evidence for the peer-review audit, from the CURRENT run only.

1. Conditional (regime-stratified) coverage for all six conformal methods on the temporal
   test partition, joined to regime variables by POSITION after verifying that the rebuilt
   test partition matches predictions.csv row-for-row (county names are not unique across
   states, so a name join would be unsafe).
2. Naive baselines that any yield-ML paper must beat:
      temporal:  FIT-only linear trend;  trend + FIT county mean anomaly (county_baseline)
      LOSO:      five-state FIT-only trend (the held-out state contributes nothing)
Read-only: nothing is trained except closed-form trends, nothing is written to outputs_master.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _fold_lab import engineered_df, cfg, DIAG_OUT
from splitting import temporal_split_4way, loso_cv_folds
from evaluation import r_squared, rmse, mae

PRED = cfg.PROJECT_ROOT / "outputs_master" / "predictions" / "predictions.csv"
LOSO_PRED = cfg.PROJECT_ROOT / "outputs_master" / "predictions" / "loso_predictions.csv"


def picp(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))


def winkler(y, lo, hi, alpha=0.10):
    w = hi - lo
    pen = np.where(y < lo, 2 / alpha * (lo - y), 0) + np.where(y > hi, 2 / alpha * (y - hi), 0)
    return float(np.mean(w + pen))


def main():
    df = engineered_df()
    out = {}

    # ---------------- temporal partitions ----------------
    fit_df, dev_df, cal_df, test_df = temporal_split_4way(df, target=cfg.PRIMARY_TARGET)
    test_df = test_df.reset_index(drop=True)

    pred = pd.read_csv(PRED)
    methods = list(dict.fromkeys(pred["Conformal_Method"]))
    blocks = {m: pred[pred.Conformal_Method == m].reset_index(drop=True) for m in methods}
    ref = blocks[methods[0]]

    align = {
        "n_test_rebuilt": int(len(test_df)),
        "n_rows_per_method": {m: int(len(b)) for m, b in blocks.items()},
        "year_sequence_identical": bool((ref["Year"].values == test_df["Year"].values).all()),
        "county_name_sequence_identical": bool((ref["County"].astype(str).values
                                                == test_df["County_Name"].astype(str).values).all()),
        "observed_yield_max_abs_diff": float(np.max(np.abs(ref["Observed_Yield"].values
                                                           - test_df[cfg.PRIMARY_TARGET].values))),
        "all_methods_same_observed": bool(all(np.allclose(b["Observed_Yield"], ref["Observed_Yield"])
                                              for b in blocks.values())),
    }
    align["verified"] = bool(align["year_sequence_identical"] and align["county_name_sequence_identical"]
                             and align["observed_yield_max_abs_diff"] < 1e-3 and align["all_methods_same_observed"])
    out["alignment"] = align
    print("ALIGNMENT", align, flush=True)
    if not align["verified"]:
        raise SystemExit("predictions.csv does not align with the rebuilt test partition; refusing to join")

    y = test_df[cfg.PRIMARY_TARGET].values
    sev = test_df["CDHW_Severity_Score"].values
    pos_sev = sev[sev > 0]
    sev_cut = float(np.median(pos_sev)) if len(pos_sev) else 0.0
    regimes = {
        "all": np.ones(len(y), bool),
        "year": test_df["Year"].astype(int).astype(str).values,
        "state": test_df["State"].values,
        "year_type": test_df["Year_Type"].astype(str).values,
        "cdhw": np.where(sev <= 0, "none", np.where(sev <= sev_cut, "moderate", "severe")),
        "drought_spei30_lt_-1": np.where(test_df["SPEI_30_min"].values < -1, "drought", "no_drought"),
        "heat_days_gt35": np.where(test_df["Tmax_Days_Above_35"].values > 0, "heat", "no_heat"),
        "enso": test_df["ENSO_Phase"].astype(str).values,
    }
    out["cdhw_severe_cut_median_of_positive"] = sev_cut

    rows = []
    for m, b in blocks.items():
        lo, hi = b["Lower_Interval"].values, b["Upper_Interval"].values
        for axis, lab in regimes.items():
            if axis == "all":
                groups = {"all": lab}
            else:
                groups = {g: lab == g for g in pd.unique(lab)}
            for g, mask in groups.items():
                if mask.sum() < 20:
                    continue
                rows.append(dict(method=m, axis=axis, stratum=str(g), n=int(mask.sum()),
                                 picp=round(picp(y[mask], lo[mask], hi[mask]), 4),
                                 mpiw=round(float(np.mean(hi[mask] - lo[mask])), 4),
                                 winkler=round(winkler(y[mask], lo[mask], hi[mask]), 4),
                                 bias=round(float(np.mean(b["Predicted_Yield"].values[mask] - y[mask])), 4)))
    strat = pd.DataFrame(rows)
    strat.to_csv(DIAG_OUT / "reports" / "p8_conditional_coverage_current_run.csv", index=False)

    marg = strat[strat.axis == "all"].set_index("method")
    marg["ace"] = (marg["picp"] - 0.90).abs()
    worst = (strat[strat.axis != "all"].groupby("method")
             .apply(lambda g: g.loc[g.picp.idxmin(), ["axis", "stratum", "n", "picp"]], include_groups=False))
    summary = marg[["picp", "ace", "mpiw", "winkler"]].join(worst.add_prefix("worst_"))
    summary["rank_marginal_ace"] = summary["ace"].rank(method="min")
    summary["rank_marginal_winkler"] = summary["winkler"].rank(method="min")
    summary["rank_worst_stratum_picp"] = summary["worst_picp"].rank(ascending=False, method="min")
    out["method_summary"] = summary.reset_index().to_dict(orient="records")
    print(summary.to_string(), flush=True)

    per_axis_worst = (strat[strat.axis != "all"].groupby(["axis", "method"])["picp"].min().unstack())
    out["worst_picp_by_axis"] = per_axis_worst.round(4).to_dict()
    print("\nWORST-STRATUM PICP BY AXIS\n", per_axis_worst.round(4).to_string(), flush=True)

    # ---------------- naive temporal baselines ----------------
    fv = fit_df[fit_df[cfg.PRIMARY_TARGET].notna()]
    coef = np.polyfit(fv["Year"].values, fv[cfg.PRIMARY_TARGET].values, 1)
    trend_te = np.polyval(coef, test_df["Year"].values)
    anom = fv[cfg.PRIMARY_TARGET].values - np.polyval(coef, fv["Year"].values)
    cb = pd.Series(anom, index=fv["GEOID"].values).groupby(level=0).mean()
    cb_te = test_df["GEOID"].map(cb).fillna(0.0).values
    ens = blocks["sa_aci"]["Predicted_Yield"].values if "sa_aci" in blocks else ref["Predicted_Yield"].values
    out["temporal_baselines"] = {
        "trend_only": dict(r2=round(float(r_squared(y, trend_te)), 4), rmse=round(float(rmse(y, trend_te)), 4)),
        "trend_plus_county_mean_anomaly": dict(r2=round(float(r_squared(y, trend_te + cb_te)), 4),
                                               rmse=round(float(rmse(y, trend_te + cb_te)), 4),
                                               mae=round(float(mae(y, trend_te + cb_te)), 4)),
        "shipped_ensemble": dict(r2=round(float(r_squared(y, ens)), 4), rmse=round(float(rmse(y, ens)), 4),
                                 mae=round(float(mae(y, ens)), 4)),
        "trend_coef": [round(float(c), 5) for c in coef],
    }
    print("\nTEMPORAL BASELINES", json.dumps(out["temporal_baselines"], indent=1), flush=True)

    # ---------------- naive LOSO baseline ----------------
    lp = pd.read_csv(LOSO_PRED)
    loso_rows = []
    for state, fit_f, dev_f, cal_f, test_f in loso_cv_folds(df, return_cal=True):
        c = np.polyfit(fit_f["Year"].values, fit_f[cfg.PRIMARY_TARGET].values, 1)
        t = test_f.reset_index(drop=True)
        base = np.polyval(c, t["Year"].values)
        mp = lp[lp.held_out_state == state].reset_index(drop=True)
        ok = (len(mp) == len(t) and (mp["year"].values == t["Year"].values).all()
              and np.allclose(mp["y_true"].values, t[cfg.PRIMARY_TARGET].values, atol=1e-3))
        yt = t[cfg.PRIMARY_TARGET].values
        loso_rows.append(dict(state=state, aligned=bool(ok),
                              trend_only_r2=round(float(r_squared(yt, base)), 4),
                              trend_only_rmse=round(float(rmse(yt, base)), 4),
                              model_r2=round(float(r_squared(yt, mp["y_pred"].values)), 4) if ok else None,
                              model_rmse=round(float(rmse(yt, mp["y_pred"].values)), 4) if ok else None))
    lt = pd.DataFrame(loso_rows)
    out["loso_baselines"] = lt.to_dict(orient="records")
    out["loso_baselines_macro"] = {"trend_only_r2": round(float(lt.trend_only_r2.mean()), 4),
                                   "model_r2": round(float(lt.model_r2.mean()), 4)}
    print("\nLOSO BASELINES\n", lt.to_string(index=False), "\n", out["loso_baselines_macro"], flush=True)

    (DIAG_OUT / "reports" / "p8_review_evidence.json").write_text(json.dumps(out, indent=2, default=str),
                                                                   encoding="utf-8")
    print("wrote", DIAG_OUT / "reports" / "p8_review_evidence.json")


if __name__ == "__main__":
    main()
