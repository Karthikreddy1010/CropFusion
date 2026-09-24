"""County fixed-effects degree-day panel — the econometric baseline (audit C4).

The paper already compares against trend + county mean, which is the baseline the
conformal literature uses. It is not the baseline an agricultural economist or an
agrometeorologist would ask for. Theirs is the reduced-form panel that has framed
crop-climate estimation since Schlenker & Roberts (2009, PNAS): county fixed
effects absorb time-invariant soil and management, a year term carries technology,
and weather enters through a beneficial growing-degree-day term, a harmful
extreme-heat term and a concave precipitation response.

    yield_ct = a_c + b1*GDD + b2*heat_days + b3*precip + b4*precip^2 + b5*year + e

**What this is not.** The canonical specification uses the full daily temperature
distribution binned at 1-3 degree intervals. The engineered frame carries growing-
season aggregates, not daily distributions, so the temperature response here is
two-piece (accumulated GDD, plus days above 35 C) rather than a flexible bin
schedule. That is the standard reduced form, but it is less flexible than the
published specification and the results section says so.

Estimated by the within transformation: county means are removed from both sides,
so 579 county effects cost nothing and are recovered afterwards as residual means.
County effects come from FIT rows only, and a county absent from FIT falls back to
the global intercept — the leave-one-state-out case, where no test county is seen.

Verified in code/diagnostics_loso/verify_fe_baseline.py.

Run:  python code/fe_degree_day_baseline.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
for p in (str(CODE_DIR), str(CODE_DIR / "diagnostics_loso")):
    if p not in sys.path:
        sys.path.insert(0, p)

import config as cfg  # noqa: E402

GDD = "GDD_Accumulated"
HEAT = "Tmax_Days_Above_35"
PRECIP = "Precip_growseason_mm"
TERMS = ("gdd", "heat", "precip", "precip_sq", "year")


@dataclass
class FEPanel:
    """A fitted panel. `predict` is the only thing callers need."""
    coefficients: Dict[str, float]
    county_effects: Dict[Any, float]
    global_intercept: float
    year_origin: float
    r2_in_sample: float
    precip_quadratic: bool = True
    n_fit: int = 0
    n_counties: int = 0
    terms: List[str] = field(default_factory=list)

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        cols = [df[GDD].values, df[HEAT].values, df[PRECIP].values]
        if self.precip_quadratic:
            cols.append(df[PRECIP].values ** 2)
        cols.append(df["Year"].values - self.year_origin)
        return np.column_stack([np.asarray(c, dtype=float) for c in cols])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._design(df)
        beta = np.array([self.coefficients[t] for t in self.terms], dtype=float)
        # A county with no FIT rows gets the global intercept. Returning NaN here
        # would be "honest" and useless: every LOSO test county is unseen, and the
        # comparison the baseline exists for would silently have no baseline.
        eff = np.array([self.county_effects.get(g, self.global_intercept)
                        for g in df["GEOID"].values], dtype=float)
        return eff + X @ beta


def fit_fe_panel(fit_df: pd.DataFrame, target: Optional[str] = None,
                 precip_quadratic: bool = True) -> FEPanel:
    """Within-county OLS on FIT rows. Nothing outside `fit_df` is read."""
    target = target or cfg.PRIMARY_TARGET
    need = [GDD, HEAT, PRECIP, "Year", "GEOID", target]
    missing = [c for c in need if c not in fit_df.columns]
    if missing:
        raise KeyError(f"FE panel needs {missing}")
    d = fit_df.dropna(subset=need).copy()
    if d.empty:
        raise ValueError("no complete FIT rows for the FE panel")

    terms = ["gdd", "heat", "precip"] + (["precip_sq"] if precip_quadratic else []) + ["year"]
    year_origin = float(d["Year"].mean())
    cols = [d[GDD].values, d[HEAT].values, d[PRECIP].values]
    if precip_quadratic:
        cols.append(d[PRECIP].values ** 2)
    cols.append(d["Year"].values - year_origin)
    X = np.column_stack([np.asarray(c, dtype=float) for c in cols])
    y = d[target].values.astype(float)

    # within transformation: subtract county means from both sides
    codes, uniq = pd.factorize(d["GEOID"].values)
    n_c = len(uniq)
    counts = np.bincount(codes, minlength=n_c).astype(float)
    y_bar = np.bincount(codes, weights=y, minlength=n_c) / counts
    X_bar = np.column_stack([np.bincount(codes, weights=X[:, j], minlength=n_c) / counts
                             for j in range(X.shape[1])])
    beta, *_ = np.linalg.lstsq(X - X_bar[codes], y - y_bar[codes], rcond=None)

    # county effects are the residual means; the global intercept serves unseen counties
    eff = y_bar - X_bar @ beta
    resid = y - (eff[codes] + X @ beta)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return FEPanel(
        coefficients={t: float(b) for t, b in zip(terms, beta)},
        county_effects={g: float(e) for g, e in zip(uniq, eff)},
        global_intercept=float(np.average(eff, weights=counts)),
        year_origin=year_origin,
        r2_in_sample=float(1.0 - np.sum(resid ** 2) / max(ss_tot, 1e-12)),
        precip_quadratic=precip_quadratic,
        n_fit=int(len(d)), n_counties=int(n_c), terms=terms,
    )


def metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    err = y - p
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
            "r2": float(1.0 - np.sum(err ** 2) / max(float(np.sum((y - y.mean()) ** 2)), 1e-12))}


def main() -> int:
    """Score the FE panel on the locked split and on every rolling origin."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import _fold_lab as lab
    import baselines as bl

    df = lab.engineered_df()
    tgt = cfg.PRIMARY_TARGET
    out: Dict[str, Any] = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "specification": "county FE + GDD + days>35C + precip + precip^2 + year",
                           "caveat": ("growing-season aggregates, not a flexible daily degree-day "
                                      "bin schedule; see the module docstring"),
                           "source": "code/fe_degree_day_baseline.py"}

    # locked temporal split
    # trend_county_mean runs polyfit on the target, so NaN yields poison its
    # coefficients; fit_fe_panel drops them internally but the baseline does not.
    fit = df[(df.Year >= 1985) & (df.Year <= 2013)].dropna(subset=[tgt])
    test = df[(df.Year >= 2019) & (df.Year <= 2023)].dropna(subset=[tgt])
    m = fit_fe_panel(fit, target=tgt)
    mt = metrics(test[tgt].values.astype(float), m.predict(test))
    base = bl.trend_county_mean(fit, test)
    bt = metrics(test[tgt].values.astype(float), base)
    out["locked_temporal"] = {"n_fit": m.n_fit, "n_counties": m.n_counties, "n_test": int(len(test)),
                              "fe_panel": {k: round(v, 4) for k, v in mt.items()},
                              "trend_county_mean": {k: round(v, 4) for k, v in bt.items()},
                              "coefficients": {k: round(v, 8) for k, v in m.coefficients.items()},
                              "r2_in_sample": round(m.r2_in_sample, 4)}
    print(f"locked TEST 2019-23: FE R2 {mt['r2']:.4f} (RMSE {mt['rmse']:.4f}) vs "
          f"trend+county {bt['r2']:.4f}")
    print("coefficients:", {k: round(v, 6) for k, v in m.coefficients.items()})

    # rolling origins, matching code/rolling_origin.py
    rows = []
    for Y in range(2008, 2024):
        f = df[(df.Year >= 1985) & (df.Year <= Y - 5)].dropna(subset=[tgt])
        t = df[df.Year == Y].dropna(subset=[tgt])
        if not len(f) or not len(t):
            continue
        try:
            mm = fit_fe_panel(f, target=tgt)
            r = metrics(t[tgt].values.astype(float), mm.predict(t))
            rb = metrics(t[tgt].values.astype(float), bl.trend_county_mean(f, t))
            rows.append({"test_year": Y, "n_test": int(len(t)),
                         "fe_r2": round(r["r2"], 4), "fe_rmse": round(r["rmse"], 4),
                         "trend_county_r2": round(rb["r2"], 4)})
        except Exception as exc:                                   # noqa: BLE001
            rows.append({"test_year": Y, "error": f"{type(exc).__name__}: {exc}"})
    ro = pd.DataFrame(rows)
    out["rolling_origin"] = ro.to_dict("records")
    ok = ro[ro.get("fe_r2").notna()] if "fe_r2" in ro else ro.iloc[0:0]
    if len(ok):
        out["rolling_origin_summary"] = {
            "n_origins": int(len(ok)), "fe_r2_mean": round(float(ok.fe_r2.mean()), 4),
            "trend_county_r2_mean": round(float(ok.trend_county_r2.mean()), 4),
            "fe_beats_trend_county": int((ok.fe_r2 > ok.trend_county_r2).sum())}
        print(f"\nrolling origins: FE mean R2 {ok.fe_r2.mean():.4f} vs "
              f"trend+county {ok.trend_county_r2.mean():.4f} "
              f"({int((ok.fe_r2 > ok.trend_county_r2).sum())}/{len(ok)} origins)")

    d = Path(cfg.PROJECT_ROOT) / "outputs_diagnostics" / "reports"
    d.mkdir(parents=True, exist_ok=True)
    (d / "fe_degree_day_baseline.json").write_text(json.dumps(out, indent=2, default=str),
                                                   encoding="utf-8")
    ro.to_csv(d / "fe_degree_day_baseline_by_origin.csv", index=False)
    print(f"\nwritten: {d / 'fe_degree_day_baseline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
