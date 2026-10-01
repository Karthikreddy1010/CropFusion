"""
07_spi_spei_cdhw.py -- daily PRISM SPI, PRISM-NLDAS SPEI and PRISM-NLDAS CDHW (county level).

Inputs : intermediate/prism_county_daily/*.parquet, intermediate/nldas_county_daily/*.parquet
Outputs: intermediate/indices/county_daily_indices_1985_2023.parquet
         audits/spi_spei_fit_diagnostics.csv, audits/spi_spei_calibration_check.csv

METHOD (documented in metadata/master_data_provenance.json)
- Accumulation first, then distribution fit, then normal transform (never averaging daily SPI).
  Trailing windows inclusive of day t: SPI-1 = 30 d, SPI-3 = 90 d, SPI-6 = 180 d, SPI-12 = 365 d of PRISM ppt;
  SPEI-30 = 30 d and SPEI-90 = 90 d of D = PRISM ppt - NLDAS FAO-56 ET0.
  A window with any missing day is missing (no partial sums).
- Calibration period 1985-2013 (locked FIT years; user decision 2026-09-14) -> no DEV/CAL/TEST information.
- Day-of-year slots: 365 (29 Feb shares the 28 Feb slot; later leap-year days shift by one). Each county x slot
  is fitted on all calibration-period values whose slot lies within +/-7 days (circular), i.e. ~29 x 15 samples.
- SPI: two-parameter gamma, maximum-likelihood (Thom 1958 estimator) on positive accumulations; zeros
  (accumulation < 0.01 mm) handled with the mixed distribution H(x) = q + (1-q) G(x), q = zero fraction,
  and H(0) = q/2 (centre of the zero probability mass). SPI = Phi^-1(H).
- SPEI: log-logistic family fitted as the generalized logistic distribution (GLO, Hosking & Wallis 1997) from
  unbiased probability-weighted moments -- the implementation of the SPEI R package (Begueria et al. 2014:
  lmomco::pwm.ub + parglo). L-moments l1=b0, l2=2b1-b0, l3=6b2-6b1+b0, t3=l3/l2; k=-t3,
  alpha = l2 sin(k pi)/(k pi), xi = l1 - alpha(1/k - pi/sin(k pi));
  F(x) = 1/(1+exp(-y)), y = -log(1 - k(x-xi)/alpha)/k (y=(x-xi)/alpha when k=0). For t3 > 0 this is exactly the
  3-parameter log-logistic of Vicente-Serrano et al. (2010); unlike that closed form it also fits negatively skewed
  water-balance samples (t3 < 0, common for summer 30/90-day D), which otherwise fail (audit: 2026-09-15, 19,439
  SPEI-30 and 72,996 SPEI-90 county-slot fits failed with the closed form, all with t3 < 0).
  Invalid fits (alpha <= 0, |k| >= 1, n < 30) -> index missing for that county-slot (recorded in diagnostics).
- Probabilities are bounded to [0.001, 0.999] -> indices in [-3.09, 3.09] (climate_indices convention).
- CDHW (project definition, methodology section 4.1): day t is CDHW iff SPEI-30_PRISM-NLDAS(t) < -1.0 AND
  PRISM Tmax(t) > 35 C. Daily joint severity = |SPEI-30| x (Tmax - 35) on CDHW days, 0 otherwise.
  Daily heat exceedance = max(Tmax - 35, 0); daily drought severity = |SPEI-30| if SPEI-30 < -1 else 0.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.special import gamma as G
from scipy.special import gammainc, ndtri

from md_common import AUDITS, INTER, SPEI_THRESHOLD, TMAX_THRESHOLD_C, get_logger, storage_guard

log = get_logger("07_spi_spei_cdhw")
OUT = INTER / "indices"
OUT.mkdir(exist_ok=True)
CAL_YEARS = (1985, 2013)
HALF_WIN = 7
P_LO, P_HI = 0.001, 0.999
ZERO_MM = 0.01
SPI_WINDOWS = {"spi1_prism": 30, "spi3_prism": 90, "spi6_prism": 180, "spi12_prism": 365}
SPEI_WINDOWS = {"spei30_prism_nldas": 30, "spei90_prism_nldas": 90}


def load_panel():
    p = pd.concat([pd.read_parquet(f) for f in sorted((INTER / "prism_county_daily").glob("prism_county_daily_*.parquet"))])
    n = pd.concat([pd.read_parquet(f, columns=["county_fips", "date", "nldas_et0"])
                   for f in sorted((INTER / "nldas_county_daily").glob("nldas_county_daily_*.parquet"))])
    dates = pd.date_range("1985-01-01", "2023-12-31")
    counties = np.array(sorted(p.county_fips.unique()))
    def to_arr(df, col):
        piv = df.pivot(index="date", columns="county_fips", values=col).reindex(index=dates, columns=counties)
        return piv.to_numpy(dtype=np.float64)
    arrs = {"ppt": to_arr(p, "prism_ppt"), "tmax": to_arr(p, "prism_tmax"), "et0": to_arr(n, "nldas_et0")}
    return dates, counties, arrs


def rolling_sum_strict(x, w):
    """Trailing w-day sum; NaN if any day in the window is missing or t < w-1."""
    valid = np.isfinite(x)
    cs = np.cumsum(np.where(valid, x, 0.0), axis=0)
    cn = np.cumsum(valid, axis=0)
    out = np.full_like(x, np.nan)
    s = cs[w - 1:].copy()
    c = cn[w - 1:].copy()
    s[1:] -= cs[:-w]
    c[1:] -= cn[:-w]
    res = np.where(c == w, s, np.nan)
    out[w - 1:] = res
    return out


def doy_slots(dates):
    doy = dates.dayofyear.values.copy()
    leap = dates.is_leap_year
    slot = doy - 1
    slot[leap & (doy >= 60)] -= 1          # 29 Feb -> 28 Feb slot; subsequent days shift
    return slot                              # 0..364


def slot_selectors(dates, slot):
    cal = (dates.year >= CAL_YEARS[0]) & (dates.year <= CAL_YEARS[1])
    sel = []
    for j in range(365):
        dist = np.abs(slot - j)
        dist = np.minimum(dist, 365 - dist)
        sel.append(np.where(cal & (dist <= HALF_WIN))[0])
    return sel


def fit_gamma(X):
    """X: (n, C) calibration accumulations (NaN allowed). Returns alpha, beta, q, n_valid per county."""
    valid = np.isfinite(X)
    n = valid.sum(0)
    pos = valid & (X >= ZERO_MM)
    npos = pos.sum(0)
    q = np.where(n > 0, (n - npos) / np.maximum(n, 1), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        xm = np.where(pos, X, 0.0).sum(0) / npos
        lnm = np.where(pos, np.log(np.where(pos, X, 1.0)), 0.0).sum(0) / npos
        A = np.log(xm) - lnm
        alpha = (1.0 + np.sqrt(1.0 + 4.0 * A / 3.0)) / (4.0 * A)
        beta = xm / alpha
    ok = (npos >= 30) & np.isfinite(alpha) & np.isfinite(beta) & (alpha > 0) & (beta > 0)
    return alpha, beta, q, n, ok


def cdf_spi(x, alpha, beta, q):
    with np.errstate(invalid="ignore", divide="ignore"):
        Gx = gammainc(alpha, np.maximum(x, 0.0) / beta)
    H = np.where(x >= ZERO_MM, q + (1.0 - q) * Gx, q / 2.0)
    return np.where(np.isfinite(x), H, np.nan)


def fit_loglogistic(X):
    """Unbiased PWM fit per county (columns). NaN-aware (NaN sorted to the end)."""
    Xs = np.sort(X, axis=0)                       # NaN last
    n = np.isfinite(Xs).sum(0).astype(np.float64)
    i = np.arange(1, Xs.shape[0] + 1, dtype=np.float64)[:, None]
    v = (i <= n)
    Xz = np.where(v, Xs, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        b0 = Xz.sum(0) / n
        b1 = (np.where(v, (i - 1) / (n - 1), 0.0) * Xz).sum(0) / n
        b2 = (np.where(v, (i - 1) * (i - 2) / ((n - 1) * (n - 2)), 0.0) * Xz).sum(0) / n
        w0, w1, w2 = b0, b0 - b1, b0 - 2 * b1 + b2
        bet = (2 * w1 - w0) / (6 * w1 - w0 - 6 * w2)
        gg = G(1 + 1 / bet) * G(1 - 1 / bet)
        alp = (w0 - 2 * w1) * bet / gg
        gam = w0 - alp * gg
    ok = (n >= 30) & np.isfinite(bet) & np.isfinite(alp) & np.isfinite(gam) & (bet > 1) & (alp > 0)
    return alp, bet, gam, n, ok


def cdf_loglogistic(x, alp, bet, gam):
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        z = x - gam
        F = np.where(z > 0, 1.0 / (1.0 + (alp / np.where(z > 0, z, 1.0)) ** bet), 0.0)
    return np.where(np.isfinite(x), F, np.nan)


def ub_pwm(X):
    """Unbiased PWMs b0, b1, b2 per column (NaN-aware) and sample size n."""
    Xs = np.sort(X, axis=0)
    n = np.isfinite(Xs).sum(0).astype(np.float64)
    i = np.arange(1, Xs.shape[0] + 1, dtype=np.float64)[:, None]
    v = (i <= n)
    Xz = np.where(v, Xs, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        b0 = Xz.sum(0) / n
        b1 = (np.where(v, (i - 1) / (n - 1), 0.0) * Xz).sum(0) / n
        b2 = (np.where(v, (i - 1) * (i - 2) / ((n - 1) * (n - 2)), 0.0) * Xz).sum(0) / n
    return b0, b1, b2, n


K_EPS = 1e-6


def fit_glo(X):
    """Generalized logistic (Hosking) from unbiased PWMs: returns xi, alpha, k, t3, n, ok."""
    b0, b1, b2, n = ub_pwm(X)
    with np.errstate(invalid="ignore", divide="ignore"):
        l1, l2, l3 = b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0
        t3 = l3 / l2
        k = -t3
        small = np.abs(k) < K_EPS
        kp = np.where(small, 1.0, k) * np.pi
        alpha = np.where(small, l2, l2 * np.sin(kp) / kp)
        xi = np.where(small, l1, l1 - alpha * (1.0 / np.where(small, 1.0, k) - np.pi / np.sin(kp)))
    ok = (n >= 30) & np.isfinite(alpha) & np.isfinite(xi) & np.isfinite(k) & (alpha > 0) & (np.abs(k) < 1)
    return xi, alpha, k, t3, n, ok


def cdf_glo(x, xi, alpha, k):
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        small = np.abs(k) < K_EPS
        arg = 1.0 - np.where(small, 0.0, k) * (x - xi) / alpha
        y = np.where(small, (x - xi) / alpha, -np.log(np.where(arg > 0, arg, 1.0)) / np.where(small, 1.0, k))
        F = 1.0 / (1.0 + np.exp(-y))
        F = np.where(small | (arg > 0), F, np.where(k > 0, 1.0, 0.0))   # outside the bounded support
    return np.where(np.isfinite(x), F, np.nan)


def standardize(acc, sel, slot, kind, name, diag):
    T, C = acc.shape
    out = np.full((T, C), np.nan)
    for j in range(365):
        X = acc[sel[j]]
        days = np.where(slot == j)[0]
        if kind == "spi":
            a, b, q, n, ok = fit_gamma(X)
            P = cdf_spi(acc[days], a, b, q)
        else:
            xi, al, k, t3, n, ok = fit_glo(X)
            P = cdf_glo(acc[days], xi, al, k)
            # evidence: where t3 > 0 the GLO must equal the closed-form 3-parameter log-logistic
            la, lb, lg, _, lok = fit_loglogistic(X)
            both = ok & lok
            eq = np.nanmax(np.abs(cdf_glo(X[:, both], xi[both], al[both], k[both]) -
                                  cdf_loglogistic(X[:, both], la[both], lb[both], lg[both]))) if both.any() else np.nan
        P = np.clip(P, P_LO, P_HI)
        z = ndtri(P)
        z[:, ~ok] = np.nan
        out[days] = z
        rec = dict(index=name, slot=j, counties_fit_ok=int(ok.sum()), counties_fit_failed=int((~ok).sum()),
                   median_n=float(np.median(n)))
        if kind != "spi":
            rec.update(counties_negative_lskew=int((t3 < 0).sum()), max_abs_cdf_diff_glo_vs_closedform_loglogistic=float(eq))
        diag.append(rec)
    return out


def main():
    t0 = time.time()
    storage_guard(1.5, logger=log)
    dates, counties, a = load_panel()
    log.info("Panel loaded: %s days x %d counties; missing ppt=%d tmax=%d et0=%d", len(dates), len(counties),
             int(np.isnan(a["ppt"]).sum()), int(np.isnan(a["tmax"]).sum()), int(np.isnan(a["et0"]).sum()))
    slot = doy_slots(dates)
    sel = slot_selectors(dates, slot)
    diag = []
    res = {}
    for name, w in SPI_WINDOWS.items():
        acc = rolling_sum_strict(a["ppt"], w)
        res[name] = standardize(acc, sel, slot, "spi", name, diag)
        log.info("%s done (%.1f min)", name, (time.time() - t0) / 60)
    wb = a["ppt"] - a["et0"]
    res["water_balance"] = wb
    for name, w in SPEI_WINDOWS.items():
        acc = rolling_sum_strict(wb, w)
        res[name] = standardize(acc, sel, slot, "spei", name, diag)
        log.info("%s done (%.1f min)", name, (time.time() - t0) / 60)
    spei = res["spei30_prism_nldas"]
    tmax = a["tmax"]
    both = np.isfinite(spei) & np.isfinite(tmax)
    flag = both & (spei < SPEI_THRESHOLD) & (tmax > TMAX_THRESHOLD_C)
    res["cdhw_valid"] = both.astype(np.int8)
    res["cdhw_flag"] = np.where(both, flag, np.nan)
    res["cdhw_severity"] = np.where(both, np.where(flag, np.abs(spei) * (tmax - TMAX_THRESHOLD_C), 0.0), np.nan)
    res["heat_exceedance"] = np.where(np.isfinite(tmax), np.maximum(tmax - TMAX_THRESHOLD_C, 0.0), np.nan)
    res["drought_severity"] = np.where(np.isfinite(spei), np.where(spei < SPEI_THRESHOLD, np.abs(spei), 0.0), np.nan)

    T, C = spei.shape
    df = pd.DataFrame({"county_fips": np.tile(counties, T), "date": np.repeat(dates.values, C)})
    for k, v in res.items():
        df[k] = v.reshape(-1).astype(np.float32)
    df["cdhw_flag"] = df["cdhw_flag"].astype("float32")
    assert not df.duplicated(["county_fips", "date"]).any()
    df.to_parquet(OUT / "county_daily_indices_1985_2023.parquet", index=False, compression="zstd")
    pd.DataFrame(diag).to_csv(AUDITS / "spi_spei_fit_diagnostics.csv", index=False)

    # calibration check: within the calibration period each index should be ~N(0,1)
    cal = (dates.year <= CAL_YEARS[1])
    chk = []
    for k in list(SPI_WINDOWS) + list(SPEI_WINDOWS):
        v = res[k][cal]
        chk.append(dict(index=k, period="calibration 1985-2013", mean=float(np.nanmean(v)), sd=float(np.nanstd(v)),
                        pct_below_minus1=float(np.nanmean(v < -1) * 100), pct_missing=float(np.isnan(v).mean() * 100),
                        n_inf=int(np.isinf(v).sum())))
        v2 = res[k][~cal]
        chk.append(dict(index=k, period="out-of-calibration 2014-2023", mean=float(np.nanmean(v2)), sd=float(np.nanstd(v2)),
                        pct_below_minus1=float(np.nanmean(v2 < -1) * 100), pct_missing=float(np.isnan(v2).mean() * 100),
                        n_inf=int(np.isinf(v2).sum())))
    pd.DataFrame(chk).to_csv(AUDITS / "spi_spei_calibration_check.csv", index=False)
    log.info("Indices written (%d rows) in %.1f min\n%s", len(df), (time.time() - t0) / 60, pd.DataFrame(chk).to_string())


if __name__ == "__main__":
    main()
