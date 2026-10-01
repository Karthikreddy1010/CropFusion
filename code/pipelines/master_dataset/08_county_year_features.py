"""
08_county_year_features.py -- aggregate county-daily PRISM, NLDAS, SPI, SPEI and CDHW to county-year features.

Windows (all computed per county-year):
  ann   : 1 Jan - 31 Dec
  gs    : 1 Apr - 31 Oct (growing season recovered from the old dataset: 214 days)
  veg   : early season / vegetative   -> 1 Apr ... last day with cumulative GDD <= 700
  silk  : flowering / silking-R1      -> 700 < cumulative GDD <= 1400
  grain : grain filling               -> cumulative GDD > 1400, up to 31 Oct
  (GDD thresholds from code/config.py; GDD accumulated from 1 Apr with PRISM Tmax/Tmin; late season omitted
   by user decision -- no project definition exists.)

Missing-data rules: means / min / max need >= 95 % valid days in the window; totals and day counts need
100 % valid days; otherwise NaN. Empty phenology windows (threshold not reached by 31 Oct): totals and
counts = 0, means/min/max = NaN, *_days = 0.
CDHW episodes = runs of consecutive CDHW days inside the window (runs truncated at window edges).
  cdhw_count             = number of episodes
  cdhw_total_days        = number of CDHW days
  cdhw_max_duration      = longest episode (days)
  cdhw_max_severity      = largest episode severity (sum of daily |SPEI30| x (Tmax-35) over the episode)
  cdhw_cumulative_severity = sum of daily joint severity over the window

Outputs (intermediate/county_year/): prism_features, nldas_features, spi_features, spei_features,
  cdhw_features, phenology_features  (.parquet, unique county_fips + year)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from md_common import (GDD_SILKING_END, GDD_VEGETATIVE_END, INTER, TMAX_THRESHOLD_C, get_logger)

log = get_logger("08_county_year_features")
OUT = INTER / "county_year"
OUT.mkdir(exist_ok=True)
MEAN_COV, TOTAL_COV = 0.95, 1.0
GDD_BASE, GDD_CAP = 10.0, 30.0


def gdd_daily(tmax, tmin, method):
    if method == "capped_86_50":
        tx = np.clip(tmax, GDD_BASE, GDD_CAP)
        tn = np.clip(tmin, GDD_BASE, GDD_CAP)
        return np.maximum((tx + tn) / 2.0 - GDD_BASE, 0.0)
    return np.maximum((tmax + tmin) / 2.0 - GDD_BASE, 0.0)   # simple base-10


def read_year(y):
    dates = pd.date_range(f"{y}-01-01", f"{y}-12-31")
    p = pd.read_parquet(INTER / "prism_county_daily" / f"prism_county_daily_{y}.parquet")
    n = pd.read_parquet(INTER / "nldas_county_daily" / f"nldas_county_daily_{y}.parquet")
    ix = pq.read_table(INTER / "indices" / "county_daily_indices_1985_2023.parquet",
                       filters=[("date", ">=", pd.Timestamp(dates[0])), ("date", "<=", pd.Timestamp(dates[-1]))]).to_pandas()
    counties = np.array(sorted(p.county_fips.unique()))
    A = {}
    for df, cols in ((p, ["prism_tmax", "prism_tmin", "prism_ppt"]),
                     (n, ["nldas_et0", "nldas_rs", "nldas_ea", "nldas_vpd", "nldas_wind", "nldas_pressure",
                          "nldas_tair_mean", "nldas_lwdown", "nldas_rainf_qc"]),
                     (ix, ["spi1_prism", "spi3_prism", "spi6_prism", "spi12_prism", "spei30_prism_nldas",
                           "spei90_prism_nldas", "water_balance", "cdhw_flag", "cdhw_severity", "heat_exceedance",
                           "drought_severity"])):
        for c in cols:
            A[c] = df.pivot(index="date", columns="county_fips", values=c).reindex(index=dates, columns=counties).to_numpy(np.float64)
    return dates, counties, A


class Agg:
    def __init__(self, M):
        self.M = M                     # (T, C) bool window mask
        self.nwin = M.sum(0)

    def _cov(self, x):
        v = self.M & np.isfinite(x)
        nv = v.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return v, nv, np.where(self.nwin > 0, nv / np.maximum(self.nwin, 1), 1.0)

    def mean(self, x):
        v, nv, cov = self._cov(x)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(v, x, 0.0).sum(0) / nv
        return np.where((cov >= MEAN_COV) & (nv > 0), m, np.nan)

    def min(self, x):
        v, nv, cov = self._cov(x)
        m = np.where(v, x, np.inf).min(0)
        return np.where((cov >= MEAN_COV) & (nv > 0), m, np.nan)

    def max(self, x):
        v, nv, cov = self._cov(x)
        m = np.where(v, x, -np.inf).max(0)
        return np.where((cov >= MEAN_COV) & (nv > 0), m, np.nan)

    def total(self, x):
        v, nv, cov = self._cov(x)
        return np.where(cov >= TOTAL_COV, np.where(v, x, 0.0).sum(0), np.nan)

    def count(self, cond, x):
        v, nv, cov = self._cov(x)
        return np.where(cov >= TOTAL_COV, (v & cond).sum(0).astype(float), np.nan)

    def valid_days(self, x):
        return (self.M & np.isfinite(x)).sum(0).astype(float)

    def runs(self, flag, sev):
        """Episode statistics of consecutive flagged days inside the window."""
        T, C = flag.shape
        cur_len = np.zeros(C)
        cur_sev = np.zeros(C)
        count = np.zeros(C)
        max_len = np.zeros(C)
        max_sev = np.zeros(C)
        for t in range(T):
            f = self.M[t] & (flag[t] == 1)
            start = f & (cur_len == 0)
            count += start
            cur_len = np.where(f, cur_len + 1, 0)
            cur_sev = np.where(f, cur_sev + np.nan_to_num(sev[t]), 0.0)
            max_len = np.maximum(max_len, cur_len)
            max_sev = np.maximum(max_sev, cur_sev)
        v, nv, cov = self._cov(flag)
        ok = cov >= TOTAL_COV
        return (np.where(ok, count, np.nan), np.where(ok, max_len, np.nan), np.where(ok, max_sev, np.nan))


def year_features(y, gdd_method):
    dates, counties, A = read_year(y)
    T, C = len(dates), len(counties)
    ann = np.ones((T, C), bool)
    gs1 = ((dates >= pd.Timestamp(f"{y}-04-01")) & (dates <= pd.Timestamp(f"{y}-10-31")))
    gs = np.repeat(gs1[:, None], C, axis=1)
    g = gdd_daily(A["prism_tmax"], A["prism_tmin"], gdd_method)
    cum = np.where(gs, np.nan_to_num(g), 0.0).cumsum(0)
    veg = gs & (cum <= GDD_VEGETATIVE_END)
    silk = gs & (cum > GDD_VEGETATIVE_END) & (cum <= GDD_SILKING_END)
    grain = gs & (cum > GDD_SILKING_END)
    wins = {"ann": ann, "gs": gs, "veg": veg, "silk": silk, "grain": grain}
    key = {"county_fips": counties, "year": np.full(C, y)}
    out = {k: dict(key) for k in ("prism", "nldas", "spi", "spei", "cdhw", "pheno")}
    tx, tn, pp = A["prism_tmax"], A["prism_tmin"], A["prism_ppt"]
    for w in ("ann", "gs"):
        a = Agg(wins[w])
        P = out["prism"]
        P[f"prism_tmax_mean_{w}"] = a.mean(tx)
        P[f"prism_tmax_max_{w}"] = a.max(tx)
        P[f"prism_tmin_mean_{w}"] = a.mean(tn)
        P[f"prism_tmin_min_{w}"] = a.min(tn)
        P[f"prism_ppt_total_{w}"] = a.total(pp)
        P[f"prism_valid_days_{w}"] = a.valid_days(tx + tn + pp)
        if w == "gs":
            P["prism_dtr_mean_gs"] = a.mean(tx - tn)
            P["prism_gdd_gs"] = a.total(g)
            P["prism_kdd30_gs"] = a.total(np.where(np.isfinite(tx), np.maximum(tx - 30.0, 0.0), np.nan))
            P["prism_tmax_days_gt35_gs"] = a.count(tx > TMAX_THRESHOLD_C, tx)
            P["prism_heat_exceedance_total_gs"] = a.total(A["heat_exceedance"])
        N = out["nldas"]
        N[f"nldas_et0_mean_{w}"] = a.mean(A["nldas_et0"])
        N[f"nldas_et0_total_{w}"] = a.total(A["nldas_et0"])
        N[f"nldas_rs_mean_{w}"] = a.mean(A["nldas_rs"])
        N[f"nldas_humidity_mean_{w}"] = a.mean(A["nldas_ea"])
        N[f"nldas_wind_mean_{w}"] = a.mean(A["nldas_wind"])
        N[f"nldas_pressure_mean_{w}"] = a.mean(A["nldas_pressure"])
        N[f"water_balance_total_{w}"] = a.total(A["water_balance"])
        N[f"nldas_valid_days_{w}"] = a.valid_days(A["nldas_et0"])
        if w == "gs":
            N["nldas_vpd_mean_gs"] = a.mean(A["nldas_vpd"])
            N["nldas_tair_mean_gs"] = a.mean(A["nldas_tair_mean"])
            N["nldas_lwdown_mean_gs"] = a.mean(A["nldas_lwdown"])
            N["nldas_rainf_total_gs_qc"] = a.total(A["nldas_rainf_qc"])
        S = out["spi"]
        for k in ("spi1_prism", "spi3_prism", "spi6_prism", "spi12_prism"):
            S[f"{k}_mean_{w}"] = a.mean(A[k])
            S[f"{k}_min_{w}"] = a.min(A[k])
        E = out["spei"]
        s30 = A["spei30_prism_nldas"]
        E[f"spei30_prism_nldas_mean_{w}"] = a.mean(s30)
        E[f"spei30_prism_nldas_min_{w}"] = a.min(s30)
        E[f"spei30_prism_nldas_max_{w}"] = a.max(s30)
        E[f"spei30_prism_nldas_drought_days_{w}"] = a.count(s30 < -1.0, s30)
        E[f"spei30_prism_nldas_drought_severity_{w}"] = a.total(A["drought_severity"])
        E[f"spei90_prism_nldas_mean_{w}"] = a.mean(A["spei90_prism_nldas"])
        E[f"spei90_prism_nldas_min_{w}"] = a.min(A["spei90_prism_nldas"])
        K = out["cdhw"]
        cnt, mdur, msev = a.runs(A["cdhw_flag"], A["cdhw_severity"])
        K[f"cdhw_count_{w}"] = cnt
        K[f"cdhw_total_days_{w}"] = a.count(A["cdhw_flag"] == 1, A["cdhw_flag"])
        K[f"cdhw_max_duration_{w}"] = mdur
        K[f"cdhw_max_severity_{w}"] = msev
        K[f"cdhw_cumulative_severity_{w}"] = a.total(A["cdhw_severity"])
        K[f"cdhw_valid_days_{w}"] = a.valid_days(A["cdhw_flag"])
        K[f"cdhw_window_days_{w}"] = a.nwin.astype(float)
    # phenology windows
    Ph = out["pheno"]
    doy = dates.dayofyear.values[:, None]
    for s in ("veg", "silk", "grain"):
        M = wins[s]
        a = Agg(M)
        nd = M.sum(0)
        Ph[f"pheno_{s}_days"] = nd.astype(float)
        Ph[f"pheno_{s}_start_doy"] = np.where(nd > 0, np.where(M, doy, 9999).min(0), np.nan)
        Ph[f"pheno_{s}_end_doy"] = np.where(nd > 0, np.where(M, doy, -1).max(0), np.nan)
        empty = nd == 0
        def z(v):  # totals / counts of an empty window are 0
            return np.where(empty, 0.0, v)
        Ph[f"prism_tmax_mean_{s}"] = a.mean(tx)
        Ph[f"prism_tmax_days_gt35_{s}"] = z(a.count(tx > TMAX_THRESHOLD_C, tx))
        Ph[f"prism_ppt_total_{s}"] = z(a.total(pp))
        Ph[f"nldas_et0_total_{s}"] = z(a.total(A["nldas_et0"]))
        Ph[f"water_balance_total_{s}"] = z(a.total(A["water_balance"]))
        Ph[f"nldas_vpd_mean_{s}"] = a.mean(A["nldas_vpd"])
        Ph[f"spi1_prism_min_{s}"] = a.min(A["spi1_prism"])
        Ph[f"spei30_prism_nldas_mean_{s}"] = a.mean(A["spei30_prism_nldas"])
        Ph[f"spei30_prism_nldas_min_{s}"] = a.min(A["spei30_prism_nldas"])
        cnt, mdur, msev = a.runs(A["cdhw_flag"], A["cdhw_severity"])
        Ph[f"cdhw_count_{s}"] = z(cnt)
        Ph[f"cdhw_total_days_{s}"] = z(a.count(A["cdhw_flag"] == 1, A["cdhw_flag"]))
        Ph[f"cdhw_cumulative_severity_{s}"] = z(a.total(A["cdhw_severity"]))
    return {k: pd.DataFrame(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdd-method", default="capped_86_50", choices=["capped_86_50", "simple_base10"])
    ap.add_argument("--years", type=int, nargs="*", default=list(range(1985, 2024)))
    a = ap.parse_args()
    t0 = time.time()
    parts = {k: [] for k in ("prism", "nldas", "spi", "spei", "cdhw", "pheno")}
    for y in a.years:
        r = year_features(y, a.gdd_method)
        for k, v in r.items():
            parts[k].append(v)
        log.info("year %d features done (%.1f min)", y, (time.time() - t0) / 60)
    names = {"prism": "prism_features", "nldas": "nldas_features", "spi": "spi_features", "spei": "spei_features",
             "cdhw": "cdhw_features", "pheno": "phenology_features"}
    for k, frames in parts.items():
        df = pd.concat(frames, ignore_index=True)
        for c in df.columns[2:]:
            df[c] = df[c].astype("float32")
        assert not df.duplicated(["county_fips", "year"]).any()
        df.to_parquet(OUT / f"{names[k]}.parquet", index=False)
        log.info("%s: %s rows x %s cols", names[k], len(df), df.shape[1])
    (OUT / "gdd_method.txt").write_text(a.gdd_method)


if __name__ == "__main__":
    main()
