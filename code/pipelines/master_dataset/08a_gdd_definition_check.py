"""
08a_gdd_definition_check.py -- recover the project's GDD definition before building phenology windows.

The old GDD_Accumulated (ERA5, season total) documents neither base/cap nor start date. Candidate PRISM GDD
definitions are compared with it on 1985-2013 county-years (six states). The candidate with the smallest
absolute mean bias and high correlation is adopted for the GDD phenology windows (thresholds 700/1400 unchanged).
This uses the old variable for VALIDATION ONLY; no ERA5 value enters the master.

Output: audits/gdd_definition_check.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from md_common import AUDITS, INTER, OLD_MEGA, STUDY_STATES, get_logger

log = get_logger("08a_gdd_definition_check")

CANDIDATES = {
    "capped_86_50_Apr1_Oct31": ("capped", (4, 1), (10, 31)),
    "simple_base10_Apr1_Oct31": ("simple", (4, 1), (10, 31)),
    "capped_86_50_May1_Sep30": ("capped", (5, 1), (9, 30)),
    "simple_base10_May1_Sep30": ("simple", (5, 1), (9, 30)),
    "capped_86_50_May1_Oct31": ("capped", (5, 1), (10, 31)),
}


def main():
    old = pd.read_csv(OLD_MEGA, usecols=["GEOID", "STATEFP", "Year", "GDD_Accumulated", "Tmax_Days_Above_35", "Precip_growseason_mm"])
    old = old[old.STATEFP.astype(str).str.zfill(2).isin(STUDY_STATES) & old.Year.between(1985, 2013)]
    old["county_fips"] = old.GEOID.astype(str).str.zfill(5)
    rows = []
    res = {k: [] for k in CANDIDATES}
    extra = []
    for y in range(1985, 2014):
        p = pd.read_parquet(INTER / "prism_county_daily" / f"prism_county_daily_{y}.parquet")
        md = p.date.dt.month * 100 + p.date.dt.day
        for k, (meth, s, e) in CANDIDATES.items():
            sel = (md >= s[0] * 100 + s[1]) & (md <= e[0] * 100 + e[1])
            q = p[sel]
            if meth == "capped":
                g = np.maximum((q.prism_tmax.clip(10, 30) + q.prism_tmin.clip(10, 30)) / 2 - 10, 0)
            else:
                g = np.maximum((q.prism_tmax + q.prism_tmin) / 2 - 10, 0)
            res[k].append(pd.DataFrame({"county_fips": q.county_fips, "year": y, "gdd": g}).groupby(["county_fips", "year"]).gdd.sum())
        gs = p[(md >= 401) & (md <= 1031)]
        extra.append(gs.groupby("county_fips").agg(ppt=("prism_ppt", "sum"), hot=("prism_tmax", lambda s: (s > 35).sum())).assign(year=y))
    for k, parts in res.items():
        s = pd.concat(parts).reset_index()
        j = old.merge(s, left_on=["county_fips", "Year"], right_on=["county_fips", "year"])
        rows.append(dict(candidate=k, n=len(j), old_mean=j.GDD_Accumulated.mean(), prism_mean=j.gdd.mean(),
                         mean_bias=(j.gdd - j.GDD_Accumulated).mean(), mae=(j.gdd - j.GDD_Accumulated).abs().mean(),
                         pearson_r=np.corrcoef(j.gdd, j.GDD_Accumulated)[0, 1]))
    out = pd.DataFrame(rows).sort_values("mae")
    ex = pd.concat(extra).reset_index().merge(old, left_on=["county_fips", "year"], right_on=["county_fips", "Year"])
    out.attrs = {}
    out.to_csv(AUDITS / "gdd_definition_check.csv", index=False)
    ctx = dict(ppt_gs_prism_vs_old_era5_r=float(np.corrcoef(ex.ppt, ex.Precip_growseason_mm)[0, 1]),
               ppt_gs_prism_mean=float(ex.ppt.mean()), ppt_gs_old_mean=float(ex.Precip_growseason_mm.mean()),
               tmax_days_gt35_prism_mean=float(ex.hot.mean()), tmax_days_gt35_old_mean=float(ex.Tmax_Days_Above_35.mean()))
    pd.DataFrame([ctx]).to_csv(AUDITS / "prism_vs_old_era5_growseason_check.csv", index=False)
    log.info("GDD definition check:\n%s\nPRISM vs old ERA5 context: %s", out.to_string(index=False), ctx)


if __name__ == "__main__":
    main()
