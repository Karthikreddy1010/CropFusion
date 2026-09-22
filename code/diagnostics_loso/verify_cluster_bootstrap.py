"""Validity check for the year-clustered bootstrap used by RO1-RO5.

Every headline number in the objectives report is a coverage or miss rate pooled
over 16 rolling origins. Rows inside one origin are not independent -- a drought
year moves every county in the panel together -- so a row-level interval around
those numbers is the wrong width, usually by a large factor. The objectives are
therefore reported with an interval that resamples whole origins.

That correction has a known answer, so it is tested rather than assumed:

  1. the point estimate is the statistic itself, not a bootstrap average;
  2. under perfect within-cluster dependence the clustered interval is much wider
     than a row-level one -- the property the whole exercise exists for;
  3. it is deterministic given a seed;
  4. resamples where the statistic is undefined are discarded and counted;
  5. too few clusters, or too few valid resamples, returns no interval instead of
     a misleading one;
  6. a non-zero null (the alpha/2 safety target) is honoured;
  7. the nominal 95% interval covers the truth about 95% of the time on data
     built to satisfy its assumptions.

Run:  python code/diagnostics_loso/verify_cluster_bootstrap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import dependence_aware_stats as das  # noqa: E402


def mean_hit(frame: pd.DataFrame) -> float:
    return float(frame["hit"].mean())


def make_blocked(n_clusters: int = 12, per_cluster: int = 200) -> pd.DataFrame:
    """Perfect within-cluster dependence: every row in a cluster shares its value."""
    rows = []
    for c in range(n_clusters):
        rows.append(pd.DataFrame({"cluster": c, "hit": float(c % 2)}, index=range(per_cluster)))
    return pd.concat(rows, ignore_index=True)


def row_level_ci(frame: pd.DataFrame, seed: int = 0, n_boot: int = 2000) -> float:
    """Width of the naive interval that ignores clustering, for comparison."""
    rng = np.random.default_rng(seed)
    v = frame["hit"].values
    boots = [float(np.mean(rng.choice(v, size=len(v), replace=True))) for _ in range(n_boot)]
    return float(np.percentile(boots, 97.5) - np.percentile(boots, 2.5))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    checks = []

    # 1 - the point estimate is the statistic on the observed data
    blocked = make_blocked()
    res = das.cluster_bootstrap_statistic(blocked, mean_hit, cluster_col="cluster",
                                          n_boot=1000, seed=1)
    checks.append(("point estimate equals the statistic on the full frame",
                   abs(res["estimate"] - 0.5) < 1e-12))
    print(f"point estimate {res['estimate']:.6f} (statistic 0.500000)")

    # 2 - the reason the module exists: clustering must widen the interval
    clustered_width = res["ci_95"][1] - res["ci_95"][0]
    naive_width = row_level_ci(blocked)
    ratio = clustered_width / naive_width
    print(f"interval width: clustered {clustered_width:.4f}, row-level {naive_width:.4f} "
          f"({ratio:.1f}x)")
    checks.append(("clustered interval is >5x the row-level interval under block dependence",
                   ratio > 5.0))

    # 3 - determinism
    again = das.cluster_bootstrap_statistic(blocked, mean_hit, cluster_col="cluster",
                                            n_boot=1000, seed=1)
    checks.append(("same seed reproduces the interval exactly", again["ci_95"] == res["ci_95"]))

    # 4 - subgroup contrasts, the shape RO1 and RO2 actually use
    rng = np.random.default_rng(7)
    frames = []
    for c in range(16):
        frames.append(pd.DataFrame({
            "cluster": c,
            "stratum": ["Normal"] * 60 + ["Extreme"] * 20,
            "hit": np.concatenate([rng.binomial(1, 0.90, 60), rng.binomial(1, 0.75, 20)]).astype(float),
        }))
    panel = pd.concat(frames, ignore_index=True)

    def gap(f: pd.DataFrame) -> float:
        return f[f.stratum == "Normal"]["hit"].mean() - f[f.stratum == "Extreme"]["hit"].mean()

    g = das.cluster_bootstrap_statistic(panel, gap, cluster_col="cluster", n_boot=2000, seed=3)
    manual = gap(panel)
    print(f"subgroup contrast {g['estimate']:.4f} (manual {manual:.4f}) "
          f"CI {g['ci_95']}")
    # the module reports to 6dp throughout, so the contract is agreement at that scale
    checks.append(("subgroup contrast point matches a direct computation",
                   abs(g["estimate"] - manual) < 1e-6))
    checks.append(("a true 0.15 contrast is detected as non-zero", g["significant_at_0.05"] is True))

    other = das.cluster_bootstrap_statistic(panel, gap, cluster_col="cluster", n_boot=2000, seed=99)
    checks.append(("a different seed gives a different interval", other["ci_95"] != g["ci_95"]))

    # 5 - undefined resamples are discarded, not silently treated as zero
    rare = panel.copy()
    rare.loc[(rare.cluster != 0) & (rare.stratum == "Extreme"), "stratum"] = "Normal"
    r = das.cluster_bootstrap_statistic(rare, gap, cluster_col="cluster", n_boot=800, seed=4)
    print(f"stratum present in 1/16 clusters: {r['n_boot_valid']}/{r['n_boot']} resamples valid, "
          f"inference_supported={r['inference_supported']}")
    checks.append(("resamples missing the subgroup are discarded",
                   r["n_boot_valid"] < r["n_boot"]))
    checks.append(("a statistic undefined in most resamples reports no interval",
                   r["inference_supported"] is False and r["ci_95"] is None))

    # 6 - too few clusters
    few = blocked[blocked.cluster < 3]
    f = das.cluster_bootstrap_statistic(few, mean_hit, cluster_col="cluster", n_boot=500, seed=5)
    checks.append(("fewer than the minimum clusters reports no interval",
                   f["inference_supported"] is False and f["ci_95"] is None))
    checks.append(("the point estimate survives when inference does not",
                   abs(f["estimate"] - float(few["hit"].mean())) < 1e-6))

    # 7 - a non-zero null: the alpha/2 safety target, not zero
    def unsafe_rate(fr: pd.DataFrame) -> float:
        return float(fr[fr.stratum == "Extreme"]["hit"].mean())

    miss = panel.copy()
    miss["hit"] = 1.0 - miss["hit"]          # miss rate: Extreme ~0.25, target 0.05
    u = das.cluster_bootstrap_statistic(miss, unsafe_rate, cluster_col="cluster",
                                        n_boot=2000, seed=6, null_value=0.05)
    print(f"unsafe rate {u['estimate']:.4f} vs target 0.05, CI {u['ci_95']}, "
          f"excludes target: {u['significant_at_0.05']}")
    checks.append(("a rate far above the target excludes it", u["significant_at_0.05"] is True))

    # misses drawn AT the target rate inside the subgroup being scored -- thinning the
    # whole frame by 5% leaves the subgroup somewhere else entirely
    rs = np.random.default_rng(11)
    at_target = panel.copy()
    at_target["hit"] = 0.0
    ex_idx = at_target.index[at_target.stratum == "Extreme"]
    at_target.loc[ex_idx, "hit"] = rs.binomial(1, 0.05, len(ex_idx)).astype(float)
    a = das.cluster_bootstrap_statistic(at_target, unsafe_rate, cluster_col="cluster",
                                        n_boot=2000, seed=6, null_value=0.05)
    print(f"rate drawn at the target: {a['estimate']:.4f}, CI {a['ci_95']}, "
          f"excludes target: {a['significant_at_0.05']}")
    checks.append(("a rate at the target does not exclude it", a["significant_at_0.05"] is False))
    checks.append(("the null used is recorded in the output", a.get("null_value") == 0.05))

    # 8 - calibration on data that satisfies the assumptions
    n_sim, truth, hits = 300, 0.9, 0
    for s in range(n_sim):
        rs = np.random.default_rng(1000 + s)
        sim = pd.concat([
            pd.DataFrame({"cluster": c,
                          "hit": rs.binomial(1, rs.uniform(0.82, 0.98), 20).astype(float)})
            for c in range(16)], ignore_index=True)
        out = das.cluster_bootstrap_statistic(sim, mean_hit, cluster_col="cluster",
                                              n_boot=400, seed=s)
        if out["ci_95"] and out["ci_95"][0] <= truth <= out["ci_95"][1]:
            hits += 1
    emp = hits / n_sim
    print(f"empirical coverage of the nominal 95% interval: {emp:.3f} over {n_sim} simulations")
    checks.append(("nominal 95% interval covers the truth 88-99% of the time",
                   0.88 <= emp <= 0.99))

    failed = 0
    print()
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        failed += 0 if ok else 1
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
