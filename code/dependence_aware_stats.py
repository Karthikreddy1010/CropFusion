"""
dependence_aware_stats.py - Inference that respects the panel structure.

The evaluation data is a county-year panel with strong temporal autocorrelation
(lag-1 autocorrelation is substantial in the diagnostics) and strong spatial
correlation between neighbouring counties within a year. Treating the ~2.3k
county-year test observations as independent samples inflates every test
statistic and shrinks every confidence interval, sometimes by an order of
magnitude in effective sample size.

Every helper here takes an explicit clustering unit and says so in its output,
so a reader can check that the unit of independence matches the claim being
made. Nothing in this module manufactures significance: when the effective
number of independent units is too small to support a test, the helpers say so
rather than returning a p-value.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("paper3")

MIN_CLUSTERS_FOR_INFERENCE = 5


def cluster_bootstrap_mean(
    values: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Cluster bootstrap for the mean of ``values``, resampling whole clusters.

    Clusters (e.g. years, or counties) are drawn with replacement; all rows of a
    drawn cluster enter the resample together, so within-cluster dependence is
    preserved instead of being averaged away.
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    n_clusters = len(uniq)
    index_by_cluster = {c: np.where(clusters == c)[0] for c in uniq}

    point = float(np.mean(values))
    if n_clusters < MIN_CLUSTERS_FOR_INFERENCE:
        return {
            "estimate": round(point, 6),
            "n_clusters": int(n_clusters),
            "cluster_unit": "unspecified",
            "ci_95": None,
            "p_value": None,
            "method": "cluster_bootstrap",
            "inference_supported": False,
            "reason": (
                f"only {n_clusters} independent clusters "
                f"(minimum {MIN_CLUSTERS_FOR_INFERENCE} required); "
                "reporting the point estimate without inferential claims"
            ),
        }

    rng = np.random.RandomState(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        drawn = rng.choice(uniq, size=n_clusters, replace=True)
        idx = np.concatenate([index_by_cluster[c] for c in drawn])
        boots[b] = np.mean(values[idx])

    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    # Two-sided bootstrap p-value for H0: mean == 0, by CI inversion.
    p_two_sided = 2.0 * min(float(np.mean(boots <= 0.0)), float(np.mean(boots >= 0.0)))
    p_two_sided = min(1.0, max(p_two_sided, 1.0 / n_boot))

    return {
        "estimate": round(point, 6),
        "n_clusters": int(n_clusters),
        "n_observations": int(len(values)),
        "ci_95": [round(lo, 6), round(hi, 6)],
        "bootstrap_se": round(float(np.std(boots, ddof=1)), 6),
        "p_value": round(p_two_sided, 6),
        "n_boot": n_boot,
        "method": "cluster_bootstrap_percentile",
        "inference_supported": True,
    }


def cluster_bootstrap_statistic(
    frame: pd.DataFrame,
    statistic,
    cluster_col: str = "test_year",
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
    null_value: float = 0.0,
    min_valid_fraction: float = 0.8,
    cluster_unit: Optional[str] = None,
) -> Dict[str, Any]:
    """Cluster bootstrap for an arbitrary statistic of a data frame.

    ``cluster_bootstrap_mean`` covers a mean of one column. The objectives need
    contrasts instead -- coverage in one stratum minus another, one method minus
    another within a stratum, a miss rate against a fixed target -- so the
    statistic is supplied as a callable and whole clusters are resampled beneath
    it.

    Two guards keep this from inventing precision it does not have. A statistic
    can be undefined in a resample (a stratum concentrated in a few origins may
    not be drawn at all); those resamples are discarded and counted rather than
    coerced to zero. If too few survive, the estimate is returned without an
    interval, because a percentile of a biased subset of resamples is not one.

    ``null_value`` is the value the interval is tested against: zero for a
    contrast, alpha/2 for a miss rate compared with its safety target.
    """
    point_raw = statistic(frame)
    point = float(point_raw) if point_raw is not None and np.isfinite(point_raw) else float("nan")
    uniq = pd.unique(frame[cluster_col])
    n_clusters = len(uniq)
    base: Dict[str, Any] = {
        "estimate": round(point, 6) if np.isfinite(point) else None,
        "n_clusters": int(n_clusters),
        "n_observations": int(len(frame)),
        "cluster_unit": cluster_unit or cluster_col,
        "null_value": null_value,
        "method": "cluster_bootstrap_percentile",
    }
    if n_clusters < MIN_CLUSTERS_FOR_INFERENCE:
        return {**base, "ci_95": None, "p_value": None, "n_boot": 0, "n_boot_valid": 0,
                "significant_at_0.05": None, "inference_supported": False,
                "reason": (f"only {n_clusters} independent clusters "
                           f"(minimum {MIN_CLUSTERS_FOR_INFERENCE} required); "
                           "reporting the point estimate without inferential claims")}

    positions = {c: np.where(frame[cluster_col].values == c)[0] for c in uniq}
    rng = np.random.RandomState(seed)
    boots: List[float] = []
    for _ in range(n_boot):
        drawn = rng.choice(n_clusters, size=n_clusters, replace=True)
        idx = np.concatenate([positions[uniq[d]] for d in drawn])
        try:
            v = statistic(frame.take(idx))
        except Exception:                      # a resample the statistic cannot score
            continue
        if v is not None and np.isfinite(v):
            boots.append(float(v))

    n_valid = len(boots)
    if n_valid < max(MIN_CLUSTERS_FOR_INFERENCE, int(min_valid_fraction * n_boot)):
        return {**base, "ci_95": None, "p_value": None, "n_boot": n_boot,
                "n_boot_valid": n_valid, "significant_at_0.05": None,
                "inference_supported": False,
                "reason": (f"the statistic was undefined in {n_boot - n_valid} of {n_boot} "
                           "resamples, so the surviving ones are not a fair sample of them; "
                           "reporting the point estimate without an interval")}

    arr = np.asarray(boots, dtype=float)
    lo = float(np.percentile(arr, 100 * alpha / 2))
    hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    p = 2.0 * min(float(np.mean(arr <= null_value)), float(np.mean(arr >= null_value)))
    p = min(1.0, max(p, 1.0 / n_valid))
    return {**base,
            "ci_95": [round(lo, 6), round(hi, 6)],
            "bootstrap_se": round(float(np.std(arr, ddof=1)), 6),
            "p_value": round(p, 6),
            "n_boot": n_boot,
            "n_boot_valid": n_valid,
            "significant_at_0.05": bool(lo > null_value or hi < null_value),
            "inference_supported": True}


def cluster_bootstrap_paired_difference(
    values_a: np.ndarray,
    values_b: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    label_a: str = "A",
    label_b: str = "B",
    cluster_unit: str = "year",
) -> Dict[str, Any]:
    """Paired A-vs-B comparison with clusters resampled as whole blocks.

    Used for model comparisons where both models predict the same observations:
    the per-observation difference is the statistic, and whole clusters of
    differences are resampled together.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired comparison requires equal-length inputs")
    diff = a - b
    res = cluster_bootstrap_mean(diff, clusters, n_boot=n_boot, seed=seed)
    res.update({
        "comparison": f"{label_a} - {label_b}",
        "cluster_unit": cluster_unit,
        "mean_a": round(float(np.mean(a)), 6),
        "mean_b": round(float(np.mean(b)), 6),
    })
    if res.get("inference_supported"):
        res["favours"] = label_a if res["estimate"] < 0 else label_b
        res["significant_at_0.05"] = bool(
            res["ci_95"] is not None and (res["ci_95"][0] > 0 or res["ci_95"][1] < 0)
        )
    else:
        res["significant_at_0.05"] = None
    return res


def cluster_robust_ols(
    y: np.ndarray,
    x: np.ndarray,
    clusters: np.ndarray,
    cluster_unit: str = "year",
    x_name: str = "x",
) -> Dict[str, Any]:
    """Simple OLS of ``y`` on ``x`` with cluster-robust (CR1) standard errors.

    Reports the slope, its cluster-robust CI and p-value, and R-squared. R-squared
    is unaffected by the SE correction -- it is reported so effect *magnitude*
    can be judged separately from statistical detectability.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    clusters = np.asarray(clusters)
    n = len(y)
    uniq = np.unique(clusters)
    n_clusters = len(uniq)

    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum(resid ** 2) / max(1e-12, ss_tot))

    result: Dict[str, Any] = {
        "predictor": x_name,
        "slope": round(float(beta[1]), 6),
        "intercept": round(float(beta[0]), 6),
        "r_squared": round(r2, 6),
        "n_observations": int(n),
        "n_clusters": int(n_clusters),
        "cluster_unit": cluster_unit,
        "method": "OLS with cluster-robust (CR1) standard errors",
    }

    if n_clusters < MIN_CLUSTERS_FOR_INFERENCE:
        result.update({
            "std_error": None, "t_statistic": None, "p_value": None, "ci_95": None,
            "inference_supported": False,
            "reason": f"only {n_clusters} clusters; too few for cluster-robust inference",
        })
        return result

    # Cluster-robust meat matrix.
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in uniq:
        idx = np.where(clusters == c)[0]
        Xc, uc = X[idx], resid[idx]
        sc = Xc.T @ uc
        meat += np.outer(sc, sc)

    # CR1 small-sample correction.
    k = X.shape[1]
    correction = (n_clusters / max(1.0, n_clusters - 1)) * ((n - 1) / max(1.0, n - k))
    vcov = correction * (XtX_inv @ meat @ XtX_inv)
    se = float(np.sqrt(max(0.0, vcov[1, 1])))

    from scipy import stats as sp_stats
    dof = n_clusters - 1
    t_stat = float(beta[1] / se) if se > 0 else np.nan
    p_val = float(2.0 * sp_stats.t.sf(abs(t_stat), df=dof)) if se > 0 else None
    t_crit = float(sp_stats.t.ppf(0.975, df=dof))

    result.update({
        "std_error": round(se, 6),
        "t_statistic": round(t_stat, 4),
        "degrees_of_freedom": int(dof),
        "p_value": round(p_val, 6) if p_val is not None else None,
        "ci_95": [round(float(beta[1] - t_crit * se), 6), round(float(beta[1] + t_crit * se), 6)],
        "inference_supported": True,
        "significant_at_0.05": bool(p_val is not None and p_val < 0.05),
    })
    return result


def year_level_paired_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    years: np.ndarray,
    label_a: str = "A",
    label_b: str = "B",
) -> Dict[str, Any]:
    """Aggregate to one number per year, then run a paired test across years.

    This is the most conservative of the three options: the unit of
    independence is the year, so the effective sample size equals the number of
    test years. With only a handful of years the test has little power -- that
    is a property of the data, not something to be worked around.
    """
    from scipy import stats as sp_stats

    dfa = pd.DataFrame({"year": years, "a": values_a, "b": values_b})
    per_year = dfa.groupby("year")[["a", "b"]].mean()
    n_years = len(per_year)
    diffs = (per_year["a"] - per_year["b"]).values

    out: Dict[str, Any] = {
        "comparison": f"{label_a} - {label_b}",
        "unit_of_independence": "year",
        "n_years": int(n_years),
        "per_year_mean_a": {str(k): round(float(v), 6) for k, v in per_year["a"].items()},
        "per_year_mean_b": {str(k): round(float(v), 6) for k, v in per_year["b"].items()},
        "mean_difference": round(float(np.mean(diffs)), 6),
        "method": "paired comparison on year-level means",
    }
    if n_years < 3:
        out.update({"p_value": None, "inference_supported": False,
                    "reason": f"only {n_years} years; paired testing not meaningful"})
        return out

    t_stat, t_p = sp_stats.ttest_rel(per_year["a"], per_year["b"])
    try:
        w_stat, w_p = sp_stats.wilcoxon(per_year["a"], per_year["b"])
    except Exception:
        w_stat, w_p = None, None

    sd = float(np.std(diffs, ddof=1))
    cohen_dz = float(np.mean(diffs) / sd) if sd > 0 else None
    se = sd / np.sqrt(n_years) if sd > 0 else 0.0
    t_crit = float(sp_stats.t.ppf(0.975, df=n_years - 1))

    out.update({
        "paired_t_statistic": round(float(t_stat), 4),
        "paired_t_p_value": round(float(t_p), 6),
        "wilcoxon_statistic": round(float(w_stat), 4) if w_stat is not None else None,
        "wilcoxon_p_value": round(float(w_p), 6) if w_p is not None else None,
        "cohens_dz": round(cohen_dz, 4) if cohen_dz is not None else None,
        "ci_95_mean_difference": [
            round(float(np.mean(diffs) - t_crit * se), 6),
            round(float(np.mean(diffs) + t_crit * se), 6),
        ],
        "inference_supported": True,
        "significant_at_0.05": bool(t_p < 0.05),
        "power_caveat": (
            f"n = {n_years} years. Non-significance here is weak evidence of "
            "no effect, not evidence of equivalence."
        ),
    })
    return out


def effective_sample_size(values: np.ndarray, clusters: np.ndarray) -> Dict[str, Any]:
    """Report how much the clustering costs in effective sample size.

    Uses the standard design-effect approximation
    ``n_eff = n / (1 + (m_bar - 1) * ICC)`` with the ICC estimated from a
    one-way ANOVA decomposition across clusters.
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    k = len(uniq)
    n = len(values)
    if k < 2 or n <= k:
        return {"n_observations": int(n), "n_clusters": int(k), "icc": None, "n_effective": int(n)}

    grand = float(np.mean(values))
    ss_between = float(sum(len(values[clusters == c]) * (np.mean(values[clusters == c]) - grand) ** 2 for c in uniq))
    ss_within = float(sum(np.sum((values[clusters == c] - np.mean(values[clusters == c])) ** 2) for c in uniq))
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    m_bar = n / k
    icc_num = ms_between - ms_within
    icc = float(icc_num / max(1e-12, ms_between + (m_bar - 1) * ms_within))
    icc = float(np.clip(icc, 0.0, 1.0))
    design_effect = 1.0 + (m_bar - 1.0) * icc
    return {
        "n_observations": int(n),
        "n_clusters": int(k),
        "mean_cluster_size": round(float(m_bar), 2),
        "icc": round(icc, 4),
        "design_effect": round(float(design_effect), 3),
        "n_effective": int(round(n / max(1.0, design_effect))),
    }


def nemenyi_critical_difference(k: int, n_blocks: int, alpha: float = 0.05) -> Dict[str, Any]:
    """Nemenyi critical difference for ``k`` methods over ``n_blocks`` blocks.

    ``CD = q_alpha * sqrt(k(k+1) / (6N))`` where ``q_alpha`` is the studentized
    range statistic at infinite degrees of freedom divided by sqrt(2). The
    previous implementation hard-coded ``2.728 if k == 5 else 2.569`` -- 2.569 is
    the value for k=4, so a six-method comparison silently used a critical value
    that was too small and therefore over-declared significance.

    The table is looked up by the actual ``k``; unknown ``k`` returns ``None``
    rather than a wrong number.
    """
    # q_alpha = q_{alpha, k, inf} / sqrt(2), the standard Nemenyi constants.
    Q_ALPHA_005 = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
        8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268,
        13: 3.313, 14: 3.354, 15: 3.391,
    }
    Q_ALPHA_010 = {
        2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693,
        8: 2.780, 9: 2.855, 10: 2.920,
    }
    table = Q_ALPHA_005 if abs(alpha - 0.05) < 1e-9 else (
        Q_ALPHA_010 if abs(alpha - 0.10) < 1e-9 else None
    )
    if table is None or k not in table or n_blocks < 1:
        return {
            "k": int(k), "n_blocks": int(n_blocks), "alpha": alpha,
            "q_alpha": None, "critical_difference": None,
            "reason": f"no tabulated q_alpha for k={k}, alpha={alpha}",
        }
    q = table[k]
    cd = float(q * np.sqrt((k * (k + 1)) / (6.0 * n_blocks)))
    return {
        "k": int(k),
        "n_blocks": int(n_blocks),
        "alpha": alpha,
        "q_alpha": q,
        "critical_difference": round(cd, 4),
        "formula": "CD = q_alpha * sqrt(k(k+1)/(6N))",
    }


def friedman_on_blocks(
    scores_by_method: Dict[str, np.ndarray],
    block_ids: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Friedman test where each *block* is an independent unit (e.g. a year).

    Per-observation Friedman testing over thousands of correlated county-year
    rows treats each row as an independent block, which it is not. Scores are
    first averaged within each block, so the number of blocks -- not the number
    of rows -- drives the test.
    """
    from scipy import stats as sp_stats

    names = list(scores_by_method.keys())
    k = len(names)
    blocks = np.asarray(block_ids)
    uniq_blocks = np.unique(blocks)
    n_blocks = len(uniq_blocks)

    out: Dict[str, Any] = {
        "methods": names,
        "k": k,
        "n_blocks": int(n_blocks),
        "block_unit": "year",
        "alpha": alpha,
    }

    if k < 3 or n_blocks < 3:
        out.update({
            "statistic": None, "p_value": None, "test_executed": False,
            "reason": (
                f"Friedman requires k>=3 methods and >=3 blocks; got k={k}, "
                f"blocks={n_blocks}. Ranking is reported as descriptive only."
            ),
        })
        return out

    block_means = np.array([
        [float(np.mean(scores_by_method[m][blocks == b])) for m in names]
        for b in uniq_blocks
    ])  # shape (n_blocks, k)

    stat, p = sp_stats.friedmanchisquare(*[block_means[:, j] for j in range(k)])
    # Average rank per method (rank 1 = best = lowest score).
    ranks = np.apply_along_axis(sp_stats.rankdata, 1, block_means)
    avg_ranks = ranks.mean(axis=0)

    cd_info = nemenyi_critical_difference(k, n_blocks, alpha=alpha)
    out.update({
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "test_executed": True,
        "significant_at_alpha": bool(p < alpha),
        "average_ranks": {names[j]: round(float(avg_ranks[j]), 4) for j in range(k)},
        "nemenyi": cd_info,
        "block_means": {
            str(uniq_blocks[i]): {names[j]: round(float(block_means[i, j]), 4) for j in range(k)}
            for i in range(n_blocks)
        },
    })

    # Post-hoc pairwise comparisons only if the omnibus test rejects.
    cd = cd_info.get("critical_difference")
    pairwise: List[Dict[str, Any]] = []
    if out["significant_at_alpha"] and cd is not None:
        for i in range(k):
            for j in range(i + 1, k):
                d = float(abs(avg_ranks[i] - avg_ranks[j]))
                pairwise.append({
                    "method_a": names[i], "method_b": names[j],
                    "rank_difference": round(d, 4),
                    "exceeds_cd": bool(d > cd),
                    "better": names[i] if avg_ranks[i] < avg_ranks[j] else names[j],
                })
    out["nemenyi_pairwise"] = pairwise
    out["n_significant_pairs"] = sum(1 for p_ in pairwise if p_["exceeds_cd"])
    return out
