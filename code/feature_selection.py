"""
feature_selection.py - Strict 4-Method Consensus Feature Selection for Paper 3.

Combines:
1. Mutual Information
2. LightGBM Feature Importance
3. Random Forest / SHAP Importance
4. Permutation Importance

Protects methodology disaster variables & static environmental context.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

import config as cfg
from utils import log_decision, save_report, save_report_markdown, save_report_csv

logger = logging.getLogger("paper3")


def select_features(
    df: pd.DataFrame,
    target: str = cfg.PRIMARY_TARGET,
) -> Tuple[List[str], Dict[str, Any]]:
    """Execute 4-method strict consensus feature selection.

    Parameters
    ----------
    df : pd.DataFrame
        Processed dataset (training fold ONLY).
    target : str
        Target variable name.

    Returns
    -------
    Tuple[List[str], Dict[str, Any]]
        List of consensus-selected feature names and detailed selection report.
    """
    logger.info("=" * 60)
    logger.info("STRICT 4-METHOD CONSENSUS FEATURE SELECTION")
    logger.info("=" * 60)

    report: Dict[str, Any] = {}

    exclude = (
        cfg.ID_COLS + cfg.TARGET_COLS + cfg.CONSTANT_COLS
        + cfg.DATA_AVAILABILITY_FLAGS + cfg.YIELD_PRESENCE_COLS
        + cfg.REDUNDANT_TARGET_COLS
        + ["Split", "ENSO_Phase", "Phenological_Window", "Year_Type"]
    )
    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]
    logger.info("Evaluating %d candidate features", len(feature_cols))

    valid = df[df[target].notna()].copy()
    for col in feature_cols:
        if valid[col].isnull().any():
            bad_row = valid[valid[col].isnull()].iloc[0]
            geoid_val = bad_row.get("GEOID", "unknown")
            year_val = bad_row.get("Year", "unknown")
            raise ValueError(
                f"FEATURE SELECTION ERROR: Unimputed NaN in feature column '{col}'! "
                f"[GEOID: {geoid_val}, Year: {year_val}]"
            )
    X = valid[feature_cols].values
    y = valid[target].values

    # ── 1. Mutual Information ────────────────────────────────
    logger.info("  Computing Mutual Information Regression scores...")
    mi_scores = mutual_info_regression(X, y, random_state=cfg.RANDOM_SEED)
    mi_rank = np.argsort(mi_scores)[::-1]
    mi_dict = [{"feature": feature_cols[idx], "score": round(float(mi_scores[idx]), 6)} for idx in mi_rank]
    report["mutual_information"] = mi_dict

    # ── 2. LightGBM Feature Importance ───────────────────────
    logger.info("  Computing LightGBM feature importances...")
    lgbm_dict = _lgbm_importance(valid, feature_cols, target)
    report["lgbm_importance"] = lgbm_dict

    # ── 3. Random Forest Feature Importance ───────────────────
    logger.info("  Computing Random Forest feature importances...")
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=cfg.RANDOM_SEED, n_jobs=-1)
    rf.fit(X, y)
    rf_imp = rf.feature_importances_
    rf_rank = np.argsort(rf_imp)[::-1]
    rf_dict = [{"feature": feature_cols[idx], "importance": round(float(rf_imp[idx]), 6)} for idx in rf_rank]
    report["rf_importance"] = rf_dict

    # ── 4. Permutation Importance ─────────────────────────────
    logger.info("  Computing Permutation Importance...")
    perm_imp = permutation_importance(rf, X, y, n_repeats=5, random_state=cfg.RANDOM_SEED, n_jobs=-1)
    perm_scores = perm_imp.importances_mean
    perm_rank = np.argsort(perm_scores)[::-1]
    perm_dict = [{"feature": feature_cols[idx], "importance": round(float(perm_scores[idx]), 6)} for idx in perm_rank]
    report["permutation_importance"] = perm_dict

    # ── 5. Consensus Decision & Protected Features Protection ──
    protected = set(
        cfg.CDHW_COLS + cfg.WEATHER_COLS + cfg.ENSO_COLS
        + cfg.DROUGHT_COLS + cfg.STORM_COLS + cfg.STATIC_CONTEXT_COLS
        + ["CDHW_Veg_Severity", "CDHW_Silking_Severity", "CDHW_GrainFill_Severity",
           "Yield_lag1", "Yield_lag2", "Hist_Normal_Precip_growseason_mm", "Hist_Normal_GDD_Accumulated"]
        + [c for c in feature_cols if c.startswith("Fourier_")]
    )
    protected_in_features = [c for c in feature_cols if c in protected]

    # Threshold for top K (top 65% of candidates per method)
    k_top = max(15, int(0.65 * len(feature_cols)))

    votes = {f: 0 for f in feature_cols}
    for item in mi_dict[:k_top]: votes[item["feature"]] += 1
    for item in lgbm_dict[:k_top]: votes[item["feature"]] += 1
    for item in rf_dict[:k_top]: votes[item["feature"]] += 1
    for item in perm_dict[:k_top]: votes[item["feature"]] += 1

    # Keep features with >= 2 votes OR protected methodology features
    selected_features = [f for f in feature_cols if votes[f] >= 2 or f in protected]
    dropped_features = [f for f in feature_cols if f not in selected_features]

    report["protected_features"] = protected_in_features
    report["consensus_votes"] = votes
    report["selected_features"] = selected_features
    report["dropped_features"] = dropped_features

    # Rank consistency dataframe
    ranks_summary = []
    for f in feature_cols:
        r_mi = next((i for i, e in enumerate(mi_dict) if e["feature"] == f), len(feature_cols))
        r_lgb = next((i for i, e in enumerate(lgbm_dict) if e["feature"] == f), len(feature_cols))
        r_rf = next((i for i, e in enumerate(rf_dict) if e["feature"] == f), len(feature_cols))
        r_perm = next((i for i, e in enumerate(perm_dict) if e["feature"] == f), len(feature_cols))

        avg_r = (r_mi + r_lgb + r_rf + r_perm) / 4.0
        ranks_summary.append({
            "feature": f,
            "votes": votes[f],
            "selected": f in selected_features,
            "protected": f in protected,
            "avg_rank": round(avg_r, 2),
            "rank_mi": r_mi + 1,
            "rank_lgbm": r_lgb + 1,
            "rank_rf": r_rf + 1,
            "rank_perm": r_perm + 1,
        })

    ranks_df = pd.DataFrame(ranks_summary).sort_values("avg_rank")
    save_report_csv(ranks_df, "feature_selection_ranks.csv")

    md = f"""# Feature Selection Consensus Report (§7)

## Method Consensus Summary
- **Total Candidate Features Evaluated**: `{len(feature_cols)}`
- **Consensus Selected Features**: `{len(selected_features)}`
- **Features Dropped**: `{len(dropped_features)}`
- **Methodology Protected Features Preserved**: `{len(protected_in_features)}`

## Top 20 Selected Features by Consensus Rank
"""
    try:
        md += ranks_df.head(20).to_markdown(index=False) + "\n"
    except Exception:
        md += ranks_df.head(20).to_string(index=False) + "\n"

    save_report_markdown(md, "feature_selection_report.md")
    save_report(report, "feature_selection_report.json")
    logger.info("Consensus feature selection report exported -> feature_selection_report.md")

    return selected_features, report


def _lgbm_importance(
    df: pd.DataFrame,
    feature_cols: List[str],
    target: str,
) -> List[Dict[str, Any]]:
    """Compute LightGBM gain feature importances."""
    import lightgbm as lgb
    X = df[feature_cols].values
    y = df[target].values

    model = lgb.LGBMRegressor(**cfg.LGBM_PARAMS)
    model.fit(X, y)

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    results = []
    for idx in sorted_idx:
        results.append({"feature": feature_cols[idx], "importance": float(importances[idx])})
    return results
