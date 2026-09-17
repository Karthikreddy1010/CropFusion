"""
visualization.py - Plot generation for Paper 3.

Generates all plots specified in the methodology and STEP 11:
- Missing-value heatmap
- Correlation heatmaps (Pearson & Spearman)
- VIF bar chart
- Feature importance plot
- Target distribution plots
- Outlier visualization
- Calibration curves (§5.6)
- PICP & MPIW comparisons across year-types (§5.7)
- LOSO-CV fold results
- Residual plots
- Interval width distributions
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import scipy.stats as sp_stats
try:
    import torch
except ImportError:
    torch = None

import config as cfg
from aci_calibrator import CalibrationResult

logger = logging.getLogger("paper3")

# ── Style setup ──────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": cfg.PLOT_DPI,
    "savefig.dpi": cfg.PLOT_DPI,
    "savefig.bbox": "tight",
    "figure.figsize": (12, 8),
})


def _save(fig: plt.Figure, name: str) -> Path:
    path = cfg.PLOT_DIR / f"{name}.{cfg.PLOT_FORMAT}"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Plot saved → %s", path)
    return path


# ─────────────────────────────────────────────────────────────
# 1. Missing-value heatmap
# ─────────────────────────────────────────────────────────────

def plot_missing_values(df: pd.DataFrame) -> Path:
    """Missing-value heatmap showing pattern across columns."""
    cols_with_missing = [c for c in df.columns if df[c].isnull().any()]
    if not cols_with_missing:
        logger.info("No missing values to plot")
        return None

    fig, ax = plt.subplots(figsize=(14, 6))
    missing_matrix = df[cols_with_missing].isnull().astype(int)

    # Sample rows for visual clarity
    if len(missing_matrix) > 500:
        missing_matrix = missing_matrix.sample(500, random_state=cfg.RANDOM_SEED)

    sns.heatmap(
        missing_matrix.T, cbar=False, cmap="YlOrRd",
        yticklabels=True, xticklabels=False, ax=ax,
    )
    ax.set_title("Missing Value Heatmap (Yellow = Present, Red = Missing)")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Features")
    return _save(fig, "missing_values_heatmap")


# ─────────────────────────────────────────────────────────────
# 2. Correlation heatmaps
# ─────────────────────────────────────────────────────────────

def plot_correlation_heatmap(
    df: pd.DataFrame,
    method: str = "pearson",
    feature_cols: Optional[List[str]] = None,
) -> Path:
    """Correlation heatmap (Pearson or Spearman)."""
    if feature_cols is None:
        exclude = cfg.ID_COLS + cfg.CONSTANT_COLS + cfg.REDUNDANT_TARGET_COLS
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]

    # Limit to top features for readability
    if len(feature_cols) > 30:
        feature_cols = feature_cols[:30]

    corr = df[feature_cols].corr(method=method)
    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, cmap="RdBu_r", center=0,
        annot=False, square=True, linewidths=0.5, ax=ax,
        vmin=-1, vmax=1,
    )
    ax.set_title(f"{method.capitalize()} Correlation Heatmap")
    return _save(fig, f"correlation_{method}")


# ─────────────────────────────────────────────────────────────
# 3. VIF bar chart
# ─────────────────────────────────────────────────────────────

def plot_vif(vif_results: List[Dict[str, Any]], top_n: int = 25) -> Path:
    """Bar chart of Variance Inflation Factors."""
    if not vif_results:
        logger.info("No VIF data to plot")
        return None

    vif_df = pd.DataFrame(vif_results).head(top_n)
    vif_df = vif_df[vif_df["vif"] > 0]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["#e74c3c" if v > 10 else "#f39c12" if v > 5 else "#2ecc71"
              for v in vif_df["vif"]]
    ax.barh(vif_df["feature"], vif_df["vif"], color=colors)
    ax.axvline(x=10, color="red", linestyle="--", label="VIF=10 threshold")
    ax.axvline(x=5, color="orange", linestyle="--", label="VIF=5 threshold")
    ax.set_xlabel("Variance Inflation Factor")
    ax.set_title("VIF Analysis")
    ax.legend()
    ax.invert_yaxis()
    return _save(fig, "vif_analysis")


# ─────────────────────────────────────────────────────────────
# 4. Feature importance
# ─────────────────────────────────────────────────────────────

def plot_feature_importance(
    importance_data: List[Dict[str, Any]],
    title: str = "LightGBM Feature Importance",
    top_n: int = 20,
) -> Path:
    """Horizontal bar chart of feature importances."""
    if not importance_data:
        return None

    imp_df = pd.DataFrame(importance_data).head(top_n)
    if "importance" in imp_df.columns:
        key = "importance"
    elif "score" in imp_df.columns:
        key = "score"
    elif "mi_score" in imp_df.columns:
        key = "mi_score"
    else:
        num_cols = [c for c in imp_df.columns if c != "feature"]
        key = num_cols[0] if num_cols else imp_df.columns[1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(imp_df["feature"], imp_df[key], color="#3498db")
    ax.set_xlabel(key.replace("_", " ").title())
    ax.set_title(title)
    ax.invert_yaxis()
    return _save(fig, f"feature_importance_{title.lower().replace(' ', '_')}")


# ─────────────────────────────────────────────────────────────
# 5. Target distribution
# ─────────────────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame) -> Path:
    """Distribution plots for target variables."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    targets = [cfg.PRIMARY_TARGET, cfg.SECONDARY_TARGET]
    titles = ["Corn Yield (t/ha)", "Soybean Yield (t/ha)"]

    for ax, target, title in zip(axes, targets, titles):
        if target not in df.columns:
            ax.set_axis_off()
            continue
        data = df[target].dropna()
        ax.hist(data, bins=50, color="#2ecc71", alpha=0.7, edgecolor="white")
        ax.axvline(data.mean(), color="red", linestyle="--",
                   label=f"Mean={data.mean():.2f}")
        ax.set_xlabel(title)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {title}")
        ax.legend()

    fig.tight_layout()
    return _save(fig, "target_distributions")


# ─────────────────────────────────────────────────────────────
# 6. Calibration curves (§5.6)
# ─────────────────────────────────────────────────────────────

def plot_calibration_curves(
    results: Dict[str, Dict[str, List[float]]],
) -> Path:
    """Plot expected vs observed coverage for multiple methods (§5.6)."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.5)

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    for i, (method, data) in enumerate(results.items()):
        color = colors[i % len(colors)]
        ax.plot(
            data["nominal"], data["empirical"],
            "o-", color=color, label=method, markersize=8,
        )

    ax.set_xlabel("Nominal Coverage")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Calibration Curves (§5.6)")
    ax.legend(loc="lower right")
    ax.set_xlim(0.75, 1.0)
    ax.set_ylim(0.75, 1.0)
    ax.set_aspect("equal")
    return _save(fig, "calibration_curves")


# ─────────────────────────────────────────────────────────────
# 7. PICP comparison across methods
# ─────────────────────────────────────────────────────────────

def plot_picp_comparison(
    eval_reports: Dict[str, Dict[str, Any]],
) -> Path:
    """Bar chart comparing PICP across methods and year-types."""
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = list(eval_reports.keys())
    picp_values = [eval_reports[m]["uncertainty"]["picp"] for m in methods]

    colors = ["#e74c3c" if p < 0.90 else "#2ecc71" for p in picp_values]
    bars = ax.bar(methods, picp_values, color=colors, edgecolor="white")
    ax.axhline(y=0.90, color="red", linestyle="--", label="90% Nominal Target")
    ax.set_ylabel("PICP")
    ax.set_title("Prediction Interval Coverage Probability (PICP) Comparison")
    ax.legend()

    for bar, val in zip(bars, picp_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10,
        )

    return _save(fig, "picp_comparison")


# ─────────────────────────────────────────────────────────────
# 8. MPIW comparison
# ─────────────────────────────────────────────────────────────

def plot_mpiw_comparison(
    eval_reports: Dict[str, Dict[str, Any]],
) -> Path:
    """Bar chart comparing MPIW across methods."""
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = list(eval_reports.keys())
    mpiw_values = [eval_reports[m]["uncertainty"]["mpiw"] for m in methods]

    ax.bar(methods, mpiw_values, color="#3498db", edgecolor="white")
    ax.set_ylabel("MPIW (t/ha)")
    ax.set_title("Mean Prediction Interval Width (MPIW) Comparison")

    for i, (m, v) in enumerate(zip(methods, mpiw_values)):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    return _save(fig, "mpiw_comparison")


# ─────────────────────────────────────────────────────────────
# 9. Interval width by year-type (§5.7) — violin/boxplot
# ─────────────────────────────────────────────────────────────

def plot_interval_width_by_yeartype(
    results: Dict[str, CalibrationResult],
    year_types: np.ndarray,
) -> Path:
    """Violin/box plot of interval widths by year-type (§5.7)."""
    records = []
    for method, result in results.items():
        widths = result.q_hi - result.q_lo
        for w, yt in zip(widths, year_types):
            records.append({"Method": method, "Year_Type": yt, "Width": w})

    plot_df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.violinplot(
        data=plot_df, x="Year_Type", y="Width", hue="Method",
        ax=ax, inner="box", palette="Set2",
        order=["Normal", "Moderate", "Extreme"],
    )
    ax.set_xlabel("Year Type")
    ax.set_ylabel("Interval Width (t/ha)")
    ax.set_title("Interval Width Distribution by Year-Type Category (§5.7)")
    ax.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    return _save(fig, "interval_width_by_yeartype")


# ─────────────────────────────────────────────────────────────
# 10. LOSO-CV fold results
# ─────────────────────────────────────────────────────────────

def plot_loso_cv_results(
    fold_metrics: List[Dict[str, Any]],
    loso_predictions: Optional[pd.DataFrame] = None,
) -> Path:
    """Publication-quality 6-panel figure of per-fold LOSO-CV metrics (§14).
    
    Panels:
      A. RMSE by held-out state (macro mean & pooled lines)
      B. R² by held-out state (macro mean & pooled lines)
      C. PICP by held-out state (nominal 90% reference line & macro mean)
      D. MPIW by held-out state (macro mean)
      E. Winkler Score by held-out state (macro mean)
      F. Ensemble Weight Allocation (NeuralCQR vs LightGBM)
    """
    if not fold_metrics:
        return None

    fold_df = pd.DataFrame(fold_metrics)
    valid_df = fold_df[fold_df["rmse"].notna()].copy() if "rmse" in fold_df.columns else fold_df.copy()

    # Enforce standard locked state order
    state_order = getattr(cfg, "LOSO_STATES", ["Illinois", "Indiana", "Iowa", "Minnesota", "Missouri", "Ohio"])
    valid_df["state_cat"] = pd.Categorical(valid_df["state"], categories=state_order, ordered=True)
    valid_df = valid_df.sort_values("state_cat").reset_index(drop=True)

    # Compute pooled metrics if predictions are available
    pooled_metrics = {}
    if loso_predictions is None or len(loso_predictions) == 0:
        pred_path = getattr(cfg, "OUTPUT_DIR", Path("outputs")) / "predictions" / "loso_predictions.csv"
        if pred_path.exists():
            try:
                loso_predictions = pd.read_csv(pred_path)
            except Exception:
                pass

    if loso_predictions is not None and len(loso_predictions) > 0:
        y_tr = loso_predictions["y_true"].values
        y_pr = loso_predictions["y_pred"].values
        q_l = loso_predictions["lower_bound"].values
        q_h = loso_predictions["upper_bound"].values
        from evaluation import rmse as calc_rmse, r_squared as calc_r2, picp as calc_picp, mpiw as calc_mpiw, winkler_score as calc_ws
        pooled_metrics["rmse"] = float(calc_rmse(y_tr, y_pr))
        pooled_metrics["r_squared"] = float(calc_r2(y_tr, y_pr))
        pooled_metrics["picp"] = float(calc_picp(y_tr, q_l, q_h))
        pooled_metrics["mpiw"] = float(calc_mpiw(q_l, q_h))
        pooled_metrics["winkler_score"] = float(calc_ws(y_tr, q_l, q_h))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=getattr(cfg, "PLOT_DPI", 300))
    axes_flat = axes.flat

    # Colors
    c_blue = "#1f77b4"
    c_orange = "#ff7f0e"
    c_green = "#2ca02c"
    c_purple = "#9467bd"
    c_teal = "#17becf"

    # Panel A: RMSE
    ax = axes_flat[0]
    bars = ax.bar(valid_df["state"], valid_df["rmse"], color=c_blue, edgecolor="white", width=0.55)
    mean_rmse = valid_df["rmse"].mean()
    ax.axhline(mean_rmse, color="crimson", linestyle="--", linewidth=1.5, label=f"Macro Mean = {mean_rmse:.4f}")
    if "rmse" in pooled_metrics:
        ax.axhline(pooled_metrics["rmse"], color="darkblue", linestyle=":", linewidth=1.5, label=f"Pooled = {pooled_metrics['rmse']:.4f}")
    ax.set_title("(A) Root Mean Squared Error (RMSE)", fontsize=11, fontweight="bold")
    ax.set_ylabel("RMSE (t/ha)", fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8, loc="upper right")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(valid_df["rmse"].max() * 1.25, 2.0))

    # Panel B: R²
    ax = axes_flat[1]
    bars = ax.bar(valid_df["state"], valid_df["r_squared"], color=c_teal, edgecolor="white", width=0.55)
    mean_r2 = valid_df["r_squared"].mean()
    ax.axhline(mean_r2, color="crimson", linestyle="--", linewidth=1.5, label=f"Macro Mean = {mean_r2:.4f}")
    if "r_squared" in pooled_metrics:
        ax.axhline(pooled_metrics["r_squared"], color="darkblue", linestyle=":", linewidth=1.5, label=f"Pooled = {pooled_metrics['r_squared']:.4f}")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.8)
    ax.set_title("(B) Coefficient of Determination (R²)", fontsize=11, fontweight="bold")
    ax.set_ylabel("R²", fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8, loc="lower right")
    for b in bars:
        h = b.get_height()
        va_pos = "bottom" if h >= 0 else "top"
        offset = 0.015 if h >= 0 else -0.04
        ax.text(b.get_x() + b.get_width() / 2, h + offset, f"{h:.3f}", ha="center", va=va_pos, fontsize=8)
    ax.set_ylim(min(0, valid_df["r_squared"].min() - 0.1), max(1.0, valid_df["r_squared"].max() * 1.2))

    # Panel C: PICP (with nominal 90% reference line)
    ax = axes_flat[2]
    # Highlight Minnesota if undercovered
    colors = [c_orange if s == "Minnesota" and v < 0.90 else c_green for s, v in zip(valid_df["state"], valid_df["picp"])]
    bars = ax.bar(valid_df["state"], valid_df["picp"], color=colors, edgecolor="white", width=0.55)
    ax.axhline(0.90, color="black", linestyle="-", linewidth=1.8, label="Nominal Target = 0.90")
    mean_picp = valid_df["picp"].mean()
    ax.axhline(mean_picp, color="crimson", linestyle="--", linewidth=1.5, label=f"Macro Mean = {mean_picp:.4f}")
    if "picp" in pooled_metrics:
        ax.axhline(pooled_metrics["picp"], color="darkblue", linestyle=":", linewidth=1.5, label=f"Pooled = {pooled_metrics['picp']:.4f}")
    ax.set_title("(C) Prediction Interval Coverage (PICP)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Coverage Probability", fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8, loc="lower right")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0.75, 1.05)

    # Panel D: MPIW (Sharpness)
    ax = axes_flat[3]
    bars = ax.bar(valid_df["state"], valid_df["mpiw"], color="#34495e", edgecolor="white", width=0.55)
    mean_mpiw = valid_df["mpiw"].mean()
    ax.axhline(mean_mpiw, color="crimson", linestyle="--", linewidth=1.5, label=f"Macro Mean = {mean_mpiw:.4f}")
    if "mpiw" in pooled_metrics:
        ax.axhline(pooled_metrics["mpiw"], color="darkblue", linestyle=":", linewidth=1.5, label=f"Pooled = {pooled_metrics['mpiw']:.4f}")
    ax.set_title("(D) Mean Prediction Interval Width (MPIW)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Width (t/ha)", fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8, loc="upper right")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(valid_df["mpiw"].max() * 1.25, 7.0))

    # Panel E: Winkler Score
    ax = axes_flat[4]
    winkler_vals = valid_df["winkler_score"] if "winkler_score" in valid_df.columns else np.zeros(len(valid_df))
    bars = ax.bar(valid_df["state"], winkler_vals, color=c_purple, edgecolor="white", width=0.55)
    mean_ws = winkler_vals.mean()
    ax.axhline(mean_ws, color="crimson", linestyle="--", linewidth=1.5, label=f"Macro Mean = {mean_ws:.4f}")
    if "winkler_score" in pooled_metrics:
        ax.axhline(pooled_metrics["winkler_score"], color="darkblue", linestyle=":", linewidth=1.5, label=f"Pooled = {pooled_metrics['winkler_score']:.4f}")
    ax.set_title("(E) Winkler Score (Interval Loss)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Winkler Score", fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8, loc="upper right")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(winkler_vals.max() * 1.25, 10.0))

    # Panel F: Ensemble Weight Allocation
    ax = axes_flat[5]
    w_neural = valid_df["neuralcqr_ensemble_weight"].astype(float).values if "neuralcqr_ensemble_weight" in valid_df.columns else np.zeros(len(valid_df))
    w_lgb = np.array([round(1.0 - w, 2) for w in w_neural])
    x_pos = np.arange(len(valid_df))

    p1 = ax.bar(valid_df["state"], w_neural, width=0.55, label="NeuralCQR Weight", color="#2980b9", edgecolor="white")
    p2 = ax.bar(valid_df["state"], w_lgb, bottom=w_neural, width=0.55, label="LightGBM Weight", color="#e67e22", edgecolor="white")
    ax.set_title("(F) Ensemble Weight Allocation (Tuned on Dev Val)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ensemble Weight", fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 1.15)
    for idx, (wn, wl) in enumerate(zip(w_neural, w_lgb)):
        if wn > 0.05:
            ax.text(idx, wn / 2, f"{wn:.2f}", ha="center", va="center", color="white", fontweight="bold", fontsize=8)
        if wl > 0.05:
            ax.text(idx, wn + wl / 2, f"{wl:.2f}", ha="center", va="center", color="white", fontweight="bold", fontsize=8)

    fig.tight_layout()
    return _save(fig, "loso_cv_results")


# ─────────────────────────────────────────────────────────────
# 11. Residual plots
# ─────────────────────────────────────────────────────────────

def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Analysis",
) -> Path:
    """Residual distribution and scatter plots."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residual scatter
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, color="#3498db")
    axes[0].axhline(y=0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"{title} — Residuals vs Predicted")

    # Residual histogram
    axes[1].hist(residuals, bins=50, color="#2ecc71", alpha=0.7, edgecolor="white")
    axes[1].axvline(x=0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"{title} — Residual Distribution")

    fig.tight_layout()
    return _save(fig, f"residuals_{title.lower().replace(' ', '_')}")


# ─────────────────────────────────────────────────────────────
# 12. ACI year-by-year tracking
# ─────────────────────────────────────────────────────────────

def plot_aci_tracking(history: List[Dict[str, Any]]) -> Path:
    """Plot ACI alpha_t, PICP, and threshold evolution over test years."""
    if not history:
        return None

    years = [h["year"] for h in history]
    picp_vals = [h.get("picp", h.get("picp_year", 0.0)) for h in history]
    alpha_vals = [h.get("alpha_t", 0.10) for h in history]
    mpiw_vals = [h.get("mean_interval_width", h.get("mean_width", 0.0)) for h in history]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(years, picp_vals, "o-", color="#2ecc71", markersize=6)
    axes[0].axhline(y=0.90, color="red", linestyle="--", label="90% Target")
    axes[0].set_ylabel("PICP")
    axes[0].set_title("ACI Online Tracking (§4.3)")
    axes[0].legend()

    axes[1].plot(years, alpha_vals, "s-", color="#e74c3c", markersize=6)
    axes[1].set_ylabel("α_t (Miscoverage Rate)")

    axes[2].plot(years, mpiw_vals, "D-", color="#3498db", markersize=6)
    axes[2].set_ylabel("MPIW (t/ha)")
    axes[2].set_xlabel("Year")

    fig.tight_layout()
    return _save(fig, "aci_tracking")


# ─────────────────────────────────────────────────────────────
# Master visualization runner
# ─────────────────────────────────────────────────────────────

def run_all_visualizations(
    df: pd.DataFrame,
    vif_results: Optional[List[Dict]] = None,
    importance_data: Optional[List[Dict]] = None,
    mi_data: Optional[List[Dict]] = None,
) -> List[Path]:
    """Generate all EDA and preprocessing visualizations."""
    logger.info("=" * 60)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 60)

    paths = []

    p = plot_missing_values(df)
    if p:
        paths.append(p)

    paths.append(plot_correlation_heatmap(df, "pearson"))
    paths.append(plot_correlation_heatmap(df, "spearman"))

    if vif_results:
        p = plot_vif(vif_results)
        if p:
            paths.append(p)

    if importance_data:
        p = plot_feature_importance(importance_data, "LightGBM Feature Importance")
        if p:
            paths.append(p)

    if mi_data:
        p = plot_feature_importance(mi_data, "Mutual Information Scores", top_n=20)
        if p:
            paths.append(p)

    paths.append(plot_target_distribution(df))

    logger.info("Generated %d plots", len(paths))
    return paths


def plot_all_calibration_curves(
    all_results: Dict[str, CalibrationResult]
) -> Path:
    """Generate statistically valid single-level nominal 90% calibration evaluation (§5.6).

    Replaces invalid multi-level quantile scaling from a single interval with
    the mathematically sound nominal 90% coverage analysis across all conformal methods.
    """
    from utils import save_report

    nom_cov = 1.0 - float(cfg.NOMINAL_ALPHA)
    curves_data = {
        "protocol": "Single-level nominal 90% conformal coverage evaluation (§5.6)",
        "nominal_coverage": round(nom_cov, 4),
        "nominal_alpha": round(float(cfg.NOMINAL_ALPHA), 4),
        "methods": {},
    }

    method_labels = []
    emp_coverages = []
    ci_lowers = []
    ci_uppers = []

    for method_name, res in all_results.items():
        y = res.y_true
        n = len(y)
        covered = (res.q_lo <= y) & (y <= res.q_hi)
        cov = float(covered.mean())
        ace_val = abs(cov - nom_cov)
        se = float(np.sqrt(cov * (1.0 - cov) / max(1, n)))
        ci_lo = max(0.0, cov - 1.96 * se)
        ci_hi = min(1.0, cov + 1.96 * se)
        mean_width = float(np.mean(res.q_hi - res.q_lo))

        curves_data["methods"][method_name] = {
            "nominal_coverage": round(nom_cov, 4),
            "empirical_coverage": round(cov, 4),
            "absolute_calibration_error": round(ace_val, 4),
            "mean_prediction_interval_width": round(mean_width, 4),
            "coverage_standard_error": round(se, 4),
            "coverage_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "n_test_samples": int(n),
        }

        clean_label = method_name.replace("_", " ").title()
        method_labels.append(clean_label)
        emp_coverages.append(cov)
        ci_lowers.append(cov - ci_lo)
        ci_uppers.append(ci_hi - cov)

    save_report(curves_data, "calibration_report.json")

    # Plot Nominal 90% vs Empirical Coverage
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(method_labels))
    colors = ["#2ecc71" if abs(c - nom_cov) <= 0.05 else "#3498db" for c in emp_coverages]

    bars = ax.bar(x_pos, emp_coverages, yerr=[ci_lowers, ci_uppers], capsize=5,
                  color=colors, edgecolor="#2c3e50", alpha=0.85, width=0.55)
    ax.axhline(nom_cov, color="#e74c3c", linestyle="--", linewidth=2,
               label=f"Nominal Target ({int(nom_cov*100)}%)")

    for bar, c, se in zip(bars, emp_coverages, ci_lowers):
        height = bar.get_height()
        ax.annotate(f"{height*100:.1f}%\n(ACE={abs(height-nom_cov):.3f})",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Empirical Coverage on TEST (2019–2023)", fontsize=11)
    ax.set_title("Conformal Calibration & Empirical Coverage at Nominal 90% (§5.6)", fontsize=13, fontweight="bold")
    ax.set_ylim(0.70, 1.02)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    path = _save(fig, "calibration_curves")

    # Mirror to calibration directory
    try:
        cal_path = cfg.CALIBRATION_DIR / f"calibration_curves.{cfg.PLOT_FORMAT}"
        fig.savefig(cal_path)
        plt.close(fig)
    except Exception:
        pass

    return path


def export_shap_consistency_report(
    models_dict: Dict[str, Any],
    feature_cols: List[str],
    X_sample: Optional[np.ndarray] = None,
    max_background: int = 200,
    max_eval: int = 500,
) -> Dict[str, Any]:
    """Compute genuine SHAP attributions across backbones and compare them (§5.8).

    What was wrong before
    ---------------------
    The previous implementation was not SHAP at all. For tree models it read
    ``feature_importances_`` (split-gain importance), and for the NeuralCQR
    backbone -- which has no such attribute -- it fell through to
    ``np.abs(np.random.RandomState(42).randn(d))``, i.e. **random numbers
    presented as feature attributions**. ``X_sample`` was accepted and never
    used. One of the four "models" in the cross-model agreement statistic was
    therefore pure noise, which is the main reason the reported similarity was
    low. The report also hard-coded the phrase "High cross-backbone attribution
    alignment" regardless of the computed value.

    What happens now
    ----------------
    * Tree backbones use ``shap.TreeExplainer`` on real data.
    * The neural backbone uses ``shap.GradientExplainer`` against the point head.
    * If a model's attributions genuinely cannot be computed, it is recorded as
      unavailable and **excluded** from the agreement statistics. Nothing is
      fabricated to fill the gap.
    * The interpretation wording is derived from the computed value against
      stated thresholds rather than asserted.

    Exports shap_consistency_report.{json,csv,md}.
    """
    from utils import save_report_markdown, save_report_csv, save_report

    logger.info("Computing SHAP attributions across %d backbones", len(models_dict))

    d = len(feature_cols)
    if X_sample is None or len(X_sample) == 0:
        raise ValueError(
            "export_shap_consistency_report requires real feature data; "
            "synthetic input would not produce meaningful attributions"
        )
    X_sample = np.asarray(X_sample, dtype=np.float64)
    if X_sample.shape[1] != d:
        raise ValueError(f"X_sample has {X_sample.shape[1]} columns but {d} feature names given")

    rng = np.random.RandomState(cfg.RANDOM_SEED)
    n = len(X_sample)
    bg_idx = rng.choice(n, size=min(max_background, n), replace=False)
    ev_idx = rng.choice(n, size=min(max_eval, n), replace=False)
    X_bg = X_sample[bg_idx]
    X_ev = X_sample[ev_idx]

    import shap as _shap

    attributions: Dict[str, np.ndarray] = {}
    provenance: Dict[str, Dict[str, Any]] = {}

    for name, mset in models_dict.items():
        try:
            pm = getattr(mset, "point_model", mset)
            is_neural = bool(getattr(mset, "is_neural", False))

            if is_neural:
                # Wrap the multi-head net so SHAP sees a single scalar output
                # (the mean head, which is what RMSE/MAE/R2 are computed from).
                class _MeanHead(torch.nn.Module):
                    def __init__(self, net):
                        super().__init__()
                        self.net = net

                    def forward(self, x):
                        return self.net(x)[0]

                wrapped = _MeanHead(pm).eval()
                explainer = _shap.GradientExplainer(
                    wrapped, torch.tensor(X_bg, dtype=torch.float32)
                )
                sv = explainer.shap_values(torch.tensor(X_ev, dtype=torch.float32))
                if isinstance(sv, list):
                    sv = sv[0]
                sv = np.asarray(sv)
                method = "shap.GradientExplainer (mean head)"
            else:
                try:
                    explainer = _shap.TreeExplainer(pm)
                    sv = explainer.shap_values(X_ev, check_additivity=False)
                    method = "shap.TreeExplainer"
                except Exception as tree_err:
                    # XGBoost >= 2 can serialise base_score as "[5.1e-2]", which
                    # shap's tree parser cannot read. Normalise it and retry, then
                    # fall back to a model-agnostic (but still genuine) SHAP
                    # explainer rather than fabricating attributions.
                    sv = None
                    try:
                        import json as _json
                        booster = pm.get_booster()
                        bcfg = _json.loads(booster.save_config())
                        bs = bcfg["learner"]["learner_model_param"]["base_score"]
                        if isinstance(bs, str) and bs.startswith("["):
                            bcfg["learner"]["learner_model_param"]["base_score"] = \
                                str(float(bs.strip("[]")))
                            booster.load_config(_json.dumps(bcfg))
                            explainer = _shap.TreeExplainer(booster)
                            sv = explainer.shap_values(X_ev, check_additivity=False)
                            method = "shap.TreeExplainer (base_score normalised)"
                    except Exception:
                        sv = None
                    if sv is None:
                        n_perm = min(120, len(X_ev))
                        explainer = _shap.PermutationExplainer(pm.predict, X_bg)
                        sv = explainer(X_ev[:n_perm], max_evals=2 * d + 1).values
                        method = (f"shap.PermutationExplainer on {n_perm} samples "
                                  f"(TreeExplainer failed: {type(tree_err).__name__})")
                if isinstance(sv, list):
                    sv = sv[0]
                sv = np.asarray(sv)

            if sv.ndim == 3:          # (n, d, outputs)
                sv = sv[..., 0]
            if sv.shape[-1] != d:
                raise ValueError(f"SHAP output has {sv.shape[-1]} features, expected {d}")

            mean_abs = np.abs(sv).mean(axis=0)
            total = float(np.sum(mean_abs))
            if not np.isfinite(total) or total <= 0:
                raise ValueError("degenerate SHAP attributions (zero or non-finite total)")

            attributions[name] = mean_abs / total
            provenance[name] = {
                "status": "computed",
                "method": method,
                "n_background": int(len(X_bg)),
                "n_explained": int(len(X_ev)),
            }
            logger.info("  %s -> SHAP computed via %s", name, method)

        except Exception as e:
            # No synthetic fallback: an unavailable model is reported as
            # unavailable and excluded from every agreement statistic.
            provenance[name] = {"status": "unavailable", "error": f"{type(e).__name__}: {e}"}
            logger.warning("  %s -> SHAP unavailable, EXCLUDED from agreement statistics: %s", name, e)

    model_names = list(attributions.keys())
    excluded = [m for m, p in provenance.items() if p["status"] != "computed"]
    n_models = len(model_names)

    spearman_matrix = np.eye(max(n_models, 1))
    kendall_matrix = np.eye(max(n_models, 1))
    pairwise_records: List[Dict[str, Any]] = []

    if n_models > 1:
        spearman_matrix = np.zeros((n_models, n_models))
        kendall_matrix = np.zeros((n_models, n_models))
        for i in range(n_models):
            for j in range(n_models):
                a1, a2 = attributions[model_names[i]], attributions[model_names[j]]
                sp_r, _ = sp_stats.spearmanr(a1, a2)
                kt_t, _ = sp_stats.kendalltau(a1, a2)
                spearman_matrix[i, j] = float(sp_r) if np.isfinite(sp_r) else np.nan
                kendall_matrix[i, j] = float(kt_t) if np.isfinite(kt_t) else np.nan
                if j > i:
                    # Top-10 overlap is easier to interpret than a global rank
                    # correlation over ~74 mostly-unimportant features.
                    top_i = set(np.argsort(a1)[::-1][:10])
                    top_j = set(np.argsort(a2)[::-1][:10])
                    pairwise_records.append({
                        "model_a": model_names[i],
                        "model_b": model_names[j],
                        "spearman": round(float(sp_r), 4),
                        "kendall_tau": round(float(kt_t), 4),
                        "top10_overlap": len(top_i & top_j),
                        "top10_jaccard": round(len(top_i & top_j) / len(top_i | top_j), 4),
                    })

    triu = np.triu_indices(n_models, k=1)
    mean_spearman = float(np.nanmean(spearman_matrix[triu])) if n_models > 1 else float("nan")
    mean_kendall = float(np.nanmean(kendall_matrix[triu])) if n_models > 1 else float("nan")
    similarity_index = (
        round(float((mean_spearman + mean_kendall) / 2.0), 4) if n_models > 1 else None
    )

    # Interpretation from stated thresholds, not asserted.
    # Bands follow the conventional reading of rank-correlation magnitude
    # (|rho| < 0.20 negligible, < 0.40 weak, < 0.60 moderate, else strong).
    def _band(v: Optional[float]) -> str:
        if v is None or not np.isfinite(v):
            return "not computable"
        av = abs(v)
        if av < 0.20:
            return "negligible agreement"
        if av < 0.40:
            return "limited agreement"
        if av < 0.60:
            return "moderate agreement"
        return "strong agreement"

    agreement_label = _band(similarity_index)
    interpretation = (
        f"Cross-backbone attribution agreement is {agreement_label} "
        f"(similarity index = {similarity_index}, mean Spearman = {round(mean_spearman, 4)}, "
        f"mean Kendall tau = {round(mean_kendall, 4)}). "
        "Rank-correlation bands used: |value| < 0.20 negligible, < 0.40 limited, "
        "< 0.60 moderate, otherwise strong. "
        "Different model families can fit similarly well while distributing credit "
        "across correlated predictors differently, so limited agreement does not by "
        "itself invalidate any single model's attributions -- but it does mean the "
        "attribution ranking should not be presented as a model-independent finding."
    )
    if n_models < 2:
        interpretation = (
            "Fewer than two backbones produced computable SHAP attributions, so no "
            "cross-model agreement statistic is reported."
        )

    csv_rows: List[Dict[str, Any]] = []
    if n_models > 0:
        avg_attr = np.mean([attributions[m] for m in model_names], axis=0)
        top20_indices = np.argsort(avg_attr)[::-1][:20]
        top20_features = [feature_cols[i] for i in top20_indices]
        for rank, idx in enumerate(top20_indices, 1):
            row = {"Rank": rank, "Feature": feature_cols[idx],
                   "Mean_Attribution": round(float(avg_attr[idx]), 6)}
            for m_name in model_names:
                row[f"Attribution_{m_name}"] = round(float(attributions[m_name][idx]), 6)
            csv_rows.append(row)
    else:
        top20_features = []

    shap_df = pd.DataFrame(csv_rows)
    if not shap_df.empty:
        save_report_csv(shap_df, "shap_consistency_report.csv")

    report = {
        "models_requested": list(models_dict.keys()),
        "models_with_computed_shap": model_names,
        "models_excluded": excluded,
        "attribution_provenance": provenance,
        "attribution_method": "mean |SHAP value| per feature, normalized to sum to 1",
        "n_background_samples": int(len(X_bg)),
        "n_explained_samples": int(len(X_ev)),
        "mean_spearman_correlation": round(mean_spearman, 4) if n_models > 1 else None,
        "mean_kendall_tau_correlation": round(mean_kendall, 4) if n_models > 1 else None,
        "shap_similarity_index": similarity_index,
        "agreement_label": agreement_label,
        "agreement_bands": {
            "negligible": "|value| < 0.20", "limited": "0.20 <= |value| < 0.40",
            "moderate": "0.40 <= |value| < 0.60", "strong": "|value| >= 0.60",
        },
        "pairwise": pairwise_records,
        "top_20_consensus_features": top20_features,
        "interpretation": interpretation,
    }

    excl_md = ""
    if excluded:
        excl_md = "\n### Excluded backbones\n" + "\n".join(
            f"- `{m}`: {provenance[m].get('error')}" for m in excluded
        ) + "\n"

    md = f"""# Cross-Backbone SHAP Attribution Consistency Report (§5.8)

## Provenance
- **Attribution method**: {report['attribution_method']}
- **Backbones with computed SHAP**: {', '.join(model_names) if model_names else 'none'}
- **Explained samples**: {report['n_explained_samples']} (background: {report['n_background_samples']})
- Per-model explainer: {', '.join(f"`{m}` = {provenance[m].get('method')}" for m in model_names) if model_names else 'n/a'}
{excl_md}
## Cross-model agreement
- **Mean Spearman ρ**: `{report['mean_spearman_correlation']}`
- **Mean Kendall τ**: `{report['mean_kendall_tau_correlation']}`
- **Similarity index** (mean of the two): `{similarity_index}` — **{agreement_label}**

### Pairwise agreement
"""
    if pairwise_records:
        try:
            md += pd.DataFrame(pairwise_records).to_markdown(index=False) + "\n\n"
        except Exception:
            md += pd.DataFrame(pairwise_records).to_string(index=False) + "\n\n"
    else:
        md += "- Not enough backbones with computed attributions for pairwise comparison.\n\n"

    md += "## Interpretation\n" + interpretation + "\n\n## Top 15 consensus features\n"
    if not shap_df.empty:
        try:
            md += shap_df.head(15).to_markdown(index=False) + "\n\n"
        except Exception:
            md += shap_df.head(15).to_string(index=False) + "\n\n"
    else:
        md += "- No attributions available.\n\n"

    save_report_markdown(md, "shap_consistency_report.md")
    save_report(report, "shap_consistency_report.json")

    if n_models > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(spearman_matrix, xticklabels=model_names, yticklabels=model_names,
                    annot=True, cmap="YlGnBu", vmin=-1, vmax=1, ax=ax)
        ax.set_title("Cross-Backbone SHAP Rank Correlation (Spearman ρ)")
        _save(fig, "shap_cross_backbone_comparison")

    if not shap_df.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap_df.head(15).plot(x="Feature", y=[f"Attribution_{m}" for m in model_names],
                              kind="barh", ax=ax2)
        ax2.set_title("Top 15 SHAP Attribution Comparison Across Backbones")
        ax2.invert_yaxis()
        _save(fig2, "shap_bar_plot")

    logger.info("SHAP consistency exported (models=%d, excluded=%d, similarity=%s, label=%s)",
                n_models, len(excluded), similarity_index, agreement_label)
    return report


def plot_backbone_comparison(benchmark_df: pd.DataFrame) -> Path:
    """Plot backbone model performance comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if "Model" in benchmark_df.columns and "RMSE" in benchmark_df.columns:
        ax.bar(benchmark_df["Model"], benchmark_df["RMSE"], color="#3498db")
        ax.set_ylabel("RMSE (t/ha)")
        ax.set_title("Multi-Backbone RMSE Comparison")
    return _save(fig, "backbone_comparison")


def plot_objective_o5_complete_suite(
    aci_result: CalibrationResult,
    test_df: pd.DataFrame,
) -> List[Path]:
    """Generate complete figure suite for Objective O5 (§3 O5 & §5.8)."""
    paths = []
    widths = aci_result.q_hi - aci_result.q_lo
    severity = test_df["CDHW_Severity_Score"].values if "CDHW_Severity_Score" in test_df.columns else np.zeros(len(widths))

    # 1. Scatter & Regression plot with 95% CI
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.regplot(x=severity, y=widths, ax=ax1, color="#e74c3c", scatter_kws={"alpha": 0.3}, line_kws={"color": "darkred"})
    ax1.set_xlabel("CDHW Severity Score (§4.1)")
    ax1.set_ylabel("ACI Prediction Interval Width (MPIW)")
    ax1.set_title("Objective O5: Interval Width Regression & 95% CI")
    paths.append(_save(fig1, "objective_o5_regression_95ci"))

    # 2. Density Plot by Year_Type
    if "Year_Type" in test_df.columns:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for yt in ["Normal", "Moderate", "Extreme"]:
            mask = test_df["Year_Type"].values == yt
            if mask.sum() > 0:
                sns.kdeplot(widths[mask], label=f"{yt} Years", ax=ax2, fill=True, alpha=0.3)
        ax2.set_xlabel("Prediction Interval Width (MPIW)")
        ax2.set_ylabel("Density")
        ax2.set_title("Objective O5: Interval Width Density by CDHW Year Type")
        ax2.legend()
        paths.append(_save(fig2, "objective_o5_density"))

        # 3. Boxplots grouped by CDHW severity / Year_Type
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        df_box = pd.DataFrame({"Year_Type": test_df["Year_Type"].values, "MPIW": widths})
        sns.boxplot(x="Year_Type", y="MPIW", data=df_box, hue="Year_Type", palette="Reds", legend=False, ax=ax3)
        ax3.set_title("Objective O5: Uncertainty Boxplots Grouped by Severity")
        paths.append(_save(fig3, "objective_o5_boxplots_by_severity"))

    # 4. Phenology Stage Attribution & Silking Severity Plot (§3 O5 & §4.1)
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 6))
    
    if "Phenological_Window" in test_df.columns:
        df_pheno = pd.DataFrame({"Phenology_Stage": test_df["Phenological_Window"].values, "MPIW": widths})
        sns.boxplot(x="Phenology_Stage", y="MPIW", data=df_pheno, hue="Phenology_Stage", palette="YlOrRd", legend=False, ax=ax4a)
        ax4a.set_xlabel("GDD Phenological Stage (§4.1)")
        ax4a.set_ylabel("ACI Prediction Interval Width (MPIW)")
        ax4a.set_title("Panel A: Interval Width by GDD Phenological Stage")

    silking_col = "CDHW_Silking_Severity" if "CDHW_Silking_Severity" in test_df.columns else "CDHW_Severity_silking"
    if silking_col in test_df.columns:
        silking_sev = test_df[silking_col].values
        sns.regplot(x=silking_sev, y=widths, ax=ax4b, color="#9b59b6", scatter_kws={"alpha": 0.3}, line_kws={"color": "purple"})
        r_silk, p_silk = sp_stats.pearsonr(widths, silking_sev)
        ax4b.set_xlabel("Silking Stage CDHW Severity (§4.1)")
        ax4b.set_ylabel("ACI Prediction Interval Width (MPIW)")
        ax4b.set_title(f"Panel B: Width vs Silking Severity (Pearson r = {r_silk:.4f}, p = {p_silk:.4e})")
    else:
        sns.regplot(x=severity, y=widths, ax=ax4b, color="#9b59b6", scatter_kws={"alpha": 0.3})
        ax4b.set_title("Panel B: Width vs Severity")

    plt.tight_layout()
    paths.append(_save(fig4, "objective_o5_phenology_attribution"))

    return paths


def plot_aci_online_adaptation_trajectory(
    aci_result: CalibrationResult,
    test_df: pd.DataFrame,
) -> Path:
    """Plot online ACI width & sliding coverage trajectory across test years (§4.3 & §5.8)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    widths = aci_result.q_hi - aci_result.q_lo
    covered = ((aci_result.y_true >= aci_result.q_lo) & (aci_result.y_true <= aci_result.q_hi)).astype(float)

    if "Year" in test_df.columns:
        df_plot = pd.DataFrame({"Year": test_df["Year"].values, "Width": widths, "Covered": covered})
        yearly = df_plot.groupby("Year").agg({"Width": ["mean", "std"], "Covered": "mean"})

        years = yearly.index.values
        w_mean = yearly[("Width", "mean")].values
        w_std = yearly[("Width", "std")].values
        cov_mean = yearly[("Covered", "mean")].values

        # Panel 1: Interval Width Trajectory
        ax1.plot(years, w_mean, "o-", color="#e74c3c", linewidth=2.5, label="Mean Interval Width (MPIW)")
        ax1.fill_between(years, w_mean - w_std, w_mean + w_std, color="#e74c3c", alpha=0.2, label="±1 Std Dev")
        ax1.set_ylabel("Prediction Interval Width (MPIW)")
        ax1.set_title("Panel A: ACI Dynamic Interval Expansion Over Test Years (2019–2023)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper left")

        # Panel 2: Online Coverage Trajectory
        ax2.plot(years, cov_mean * 100, "s-", color="#2ecc71", linewidth=2.5, label="Observed Coverage (PICP %)")
        ax2.axhline(90.0, color="darkred", linestyle="--", linewidth=1.5, label="Target Coverage (90%)")
        ax2.set_xlabel("Test Year")
        ax2.set_ylabel("Coverage Probability (%)")
        ax2.set_title("Panel B: Online Target Coverage Recovery Trajectory")
        ax2.set_ylim(60, 100)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="lower right")

def plot_residual_diagnostics_suite(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_df: pd.DataFrame,
) -> Path:
    """Generate complete 6-panel residual diagnostic suite (§6)."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Panel 1: Residual vs Prediction
    axes[0, 0].scatter(y_pred, residuals, alpha=0.3, color="#2980b9")
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0, 0].set_xlabel("Predicted Yield (t/ha)")
    axes[0, 0].set_ylabel("Residual (True - Pred)")
    axes[0, 0].set_title("Panel A: Residuals vs Prediction")
    axes[0, 0].grid(True, linestyle="--", alpha=0.4)

    # Panel 2: Residual vs Year
    if "Year" in test_df.columns:
        sns.boxplot(x=test_df["Year"], y=residuals, ax=axes[0, 1], color="#3498db")
        axes[0, 1].axhline(0, color="red", linestyle="--", linewidth=1.5)
        axes[0, 1].set_xlabel("Test Year")
        axes[0, 1].set_ylabel("Residual")
        axes[0, 1].set_title("Panel B: Residual Distribution vs Year")
        axes[0, 1].grid(True, linestyle="--", alpha=0.4)

    # Panel 3: Residual vs State
    if "State" in test_df.columns:
        sns.boxplot(x=test_df["State"], y=residuals, ax=axes[0, 2], color="#e74c3c")
        axes[0, 2].axhline(0, color="red", linestyle="--", linewidth=1.5)
        axes[0, 2].set_xlabel("State")
        axes[0, 2].set_ylabel("Residual")
        axes[0, 2].tick_params(axis="x", rotation=45)
        axes[0, 2].set_title("Panel C: Residual Distribution vs State")
        axes[0, 2].grid(True, linestyle="--", alpha=0.4)

    # Panel 4: Residual vs CDHW Severity
    sev_col = "CDHW_Severity_Score" if "CDHW_Severity_Score" in test_df.columns else "CDHW_Flag"
    if sev_col in test_df.columns:
        axes[1, 0].scatter(test_df[sev_col], residuals, alpha=0.3, color="#9b59b6")
        axes[1, 0].axhline(0, color="red", linestyle="--", linewidth=1.5)
        axes[1, 0].set_xlabel(sev_col)
        axes[1, 0].set_ylabel("Residual")
        axes[1, 0].set_title("Panel D: Residuals vs CDHW Severity")
        axes[1, 0].grid(True, linestyle="--", alpha=0.4)

    # Panel 5: Residual QQ Plot
    sp_stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Panel E: Residual Normal Q-Q Plot")
    axes[1, 1].grid(True, linestyle="--", alpha=0.4)

    # Panel 6: Residual Histogram
    sns.histplot(residuals, kde=True, ax=axes[1, 2], color="#2ecc71", bins=30)
    axes[1, 2].set_xlabel("Residual (t/ha)")
    axes[1, 2].set_title("Panel F: Residual Histogram & KDE Density")
    axes[1, 2].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    return _save(fig, "residual_diagnostics_suite")




def plot_feature_selection_stability_visualizations() -> List[Path]:
    """Generate 4 feature selection stability plots (§5.2)."""
    paths = []
    stab_csv = cfg.REPORT_DIR / "feature_selection_stability.csv"
    if not stab_csv.exists():
        return paths

    df = pd.read_csv(stab_csv)
    top15 = df.head(15)

    # 1. Selection Frequency Plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.barh(top15["Feature"], top15["Selection_Frequency"], color="#2ecc71")
    ax1.set_xlabel("Selection Frequency Across Folds & Bootstraps")
    ax1.set_title("Feature Selection Frequency (§5.2)")
    ax1.invert_yaxis()
    paths.append(_save(fig1, "selection_frequency_plot"))

    # 2. Rank Consistency Plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.errorbar(top15["Average_Rank"], top15["Feature"], xerr=top15["Rank_Std"], fmt="o", color="#3498db", ecolor="#e74c3c", capsize=4)
    ax2.set_xlabel("Average Rank (± Std Dev)")
    ax2.set_title("Feature Rank Consistency Across Folds (§5.2)")
    ax2.invert_yaxis()
    paths.append(_save(fig2, "rank_consistency_plot"))

    return paths


# ─────────────────────────────────────────────────────────────
# Split comparison: Temporal (out-of-time) vs Random row-level split
# ─────────────────────────────────────────────────────────────

def plot_split_comparison(split_comparison_df: pd.DataFrame) -> Path:
    """Grouped bar chart contrasting R2 (and RMSE) between the temporal
    (out-of-time forecast) split and the random row-level (interpolation
    benchmark) split, for each model. Makes the "these answer different
    questions" point from PHASE 3B's docstring visually obvious: random
    split R2 should be visibly higher, since it's an easier interpolation
    task, not a fairer benchmark.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = split_comparison_df["Model"].unique().tolist()
    split_types = split_comparison_df["Split_Type"].unique().tolist()
    x = np.arange(len(models))
    width = 0.8 / max(1, len(split_types))
    colors = sns.color_palette("Set2", len(split_types))

    for ax, metric, ylabel in [(axes[0], "R2", "R\u00b2 (higher is better)"),
                                (axes[1], "RMSE", "RMSE, t/ha (lower is better)")]:
        for i, split_type in enumerate(split_types):
            sub = split_comparison_df[split_comparison_df["Split_Type"] == split_type]
            vals = [sub[sub["Model"] == m][metric].values[0] if m in sub["Model"].values and
                    pd.notna(sub[sub["Model"] == m][metric].values[0]) else np.nan
                    for m in models]
            ax.bar(x + i * width, vals, width, label=split_type, color=colors[i])
        ax.set_xticks(x + width * (len(split_types) - 1) / 2)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.margins(x=0.15)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric}: Temporal vs. Random Split")

    axes[0].legend(loc="upper right", fontsize=8, ncol=1)
    fig.suptitle(
        "Temporal (out-of-time forecast) vs. Random row-level (interpolation) split\n"
        "Random-split R\u00b2 is expected to be higher -- it answers a different, easier question",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return _save(fig, "temporal_vs_random_split_comparison")


# ─────────────────────────────────────────────────────────────
# Yield trends over time by state
# ─────────────────────────────────────────────────────────────

def plot_yield_trends_by_state(
    df: pd.DataFrame,
    target_col: str = None,
    train_years: tuple = None,
    val_years: tuple = None,
    test_years: tuple = None,
) -> Path:
    """Mean yield per state per year, with the train/val/test temporal
    split boundaries shaded, so the reader can see both the underlying
    upward yield trend (motivating DETREND_TARGET) and where the
    forecast horizon (test period) sits relative to it."""
    target_col = target_col or cfg.PRIMARY_TARGET
    train_years = train_years or cfg.TRAIN_YEARS
    val_years = val_years or cfg.VAL_YEARS
    test_years = test_years or cfg.TEST_YEARS

    trend = df.groupby(["Year", "State"])[target_col].mean().reset_index()

    fig, ax = plt.subplots(figsize=(13, 7))
    for state, sub in trend.groupby("State"):
        ax.plot(sub["Year"], sub[target_col], marker="o", markersize=3, linewidth=1.5, label=state)

    ax.axvspan(val_years[0], val_years[1], color="orange", alpha=0.12, label="Validation (2016-2018)")
    ax.axvspan(test_years[0], test_years[1], color="red", alpha=0.10, label="Test (2019-2023)")

    ax.set_xlabel("Year")
    ax.set_ylabel(f"Mean {target_col}")
    ax.set_title(f"State-level mean {target_col} over time, with temporal split boundaries")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    return _save(fig, "yield_trends_by_state")


