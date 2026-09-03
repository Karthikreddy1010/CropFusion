"""
geo_maps.py - Geographic map visualizations for the 6-state study region.

Adds county- and state-level choropleth maps using real US Census county
boundaries (fetched once from a public GeoJSON mirror and cached locally
under outputs/geo_cache/), so no shapefiles need to ship with the repo.

All functions degrade gracefully: if the boundary GeoJSON can't be
fetched (no network, or the mirror is unreachable) they log a warning
and return None instead of crashing the pipeline. Every other report/
figure in main() is produced whether or not these succeed.

Maps included:
    plot_study_region_map()      - county boundaries colored by state
                                    (locator map for the 6-state region)
    plot_yield_choropleth()      - mean target value by county
    plot_loso_r2_choropleth()    - LOSO-CV R2 by held-out state
    plot_residual_choropleth()   - mean backbone test-set residual by
                                    county (directly visualizes the
                                    "spatial heterogeneity dominates the
                                    error" diagnostic from negative_r2
                                    diagnostics)
"""
from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg

logger = logging.getLogger("paper3")

# US Census state FIPS codes for the 6 LOSO_STATES (order matches
# cfg.LOSO_STATES so state name <-> FIPS is always unambiguous even if
# LOSO_STATES is edited later).
_STATE_NAME_TO_FIPS: Dict[str, str] = {
    "Illinois": "17", "Indiana": "18", "Iowa": "19",
    "Minnesota": "27", "Missouri": "29", "Ohio": "39",
    "Nebraska": "31",  # kept for completeness even though DROP_STATES excludes it
}

_COUNTY_GEOJSON_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
_STATE_GEOJSON_URL = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"

_GEO_CACHE_DIR = cfg.OUTPUT_DIR / "geo_cache"


def _fetch_geojson(url: str, cache_name: str) -> Optional[Dict[str, Any]]:
    """Download a GeoJSON file once and cache it under outputs/geo_cache/.
    Returns the parsed dict, or None (with a logged warning) on any
    failure -- callers must handle None gracefully."""
    _GEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _GEO_CACHE_DIR / cache_name

    if cache_path.exists():
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Cached geojson %s unreadable (%s), re-fetching.", cache_path, e)

    try:
        logger.info("Fetching boundary data: %s", url)
        urllib.request.urlretrieve(url, cache_path)
        with open(cache_path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(
            "Could not fetch %s (%s). Geographic maps will be skipped -- "
            "this requires outbound network access to raw.githubusercontent.com. "
            "Every other report/figure is unaffected.", url, e
        )
        return None


def _load_county_geometries():
    """Returns a GeoDataFrame of county polygons for cfg.LOSO_STATES only,
    or None if boundary data couldn't be fetched or geopandas isn't
    installed."""
    try:
        import geopandas as gpd
    except ImportError:
        logger.warning("geopandas not installed -- run `pip install geopandas --break-system-packages` "
                        "to enable geographic maps. Skipping.")
        return None

    geo = _fetch_geojson(_COUNTY_GEOJSON_URL, "counties_fips.json")
    if geo is None:
        return None

    gdf = gpd.GeoDataFrame.from_features(geo["features"])
    gdf["GEOID"] = [f["id"] for f in geo["features"]]
    gdf["STATEFP"] = gdf["GEOID"].str[:2]

    target_fips = {_STATE_NAME_TO_FIPS[s] for s in cfg.LOSO_STATES if s in _STATE_NAME_TO_FIPS}
    return gdf[gdf["STATEFP"].isin(target_fips)].copy()


def _load_state_geometries():
    """Returns a GeoDataFrame of state polygons for cfg.LOSO_STATES only,
    or None on failure."""
    try:
        import geopandas as gpd
    except ImportError:
        return None

    geo = _fetch_geojson(_STATE_GEOJSON_URL, "us_states.json")
    if geo is None:
        return None

    gdf = gpd.GeoDataFrame.from_features(geo["features"])
    name_col = "name" if "name" in gdf.columns else "NAME"
    return gdf[gdf[name_col].isin(cfg.LOSO_STATES)].copy(), name_col


def plot_study_region_map(df: pd.DataFrame) -> Optional[Path]:
    """County-boundary locator map of the 6-state study region, colored
    by state. Shows which counties are actually in the dataset (not
    just the state outline) since county coverage can be uneven."""
    county_gdf = _load_county_geometries()
    if county_gdf is None:
        return None

    included_geoids = set(df["GEOID"].astype(str).str.zfill(5).unique())
    county_gdf["in_dataset"] = county_gdf["GEOID"].isin(included_geoids)
    state_fp_to_name = {v: k for k, v in _STATE_NAME_TO_FIPS.items()}
    county_gdf["State"] = county_gdf["STATEFP"].map(state_fp_to_name)

    fig, ax = plt.subplots(figsize=(12, 9))
    county_gdf.plot(
        ax=ax, column="State", categorical=True, legend=True,
        edgecolor="white", linewidth=0.3, cmap="Set2",
        legend_kwds={"loc": "lower left", "title": "State", "fontsize": 9},
    )
    n_missing = (~county_gdf["in_dataset"]).sum()
    if n_missing > 0:
        county_gdf[~county_gdf["in_dataset"]].boundary.plot(ax=ax, edgecolor="red", linewidth=1.2)
    ax.set_title(
        f"Study region: {county_gdf['in_dataset'].sum()} counties across "
        f"{len(cfg.LOSO_STATES)} states"
        + (f" ({n_missing} county boundary shown in red has no matching data row)" if n_missing else "")
    )
    ax.axis("off")

    path = cfg.PLOT_DIR / f"study_region_map.{cfg.PLOT_FORMAT}"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Plot saved -> %s", path)
    return path


def plot_yield_choropleth(
    df: pd.DataFrame,
    target_col: str = None,
    title: Optional[str] = None,
) -> Optional[Path]:
    """County-level choropleth of the mean target value across all years
    in the dataset (e.g. mean corn yield 1985-2023)."""
    target_col = target_col or cfg.PRIMARY_TARGET
    county_gdf = _load_county_geometries()
    if county_gdf is None:
        return None

    df_local = df.copy()
    df_local["GEOID_str"] = df_local["GEOID"].astype(str).str.zfill(5)
    mean_by_county = df_local.groupby("GEOID_str")[target_col].mean().reset_index()

    merged = county_gdf.merge(mean_by_county, left_on="GEOID", right_on="GEOID_str", how="inner")
    if merged.empty:
        logger.warning("plot_yield_choropleth: no counties matched between boundary data and dataset GEOIDs.")
        return None

    fig, ax = plt.subplots(figsize=(12, 9))
    merged.plot(
        ax=ax, column=target_col, cmap="YlGn", legend=True,
        edgecolor="white", linewidth=0.3,
        legend_kwds={"label": f"Mean {target_col}", "shrink": 0.7},
    )
    ax.set_title(title or f"Mean county-level {target_col} across the study region ({merged.shape[0]} counties)")
    ax.axis("off")

    safe_name = target_col.lower().replace(" ", "_")
    path = cfg.PLOT_DIR / f"yield_choropleth_{safe_name}.{cfg.PLOT_FORMAT}"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Plot saved -> %s", path)
    return path


def plot_state_metric_choropleth(
    values_by_state: Dict[str, float],
    metric_label: str,
    title: str,
    output_name: str,
    cmap: str = "RdYlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    value_fmt: str = "{:.2f}",
) -> Optional[Path]:
    """Generic state-level choropleth with the state name + metric value
    labeled directly on each polygon (same style as the LOSO R2 map).
    Reusable for any per-state metric -- ensemble weight, yield
    volatility, CDHW severity, PICP, etc.

    values_by_state: {state name (matching cfg.LOSO_STATES) -> value}
    vmin/vmax: color scale bounds; None lets matplotlib auto-scale to
        the data (use explicit bounds like 0/1 for metrics with a
        natural fixed range, e.g. R2 or PICP, so maps are comparable
        across runs).
    """
    result = _load_state_geometries()
    if result is None:
        return None
    state_gdf, name_col = result

    state_gdf["_metric"] = state_gdf[name_col].map(values_by_state)
    if state_gdf["_metric"].isna().all():
        logger.warning("plot_state_metric_choropleth(%s): no values matched to state polygons.", output_name)
        return None

    fig, ax = plt.subplots(figsize=(12, 9))
    state_gdf.plot(
        ax=ax, column="_metric", cmap=cmap, legend=True,
        edgecolor="black", linewidth=0.8, vmin=vmin, vmax=vmax,
        missing_kwds={"color": "lightgrey", "label": "No data"},
        legend_kwds={"label": metric_label, "shrink": 0.7},
    )
    for _, row in state_gdf.iterrows():
        if pd.notna(row["_metric"]):
            centroid = row.geometry.centroid
            ax.annotate(
                f"{row[name_col]}\n{value_fmt.format(row['_metric'])}",
                xy=(centroid.x, centroid.y), ha="center", fontsize=9,
                fontweight="bold", color="black",
            )
    ax.set_title(title)
    ax.axis("off")

    path = cfg.PLOT_DIR / f"{output_name}.{cfg.PLOT_FORMAT}"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Plot saved -> %s", path)
    return path


def plot_loso_r2_choropleth(loso_metrics: List[Dict[str, Any]]) -> Optional[Path]:
    """State-level choropleth colored by each state's LOSO-CV R2 -- the
    single clearest visual for the paper's spatial-generalization
    result (e.g. the Missouri recovery story)."""
    r2_by_state = {m["state"]: m.get("r_squared") for m in loso_metrics if "r_squared" in m}
    return plot_state_metric_choropleth(
        r2_by_state,
        metric_label="LOSO-CV R\u00b2 (held-out state)",
        title="Leave-One-State-Out CV: R\u00b2 by held-out state",
        output_name="loso_r2_choropleth",
        cmap="RdYlGn", vmin=0, vmax=1, value_fmt="R\u00b2={:.2f}",
    )


def plot_loso_picp_choropleth(loso_metrics: List[Dict[str, Any]]) -> Optional[Path]:
    """State-level choropleth of LOSO-CV interval coverage (PICP) by
    held-out state -- calibration quality, not just point-prediction
    accuracy, can vary by state independently of R2. Nominal target is
    0.90; the diverging colormap centered there makes over/under
    coverage visually distinct."""
    picp_by_state = {m["state"]: m.get("picp") for m in loso_metrics if "picp" in m}
    return plot_state_metric_choropleth(
        picp_by_state,
        metric_label="LOSO-CV PICP (target = 0.90)",
        title="Leave-One-State-Out CV: interval coverage (PICP) by held-out state",
        output_name="loso_picp_choropleth",
        cmap="RdYlGn", vmin=0.5, vmax=1.0, value_fmt="PICP={:.2f}",
    )


def plot_ensemble_weight_choropleth(loso_metrics: List[Dict[str, Any]]) -> Optional[Path]:
    """State-level choropleth of the NeuralCQR ensemble weight chosen
    per LOSO fold (0 = pure LightGBM fallback, 1 = pure NeuralCQR).
    This directly visualizes *why* the ensemble fix works: states where
    the neural net fails to generalize (weight -> 0, e.g. Missouri)
    fall back to the tree model; states it generalizes to fine keep a
    higher neural weight. Ties the Missouri-recovery narrative to a
    mechanism, not just an outcome."""
    w_by_state = {
        m["state"]: m.get("neuralcqr_ensemble_weight")
        for m in loso_metrics if "neuralcqr_ensemble_weight" in m
    }
    return plot_state_metric_choropleth(
        w_by_state,
        metric_label="NeuralCQR weight in ensemble (0=pure LightGBM, 1=pure NeuralCQR)",
        title="Which held-out states does the neural backbone generalize to?\n(ensemble weight chosen per LOSO fold, tuned on val only)",
        output_name="ensemble_weight_choropleth",
        cmap="RdYlGn", vmin=0, vmax=1, value_fmt="w={:.2f}",
    )


def plot_yield_volatility_choropleth(df: pd.DataFrame, target_col: str = None) -> Optional[Path]:
    """State-level choropleth of yield volatility (coefficient of
    variation across 1985-2023) -- a direct, physical explanation for
    why some states (e.g. Missouri) are harder LOSO folds: it isn't
    just 'different soil', it's a genuinely more volatile yield series
    to predict in the first place. Complements the yield_trends_by_state
    line chart with a single labeled summary number per state."""
    target_col = target_col or cfg.PRIMARY_TARGET
    stats = df.groupby("State")[target_col].agg(["mean", "std"])
    cv_by_state = (stats["std"] / stats["mean"]).to_dict()
    return plot_state_metric_choropleth(
        cv_by_state,
        metric_label="Yield coefficient of variation (std/mean, 1985-2023)",
        title=f"State-level {target_col} volatility (higher = more erratic, harder to forecast)",
        output_name="yield_volatility_choropleth",
        cmap="RdYlGn_r", value_fmt="CV={:.3f}",  # reversed: high volatility = red = bad
    )


def plot_residual_choropleth(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Mean backbone test-set residual by county",
) -> Optional[Path]:
    """County-level choropleth of mean (predicted - actual) residual on
    the temporal-split test set. Directly visualizes the spatial
    heterogeneity flagged by diagnose_r2_source() (negative_r2_diagnostic
    report) -- counties/regions the backbone systematically over- or
    under-predicts should show up as spatially clustered red/blue here,
    not scattered noise, if spatial heterogeneity is really the dominant
    error source."""
    county_gdf = _load_county_geometries()
    if county_gdf is None:
        return None

    if len(test_df) != len(y_true):
        logger.warning(
            "plot_residual_choropleth: test_df (%d rows) and y_true (%d) length mismatch, skipping.",
            len(test_df), len(y_true),
        )
        return None

    resid_df = pd.DataFrame({
        "GEOID_str": test_df["GEOID"].astype(str).str.zfill(5).values,
        "residual": np.asarray(y_pred) - np.asarray(y_true),
    })
    mean_resid = resid_df.groupby("GEOID_str")["residual"].mean().reset_index()

    merged = county_gdf.merge(mean_resid, left_on="GEOID", right_on="GEOID_str", how="inner")
    if merged.empty:
        logger.warning("plot_residual_choropleth: no counties matched.")
        return None

    vmax = float(np.nanmax(np.abs(merged["residual"])))
    fig, ax = plt.subplots(figsize=(12, 9))
    merged.plot(
        ax=ax, column="residual", cmap="RdBu_r", legend=True,
        edgecolor="white", linewidth=0.3, vmin=-vmax, vmax=vmax,
        legend_kwds={"label": "Mean residual (predicted - actual, t/ha)", "shrink": 0.7},
    )
    ax.set_title(f"{title}\n(red = over-predicts, blue = under-predicts; n={merged.shape[0]} counties)")
    ax.axis("off")

    path = cfg.PLOT_DIR / f"residual_choropleth.{cfg.PLOT_FORMAT}"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Plot saved -> %s", path)
    return path


def run_all_geo_maps(
    df: pd.DataFrame,
    loso_metrics: Optional[List[Dict[str, Any]]] = None,
    test_df: Optional[pd.DataFrame] = None,
    y_test: Optional[np.ndarray] = None,
    preds_test: Optional[np.ndarray] = None,
) -> Dict[str, Optional[Path]]:
    """Convenience wrapper: generate every geo map that has the inputs
    available for it. Never raises -- each map is independently
    try/excepted so one failure doesn't block the others."""
    results: Dict[str, Optional[Path]] = {}

    for name, fn in [
        ("study_region_map", lambda: plot_study_region_map(df)),
        ("yield_choropleth", lambda: plot_yield_choropleth(df)),
    ]:
        try:
            results[name] = fn()
        except Exception as e:
            logger.warning("Geo map '%s' failed (%s), skipping.", name, e)
            results[name] = None

    if loso_metrics:
        try:
            results["loso_r2_choropleth"] = plot_loso_r2_choropleth(loso_metrics)
        except Exception as e:
            logger.warning("Geo map 'loso_r2_choropleth' failed (%s), skipping.", e)
            results["loso_r2_choropleth"] = None
        try:
            results["loso_picp_choropleth"] = plot_loso_picp_choropleth(loso_metrics)
        except Exception as e:
            logger.warning("Geo map 'loso_picp_choropleth' failed (%s), skipping.", e)
            results["loso_picp_choropleth"] = None
        try:
            results["ensemble_weight_choropleth"] = plot_ensemble_weight_choropleth(loso_metrics)
        except Exception as e:
            logger.warning("Geo map 'ensemble_weight_choropleth' failed (%s), skipping.", e)
            results["ensemble_weight_choropleth"] = None

    try:
        results["yield_volatility_choropleth"] = plot_yield_volatility_choropleth(df)
    except Exception as e:
        logger.warning("Geo map 'yield_volatility_choropleth' failed (%s), skipping.", e)
        results["yield_volatility_choropleth"] = None

    if test_df is not None and y_test is not None and preds_test is not None:
        try:
            results["residual_choropleth"] = plot_residual_choropleth(test_df, y_test, preds_test)
        except Exception as e:
            logger.warning("Geo map 'residual_choropleth' failed (%s), skipping.", e)
            results["residual_choropleth"] = None

    n_ok = sum(1 for v in results.values() if v is not None)
    logger.info("Geographic maps: %d/%d generated successfully.", n_ok, len(results))
    return results
