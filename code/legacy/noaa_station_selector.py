"""
noaa_station_selector.py - Station inventory compiler & county spatial mapping.

Tasks:
1. Loads all 583 corn counties across 6 states from Paper3_MegaDataset_SPEI_FINAL.csv
2. Queries NOAA CDO API for candidate GHCND stations across the 6 states
3. Calculates Haversine distances between all stations and county centroids
4. Evaluates station record lengths (1985-2023) and completeness
5. Selects primary and secondary stations per county
6. Generates outputs/noaa_weather/noaa_station_inventory.csv
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from noaa_cdo_client import NoaaCdoClient

logger = logging.getLogger("noaa_weather")


def haversine_matrix_km(lats1: np.ndarray, lons1: np.ndarray, lats2: np.ndarray, lons2: np.ndarray) -> np.ndarray:
    """Compute Haversine distance matrix (N, M) between points (N) and points (M) in km."""
    R = 6371.0
    phi1 = np.radians(lats1)[:, np.newaxis]
    phi2 = np.radians(lats2)[np.newaxis, :]
    delta_phi = np.radians(lats2[np.newaxis, :] - lats1[:, np.newaxis])
    delta_lam = np.radians(lons2[np.newaxis, :] - lons1[:, np.newaxis])

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lam / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def load_county_centroids(dataset_path: Path | str, target_states: List[str] | None = None) -> pd.DataFrame:
    """Extract authoritative county centroids for the locked corn counties."""
    df = pd.read_csv(dataset_path, low_memory=False)
    if target_states is not None:
        df = df[df["State"].isin(target_states)].copy()
    counties = (
        df[["GEOID", "State", "County_Name", "Lat", "Lon"]]
        .drop_duplicates(subset=["GEOID"])
        .sort_values(["State", "County_Name"])
        .reset_index(drop=True)
    )
    counties["GEOID"] = counties["GEOID"].astype(int).apply(lambda x: f"{x:05d}")
    logger.info("Loaded %d county centroids across %d states", len(counties), counties["State"].nunique())
    return counties


def discover_state_stations(
    client: NoaaCdoClient,
    states: Dict[str, str],
    dataset_id: str = "GHCND",
    datacategory_id: str = "TEMP",
    datatype_id: str = "TMAX",
) -> pd.DataFrame:
    """Query NOAA CDO API for candidate GHCND stations across target states."""
    all_stations = []
    for state_name, fips in states.items():
        loc_id = f"FIPS:{fips}"
        logger.info("Discovering GHCND stations for %s (FIPS %s)...", state_name, fips)

        st_records = client.paginate(
            endpoint="stations",
            params={
                "datasetid": dataset_id,
                "locationid": loc_id,
                "datacategoryid": datacategory_id,
                "datatypeid": datatype_id,
            },
        )

        for s in st_records:
            all_stations.append(
                {
                    "station_id": s.get("id"),
                    "name": s.get("name"),
                    "state_name": state_name,
                    "state_fips": fips,
                    "latitude": s.get("latitude"),
                    "longitude": s.get("longitude"),
                    "elevation": s.get("elevation"),
                    "elevationUnit": s.get("elevationUnit"),
                    "first_date": s.get("mindate"),
                    "last_date": s.get("maxdate"),
                    "datacoverage": s.get("datacoverage"),
                }
            )

    st_df = pd.DataFrame(all_stations).drop_duplicates(subset=["station_id"]).reset_index(drop=True)
    logger.info("Discovered %d unique candidate GHCND stations across all target states", len(st_df))
    return st_df


def build_station_inventory(
    counties_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    max_radius_km: float = 50.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Map stations to counties, score historical coverage, and select optimal networks.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (station_inventory_df, county_station_mapping_df)
    """
    logger.info("Building county-station spatial mapping (max radius: %.1f km)...", max_radius_km)

    # Filter stations with valid coordinates and active in the study period (>= 1985)
    valid_st = stations_df[
        stations_df["latitude"].notna()
        & stations_df["longitude"].notna()
        & (stations_df["last_date"] >= "1985-01-01")
    ].copy().reset_index(drop=True)

    # Classify long-record status
    valid_st["is_long_record"] = (
        (valid_st["first_date"] <= "1985-01-01")
        & (valid_st["last_date"] >= "2023-01-01")
        & (valid_st["datacoverage"] >= 0.80)
    )

    c_lats = counties_df["Lat"].to_numpy(dtype=float)
    c_lons = counties_df["Lon"].to_numpy(dtype=float)
    s_lats = valid_st["latitude"].to_numpy(dtype=float)
    s_lons = valid_st["longitude"].to_numpy(dtype=float)
    s_is_long = valid_st["is_long_record"].to_numpy(dtype=bool)

    # Vectorized (N_counties, M_stations) distance matrix
    dist_matrix = haversine_matrix_km(c_lats, c_lons, s_lats, s_lons)

    mapping_records = []
    st_dict = valid_st.to_dict(orient="records")
    co_dict = counties_df.to_dict(orient="records")

    for i, c_row in enumerate(co_dict):
        geoid = c_row["GEOID"]
        c_state = c_row["State"]
        c_name = c_row["County_Name"]
        dists = dist_matrix[i]

        within_idx = np.where(dists <= max_radius_km)[0]
        if len(within_idx) == 0:
            nearest_idx = int(np.argmin(dists))
            within_idx = np.array([nearest_idx])
            logger.warning(
                "County %s (%s) has no stations within %.1f km! Nearest station %s is %.1f km away",
                c_name, geoid, max_radius_km, st_dict[nearest_idx]["station_id"], dists[nearest_idx]
            )

        # Rank within radius: prioritize long-record stations, then shortest distance
        scores = (~s_is_long[within_idx]).astype(float) * 1000.0 + dists[within_idx]
        sorted_order = np.argsort(scores)
        ranked_indices = within_idx[sorted_order]

        for rank_num, s_idx in enumerate(ranked_indices, start=1):
            s_row = st_dict[s_idx]
            mapping_records.append(
                {
                    "GEOID": geoid,
                    "state": c_state,
                    "county": c_name,
                    "station_id": s_row["station_id"],
                    "station_name": s_row["name"],
                    "distance_km": round(float(dists[s_idx]), 2),
                    "latitude": s_row["latitude"],
                    "longitude": s_row["longitude"],
                    "elevation": s_row["elevation"],
                    "first_date": s_row["first_date"],
                    "last_date": s_row["last_date"],
                    "datacoverage": s_row["datacoverage"],
                    "is_long_record": s_row["is_long_record"],
                    "rank": rank_num,
                    "is_primary": (rank_num == 1),
                }
            )

    mapping_df = pd.DataFrame(mapping_records)

    # Compile station inventory: 1 row per unique station mapped to any county
    inventory_records = []
    for s_id, s_group in mapping_df.groupby("station_id"):
        rep_row = s_group.sort_values("distance_km").iloc[0]
        cov = rep_row["datacoverage"]
        inventory_records.append(
            {
                "station_id": s_id,
                "state": rep_row["state"],
                "county": rep_row["county"],
                "GEOID": rep_row["GEOID"],
                "latitude": rep_row["latitude"],
                "longitude": rep_row["longitude"],
                "elevation": rep_row["elevation"],
                "first_date": rep_row["first_date"],
                "last_date": rep_row["last_date"],
                "distance_to_primary_county_km": rep_row["distance_km"],
                "is_long_record": rep_row["is_long_record"],
                "datacoverage": round(float(cov), 4) if pd.notna(cov) else 0.0,
                "coverage_TMAX": round(float(cov), 4) if pd.notna(cov) else 0.0,
                "coverage_TMIN": round(float(cov), 4) if pd.notna(cov) else 0.0,
                "coverage_PRCP": round(float(cov), 4) if pd.notna(cov) else 0.0,
            }
        )

    inventory_df = pd.DataFrame(inventory_records).sort_values(["state", "county", "station_id"]).reset_index(drop=True)
    logger.info("Compiled inventory of %d distinct weather stations serving %d counties", len(inventory_df), len(counties_df))
    return inventory_df, mapping_df


def run_station_inventory(config_path: Path | str = "noaa_config.yaml") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete station inventory step and save artifacts."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_out_path = Path(cfg["paths"]["station_inventory"])

    client = NoaaCdoClient(
        cache_dir=cfg["paths"]["cache_dir"],
        rate_limit_per_sec=cfg["api"]["rate_limit_req_per_sec"],
        max_retries=cfg["api"]["max_retries"],
        timeout=cfg["api"]["timeout_seconds"],
    )

    data_file = Path("Paper3_MegaDataset_SPEI_FINAL.csv")
    counties_df = load_county_centroids(data_file, target_states=list(cfg["geography"]["states"].keys()))
    stations_df = discover_state_stations(client, cfg["geography"]["states"])

    inventory_df, mapping_df = build_station_inventory(
        counties_df,
        stations_df,
        max_radius_km=cfg["station_selection"]["max_search_radius_km"],
    )

    inventory_df.to_csv(inventory_out_path, index=False)
    mapping_df.to_csv(output_dir / "noaa_county_station_mapping.csv", index=False)
    logger.info("Station inventory saved -> %s", inventory_out_path)
    logger.info("County-station mapping saved -> %s", output_dir / "noaa_county_station_mapping.csv")

    return inventory_df, mapping_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_station_inventory()
