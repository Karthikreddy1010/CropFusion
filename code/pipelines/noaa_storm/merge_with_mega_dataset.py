#!/usr/bin/env python3
"""
merge_with_mega_dataset.py

Merges noaa_storm_county_year_1985_2023.parquet with
Paper3_MegaDataset_SPEI_FINAL(1).csv on GEOID + Year, producing
Paper3_NOAA_Storm_Enhanced.csv. Never overwrites existing mega-dataset
columns; never merges on County_Name.
"""

from __future__ import annotations

import argparse

import pandas as pd

from storm_common import load_config, resolve_path, setup_logger


def main():
    parser = argparse.ArgumentParser(description="Merge storm county-year features into Paper 3 mega dataset")
    parser.add_argument("--config", default="config/storm_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("merge_with_mega_dataset", cfg, "merge_with_mega_dataset.log")

    mega_path = resolve_path(cfg, "paths.mega_dataset_csv")
    storm_path = resolve_path(cfg, "paths.processed_dir") / "noaa_storm_county_year_1985_2023.parquet"

    if not mega_path.exists():
        logger.error(f"Mega dataset not found at {mega_path}.")
        return
    if not storm_path.exists():
        logger.error(f"{storm_path} not found — run aggregate_storm_county_year.py first.")
        return

    mega_df = pd.read_csv(mega_path, dtype={"GEOID": str})
    mega_df["GEOID"] = mega_df["GEOID"].astype(str).str.zfill(5)

    storm_df = pd.read_parquet(storm_path)
    storm_df["GEOID"] = storm_df["GEOID"].astype(str).str.zfill(5)

    overlap = (set(mega_df.columns) & set(storm_df.columns)) - {"GEOID", "Year", "State", "County_Name"}
    if overlap:
        logger.warning(
            f"Column name overlap between mega dataset and storm features: {overlap}. "
            "Storm columns will be suffixed '_storm' to avoid overwriting existing variables."
        )
        storm_df = storm_df.rename(columns={c: f"{c}_storm" for c in overlap})

    # Drop duplicate metadata columns from storm_df before merging
    storm_feature_cols = [c for c in storm_df.columns if c not in ["State", "County_Name"]]
    merged = mega_df.merge(storm_df[storm_feature_cols], on=["GEOID", "Year"], how="left")

    n_matched = (merged["storm_data_status"].notna()).sum() if "storm_data_status" in merged else 0
    logger.info(f"Merged {len(merged)} rows; {n_matched}/{len(merged)} matched the storm county-year skeleton.")

    out_dir = resolve_path(cfg, "paths.processed_dir")
    out_path = out_dir / "Paper3_NOAA_Storm_Enhanced.csv"
    merged.to_csv(out_path, index=False)
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
