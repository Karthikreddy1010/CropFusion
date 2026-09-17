import pandas as pd
from pathlib import Path

pq_path = Path("data/prism_pipeline/processed/prism_county_daily_stage_checkpoint.parquet")
prod_path = Path("data/prism_pipeline/processed/prism_county_daily_1985_2023.parquet")

print("Checking Parquet Datasets:")
print(f"  Production Parquet Exists: {prod_path.exists()}")
if prod_path.exists():
    print(f"  WARNING: Production parquet exists unexpectedly! Size: {prod_path.stat().st_size}")

if pq_path.exists():
    df = pd.read_parquet(pq_path)
    print(f"Staging Checkpoint Parquet:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Unique counties (GEOID): {df['GEOID'].nunique()}")
    print(f"  Unique dates: {df['Date'].nunique()}")
    print(f"  Columns: {list(df.columns)}")
    
    inv = (df['TMIN'] > df['TMAX']).sum()
    neg_ppt = (df['PPT'] < 0).sum()
    null_tmax = df['TMAX'].isna().sum()
    null_tmin = df['TMIN'].isna().sum()
    null_ppt = df['PPT'].isna().sum()
    print(f"  Inversions (Tmin > Tmax): {inv}")
    print(f"  Negative PPT: {neg_ppt}")
    print(f"  Nulls -> TMAX: {null_tmax}, TMIN: {null_tmin}, PPT: {null_ppt}")
else:
    print("Staging parquet not found!")
