import os
from pathlib import Path

def scan_era5_data():
    project_root = Path(__file__).resolve().parents[3]
    print(f"Scanning project root: {project_root}\n")

    era5_candidates = []
    
    # Check known configured directories
    known_dirs = [
        project_root / "data" / "era5_daily",
        project_root / "data" / "era5",
        project_root / "data" / "era5_hourly",
        project_root / "data" / "era5_land",
        project_root / "outputs" / "era5_daily_download",
        project_root / "outputs" / "era5_data_audit",
    ]

    for kd in known_dirs:
        if kd.exists():
            for p in kd.rglob("*"):
                if p.is_file():
                    era5_candidates.append(p)

    # General recursive scan across project for era5-land data files
    for p in project_root.rglob("*"):
        if p.is_file():
            name_lower = p.name.lower()
            path_lower = str(p).lower()
            
            # Skip prism, git, pycache, .gemini
            if any(skip in path_lower for skip in ["prism", ".git", "__pycache__", ".gemini", "noaa_storm"]):
                continue

            if any(term in name_lower or term in path_lower for term in ["era5_daily", "era5_land", "reanalysis-era5-land"]):
                if p not in era5_candidates:
                    era5_candidates.append(p)

    # Sort and group candidates
    print(f"Total ERA5 files/data identified: {len(era5_candidates)}")
    total_bytes = 0
    by_ext = {}
    by_folder = {}

    for p in sorted(era5_candidates):
        sz = p.stat().st_size
        total_bytes += sz
        ext = p.suffix.lower()
        by_ext[ext] = by_ext.get(ext, 0) + 1
        folder = str(p.parent.relative_to(project_root))
        by_folder[folder] = by_folder.get(folder, 0) + 1

    print(f"Total Size: {total_bytes / (1024**2):.2f} MB ({total_bytes / (1024**3):.2f} GB)\n")

    print("Breakdown by Folder:")
    for folder, count in sorted(by_folder.items()):
        print(f"  {folder}: {count} files")

    print("\nBreakdown by Extension:")
    for ext, count in sorted(by_ext.items()):
        print(f"  {ext or '[no ext]'}: {count} files")

    print("\nSample of files identified (first 25):")
    for p in sorted(era5_candidates)[:25]:
        rel = p.relative_to(project_root)
        print(f"  {rel} ({p.stat().st_size / (1024**2):.2f} MB)")

if __name__ == "__main__":
    scan_era5_data()
