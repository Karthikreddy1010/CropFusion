"""
run_both_crops.py - Run the full Paper 3 pipeline once for corn and once
for soybean (parallel single-crop runs, not a shared multi-task model --
see CROP_COMPARISON_NOTES.md for why).

Each crop is run as a SEPARATE subprocess (not an in-process function
call) because config.py reads PAPER3_CROP_TARGET at import time to set
PRIMARY_TARGET/SECONDARY_TARGET and OUTPUT_DIR -- a fresh subprocess is
the simplest way to guarantee a completely clean config for each crop,
with zero risk of state leaking between runs (module-level constants in
config.py can't be "reset" mid-process once other modules have already
imported and cached them).

Usage:
    python3 run_both_crops.py                  # both crops, full pipeline
    python3 run_both_crops.py --crop corn       # just corn
    python3 run_both_crops.py --crop soy        # just soy
    python3 run_both_crops.py --skip-compare    # skip the final comparison step

Outputs:
    outputs/        -- corn results (backward-compatible path, same as
                        every prior single-crop run)
    outputs_soy/    -- soybean results, same structure
    crop_comparison_report.json / .md  -- side-by-side summary (written
        to the project root, not inside either crop's output dir, since
        it's neither crop's result alone)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("run_both_crops")

CODE_DIR = Path(__file__).parent
PROJECT_ROOT = CODE_DIR.parent


def _output_dir_for_crop(crop: str) -> Path:
    """Must exactly mirror config.py's OUTPUT_DIR logic: corn (the
    default) writes to outputs/ for backward compatibility with every
    prior run; only soy gets its own outputs_soy/ directory. This is
    the single source of truth for that mapping in this file -- do not
    hardcode f"outputs_{crop}" anywhere else below."""
    return PROJECT_ROOT / "outputs" if crop == "corn" else PROJECT_ROOT / f"outputs_{crop}"


def run_crop(crop: str) -> bool:
    """Run main.py as a subprocess with PAPER3_CROP_TARGET set. Returns
    True on success (exit code 0), False otherwise -- does not raise, so
    one crop failing doesn't prevent the other from being attempted."""
    assert crop in ("corn", "soy")
    env = os.environ.copy()
    env["PAPER3_CROP_TARGET"] = crop

    logger.info("=" * 70)
    logger.info("Starting full pipeline run for crop=%s (%s/)", crop, _output_dir_for_crop(crop).name)
    logger.info("=" * 70)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "main.py"],
        cwd=str(CODE_DIR),
        env=env,
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        logger.info("crop=%s finished OK in %.0fs (%.1f min)", crop, elapsed, elapsed / 60)
        return True
    else:
        logger.error(
            "crop=%s FAILED (exit code %d) after %.0fs -- see %s/pipeline.log for details",
            crop, result.returncode, elapsed, _output_dir_for_crop(crop),
        )
        return False


def _load_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not read %s: %s", path, e)
        return None


def build_comparison() -> None:
    """Pull the headline numbers from both crops' reports/ directories
    into a single side-by-side comparison file. Purely a reporting step
    -- reads already-computed reports, does not retrain anything."""
    rows = []
    for crop in ("corn", "soy"):
        out_dir = _output_dir_for_crop(crop)
        ensemble = _load_json(out_dir / "reports" / "ensemble_blend_report.json") or {}
        loso_summary_csv = out_dir / "reports" / "loso_summary.csv"
        robustness = _load_json(out_dir / "reports" / "backbone_robustness_summary.json") or {}

        loso_r2_mean, loso_r2_std, loso_picp_mean = None, None, None
        if loso_summary_csv.exists():
            import pandas as pd
            loso_df = pd.read_csv(loso_summary_csv).set_index("Metric")
            if "R_SQUARED" in loso_df.index:
                loso_r2_mean = float(loso_df.loc["R_SQUARED", "Mean"])
                loso_r2_std = float(loso_df.loc["R_SQUARED", "Std"])
            if "PICP" in loso_df.index:
                loso_picp_mean = float(loso_df.loc["PICP", "Mean"])

        rows.append({
            "crop": crop,
            "backbone_neuralcqr_r2": ensemble.get("neuralcqr_alone_test_r2"),
            "backbone_ensemble_r2": ensemble.get("test_r2"),
            "backbone_ensemble_r2_multiseed_mean": robustness.get("ensemble_test_r2_mean"),
            "backbone_ensemble_r2_multiseed_std": robustness.get("ensemble_test_r2_std"),
            "backbone_ensemble_rmse": ensemble.get("test_rmse"),
            "backbone_ensemble_picp": ensemble.get("test_picp"),
            "loso_r2_mean": loso_r2_mean,
            "loso_r2_std": loso_r2_std,
            "loso_picp_mean": loso_picp_mean,
        })

    comparison = {"crops": rows}
    out_json = PROJECT_ROOT / "crop_comparison_report.json"
    with open(out_json, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Comparison report saved -> %s", out_json)

    # Markdown table for easy copy-paste into the paper
    md = "# Corn vs. Soybean -- Pipeline Comparison\n\n"
    md += "| Metric | Corn | Soybean |\n|---|---|---|\n"
    corn_row = rows[0] if rows[0]["crop"] == "corn" else rows[1]
    soy_row = rows[0] if rows[0]["crop"] == "soy" else rows[1]
    metric_labels = [
        ("backbone_neuralcqr_r2", "Backbone NeuralCQR R\u00b2 (temporal split)"),
        ("backbone_ensemble_r2", "Backbone Ensemble R\u00b2 (temporal split)"),
        ("backbone_ensemble_r2_multiseed_mean", "Backbone Ensemble R\u00b2 (multi-seed mean)"),
        ("backbone_ensemble_r2_multiseed_std", "Backbone Ensemble R\u00b2 (multi-seed std)"),
        ("backbone_ensemble_rmse", "Backbone Ensemble RMSE"),
        ("backbone_ensemble_picp", "Backbone Ensemble PICP"),
        ("loso_r2_mean", "LOSO-CV R\u00b2 (mean across 6 states)"),
        ("loso_r2_std", "LOSO-CV R\u00b2 (std across 6 states)"),
        ("loso_picp_mean", "LOSO-CV PICP (mean across 6 states)"),
    ]
    for key, label in metric_labels:
        c_val = corn_row.get(key)
        s_val = soy_row.get(key)
        c_str = f"{c_val:.4f}" if isinstance(c_val, (int, float)) else "N/A"
        s_str = f"{s_val:.4f}" if isinstance(s_val, (int, float)) else "N/A"
        md += f"| {label} | {c_str} | {s_str} |\n"

    out_md = PROJECT_ROOT / "crop_comparison_report.md"
    with open(out_md, "w") as f:
        f.write(md)
    logger.info("Comparison markdown saved -> %s", out_md)
    print("\n" + md)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", choices=["corn", "soy", "both"], default="both")
    parser.add_argument("--skip-compare", action="store_true", help="Don't build the final comparison report")
    args = parser.parse_args()

    crops = ["corn", "soy"] if args.crop == "both" else [args.crop]

    results = {}
    for crop in crops:
        results[crop] = run_crop(crop)

    logger.info("=" * 70)
    logger.info("SUMMARY: %s", {k: ("OK" if v else "FAILED") for k, v in results.items()})
    logger.info("=" * 70)

    if not args.skip_compare and len(crops) == 2 and all(results.values()):
        build_comparison()
    elif not args.skip_compare and len(crops) == 2:
        logger.warning("Skipping comparison report: at least one crop's run failed.")


if __name__ == "__main__":
    main()
