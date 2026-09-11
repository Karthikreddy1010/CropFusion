"""Canonical full re-run — round-5 (calibration-independent) configuration.

Runs the complete pipeline at the configured epoch budgets with config.py
UNMODIFIED. This is the run whose numbers are citable.

    python run_final_pipeline.py

Afterwards, both gates must pass:

    python code/results_integrity_check.py
    python code/final_quality_gate.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))

from main import main

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nFULL PIPELINE COMPLETE in {time.time() - t0:.1f}s")
