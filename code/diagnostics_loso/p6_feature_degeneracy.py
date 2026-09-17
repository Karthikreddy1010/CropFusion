"""p6_feature_degeneracy.py - which selected features are degenerate on the partitions
that actually get scored.

A feature can rank first on the FIT partition and still carry zero information on
DEV/CAL/TEST if its construction rule makes it unavailable there and the
train-fitted imputer replaces it with a single constant. This measures that
directly, per partition, on the arrays the models are handed.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from _fold_lab import build_loso_fold, build_temporal, cfg, DIAG_OUT


def constant_columns(X, cols, tol=1e-9):
    sd = X.std(axis=0)
    return [c for c, s in zip(cols, sd) if s <= tol]


def audit(state: str) -> dict:
    fold = build_loso_fold(state)
    cols = fold["feature_cols"]
    ranks = fold["selection_report"].get("consensus_votes", {})
    out = {"state": state, "n_features": len(cols)}
    for part in ("fit", "dev", "cal", "test"):
        const = constant_columns(fold["X"][part], cols)
        out[part] = {"n_rows": int(len(fold["X"][part])), "n_constant_features": len(const),
                     "constant_features": const}
    # where do the constants sit in the fold's own consensus ranking?
    sel = fold["selection_report"].get("feature_ranking", [])
    order = {r.get("feature"): i + 1 for i, r in enumerate(sel)} if sel else {}
    out["test_constant_feature_ranks"] = {c: order.get(c) for c in out["test"]["constant_features"]}
    out["lag_values_on_test"] = {}
    for c in ("Yield_lag1", "Yield_lag2"):
        if c in fold["frames"]["test"].columns:
            v = fold["frames"]["test"][c].values
            out["lag_values_on_test"][c] = {"unique": int(len(np.unique(v))),
                                            "value": round(float(v[0]), 4)}
    out["train_target_median_fill"] = round(float(fold["prep"].train_target_median_), 4)
    return out


def temporal_audit() -> dict:
    tp = build_temporal()
    cols = tp["feature_cols"]
    out = {"n_features": len(cols)}
    for part in ("fit", "dev", "cal", "test"):
        const = constant_columns(tp["X"][part], cols)
        out[part] = {"n_rows": int(len(tp["X"][part])), "n_constant_features": len(const),
                     "constant_features": const}
    return out


if __name__ == "__main__":
    states = sys.argv[1:] or list(cfg.LOSO_STATES)
    res = {"loso": {}, "temporal": temporal_audit()}
    print("TEMPORAL constant features per partition:")
    for p in ("fit", "dev", "cal", "test"):
        print(f"  {p:5s} n={res['temporal'][p]['n_rows']:6d}  constant={res['temporal'][p]['n_constant_features']:3d} "
              f"{res['temporal'][p]['constant_features']}", flush=True)
    for st in states:
        r = audit(st)
        res["loso"][st] = r
        print(f"\nLOSO {st}: {r['n_features']} features")
        for p in ("fit", "dev", "cal", "test"):
            print(f"  {p:5s} n={r[p]['n_rows']:6d}  constant={r[p]['n_constant_features']:3d} "
                  f"{r[p]['constant_features']}", flush=True)
        print(f"  lag values on TEST: {r['lag_values_on_test']}  (fill={r['train_target_median_fill']})", flush=True)
    out = DIAG_OUT / "reports" / "p6_feature_degeneracy.json"
    out.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {out}")
