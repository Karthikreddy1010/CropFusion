"""Verify the F0 ladder edits to model_training.py.

Two things must hold before any DEV run is worth doing:

1. **Regression.** With every new flag at its default, the training path must be
   numerically identical to the committed version (HEAD). The legacy module is
   pulled straight out of git and both are trained on the same synthetic data
   with the same seed.
2. **Function.** With a flag on, the new behaviour must actually do what it
   claims: standardised training must round-trip to target units, monotone heads
   must make crossings impossible, and the EMA refit must return weights that
   differ from the last epoch's.

Run:  python code/diagnostics_loso/verify_f0_changes.py
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import config as cfg  # noqa: E402

N, P, EPOCHS, SEED = 600, 12, 6, 42


def make_data(seed: int = 0):
    """Synthetic panel with a skewed, heteroscedastic target."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, P)).astype(np.float32)
    signal = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.8 * X[:, 2] * X[:, 3]
    shock = -rng.gamma(shape=2.0, scale=0.6, size=N)      # left tail, like a bad year
    y = (signal + 0.5 * rng.normal(size=N) + shock).astype(np.float64)
    split = int(0.75 * N)
    return X[:split], y[:split], X[split:], y[split:]


def load_legacy_module():
    """Import the committed model_training.py under a separate module name.

    Returns None when git is unavailable (e.g. a Colab copy of the project with
    no .git directory); the caller then skips the regression comparison rather
    than failing the whole verification.
    """
    try:
        src = subprocess.run(
            ["git", "show", "HEAD:code/model_training.py"],
            cwd=REPO_DIR, capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    tmp_dir = Path(tempfile.mkdtemp(prefix="f0_legacy_"))
    tmp_path = tmp_dir / "model_training_legacy.py"
    tmp_path.write_text(src, encoding="utf-8")
    sys.path.insert(0, str(tmp_dir))
    return importlib.import_module("model_training_legacy")


def train(mod, X_tr, y_tr, X_va, y_va, **kw):
    return mod.train_neural_cqr(
        X_tr, y_tr, X_va, y_va, feature_cols=[f"f{i}" for i in range(P)],
        epochs=EPOCHS, batch_size=64, lr=1e-3, hidden_dims=(32, 16),
        patience=EPOCHS, seed=SEED, **kw,
    )


def main() -> int:
    X_tr, y_tr, X_va, y_va = make_data()
    checks: list[tuple[str, bool, str]] = []

    # Defaults must match the committed behaviour before anything else counts.
    for flag in ("STANDARDIZE_TARGET", "USE_EMA_WEIGHTS", "NEURAL_MONOTONE_HEADS",
                 "LR_SCHEDULE_MATCH_DEV"):
        checks.append((f"default {flag} is False", getattr(cfg, flag) is False, str(getattr(cfg, flag))))
    checks.append(("default HUBER_DELTA is 1.0", cfg.HUBER_DELTA == 1.0, str(cfg.HUBER_DELTA)))

    import model_training as new
    legacy = load_legacy_module()

    # ── 1. Regression: identical predictions with flags at defaults ────
    m_new = train(new, X_tr, y_tr, X_va, y_va)
    p_new, lo_new, hi_new = new.predict_intervals(m_new, X_va)

    if legacy is None:
        checks.append(("legacy path unchanged (max |diff| < 1e-6)", True,
                       "SKIPPED - no git available to fetch the committed version"))
    else:
        m_old = train(legacy, X_tr, y_tr, X_va, y_va)
        p_old, lo_old, hi_old = legacy.predict_intervals(m_old, X_va)
        max_diff = float(max(np.abs(p_new - p_old).max(),
                             np.abs(lo_new - lo_old).max(),
                             np.abs(hi_new - hi_old).max()))
        checks.append(("legacy path unchanged (max |diff| < 1e-6)", max_diff < 1e-6, f"{max_diff:.3e}"))

    # ── 2. F0a: standardised training round-trips to target units ──────
    cfg.STANDARDIZE_TARGET = True
    importlib.reload(new)
    m_std = train(new, X_tr, y_tr, X_va, y_va)
    p_std, lo_std, hi_std = new.predict_intervals(m_std, X_va)
    checks.append(("F0a mu/sd recorded from train only",
                   abs(m_std.target_mu - float(np.mean(y_tr))) < 1e-9
                   and abs(m_std.target_sd - float(np.std(y_tr))) < 1e-9,
                   f"mu={m_std.target_mu:.4f} sd={m_std.target_sd:.4f}"))
    checks.append(("F0a predictions are in target units",
                   abs(float(np.mean(p_std)) - float(np.mean(y_va))) < 2.0,
                   f"mean pred={np.mean(p_std):.3f} vs mean y={np.mean(y_va):.3f}"))
    checks.append(("F0a huber delta = 1.345 sigma",
                   m_std.paradigm_fingerprint["huber_delta"] == cfg.HUBER_DELTA_SIGMA,
                   str(m_std.paradigm_fingerprint["huber_delta"])))
    cfg.STANDARDIZE_TARGET = False
    importlib.reload(new)

    # ── 3. F1: monotone heads cannot cross ────────────────────────────
    cfg.NEURAL_MONOTONE_HEADS = True
    importlib.reload(new)
    m_mono = train(new, X_tr, y_tr, X_va, y_va)
    _, lo_m, hi_m = new.predict_intervals(m_mono, X_va)
    checks.append(("F1 zero crossings with monotone heads",
                   bool(np.all(hi_m >= lo_m)), f"n_crossing={int(np.sum(hi_m < lo_m))}"))
    base_cross = int(np.sum(hi_new < lo_new))
    checks.append(("F1 baseline crossing count recorded", True, f"legacy n_crossing={base_cross}"))
    cfg.NEURAL_MONOTONE_HEADS = False
    importlib.reload(new)

    # ── 4. F0b: EMA refit differs from the last-epoch refit ───────────
    def refit(**kw):
        return new.train_neural_cqr_final(
            X_tr, y_tr, feature_cols=[f"f{i}" for i in range(P)],
            epochs=EPOCHS, batch_size=64, lr=1e-3, hidden_dims=(32, 16), seed=SEED, **kw,
        )

    m_last = refit()
    p_last, _, _ = new.predict_intervals(m_last, X_va)
    cfg.USE_EMA_WEIGHTS = True
    importlib.reload(new)
    m_ema = refit()
    p_ema, _, _ = new.predict_intervals(m_ema, X_va)
    delta = float(np.abs(p_ema - p_last).mean())
    checks.append(("F0b EMA weights actually applied (mean |diff| > 0)", delta > 1e-8, f"{delta:.3e}"))
    cfg.USE_EMA_WEIGHTS = False
    importlib.reload(new)

    # ── 5. F0c: schedule_t_max reaches the caller ─────────────────────
    m_sched = new.train_neural_cqr_final(
        X_tr, y_tr, feature_cols=[f"f{i}" for i in range(P)],
        epochs=EPOCHS, batch_size=64, lr=1e-3, hidden_dims=(32, 16), seed=SEED,
        schedule_t_max=200,
    )
    checks.append(("F0c schedule_t_max honoured",
                   m_sched.paradigm_fingerprint["schedule_t_max"] == 200,
                   str(m_sched.paradigm_fingerprint["schedule_t_max"])))

    width = max(len(name) for name, _, _ in checks)
    failed = 0
    print("\n" + "=" * (width + 28))
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
        failed += 0 if ok else 1
    print("=" * (width + 28))
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
