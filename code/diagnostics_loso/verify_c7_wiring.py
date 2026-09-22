"""Check how main.py obtains the DEV-selected LOSO hyperparameters (audit C7).

The tuning must happen exactly once and then be reused. If it reran on every
pipeline invocation the selected configuration could drift between runs, and the
frozen protocol would be describing a moving target; if it never ran, the LOSO
numbers would silently keep the held-out-informed settings the audit objected to.

So the contract is: run the tuner when the selection file is absent, read it when
it is present, never touch it when the flag is off, and if tuning fails, carry on
with the config defaults and say so rather than taking the pipeline down 40
minutes into a run.

The tuner itself is injected here, so these checks cost nothing; the statistics
inside it are not under test (that is loso_dev_tuning.py's own job).

Run:  python code/diagnostics_loso/verify_c7_wiring.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import config as cfg  # noqa: E402
import main as pipeline  # noqa: E402


class SpyTuner:
    """Stands in for loso_dev_tuning.main, and counts how often it was asked to run."""

    def __init__(self, path: Path, states, fail: bool = False) -> None:
        self.calls = 0
        self.path = path
        self.states = states
        self.fail = fail

    def __call__(self) -> int:
        self.calls += 1
        if self.fail:
            raise RuntimeError("simulated tuning failure (e.g. CUDA out of memory)")
        self.path.write_text(json.dumps({
            "per_state_selection": {s: {"hidden_dims": [128, 64], "lr": 5e-4, "dropout": 0.2,
                                        "weight_decay": 5e-4, "batch_size": 96}
                                    for s in self.states}}), encoding="utf-8")
        return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    checks = []
    states = list(cfg.LOSO_STATES)

    with tempfile.TemporaryDirectory() as tmp:
        sel = Path(tmp) / "loso_dev_selected_hyperparams.json"

        # 1 - flag on, no selection yet: the tuner runs and the file appears
        spy = SpyTuner(sel, states)
        src = pipeline.ensure_loso_dev_hyperparameters(enabled=True, path=sel, runner=spy)
        print(f"missing file -> tuner calls={spy.calls}, source={src!r}")
        checks.append(("a missing selection triggers exactly one tuning run", spy.calls == 1))
        checks.append(("the selection file exists afterwards", sel.exists()))
        checks.append(("the fold hyperparameters are reported as DEV-selected",
                       "DEV-selected" in src))

        # 2 - second call with the file present: cached, not recomputed
        src2 = pipeline.ensure_loso_dev_hyperparameters(enabled=True, path=sel, runner=spy)
        print(f"file present -> tuner calls={spy.calls} (unchanged), source={src2!r}")
        checks.append(("an existing selection is reused, not recomputed", spy.calls == 1))
        checks.append(("the reused selection is still reported as DEV-selected",
                       "DEV-selected" in src2))

        # 3 - every held-out state must have an entry, or folds fall back silently
        payload = json.loads(sel.read_text(encoding="utf-8"))
        covered = set(payload.get("per_state_selection", {}))
        missing = [s for s in states if s not in covered]
        print(f"states covered by the selection: {len(covered)}/{len(states)}")
        checks.append(("every LOSO state has a selected configuration", not missing))

        # 4 - flag off: the tuner is never invoked, defaults are declared as such
        sel_off = Path(tmp) / "unused.json"
        spy_off = SpyTuner(sel_off, states)
        src_off = pipeline.ensure_loso_dev_hyperparameters(enabled=False, path=sel_off,
                                                           runner=spy_off)
        print(f"flag off -> tuner calls={spy_off.calls}, source={src_off!r}")
        checks.append(("the tuner does not run when the flag is off", spy_off.calls == 0))
        checks.append(("no selection file is created when the flag is off", not sel_off.exists()))
        checks.append(("config defaults are declared with the C7 caveat",
                       "C7" in src_off or "config defaults" in src_off))

        # 5 - a partial selection must not pass as a complete one. The tuner has a
        # smoke mode (--states Minnesota --grid 2), and if that output landed at the
        # cache path the next pipeline run would reuse it and quietly leave five
        # folds on the held-out-informed defaults.
        partial = Path(tmp) / "sel_one_state.json"
        partial.write_text(json.dumps({"per_state_selection": {states[0]: {"lr": 5e-4}}}),
                           encoding="utf-8")
        spy_partial = SpyTuner(partial, states)
        src_partial = pipeline.ensure_loso_dev_hyperparameters(enabled=True, path=partial,
                                                               runner=spy_partial)
        print(f"partial selection (1/{len(states)} states) -> source={src_partial!r}")
        checks.append(("a selection covering only some states is reported as partial",
                       "PARTIAL" in src_partial))
        # Counted exactly, and the covered state must not appear in the missing list.
        # A looser assertion here passed while the count was wrong for every state.
        checks.append((f"exactly {len(states) - 1} states are reported missing",
                       f"{len(states) - 1} of {len(states)} states missing" in src_partial))
        checks.append(("the state that IS covered is not listed as missing",
                       states[0] not in src_partial.split("missing:")[-1]))

        # 6 - tuning failure must not take the pipeline down
        sel_fail = Path(tmp) / "fails.json"
        spy_fail = SpyTuner(sel_fail, states, fail=True)
        try:
            src_fail = pipeline.ensure_loso_dev_hyperparameters(enabled=True, path=sel_fail,
                                                                runner=spy_fail)
            survived = True
        except Exception as exc:                       # noqa: BLE001 - that is the failure
            src_fail, survived = f"raised {type(exc).__name__}", False
        print(f"tuning failure -> survived={survived}, source={src_fail!r}")
        checks.append(("a tuning failure does not abort the pipeline", survived))
        checks.append(("a failed tuning falls back to declared config defaults",
                       survived and "config defaults" in src_fail))

    # 7 - the C7 path is only reached when the flag is on, so a NameError inside it
    # stays dormant until the run that matters. Two such bugs were found this way:
    # both this module and the per-fold block used `json` without importing it.
    # Undefined names are cheap to check and cost 40 minutes to discover at runtime.
    import subprocess
    pf = subprocess.run([sys.executable, "-m", "pyflakes", str(CODE_DIR / "main.py")],
                        capture_output=True, text=True)
    undefined = [ln for ln in pf.stdout.splitlines() if "undefined name" in ln]
    print(f"undefined names in main.py: {len(undefined)}")
    for ln in undefined:
        print(f"   {ln}")
    checks.append(("main.py has no undefined names", not undefined))

    failed = 0
    print()
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        failed += 0 if ok else 1
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
