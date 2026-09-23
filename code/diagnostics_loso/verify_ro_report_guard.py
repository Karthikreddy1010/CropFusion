"""A degraded objectives report must not overwrite a complete one.

This has now cost the project twice. `objectives_ro.py` computes RO1, RO2, RO4
and RO5 from `rolling_origin_rows.csv`, which is produced locally and does not
exist on a fresh Colab session. A pipeline run there therefore writes a report
in which four of the five objectives read "requires rolling_origin_rows.csv" --
correct behaviour for that machine. When those outputs are synced back, the
complete local report is replaced by the degraded one, silently, and the paper's
objective results appear to have vanished.

The fix is not to make Colab compute them. It is that a writer holding less
information than the file already on disk should not win. So:

  - strictly more objectives computed  -> write, as normal;
  - equal                              -> write, since it is a refresh;
  - strictly fewer                     -> keep the existing file, and put the
                                          degraded one beside it as
                                          *.partial.json so nothing is hidden.

Run:  python code/diagnostics_loso/verify_ro_report_guard.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import objectives_ro as ro  # noqa: E402

KEYS = ("RO1", "RO2", "RO3", "RO4", "RO5")


def report(computed) -> dict:
    """A report in which `computed` carry results and the rest carry a status."""
    out = {"generated": "test"}
    for k in KEYS:
        out[k] = {"objective": k, "headline": "x"} if k in computed else \
                 {"objective": k, "status": "requires rolling_origin_rows.csv"}
    return out


def n_computed(path: Path) -> int:
    d = json.loads(path.read_text(encoding="utf-8"))
    return sum(1 for k in KEYS if "status" not in d.get(k, {"status": 1}))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    checks = []

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "objectives_RO_report.json"
        partial = p.with_suffix(".partial.json")

        # a complete report is written to an empty directory
        ro.persist_report(report(KEYS), p)
        checks.append(("a complete report writes when nothing is there", n_computed(p) == 5))

        # the Colab case: one objective against five already on disk
        ro.persist_report(report(("RO3",)), p)
        print(f"after a 1-objective write over a 5-objective file: {n_computed(p)} computed")
        checks.append(("a degraded report does not overwrite a complete one", n_computed(p) == 5))
        checks.append(("the degraded report is kept beside it, not discarded", partial.exists()))
        checks.append(("the kept copy is the degraded one", partial.exists() and
                       n_computed(partial) == 1))

        # a refresh with the same coverage must still land
        fresh = report(KEYS); fresh["generated"] = "refreshed"
        ro.persist_report(fresh, p)
        checks.append(("an equal-coverage refresh does write",
                       json.loads(p.read_text(encoding="utf-8"))["generated"] == "refreshed"))

        # and a genuine improvement must land
        p2 = Path(tmp) / "b.json"
        ro.persist_report(report(("RO3",)), p2)
        ro.persist_report(report(KEYS), p2)
        checks.append(("a more complete report does overwrite a degraded one",
                       n_computed(p2) == 5))

        # an unreadable existing file must not block the write
        p3 = Path(tmp) / "c.json"
        p3.write_text("{ this is not json", encoding="utf-8")
        ro.persist_report(report(("RO3",)), p3)
        checks.append(("a corrupt existing file does not block writing", n_computed(p3) == 1))

    failed = 0
    print()
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        failed += 0 if ok else 1
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
