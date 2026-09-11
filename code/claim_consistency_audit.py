"""
claim_consistency_audit.py - Scan generated outputs for unsupported claims.

After the pipeline runs, every generated ``.json`` / ``.csv`` / ``.md`` / ``.txt``
/ ``.log`` file is scanned for phrases that assert a conclusion. Each hit is
checked against the artefact that would have to support it. A phrase is only
allowed to stand if the corresponding computed result backs it.

This exists because the previous outputs asserted "100% compliance", "zero
leakage", "7 folds", "High cross-backbone attribution alignment", "joint
training improves RMSE", "significantly positively correlated" and "ACI achieves
top rank" while the implementation demonstrated none of those things.

Run standalone:  python claim_consistency_audit.py
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import config as cfg

logger = logging.getLogger("paper3")

SCAN_SUFFIXES = {".json", ".csv", ".md", ".txt", ".log"}


@dataclass
class ClaimRule:
    name: str
    pattern: str
    requires: str          # human-readable description of the supporting evidence
    checker: str           # key into the SUPPORT dispatch below


RULES: List[ClaimRule] = [
    ClaimRule("full_compliance", r"100\s*%\s*(fully\s*)?complian|FULL_METHODOLOGY_COMPLIANCE",
              "methodology_validation_report.json shows zero FAIL and zero WARNING",
              "compliance"),
    ClaimRule("zero_leakage", r"zero\s+(data\s+)?leakage|ZERO_DATA_LEAKAGE",
              "leakage_provenance_audit.json shows all_passed = true", "leakage"),
    ClaimRule("seven_folds", r"\b(7|seven)[\s-]+(state\s+)?(folds?|spatial\s+folds?)\b",
              f"the protocol has exactly {len(cfg.LOSO_STATES)} folds — this phrase is always stale",
              "never"),
    ClaimRule("high_shap_alignment", r"[Hh]igh\s+cross-?backbone|high\s+attribution\s+alignment|[Hh]igh\s+SHAP",
              "shap_consistency_report.json agreement_label is 'strong agreement'", "shap_high"),
    ClaimRule("joint_improves", r"[Jj]oint\s+(end-to-end\s+)?(Neural\s+CQR\s+)?(training\s+)?improves",
              "objective_O4_report.json shows a supported improvement", "o4_improves"),
    ClaimRule("strong_relationship", r"strong(ly)?\s+(positive\s+)?(correlat|relationship|associat)"
                                     r"|significantly\s+positively\s+correlated",
              "objective_o5_report.json effect_magnitude_label is 'substantial'", "o5_strong"),
    ClaimRule("statistically_superior", r"statistically\s+superior|achieves\s+top\s+rank"
                                        r"|superior\s+coverage\s+guarantee",
              "objective_O6_report.json claims_statistical_superiority is true", "o6_superior"),
]


def _read_json(name: str) -> Optional[Dict[str, Any]]:
    p = cfg.REPORT_DIR / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _support() -> Dict[str, Dict[str, Any]]:
    """Evaluate whether each claim type is currently supported by the results."""
    val = _read_json("methodology_validation_report.json")
    leak = _read_json("leakage_provenance_audit.json")
    shap_rpt = _read_json("shap_consistency_report.json")
    o4 = _read_json("objective_O4_report.json")
    o5 = _read_json("objective_o5_report.json")
    o6 = _read_json("objective_O6_report.json")

    counts = (val or {}).get("counts", {})
    compliance_ok = bool(val) and counts.get("FAIL", 1) == 0 and counts.get("WARNING", 1) == 0

    return {
        "compliance": {
            "supported": compliance_ok,
            "evidence": (f"validator status={(val or {}).get('overall_status')}, "
                         f"FAIL={counts.get('FAIL')}, WARNING={counts.get('WARNING')}"),
        },
        "leakage": {
            "supported": bool((leak or {}).get("all_passed")),
            "evidence": (f"leakage audit all_passed={(leak or {}).get('all_passed')} over "
                         f"{(leak or {}).get('n_experiments')} experiments"),
        },
        "never": {"supported": False,
                  "evidence": f"config.LOSO_STATES has {len(cfg.LOSO_STATES)} states"},
        "shap_high": {
            "supported": (shap_rpt or {}).get("agreement_label") == "strong agreement",
            "evidence": f"SHAP agreement_label={(shap_rpt or {}).get('agreement_label')}, "
                        f"similarity={(shap_rpt or {}).get('shap_similarity_index')}",
        },
        "o4_improves": {
            "supported": bool((o4 or {}).get("point_prediction_improvement_supported")
                              or (o4 or {}).get("interval_quality_improvement_supported")),
            "evidence": f"O4 verdict={(o4 or {}).get('verdict')}",
        },
        "o5_strong": {
            "supported": (o5 or {}).get("effect_magnitude_label") == "substantial",
            "evidence": (f"O5 magnitude={(o5 or {}).get('effect_magnitude_label')}, "
                         f"R2={(o5 or {}).get('variance_explained_fraction')}, "
                         f"detectable={(o5 or {}).get('statistically_detectable')}"),
        },
        "o6_superior": {
            "supported": bool((o6 or {}).get("claims_statistical_superiority")),
            "evidence": (f"O6 ranking_status={(o6 or {}).get('ranking_status')}, "
                         f"claims_superiority={(o6 or {}).get('claims_statistical_superiority')}"),
        },
    }


def run_claim_consistency_audit(scan_root: Optional[Path] = None) -> Dict[str, Any]:
    """Scan every generated artefact and flag claims the results do not support."""
    from utils import save_report, save_report_markdown

    root = scan_root or cfg.OUTPUT_DIR
    support = _support()
    compiled = [(r, re.compile(r.pattern)) for r in RULES]

    findings: List[Dict[str, Any]] = []
    files_scanned = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SCAN_SUFFIXES:
            continue
        # Do not flag this audit's own output, which necessarily quotes the phrases.
        if path.name.startswith("claim_consistency_audit"):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        files_scanned += 1

        for line_no, line in enumerate(text.splitlines(), 1):
            for rule, rx in compiled:
                if not rx.search(line):
                    continue
                sup = support.get(rule.checker, {"supported": False, "evidence": "unknown"})
                findings.append({
                    "file": str(path.relative_to(root)),
                    "line": line_no,
                    "claim_type": rule.name,
                    "matched_text": line.strip()[:220],
                    "requires": rule.requires,
                    "supported_by_results": bool(sup["supported"]),
                    "evidence": sup["evidence"],
                    "status": "OK" if sup["supported"] else "UNSUPPORTED",
                })

    unsupported = [f for f in findings if f["status"] == "UNSUPPORTED"]
    by_type: Dict[str, int] = {}
    for f in unsupported:
        by_type[f["claim_type"]] = by_type.get(f["claim_type"], 0) + 1

    report = {
        "files_scanned": files_scanned,
        "scan_root": str(root),
        "claim_types_checked": [r.name for r in RULES],
        "support_evaluation": support,
        "n_matches": len(findings),
        "n_unsupported": len(unsupported),
        "unsupported_by_type": by_type,
        "status": "PASS" if not unsupported else "FAIL",
        "findings": findings,
    }
    save_report(report, "claim_consistency_audit.json")

    md = f"""# Output Claim Consistency Audit

Scanned **{files_scanned}** generated files under `{root}` for phrases that assert a
conclusion, and checked each against the artefact that would have to support it.

**Status: `{report['status']}`** — {len(unsupported)} unsupported claim(s) out of
{len(findings)} phrase match(es).

## Support evaluation
| Claim type | Currently supported | Evidence |
|---|---|---|
"""
    for r in RULES:
        s = support[r.checker]
        md += f"| {r.name} | {s['supported']} | {s['evidence']} |\n"

    if unsupported:
        md += "\n## Unsupported claims found\n\n| File | Line | Claim type | Text |\n|---|---:|---|---|\n"
        for f in unsupported[:200]:
            txt = f["matched_text"].replace("|", "\\|")
            md += f"| `{f['file']}` | {f['line']} | {f['claim_type']} | {txt} |\n"
        md += ("\nEach of these must be fixed at the source that generates the file, "
               "not by editing the artefact.\n")
    else:
        md += "\nNo unsupported claims found in the generated outputs.\n"

    save_report_markdown(md, "claim_consistency_audit.md")
    logger.info("Claim consistency audit: %s (%d unsupported of %d matches across %d files)",
                report["status"], len(unsupported), len(findings), files_scanned)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = run_claim_consistency_audit()
    print(json.dumps({k: v for k, v in res.items() if k != "findings"}, indent=2))
    for f in res["findings"]:
        if f["status"] == "UNSUPPORTED":
            print(f"  UNSUPPORTED {f['file']}:{f['line']} [{f['claim_type']}] {f['matched_text'][:120]}")
