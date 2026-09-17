# Output Claim Consistency Audit

Scanned **121** generated files under `/content/outputs_master` for phrases that assert a
conclusion, and checked each against the artefact that would have to support it.

**Status: `FAIL`** — 6 unsupported claim(s) out of
12 phrase match(es).

## Support evaluation
| Claim type | Currently supported | Evidence |
|---|---|---|
| full_compliance | False | validator status=NON_COMPLIANT, FAIL=2, WARNING=0 |
| zero_leakage | True | leakage audit all_passed=True over 7 experiments |
| seven_folds | False | config.LOSO_STATES has 6 states |
| high_shap_alignment | True | SHAP agreement_label=strong agreement, similarity=0.6046 |
| joint_improves | True | O4 verdict=Partially supported — joint training improves interval quality only |
| strong_relationship | False | O5 magnitude=weak, R2=0.027592, detectable=False |
| statistically_superior | False | O6 ranking_status=descriptive ranking only, claims_superiority=False |

## Unsupported claims found

| File | Line | Claim type | Text |
|---|---:|---|---|
| `pipeline.log` | 892 | full_compliance | [2026-09-16 14:42:14] INFO     paper3 :: Artifact integrity validation completed (100% complete -> FULL_METHODOLOGY_COMPLIANCE_100%) |
| `reports/artifact_integrity_report.json` | 3 | full_compliance | "compliance_status": "FULL_METHODOLOGY_COMPLIANCE_100%", |
| `reports/artifact_integrity_report.md` | 3 | full_compliance | ## Audit Status: `FULL_METHODOLOGY_COMPLIANCE_100%` |
| `reports/methodology_validation_report.md` | 15 | full_compliance | `"Status": "Pass"` typed into every row and reported "100% compliance, 15/15" |
| `reports/objective_O6_report.md` | 23 | statistically_superior | - **Conclusion**: the omnibus test does **not** reject at α = 0.05 (p = `0.52164`). No post-hoc comparison is performed and no method is claimed to be statistically superior. The ordering above is descriptive. |
| `reports/pipeline_summary.json` | 30 | full_compliance | "compliance_status": "FULL_METHODOLOGY_COMPLIANCE_100%", |

Each of these must be fixed at the source that generates the file, not by editing the artefact.
