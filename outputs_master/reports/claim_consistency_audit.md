# Output Claim Consistency Audit

Scanned **122** generated files under `/content/outputs_master` for phrases that assert a
conclusion, and checked each against the artefact that would have to support it.

**Status: `PASS`** — 0 unsupported claim(s) out of
7 phrase match(es).

## Support evaluation
| Claim type | Currently supported | Evidence |
|---|---|---|
| full_compliance | True | validator status=COMPLIANT, FAIL=0, WARNING=0 |
| zero_leakage | True | leakage audit all_passed=True over 7 experiments |
| seven_folds | False | config.LOSO_STATES has 6 states |
| high_shap_alignment | True | SHAP agreement_label=strong agreement, similarity=0.6046 |
| joint_improves | True | O4 verdict=Partially supported — joint training improves interval quality only |
| strong_relationship | False | O5 magnitude=weak, R2=0.075887, detectable=False |
| statistically_superior | False | O6 ranking_status=descriptive ranking only, claims_superiority=False |

No unsupported claims found in the generated outputs.
