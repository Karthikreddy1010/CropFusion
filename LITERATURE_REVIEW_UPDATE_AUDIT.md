# Literature Review Update Audit

**Format note, stated honestly rather than fabricating structure that isn't
there:** `Literature_Review_Paper3_ConformalInference.xlsx` is a flat table
of 36 papers (Title/Year/Gap/Methods/Findings/Dataset/Category columns) — it
has no narrative "subsections" for a KEEP/UPDATE/REWRITE/ADD/REMOVE audit to
act on directly. That kind of subsection structure belongs to the paper's
own **Related Work / Literature Review section** (in `Paper3_Methodology_Updated_v2.md`
or the eventual manuscript), which *is* organized thematically. This audit
therefore maps the requested action onto **thematic paper groupings**
(which of the 36 reviewed papers belong together when the Related Work
section is drafted), not onto the spreadsheet's rows directly.

| Thematic Group (papers) | Action | Reason |
|---|---|---|
| Conformal prediction foundations (rows 1, 2, 3, 6, 7, 8, 10) | **KEEP** | Correctly attributed already (§1 of the methodology explicitly credits Romano et al., Gibbs & Candès); no code or claim changes affect this group |
| Static conformal failure in agriculture (row 18) | **KEEP, but strengthen the discussion** | This is the paper's central motivating citation. The real result (PICP degrades to 0.82 in extremes, not fully restored by ACI) should be discussed *relative to* Paper 18's 73% finding — is 0.82 a meaningful improvement over Paper 18's 0.73, or not? This comparison is not currently drawn out in the methodology text and should be, since it's the most direct evidence for the paper's central claim |
| Shift-robust conformal alternatives (rows 4, 5, 9, 17, 23) | **KEEP** | All four are genuinely implemented and compared (O6) — no change needed, but see the note below about not overclaiming until the Round-3 Friedman fix is re-verified |
| Phenology-aware calibration (row 21) | **UPDATE** | The current framing ("this paper extends Paper 21 by adding online recalibration") is directionally fine, but the real result (phenology-stratified CQR ties/slightly underperforms static conformal in this run, `Silking_R1` n=35) should be added — a reviewer familiar with Paper 21 will ask why the extension doesn't clearly outperform the base method it extends |
| Compound drought-heat magnitude studies (rows 13, 29, 32, 33, 37, 38) | **REWRITE the framing, not the citations** | These papers are correctly cited as *motivation* for constructing CDHW, but the methodology currently implies this paper's own results will be of a similar character (compound stress worsens outcomes). The real ablation result (CDHW reduces this backbone's R²) means the Related Work framing must shift from "we show CDHW matters, consistent with Papers 29/32" to "we test whether CDHW improves a specific predictive backbone; unlike the crop-model literature's magnitude estimates, this is a narrower predictive-contribution question, not a mechanistic damage estimate" — do not let citations to Papers 29/32's specific multipliers (2.3×, ~50%) imply this paper reproduces them |
| Drought-timing / phenology sensitivity meta-analysis (row 24) | **KEEP** | Correctly used to justify the silking-stage severity weighting; unaffected by any code change |
| ENSO/climate mediation (row 22) | **KEEP** | Used for year-type/ENSO stratification; unaffected |
| Spatial conformal / geographic shift (row 26) | **KEEP, with a scope caveat already added** | The six-state LOSO section now correctly cites this as motivation while explicitly scoping the claim to six Corn Belt states, not nationally — already fixed in the methodology update |
| Extreme value theory / tail risk (row 27) | **ADD a discussion, not a new citation** | This paper is already in the review but is not discussed anywhere in the methodology's actual text. Since this audit confirmed no EVT/tail-risk analysis is implemented, add an explicit sentence acknowledging Paper 27's approach as a natural extension not pursued here, rather than leaving it uncited in the write-up despite being in the review spreadsheet |
| ML-in-agriculture survey (row 30) | **KEEP, but note its role** | This is the most likely source of the original PhenoFormer/Transformer confusion (its finding that "Transformers outperform LSTM on long-horizon predictions" is a field-wide trend, not a statement about this paper's own architecture) — already noted in `AUDIT_REPORT.md` Part 2. No spreadsheet change needed, just don't let this general survey finding bleed into this paper's own architecture description again |
| Quantile regression forests / classical QR (rows 6, 16) | **KEEP** | Used correctly as the "conventional prediction interval" comparison point in the literature framing; the codebase's own tree-based quantile models (LightGBM/CatBoost/XGBoost) are the closest actual implementation |
| Extreme weather/agricultural risk insurance framing (rows 14, 15, 20) | **KEEP** | General motivation for probabilistic yield forecasting; unaffected |

## What NOT to do

- **Do not add new papers to make a gap look more closed than the code
  actually demonstrates.** Every row in `LITERATURE_GAP_COVERAGE_TABLE.md`
  marked "NOT ADDRESSED" or "PARTIALLY ADDRESSED" should stay that way in
  the write-up regardless of how the literature section is worded — wording
  changes cannot substitute for missing experiments.
- **Do not remove citations to Papers 29/32/etc. just because this paper's
  own CDHW ablation came back negative.** Those papers remain valid,
  correctly-cited motivation for *why CDHW is a scientifically reasonable
  thing to try* — the negative result is about this paper's specific
  backbone/architecture, not a refutation of the compound-stress literature
  itself. Rewrite the *connecting sentence*, not the citation list.
- **Do not use this document to make the code appear to address a gap
  it does not.** The order is, and must remain: literature → genuine gap →
  methodological requirement → implementation → experiment → evidence →
  claim. Every "UPDATE"/"REWRITE" above is a *writing* fix triggered by a
  *real, already-measured* result, not a preemptive rewrite designed to
  make a future result look better.
