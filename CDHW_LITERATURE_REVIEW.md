# CDHW Literature Review

Searched via Crossref/scite full-text index, 2015–2026 plus foundational work.
Every entry below was retrieved as a real record — DOI, journal, volume, pages
verified in the returned metadata. Nothing here is recalled from memory.

---

## 1. Compound drought–heatwave definition

| ID | Citation | Method / definition | Relevance |
|---|---|---|---|
| **L1** | Li, Wang & Liu (2021), *Sustainability* 13(22):12774 | CDHE = monthly SPI ≤ −1 co-occurring with percentile-based heatwave | States plainly: **"There is no unified standard for the definition of CDHEs"** |
| **L2** | He, Fang & Xu (2022), *Int. J. Climatology* 42(10):5038–5054 | CDHE identified **using the crop growing season as the accumulation period** | Directly supports a growing-season-accumulated CDHW score — the design this project uses |
| **L3** | Simanjuntak, Gaiser & Ahrends (2023), *Scientific Reports* 13 | Heatwave = ≥3 consecutive days Tmax > 90th percentile of a **31-day centred window**; drought via SPEI; modified Combined Stress Index | The most-cited operational recipe; CSIm explained 25–46% of maize yield variability by province |
| **L4** | Schillerberg & Tian (2024), *Earth's Future* 12(9), e2024EF004845 | Compound heatwave / flash-drought with **location-dependent thresholds** | Supports local rather than absolute thresholds |
| **L5** | Zhang & Zhou (2024), *Environmental Research Letters* 19(10):104073 | Percentile-threshold heatwaves over global croplands | Confirms percentile thresholds are the field standard, and notes impacts **"depend on climate zones, crop types and growth stages"** |
| **L6** | Kamruzzaman (2026), *Int. J. Climatology* 46(9) | EDI ≤ −1.0 + ≥3 days > 90th pct Tmax; CDHW when co-occurring | Recent confirmation of the ≥3-day / 90th-percentile convention |

**Convergent convention.** Drought: a standardized index (SPI / SPEI / PDSI /
EDI) at threshold ≈ −1. Heat: ≥3 consecutive days above the 90th percentile of
daily Tmax, computed against a moving climatological window. Compound: temporal
co-occurrence. Severity: accumulated over the event or the growing season.

## 2. Stage-specific crop sensitivity — the decisive band

| ID | Citation | Finding |
|---|---|---|
| **L7** | Vennam, Poudel & Ramamoorthy (2023), *Physiologia Plantarum* 175(5):e14029 | Soil-moisture stress at flowering cut kernel number **53%** and weight **54%**; **"the silking period (VT–R2) is highly sensitive to moisture stress compared to R2–R5, the grain-filling period"** |
| **L8** | Zhang, Han & Comas (2019), *Agronomy Journal* 111(6):3244–3252 | Growth-stage deficit irrigation; **maturation-stage deficit had a larger yield impact than late-vegetative**; stage interaction present |
| **L9** | Niu, Zhang & Chen (2021), *Agronomy* 11(5):992 | Nocturnal warming: reproductive stages reduced yield 3.4–13.0%, vegetative 1.0% and non-significant. **"Vegetative stages … are less sensitive to high temperatures compared to the reproductive stages"** |
| **L10** | Poudel & Poudel (2023), *InWascon Tech. Mag.* 5:22–25 (review) | **"In maize, reproductive stage is the most susceptible stage"**; >35 °C around flowering cuts grain number up to 45% |
| **L11** | Li, Chen & Tian (2022), *Plants* 11(21):3007 | Deficit at seedling stage ≈ recoverable; deficit at reproductive stages reduces yield materially |

**Convergent finding, five independent sources: the reproductive / silking
window is the most yield-sensitive stage in maize; the vegetative stage is the
least.** This is the single strongest agronomic prior available for this project.

## 3. What the literature does *not* settle

- **No consensus threshold set.** SPI vs SPEI vs PDSI vs EDI, −1 vs −1.5, 90th
  vs 95th percentile — all in current use (L1 says so explicitly). Any specific
  choice is defensible; none is canonical.
- **No consensus on severity aggregation.** Duration, magnitude, count and
  cumulative severity all appear; L1 notes some studies skip events entirely and
  describe CDHEs by metric alone.
- **Compound-event yield attribution is contested.** L3 finds drought dominant
  (25% of variability) with heat secondary (35% in one province only).

## 4. Implication for this project

1. A **growing-season-accumulated** compound severity score is a
   literature-supported construction (L2), not an ad-hoc one.
2. **Stage-resolved exposure is scientifically the right target** (L7–L11), and
   the shipped dataset genuinely provides it (verified empirically — see
   `CDHW_FINAL_AUDIT.md` §2).
3. Because sensitivity is concentrated in the reproductive window, a
   **silking-weighted** representation has the strongest prior — which is why it
   was tested as an explicit arm rather than assumed.
4. The literature gives **no basis** for expecting CDHW features to improve a
   statistical yield model that already contains `Tmax_Days_Above_35`,
   `SPEI_30_min` and GDD. It establishes that the *phenomenon* is real, which is
   a claim about agronomy, not about feature-set marginal value.

## 5. Excluded

- Flash-drought / NDVI work (e.g. *Agriculture* 16(13):1468, 2026) — vegetation
  greenness response, not yield; different target.
- China / Saudi / South Africa regional trend studies — used only for definition
  conventions, not for transferring effect sizes to the US Corn Belt.
