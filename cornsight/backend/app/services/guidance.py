"""Alerts and preparedness guidance derived from a county-season assessment.

All alerts are historical assessments of a completed season in the record. Thresholds marked
"proposed" are platform choices, not model outputs. Guidance is general preparedness awareness that
points to established public services; it never prescribes irrigation, fertilizer or pesticide use.
"""
from __future__ import annotations

RESOURCES = {
    "drought_monitor": {"name": "U.S. Drought Monitor", "url": "https://droughtmonitor.unl.edu"},
    "drought_gov": {"name": "NIDIS Drought.gov", "url": "https://www.drought.gov"},
    "nws": {"name": "National Weather Service forecasts and heat products", "url": "https://www.weather.gov"},
    "rma": {"name": "USDA Risk Management Agency (crop insurance)", "url": "https://www.rma.usda.gov"},
    "climate_hub": {"name": "USDA Midwest Climate Hub", "url": "https://www.climatehubs.usda.gov/hubs/midwest"},
    "extension": {"name": "Your state's land-grant university extension service", "url": None},
}

DECLINE_PCT = -10.0          # proposed
CHANGE_PCT = 15.0            # proposed


def _period(year):
    return f"Season {year}, 1 April – 31 October (historical record)"


def alerts(info: dict, yb: dict, risk: dict, stages: dict) -> list[dict]:
    where = info["label"]
    y = yb["season"]
    out = []
    comp = risk.get("compound", {})
    if comp.get("level") == "High":
        sstage = (stages.get("most_stressed") or {}).get("compound_days")
        out.append({
            "id": "compound", "severity": "High", "title": "Compound drought–heat season",
            "what": f"{int(comp['value'])} days with maximum temperature above 35 °C while SPEI-30 was below −1"
                    + (f", most of them in the {sstage['stage'].lower()} stage." if sstage else "."),
            "where": where, "period": _period(y),
            "impact": "The model groups this season with its 'Extreme' exposure class. Check how often past intervals "
                      "in this class contained the observed yield (Crop yield intelligence, reliability table) before "
                      "relying on the lower bound.",
            "consider": "Treat the lower end of the yield range as a realistic outcome when reviewing insurance "
                        "coverage, contracts and storage plans.",
            "basis": "documented", "threshold": "≥ 3 compound days (model 'Extreme' class)",
        })
    for key, title, what_unit in [("drought", "High drought exposure", "days with SPEI-30 below −1"),
                                  ("heat", "Extreme heat exposure", "days above 35 °C")]:
        b = risk.get(key, {})
        if b.get("level") in ("High", "Very High"):
            out.append({
                "id": key, "severity": b["level"], "title": title,
                "what": f"{int(b['value'])} {what_unit}, more than in {round(100 * (b['percentile'] or 0))}% of this "
                        "county's seasons since 1985.",
                "where": where, "period": _period(y),
                "impact": "Water shortage and heat around silking reduce kernel set; during grain filling they shorten "
                          "the fill period and reduce kernel weight.",
                "consider": "Compare with the stage chart to see when the stress occurred, and check current drought "
                            "status and forecasts before relying on this history.",
                "basis": "proposed", "threshold": "Above the county's 75th percentile (proposed relative scale)",
            })
    e = yb.get("estimate")
    if e and e.get("vs_average_pct") is not None and e["vs_average_pct"] <= DECLINE_PCT:
        out.append({
            "id": "decline", "severity": "High" if e["vs_average_pct"] <= 2 * DECLINE_PCT else "Moderate",
            "title": "Potential production decline",
            "what": f"Model estimate {e['t_ha']:.1f} t/ha is {abs(e['vs_average_pct']):.0f}% below the five-season "
                    f"average ({yb['prior5_average']['t_ha']:.1f} t/ha). Interval {e['lower_t_ha']:.1f}–{e['upper_t_ha']:.1f} t/ha.",
            "where": where, "period": _period(y),
            "impact": "Lower expected production for the county's maize area.",
            "consider": "Use the full interval, not only the central estimate, when planning volumes.",
            "basis": "proposed", "threshold": f"Estimate ≥ {abs(DECLINE_PCT):.0f}% below the prior five-season average",
        })
    if e and e.get("change_vs_previous_estimate_pct") is not None and abs(e["change_vs_previous_estimate_pct"]) >= CHANGE_PCT:
        ch = e["change_vs_previous_estimate_pct"]
        out.append({
            "id": "change", "severity": "Moderate", "title": "Large change in yield estimate",
            "what": f"The estimate is {abs(ch):.0f}% {'lower' if ch < 0 else 'higher'} than the previous season's estimate "
                    f"({e['previous_season_estimate_t_ha']:.1f} t/ha).",
            "where": where, "period": f"Seasons {y-1} and {y} (historical record)",
            "impact": "Season-to-season swings of this size change regional supply and revenue expectations.",
            "consider": "Check which conditions differ between the two seasons in the climate risk view.",
            "basis": "proposed", "threshold": f"Change of {CHANGE_PCT:.0f}% or more between consecutive estimates",
        })
    return out


def recommendations(yb: dict, risk: dict, stages: dict) -> list[dict]:
    """General preparedness topics, triggered by the assessment. Not prescriptions."""
    recs = []
    lvl = lambda k: (risk.get(k) or {}).get("level")
    if lvl("drought") in ("Moderate", "High", "Very High") or lvl("compound") in ("Moderate", "High"):
        recs.append({
            "topic": "Drought preparedness",
            "why": f"Drought exposure was {lvl('drought') or 'not available'} and compound drought–heat was "
                   f"{lvl('compound') or 'not available'} in this season.",
            "consider": ["Follow the weekly U.S. Drought Monitor map for your county during the season.",
                         "Review how much of your production is protected if yield falls to the lower end of the range.",
                         "Talk to your extension agronomist about drought-tolerant hybrid options for fields with low water-holding soils."],
            "sources": ["drought_monitor", "drought_gov", "rma", "extension"],
        })
    if lvl("heat") in ("High", "Very High") or ((stages.get("most_stressed") or {}).get("heat_days")):
        recs.append({
            "topic": "Heat-stress preparedness",
            "why": f"Heat exposure was {lvl('heat') or 'not available'}; heat around silking is the most yield-sensitive period for maize.",
            "consider": ["Watch National Weather Service heat outlooks in the weeks around expected silking.",
                         "Note which fields reach silking during typical July heat when planning planting windows and hybrid maturity with your agronomist."],
            "sources": ["nws", "climate_hub", "extension"],
        })
    if lvl("rainfall") in ("High", "Very High"):
        recs.append({
            "topic": "Wet-season awareness",
            "why": "Season precipitation was well above this county's usual amount.",
            "consider": ["Keep records of flooded or ponded areas for insurance and field planning.",
                         "Ask your extension service about field access and harvest-timing issues in wet seasons."],
            "sources": ["rma", "extension"],
        })
    e = yb.get("estimate")
    if e:
        recs.append({
            "topic": "Production-risk awareness",
            "why": f"The 90% prediction interval spans {e['width_t_ha']:.1f} t/ha.",
            "consider": ["Plan marketing and storage around the range, not a single number.",
                         "Compare the lower bound with your insurance guarantee."],
            "sources": ["rma"],
        })
    recs.append({
        "topic": "Weather preparedness",
        "why": "This platform describes completed seasons. It has no current weather forecast yet.",
        "consider": ["Use National Weather Service forecasts for in-season decisions."],
        "sources": ["nws"],
    })
    for r in recs:
        r["sources"] = [RESOURCES[k] for k in r["sources"]]
        r["note"] = "General guidance for awareness, not a field prescription."
    return recs
