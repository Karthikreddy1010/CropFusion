import { ArrowRight } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { CountyMap } from "@/components/CountyMap";
import { ErrorState, Loading, PageHeader, RiskBadge, Section } from "@/components/common";
import { CountyPicker, SeasonPicker, StatePicker } from "@/components/pickers";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { api, type Layer } from "@/lib/api";
import { n, pct, precip, share, yld, yldRange } from "@/lib/format";
import { useSettings } from "@/lib/settings";
import { useApi } from "@/lib/useApi";
import { cn } from "@/lib/utils";

const GROUPS: { title: string; items: [string, string][] }[] = [
  { title: "Yield", items: [["estimate", "Predicted maize yield"], ["observed", "Historical (observed) yield"], ["anomaly", "Observed yield anomaly"], ["estimate_anomaly", "Predicted yield anomaly"], ["width", "Prediction interval width"]] },
  { title: "Climate", items: [["drought", "Drought exposure"], ["heat", "Heat exposure"], ["compound", "Compound drought–heat class"], ["compound_severity", "Compound drought–heat severity"], ["ppt", "Season precipitation"], ["storms", "Historical storm reports"]] },
  { title: "Land", items: [["soil_clay", "Soil clay"], ["soil_oc", "Soil organic carbon"], ["soil_awc", "Available water capacity"], ["cropland", "Cropland share"], ["forest", "Forest share"]] },
];

export default function GeoIntel() {
  const { s, set, season, meta } = useSettings();
  const [layer, setLayer] = useState("estimate");
  const [stateF, setStateF] = useState("");
  const lay = useApi(() => api.layer(layer, season, s.method), [layer, season, s.method]);
  const prof = useApi(() => api.profile(s.fips, season, s.method), [s.fips, season, s.method]);
  const p = prof.data;
  const shown: Layer | null = useMemo(() => {
    if (!lay.data) return null;
    if (!stateF) return lay.data;
    const pre = Object.entries(meta?.states ?? {}).length ? { IL: "17", IN: "18", IA: "19", MN: "27", MO: "29", OH: "39" }[stateF] : "";
    return { ...lay.data, values: Object.fromEntries(Object.entries(lay.data.values).filter(([f]) => f.startsWith(pre ?? ""))) };
  }, [lay.data, stateF, meta]);
  const fmt = (v: number) => (lay.data?.unit === "t/ha" ? yld(v, s.units) : lay.data?.unit === "%" ? (layer.includes("anomaly") ? pct(v, 1) : `${n(v, 0)}%`) : lay.data?.unit === "mm" ? precip(v, s.units) : `${n(v, Math.abs(v) < 20 ? 1 : 0)} ${lay.data?.unit ?? ""}`);

  return (
    <>
      <PageHeader title="Geographic intelligence" subtitle="County maps of yield, climate exposure and land characteristics across the six study states." />
      <div className="grid gap-5 xl:grid-cols-[16rem_1fr_20rem]">
        <Section title="Layers">
          <div className="flex flex-col gap-4">
            {GROUPS.map((g) => (
              <fieldset key={g.title} className="flex flex-col gap-1">
                <legend className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">{g.title}</legend>
                {g.items.map(([k, l]) => (
                  <button key={k} type="button" onClick={() => setLayer(k)} aria-pressed={layer === k}
                    className={cn("rounded-md px-2.5 py-1.5 text-left text-sm hover:bg-accent", layer === k && "bg-accent font-semibold text-accent-foreground")}>{l}</button>
                ))}
              </fieldset>
            ))}
            <div className="flex flex-col gap-1.5"><Label>Season</Label><SeasonPicker className="w-full" /></div>
            <div className="flex flex-col gap-1.5"><Label>State</Label><StatePicker value={stateF} onChange={setStateF} /></div>
            <div className="flex flex-col gap-1.5"><Label htmlFor="map-search">Find county</Label><CountyPicker id="map-search" value={s.fips} onChange={(f) => set({ fips: f })} /></div>
          </div>
        </Section>
        <Section title={lay.data?.label ?? "Map"} description={lay.data ? `${lay.data.seasonal ? `Season ${season}. ` : "Does not change by season. "}${lay.data.counties_with_value} counties with data.` : undefined}>
          {lay.error ? <ErrorState message={lay.error} /> : <CountyMap layer={shown} selected={s.fips} onSelect={(f) => set({ fips: f })} format={fmt} height={620} />}
          {lay.data?.note && <p className="mt-2 text-xs text-muted-foreground">{lay.data.note}</p>}
          {layer === "soil_awc" && <p className="mt-2 text-xs text-muted-foreground">Units of available water capacity are not verified in the source data; compare counties relative to each other only.</p>}
        </Section>
        <Section title="County panel">
          {!p ? <Loading /> : (
            <div className="flex flex-col gap-3 text-sm">
              <p className="text-base font-semibold">{p.county.label}</p>
              <dl className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1.5">
                <dt className="text-muted-foreground">Estimate {season}</dt><dd className="num text-right">{yld(p.yield.estimate?.t_ha, s.units)}</dd>
                <dt className="text-muted-foreground">Interval</dt><dd className="num text-right">{p.yield.estimate ? yldRange(p.yield.estimate.lower_t_ha, p.yield.estimate.upper_t_ha, s.units) : "–"}</dd>
                <dt className="text-muted-foreground">Observed</dt><dd className="num text-right">{yld(p.yield.observed.t_ha, s.units)}</dd>
                <dt className="text-muted-foreground">vs 5-season avg.</dt><dd className="num text-right">{pct(p.yield.observed_vs_average_pct, 1)}</dd>
              </dl>
              <div className="flex flex-col gap-1.5">
                {(["drought", "heat", "rainfall", "compound"] as const).map((k) => (
                  <div key={k} className="flex items-center justify-between"><span className="capitalize">{k === "compound" ? "Compound" : k}</span><RiskBadge level={p.risk[k]?.level} /></div>
                ))}
              </div>
              <dl className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1.5 border-t pt-3">
                <dt className="text-muted-foreground">Cropland</dt><dd className="num text-right">{share(p.county.land_cover.cropland)}</dd>
                <dt className="text-muted-foreground">Clay</dt><dd className="num text-right">{n(p.county.soil.clay.value, 0)} g/kg</dd>
                <dt className="text-muted-foreground">Organic carbon</dt><dd className="num text-right">{n(p.county.soil.organic_carbon.value, 0)} dg/kg</dd>
                <dt className="text-muted-foreground">Elevation</dt><dd className="num text-right">{n(p.county.terrain.elevation_m, 0)} m</dd>
                <dt className="text-muted-foreground">Slope</dt><dd className="num text-right">{n(p.county.terrain.slope_deg, 1)}°</dd>
              </dl>
              <Button asChild size="sm" variant="secondary"><Link to="/yield">Open yield intelligence <ArrowRight /></Link></Button>
            </div>
          )}
        </Section>
      </div>
    </>
  );
}
