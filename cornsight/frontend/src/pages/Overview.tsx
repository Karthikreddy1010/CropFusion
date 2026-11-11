import { ArrowRight } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { AlertList, IntervalBar, RiskCards } from "@/components/blocks";
import { CountyMap } from "@/components/CountyMap";
import { ErrorState, HistoricalNote, Loading, PageHeader, RiskBadge, Section, Stat } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api } from "@/lib/api";
import { n, pct, yld, yldRange } from "@/lib/format";
import { useSettings } from "@/lib/settings";
import { useApi } from "@/lib/useApi";

const OVERVIEW_LAYERS = [
  ["estimate", "Model yield estimate"], ["observed", "Observed yield"], ["estimate_anomaly", "Estimate vs 5-season average"],
  ["drought", "Drought days"], ["heat", "Days above 35 °C"], ["compound", "Compound drought–heat"],
] as const;

export default function Overview() {
  const { s, set, season, meta } = useSettings();
  const [layer, setLayer] = useState<string>("estimate");
  const prof = useApi(() => api.profile(s.fips, season, s.method), [s.fips, season, s.method]);
  const states = useApi(() => api.states(season, s.method), [season, s.method]);
  const lay = useApi(() => api.layer(layer, season, s.method), [layer, season, s.method]);
  const p = prof.data;
  const st = states.data?.states.find((x) => x.state === p?.county.state);
  const regionAvg = (() => {
    const xs = states.data?.states.filter((x) => x.estimate_t_ha_acre_weighted !== null && x.harvested_acres) ?? [];
    const w = xs.reduce((a, x) => a + (x.harvested_acres ?? 0), 0);
    return w ? { est: xs.reduce((a, x) => a + (x.estimate_t_ha_acre_weighted ?? 0) * (x.harvested_acres ?? 0), 0) / w,
      obs: xs.reduce((a, x) => a + (x.observed_t_ha_acre_weighted ?? 0) * (x.harvested_acres ?? 0), 0) / w } : null;
  })();
  const fmtLayer = (v: number) => (lay.data?.unit === "t/ha" ? yld(v, s.units) : `${n(v, Math.abs(v) < 20 ? 1 : 0)} ${lay.data?.unit ?? ""}`);

  return (
    <>
      <PageHeader title="Agricultural overview" subtitle={`Maize · six Corn Belt states · season ${season}`} />
      <HistoricalNote season={season} />
      {prof.error && <ErrorState message={prof.error} />}
      {!p && prof.loading && <Loading />}
      {p && (
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-6">
          <Stat label="Predicted yield" value={p.yield.estimate ? yld(p.yield.estimate.t_ha, s.units) : "–"}
            sub={p.yield.estimate ? yldRange(p.yield.estimate.lower_t_ha, p.yield.estimate.upper_t_ha, s.units) : "No estimate"} accent="var(--primary)"
            hint="Out-of-sample model estimate with its 90% prediction interval." />
          <Stat label="Historical average" value={yld(p.yield.prior5_average.t_ha, s.units)} sub="Prior 5 seasons, observed" />
          <Stat label="Yield deviation" value={pct(p.yield.estimate?.vs_average_pct, 1)} sub={`Observed ${pct(p.yield.observed_vs_average_pct, 1)}`} />
          <Stat label="Drought risk" value={<RiskBadge level={p.risk.drought?.level} />} sub={`${n(p.risk.drought?.value, 0)} drought days`} />
          <Stat label="Heat stress" value={<RiskBadge level={p.risk.heat?.level} />} sub={`${n(p.risk.heat?.value, 0)} days > 35 °C`} />
          <Stat label="Compound risk" value={<RiskBadge level={p.risk.compound?.level} />} sub={`${n(p.risk.compound?.value, 0)} compound days`} />
        </div>
      )}

      <div className="grid gap-5 xl:grid-cols-[1fr_22rem]">
        <Section title="Geographic overview" description="Click a county to select it. Colours group counties into six equal-sized classes unless stated."
          actions={
            <Select value={layer} onValueChange={setLayer}>
              <SelectTrigger className="w-[220px]" aria-label="Map layer"><SelectValue /></SelectTrigger>
              <SelectContent>{OVERVIEW_LAYERS.map(([k, l]) => <SelectItem key={k} value={k}>{l}</SelectItem>)}</SelectContent>
            </Select>
          }>
          {lay.error ? <ErrorState message={lay.error} /> : <CountyMap layer={lay.data} selected={s.fips} onSelect={(f) => set({ fips: f })} format={fmtLayer} />}
          {lay.data?.note && <p className="mt-2 text-xs text-muted-foreground">{lay.data.note}</p>}
        </Section>
        {p && (
          <Section title={p.county.label} description={`Season ${season}`} actions={<Button asChild size="sm" variant="ghost"><Link to="/yield">Details <ArrowRight /></Link></Button>}>
            <div className="flex flex-col gap-4 text-sm">
              {p.yield.estimate && <IntervalBar p={p.yield} units={s.units} />}
              <Table>
                <TableHeader><TableRow><TableHead>Yield</TableHead><TableHead className="text-right">Estimate</TableHead><TableHead className="text-right">Observed</TableHead></TableRow></TableHeader>
                <TableBody>
                  <TableRow><TableCell>{p.county.county} County</TableCell><TableCell className="num text-right">{yld(p.yield.estimate?.t_ha, s.units, false)}</TableCell><TableCell className="num text-right">{yld(p.yield.observed.t_ha, s.units, false)}</TableCell></TableRow>
                  <TableRow><TableCell>{meta?.states[p.county.state]} avg.</TableCell><TableCell className="num text-right">{yld(st?.estimate_t_ha_acre_weighted, s.units, false)}</TableCell><TableCell className="num text-right">{yld(st?.observed_t_ha_acre_weighted, s.units, false)}</TableCell></TableRow>
                  <TableRow><TableCell>Six-state region</TableCell><TableCell className="num text-right">{yld(regionAvg?.est, s.units, false)}</TableCell><TableCell className="num text-right">{yld(regionAvg?.obs, s.units, false)}</TableCell></TableRow>
                </TableBody>
              </Table>
              <p className="text-xs text-muted-foreground">Yields in {s.units === "metric" ? "t/ha" : "bu/acre"}. State and region values are means weighted by harvested acres, not official NASS figures.</p>
              <div className="flex flex-col gap-1.5">
                {(["drought", "heat", "compound"] as const).map((k) => (
                  <div key={k} className="flex items-center justify-between"><span className="capitalize">{k === "compound" ? "Compound drought–heat" : k}</span><RiskBadge level={p.risk[k]?.level} /></div>
                ))}
              </div>
            </div>
          </Section>
        )}
      </div>

      {p && (
        <>
          <Section title="Climate risk overview" description={`${p.county.label}, growing season ${season}.`}><RiskCards p={p} units={s.units} tempUnit={s.temp} /></Section>
          <Section title="Alerts"><AlertList alerts={p.alerts} /></Section>
        </>
      )}
    </>
  );
}
