import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { IntervalBar, UncertaintyText, YieldCards } from "@/components/blocks";
import { MultiLine, STATE_COLORS, YieldHistoryChart } from "@/components/charts";
import { CountyMap } from "@/components/CountyMap";
import { ErrorState, HistoricalNote, Loading, Note, PageHeader, Section, Stat } from "@/components/common";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api } from "@/lib/api";
import { n, pct, production, share, yieldUnit, yieldVal, yld, yldRange } from "@/lib/format";
import { useSettings } from "@/lib/settings";
import { useApi } from "@/lib/useApi";

function CountyView() {
  const { s, season } = useSettings();
  const prof = useApi(() => api.profile(s.fips, season, s.method), [s.fips, season, s.method]);
  const ser = useApi(() => api.series(s.fips, s.method), [s.fips, s.method]);
  const rkState = useApi(() => (prof.data ? api.rankings("estimate_t_ha", season, s.method, prof.data.county.state, false, 200) : Promise.resolve(null)), [prof.data?.county.state, season, s.method]);
  const rkAll = useApi(() => api.rankings("estimate_t_ha", season, s.method, undefined, false, 600), [season, s.method]);
  const p = prof.data;
  if (prof.error) return <ErrorState message={prof.error} />;
  if (!p) return <Loading />;
  const e = p.yield.estimate;
  const posState = rkState.data?.rows.find((r) => r.fips === s.fips);
  const posAll = rkAll.data?.rows.find((r) => r.fips === s.fips);
  const prev = ser.data?.rows.find((r) => r.year === season - 1);
  return (
    <div className="flex flex-col gap-5">
      <YieldCards p={p} units={s.units} />
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Year-over-year (observed)" value={prev?.observed_t_ha && p.yield.observed.t_ha ? pct(100 * (p.yield.observed.t_ha - prev.observed_t_ha) / prev.observed_t_ha, 1) : "–"} sub={`vs ${season - 1}: ${yld(prev?.observed_t_ha, s.units)}`} />
        <Stat label="Change in estimate" value={pct(e?.change_vs_previous_estimate_pct, 1)} sub={e?.previous_season_estimate_t_ha ? `${season - 1} estimate ${yld(e.previous_season_estimate_t_ha, s.units)}` : "No previous estimate"} />
        <Stat label="Rank in state" value={posState ? `${posState.rank} of ${rkState.data?.counties_ranked}` : "–"} sub="by model estimate, highest first" />
        <Stat label="Rank in region" value={posAll ? `${posAll.rank} of ${rkAll.data?.counties_ranked}` : "–"} sub="all six states" />
      </div>
      <div className="grid gap-5 lg:grid-cols-2">
        <Section title="Prediction uncertainty" description="Lower bound, central estimate and upper bound.">
          {e && <IntervalBar p={p.yield} units={s.units} />}
          <div className="mt-4"><UncertaintyText p={p} units={s.units} /></div>
        </Section>
        <Section title="How reliable have past intervals been?" description={`${p.reliability.method_label}, seasons ${p.reliability.seasons.join("–")}.`}>
          <Table>
            <TableHeader><TableRow><TableHead>Season type</TableHead><TableHead className="text-right">County-seasons</TableHead><TableHead className="text-right">Observed inside</TableHead><TableHead className="text-right">Fell below</TableHead></TableRow></TableHeader>
            <TableBody>
              {p.reliability.by_class.map((c) => (
                <TableRow key={c.exposure_class}>
                  <TableCell>{c.exposure_class === "ALL" ? "All seasons" : `${c.exposure_class} compound exposure`}</TableCell>
                  <TableCell className="num text-right">{n(c.rows, 0)}</TableCell>
                  <TableCell className="num text-right">{share(c.inside_rate)}</TableCell>
                  <TableCell className="num text-right">{share(c.below_lower_rate)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          {p.reliability.county && <p className="mt-3 text-sm">In {p.county.county} County the observed yield fell inside the interval in {p.reliability.county.inside} of {p.reliability.county.seasons} seasons, and below it in {p.reliability.county.below_lower}.</p>}
          <div className="mt-2"><Note>Target is 90% inside and about 5% below. {p.reliability.note}</Note></div>
        </Section>
      </div>
      <Section title="Historical versus predicted yield" description="Observed USDA yields since 1985; model estimates and intervals since 2008.">
        {ser.data ? <YieldHistoryChart rows={ser.data.rows} units={s.units} highlight={season} /> : <Loading />}
      </Section>
    </div>
  );
}

function StateView() {
  const { s, season, meta } = useSettings();
  const st = useApi(() => api.states(season, s.method), [season, s.method]);
  const ss = useApi(() => api.stateSeries(s.method), [s.method]);
  const rows = useMemo(() => {
    if (!ss.data) return [];
    const by = new Map<number, Record<string, number | null>>();
    ss.data.rows.forEach((r) => {
      const o = by.get(r.year) ?? { year: r.year };
      o[r.state] = yieldVal(r.observed_t_ha, s.units);
      by.set(r.year, o);
    });
    return [...by.values()].sort((a, b) => (a.year as number) - (b.year as number));
  }, [ss.data, s.units]);
  if (st.error) return <ErrorState message={st.error} />;
  if (!st.data) return <Loading />;
  return (
    <div className="flex flex-col gap-5">
      <Section title={`State comparison, season ${season}`} description={st.data.states[0]?.definition}>
        <div className="overflow-x-auto"><Table>
          <TableHeader><TableRow><TableHead>State</TableHead><TableHead className="text-right">Estimate</TableHead><TableHead className="text-right">90% interval</TableHead><TableHead className="text-right">Observed</TableHead><TableHead className="text-right">Prior 5-season avg.</TableHead><TableHead className="text-right">Counties estimated</TableHead></TableRow></TableHeader>
          <TableBody>{st.data.states.map((x) => (
            <TableRow key={x.state}><TableCell className="font-medium">{x.state_name}</TableCell>
              <TableCell className="num text-right">{yld(x.estimate_t_ha_acre_weighted, s.units)}</TableCell>
              <TableCell className="num text-right">{x.lower_t_ha_acre_weighted !== null ? yldRange(x.lower_t_ha_acre_weighted, x.upper_t_ha_acre_weighted, s.units) : "–"}</TableCell>
              <TableCell className="num text-right">{yld(x.observed_t_ha_acre_weighted, s.units)}</TableCell>
              <TableCell className="num text-right">{yld(x.prior5_avg_t_ha_acre_weighted, s.units)}</TableCell>
              <TableCell className="num text-right">{x.counties_with_estimate} of {x.counties}</TableCell></TableRow>))}
          </TableBody></Table></div>
        <div className="mt-2"><Note>State intervals are acre-weighted means of the county bounds. They are a summary, not a calibrated 90% interval for the state total.</Note></div>
      </Section>
      <Section title="Yield trend by state" description={`Observed yield, acre-weighted, ${meta?.seasons.join("–")}.`}>
        <MultiLine rows={rows} series={Object.entries(meta?.states ?? {}).map(([k, v]) => ({ key: k, name: v, color: STATE_COLORS[k] }))} unitLabel={yieldUnit(s.units)} digits={s.units === "metric" ? 1 : 0} />
      </Section>
    </div>
  );
}

function RegionView() {
  const { s, set, season } = useSettings();
  const st = useApi(() => api.states(season, s.method), [season, s.method]);
  const rk = useApi(() => api.rankings("estimate_t_ha", season, s.method, undefined, false, 600), [season, s.method]);
  const lay = useApi(() => api.layer("estimate_anomaly", season, s.method), [season, s.method]);
  const hist = useMemo(() => {
    const v = (rk.data?.rows ?? []).map((r) => yieldVal(r.value, s.units)).filter((x): x is number => x !== null);
    if (!v.length) return [];
    const w = s.units === "metric" ? 0.5 : 10;
    const lo = Math.floor(Math.min(...v) / w) * w;
    const bins = new Map<number, number>();
    v.forEach((x) => { const b = Math.floor((x - lo) / w); bins.set(b, (bins.get(b) ?? 0) + 1); });
    return [...bins.entries()].sort((a, b) => a[0] - b[0]).map(([b, c]) => ({ bin: `${n(lo + b * w, s.units === "metric" ? 1 : 0)}`, counties: c }));
  }, [rk.data, s.units]);
  const tot = st.data?.states.reduce((a, x) => ({ est: a.est + (x.production_bu_estimated ?? 0), obs: a.obs + (x.production_bu_observed ?? 0), n: a.n + x.counties_with_estimate }), { est: 0, obs: 0, n: 0 });
  return (
    <div className="flex flex-col gap-5">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
        <Stat label="Estimated production" value={production(tot?.est, s.units)} sub={`${tot?.n ?? 0} counties with an estimate`} />
        <Stat label="Reported production" value={production(tot?.obs, s.units)} sub="USDA, all panel counties" />
        <Stat label="Counties ranked" value={n(rk.data?.counties_ranked, 0)} sub={`Season ${season}`} />
      </div>
      <div className="grid gap-5 lg:grid-cols-2">
        <Section title="County-level yield distribution" description={`Model estimates, ${yieldUnit(s.units)}.`}>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={hist} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid vertical={false} stroke="oklch(0.92 0.01 95)" />
              <XAxis dataKey="bin" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis fontSize={12} tickLine={false} axisLine={false} width={36} allowDecimals={false} />
              <Tooltip formatter={(v: number) => [v, "Counties"]} labelFormatter={(l) => `From ${l} ${yieldUnit(s.units)}`} />
              <Bar dataKey="counties" fill="oklch(0.5 0.13 145)" radius={[3, 3, 0, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </Section>
        <Section title="Yield anomaly map" description="Model estimate compared with each county's prior five-season average.">
          {lay.data ? <CountyMap layer={lay.data} selected={s.fips} onSelect={(f) => set({ fips: f })} height={380} format={(v) => pct(v, 1)} /> : <Loading />}
        </Section>
      </div>
    </div>
  );
}

export default function YieldIntel() {
  const { season } = useSettings();
  const [tab, setTab] = useState("county");
  return (
    <>
      <PageHeader title="Crop yield intelligence" subtitle="Expected maize yield, how it compares, and how certain the estimate is." />
      <HistoricalNote season={season} />
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList><TabsTrigger value="county">County</TabsTrigger><TabsTrigger value="state">State</TabsTrigger><TabsTrigger value="region">Region</TabsTrigger></TabsList>
        <TabsContent value="county" className="mt-4"><CountyView /></TabsContent>
        <TabsContent value="state" className="mt-4"><StateView /></TabsContent>
        <TabsContent value="region" className="mt-4"><RegionView /></TabsContent>
      </Tabs>
    </>
  );
}
