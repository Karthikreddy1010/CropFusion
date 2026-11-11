import { Download, Plus, X } from "lucide-react";
import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { MultiLine, STATE_COLORS } from "@/components/charts";
import { CountyMap } from "@/components/CountyMap";
import { ErrorState, HistoricalNote, Loading, Note, PageHeader, RiskBadge, Section, Stat } from "@/components/common";
import { CountyPicker, StatePicker } from "@/components/pickers";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api, reportUrl, type Level } from "@/lib/api";
import { big, n, pct, production, share, T_PER_BU, yieldUnit, yieldVal, yld, yldRange } from "@/lib/format";
import { useSettings } from "@/lib/settings";
import { useApi } from "@/lib/useApi";

/* ---------------------------------------------------------------- Regional production */
export function RegionalProduction() {
  const { s, season, meta } = useSettings();
  const st = useApi(() => api.states(season, s.method), [season, s.method]);
  const ss = useApi(() => api.stateSeries(s.method), [s.method]);
  const tot = st.data?.states.reduce((a, x) => ({ est: a.est + (x.production_bu_estimated ?? 0), obs: a.obs + (x.production_bu_observed ?? 0), ac: a.ac + (x.harvested_acres ?? 0) }), { est: 0, obs: 0, ac: 0 });
  const prodRows = useMemo(() => {
    if (!ss.data) return [];
    const by = new Map<number, Record<string, number | null>>();
    ss.data.rows.forEach((r) => { const o = by.get(r.year) ?? { year: r.year }; o[r.state] = r.production_bu === null ? null : (s.units === "metric" ? r.production_bu * T_PER_BU : r.production_bu) / 1e6; by.set(r.year, o); });
    return [...by.values()].sort((a, b) => (a.year as number) - (b.year as number));
  }, [ss.data, s.units]);
  const bars = st.data?.states.map((x) => ({
    state: x.state, estimated: x.production_bu_estimated === null ? null : (s.units === "metric" ? x.production_bu_estimated * T_PER_BU : x.production_bu_estimated) / 1e6,
    reported: x.production_bu_observed === null ? null : (s.units === "metric" ? x.production_bu_observed * T_PER_BU : x.production_bu_observed) / 1e6,
  })) ?? [];
  const unitM = s.units === "metric" ? "million t" : "million bu";
  return (
    <>
      <PageHeader title="Regional production" subtitle={`State-level maize production, season ${season}.`} />
      <HistoricalNote season={season} />
      {st.error && <ErrorState message={st.error} />}
      {!st.data ? <Loading /> : (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Stat label="Estimated production" value={production(tot?.est, s.units)} sub="counties with a model estimate" />
            <Stat label="Reported production" value={production(tot?.obs, s.units)} sub="USDA, all panel counties" />
            <Stat label="Harvested area" value={`${big(tot?.ac)} acres`} />
            <Stat label="Counties, high compound exposure" value={n(st.data.states.reduce((a, x) => a + x.counties_compound_high, 0), 0)} sub="3 or more compound days" />
          </div>
          <Section title="Production by state" description={`${unitM}. Estimated = model yield × USDA harvested acres for counties with an estimate; reported = USDA county production summed.`}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={bars} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid vertical={false} stroke="oklch(0.92 0.01 95)" />
                <XAxis dataKey="state" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis fontSize={12} tickLine={false} axisLine={false} width={48} />
                <Tooltip formatter={(v: number, nm: string) => [`${n(v, 0)} ${unitM}`, nm]} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="estimated" name="Estimated" fill="oklch(0.5 0.13 145)" radius={[3, 3, 0, 0]} isAnimationActive={false} />
                <Bar dataKey="reported" name="Reported (USDA)" fill="oklch(0.6 0.11 60)" radius={[3, 3, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
            <Note>Estimated production covers fewer counties than reported production when some counties have no estimate; compare within the same county set before drawing conclusions.</Note>
          </Section>
          <Section title="State detail" actions={<Button asChild variant="secondary" size="sm"><a href={reportUrl.states(season, s.method)} download><Download />CSV</a></Button>}>
            <div className="overflow-x-auto"><Table>
              <TableHeader><TableRow><TableHead>State</TableHead><TableHead className="text-right">Estimate</TableHead><TableHead className="text-right">Interval (acre-wtd.)</TableHead><TableHead className="text-right">vs 5-season avg.</TableHead><TableHead className="text-right">Est. production</TableHead><TableHead className="text-right">Reported</TableHead><TableHead className="text-right">High / moderate compound counties</TableHead></TableRow></TableHeader>
              <TableBody>{st.data.states.map((x) => (
                <TableRow key={x.state}><TableCell className="font-medium">{x.state_name}</TableCell>
                  <TableCell className="num text-right">{yld(x.estimate_t_ha_acre_weighted, s.units)}</TableCell>
                  <TableCell className="num text-right">{x.lower_t_ha_acre_weighted !== null ? yldRange(x.lower_t_ha_acre_weighted, x.upper_t_ha_acre_weighted, s.units) : "–"}</TableCell>
                  <TableCell className="num text-right">{x.estimate_t_ha_acre_weighted && x.prior5_avg_t_ha_acre_weighted ? pct(100 * (x.estimate_t_ha_acre_weighted - x.prior5_avg_t_ha_acre_weighted) / x.prior5_avg_t_ha_acre_weighted, 1) : "–"}</TableCell>
                  <TableCell className="num text-right">{production(x.production_bu_estimated, s.units)}</TableCell>
                  <TableCell className="num text-right">{production(x.production_bu_observed, s.units)}</TableCell>
                  <TableCell className="num text-right">{x.counties_compound_high} / {x.counties_compound_moderate}</TableCell></TableRow>))}
              </TableBody></Table></div>
          </Section>
          <Section title="Historical production trend" description={`Reported production by state, ${unitM}.`}>
            <MultiLine rows={prodRows} series={Object.entries(meta?.states ?? {}).map(([k, v]) => ({ key: k, name: v, color: STATE_COLORS[k] }))} unitLabel={unitM} digits={0} />
          </Section>
        </>
      )}
    </>
  );
}

/* ---------------------------------------------------------------- Supply chain */
export function SupplyChain() {
  const { s, set, season } = useSettings();
  const all = useApi(() => api.rankings("production_bu_estimated", season, s.method, undefined, false, 600), [season, s.method]);
  const lay = useApi(() => api.layer("estimate_anomaly", season, s.method), [season, s.method]);
  const conc = useMemo(() => {
    const rows = all.data?.rows ?? [];
    const total = rows.reduce((a, r) => a + (r.value ?? 0), 0);
    const part = (f: (r: (typeof rows)[number]) => boolean) => { const x = rows.filter(f); return { n: x.length, share: total ? x.reduce((a, r) => a + (r.value ?? 0), 0) / total : null }; };
    const top = rows.slice(0, Math.ceil(rows.length * 0.2));
    return {
      total, counties: rows.length,
      top20: total ? top.reduce((a, r) => a + (r.value ?? 0), 0) / total : null,
      compound: part((r) => (r.cdhw_days ?? 0) >= 3), drought: part((r) => (r.drought_days ?? 0) >= 30),
      heat: part((r) => (r.heat_days ?? 0) >= 5),
      exposedTop: rows.filter((r) => (r.cdhw_days ?? 0) >= 3).slice(0, 12),
    };
  }, [all.data]);
  return (
    <>
      <PageHeader title="Supply chain intelligence" subtitle={`Where production and climate exposure overlap, season ${season}.`} />
      <HistoricalNote season={season}>Shares are of estimated production in counties with a model estimate. The platform does not convert yields into prices or grain availability, which needs inventory and market data.</HistoricalNote>
      {all.error && <ErrorState message={all.error} />}
      {!all.data ? <Loading /> : (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Stat label="Top 20% of counties" value={share(conc.top20)} sub="share of estimated production" />
            <Stat label="In high compound exposure" value={share(conc.compound.share)} sub={`${conc.compound.n} counties with 3+ compound days (model class)`} accent="var(--risk-high)" />
            <Stat label="In 30+ drought days" value={share(conc.drought.share)} sub={`${conc.drought.n} counties · proposed cut-off`} accent="var(--risk-moderate)" />
            <Stat label="In 5+ days above 35 °C" value={share(conc.heat.share)} sub={`${conc.heat.n} counties · proposed cut-off`} accent="var(--risk-moderate)" />
          </div>
          <div className="grid gap-5 xl:grid-cols-[1fr_26rem]">
            <Section title="Regions with potential yield reductions" description="Model estimate vs each county's prior five-season average.">
              {lay.data ? <CountyMap layer={lay.data} selected={s.fips} onSelect={(f) => set({ fips: f })} format={(v) => pct(v, 1)} height={480} /> : <Loading />}
            </Section>
            <Section title="Largest producers with high compound exposure">
              {conc.exposedTop.length === 0 ? <p className="text-sm text-muted-foreground">No county had 3 or more compound days in {season}.</p> : (
                <Table>
                  <TableHeader><TableRow><TableHead>County</TableHead><TableHead className="text-right">Est. production</TableHead><TableHead className="text-right">Compound days</TableHead></TableRow></TableHeader>
                  <TableBody>{conc.exposedTop.map((r) => (
                    <TableRow key={r.fips} className="cursor-pointer" onClick={() => set({ fips: r.fips })}><TableCell>{r.county}, {r.state}</TableCell><TableCell className="num text-right">{production(r.value, s.units)}</TableCell><TableCell className="num text-right">{n(r.cdhw_days, 0)}</TableCell></TableRow>))}
                  </TableBody>
                </Table>
              )}
            </Section>
          </div>
          <Note>Drought (30+ days with SPEI-30 below −1) and heat (5+ days above 35 °C) cut-offs on this page are proposed screening thresholds, not model classes.</Note>
        </>
      )}
    </>
  );
}

/* ---------------------------------------------------------------- County comparison */
export function CountyCompare() {
  const { s, season, countyName } = useSettings();
  const [list, setList] = useState<string[]>(() => [s.fips]);
  const cmp = useApi(() => (list.length ? api.compare(list, season, s.method) : Promise.resolve(null)), [list.join(","), season, s.method]);
  const colors = ["oklch(0.5 0.13 145)", "oklch(0.62 0.15 60)", "oklch(0.5 0.12 250)", "oklch(0.55 0.16 320)", "oklch(0.6 0.1 200)", "oklch(0.45 0.05 50)"];
  const rows = useMemo(() => {
    if (!cmp.data) return [];
    const by = new Map<number, Record<string, number | null>>();
    cmp.data.counties.forEach((c) => c.series.forEach((r) => { const o = by.get(r.year) ?? { year: r.year }; o[c.county.fips] = yieldVal(r.observed_t_ha, s.units); by.set(r.year, o); }));
    return [...by.values()].sort((a, b) => (a.year as number) - (b.year as number));
  }, [cmp.data, s.units]);
  return (
    <>
      <PageHeader title="County comparison" subtitle={`Up to six counties side by side, season ${season}.`} />
      <Section title="Counties">
        <div className="flex flex-wrap items-center gap-2">
          {list.map((f) => (
            <span key={f} className="inline-flex items-center gap-1 rounded-full border bg-card px-3 py-1 text-sm">{countyName(f)}
              <button type="button" aria-label={`Remove ${countyName(f)}`} onClick={() => setList(list.filter((x) => x !== f))}><X className="size-3.5" /></button></span>
          ))}
          {list.length < 6 && <CountyPicker id="cmp-add" className="w-64" placeholder="Add a county" onChange={(f) => !list.includes(f) && setList([...list, f])} />}
        </div>
      </Section>
      {cmp.error && <ErrorState message={cmp.error} />}
      {cmp.data && (
        <>
          <Section title="Side by side">
            <div className="overflow-x-auto"><Table>
              <TableHeader><TableRow><TableHead>County</TableHead><TableHead className="text-right">Estimate</TableHead><TableHead className="text-right">90% interval</TableHead><TableHead className="text-right">Observed</TableHead><TableHead className="text-right">vs 5-season avg.</TableHead><TableHead>Drought</TableHead><TableHead>Heat</TableHead><TableHead>Compound</TableHead></TableRow></TableHeader>
              <TableBody>{cmp.data.counties.map((c) => (
                <TableRow key={c.county.fips}><TableCell className="font-medium">{c.county.label}</TableCell>
                  <TableCell className="num text-right">{yld(c.yield.estimate?.t_ha, s.units)}</TableCell>
                  <TableCell className="num text-right">{c.yield.estimate ? yldRange(c.yield.estimate.lower_t_ha, c.yield.estimate.upper_t_ha, s.units) : "–"}</TableCell>
                  <TableCell className="num text-right">{yld(c.yield.observed.t_ha, s.units)}</TableCell>
                  <TableCell className="num text-right">{pct(c.yield.estimate?.vs_average_pct, 1)}</TableCell>
                  {(["drought", "heat", "compound"] as const).map((k) => <TableCell key={k}><RiskBadge level={c.risk[k]?.level as Level} /></TableCell>)}</TableRow>))}
              </TableBody></Table></div>
          </Section>
          <Section title="Observed yield history"><MultiLine rows={rows} series={cmp.data.counties.map((c, i) => ({ key: c.county.fips, name: c.county.label, color: colors[i] }))} unitLabel={yieldUnit(s.units)} digits={s.units === "metric" ? 1 : 0} /></Section>
        </>
      )}
    </>
  );
}

/* ---------------------------------------------------------------- Production risk ranking */
export function ProductionRisk() {
  const { s, set, season } = useSettings();
  const [metric, setMetric] = useState("estimate_vs_average_pct");
  const [st, setSt] = useState("");
  const mets = useApi(() => api.rankMetrics(), []);
  const asc = metric === "estimate_vs_average_pct" || metric === "observed_vs_average_pct" || metric === "estimate_t_ha";
  const rk = useApi(() => api.rankings(metric, season, s.method, st || undefined, asc, 50), [metric, season, s.method, st]);
  const fmt = (v: number | null) => {
    const u = rk.data?.unit;
    if (u === "t/ha") return yld(v, s.units);
    if (u === "%") return pct(v, 1);
    if (u === "bu") return production(v, s.units);
    return `${n(v, u === "°C·day" ? 0 : 0)} ${u ?? ""}`;
  };
  return (
    <>
      <PageHeader title="Production risk analysis" subtitle="Rank counties by one transparent metric at a time." />
      <HistoricalNote season={season} />
      <Section title="Ranking" actions={
        <div className="flex flex-wrap gap-2">
          <Select value={metric} onValueChange={setMetric}>
            <SelectTrigger className="w-[260px]" aria-label="Metric"><SelectValue /></SelectTrigger>
            <SelectContent>{Object.entries(mets.data ?? {}).map(([k, v]) => <SelectItem key={k} value={k}>{v.label}</SelectItem>)}</SelectContent>
          </Select>
          <StatePicker value={st} onChange={setSt} />
          <Button asChild variant="secondary"><a href={reportUrl.rankings(metric, season, s.method, st || undefined)} download><Download />CSV</a></Button>
        </div>}>
        {rk.error && <ErrorState message={rk.error} />}
        {!rk.data ? <Loading /> : (
          <>
            <p className="mb-2 text-sm text-muted-foreground">{rk.data.label}, {asc ? "lowest first (largest decline or lowest yield)" : "highest first"}. {rk.data.counties_ranked} counties have a value.</p>
            <div className="overflow-x-auto"><Table>
              <TableHeader><TableRow><TableHead>#</TableHead><TableHead>County</TableHead><TableHead className="text-right">{rk.data.label}</TableHead><TableHead className="text-right">Estimate</TableHead><TableHead className="text-right">Interval</TableHead><TableHead className="text-right">Compound days</TableHead><TableHead className="text-right">Drought days</TableHead><TableHead className="text-right">Heat days</TableHead></TableRow></TableHeader>
              <TableBody>{rk.data.rows.map((r) => (
                <TableRow key={r.fips} className="cursor-pointer" onClick={() => set({ fips: r.fips })}><TableCell>{r.rank}</TableCell><TableCell>{r.county}, {r.state}</TableCell>
                  <TableCell className="num text-right font-medium">{fmt(r.value)}</TableCell><TableCell className="num text-right">{yld(r.estimate_t_ha, s.units)}</TableCell>
                  <TableCell className="num text-right">{r.lower_t_ha !== null ? yldRange(r.lower_t_ha, r.upper_t_ha, s.units) : "–"}</TableCell>
                  <TableCell className="num text-right">{n(r.cdhw_days, 0)}</TableCell><TableCell className="num text-right">{n(r.drought_days, 0)}</TableCell><TableCell className="num text-right">{n(r.heat_days, 0)}</TableCell></TableRow>))}
              </TableBody></Table></div>
          </>
        )}
      </Section>
      <Note>The platform does not combine metrics into a single risk score, because no validated weighting exists. Rank by the metric that matches your decision.</Note>
    </>
  );
}

/* ---------------------------------------------------------------- Cooperative portfolio */
export function Portfolio() {
  const { s, set, season, countyName } = useSettings();
  const list = s.watchlist.length ? s.watchlist : [];
  const chunks = useMemo(() => { const c: string[][] = []; for (let i = 0; i < list.length; i += 6) c.push(list.slice(i, i + 6)); return c; }, [list]);
  const data = useApi(() => Promise.all(chunks.map((c) => api.compare(c, season, s.method))).then((r) => r.flatMap((x) => x.counties)), [chunks.map((c) => c.join(",")).join("|"), season, s.method]);
  const totals = useMemo(() => {
    const d = data.data ?? [];
    return {
      prod: d.reduce((a, c) => a + (c.yield.estimate?.production_bu ?? 0), 0),
      low: d.reduce((a, c) => a + (c.yield.estimate?.production_bu_lower ?? 0), 0),
      high: d.filter((c) => c.risk.compound?.level === "High").length,
      below: d.filter((c) => (c.yield.estimate?.vs_average_pct ?? 0) <= -10).length,
    };
  }, [data.data]);
  return (
    <>
      <PageHeader title="Portfolio" subtitle="Counties your organisation monitors. Saved in this browser." />
      <HistoricalNote season={season} />
      <Section title="Monitored counties">
        <div className="flex flex-wrap items-center gap-2">
          {list.map((f) => (
            <span key={f} className="inline-flex items-center gap-1 rounded-full border bg-card px-3 py-1 text-sm">{countyName(f)}
              <button type="button" aria-label={`Remove ${countyName(f)}`} onClick={() => set({ watchlist: list.filter((x) => x !== f) })}><X className="size-3.5" /></button></span>
          ))}
          <CountyPicker id="pf-add" className="w-64" placeholder="Add a county" onChange={(f) => !list.includes(f) && set({ watchlist: [...list, f] })} />
          {!list.includes(s.fips) && <Button variant="ghost" size="sm" onClick={() => set({ watchlist: [...list, s.fips] })}><Plus />Add {countyName(s.fips)}</Button>}
        </div>
      </Section>
      {list.length === 0 ? (
        <p className="text-sm text-muted-foreground">Add the counties your members farm in. The portfolio then summarises their estimated yield, production range and climate exposure for the selected season.</p>
      ) : data.error ? <ErrorState message={data.error} /> : !data.data ? <Loading /> : (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Stat label="Estimated production" value={production(totals.prod, s.units)} sub={`Lower-bound total ${production(totals.low, s.units)}`} hint="Sum of county estimates × USDA harvested acres. The lower-bound total adds county lower bounds, which is a cautious scenario, not a 90% bound for the total." />
            <Stat label="Counties monitored" value={n(list.length, 0)} />
            <Stat label="High compound exposure" value={n(totals.high, 0)} accent="var(--risk-high)" />
            <Stat label="Estimate 10%+ below average" value={n(totals.below, 0)} accent="var(--risk-moderate)" sub="proposed screening cut-off" />
          </div>
          <Section title="Risk overview">
            <div className="overflow-x-auto"><Table>
              <TableHeader><TableRow><TableHead>County</TableHead><TableHead className="text-right">Estimate</TableHead><TableHead className="text-right">Interval</TableHead><TableHead className="text-right">vs avg.</TableHead><TableHead className="text-right">Est. production</TableHead><TableHead>Drought</TableHead><TableHead>Heat</TableHead><TableHead>Compound</TableHead></TableRow></TableHeader>
              <TableBody>{[...data.data].sort((a, b) => (a.yield.estimate?.vs_average_pct ?? 0) - (b.yield.estimate?.vs_average_pct ?? 0)).map((c) => (
                <TableRow key={c.county.fips} className="cursor-pointer" onClick={() => set({ fips: c.county.fips })}><TableCell className="font-medium">{c.county.label}</TableCell>
                  <TableCell className="num text-right">{yld(c.yield.estimate?.t_ha, s.units)}</TableCell>
                  <TableCell className="num text-right">{c.yield.estimate ? yldRange(c.yield.estimate.lower_t_ha, c.yield.estimate.upper_t_ha, s.units) : "–"}</TableCell>
                  <TableCell className="num text-right">{pct(c.yield.estimate?.vs_average_pct, 1)}</TableCell>
                  <TableCell className="num text-right">{production(c.yield.estimate?.production_bu, s.units)}</TableCell>
                  {(["drought", "heat", "compound"] as const).map((k) => <TableCell key={k}><RiskBadge level={c.risk[k]?.level as Level} /></TableCell>)}</TableRow>))}
              </TableBody></Table></div>
          </Section>
          <Section title="Notifications"><p className="text-sm text-muted-foreground">Email and mobile notifications are a planned integration. Alerts for each county are on its overview and recommendations pages.</p></Section>
        </>
      )}
    </>
  );
}
