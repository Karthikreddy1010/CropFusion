import { StageBars, SeasonBars } from "@/components/charts";
import { ErrorState, HistoricalNote, Loading, PageHeader, Section } from "@/components/common";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api, type Stage } from "@/lib/api";
import { dateShort, n, precip, temp } from "@/lib/format";
import { useSettings } from "@/lib/settings";
import { useApi } from "@/lib/useApi";

export default function CropGrowth() {
  const { s, season } = useSettings();
  const prof = useApi(() => api.profile(s.fips, season, s.method), [s.fips, season, s.method]);
  const hist = useApi(() => api.stageHistory(s.fips), [s.fips]);
  const p = prof.data;
  const st = p?.stages;
  const rows: { label: string; f: (x: Stage) => string }[] = [
    { label: "Modelled dates", f: (x) => x.start ? `${dateShort(x.start)} – ${dateShort(x.end)}` : "Not reached" },
    { label: "Length", f: (x) => x.days !== null ? `${n(x.days, 0)} days` : "–" },
    { label: "Mean daily maximum temperature", f: (x) => temp(x.tmax_mean_c, s.temp) },
    { label: `Days above ${s.temp === "C" ? "35 °C" : "95 °F"}`, f: (x) => n(x.days_above_35c, 0) },
    { label: "Precipitation", f: (x) => precip(x.ppt_mm, s.units) },
    { label: "Reference evapotranspiration (ET₀)", f: (x) => precip(x.et0_mm, s.units) },
    { label: "Water balance (P − ET₀)", f: (x) => precip(x.water_balance_mm, s.units) },
    { label: "Mean vapour-pressure deficit", f: (x) => `${n(x.vpd_kpa, 2)} kPa` },
    { label: "Mean SPEI-30", f: (x) => n(x.spei30_mean, 2) },
    { label: "Lowest SPEI-30", f: (x) => n(x.spei30_min, 2) },
    { label: "Compound drought–heat days", f: (x) => n(x.cdhw_days, 0) },
    { label: "Compound severity", f: (x) => n(x.cdhw_severity, 1) },
  ];
  const most = st?.most_stressed ?? {};
  return (
    <>
      <PageHeader title="Crop growth analysis" subtitle="Conditions in each modelled growth stage of maize." />
      <HistoricalNote season={season} />
      {prof.error && <ErrorState message={prof.error} />}
      {!p || !st ? <Loading /> : (
        <>
          <div className="rounded-md border bg-muted/40 p-3 text-sm"><span className="font-semibold">Modelled stages, not observed dates.</span> {st.note}</div>
          <Section title={`Which stage was most stressed? ${p.county.label}, ${season}`}>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {[["Most compound drought–heat days", most.compound_days, "days"], ["Most days above 35 °C", most.heat_days, "days"],
                ["Driest stage (lowest mean SPEI-30)", most.driest_spei30, ""], ["Highest evaporative demand (VPD)", most.highest_vpd, "kPa"]].map(([l, v, u]) => {
                const item = v as { stage: string; value: number } | null | undefined;
                return (
                  <div key={String(l)} className="rounded-lg border p-4">
                    <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{String(l)}</p>
                    <p className="mt-1 text-lg font-semibold">{item ? item.stage : "None"}</p>
                    <p className="num text-sm text-muted-foreground">{item ? `${n(item.value, u === "days" ? 0 : 2)} ${u}` : "No exposure in any stage"}</p>
                  </div>
                );
              })}
            </div>
          </Section>
          <Section title="Stage-by-stage conditions">
            <div className="overflow-x-auto"><Table>
              <TableHeader><TableRow><TableHead>Indicator</TableHead>{st.stages.map((x) => <TableHead key={x.stage} className="text-right">{x.label}</TableHead>)}</TableRow></TableHeader>
              <TableBody>{rows.map((r) => (
                <TableRow key={r.label}><TableCell>{r.label}</TableCell>{st.stages.map((x) => <TableCell key={x.stage} className="num text-right">{r.f(x)}</TableCell>)}</TableRow>))}
              </TableBody></Table></div>
          </Section>
          <div className="grid gap-5 lg:grid-cols-2">
            <Section title="Heat and compound exposure by stage" description="Days in each stage.">
              <StageBars stages={st.stages} metrics={[{ key: "days_above_35c", name: "Days above 35 °C", color: "oklch(0.63 0.17 55)" }, { key: "cdhw_days", name: "Compound days", color: "oklch(0.55 0.2 28)" }]} />
            </Section>
            <Section title="Water supply and demand by stage" description="Millimetres in each stage.">
              <StageBars stages={st.stages} metrics={[{ key: "ppt_mm", name: "Precipitation", color: "oklch(0.5 0.09 240)" }, { key: "et0_mm", name: "ET₀", color: "oklch(0.6 0.11 60)" }]} />
            </Section>
          </div>
          <Section title="Compound drought–heat days by stage, every season" description="Shows in which stage compound stress has tended to fall in this county.">
            {hist.data ? <SeasonBars rows={hist.data.rows} highlight={season} keys={[
              { key: "veg_cdhw_days", name: "Vegetative", color: "oklch(0.6 0.12 145)" },
              { key: "silk_cdhw_days", name: "Silking", color: "oklch(0.55 0.2 28)" },
              { key: "grain_cdhw_days", name: "Grain filling", color: "oklch(0.62 0.14 60)" }]} height={260} /> : <Loading />}
          </Section>
        </>
      )}
    </>
  );
}
