import { Link } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, ErrorBar, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { RiskCards, StormCard } from "@/components/blocks";
import { SeasonBars } from "@/components/charts";
import { CountyMap } from "@/components/CountyMap";
import { ErrorState, HistoricalNote, Loading, Note, PageHeader, Section, Stat } from "@/components/common";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api } from "@/lib/api";
import { ddUnit, n, share, temp, yld } from "@/lib/format";
import { useSettings } from "@/lib/settings";
import { useApi } from "@/lib/useApi";

function droughtWord(v: number | null | undefined) {
  if (v === null || v === undefined) return "–";
  if (v <= -2) return "extremely dry";
  if (v <= -1.5) return "severely dry";
  if (v <= -1) return "moderately dry";
  if (v < 1) return "near normal";
  if (v < 1.5) return "moderately wet";
  return "very wet";
}

export default function ClimateRisk() {
  const { s, set, season } = useSettings();
  const prof = useApi(() => api.profile(s.fips, season, s.method), [s.fips, season, s.method]);
  const ser = useApi(() => api.series(s.fips, s.method), [s.fips, s.method]);
  const rel = useApi(() => api.compoundYield(prof.data?.county.state), [prof.data?.county.state]);
  const lay = useApi(() => api.layer("compound", season, s.method), [season, s.method]);
  const top = useApi(() => api.rankings("cdhw_days", season, s.method, undefined, false, 10), [season, s.method]);
  const p = prof.data;
  const pro = s.role !== "farmer";

  return (
    <>
      <PageHeader title="Climate risk assessment" subtitle="Drought, heat, rainfall, compound drought–heat and storm exposure in the growing season (1 April – 31 October)." />
      <HistoricalNote season={season}>Storm counts and climate indices describe the completed {season} season. They are not weather forecasts.</HistoricalNote>
      {prof.error && <ErrorState message={prof.error} />}
      {!p ? <Loading /> : (
        <>
          <Section title={`${p.county.label}, season ${season}`}><RiskCards p={p} units={s.units} tempUnit={s.temp} simple={!pro} /></Section>

          <div className="grid gap-5 lg:grid-cols-2">
            <Section title="Drought" description="Standardized precipitation (SPI) and precipitation–evapotranspiration (SPEI) indices. Below −1 is dry; below −2 extremely dry.">
              <Table>
                <TableHeader><TableRow><TableHead>Index</TableHead><TableHead className="text-right">Value</TableHead><TableHead>Meaning</TableHead></TableRow></TableHeader>
                <TableBody>
                  {Object.entries(p.risk.drought.indices as Record<string, number | null>).map(([k, v]) => (
                    <TableRow key={k}><TableCell>{k}</TableCell><TableCell className="num text-right">{n(v, 2)}</TableCell><TableCell className="text-muted-foreground">{droughtWord(v)}</TableCell></TableRow>
                  ))}
                  <TableRow><TableCell>Drought days (SPEI-30 &lt; −1)</TableCell><TableCell className="num text-right">{n(p.risk.drought.value, 0)}</TableCell><TableCell className="text-muted-foreground">of 214 season days</TableCell></TableRow>
                  <TableRow><TableCell>Drought severity</TableCell><TableCell className="num text-right">{n(p.risk.drought.severity as number, 1)}</TableCell><TableCell className="text-muted-foreground">SPEI deficit summed over drought days</TableCell></TableRow>
                </TableBody>
              </Table>
              <div className="mt-2"><Note>SPI-1, 3, 6 and 12 use 1-, 3-, 6- and 12-month windows (about 30, 90, 180 and 365 days). Values are the lowest reached during the season unless marked mean.</Note></div>
            </Section>
            <Section title="Heat stress">
              <div className="grid grid-cols-2 gap-3">
                <Stat label="Mean daily maximum" value={temp(p.risk.heat.tmax_mean_c as number, s.temp)} />
                <Stat label="Hottest day" value={temp(p.risk.heat.tmax_max_c as number, s.temp)} />
                <Stat label={`Days above ${s.temp === "C" ? "35 °C" : "95 °F"}`} value={n(p.risk.heat.value, 0)} />
                <Stat label="Killing degree days > 30 °C" value={`${n(s.temp === "C" ? (p.risk.heat.kdd30 as number) : (p.risk.heat.kdd30 as number) * 1.8, 0)}`} sub={ddUnit(s.temp)}
                  hint="Sum of daily maximum temperature above 30 °C (86 °F). Heat above this level damages maize." />
                <Stat label="Heat exceedance > 35 °C" value={`${n(s.temp === "C" ? (p.risk.heat.heat_exceedance as number) : (p.risk.heat.heat_exceedance as number) * 1.8, 1)}`} sub={ddUnit(s.temp)} />
              </div>
              <p className="mt-3 text-sm">Heat by growth stage is in <Link to="/growth" className="text-primary underline">Crop growth analysis</Link>.</p>
            </Section>
          </div>

          <Section title="Compound drought–heat" description="A compound day has maximum temperature above 35 °C while SPEI-30 is below −1. The model groups seasons into Normal (0 days), Moderate (1–2) and Extreme (3 or more).">
            <div className="grid gap-5 xl:grid-cols-2">
              <div className="flex flex-col gap-4">
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <Stat label="Compound days" value={n(p.risk.compound.value, 0)} />
                  <Stat label="Episodes" value={n(p.risk.compound.episodes as number, 0)} />
                  <Stat label="Longest episode" value={`${n(p.risk.compound.longest_episode_days as number, 0)} d`} />
                  <Stat label="Severity" value={n(p.risk.compound.cumulative_severity as number, 1)} hint="Sum over compound days of |SPEI-30| × (Tmax − 35 °C)." />
                </div>
                <p className="text-sm font-semibold">Compound days per season, {p.county.label}</p>
                {ser.data ? <SeasonBars rows={ser.data.rows.map((r) => ({ year: r.year, cdhw: r.cdhw_days }))} keys={[{ key: "cdhw", name: "Compound days", color: "oklch(0.63 0.17 55)" }]} highlight={season} /> : <Loading />}
              </div>
              <div className="flex flex-col gap-2">
                <p className="text-sm font-semibold">Where it happened in {season}</p>
                {lay.data ? <CountyMap layer={lay.data} selected={s.fips} onSelect={(f) => set({ fips: f })} height={360} /> : <Loading />}
              </div>
            </div>
            {pro && rel.data && (
              <div className="mt-6 grid gap-5 lg:grid-cols-2">
                <div>
                  <p className="text-sm font-semibold">Relationship with historical yield{rel.data.state ? `, ${rel.data.state}` : ""}</p>
                  <p className="text-xs text-muted-foreground">Median observed yield vs the county's prior five-season average, with the middle 50% of county-seasons.</p>
                  <ResponsiveContainer width="100%" height={240}>
                    <BarChart data={rel.data.classes.map((c) => ({ ...c, err: [c.median_dev_pct - c.p25_dev_pct, c.p75_dev_pct - c.median_dev_pct] }))} margin={{ top: 12, right: 8, bottom: 0, left: 0 }}>
                      <CartesianGrid vertical={false} stroke="oklch(0.92 0.01 95)" />
                      <XAxis dataKey="class" fontSize={12} tickLine={false} axisLine={false} />
                      <YAxis fontSize={12} tickLine={false} axisLine={false} width={40} unit="%" />
                      <ReferenceLine y={0} stroke="oklch(0.5 0.02 100)" />
                      <Tooltip formatter={(v: number) => [`${n(v, 1)}%`, "Median deviation"]} />
                      <Bar dataKey="median_dev_pct" fill="oklch(0.63 0.17 55)" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                        <ErrorBar dataKey="err" width={6} stroke="oklch(0.3 0.03 120)" />
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div>
                  <Table>
                    <TableHeader><TableRow><TableHead>Compound class</TableHead><TableHead className="text-right">County-seasons</TableHead><TableHead className="text-right">Median vs avg.</TableHead><TableHead className="text-right">&gt;10% below avg.</TableHead></TableRow></TableHeader>
                    <TableBody>{rel.data.classes.map((c) => (
                      <TableRow key={c.class}><TableCell>{c.class}</TableCell><TableCell className="num text-right">{n(c.county_seasons, 0)}</TableCell><TableCell className="num text-right">{n(c.median_dev_pct, 1)}%</TableCell><TableCell className="num text-right">{share(c.share_below_avg_10pct)}</TableCell></TableRow>))}
                    </TableBody>
                  </Table>
                  <div className="mt-2"><Note>{rel.data.note} Seasons {rel.data.seasons.join("–")}.</Note></div>
                </div>
              </div>
            )}
            {pro && top.data && (
              <div className="mt-6">
                <p className="mb-2 text-sm font-semibold">Counties with the most compound days in {season}</p>
                <Table>
                  <TableHeader><TableRow><TableHead>#</TableHead><TableHead>County</TableHead><TableHead className="text-right">Compound days</TableHead><TableHead className="text-right">Drought days</TableHead><TableHead className="text-right">Heat days</TableHead><TableHead className="text-right">Estimate</TableHead></TableRow></TableHeader>
                  <TableBody>{top.data.rows.map((r) => (
                    <TableRow key={r.fips} className="cursor-pointer" onClick={() => set({ fips: r.fips })}><TableCell>{r.rank}</TableCell><TableCell>{r.county}, {r.state}</TableCell>
                      <TableCell className="num text-right">{n(r.cdhw_days, 0)}</TableCell><TableCell className="num text-right">{n(r.drought_days, 0)}</TableCell><TableCell className="num text-right">{n(r.heat_days, 0)}</TableCell><TableCell className="num text-right">{yld(r.estimate_t_ha, s.units)}</TableCell></TableRow>))}
                  </TableBody>
                </Table>
              </div>
            )}
          </Section>

          <Section title="Storm exposure" description="NOAA Storm Events reports during the season."><StormCard p={p} /></Section>
        </>
      )}
    </>
  );
}
