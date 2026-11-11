import { useMemo } from "react";
import { MultiLine, SeasonBars, SeasonLine, STATE_COLORS, YieldHistoryChart } from "@/components/charts";
import { ErrorState, Loading, Note, PageHeader, Section, Stat } from "@/components/common";
import { api } from "@/lib/api";
import { n, yld } from "@/lib/format";
import { useSettings } from "@/lib/settings";
import { useApi } from "@/lib/useApi";

function slope(xs: number[], ys: number[]) {
  const mx = xs.reduce((a, b) => a + b, 0) / xs.length;
  const my = ys.reduce((a, b) => a + b, 0) / ys.length;
  const num = xs.reduce((a, x, i) => a + (x - mx) * (ys[i] - my), 0);
  const den = xs.reduce((a, x) => a + (x - mx) ** 2, 0);
  return den ? num / den : null;
}

export default function Trends() {
  const { s, countyName, meta } = useSettings();
  const ser = useApi(() => api.series(s.fips, s.method), [s.fips, s.method]);
  const ss = useApi(() => api.stateSeries(s.method), [s.method]);
  const rows = ser.data?.rows ?? [];
  const trend = useMemo(() => {
    const d = rows.filter((r) => r.observed_t_ha !== null);
    return d.length > 5 ? slope(d.map((r) => r.year), d.map((r) => r.observed_t_ha as number)) : null;
  }, [rows]);
  const mean = (k: keyof (typeof rows)[number], a: number, b: number) => {
    const v = rows.filter((r) => r.year >= a && r.year <= b).map((r) => r[k] as number | null).filter((x): x is number => x !== null);
    return v.length ? v.reduce((x, y) => x + y, 0) / v.length : null;
  };
  const stRows = useMemo(() => {
    if (!ss.data) return [];
    const by = new Map<number, Record<string, number | null>>();
    ss.data.rows.forEach((r) => { const o = by.get(r.year) ?? { year: r.year }; o[r.state] = r.compound_days_mean; by.set(r.year, o); });
    return [...by.values()].sort((a, b) => (a.year as number) - (b.year as number));
  }, [ss.data]);

  return (
    <>
      <PageHeader title="Historical trends" subtitle={`${countyName(s.fips)}, seasons ${meta?.seasons.join("–")}.`} />
      {ser.error && <ErrorState message={ser.error} />}
      {!ser.data ? <Loading /> : (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Stat label="Yield trend" value={trend !== null ? `${trend > 0 ? "+" : ""}${yld(trend, s.units)}` : "–"} sub="per season, straight-line fit 1985–2023" />
            <Stat label="Average yield 1985–1994" value={yld(mean("observed_t_ha", 1985, 1994), s.units)} />
            <Stat label="Average yield 2014–2023" value={yld(mean("observed_t_ha", 2014, 2023), s.units)} />
            <Stat label="Compound days per season" value={`${n(mean("cdhw_days", 1985, 2003), 1)} → ${n(mean("cdhw_days", 2004, 2023), 1)}`} sub="mean 1985–2003 → 2004–2023" />
          </div>
          <Section title="Maize yield" description="Observed yields, model estimates and 90% intervals."><YieldHistoryChart rows={rows} units={s.units} /></Section>
          <div className="grid gap-5 lg:grid-cols-2">
            <Section title="Growing degree days"><SeasonLine rows={rows.map((r) => ({ year: r.year, v: s.temp === "C" ? r.gdd : r.gdd !== null ? r.gdd * 1.8 : null }))} k="v" name="GDD" unit={s.temp === "C" ? "°C·day" : "°F·day"} reference={mean("gdd", 1985, 2023) !== null ? (s.temp === "C" ? mean("gdd", 1985, 2023) : (mean("gdd", 1985, 2023) as number) * 1.8) : null} /></Section>
            <Section title="Season precipitation"><SeasonLine rows={rows.map((r) => ({ year: r.year, v: s.units === "metric" ? r.ppt_mm : r.ppt_mm !== null ? r.ppt_mm / 25.4 : null }))} k="v" name="Precipitation" unit={s.units === "metric" ? "mm" : "in"} digits={s.units === "metric" ? 0 : 1}
              reference={mean("ppt_mm", 1985, 2023) !== null ? (s.units === "metric" ? mean("ppt_mm", 1985, 2023) : (mean("ppt_mm", 1985, 2023) as number) / 25.4) : null} /></Section>
            <Section title="Heat and drought days"><SeasonBars rows={rows.map((r) => ({ year: r.year, heat: r.heat_days, drought: r.drought_days }))} keys={[{ key: "drought", name: "Drought days (SPEI-30 < −1)" }, { key: "heat", name: "Days above 35 °C" }]} /></Section>
            <Section title="Mean daily maximum temperature"><SeasonLine rows={rows.map((r) => ({ year: r.year, v: r.tmax_mean_c === null ? null : s.temp === "C" ? r.tmax_mean_c : r.tmax_mean_c * 1.8 + 32 }))} k="v" name="Mean Tmax" unit={`°${s.temp}`} digits={1} /></Section>
          </div>
          <Section title="Compound drought–heat days by state" description="Mean compound days per county, each season.">
            {ss.data ? <MultiLine rows={stRows} series={Object.entries(meta?.states ?? {}).map(([k, v]) => ({ key: k, name: v, color: STATE_COLORS[k] }))} unitLabel="days" /> : <Loading />}
            <div className="mt-2"><Note>Trends describe the historical record. A straight-line fit summarises the change; it is not a projection.</Note></div>
          </Section>
        </>
      )}
    </>
  );
}
