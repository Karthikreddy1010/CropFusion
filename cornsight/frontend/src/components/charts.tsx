import {
  Area, Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { SeriesRow, Stage } from "@/lib/api";
import { n, yieldDigits, yieldUnit, yieldVal, type Units } from "@/lib/format";

const C = {
  obs: "oklch(0.3 0.03 120)", est: "oklch(0.5 0.13 145)", band: "oklch(0.75 0.1 145)", avg: "oklch(0.6 0.11 60)",
  grid: "oklch(0.92 0.01 95)", axis: "oklch(0.5 0.02 100)",
  s1: "oklch(0.55 0.12 145)", s2: "oklch(0.62 0.14 60)", s3: "oklch(0.5 0.09 240)",
};
const axisProps = { stroke: C.axis, fontSize: 12, tickLine: false, axisLine: false } as const;

/** Observed yield, model estimate with 90% interval band, and prior five-season average. */
export function YieldHistoryChart({ rows, units, from = 1985, highlight }: { rows: SeriesRow[]; units: Units; from?: number; highlight?: number }) {
  const d = rows.filter((r) => r.year >= from).map((r) => ({
    year: r.year,
    observed: yieldVal(r.observed_t_ha, units),
    estimate: yieldVal(r.pred_t_ha, units),
    band: r.lower_t_ha !== null && r.upper_t_ha !== null ? [yieldVal(r.lower_t_ha, units), yieldVal(r.upper_t_ha, units)] : null,
    average: yieldVal(r.prior5_avg_t_ha, units),
  }));
  const dg = yieldDigits(units);
  return (
    <ResponsiveContainer width="100%" height={320}>
      <ComposedChart data={d} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="year" {...axisProps} minTickGap={16} />
        <YAxis {...axisProps} width={48} domain={["auto", "auto"]} tickFormatter={(v) => n(v, 0)}
          label={{ value: yieldUnit(units), angle: -90, position: "insideLeft", fill: C.axis, fontSize: 12 }} />
        <Tooltip formatter={(v: unknown, name: string) => Array.isArray(v) ? [`${n(v[0], dg)}–${n(v[1], dg)}`, name] : [n(v as number, dg), name]}
          labelFormatter={(l) => `Season ${l}`} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Area dataKey="band" name="90% prediction interval" fill={C.band} fillOpacity={0.35} stroke="none" isAnimationActive={false} connectNulls={false} />
        <Line dataKey="average" name="Prior 5-season average" stroke={C.avg} strokeDasharray="5 4" dot={false} strokeWidth={1.5} isAnimationActive={false} />
        <Line dataKey="observed" name="Observed (USDA)" stroke={C.obs} strokeWidth={2} dot={{ r: 2.5 }} isAnimationActive={false} />
        <Line dataKey="estimate" name="Model estimate" stroke={C.est} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} connectNulls={false} />
        {highlight && <ReferenceLine x={highlight} stroke={C.axis} strokeDasharray="3 3" />}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/** One metric across seasons as bars (e.g. compound drought–heat days). */
export function SeasonBars({ rows, keys, height = 220, highlight }: {
  rows: Record<string, number | null>[]; keys: { key: string; name: string; color?: string }[]; height?: number; highlight?: number;
}) {
  const cols = [C.s2, C.s1, C.s3];
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 0 }} barGap={2}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="year" {...axisProps} minTickGap={16} />
        <YAxis {...axisProps} width={40} allowDecimals={false} />
        <Tooltip labelFormatter={(l) => `Season ${l}`} formatter={(v: number, nm: string) => [n(v, 0), nm]} />
        {keys.length > 1 && <Legend wrapperStyle={{ fontSize: 12 }} />}
        {keys.map((k, i) => <Bar key={k.key} dataKey={k.key} name={k.name} fill={k.color ?? cols[i % 3]} radius={[3, 3, 0, 0]} isAnimationActive={false} />)}
        {highlight && <ReferenceLine x={highlight} stroke={C.axis} strokeDasharray="3 3" />}
      </BarChart>
    </ResponsiveContainer>
  );
}

/** A metric across seasons as a line with an optional reference (county mean). */
export function SeasonLine({ rows, k, name, unit, reference, digits = 0, height = 220 }: {
  rows: Record<string, number | null>[]; k: string; name: string; unit: string; reference?: number | null; digits?: number; height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="year" {...axisProps} minTickGap={16} />
        <YAxis {...axisProps} width={48} domain={["auto", "auto"]} tickFormatter={(v) => n(v, digits)} />
        <Tooltip labelFormatter={(l) => `Season ${l}`} formatter={(v: number) => [`${n(v, digits)} ${unit}`, name]} />
        {reference !== undefined && reference !== null && <ReferenceLine y={reference} stroke={C.avg} strokeDasharray="5 4" label={{ value: "1985–2023 mean", fill: C.axis, fontSize: 11, position: "insideTopRight" }} />}
        <Line dataKey={k} name={name} stroke={C.s1} strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/** Grouped bars of a stage metric for the three modelled growth stages. */
export function StageBars({ stages, metrics }: { stages: Stage[]; metrics: { key: keyof Stage; name: string; color: string }[] }) {
  const d = stages.map((s) => ({ stage: s.label, ...Object.fromEntries(metrics.map((m) => [m.key, s[m.key]])) }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={d} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="stage" {...axisProps} />
        <YAxis {...axisProps} width={40} />
        <Tooltip formatter={(v: number, nm: string) => [n(v, 1), nm]} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {metrics.map((m) => <Bar key={String(m.key)} dataKey={String(m.key)} name={m.name} fill={m.color} radius={[3, 3, 0, 0]} isAnimationActive={false} />)}
      </BarChart>
    </ResponsiveContainer>
  );
}

/** State yield trends, one line per state. */
export function MultiLine({ rows, series, unitLabel, digits = 1, height = 320 }: {
  rows: Record<string, number | null>[]; series: { key: string; name: string; color: string }[]; unitLabel: string; digits?: number; height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="year" {...axisProps} minTickGap={16} />
        <YAxis {...axisProps} width={48} domain={["auto", "auto"]} tickFormatter={(v) => n(v, 0)}
          label={{ value: unitLabel, angle: -90, position: "insideLeft", fill: C.axis, fontSize: 12 }} />
        <Tooltip labelFormatter={(l) => `Season ${l}`} formatter={(v: number, nm: string) => [n(v, digits), nm]} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {series.map((s) => <Line key={s.key} dataKey={s.key} name={s.name} stroke={s.color} strokeWidth={1.8} dot={false} isAnimationActive={false} />)}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/** Six distinguishable colours for the six states, fixed order IL IN IA MN MO OH. */
export const STATE_COLORS: Record<string, string> = {
  IL: "oklch(0.5 0.13 145)", IN: "oklch(0.62 0.15 60)", IA: "oklch(0.5 0.12 250)",
  MN: "oklch(0.55 0.16 320)", MO: "oklch(0.6 0.1 200)", OH: "oklch(0.45 0.05 50)",
};
