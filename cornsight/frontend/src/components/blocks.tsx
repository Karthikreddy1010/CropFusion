import { AlertTriangle, ExternalLink } from "lucide-react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { Alert, Profile, Recommendation } from "@/lib/api";
import { n, pct, precip, production, share, temp, yld, yldBoth, yldRange, type TempUnit, type Units } from "@/lib/format";
import { cn } from "@/lib/utils";
import { Hint, LEVEL_COLOR, Note, RiskBadge, Stat, Unavailable } from "./common";

/** Visual of a prediction interval on a value axis with the observed yield and the 5-season average. */
export function IntervalBar({ p, units }: { p: Profile["yield"]; units: Units }) {
  const e = p.estimate;
  if (!e) return null;
  const obs = p.observed.t_ha;
  const avg = p.prior5_average.t_ha;
  const vals = [e.lower_t_ha, e.upper_t_ha, e.t_ha, obs ?? e.t_ha, avg ?? e.t_ha];
  const lo = Math.min(...vals) - 0.6;
  const hi = Math.max(...vals) + 0.6;
  const x = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
  return (
    <div className="mt-2" aria-label="Prediction interval">
      <div className="relative h-14">
        <div className="absolute top-6 h-3 rounded-full bg-[oklch(0.82_0.09_145)]" style={{ left: x(e.lower_t_ha), width: `calc(${x(e.upper_t_ha)} - ${x(e.lower_t_ha)})` }} />
        <div className="absolute top-4 h-7 w-1 -translate-x-1/2 rounded bg-primary" style={{ left: x(e.t_ha) }} title="Model estimate" />
        {avg !== null && <div className="absolute top-3 h-9 w-0 -translate-x-1/2 border-l-2 border-dashed border-[oklch(0.6_0.11_60)]" style={{ left: x(avg) }} title="5-season average" />}
        {obs !== null && <div className="absolute top-5 size-4 -translate-x-1/2 rounded-full border-2 border-white bg-foreground shadow" style={{ left: x(obs) }} title="Observed" />}
        <span className="num absolute top-0 -translate-x-1/2 text-[11px] text-muted-foreground" style={{ left: x(e.lower_t_ha) }}>{yld(e.lower_t_ha, units, false)}</span>
        <span className="num absolute top-0 -translate-x-1/2 text-[11px] text-muted-foreground" style={{ left: x(e.upper_t_ha) }}>{yld(e.upper_t_ha, units, false)}</span>
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span className="inline-flex items-center gap-1"><span className="h-2.5 w-5 rounded-full bg-[oklch(0.82_0.09_145)]" />90% prediction interval</span>
        <span className="inline-flex items-center gap-1"><span className="h-3 w-1 rounded bg-primary" />Model estimate</span>
        {obs !== null && <span className="inline-flex items-center gap-1"><span className="size-3 rounded-full bg-foreground" />Observed (USDA)</span>}
        {avg !== null && <span className="inline-flex items-center gap-1"><span className="h-3 w-0 border-l-2 border-dashed border-[oklch(0.6_0.11_60)]" />Prior 5-season average</span>}
      </div>
    </div>
  );
}

export function YieldCards({ p, units, farmAcres }: { p: Profile; units: Units; farmAcres?: number | null }) {
  const y = p.yield;
  const e = y.estimate;
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
      <Stat label="Model yield estimate" value={e ? yld(e.t_ha, units) : "Not available"}
        sub={e ? `90% interval ${yldRange(e.lower_t_ha, e.upper_t_ha, units)}` : y.unavailable_reason}
        hint={e ? e.explanation : undefined} accent="var(--primary)" />
      <Stat label="Prior 5-season average" value={yld(y.prior5_average.t_ha, units)} sub={y.prior5_average.definition} />
      <Stat label="Estimate vs average" value={e?.vs_average_pct !== undefined ? pct(e.vs_average_pct, 1) : "–"}
        sub={y.observed.t_ha !== null ? `Observed: ${yld(y.observed.t_ha, units)} (${pct(y.observed_vs_average_pct, 1)})` : "Observed yield not reported"}
        accent={e?.vs_average_pct !== undefined ? (e.vs_average_pct <= -10 ? "var(--risk-high)" : e.vs_average_pct < 0 ? "var(--risk-moderate)" : "var(--risk-low)") : undefined} />
      <Stat label={farmAcres ? "Estimated farm production" : "Estimated county production"}
        value={farmAcres ? production(e?.farm_production_bu, units) : production(e?.production_bu, units)}
        sub={farmAcres
          ? (e ? `${n(farmAcres, 0)} acres × estimate · range ${production(e.farm_production_bu_lower, units)}–${production(e.farm_production_bu_upper, units)}` : "Needs a model estimate")
          : (e?.production_bu ? `USDA harvested acres × estimate. Reported: ${production(y.production_bu_observed, units)}` : "Harvested acres or estimate missing")}
        hint="Production = yield × harvested area. The range uses the 90% interval bounds." />
    </div>
  );
}

export function UncertaintyText({ p, units }: { p: Profile; units: Units }) {
  const e = p.yield.estimate;
  if (!e) return <Unavailable title="No model estimate for this season">{p.yield.unavailable_reason}</Unavailable>;
  return (
    <div className="space-y-2 text-sm leading-relaxed">
      <p><span className="font-semibold">Estimated yield:</span> {yldBoth(e.t_ha, units)}.</p>
      <p><span className="font-semibold">Prediction interval:</span> {yldRange(e.lower_t_ha, e.upper_t_ha, units)} (90% nominal coverage, width {yld(e.width_t_ha, units)}).</p>
      <p className="text-muted-foreground">These bounds describe prediction uncertainty, not a guaranteed range. {e.interval_explanation.split(". ").slice(1).join(". ")}</p>
      {e.inside_interval !== null && (
        <p>
          <span className="font-semibold">Outcome:</span> the observed yield ({yld(p.yield.observed.t_ha, units)}) fell{" "}
          {e.inside_interval ? "inside" : (p.yield.observed.t_ha ?? 0) < e.lower_t_ha ? "below" : "above"} the interval.
        </p>
      )}
      <Note>{e.explanation} Interval method: {p.yield.method_label}. Exposure class used by the model: {e.exposure_class}.</Note>
    </div>
  );
}

const RISK_ORDER: { key: keyof Profile["risk"]; label: string }[] = [
  { key: "drought", label: "Drought exposure" }, { key: "heat", label: "Heat stress" },
  { key: "rainfall", label: "Season rainfall" }, { key: "compound", label: "Compound drought–heat" },
];

export function RiskCards({ p, units, tempUnit, simple = false }: { p: Profile; units: Units; tempUnit: TempUnit; simple?: boolean }) {
  const r = p.risk;
  const days = (v: number | null | undefined) => `${n(v, 0)} ${v === 1 ? "day" : "days"}`;
  const detail: Record<string, string> = {
    drought: `${days(r.drought?.value)} with SPEI-30 below −1`,
    heat: `${days(r.heat?.value)} above ${tempUnit === "C" ? "35 °C" : "95 °F"}`,
    rainfall: `${precip(r.rainfall?.value, units)} from 1 Apr to 31 Oct`,
    compound: `${days(r.compound?.value)} hot (> ${tempUnit === "C" ? "35 °C" : "95 °F"}) and dry (SPEI-30 < −1) together`,
  };
  const plain: Record<string, string> = {
    drought: "How many days the 30-day water balance was in deficit.",
    heat: "Days hot enough to damage maize, especially around silking.",
    rainfall: "Total rain compared with this county's usual season. High means wetter than usual.",
    compound: "Days with heat and drought at the same time, the combination most harmful to yield.",
  };
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
      {RISK_ORDER.map(({ key, label }) => {
        const it = r[key];
        return (
          <div key={key} className="rounded-lg border bg-card p-4" style={{ borderLeft: `4px solid ${it?.level ? LEVEL_COLOR[it.level] : "var(--border)"}` }}>
            <div className="flex items-center justify-between gap-2">
              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{label}</p>
              {it?.rule && <Hint>{String(it.rule)}</Hint>}
            </div>
            <div className="mt-2"><RiskBadge level={it?.level} basis={simple ? undefined : it?.basis} /></div>
            <p className="num mt-2 text-sm font-medium">{detail[key]}</p>
            <p className="mt-1 text-xs text-muted-foreground">{simple ? plain[key] : it?.percentile !== undefined && it?.percentile !== null ? `More than in ${share(it.percentile)} of this county's seasons 1985–2023.` : plain[key]}</p>
          </div>
        );
      })}
    </div>
  );
}

export function StormCard({ p }: { p: Profile }) {
  const s = p.risk.storms as Record<string, number | string | null>;
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {[["Hail reports", s.hail], ["Thunderstorm wind", s.thunderstorm_wind], ["Tornado reports", s.tornado], ["Largest hail", s.max_hail_in !== null && s.max_hail_in !== undefined ? `${n(Number(s.max_hail_in), 2)} in` : "–"]].map(([l, v]) => (
        <div key={String(l)} className="rounded-md border p-3">
          <p className="text-xs text-muted-foreground">{l}</p>
          <p className="num text-xl font-semibold">{typeof v === "number" ? n(v, 0) : (v ?? "–")}</p>
        </div>
      ))}
      <div className="col-span-full"><Note>{String(s.note)}</Note></div>
    </div>
  );
}

export function AlertList({ alerts, empty }: { alerts: Alert[]; empty?: string }) {
  if (!alerts.length) return <p className="text-sm text-muted-foreground">{empty ?? "No alerts for this county and season."}</p>;
  return (
    <ul className="flex flex-col gap-3">
      {alerts.map((a) => (
        <li key={a.id} className="rounded-lg border bg-card p-4" style={{ borderLeft: `4px solid ${LEVEL_COLOR[a.severity] ?? "var(--border)"}` }}>
          <div className="flex flex-wrap items-center gap-2">
            <AlertTriangle className="size-4" style={{ color: LEVEL_COLOR[a.severity] }} />
            <p className="font-semibold">{a.title}</p>
            <RiskBadge level={a.severity as Alert["severity"] as never} />
            <span className="text-xs text-muted-foreground">{a.basis === "documented" ? "Model-defined threshold" : "Proposed threshold"}: {a.threshold}</span>
          </div>
          <dl className="mt-3 grid gap-x-6 gap-y-2 text-sm sm:grid-cols-[8rem_1fr]">
            <dt className="text-muted-foreground">What happened</dt><dd>{a.what}</dd>
            <dt className="text-muted-foreground">Where</dt><dd>{a.where}</dd>
            <dt className="text-muted-foreground">Period</dt><dd>{a.period}</dd>
            <dt className="text-muted-foreground">Possible impact</dt><dd>{a.impact}</dd>
            <dt className="text-muted-foreground">Consider</dt><dd>{a.consider}</dd>
          </dl>
        </li>
      ))}
    </ul>
  );
}

export function RecommendationList({ recs }: { recs: Recommendation[] }) {
  return (
    <div className="grid gap-3 md:grid-cols-2">
      {recs.map((r) => (
        <div key={r.topic} className="flex flex-col gap-2 rounded-lg border bg-card p-4">
          <p className="font-semibold">{r.topic}</p>
          <p className="text-sm text-muted-foreground">{r.why}</p>
          <ul className="list-disc space-y-1 pl-5 text-sm">{r.consider.map((c) => <li key={c}>{c}</li>)}</ul>
          <div className="mt-auto flex flex-wrap gap-x-3 gap-y-1 pt-1 text-xs">
            {r.sources.map((s) => s.url
              ? <a key={s.name} href={s.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-primary underline-offset-2 hover:underline">{s.name}<ExternalLink className="size-3" /></a>
              : <span key={s.name} className="text-muted-foreground">{s.name}</span>)}
          </div>
          <p className="text-[11px] text-muted-foreground">{r.note}</p>
        </div>
      ))}
    </div>
  );
}

export function ConditionsTable({ p, units, tempUnit, compact = false }: { p: Profile; units: Units; tempUnit: TempUnit; compact?: boolean }) {
  if (!p.conditions.available) return <Unavailable title="No climate record">This county has no climate data for the season.</Unavailable>;
  const fmt = (key: string, unit: string, v: number | null, diff = false) => {
    if (v === null) return "–";
    if (unit === "°C") return diff ? `${v > 0 ? "+" : ""}${n(tempUnit === "C" ? v : v * 1.8, 1)} °${tempUnit}` : temp(v, tempUnit);
    if (unit === "mm") return diff ? `${v > 0 ? "+" : ""}${precip(v, units)}` : precip(v, units);
    if (unit === "°C·day") return `${diff && v > 0 ? "+" : ""}${n(tempUnit === "C" ? v : v * 1.8, 0)} °${tempUnit}·day`;
    return `${diff && v > 0 ? "+" : ""}${n(v, key.includes("spei") || unit === "kPa" ? 2 : 0)} ${unit === "index" ? "" : unit}`;
  };
  const items = compact ? p.conditions.items.filter((i) => ["prism_ppt_total_gs", "prism_tmax_mean_gs", "prism_gdd_gs", "water_balance_total_gs", "spei90_prism_nldas_mean_gs"].includes(i.key)) : p.conditions.items;
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow><TableHead>Indicator ({p.conditions.period})</TableHead><TableHead className="text-right">Season {p.season}</TableHead>
            <TableHead className="text-right">County mean 1985–2023</TableHead><TableHead className="text-right">Difference</TableHead><TableHead className="text-right">Percentile <Hint>Share of this county's 1985–2023 seasons with the same or a lower value.</Hint></TableHead></TableRow>
        </TableHeader>
        <TableBody>
          {items.map((i) => (
            <TableRow key={i.key}>
              <TableCell>{i.label}</TableCell>
              <TableCell className="num text-right font-medium">{fmt(i.key, i.unit, i.value)}</TableCell>
              <TableCell className="num text-right text-muted-foreground">{fmt(i.key, i.unit, i.county_mean_1985_2023)}</TableCell>
              <TableCell className={cn("num text-right")}>{fmt(i.key, i.unit, i.difference, true)}</TableCell>
              <TableCell className="num text-right text-muted-foreground">{i.percentile !== null ? n(100 * i.percentile, 0) : "–"}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
