import { Bot, Database, Loader2, Send, User } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { ErrorState, Note, PageHeader, Section, Unavailable } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Textarea } from "@/components/ui/textarea";
import { api } from "@/lib/api";
import { n, pct, share, yld, yldRange, type Units } from "@/lib/format";
import { useSettings } from "@/lib/settings";
import { useApi } from "@/lib/useApi";

/* eslint-disable @typescript-eslint/no-explicit-any */
type Q = { id: string; label: string; tool: string; args: Record<string, unknown>; render: (r: any, u: Units) => ReactNode };

function Sources({ r }: { r: any }) {
  return r?.sources ? <p className="mt-3 text-xs text-muted-foreground"><span className="font-semibold">Sources:</span> {r.sources.join("; ")}</p> : null;
}

function questions(fips: string, county: string, state: string, season: number): Q[] {
  return [
    { id: "trend", label: `What is the historical maize yield trend in ${county}?`, tool: "county_history", args: { fips },
      render: (r, u) => {
        const d = (r.rows as any[]).filter((x) => x.observed_t_ha !== null);
        const first = d.slice(0, 10), last = d.slice(-10);
        const avg = (a: any[]) => a.reduce((s, x) => s + x.observed_t_ha, 0) / a.length;
        return (<div className="text-sm"><p>{r.county}: average observed yield was {yld(avg(first), u)} in {first[0]?.year}–{first.at(-1)?.year} and {yld(avg(last), u)} in {last[0]?.year}–{last.at(-1)?.year} ({pct(100 * (avg(last) - avg(first)) / avg(first), 0)}). These are historical USDA yields.</p><Sources r={r} /></div>);
      } },
    { id: "compound", label: `Which counties had the most compound drought–heat days in ${season}?`, tool: "rank_counties", args: { season, metric: "cdhw_days", limit: 10 },
      render: (r, u) => (<div className="text-sm"><Table><TableHeader><TableRow><TableHead>County</TableHead><TableHead className="text-right">Compound days</TableHead><TableHead className="text-right">Estimate</TableHead><TableHead className="text-right">Observed</TableHead></TableRow></TableHeader>
        <TableBody>{r.rows.map((x: any) => <TableRow key={x.fips}><TableCell>{x.county}, {x.state}</TableCell><TableCell className="num text-right">{n(x.value, 0)}</TableCell><TableCell className="num text-right">{yld(x.estimate_t_ha, u)}</TableCell><TableCell className="num text-right">{yld(x.observed_t_ha, u)}</TableCell></TableRow>)}</TableBody></Table>
        <p className="mt-2">Season {r.season}, {r.counties_ranked} counties ranked. A compound day has Tmax above 35 °C while SPEI-30 is below −1.</p><Sources r={r} /></div>) },
    { id: "uncert", label: `What is the uncertainty of ${county}'s ${season} yield estimate?`, tool: "county_season", args: { fips, season },
      render: (r, u) => { const e = r.yield.estimate; return (<div className="text-sm">{e ? <p>{r.county}, {r.season}: the model estimated {yld(e.t_ha, u)} with a 90% prediction interval of {yldRange(e.lower_t_ha, e.upper_t_ha, u)} (width {yld(e.width_t_ha, u)}). The interval describes prediction uncertainty, not a guaranteed range. The observed yield was {yld(r.yield.observed.t_ha, u)}, {e.inside_interval ? "inside" : "outside"} the interval. This is a reconstruction of a completed season, not a forecast.</p> : <p>{r.yield.unavailable_reason}</p>}<Sources r={r} /></div>); } },
    { id: "lower", label: `Why might ${county}'s ${season} estimate differ from its average?`, tool: "county_season", args: { fips, season },
      render: (r, u) => { const e = r.yield.estimate; const m = r.most_stressed_stage ?? {}; return (<div className="space-y-2 text-sm">
        <p>{r.county}, {r.season}: estimate {e ? yld(e.t_ha, u) : "not available"} against a prior five-season average of {yld(r.yield.prior5_average.t_ha, u)}{e?.vs_average_pct !== undefined ? ` (${pct(e.vs_average_pct, 1)})` : ""}.</p>
        <p>Season conditions: drought {r.risk.drought.level} ({n(r.risk.drought.value, 0)} days with SPEI-30 &lt; −1), heat {r.risk.heat.level} ({n(r.risk.heat.value, 0)} days above 35 °C), compound drought–heat {r.risk.compound.level} ({n(r.risk.compound.value, 0)} days){m.compound_days ? `, mostly in the ${m.compound_days.stage.toLowerCase()} stage` : ""}.</p>
        <p className="text-muted-foreground">The platform has the model's overall input importance, not a breakdown for each county, so these conditions are associated with the estimate rather than proven causes.</p><Sources r={r} /></div>); } },
    { id: "stage", label: `Which growth stage had the most heat stress in ${county} in ${season}?`, tool: "county_season", args: { fips, season },
      render: (r) => { const m = r.most_stressed_stage ?? {}; return (<div className="text-sm"><p>{r.county}, {r.season}: {m.heat_days ? `most days above 35 °C fell in the ${m.heat_days.stage.toLowerCase()} stage (${n(m.heat_days.value, 0)} days)` : "no stage had days above 35 °C"}; {m.compound_days ? `most compound drought–heat days fell in the ${m.compound_days.stage.toLowerCase()} stage (${n(m.compound_days.value, 0)})` : "no compound drought–heat days occurred"}.</p><p className="mt-1 text-muted-foreground">{r.stage_note}</p><Sources r={r} /></div>); } },
    { id: "states", label: `Compare maize yield estimates between Illinois and Indiana in ${season}`, tool: "state_season", args: { season },
      render: (r, u) => (<div className="text-sm"><Table><TableHeader><TableRow><TableHead>State</TableHead><TableHead className="text-right">Estimate</TableHead><TableHead className="text-right">Interval (acre-weighted)</TableHead><TableHead className="text-right">Observed</TableHead><TableHead className="text-right">High compound counties</TableHead></TableRow></TableHeader>
        <TableBody>{r.states.filter((x: any) => ["IL", "IN"].includes(x.state)).map((x: any) => <TableRow key={x.state}><TableCell>{x.state_name}</TableCell><TableCell className="num text-right">{yld(x.estimate_t_ha_acre_weighted, u)}</TableCell><TableCell className="num text-right">{x.lower_t_ha_acre_weighted ? yldRange(x.lower_t_ha_acre_weighted, x.upper_t_ha_acre_weighted, u) : "–"}</TableCell><TableCell className="num text-right">{yld(x.observed_t_ha_acre_weighted, u)}</TableCell><TableCell className="num text-right">{x.counties_compound_high}</TableCell></TableRow>)}</TableBody></Table>
        <p className="mt-2 text-muted-foreground">Season {r.season}. Acre-weighted means of panel counties, not official NASS state estimates.</p><Sources r={r} /></div>) },
    { id: "iowa", label: `How was compound drought–heat associated with maize yield in ${state}?`, tool: "compound_vs_yield", args: { state },
      render: (r) => (<div className="text-sm"><Table><TableHeader><TableRow><TableHead>Compound class</TableHead><TableHead className="text-right">County-seasons</TableHead><TableHead className="text-right">Median yield vs avg.</TableHead><TableHead className="text-right">&gt;10% below avg.</TableHead></TableRow></TableHeader>
        <TableBody>{r.classes.map((c: any) => <TableRow key={c.class}><TableCell>{c.class}</TableCell><TableCell className="num text-right">{n(c.county_seasons, 0)}</TableCell><TableCell className="num text-right">{pct(c.median_dev_pct, 1)}</TableCell><TableCell className="num text-right">{share(c.share_below_avg_10pct)}</TableCell></TableRow>)}</TableBody></Table>
        <p className="mt-2 text-muted-foreground">{r.note}</p><Sources r={r} /></div>) },
    { id: "prio", label: `Which counties should organisations prioritise for risk review in ${season}?`, tool: "rank_counties", args: { season, metric: "estimate_vs_average_pct", lowest_first: true, limit: 10 },
      render: (r, u) => (<div className="text-sm"><p className="mb-2">Counties with the largest estimated drop below their own five-season average (one transparent criterion; combine with exposure and acreage for decisions):</p>
        <Table><TableHeader><TableRow><TableHead>County</TableHead><TableHead className="text-right">Estimate vs avg.</TableHead><TableHead className="text-right">Estimate</TableHead><TableHead className="text-right">Compound days</TableHead></TableRow></TableHeader>
        <TableBody>{r.rows.map((x: any) => <TableRow key={x.fips}><TableCell>{x.county}, {x.state}</TableCell><TableCell className="num text-right">{pct(x.value, 1)}</TableCell><TableCell className="num text-right">{yld(x.estimate_t_ha, u)}</TableCell><TableCell className="num text-right">{n(x.cdhw_days, 0)}</TableCell></TableRow>)}</TableBody></Table><Sources r={r} /></div>) },
  ];
}

function Lookup() {
  const { s, season, counties } = useSettings();
  const c = counties.find((x) => x.fips === s.fips);
  const qs = questions(s.fips, c ? `${c.county} County` : s.fips, c?.state ?? "IA", Math.max(season, 2008));
  const [active, setActive] = useState<Q | null>(null);
  const [res, setRes] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const run = (q: Q) => {
    setActive(q); setRes(null); setErr(null); setBusy(true);
    api.lookup(q.tool, q.args).then((r) => { if ((r as any).error) setErr(String((r as any).error)); else setRes(r); }, (e) => setErr(e.message)).finally(() => setBusy(false));
  };
  return (
    <div className="grid gap-4 lg:grid-cols-[22rem_1fr]">
      <div className="flex flex-col gap-2">
        {qs.map((q) => (
          <button key={q.id} type="button" onClick={() => run(q)} aria-pressed={active?.id === q.id}
            className="rounded-md border bg-card px-3 py-2 text-left text-sm hover:border-primary hover:bg-accent aria-pressed:border-primary aria-pressed:bg-accent">{q.label}</button>
        ))}
      </div>
      <div className="min-w-0 rounded-lg border bg-card p-4">
        {!active && <p className="text-sm text-muted-foreground">Choose a question. Answers are computed directly from the platform data, with sources.</p>}
        {busy && <p className="flex items-center gap-2 text-sm text-muted-foreground"><Loader2 className="size-4 animate-spin" />Looking up…</p>}
        {err && <ErrorState message={err} />}
        {active && res && <><p className="mb-3 font-semibold">{active.label}</p>{active.render(res, s.units)}</>}
      </div>
    </div>
  );
}

function Chat() {
  const { s, season, countyName } = useSettings();
  const [msgs, setMsgs] = useState<{ role: "user" | "assistant"; content: string; tools?: { tool: string; ok: boolean }[] }[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }); }, [msgs, busy]);
  const send = () => {
    const t = text.trim();
    if (!t || busy) return;
    const ctx = `[Context: user is viewing ${countyName(s.fips)} (FIPS ${s.fips}), season ${season}, units ${s.units === "metric" ? "metric" : "US"}.]\n`;
    const next = [...msgs, { role: "user" as const, content: t }];
    setMsgs(next); setText(""); setBusy(true); setErr(null);
    api.chat(next.map((m, i) => ({ role: m.role, content: i === next.length - 1 ? ctx + m.content : m.content })))
      .then((r) => setMsgs([...next, { role: "assistant", content: r.answer, tools: r.tool_calls }]), (e) => setErr(e.message))
      .finally(() => setBusy(false));
  };
  return (
    <div className="flex flex-col gap-3">
      <div className="flex max-h-[28rem] min-h-48 flex-col gap-3 overflow-y-auto rounded-lg border bg-muted/30 p-3">
        {!msgs.length && <p className="text-sm text-muted-foreground">Ask about yields, drought, heat, compound drought–heat or model uncertainty for any county or state in the platform. Answers use only platform data and end with their sources.</p>}
        {msgs.map((m, i) => (
          <div key={i} className="flex gap-2">
            <span className="mt-0.5 grid size-7 shrink-0 place-items-center rounded-full bg-card">{m.role === "user" ? <User className="size-4" /> : <Bot className="size-4 text-primary" />}</span>
            <div className="min-w-0 flex-1 text-sm">
              <p className="whitespace-pre-wrap leading-relaxed">{m.content}</p>
              {m.tools && m.tools.length > 0 && <p className="mt-1 flex flex-wrap items-center gap-1 text-[11px] text-muted-foreground"><Database className="size-3" />Looked up: {m.tools.map((x) => x.tool.replace(/_/g, " ")).join(", ")}</p>}
            </div>
          </div>
        ))}
        {busy && <p className="flex items-center gap-2 text-sm text-muted-foreground"><Loader2 className="size-4 animate-spin" />Looking up platform data…</p>}
        <div ref={end} />
      </div>
      {err && <ErrorState message={err} />}
      <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); send(); }}>
        <Textarea id="chat-input" value={text} onChange={(e) => setText(e.target.value)} rows={2} placeholder={`e.g. Why was ${countyName(s.fips)}'s ${season} estimate below average?`}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }} className="min-h-0" />
        <Button type="submit" disabled={busy || !text.trim()} aria-label="Send"><Send /></Button>
      </form>
    </div>
  );
}

export default function Assistant() {
  const st = useApi(() => api.assistantStatus(), []);
  return (
    <>
      <PageHeader title="AI agricultural assistant" subtitle="Answers questions from the platform's yield, climate and model data, and says when data are not available." />
      <Section title="Ask in your own words" description={st.data?.configured ? `Uses ${st.data.model} with read-only access to the platform's data tools.` : undefined}>
        {st.loading ? null : st.data?.configured ? <Chat /> : (
          <Unavailable title="The conversational assistant is not switched on">
            It needs an Anthropic API key on the computer running the backend. {st.data?.how_to_enable} Until then, use Data lookup below; it answers the same kinds of questions straight from the data.
          </Unavailable>
        )}
      </Section>
      <Section title="Data lookup" description="Common questions answered directly from platform data, no AI model involved."><Lookup /></Section>
      <Note>The assistant cannot see current weather, forecasts, satellite images or your fields. It does not give irrigation, fertilizer or pesticide prescriptions.</Note>
    </>
  );
}
