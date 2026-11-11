import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ErrorState, Loading, Note, PageHeader, Section } from "@/components/common";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api } from "@/lib/api";
import { n, share, yld } from "@/lib/format";
import { useSettings } from "@/lib/settings";
import { useApi } from "@/lib/useApi";

const LABELS: Record<string, string> = {
  county_baseline: "County yield level (baseline)", Yield_lag1: "Previous season's yield", Yield_lag2: "Yield two seasons back",
  spei30_prism_nldas_mean_silk: "SPEI-30 during silking", spi1_prism_min_silk: "Lowest SPI-1 during silking",
};

export default function Model() {
  const { s, meta } = useSettings();
  const att = useApi(() => api.attribution(), []);
  const rels = useApi(() => Promise.all(Object.keys(meta?.methods ?? {}).map((m) => api.reliability(m))), [meta]);
  return (
    <>
      <PageHeader title="Model and uncertainty" subtitle="What the yield model relies on and how well its prediction intervals have held up." />
      <Section title="How estimates are produced">
        <ul className="list-disc space-y-1.5 pl-5 text-sm">
          <li>For each season from {meta?.estimate_seasons[0]} to {meta?.estimate_seasons[1]}, a LightGBM model was trained only on seasons at least five years earlier, tuned on the next two, and calibrated on the two seasons just before.</li>
          <li>Conformal calibration turns raw 5% and 95% quantile predictions into a 90% prediction interval. The group-conditional method calibrates separately for seasons with Normal, Moderate and Extreme compound drought–heat exposure.</li>
          <li>Inputs: growing-season and stage-specific weather, SPI/SPEI drought indices, compound drought–heat, storm reports, soil, terrain, land cover and past yields, as selected by the research pipeline.</li>
          <li>The research also trained NeuralCQR, CatBoost and XGBoost models; their stored outputs are not shown here because they were evaluated under a different validation design.</li>
        </ul>
      </Section>
      <Section title="Interval reliability by method" description="Share of county-seasons where the observed yield fell inside (target 90%) and below (reference 5%) the interval.">
        {rels.error && <ErrorState message={rels.error} />}
        {!rels.data ? <Loading /> : (
          <div className="overflow-x-auto"><Table>
            <TableHeader><TableRow><TableHead>Method</TableHead>{["Normal", "Moderate", "Extreme", "ALL"].map((c) => <TableHead key={c} className="text-right">{c === "ALL" ? "All seasons" : c}</TableHead>)}<TableHead className="text-right">Mean width</TableHead></TableRow></TableHeader>
            <TableBody>{rels.data.map((r) => (
              <TableRow key={r.method} className={r.method === s.method ? "bg-accent/50" : ""}>
                <TableCell className="font-medium">{r.method_label}</TableCell>
                {r.by_class.map((c) => <TableCell key={c.exposure_class} className="num text-right">{share(c.inside_rate, 1)} <span className="text-muted-foreground">/ {share(c.below_lower_rate, 1)}</span></TableCell>)}
                <TableCell className="num text-right">{yld(r.by_class.find((c) => c.exposure_class === "ALL")?.mean_width_t_ha, s.units)}</TableCell>
              </TableRow>))}
            </TableBody></Table></div>
        )}
        <div className="mt-2"><Note>Cells show inside / below. Rows: {rels.data?.[0]?.by_class.map((c) => `${c.exposure_class} ${n(c.rows, 0)}`).join(", ")}. {rels.data?.[0]?.note}</Note></div>
      </Section>
      <Section title="What the models rely on overall" description={att.data?.note}>
        {!att.data ? <Loading /> : (
          <ResponsiveContainer width="100%" height={460}>
            <BarChart data={att.data.rows.slice(0, 15).map((r) => ({ name: LABELS[r.Feature] ?? r.Feature, v: 100 * r.Mean_Attribution }))} layout="vertical" margin={{ top: 4, right: 16, bottom: 0, left: 8 }}>
              <CartesianGrid horizontal={false} stroke="oklch(0.92 0.01 95)" />
              <XAxis type="number" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis type="category" dataKey="name" width={230} fontSize={11} tickLine={false} axisLine={false} />
              <Tooltip formatter={(v: number) => [`${n(v, 1)}`, "Mean attribution × 100"]} />
              <Bar dataKey="v" fill="oklch(0.5 0.13 145)" radius={[0, 3, 3, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </Section>
    </>
  );
}
