import { Download } from "lucide-react";
import { useState } from "react";
import { PageHeader, Section } from "@/components/common";
import { StatePicker } from "@/components/pickers";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { api, reportUrl } from "@/lib/api";
import { useSettings } from "@/lib/settings";
import { useApi } from "@/lib/useApi";

function Row({ title, desc, href, children }: { title: string; desc: string; href: string; children?: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-3 border-b py-4 last:border-0 sm:flex-row sm:items-center sm:justify-between">
      <div className="min-w-0"><p className="font-semibold">{title}</p><p className="text-sm text-muted-foreground">{desc}</p></div>
      <div className="flex flex-wrap items-center gap-2">{children}<Button asChild><a href={href} download><Download />CSV</a></Button></div>
    </div>
  );
}

export default function Reports() {
  const { s, season, countyName } = useSettings();
  const [metric, setMetric] = useState("cdhw_days");
  const [st, setSt] = useState("");
  const mets = useApi(() => api.rankMetrics(), []);
  const pro = s.role !== "farmer";
  return (
    <>
      <PageHeader title="Reports and exports" subtitle="CSV files open in Excel. Every file starts with lines naming the source files and their checksums, so figures can be traced." />
      <Section title="Exports">
        <Row title={`County history: ${countyName(s.fips)}`} desc="Every season: observed yield, model estimate and 90% interval, climate and compound drought–heat indicators." href={reportUrl.county(s.fips, s.method)} />
        {pro && <Row title={`Regional county table, season ${season}`} desc="All counties: yields, estimates, intervals, exposure, harvested acres and estimated production." href={reportUrl.region(season, s.method)} />}
        {pro && <Row title={`State summary, season ${season}`} desc="Acre-weighted yields, production and high-risk county counts for the six states." href={reportUrl.states(season, s.method)} />}
        {pro && (
          <Row title={`County ranking, season ${season}`} desc="All counties ranked by the chosen metric." href={reportUrl.rankings(metric, season, s.method, st || undefined)}>
            <Select value={metric} onValueChange={setMetric}>
              <SelectTrigger className="w-[230px]" aria-label="Ranking metric"><SelectValue /></SelectTrigger>
              <SelectContent>{Object.entries(mets.data ?? {}).map(([k, v]) => <SelectItem key={k} value={k}>{v.label}</SelectItem>)}</SelectContent>
            </Select>
            <StatePicker value={st} onChange={setSt} />
          </Row>
        )}
      </Section>
      <p className="text-xs text-muted-foreground">Reports describe completed seasons. Yields are in t/ha in the files (multiply by 15.95 for bu/acre).</p>
    </>
  );
}
