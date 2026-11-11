import { PageHeader, Section } from "@/components/common";
import { CountyPicker } from "@/components/pickers";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api } from "@/lib/api";
import { ROLE_HINT, ROLE_LABEL } from "@/lib/roles";
import { useSettings, type Role } from "@/lib/settings";
import { useApi } from "@/lib/useApi";

export default function SettingsPage() {
  const { s, set, meta } = useSettings();
  const integ = useApi(() => api.integrations(), []);
  return (
    <>
      <PageHeader title="Settings" subtitle="Saved in this browser on this computer." />
      <div className="grid gap-5 lg:grid-cols-2">
        <Section title="Your view">
          <div className="flex flex-col gap-4">
            <div className="flex flex-col gap-1.5"><Label htmlFor="set-role">I am a</Label>
              <Select value={s.role} onValueChange={(v) => set({ role: v as Role })}>
                <SelectTrigger id="set-role"><SelectValue /></SelectTrigger>
                <SelectContent>{(Object.keys(ROLE_LABEL) as Role[]).map((r) => <SelectItem key={r} value={r}>{ROLE_LABEL[r]}</SelectItem>)}</SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">{ROLE_HINT[s.role]}</p>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="flex flex-col gap-1.5"><Label htmlFor="set-units">Units</Label>
                <Select value={s.units} onValueChange={(v) => set({ units: v as "metric" | "us" })}>
                  <SelectTrigger id="set-units"><SelectValue /></SelectTrigger>
                  <SelectContent><SelectItem value="us">US (bu/acre, in)</SelectItem><SelectItem value="metric">Metric (t/ha, mm)</SelectItem></SelectContent>
                </Select></div>
              <div className="flex flex-col gap-1.5"><Label htmlFor="set-temp">Temperature</Label>
                <Select value={s.temp} onValueChange={(v) => set({ temp: v as "C" | "F" })}>
                  <SelectTrigger id="set-temp"><SelectValue /></SelectTrigger>
                  <SelectContent><SelectItem value="F">Fahrenheit</SelectItem><SelectItem value="C">Celsius</SelectItem></SelectContent>
                </Select></div>
            </div>
            {s.role !== "farmer" && (
              <div className="flex flex-col gap-1.5"><Label htmlFor="set-method">Prediction interval method</Label>
                <Select value={s.method} onValueChange={(v) => set({ method: v })}>
                  <SelectTrigger id="set-method"><SelectValue /></SelectTrigger>
                  <SelectContent>{Object.entries(meta?.methods ?? {}).map(([k, v]) => <SelectItem key={k} value={k}>{v}</SelectItem>)}</SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">Group-conditional (default) calibrates intervals separately for Normal, Moderate and Extreme compound-exposure seasons. All methods share the same central estimate.</p>
              </div>
            )}
          </div>
        </Section>
        <Section title="My farm">
          <div className="flex flex-col gap-4">
            <div className="flex flex-col gap-1.5"><Label htmlFor="farm-name">Farm name (optional)</Label><Input id="farm-name" value={s.farmName} onChange={(e) => set({ farmName: e.target.value })} /></div>
            <div className="flex flex-col gap-1.5"><Label htmlFor="farm-county2">County</Label><CountyPicker id="farm-county2" value={s.fips} onChange={(f) => set({ fips: f, farmCountySet: true })} /></div>
            <div className="flex flex-col gap-1.5"><Label htmlFor="farm-acres2">Maize acres (optional)</Label>
              <Input id="farm-acres2" type="number" min={0} value={s.farmAcres ?? ""} onChange={(e) => set({ farmAcres: e.target.value ? Number(e.target.value) : null })} />
              <p className="text-xs text-muted-foreground">Used only to multiply the county yield estimate into farm production. The platform has no field-level model.</p></div>
          </div>
        </Section>
      </div>

      <Section title="Data sources" description={meta ? `Data store built ${new Date(meta.data_built_utc).toLocaleString()}. ${meta.framing}` : undefined}>
        <div className="overflow-x-auto"><Table>
          <TableHeader><TableRow><TableHead>Source</TableHead><TableHead>File</TableHead><TableHead>Modified</TableHead><TableHead>SHA-256</TableHead></TableRow></TableHeader>
          <TableBody>{Object.entries(meta?.sources ?? {}).map(([k, v]) => (
            <TableRow key={k}><TableCell>{k.replace(/_/g, " ")}</TableCell><TableCell className="max-w-xs truncate font-mono text-xs">{v.path}</TableCell><TableCell className="text-xs">{new Date(v.modified_utc).toLocaleDateString()}</TableCell><TableCell className="font-mono text-xs">{v.sha256}…</TableCell></TableRow>))}
          </TableBody></Table></div>
      </Section>

      <Section title="Planned integrations" description="Not operational. Shown so you know what is coming and what each needs; no placeholder data is displayed anywhere.">
        <div className="grid gap-3 md:grid-cols-2">
          {(integ.data ?? []).map((i) => (
            <div key={i.id} className="flex flex-col gap-2 rounded-lg border border-dashed p-4">
              <div className="flex items-center justify-between gap-2"><p className="font-semibold">{i.name}</p><Badge variant="outline">Planned</Badge></div>
              <p className="text-sm"><span className="text-muted-foreground">Would provide:</span> {i.products.join(", ")}</p>
              <p className="text-sm"><span className="text-muted-foreground">Candidate sources:</span> {i.candidates.join("; ")}</p>
              <p className="text-sm"><span className="text-muted-foreground">Still needed:</span> {i.needs.join("; ")}</p>
              <p className="font-mono text-xs text-muted-foreground">{i.interface}</p>
            </div>
          ))}
        </div>
      </Section>
    </>
  );
}
