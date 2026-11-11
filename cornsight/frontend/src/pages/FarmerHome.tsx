import { ArrowRight, MapPin, Tractor } from "lucide-react";
import { Link } from "react-router-dom";
import { AlertList, ConditionsTable, IntervalBar, RiskCards, YieldCards } from "@/components/blocks";
import { YieldHistoryChart } from "@/components/charts";
import { ErrorState, HistoricalNote, Loading, PageHeader, Section, Unavailable } from "@/components/common";
import { CountyPicker } from "@/components/pickers";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api } from "@/lib/api";
import { n, yld, yldRange } from "@/lib/format";
import { useSettings } from "@/lib/settings";
import { useApi } from "@/lib/useApi";

export default function FarmerHome() {
  const { s, set, season, countyName } = useSettings();
  const prof = useApi(() => api.profile(s.fips, season, s.method, s.farmAcres), [s.fips, season, s.method, s.farmAcres]);
  const ser = useApi(() => api.series(s.fips, s.method), [s.fips, s.method]);
  const p = prof.data;

  return (
    <>
      <PageHeader title={s.farmName ? s.farmName : "My farm"} subtitle={`${countyName(s.fips)} · maize · season ${season}`} />

      {!s.farmCountySet && (
        <Section title="Set up your farm" description="Choose the county your fields are in. Acreage is optional and only used to turn yield into production. This is saved on this computer only.">
          <div className="grid gap-3 sm:grid-cols-[1fr_10rem_auto] sm:items-end">
            <div className="flex flex-col gap-1.5"><Label htmlFor="farm-county">County</Label>
              <CountyPicker id="farm-county" value={s.fips} onChange={(f) => set({ fips: f, farmCountySet: true })} placeholder="Type your county, e.g. Story, IA" /></div>
            <div className="flex flex-col gap-1.5"><Label htmlFor="farm-acres">Maize acres</Label>
              <Input id="farm-acres" type="number" min={0} value={s.farmAcres ?? ""} onChange={(e) => set({ farmAcres: e.target.value ? Number(e.target.value) : null })} /></div>
            <Button variant="secondary" onClick={() => set({ farmCountySet: true })}>Keep {countyName(s.fips)}</Button>
          </div>
        </Section>
      )}

      <HistoricalNote season={season}>
        This page describes the {season} season after it finished. The yield estimate was made by a model that had not seen that season's yield. There is no forecast for the current season yet.
      </HistoricalNote>

      {prof.loading && !p && <Loading />}
      {prof.error && <ErrorState message={prof.error} />}
      {p && (
        <>
          <Section title="Farm and crop overview">
            <dl className="grid grid-cols-2 gap-4 text-sm md:grid-cols-4">
              <div><dt className="text-muted-foreground">Location</dt><dd className="flex items-center gap-1 font-medium"><MapPin className="size-3.5" />{p.county.label}</dd></div>
              <div><dt className="text-muted-foreground">Crop and season</dt><dd className="font-medium">Maize · 1 Apr–31 Oct {season}</dd></div>
              <div><dt className="text-muted-foreground">Farm acreage</dt><dd className="font-medium">{s.farmAcres ? `${n(s.farmAcres, 0)} acres` : <Link className="text-primary underline" to="/settings">Add in settings</Link>}</dd></div>
              <div><dt className="text-muted-foreground">County yield records</dt><dd className="font-medium">{p.county.yield_years ? `${p.county.yield_seasons} seasons, ${p.county.yield_years.join("–")}` : "None"}</dd></div>
            </dl>
          </Section>

          <Section title="What yield to expect" description="The model's estimate for your county, with the range it could reasonably fall in.">
            <YieldCards p={p} units={s.units} farmAcres={s.farmAcres} />
            {p.yield.estimate ? (
              <div className="mt-4 rounded-lg bg-muted/50 p-4">
                <p className="text-sm">
                  For {p.county.county} County the model estimated <span className="font-semibold">{yld(p.yield.estimate.t_ha, s.units)}</span>.
                  The yield could reasonably have been anywhere from <span className="font-semibold">{yldRange(p.yield.estimate.lower_t_ha, p.yield.estimate.upper_t_ha, s.units)}</span>.
                  {" "}This range shows how uncertain the estimate is; it is not a guarantee.
                </p>
                <IntervalBar p={p.yield} units={s.units} />
              </div>
            ) : (
              <div className="mt-4"><Unavailable title="No model estimate for this season">{p.yield.unavailable_reason}</Unavailable></div>
            )}
          </Section>

          <Section title="Weather risks this season" description="How this season compared with your county's own history since 1985.">
            <RiskCards p={p} units={s.units} tempUnit={s.temp} simple />
          </Section>

          <Section title="Important alerts" description="Each alert describes the finished season and what to keep in mind.">
            <AlertList alerts={p.alerts} empty={`No alerts for ${p.county.county} County in ${season}. Conditions stayed within the usual range for this county.`} />
          </Section>

          <div className="grid gap-5 lg:grid-cols-2">
            <Section title="Growing conditions" description="Season totals and averages compared with the county's usual values.">
              <ConditionsTable p={p} units={s.units} tempUnit={s.temp} compact />
            </Section>
            <Section title="What to consider" actions={<Button asChild variant="ghost" size="sm"><Link to="/recommendations">All guidance <ArrowRight /></Link></Button>}>
              <ul className="flex flex-col gap-3 text-sm">
                {p.recommendations.slice(0, 3).map((r) => (
                  <li key={r.topic} className="flex gap-2"><Tractor className="mt-0.5 size-4 shrink-0 text-primary" />
                    <div><p className="font-semibold">{r.topic}</p><p className="text-muted-foreground">{r.consider[0]}</p></div></li>
                ))}
              </ul>
            </Section>
          </div>

          <Section title="Your county's yield over the years" description="Observed yields from USDA surveys, and model estimates with their range since 2008.">
            {ser.data ? <YieldHistoryChart rows={ser.data.rows} units={s.units} from={2000} highlight={season} /> : <Loading />}
          </Section>
        </>
      )}
    </>
  );
}
