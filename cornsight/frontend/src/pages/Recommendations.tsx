import { AlertList, RecommendationList } from "@/components/blocks";
import { ErrorState, HistoricalNote, Loading, PageHeader, Section, Unavailable } from "@/components/common";
import { api } from "@/lib/api";
import { useSettings } from "@/lib/settings";
import { useApi } from "@/lib/useApi";

export default function Recommendations() {
  const { s, season, countyName } = useSettings();
  const prof = useApi(() => api.profile(s.fips, season, s.method, s.farmAcres), [s.fips, season, s.method, s.farmAcres]);
  return (
    <>
      <PageHeader title="Agricultural recommendations" subtitle={`Preparedness topics for ${countyName(s.fips)}, based on the ${season} season assessment.`} />
      <HistoricalNote season={season}>Guidance is drawn from how this past season played out and points to public services for current conditions. It is general awareness, not a field prescription.</HistoricalNote>
      {prof.error && <ErrorState message={prof.error} />}
      {!prof.data ? <Loading /> : (
        <>
          <Section title="Alerts behind this guidance"><AlertList alerts={prof.data.alerts} /></Section>
          <Section title="What to consider"><RecommendationList recs={prof.data.recommendations} /></Section>
          <Section title="Not available yet">
            <Unavailable title="Irrigation, fertilizer, pest and disease recommendations">
              These need field-level data and agronomic validation that the platform does not have. They are listed as planned integrations in Settings. Talk to your extension agronomist for field decisions.
            </Unavailable>
          </Section>
        </>
      )}
    </>
  );
}
