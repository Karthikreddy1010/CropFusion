/* Thin typed client for the CornSight API. Every call goes to the local backend (/api, proxied by Vite). */

export type Level = "Low" | "Moderate" | "High" | "Very High" | null;

export interface Meta {
  product: string;
  crop: string;
  states: Record<string, string>;
  seasons: [number, number];
  estimate_seasons: [number, number];
  default_season: number;
  methods: Record<string, string>;
  default_method: string;
  counties: number;
  counties_with_estimates: number;
  data_built_utc: string;
  constants: Record<string, number | string>;
  sources: Record<string, { path: string; modified_utc: string; sha256: string }>;
  framing: string;
  assistant_configured: boolean;
}

export interface CountyListItem { fips: string; county: string; state: string; has_estimates: boolean }

export interface Estimate {
  t_ha: number; bu_acre: number; lower_t_ha: number; upper_t_ha: number; lower_bu_acre: number; upper_bu_acre: number;
  width_t_ha: number; nominal_coverage: number; exposure_class: string; inside_interval: boolean | null;
  error_t_ha: number | null; explanation: string; interval_explanation: string; model: string;
  vs_average_pct?: number; lower_vs_average_pct?: number;
  production_bu?: number; production_bu_lower?: number; production_bu_upper?: number;
  farm_production_bu?: number; farm_production_bu_lower?: number; farm_production_bu_upper?: number;
  previous_season_estimate_t_ha?: number; change_vs_previous_estimate_pct?: number;
}

export interface YieldBlock {
  season: number; method: string; method_label: string;
  observed: { t_ha: number | null; bu_acre: number | null; source: string };
  prior5_average: { t_ha: number | null; bu_acre: number | null; definition: string };
  harvested_acres: number | null; production_bu_observed: number | null;
  observed_vs_average_pct?: number;
  estimate: Estimate | null; available: boolean; unavailable_reason?: string;
}

export interface RiskItem {
  level: Level; basis: string; rule?: string; value: number | null; unit?: string; percentile?: number | null;
  [k: string]: unknown;
}

export interface Stage {
  stage: string; label: string; start: string | null; end: string | null; days: number | null;
  tmax_mean_c: number | null; days_above_35c: number | null; ppt_mm: number | null; et0_mm: number | null;
  water_balance_mm: number | null; vpd_kpa: number | null; spei30_mean: number | null; spei30_min: number | null;
  spi1_min: number | null; cdhw_days: number | null; cdhw_episodes: number | null; cdhw_severity: number | null;
}

export interface Alert {
  id: string; severity: string; title: string; what: string; where: string; period: string;
  impact: string; consider: string; basis: string; threshold: string;
}

export interface Recommendation {
  topic: string; why: string; consider: string[]; sources: { name: string; url: string | null }[]; note: string;
}

export interface Reliability {
  method: string; method_label: string; nominal_coverage: number; seasons: [number, number]; note: string;
  by_class: { exposure_class: string; rows: number; inside_rate: number | null; below_lower_rate: number | null; mean_width_t_ha: number | null }[];
  county?: { fips: string; seasons: number; inside: number; below_lower: number };
}

export interface CountyInfo {
  fips: string; county: string; state: string; state_name: string; label: string; lat: number; lon: number;
  yield_years: [number, number] | null; yield_seasons: number; has_estimates: boolean;
  soil: Record<string, { value: number | null; unit: string }>;
  terrain: { elevation_m: number | null; elevation_sd_m: number | null; slope_deg: number | null };
  land_cover: Record<string, number | null>;
}

export interface Profile {
  county: CountyInfo; season: number; method: string; framing: string;
  yield: YieldBlock;
  risk: Record<"drought" | "heat" | "rainfall" | "compound" | "storms", RiskItem>;
  conditions: { available: boolean; season: number; period: string; items: { key: string; label: string; unit: string; value: number | null; county_mean_1985_2023: number | null; difference: number | null; percentile: number | null }[] };
  stages: { available: boolean; season: number; note: string; stages: Stage[]; most_stressed: Record<string, { stage: string; value: number } | null> };
  alerts: Alert[];
  recommendations: Recommendation[];
  reliability: Reliability;
}

export interface SeriesRow {
  year: number; observed_t_ha: number | null; prior5_avg_t_ha: number | null; pred_t_ha: number | null;
  lower_t_ha: number | null; upper_t_ha: number | null; exposure_class: string | null;
  cdhw_days: number | null; drought_days: number | null; heat_days: number | null; ppt_mm: number | null;
  gdd: number | null; kdd30: number | null; tmax_mean_c: number | null; spei30_mean: number | null;
  et0_mm: number | null; vpd_kpa: number | null; water_balance_mm: number | null;
}

export interface StateSummary {
  state: string; state_name: string; season: number; counties: number; counties_with_yield: number; counties_with_estimate: number;
  observed_t_ha_acre_weighted: number | null; prior5_avg_t_ha_acre_weighted: number | null;
  estimate_t_ha_acre_weighted: number | null; lower_t_ha_acre_weighted: number | null; upper_t_ha_acre_weighted: number | null;
  production_bu_observed: number | null; production_bu_estimated: number | null; harvested_acres: number | null;
  counties_compound_high: number; counties_compound_moderate: number; mean_drought_days: number | null; mean_heat_days: number | null;
  definition: string;
}

export interface Ranking {
  season: number; metric: string; label: string; unit: string; order: string; counties_ranked: number;
  rows: { rank: number; fips: string; county: string; state: string; value: number | null; estimate_t_ha: number | null; lower_t_ha: number | null; upper_t_ha: number | null; observed_t_ha: number | null; cdhw_days: number | null; drought_days: number | null; heat_days: number | null }[];
}

export interface Layer {
  layer: string; label: string; unit: string; scale: "sequential" | "diverging" | "classes"; seasonal: boolean;
  season: number | null; values: Record<string, number>; counties_with_value: number; note: string | null;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) { super(message); this.status = status; }
}

async function get<T>(path: string, params?: Record<string, string | number | boolean | null | undefined>): Promise<T> {
  const q = new URLSearchParams();
  Object.entries(params ?? {}).forEach(([k, v]) => { if (v !== null && v !== undefined && v !== "") q.set(k, String(v)); });
  const r = await fetch(`/api${path}${q.toString() ? `?${q}` : ""}`);
  if (!r.ok) {
    let msg = `Request failed (${r.status})`;
    try { msg = (await r.json()).detail ?? msg; } catch { /* keep default */ }
    throw new ApiError(r.status, msg);
  }
  return r.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`/api${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) {
    let msg = `Request failed (${r.status})`;
    try { msg = (await r.json()).detail ?? msg; } catch { /* keep default */ }
    throw new ApiError(r.status, msg);
  }
  return r.json() as Promise<T>;
}

export const api = {
  meta: () => get<Meta>("/meta"),
  counties: () => get<CountyListItem[]>("/counties"),
  profile: (fips: string, season: number, method: string, farm_acres?: number | null) =>
    get<Profile>(`/county/${fips}/profile`, { season, method, farm_acres }),
  series: (fips: string, method: string) => get<{ fips: string; rows: SeriesRow[] }>(`/county/${fips}/series`, { method }),
  stageHistory: (fips: string) => get<{ fips: string; note: string; rows: Record<string, number | null>[] }>(`/county/${fips}/stages/history`),
  compare: (fips: string[], season: number, method: string) =>
    get<{ season: number; method: string; counties: { county: CountyInfo; yield: YieldBlock; risk: Record<string, { level: Level; value: number | null; unit: string }>; series: SeriesRow[] }[] }>("/compare", { fips: fips.join(","), season, method }),
  states: (season: number, method: string) => get<{ season: number; states: StateSummary[] }>("/states/summary", { season, method }),
  stateSeries: (method: string) => get<{ rows: { state: string; year: number; observed_t_ha: number | null; estimate_t_ha: number | null; lower_t_ha: number | null; upper_t_ha: number | null; production_bu: number | null; compound_days_mean: number | null }[] }>("/states/series", { method }),
  rankings: (metric: string, season: number, method: string, state?: string, ascending = false, limit = 25) =>
    get<Ranking>("/rankings", { metric, season, method, state, ascending, limit }),
  rankMetrics: () => get<Record<string, { label: string; unit: string }>>("/rank-metrics"),
  compoundYield: (state?: string) => get<{ state: string | null; seasons: [number, number]; note: string; classes: { class: string; county_seasons: number; median_dev_pct: number; p25_dev_pct: number; p75_dev_pct: number; share_below_avg_10pct: number }[] }>("/climate/compound-yield", { state }),
  reliability: (method: string, fips?: string) => get<Reliability>("/model/reliability", { method, fips }),
  attribution: () => get<{ note: string; rows: { Rank: number; Feature: string; Mean_Attribution: number }[] }>("/model/attribution"),
  geoCounties: () => get<GeoJSON.FeatureCollection>("/geo/counties"),
  geoStates: () => get<GeoJSON.FeatureCollection>("/geo/states"),
  layers: () => get<Record<string, { label: string; unit: string; scale: string; seasonal: boolean }>>("/map/layers"),
  layer: (layer: string, season: number, method: string) => get<Layer>("/map/layer", { layer, season, method }),
  integrations: () => get<{ id: string; name: string; status: string; candidates: string[]; products: string[]; needs: string[]; interface: string; feeds: string[] }[]>("/integrations"),
  assistantStatus: () => get<{ configured: boolean; model: string; how_to_enable: string; tools: { name: string; description: string }[] }>("/assistant/status"),
  chat: (messages: { role: string; content: string }[]) => post<{ answer: string; tool_calls: { tool: string; input: Record<string, unknown>; ok: boolean }[]; stop_reason: string }>("/assistant/chat", { messages }),
  lookup: (tool: string, args: Record<string, unknown>) => post<Record<string, unknown>>("/assistant/lookup", { tool, args }),
};

export const reportUrl = {
  county: (fips: string, method: string) => `/api/reports/county/${fips}.csv?method=${method}`,
  rankings: (metric: string, season: number, method: string, state?: string) =>
    `/api/reports/rankings.csv?metric=${metric}&season=${season}&method=${method}${state ? `&state=${state}` : ""}`,
  states: (season: number, method: string) => `/api/reports/states.csv?season=${season}&method=${method}`,
  region: (season: number, method: string) => `/api/reports/region.csv?season=${season}&method=${method}`,
};
