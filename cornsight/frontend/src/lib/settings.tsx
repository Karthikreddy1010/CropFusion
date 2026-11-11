/* App-wide context: role, units, selected county and season. Stored in the browser of this machine only. */
import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api, type CountyListItem, type Meta } from "./api";
import type { TempUnit, Units } from "./format";

export type Role = "farmer" | "agronomist" | "industry" | "cooperative";

export interface Settings {
  role: Role;
  units: Units;
  temp: TempUnit;
  fips: string;
  farmCountySet: boolean;
  season: number | null; // null = latest modelled season
  method: string;
  farmName: string;
  farmAcres: number | null;
  watchlist: string[];
}

const DEFAULTS: Settings = {
  role: "farmer", units: "us", temp: "F", fips: "19169", farmCountySet: false, season: null,
  method: "group_conditional_cp", farmName: "", farmAcres: null, watchlist: [],
};
const KEY = "cornsight.settings.v1";

function load(): Settings {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? { ...DEFAULTS, ...JSON.parse(raw) } : DEFAULTS;
  } catch {
    return DEFAULTS;
  }
}

interface Ctx {
  s: Settings;
  set: (patch: Partial<Settings>) => void;
  meta: Meta | null;
  metaError: string | null;
  counties: CountyListItem[];
  season: number; // resolved season
  countyName: (fips: string) => string;
}

const SettingsCtx = createContext<Ctx | null>(null);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [s, setS] = useState<Settings>(load);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [metaError, setMetaError] = useState<string | null>(null);
  const [counties, setCounties] = useState<CountyListItem[]>([]);

  useEffect(() => {
    api.meta().then(setMeta).catch((e) => setMetaError(e.message));
    api.counties().then(setCounties).catch(() => setCounties([]));
  }, []);

  useEffect(() => {
    try { localStorage.setItem(KEY, JSON.stringify(s)); } catch { /* storage unavailable */ }
  }, [s]);

  const value = useMemo<Ctx>(() => {
    const byFips = new Map(counties.map((c) => [c.fips, c]));
    return {
      s,
      set: (patch) => setS((p) => ({ ...p, ...patch })),
      meta,
      metaError,
      counties,
      season: s.season ?? meta?.default_season ?? 2023,
      countyName: (f) => {
        const c = byFips.get(f);
        return c ? `${c.county} County, ${c.state}` : f;
      },
    };
  }, [s, meta, metaError, counties]);

  return <SettingsCtx.Provider value={value}>{children}</SettingsCtx.Provider>;
}

export function useSettings(): Ctx {
  const c = useContext(SettingsCtx);
  if (!c) throw new Error("useSettings outside provider");
  return c;
}
