import { geoAlbers, geoPath } from "d3-geo";
import { scaleQuantile, scaleThreshold } from "d3-scale";
import {
  interpolateBlues, interpolateBrBG, interpolateGreens, interpolateOrRd, interpolatePurples, interpolateYlGn, interpolateYlOrBr,
} from "d3-scale-chromatic";
import { useEffect, useMemo, useState } from "react";
import { api, type Layer } from "@/lib/api";
import { n } from "@/lib/format";
import { LEVEL_COLOR } from "./common";

let GEO: Promise<[GeoJSON.FeatureCollection, GeoJSON.FeatureCollection]> | null = null;
const loadGeo = () => (GEO ??= Promise.all([api.geoCounties(), api.geoStates()]));

const SCHEME: Record<string, (t: number) => string> = {
  estimate: interpolateYlGn, observed: interpolateYlGn, width: interpolatePurples, drought: interpolateYlOrBr,
  heat: interpolateOrRd, compound_severity: interpolateOrRd, storms: interpolatePurples, ppt: interpolateBlues,
  soil_clay: interpolateYlOrBr, soil_oc: interpolateYlOrBr, soil_awc: interpolateBlues, cropland: interpolateGreens, forest: interpolateGreens,
};
const NODATA = "oklch(0.93 0.005 95)";

export interface Scale { color: (v: number) => string; legend: { color: string; label: string }[] }

export function buildScale(layer: Layer): Scale {
  const vals = Object.values(layer.values).filter((v) => Number.isFinite(v));
  if (layer.scale === "classes") {
    const c = (v: number) => (v >= 3 ? LEVEL_COLOR.High : v >= 1 ? LEVEL_COLOR.Moderate : LEVEL_COLOR.Low);
    return { color: c, legend: [
      { color: LEVEL_COLOR.Low, label: "Low · 0 days" }, { color: LEVEL_COLOR.Moderate, label: "Moderate · 1–2" }, { color: LEVEL_COLOR.High, label: "High · 3 or more" },
    ] };
  }
  if (layer.scale === "diverging") {
    const m = Math.max(5, Math.ceil(Math.max(...vals.map(Math.abs), 0) / 5) * 5);
    const edges = [-m * 0.6, -m * 0.3, -m * 0.1, m * 0.1, m * 0.3, m * 0.6].map((x) => Math.round(x));
    const cols = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95].map((t) => interpolateBrBG(t));
    const s = scaleThreshold<number, string>().domain(edges).range(cols);
    const labels = [`below ${edges[0]}`, ...edges.slice(0, -1).map((e, i) => `${e} to ${edges[i + 1]}`), `above ${edges[edges.length - 1]}`];
    return { color: s, legend: cols.map((c, i) => ({ color: c, label: `${labels[i]}${layer.unit === "%" ? "%" : ""}` })) };
  }
  const interp = SCHEME[layer.layer] ?? interpolateYlGn;
  const cols = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9].map(interp);
  if (vals.length < 2) return { color: () => cols[3], legend: [] };
  const allZeroHeavy = vals.filter((v) => v === 0).length > vals.length / 2;
  if (allZeroHeavy) {
    const pos = vals.filter((v) => v > 0);
    const q = scaleQuantile<string>().domain(pos.length ? pos : [1]).range(cols.slice(1));
    const th = q.quantiles();
    return {
      color: (v) => (v === 0 ? cols[0] : q(v)),
      legend: [{ color: cols[0], label: "0" }, ...cols.slice(1).map((c, i) => ({
        color: c, label: i === 0 ? `up to ${n(th[0], 0)}` : i === cols.length - 2 ? `${n(th[i - 1], 0)} +` : `${n(th[i - 1], 0)}–${n(th[i], 0)}`,
      }))],
    };
  }
  const q = scaleQuantile<string>().domain(vals).range(cols);
  const th = q.quantiles();
  const d = Math.max(...vals) < 20 ? 1 : 0;
  return {
    color: q,
    legend: cols.map((c, i) => ({
      color: c, label: i === 0 ? `below ${n(th[0], d)}` : i === cols.length - 1 ? `${n(th[i - 1], d)} +` : `${n(th[i - 1], d)}–${n(th[i], d)}`,
    })),
  };
}

export function CountyMap({ layer, selected, onSelect, height = 520, format }: {
  layer: Layer | null; selected?: string; onSelect?: (fips: string) => void; height?: number;
  format?: (v: number) => string;
}) {
  const [geo, setGeo] = useState<[GeoJSON.FeatureCollection, GeoJSON.FeatureCollection] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [hover, setHover] = useState<{ fips: string; name: string; x: number; y: number } | null>(null);
  useEffect(() => { loadGeo().then(setGeo, (e) => setErr(e.message)); }, []);

  const W = 900;
  const H = 640;
  const paths = useMemo(() => {
    if (!geo) return null;
    const proj = geoAlbers().rotate([90, 0]).fitSize([W, H], geo[0]);
    const p = geoPath(proj);
    return {
      counties: geo[0].features.map((f) => ({ fips: String(f.id), name: String(f.properties?.name ?? ""), d: p(f) ?? "" })),
      states: geo[1].features.map((f) => ({ id: String(f.id), d: p(f) ?? "" })),
    };
  }, [geo]);
  const scale = useMemo(() => (layer ? buildScale(layer) : null), [layer]);
  const fmt = format ?? ((v: number) => `${n(v, Math.abs(v) < 20 ? 1 : 0)} ${layer?.unit ?? ""}`);

  if (err) return <p className="text-sm text-destructive">Map geometry could not be loaded: {err}</p>;
  if (!paths) return <div className="grid place-items-center text-sm text-muted-foreground" style={{ height }}>Loading map…</div>;

  return (
    <div className="flex flex-col gap-3">
      <div className="relative w-full" onMouseLeave={() => setHover(null)}>
        <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" style={{ maxHeight: height }} role="img"
          aria-label={layer ? `Map of ${layer.label}` : "County map"}>
          <g>
            {paths.counties.map((c) => {
              const v = layer?.values[c.fips];
              const fill = v === undefined || v === null || !scale ? NODATA : scale.color(v);
              return (
                <path key={c.fips} d={c.d} fill={fill} stroke="white" strokeWidth={0.4}
                  className="cursor-pointer transition-opacity hover:opacity-80"
                  onClick={() => onSelect?.(c.fips)}
                  onMouseMove={(e) => {
                    const r = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                    setHover({ fips: c.fips, name: c.name, x: e.clientX - r.left, y: e.clientY - r.top });
                  }}>
                  <title>{c.name}</title>
                </path>
              );
            })}
          </g>
          <g pointerEvents="none">
            {paths.states.map((s) => <path key={s.id} d={s.d} fill="none" stroke="oklch(0.35 0.03 120)" strokeWidth={1.2} />)}
            {selected && paths.counties.filter((c) => c.fips === selected).map((c) =>
              <path key="sel" d={c.d} fill="none" stroke="oklch(0.2 0.03 120)" strokeWidth={2.4} />)}
          </g>
        </svg>
        {hover && (
          <div className="pointer-events-none absolute z-10 rounded-md border bg-popover px-2.5 py-1.5 text-xs shadow-md"
            style={{ left: Math.min(hover.x + 12, 9999), top: hover.y + 12 }}>
            <p className="font-semibold">{hover.name} County</p>
            <p className="num">{layer?.values[hover.fips] !== undefined ? fmt(layer.values[hover.fips]) : "No data"}</p>
          </div>
        )}
      </div>
      {scale && layer && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-xs" aria-label="Legend">
          <span className="font-semibold">{layer.label}{layer.unit && layer.scale !== "classes" ? ` (${layer.unit})` : ""}</span>
          {scale.legend.map((l) => (
            <span key={l.label} className="inline-flex items-center gap-1"><span className="h-3 w-4 rounded-sm" style={{ background: l.color }} />{l.label}</span>
          ))}
          <span className="inline-flex items-center gap-1"><span className="h-3 w-4 rounded-sm border" style={{ background: NODATA }} />no data</span>
        </div>
      )}
    </div>
  );
}
