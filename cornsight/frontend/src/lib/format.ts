/* Unit handling and number formatting. Conversion factors match the research pipeline. */
export const T_HA_PER_BU_ACRE = 0.0627; // Paper3 master_data_loader.py
export const T_PER_BU = 0.0254012; // 56-lb bushel of shelled corn
export const MM_PER_IN = 25.4;
export const ACRES_PER_HA = 2.47105;

export type Units = "metric" | "us";
export type TempUnit = "C" | "F";

const nf = (d: number) => new Intl.NumberFormat("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

export function n(v: number | null | undefined, d = 1): string {
  return v === null || v === undefined || !Number.isFinite(v) ? "–" : nf(d).format(v);
}

export function big(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1e9) return `${n(v / 1e9, 2)} billion`;
  if (a >= 1e6) return `${n(v / 1e6, 1)} million`;
  if (a >= 1e4) return `${n(v / 1e3, 0)} thousand`;
  return n(v, 0);
}

export function pct(v: number | null | undefined, d = 0, signed = true): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  const s = nf(d).format(v);
  return `${signed && v > 0 ? "+" : ""}${s}%`;
}

export function share(v: number | null | undefined, d = 0): string {
  return v === null || v === undefined ? "–" : `${nf(d).format(100 * v)}%`;
}

/** Yield value from t/ha in the chosen unit system. */
export function yieldVal(tha: number | null | undefined, u: Units): number | null {
  if (tha === null || tha === undefined) return null;
  return u === "metric" ? tha : tha / T_HA_PER_BU_ACRE;
}
export const yieldUnit = (u: Units) => (u === "metric" ? "t/ha" : "bu/acre");
export const yieldDigits = (u: Units) => (u === "metric" ? 1 : 0);

export function yld(tha: number | null | undefined, u: Units, withUnit = true): string {
  const v = yieldVal(tha, u);
  return `${n(v, yieldDigits(u))}${withUnit ? ` ${yieldUnit(u)}` : ""}`;
}
/** Both units, primary first: "10.2 t/ha (163 bu/acre)". */
export function yldBoth(tha: number | null | undefined, u: Units): string {
  if (tha === null || tha === undefined) return "–";
  const other: Units = u === "metric" ? "us" : "metric";
  return `${yld(tha, u)} (${yld(tha, other)})`;
}
export function yldRange(lo: number | null | undefined, hi: number | null | undefined, u: Units): string {
  return `${yld(lo, u, false)}–${yld(hi, u)}`;
}

export function temp(c: number | null | undefined, t: TempUnit, d = 1): string {
  if (c === null || c === undefined) return "–";
  return t === "C" ? `${n(c, d)} °C` : `${n(c * 1.8 + 32, d)} °F`;
}
/** Temperature differences and degree-days scale by 1.8 without offset. */
export function tempDelta(c: number | null | undefined, t: TempUnit, d = 1): number | null {
  if (c === null || c === undefined) return null;
  return t === "C" ? Number(c.toFixed(d + 2)) : c * 1.8;
}
export const ddUnit = (t: TempUnit) => (t === "C" ? "°C·day" : "°F·day");

export function precip(mm: number | null | undefined, u: Units): string {
  if (mm === null || mm === undefined) return "–";
  return u === "metric" ? `${n(mm, 0)} mm` : `${n(mm / MM_PER_IN, 1)} in`;
}

export function production(bu: number | null | undefined, u: Units): string {
  if (bu === null || bu === undefined) return "–";
  return u === "metric" ? `${big(bu * T_PER_BU)} t` : `${big(bu)} bu`;
}

export function dateShort(iso: string | null): string {
  if (!iso) return "–";
  const d = new Date(iso + "T00:00:00");
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}
