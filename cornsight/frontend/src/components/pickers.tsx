import { Search } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useSettings } from "@/lib/settings";
import { cn } from "@/lib/utils";

export function CountyPicker({ value, onChange, className, placeholder = "Search county", id = "county-search" }: {
  value?: string; onChange: (fips: string) => void; className?: string; placeholder?: string; id?: string;
}) {
  const { counties, countyName } = useSettings();
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [hi, setHi] = useState(0);
  const box = useRef<HTMLDivElement>(null);

  const hits = useMemo(() => {
    const t = q.trim().toLowerCase().replace(/ county/, "");
    if (!t) return [];
    const [name, st] = t.split(",").map((x) => x.trim());
    return counties
      .filter((c) => c.county.toLowerCase().startsWith(name) && (!st || c.state.toLowerCase() === st))
      .slice(0, 8);
  }, [q, counties]);

  const pick = (f: string) => { onChange(f); setQ(""); setOpen(false); };

  return (
    <div ref={box} className={cn("relative", className)}>
      <Search className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
      <Input
        id={id}
        value={q}
        placeholder={value ? countyName(value) : placeholder}
        aria-label="Search county"
        aria-expanded={open && hits.length > 0}
        aria-controls={`${id}-list`}
        role="combobox"
        className="pl-8"
        onChange={(e) => { setQ(e.target.value); setOpen(true); setHi(0); }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") { setHi((h) => Math.min(h + 1, hits.length - 1)); e.preventDefault(); }
          if (e.key === "ArrowUp") { setHi((h) => Math.max(h - 1, 0)); e.preventDefault(); }
          if (e.key === "Enter" && hits[hi]) pick(hits[hi].fips);
          if (e.key === "Escape") setOpen(false);
        }}
      />
      {open && hits.length > 0 && (
        <ul id={`${id}-list`} role="listbox" className="absolute z-50 mt-1 w-full overflow-hidden rounded-md border bg-popover py-1 text-sm shadow-md">
          {hits.map((c, i) => (
            <li key={c.fips} role="option" aria-selected={i === hi}
              className={cn("flex cursor-pointer items-center justify-between px-3 py-1.5", i === hi && "bg-accent")}
              onMouseDown={(e) => { e.preventDefault(); pick(c.fips); }} onMouseEnter={() => setHi(i)}>
              <span>{c.county} County, {c.state}</span>
              {!c.has_estimates && <span className="text-xs text-muted-foreground">no model estimates</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function SeasonPicker({ className, onlyEstimates = false }: { className?: string; onlyEstimates?: boolean }) {
  const { meta, season, set } = useSettings();
  if (!meta) return null;
  const [a, b] = onlyEstimates ? meta.estimate_seasons : meta.seasons;
  const years: number[] = [];
  for (let y = b; y >= a; y--) years.push(y);
  return (
    <Select value={String(season)} onValueChange={(v) => set({ season: Number(v) === meta.default_season ? null : Number(v) })}>
      <SelectTrigger className={cn("w-[132px]", className)} aria-label="Season">
        <SelectValue />
      </SelectTrigger>
      <SelectContent className="max-h-72">
        {years.map((y) => (
          <SelectItem key={y} value={String(y)}>
            Season {y}{y < meta.estimate_seasons[0] ? " · no estimate" : ""}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

export function StatePicker({ value, onChange, allowAll = true }: { value: string; onChange: (v: string) => void; allowAll?: boolean }) {
  const { meta } = useSettings();
  return (
    <Select value={value || "ALL"} onValueChange={(v) => onChange(v === "ALL" ? "" : v)}>
      <SelectTrigger className="w-[150px]" aria-label="State"><SelectValue /></SelectTrigger>
      <SelectContent>
        {allowAll && <SelectItem value="ALL">All six states</SelectItem>}
        {Object.entries(meta?.states ?? {}).map(([k, v]) => <SelectItem key={k} value={k}>{v}</SelectItem>)}
      </SelectContent>
    </Select>
  );
}
