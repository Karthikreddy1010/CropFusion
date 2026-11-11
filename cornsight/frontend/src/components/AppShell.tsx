import { Database, Menu, Wheat } from "lucide-react";
import { useState, type ReactNode } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Sheet, SheetContent, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { homeFor, navFor, ROLE_LABEL } from "@/lib/roles";
import { useSettings, type Role } from "@/lib/settings";
import { cn } from "@/lib/utils";
import { CountyPicker, SeasonPicker } from "./pickers";

function Nav({ onNavigate }: { onNavigate?: () => void }) {
  const { s } = useSettings();
  const items = navFor(s.role);
  let lastGroup: string | undefined;
  return (
    <nav className="flex flex-col gap-0.5 px-3" aria-label="Main">
      {items.map((it) => {
        const header = it.group && it.group !== lastGroup ? it.group : null;
        lastGroup = it.group;
        return (
          <div key={it.path}>
            {header && <p className="mt-4 mb-1 px-3 text-[11px] font-semibold uppercase tracking-wider text-sidebar-foreground/60">{header}</p>}
            <NavLink to={it.path} onClick={onNavigate}
              className={({ isActive }) => cn(
                "flex items-center gap-2.5 rounded-md px-3 py-2 text-sm text-sidebar-foreground/90 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                isActive && "bg-sidebar-accent font-semibold text-sidebar-accent-foreground",
              )}>
              <it.icon className="size-4" /> {it.label}
            </NavLink>
          </div>
        );
      })}
    </nav>
  );
}

function RoleSwitch() {
  const { s, set } = useSettings();
  const nav = useNavigate();
  return (
    <div className="px-6">
      <label htmlFor="role" className="text-[11px] font-semibold uppercase tracking-wider text-sidebar-foreground/60">View as</label>
      <Select value={s.role} onValueChange={(v) => { set({ role: v as Role }); nav(homeFor(v as Role)); }}>
        <SelectTrigger id="role" className="mt-1 w-full border-sidebar-border bg-sidebar-accent text-sidebar-accent-foreground">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {(Object.keys(ROLE_LABEL) as Role[]).map((r) => <SelectItem key={r} value={r}>{ROLE_LABEL[r]}</SelectItem>)}
        </SelectContent>
      </Select>
    </div>
  );
}

function Brand() {
  return (
    <div className="flex items-center gap-2 px-6 py-5">
      <span className="grid size-8 place-items-center rounded-md bg-sidebar-primary text-sidebar-primary-foreground"><Wheat className="size-5" /></span>
      <div className="leading-tight">
        <p className="font-[family-name:var(--font-display)] text-lg font-bold text-sidebar-accent-foreground">CornSight</p>
        <p className="text-[11px] text-sidebar-foreground/70">Corn Belt maize intelligence</p>
      </div>
    </div>
  );
}

function ContextBar() {
  const { s, set, meta, countyName, season } = useSettings();
  const built = meta ? new Date(meta.data_built_utc).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" }) : "–";
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b bg-card px-4 py-3 sm:px-6">
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded-full bg-accent px-2.5 py-0.5 text-xs font-semibold text-accent-foreground">Maize (corn)</span>
        <span className="text-sm font-semibold">{countyName(s.fips)}</span>
      </div>
      <div className="flex flex-1 flex-wrap items-center justify-end gap-2">
        <CountyPicker className="w-full sm:w-56" value={s.fips} onChange={(f) => set({ fips: f, ...(s.role === "farmer" ? { farmCountySet: true } : {}) })} id="ctx-county" />
        <SeasonPicker />
      </div>
      <div className="flex w-full flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span className="inline-flex items-center gap-1"><Database className="size-3.5" />
          Data: seasons {meta?.seasons.join("–")}, model estimates {meta?.estimate_seasons.join("–")}
        </span>
        <span>Data store built {built}</span>
        <span>Showing season {season} · historical</span>
      </div>
    </div>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const { metaError } = useSettings();
  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-64 shrink-0 flex-col overflow-y-auto bg-sidebar pb-6 lg:flex">
        <Brand />
        <RoleSwitch />
        <div className="mt-5"><Nav /></div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-2 border-b bg-sidebar px-4 py-2 lg:hidden">
          <Sheet open={open} onOpenChange={setOpen}>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon" className="text-sidebar-foreground" aria-label="Open menu"><Menu /></Button>
            </SheetTrigger>
            <SheetContent side="left" className="w-72 bg-sidebar p-0">
              <SheetTitle className="sr-only">Navigation</SheetTitle>
              <Brand />
              <RoleSwitch />
              <div className="mt-5 pb-6"><Nav onNavigate={() => setOpen(false)} /></div>
            </SheetContent>
          </Sheet>
          <span className="font-[family-name:var(--font-display)] text-lg font-bold text-sidebar-accent-foreground">CornSight</span>
        </div>
        <ContextBar />
        {metaError && (
          <div className="mx-4 mt-4 rounded-md border border-destructive/40 bg-destructive/5 p-3 text-sm text-destructive sm:mx-6">
            The CornSight backend is not reachable ({metaError}). Start it with <code>run_backend.ps1</code>.
          </div>
        )}
        <main className="mx-auto flex w-full max-w-7xl flex-1 flex-col gap-5 px-4 py-6 sm:px-6">{children}</main>
        <footer className="border-t px-6 py-3 text-xs text-muted-foreground">
          Sources: USDA NASS, PRISM AN81d, NLDAS-2, NOAA Storm Events, soil and NLCD layers, USGS 3DEP. Model: rolling-origin LightGBM with conformal 90% intervals. All seasons shown are historical.
        </footer>
      </div>
    </div>
  );
}
