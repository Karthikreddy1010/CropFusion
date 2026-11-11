import { AlertTriangle, CircleHelp, Clock, Info, Loader2 } from "lucide-react";
import type { ReactNode } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { Level } from "@/lib/api";
import { cn } from "@/lib/utils";

const LEVEL_STYLE: Record<string, string> = {
  Low: "bg-[var(--risk-low-bg)] text-[oklch(0.38_0.1_150)] border-[var(--risk-low)]",
  Moderate: "bg-[var(--risk-moderate-bg)] text-[oklch(0.42_0.1_90)] border-[var(--risk-moderate)]",
  High: "bg-[var(--risk-high-bg)] text-[oklch(0.45_0.14_50)] border-[var(--risk-high)]",
  "Very High": "bg-[var(--risk-veryhigh-bg)] text-[oklch(0.45_0.18_28)] border-[var(--risk-veryhigh)]",
};
export const LEVEL_COLOR: Record<string, string> = {
  Low: "var(--risk-low)", Moderate: "var(--risk-moderate)", High: "var(--risk-high)", "Very High": "var(--risk-veryhigh)",
};

export function RiskBadge({ level, basis, className }: { level: Level | undefined; basis?: string; className?: string }) {
  if (!level) return <span className="text-xs text-muted-foreground">Not available</span>;
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-semibold", LEVEL_STYLE[level], className)}>
      <span className="size-1.5 rounded-full" style={{ background: LEVEL_COLOR[level] }} />
      {level}
      {basis === "proposed" && <span className="font-normal opacity-75">· proposed scale</span>}
    </span>
  );
}

export function Hint({ children }: { children: ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button type="button" className="inline-flex text-muted-foreground hover:text-foreground" aria-label="More information">
          <CircleHelp className="size-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs text-xs leading-relaxed">{children}</TooltipContent>
    </Tooltip>
  );
}

export function Stat({ label, value, sub, hint, accent, children }: {
  label: string; value: ReactNode; sub?: ReactNode; hint?: ReactNode; accent?: string; children?: ReactNode;
}) {
  return (
    <Card className="gap-1 py-4">
      <CardContent className="flex flex-col gap-1 px-4">
        <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          {accent && <span className="h-3 w-1 rounded-full" style={{ background: accent }} />}
          {label}
          {hint && <Hint>{hint}</Hint>}
        </div>
        <div className="num text-2xl font-semibold leading-tight">{value}</div>
        {sub && <div className="num text-sm text-muted-foreground">{sub}</div>}
        {children}
      </CardContent>
    </Card>
  );
}

export function Section({ title, description, actions, children, className }: {
  title: string; description?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <Card className={cn("min-w-0 gap-4", className)}>
      <CardHeader className="flex flex-row flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <CardTitle className="font-[family-name:var(--font-display)] text-lg">{title}</CardTitle>
          {description && <CardDescription className="mt-1 max-w-3xl">{description}</CardDescription>}
        </div>
        {actions}
      </CardHeader>
      <CardContent className="min-w-0">{children}</CardContent>
    </Card>
  );
}

export function PageHeader({ title, subtitle, children }: { title: string; subtitle?: ReactNode; children?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-3">
      <div className="min-w-0">
        <h1 className="text-2xl font-semibold sm:text-3xl">{title}</h1>
        {subtitle && <p className="mt-1 max-w-3xl text-sm text-muted-foreground">{subtitle}</p>}
      </div>
      {children}
    </div>
  );
}

export function Loading({ label = "Loading data" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 py-10 text-sm text-muted-foreground" role="status">
      <Loader2 className="size-4 animate-spin" /> {label}…
    </div>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">
      <AlertTriangle className="mt-0.5 size-4 shrink-0" />
      <div>
        <p className="font-semibold">Could not load this data.</p>
        <p>{message}</p>
        {message.includes("fetch") || message.includes("Failed") ? <p className="mt-1">Check that the CornSight backend is running on port 8010.</p> : null}
      </div>
    </div>
  );
}

export function Unavailable({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="rounded-md border border-dashed bg-muted/40 p-4 text-sm">
      <p className="font-semibold text-foreground">{title}</p>
      <div className="mt-1 text-muted-foreground">{children}</div>
    </div>
  );
}

export function HistoricalNote({ season, children }: { season: number; children?: ReactNode }) {
  return (
    <div className="flex items-start gap-2 rounded-md border border-[var(--risk-moderate)]/50 bg-[var(--risk-moderate-bg)] px-3 py-2 text-sm text-[oklch(0.38_0.07_90)]">
      <Clock className="mt-0.5 size-4 shrink-0" />
      <p>
        <span className="font-semibold">Historical assessment, season {season}.</span>{" "}
        {children ?? "Model estimates are reconstructions of a completed season made without that season's yield. They are not forecasts. Current-season forecasts are a planned integration."}
      </p>
    </div>
  );
}

export function Note({ children }: { children: ReactNode }) {
  return (
    <p className="flex items-start gap-1.5 text-xs leading-relaxed text-muted-foreground">
      <Info className="mt-0.5 size-3.5 shrink-0" />
      <span>{children}</span>
    </p>
  );
}
