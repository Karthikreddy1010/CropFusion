import {
  BarChart3, Bot, CloudSun, Factory, FileDown, GitCompare, Home, LayoutDashboard, LineChart, Lightbulb,
  Map, Network, Settings, ShieldAlert, Sprout, TrendingUp, Users, type LucideIcon,
} from "lucide-react";
import type { Role } from "./settings";

export interface NavItem { path: string; label: string; icon: LucideIcon; group?: string }

export const ROLE_LABEL: Record<Role, string> = {
  farmer: "Farmer", agronomist: "Agronomist", industry: "Agricultural industry", cooperative: "Cooperative",
};
export const ROLE_HINT: Record<Role, string> = {
  farmer: "Your county's yield, risks and what to watch, in plain terms.",
  agronomist: "Stage-by-stage conditions, drought and heat indices, model detail.",
  industry: "Regional production, county risk rankings and comparisons.",
  cooperative: "Several counties at once: portfolio, rankings, reports.",
};

const ALL: Record<string, NavItem> = {
  farm: { path: "/farm", label: "My farm", icon: Home },
  overview: { path: "/overview", label: "Agricultural overview", icon: LayoutDashboard },
  yield: { path: "/yield", label: "Crop yield intelligence", icon: LineChart },
  climate: { path: "/climate", label: "Climate risk", icon: CloudSun },
  growth: { path: "/growth", label: "Crop growth analysis", icon: Sprout },
  map: { path: "/map", label: "Geographic intelligence", icon: Map },
  trends: { path: "/trends", label: "Historical trends", icon: TrendingUp },
  recs: { path: "/recommendations", label: "Recommendations", icon: Lightbulb },
  assistant: { path: "/assistant", label: "AI assistant", icon: Bot },
  reports: { path: "/reports", label: "Reports and exports", icon: FileDown },
  settings: { path: "/settings", label: "Settings", icon: Settings },
  regional: { path: "/regional", label: "Regional production", icon: Factory, group: "Industry" },
  supply: { path: "/supply-chain", label: "Supply chain intelligence", icon: Network, group: "Industry" },
  compare: { path: "/compare", label: "County comparison", icon: GitCompare, group: "Industry" },
  risk: { path: "/production-risk", label: "Production risk", icon: ShieldAlert, group: "Industry" },
  portfolio: { path: "/portfolio", label: "Portfolio", icon: Users, group: "Cooperative" },
  model: { path: "/model", label: "Model and uncertainty", icon: BarChart3 },
};

const NAV: Record<Role, string[]> = {
  farmer: ["farm", "yield", "climate", "trends", "recs", "assistant", "reports", "settings"],
  agronomist: ["overview", "yield", "climate", "growth", "map", "trends", "model", "recs", "assistant", "reports", "settings"],
  industry: ["overview", "yield", "climate", "map", "trends", "regional", "supply", "compare", "risk", "assistant", "reports", "settings"],
  cooperative: ["overview", "portfolio", "yield", "climate", "map", "trends", "compare", "risk", "recs", "assistant", "reports", "settings"],
};

export const navFor = (r: Role): NavItem[] => NAV[r].map((k) => ALL[k]);
export const homeFor = (r: Role) => (r === "farmer" ? "/farm" : r === "cooperative" ? "/portfolio" : "/overview");
