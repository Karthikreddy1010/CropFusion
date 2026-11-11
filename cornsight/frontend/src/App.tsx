import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "@/components/AppShell";
import { TooltipProvider } from "@/components/ui/tooltip";
import { homeFor } from "@/lib/roles";
import { useSettings } from "@/lib/settings";
import Assistant from "@/pages/Assistant";
import ClimateRisk from "@/pages/ClimateRisk";
import CropGrowth from "@/pages/CropGrowth";
import FarmerHome from "@/pages/FarmerHome";
import GeoIntel from "@/pages/GeoIntel";
import { CountyCompare, Portfolio, ProductionRisk, RegionalProduction, SupplyChain } from "@/pages/Industry";
import Model from "@/pages/Model";
import Overview from "@/pages/Overview";
import Recommendations from "@/pages/Recommendations";
import Reports from "@/pages/Reports";
import SettingsPage from "@/pages/Settings";
import Trends from "@/pages/Trends";
import YieldIntel from "@/pages/YieldIntel";

export default function App() {
  const { s } = useSettings();
  return (
    <TooltipProvider delayDuration={150}>
      <AppShell>
        <Routes>
          <Route path="/" element={<Navigate to={homeFor(s.role)} replace />} />
          <Route path="/farm" element={<FarmerHome />} />
          <Route path="/overview" element={<Overview />} />
          <Route path="/yield" element={<YieldIntel />} />
          <Route path="/climate" element={<ClimateRisk />} />
          <Route path="/growth" element={<CropGrowth />} />
          <Route path="/map" element={<GeoIntel />} />
          <Route path="/trends" element={<Trends />} />
          <Route path="/recommendations" element={<Recommendations />} />
          <Route path="/assistant" element={<Assistant />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/model" element={<Model />} />
          <Route path="/regional" element={<RegionalProduction />} />
          <Route path="/supply-chain" element={<SupplyChain />} />
          <Route path="/compare" element={<CountyCompare />} />
          <Route path="/production-risk" element={<ProductionRisk />} />
          <Route path="/portfolio" element={<Portfolio />} />
          <Route path="*" element={<Navigate to={homeFor(s.role)} replace />} />
        </Routes>
      </AppShell>
    </TooltipProvider>
  );
}
