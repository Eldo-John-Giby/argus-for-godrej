import { Routes, Route, NavLink } from "react-router-dom";
import { AlertCenterProvider } from "./lib/alerts";
import AlertToaster from "./components/AlertToaster";
import LiveFeed from "./pages/LiveFeed";
import IncidentReplay from "./pages/IncidentReplay";
import Dashboard from "./pages/Dashboard";
import Heatmap from "./pages/Heatmap";
import Scorecards from "./pages/Scorecards";
import Assistant from "./pages/Assistant";
import { Eye, AlertTriangle, BarChart3, MapPin, Users, MessageSquare } from "lucide-react";

const navItems = [
  { to: "/", label: "Live Feed", icon: Eye },
  { to: "/incidents", label: "Incidents", icon: AlertTriangle },
  { to: "/dashboard", label: "Dashboard", icon: BarChart3 },
  { to: "/heatmap", label: "Heatmap", icon: MapPin },
  { to: "/scorecards", label: "Scorecards", icon: Users },
  { to: "/assistant", label: "Assistant", icon: MessageSquare },
];

export default function App() {
  return (
    <AlertCenterProvider>
      <div className="min-h-screen bg-argus-dark">
        {/* Top Nav */}
      <nav className="bg-argus-card border-b border-argus-border px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-argus-accent rounded-lg flex items-center justify-center">
            <Eye className="w-5 h-5 text-white" />
          </div>
          <h1 className="text-xl font-bold text-white">
            ARGUS <span className="text-sm font-normal text-gray-400">— AI Field Intelligence</span>
          </h1>
        </div>

        <div className="flex items-center gap-1">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-argus-accent text-white"
                    : "text-gray-400 hover:text-white hover:bg-argus-border"
                }`
              }
            >
              <item.icon className="w-4 h-4" />
              {item.label}
            </NavLink>
          ))}
        </div>
      </nav>

        {/* Main Content */}
        <main className="p-6">
          <Routes>
            <Route path="/" element={<LiveFeed />} />
            <Route path="/incidents" element={<IncidentReplay />} />
            <Route path="/incidents/:eventId" element={<IncidentReplay />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/heatmap" element={<Heatmap />} />
            <Route path="/scorecards" element={<Scorecards />} />
            <Route path="/assistant" element={<Assistant />} />
          </Routes>
        </main>

        {/* Real-time alert toasts (sound + multilingual voice) */}
        <AlertToaster />
      </div>
    </AlertCenterProvider>
  );
}
