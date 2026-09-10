import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { RiskBadge, TIER_COLOR } from "../components/RiskBadge";

interface BayData {
  bay_id: string;
  event_count: number;
  avg_risk: number;
  critical: number;
  high: number;
  medium: number;
  low: number;
}

// Single naming convention shared with the seeder/assistant — the heatmap
// used to say "A" while the assistant said "Bay A — Unloading Dock".
const BAY_NAMES: Record<string, string> = {
  A: "Bay A - Unloading Dock",
  B: "Bay B - Staging & Stacking",
  C: "Bay C - KD Assembly & Loading",
};

function bayName(id: string): string {
  return BAY_NAMES[id] ?? `Bay ${id}`;
}

const DEMO_BAYS: BayData[] = [
  { bay_id: "B", event_count: 39, avg_risk: 34.6, critical: 0, high: 12, medium: 27, low: 0 },
  { bay_id: "C", event_count: 29, avg_risk: 33.8, critical: 0, high: 8, medium: 21, low: 0 },
  { bay_id: "A", event_count: 12, avg_risk: 31.2, critical: 0, high: 0, medium: 12, low: 0 },
];

function worstTier(bay: BayData): string {
  if (bay.critical > 0) return "Critical";
  if (bay.high > 0) return "High";
  if (bay.medium > 0) return "Medium";
  if (bay.low > 0) return "Low";
  return "Medium";
}

/**
 * Cell color = worst open tier in the bay (hue) × event density (fill
 * opacity) — matches the design spec: bay cells glance like status, not
 * content. Density keeps two same-tier bays visually distinct.
 */
function cellStyle(bay: BayData, maxEvents: number): React.CSSProperties {
  const color = TIER_COLOR[worstTier(bay)] ?? "#94a3b8";
  const density = 0.18 + 0.32 * (bay.event_count / maxEvents);
  return {
    backgroundColor: `${color}${Math.round(density * 255)
      .toString(16)
      .padStart(2, "0")}`,
    borderColor: color,
  };
}

export default function Heatmap() {
  const [bays, setBays] = useState<BayData[]>([]);
  const [selectedBay, setSelectedBay] = useState<BayData | null>(null);
  const [isDemo, setIsDemo] = useState(false);

  useEffect(() => {
    fetchBays();
  }, []);

  async function fetchBays() {
    try {
      const res = await fetch("/api/bays/");
      if (res.ok) {
        const data = await res.json();
        if (data.length > 0) {
          setBays(data);
          return;
        }
      }
    } catch {}
    // Fallback to demo data
    setBays(DEMO_BAYS);
    setIsDemo(true);
  }

  const maxEvents = Math.max(...bays.map((b) => b.event_count), 1);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Bay Risk Heatmap</h2>
        {isDemo && <span className="text-xs text-yellow-400">backend unavailable — demo data</span>}
      </div>

      {/* Legend — cell hue = worst open tier in that bay */}
      <div className="flex items-center gap-4 text-sm">
        <span className="text-gray-400">Worst open tier:</span>
        {["Critical", "High", "Medium", "Low"].map((tier) => (
          <div key={tier} className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: TIER_COLOR[tier] }} />
            <span className="text-gray-300">{tier}</span>
          </div>
        ))}
        <span className="text-gray-500 text-xs">· darker fill = more events</span>
      </div>

      {/* Heatmap Grid — real bays only, no placeholder slots */}
      <div className="grid grid-cols-3 gap-4">
        {bays.map((bay) => (
          <div
            key={bay.bay_id}
            onClick={() => setSelectedBay(bay)}
            className={`rounded-xl border-2 p-4 cursor-pointer transition-transform hover:scale-[1.02] ${
              selectedBay?.bay_id === bay.bay_id ? "ring-2 ring-argus-accent" : ""
            }`}
            style={cellStyle(bay, maxEvents)}
          >
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-semibold text-sm text-gray-100">{bayName(bay.bay_id)}</h3>
              <RiskBadge tier={worstTier(bay)} />
            </div>

            {/* Event Count Bar — neutral UI accent, not a risk color */}
            <div className="mb-3">
              <div className="flex justify-between text-xs text-gray-400 mb-1">
                <span>Events</span>
                <span className="font-mono">{bay.event_count}</span>
              </div>
              <div className="w-full bg-black/30 rounded-full h-2">
                <div
                  className="h-2 rounded-full bg-argus-accent"
                  style={{ width: `${(bay.event_count / maxEvents) * 100}%` }}
                />
              </div>
            </div>

            {/* Tier Breakdown */}
            <div className="grid grid-cols-4 gap-1 text-center text-xs">
              <div>
                <div className="font-bold font-mono" style={{ color: TIER_COLOR.Critical }}>{bay.critical}</div>
                <div className="text-gray-500">Crit</div>
              </div>
              <div>
                <div className="font-bold font-mono" style={{ color: TIER_COLOR.High }}>{bay.high}</div>
                <div className="text-gray-500">High</div>
              </div>
              <div>
                <div className="font-bold font-mono" style={{ color: TIER_COLOR.Medium }}>{bay.medium}</div>
                <div className="text-gray-500">Med</div>
              </div>
              <div>
                <div className="font-bold font-mono" style={{ color: TIER_COLOR.Low }}>{bay.low}</div>
                <div className="text-gray-500">Low</div>
              </div>
            </div>

            <div className="mt-2 text-center text-xs text-gray-400">
              Avg risk: <span className="font-mono">{bay.avg_risk.toFixed(1)}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Selected Bay Detail — context, not a repeat of the card numbers */}
      {selectedBay && (
        <div className="bg-argus-card rounded-xl border border-argus-border p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold">{bayName(selectedBay.bay_id)}</h3>
            <button onClick={() => setSelectedBay(null)} className="text-gray-400 hover:text-white">✕</button>
          </div>

          <div className="grid grid-cols-3 gap-6 items-start">
            <div className="col-span-2">
              <h4 className="text-sm font-medium text-gray-400 mb-2">What's driving this bay's risk</h4>
              <p className="text-sm text-gray-300 leading-relaxed">
                {bayDetailNarrative(selectedBay)}
              </p>
              <div className="mt-3 flex gap-3">
                <Link
                  to="/"
                  className="text-sm text-argus-accent hover:underline"
                >
                  Open Live Feed →
                </Link>
                <Link
                  to="/incidents"
                  className="text-sm text-argus-accent hover:underline"
                >
                  Review incidents →
                </Link>
              </div>
            </div>

            <div className="text-center">
              <div className="text-5xl font-bold font-mono" style={{ color: TIER_COLOR[worstTier(selectedBay)] }}>
                {selectedBay.event_count}
              </div>
              <div className="text-gray-400 mt-1 text-sm">events this shift</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function bayDetailNarrative(bay: BayData): string {
  const parts: string[] = [];
  if (bay.critical > 0) parts.push(`${bay.critical} critical`);
  if (bay.high > 0) parts.push(`${bay.high} high-tier`);
  const dominant = bay.medium >= bay.low ? "medium" : "low";
  parts.push(`${bay.medium + bay.low} ${dominant}-tier handling events`);
  const worst = worstTier(bay);
  const action =
    worst === "Critical"
      ? "Immediate supervisor walkthrough recommended."
      : worst === "High"
        ? "Priority for the next safety briefing — thrown/dropped goods dominate here."
        : "Routine monitoring is sufficient for now.";
  return `This bay logged ${parts.join(", ")} this shift. Worst open tier is ${worst}. ${action}`;
}
