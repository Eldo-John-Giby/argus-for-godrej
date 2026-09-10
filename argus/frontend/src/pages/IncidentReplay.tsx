import { useState, useEffect, useRef } from "react";
import { useParams, Link } from "react-router-dom";
import { AlertTriangle, CheckCircle, XCircle, Info } from "lucide-react";
import OverlayCanvas from "../components/OverlayCanvas";
import { RiskBadge, StatusBadge, TIER_COLOR } from "../components/RiskBadge";

interface EventDetail {
  event_id: number;
  behaviour_type: string;
  risk_score: number;
  tier: string;
  confidence: number;
  status: string;
  features: Record<string, any>;
  vlm_explanation: string;
  risk_breakdown: Record<string, number>;
  explanation: string;
  clip_url?: string | null;
  clip_path?: string | null;
}

export default function IncidentReplay() {
  const { eventId } = useParams();
  const [event, setEvent] = useState<EventDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (eventId) {
      fetchEvent(parseInt(eventId));
    } else {
      setLoading(false);
    }
  }, [eventId]);

  async function fetchEvent(id: number) {
    try {
      const res = await fetch(`/api/events/${id}`);
      if (res.ok) {
        setEvent(await res.json());
      }
    } catch {
      // Demo data
      setEvent(DEMO_EVENT);
    } finally {
      setLoading(false);
    }
  }

  async function handleFeedback(action: "confirm" | "dismiss") {
    if (!event) return;
    try {
      await fetch("/api/feedback/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          event_id: event.event_id,
          reviewer: "supervisor",
          action,
        }),
      });
      setEvent({ ...event, status: action === "confirm" ? "confirmed" : "dismissed" });
    } catch {
      // Optimistic update
      setEvent({ ...event, status: action === "confirm" ? "confirmed" : "dismissed" });
    }
  }

  if (loading) return <div className="text-gray-500 p-8">Loading incident details...</div>;
  if (!event && eventId) return <div className="text-gray-500 p-8">Event not found.</div>;
  if (!event) return <IncidentList />;

  const breakdown = event.risk_breakdown || {};
  const maxBreakdown = Math.max(...Object.values(breakdown).map(Math.abs), 1);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Incident Replay — Event #{event.event_id}</h2>
        <div className="flex gap-2">
          <button
            onClick={() => handleFeedback("confirm")}
            className="flex items-center gap-2 px-4 py-2 bg-green-800 hover:bg-green-700 text-green-200 rounded-lg transition-colors"
          >
            <CheckCircle className="w-4 h-4" />
            Confirm
          </button>
          <button
            onClick={() => handleFeedback("dismiss")}
            className="flex items-center gap-2 px-4 py-2 bg-red-900 hover:bg-red-800 text-red-300 rounded-lg transition-colors"
          >
            <XCircle className="w-4 h-4" />
            Dismiss
          </button>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Video + Overlay */}
        <div className="col-span-2 space-y-4">
          <div className="bg-argus-card rounded-xl border border-argus-border overflow-hidden">
            <div className="aspect-video bg-gray-900 relative flex items-center justify-center">
              {event.clip_url ? (
                <video
                  key={event.clip_url}
                  src={event.clip_url}
                  className="w-full h-full object-contain"
                  controls
                  autoPlay
                  loop
                  muted
                  playsInline
                />
              ) : event.features && event.features.detections && event.features.detections.length > 0 ? (
                <OverlayCanvas
                  width={640}
                  height={360}
                  detections={event.features.detections}
                  showSkeleton={false}
                  showLabels={true}
                />
              ) : (
                <div className="text-gray-600 text-center z-10">
                  <AlertTriangle className="w-12 h-12 mx-auto mb-2" />
                  <p>Clip evidence unavailable</p>
                  <p className="text-sm">No clip was stored for this event</p>
                </div>
              )}
            </div>
            <div className="px-4 py-3 border-t border-argus-border text-sm text-gray-400">
              Clip: ±2s window around event · Keyframe highlighted at t=0
            </div>
          </div>

        </div>

        {/* Sidebar — breakdown first (above the fold), then summary + narrative */}
        <div className="space-y-4">
          {/* Risk Score Breakdown — the running total that makes the score defensible */}
          <div className="bg-argus-card rounded-xl border border-argus-border p-4">
            <h3 className="font-semibold mb-3 flex items-center gap-2">
              <Info className="w-4 h-4 text-argus-accent" />
              Risk Score Breakdown
            </h3>
            <div className="space-y-2">
              {Object.entries(breakdown).map(([key, value]) => (
                <div key={key} className="flex items-center gap-3">
                  <span className="text-sm text-gray-400 w-40 truncate">{key.replace(/_/g, " ")}</span>
                  <div className="flex-1 bg-gray-800 rounded-full h-3">
                    <div
                      className={`h-3 rounded-full ${value >= 0 ? "bg-sky-400" : "bg-slate-500"}`}
                      style={{ width: `${Math.abs(value) / maxBreakdown * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-mono w-16 text-right">{typeof value === "number" ? value.toFixed(1) : value}</span>
                </div>
              ))}
            </div>
            <div className="mt-3 pt-3 border-t border-argus-border flex justify-between">
              <span className="font-semibold">Total Risk Score</span>
              <span className="text-xl font-bold font-mono" style={{ color: TIER_COLOR[event.tier] }}>
                {event.risk_score.toFixed(1)} / 100
              </span>
            </div>
          </div>

          <div className="bg-argus-card rounded-xl border border-argus-border p-4">
            <h3 className="font-semibold mb-3">Event Summary</h3>
            <div className="space-y-3 text-sm">
              <div>
                <span className="text-gray-400">Behaviour:</span>
                <span className="ml-2 font-medium">{event.behaviour_type.replace(/_/g, " ")}</span>
              </div>
              <div>
                <span className="text-gray-400">Tier:</span>
                <span className="ml-2"><RiskBadge tier={event.tier} /></span>
              </div>
              <div>
                <span className="text-gray-400">Status:</span>
                <span className="ml-2"><StatusBadge status={event.status} tier={event.tier} /></span>
              </div>
              <div>
                <span className="text-gray-400">Confidence:</span>
                <span className="ml-2 font-mono">{(event.confidence * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>

          {/* Why-flagged narrative — the detector's own feature-grounded
              reasoning. The composite tier/score sentence is deliberately not
              repeated here; those fields sit directly above. */}
          <div className="bg-argus-card rounded-xl border border-argus-border p-4">
            <h3 className="font-semibold mb-3">Why Was This Flagged?</h3>
            <p className="text-sm text-gray-300 leading-relaxed">
              {event.vlm_explanation || "Flagged by the behaviour FSM on kinematic and geometric features — see the computed features below."}
            </p>
          </div>

          {/* Features */}
          <div className="bg-argus-card rounded-xl border border-argus-border p-4">
            <h3 className="font-semibold mb-3">Computed Features</h3>
            <pre className="text-xs text-gray-400 overflow-x-auto font-mono bg-black/30 rounded-lg p-3 leading-relaxed">
              {JSON.stringify(event.features, null, 2)}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}

function IncidentList() {
  const [events, setEvents] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/events/?limit=50")
      .then((r) => r.ok ? r.json() : [])
      .then(setEvents)
      .catch(() => setEvents(DEMO_EVENTS_LIST))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-gray-500 p-8">Loading incidents...</div>;

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold">Incidents</h2>
      <p className="text-gray-400 text-sm">Select an incident to view details, risk breakdown, and replay.</p>
      {events.length === 0 ? (
        <div className="bg-argus-card rounded-xl border border-argus-border p-8 text-center text-gray-500">
          No incidents detected yet.
        </div>
      ) : (
        <div className="bg-argus-card rounded-xl border border-argus-border overflow-hidden">
          <div className="divide-y divide-argus-border">
            {events.map((ev) => (
              <Link to={`/incidents/${ev.id}`} key={ev.id} className="block px-4 py-3 hover:bg-argus-dark/50 transition-colors">
                <div className="flex items-center gap-3">
                  <span className="font-medium">{ev.behaviour_type.replace(/_/g, " ")}</span>
                  <RiskBadge tier={ev.tier} />
                  <span className="text-sm text-gray-400 font-mono">Score: {ev.risk_score?.toFixed(1) ?? "—"}</span>
                  <span className="text-xs text-gray-500 ml-auto">{ev.timestamp ? new Date(ev.timestamp).toLocaleString() : ""}</span>
                </div>
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

const DEMO_EVENTS_LIST = [
  { id: 1, behaviour_type: "3_product_thrown", tier: "Critical", risk_score: 87.2, timestamp: new Date().toISOString() },
  { id: 2, behaviour_type: "1_product_dropped", tier: "High", risk_score: 68.5, timestamp: new Date(Date.now() - 120000).toISOString() },
  { id: 3, behaviour_type: "6_stepping_on_packages", tier: "High", risk_score: 72.1, timestamp: new Date(Date.now() - 300000).toISOString() },
];

const DEMO_EVENT: EventDetail = {
  event_id: 1,
  behaviour_type: "3_product_thrown",
  risk_score: 87.2,
  tier: "Critical",
  confidence: 0.92,
  status: "confirmed",
  features: {
    throw_velocity: [180, -220],
    trajectory_r_squared: 0.91,
    mass_proxy: 1.5,
    velocity_at_contact: 280,
    fragility_multiplier: 1.5,
  },
  vlm_explanation: "Operator threw package from loading dock height (~1.2m) to ground level. Ballistic trajectory confirmed with R²=0.91. Impact velocity estimated at 2.8 m/s — sufficient to damage standard packaging.",
  risk_breakdown: {
    behaviour_severity: 90,
    impact_proxy: 72.5,
    fragility: 45.0,
    stack_instability: 0,
    recurrence: 15.0,
    confidence_penalty: -3.2,
  },
  explanation: "This event was classified as Critical risk (score: 87.2/100) because the system detected behaviour 'product thrown' with 92% confidence. Operator threw package from loading dock — ballistic trajectory detected with parabolic fit R²=0.91. Impact velocity proxy suggests significant damage risk for fragile goods.",
};
