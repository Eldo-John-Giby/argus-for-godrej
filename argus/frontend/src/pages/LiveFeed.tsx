import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { Clock } from "lucide-react";
import { useAlertCenter } from "../lib/alerts";
import { RiskBadge, StatusBadge, TIER_COLOR } from "../components/RiskBadge";

interface Event {
  id: number;
  behaviour_type: string;
  tier: string;
  risk_score: number;
  status: string;
  confidence: number;
  timestamp: string;
  vlm_explanation?: string;
}

const LOW_CONFIDENCE = 0.35;

const TIER_STAT_CLASSES: Record<string, string> = {
  Critical: "bg-red-900/40 text-red-300 border-red-700/60",
  High: "bg-amber-900/40 text-amber-300 border-amber-700/60",
  Medium: "bg-yellow-900/40 text-yellow-200 border-yellow-700/60",
  Low: "bg-green-900/40 text-green-300 border-green-700/60",
};

/** Collapse runs of consecutive near-identical rows (same behaviour + tier +
 *  status) into one lead row with a "+N similar" expander — raw feed reads as
 *  spam when one handling pattern repeats 5× back to back. */
function groupFeed(events: Event[]): Event[][] {
  const groups: Event[][] = [];
  for (const e of events) {
    const last = groups[groups.length - 1];
    if (
      last &&
      last[0].behaviour_type === e.behaviour_type &&
      last[0].tier === e.tier &&
      last[0].status === e.status
    ) {
      last.push(e);
    } else {
      groups.push([e]);
    }
  }
  return groups;
}

export default function LiveFeed() {
  const [events, setEvents] = useState<Event[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  const { newCount, clearNewCount, replayAlert, muted, setMuted, lang, setLang, languages } = useAlertCenter();

  // Viewing the live feed acknowledges the new-alert badge.
  useEffect(() => {
    clearNewCount();
  }, [clearNewCount]);

  useEffect(() => {
    fetchEvents();
    const interval = setInterval(fetchEvents, 5000); // Poll every 5s
    return () => clearInterval(interval);
  }, []);

  async function fetchEvents() {
    try {
      const res = await fetch("/api/events/?limit=50");
      if (res.ok) {
        setEvents(await res.json());
        setError(null);
      } else {
        setError(`Backend error: ${res.status} ${res.statusText}`);
      }
    } catch {
      // Backend not available — show demo data
      setEvents(DEMO_EVENTS);
      setError("Backend not available — showing demo data");
    } finally {
      setLoading(false);
    }
  }

  function toggleGroup(leadId: number) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(leadId)) next.delete(leadId);
      else next.add(leadId);
      return next;
    });
  }

  function eventRow(event: Event) {
    const dim = event.confidence < LOW_CONFIDENCE;
    return (
      <Link
        to={`/incidents/${event.id}`}
        key={event.id}
        className={`block px-4 py-3 flex items-center gap-4 hover:bg-argus-dark/50 transition-colors ${
          dim ? "opacity-55 hover:opacity-100" : ""
        }`}
      >
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium truncate">{event.behaviour_type.replace(/_/g, " ")}</span>
            <RiskBadge tier={event.tier} />
            <StatusBadge status={event.status} tier={event.tier} />
          </div>
          {event.vlm_explanation && (
            <p className="text-sm text-gray-400 mt-1 truncate">{event.vlm_explanation}</p>
          )}
        </div>

        <div className="text-right text-sm text-gray-400">
          <div className="font-mono" style={{ color: TIER_COLOR[event.tier] ?? undefined }}>
            Score: {event.risk_score.toFixed(1)}
          </div>
          <div className={`text-xs font-mono ${dim ? "text-gray-600" : "text-gray-500"}`}>
            Conf: {(event.confidence * 100).toFixed(0)}%
          </div>
        </div>

        <div className="text-right text-xs text-gray-500">
          <Clock className="w-3 h-3 inline mr-1" />
          {new Date(event.timestamp).toLocaleTimeString()}
        </div>

        {/* Replay the spoken alert for this event */}
        <button
          onClick={(e) => {
            e.preventDefault();
            replayAlert(event);
          }}
          title="Play alert (sound + voice)"
          className="text-gray-500 hover:text-white"
        >
          🔊
        </button>
      </Link>
    );
  }

  const groups = groupFeed(events);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold flex items-center gap-3">
          Live Event Feed
          {newCount > 0 && (
            <span className="bg-red-600 text-white text-xs font-semibold px-2 py-0.5 rounded-full">
              +{newCount} new
            </span>
          )}
        </h2>
        <div className="flex items-center gap-3 text-sm text-gray-400">
          {/* Voice alert language */}
          <label className="flex items-center gap-1.5">
            <span>Alert voice</span>
            <select
              value={lang}
              onChange={(e) => setLang(e.target.value)}
              className="bg-argus-dark border border-argus-border rounded px-2 py-1 text-xs text-gray-200"
              title="Language for spoken alerts on new High/Critical events"
            >
              {languages.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.name}
                </option>
              ))}
            </select>
          </label>
          <button
            onClick={() => setMuted(!muted)}
            className="border border-argus-border rounded px-2 py-1 text-xs hover:text-white"
            title={muted ? "Unmute alert sound and voice" : "Mute alert sound and voice"}
          >
            {muted ? "🔇 Muted" : "🔔 Sound on"}
          </button>
          {error ? (
            <>
              <div className="w-2 h-2 bg-yellow-500 rounded-full" />
              <span className="text-yellow-400">{error}</span>
            </>
          ) : (
            <>
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              Live — polling every 5s
            </>
          )}
        </div>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-4 gap-4">
        {["Critical", "High", "Medium", "Low"].map((tier) => {
          const count = events.filter((e) => e.tier === tier).length;
          return (
            <div key={tier} className={`rounded-xl border p-4 ${TIER_STAT_CLASSES[tier]}`}>
              <div className="text-2xl font-bold font-mono">{count}</div>
              <div className="text-sm opacity-75">{tier} Risk</div>
            </div>
          );
        })}
      </div>

      {/* Event List */}
      <div className="bg-argus-card rounded-xl border border-argus-border overflow-hidden">
        <div className="px-4 py-3 border-b border-argus-border flex items-center justify-between">
          <h3 className="font-semibold text-gray-300">Recent Events</h3>
          <span className="text-xs text-gray-500">
            rows below {LOW_CONFIDENCE * 100}% confidence are dimmed
          </span>
        </div>
        {loading ? (
          <div className="p-8 text-center text-gray-500">Loading events...</div>
        ) : events.length === 0 ? (
          <div className="p-8 text-center text-gray-500">No events detected yet. Feed a video to start monitoring.</div>
        ) : (
          <div className="divide-y divide-argus-border">
            {groups.map((group) => {
              const lead = group[0];
              if (group.length === 1 || expanded.has(lead.id)) {
                return group.map((e) => eventRow(e));
              }
              return (
                <div key={lead.id}>
                  {eventRow(lead)}
                  <button
                    onClick={() => toggleGroup(lead.id)}
                    className="w-full px-4 pb-2 -mt-1 text-left text-xs text-gray-500 hover:text-gray-300"
                  >
                    + {group.length - 1} similar consecutive events — click to expand
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// Demo data for when backend is not running
const DEMO_EVENTS: Event[] = [
  { id: 1, behaviour_type: "3_product_thrown", tier: "Critical", risk_score: 87.2, status: "confirmed", confidence: 0.92, timestamp: new Date().toISOString(), vlm_explanation: "Operator threw package from loading dock to ground level instead of using ramp." },
  { id: 2, behaviour_type: "1_product_dropped", tier: "High", risk_score: 68.5, status: "potential", confidence: 0.74, timestamp: new Date(Date.now() - 120000).toISOString(), vlm_explanation: "Box slipped from operator's grip during transfer to pallet." },
  { id: 3, behaviour_type: "6_stepping_on_packages", tier: "High", risk_score: 72.1, status: "observed", confidence: 0.45, timestamp: new Date(Date.now() - 300000).toISOString() },
  { id: 4, behaviour_type: "2_product_dragged", tier: "Medium", risk_score: 45.3, status: "dismissed", confidence: 0.62, timestamp: new Date(Date.now() - 600000).toISOString(), vlm_explanation: "Operator dragged box along floor instead of lifting." },
  { id: 5, behaviour_type: "5_unstable_stacking", tier: "Medium", risk_score: 38.7, status: "observed", confidence: 0.31, timestamp: new Date(Date.now() - 900000).toISOString() },
];
