import { useState, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";
import { TrendingUp, AlertTriangle, Target, Shield } from "lucide-react";
import { getJson } from "../lib/api";

interface ShiftReport {
  title: string;
  start_time: string;
  end_time: string;
  bay_id: string | null;
  total_events: number;
  tier_counts: Record<string, number>;
  top_behaviour: string | null;
  report_text: string;
  filename: string;
  feedback: {
    confirms: number;
    dismissals: number;
    total: number;
    false_positive_rate: number | null;
  };
}

interface TopBehaviour {
  behaviour_type: string;
  count: number;
  avg_risk: number;
}

interface FeedbackStats {
  confirms: number;
  dismissals: number;
  total: number;
  false_positive_rate: number;
}

// Tier palette — mirrors RiskBadge; amber ONLY ever means High risk.
const TIER_HEX: Record<string, string> = {
  Critical: "#ef4444",
  High: "#f59e0b",
  Medium: "#eab308",
  Low: "#22c55e",
};

function tierForScore(score: number): string {
  if (score >= 50) return "Critical";
  if (score >= 35) return "High";
  if (score >= 20) return "Medium";
  return "Low";
}

export default function Dashboard() {
  const [topBehaviours, setTopBehaviours] = useState<TopBehaviour[]>([]);
  const [feedbackStats, setFeedbackStats] = useState<FeedbackStats | null>(null);
  const [recalibration, setRecalibration] = useState<any>(null);
  const [totalEventsToday, setTotalEventsToday] = useState(0);
  const [report, setReport] = useState<ShiftReport | null>(null);
  const [reportLoading, setReportLoading] = useState(false);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  async function fetchDashboardData() {
    try {
      const now = new Date();
      const startOfDay = new Date(now.getFullYear(), now.getMonth(), now.getDate()).toISOString();

      const [behavioursRes, feedbackRes, eventsRes] = await Promise.all([
        fetch(`/api/events/top-behaviours?start_time=${startOfDay}&end_time=${now.toISOString()}&limit=10`),
        fetch("/api/feedback/stats"),
        fetch(`/api/events/?start_time=${startOfDay}&end_time=${now.toISOString()}&limit=500`),
      ]);

      if (behavioursRes.ok) setTopBehaviours(await behavioursRes.json());
      if (feedbackRes.ok) setFeedbackStats(await feedbackRes.json());
      if (eventsRes.ok) {
        const events = await eventsRes.json();
        setTotalEventsToday(events.length);
      }
    } catch {
      // Use demo data
      setTopBehaviours(DEMO_BEHAVIOURS);
      setFeedbackStats(DEMO_FEEDBACK);
      setTotalEventsToday(34);
    }
  }

  async function generateReport() {
    setReportLoading(true);
    try {
      const data = await getJson<ShiftReport>("/api/reports/shift?hours=8");
      if (data) {
        setReport(data);
      } else {
        setReport(DEMO_REPORT);
      }
    } finally {
      setReportLoading(false);
    }
  }

  function speakReport() {
    if (!report || !("speechSynthesis" in window)) return;
    const u = new SpeechSynthesisUtterance(report.report_text);
    u.lang = "en-US";
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(u);
  }

  function downloadReport() {
    if (!report) return;
    const blob = new Blob([report.report_text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = report.filename || "shift-report.txt";
    a.click();
    URL.revokeObjectURL(url);
  }

  async function triggerRecalibration() {
    try {
      const res = await fetch("/api/feedback/recalibrate", { method: "POST" });
      if (res.ok) {
        setRecalibration(await res.json());
      }
    } catch {
      setRecalibration(DEMO_RECALIBRATION);
    }
  }

  const behaviourChartData = topBehaviours.map((b) => ({
    name: b.behaviour_type.replace(/^\d+_/, "").replace(/_/g, " "),
    count: b.count,
    avgRisk: Math.round(b.avg_risk),
  }));

  const tierDistribution = feedbackStats
    ? [
        { name: "Confirmed True", value: feedbackStats.confirms, color: "#22c55e" },
        { name: "False Positive", value: feedbackStats.dismissals, color: "#ef4444" },
      ].filter((d) => d.value > 0)
    : [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Operations Dashboard</h2>
        <button
          onClick={triggerRecalibration}
          className="px-4 py-2 bg-argus-accent text-white rounded-lg font-medium hover:bg-blue-500 transition-colors"
        >
          Recalibrate Thresholds
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard icon={AlertTriangle} label="Total Events Today" value={totalEventsToday} color="text-sky-400" />
        <StatCard icon={Shield} label="Confirms" value={feedbackStats?.confirms ?? 0} color="text-green-400" />
        <StatCard icon={Target} label="False Positive Rate" value={`${((feedbackStats?.false_positive_rate ?? 0) * 100).toFixed(1)}%`} color="text-red-400" />
        <StatCard icon={TrendingUp} label="Avg Risk Score" value={topBehaviours.length > 0 ? (() => {
          const totalWeighted = topBehaviours.reduce((sum, b) => sum + b.avg_risk * b.count, 0);
          const totalCount = topBehaviours.reduce((sum, b) => sum + b.count, 0);
          return totalCount > 0 ? (totalWeighted / totalCount).toFixed(1) : "0";
        })() : "0"} color="text-blue-400" />
      </div>

      {/* Automatic shift report — generated from stored events, no manual writing */}
      <div className="bg-argus-card rounded-xl border border-argus-border p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-semibold">Automatic shift report</h3>
          <div className="flex gap-2">
            <button
              onClick={generateReport}
              disabled={reportLoading}
              className="px-3 py-1.5 bg-argus-accent text-white rounded-lg text-sm font-medium hover:bg-blue-500 transition-colors disabled:opacity-50"
            >
              {reportLoading ? "Generating…" : report ? "Regenerate" : "Generate report"}
            </button>
            {report && (
              <>
                <button
                  onClick={speakReport}
                  className="px-3 py-1.5 border border-argus-border rounded-lg text-sm text-gray-300 hover:text-white transition-colors"
                  title="Read the report aloud (browser voice)"
                >
                  🔊 Read aloud
                </button>
                <button
                  onClick={downloadReport}
                  className="px-3 py-1.5 border border-argus-border rounded-lg text-sm text-gray-300 hover:text-white transition-colors"
                  title="Download as .txt"
                >
                  Download .txt
                </button>
              </>
            )}
          </div>
        </div>
        {report ? (
          <pre className="text-sm text-gray-300 whitespace-pre-wrap font-mono leading-relaxed">
            {report.report_text}
          </pre>
        ) : (
          <p className="text-sm text-gray-500">
            Generate a formatted summary of the last 8 hours — event counts, top behaviour, recommended actions —
            ready to paste into the shift log or read out at handover.
          </p>
        )}
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Top Behaviours Chart */}
        <div className="bg-argus-card rounded-xl border border-argus-border p-4">
          <h3 className="font-semibold mb-4">Top Risky Behaviours (Today)</h3>
          {behaviourChartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={behaviourChartData} margin={{ bottom: 8 }}>
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 10, fill: "#94a3b8" }}
                  angle={-32}
                  textAnchor="end"
                  height={95}
                  interval={0}
                />
                <YAxis tick={{ fontSize: 12, fill: "#94a3b8" }} allowDecimals={false} />
                <Tooltip
                  contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8 }}
                  labelStyle={{ color: "#e2e8f0" }}
                  cursor={{ fill: "#33415540" }}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {/* Bar color = severity tier of that behaviour's avg risk —
                      color carries the same meaning as every badge in the app. */}
                  {behaviourChartData.map((entry) => (
                    <Cell key={entry.name} fill={TIER_HEX[tierForScore(entry.avgRisk)] ?? "#3b82f6"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[340px] flex items-center justify-center text-gray-500">No data yet</div>
          )}
        </div>

        {/* Feedback Distribution */}
        <div className="bg-argus-card rounded-xl border border-argus-border p-4">
          <h3 className="font-semibold mb-4">Human Feedback Distribution</h3>
          {tierDistribution.length > 0 ? (
            <div>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie
                    data={tierDistribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={90}
                    dataKey="value"
                    label={({ value, percent }) => `${value} (${((percent ?? 0) * 100).toFixed(0)}%)`}
                  >
                    {tierDistribution.map((entry, index) => (
                      <Cell key={index} fill={entry.color} />
                    ))
                    }
                  </Pie>
                  <Tooltip
                    contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8 }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="flex items-center justify-center gap-6 text-sm pb-2">
                {tierDistribution.map((d) => (
                  <div key={d.name} className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded" style={{ backgroundColor: d.color }} />
                    <span className="text-gray-300">
                      {d.name}: <span className="font-mono">{d.value}</span>
                    </span>
                  </div>
                ))}
                {feedbackStats && (
                  <span className="text-xs text-gray-500">
                    FPR {((feedbackStats.false_positive_rate ?? 0) * 100).toFixed(0)}% · supervisor-verified sample
                  </span>
                )}
              </div>
            </div>
          ) : (
            <div className="h-[220px] flex items-center justify-center text-gray-500">
              No supervisor feedback yet — confirm or dismiss events in the Live Feed
            </div>
          )}
        </div>
      </div>

      {/* Recalibration Results */}
      {recalibration && (
        <div className="bg-argus-card rounded-xl border border-argus-accent p-4">
          <h3 className="font-semibold mb-3 text-argus-accent">⚡ Recalibration Results</h3>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-gray-400">False Positive Rate:</span>
              <span className="ml-2 font-mono">{(recalibration.false_positive_rate * 100).toFixed(1)}%</span>
            </div>
            <div>
              <span className="text-gray-400">Adjustment Direction:</span>
              <span className="ml-2 font-mono">{recalibration.adjustment_direction}</span>
            </div>
            <div>
              <span className="text-gray-400">Confidence Threshold Δ:</span>
              <span className="ml-2 font-mono">{recalibration.confidence_threshold_adjustment > 0 ? "+" : ""}{recalibration.confidence_threshold_adjustment.toFixed(3)}</span>
            </div>
          </div>
          {recalibration.per_behaviour && Object.keys(recalibration.per_behaviour).length > 0 && (
            <div className="mt-3 pt-3 border-t border-argus-border">
              <h4 className="text-sm font-medium mb-2">Per-Behaviour Adjustments:</h4>
              <div className="grid grid-cols-4 gap-2 text-xs">
                {Object.entries(recalibration.per_behaviour).map(([behaviour, data]: [string, any]) => (
                  <div key={behaviour} className="bg-argus-dark rounded p-2">
                    <div className="truncate">{behaviour.replace(/^\d+_/, "")}</div>
                    <div className="text-gray-400">FPR: {(data.fpr * 100).toFixed(0)}% · {data.adjustment}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function StatCard({ icon: Icon, label, value, color }: { icon: any; label: string; value: any; color: string }) {
  return (
    <div className="bg-argus-card rounded-xl border border-argus-border p-4">
      <div className="flex items-center gap-3">
        <Icon className={`w-8 h-8 ${color}`} />
        <div>
          <div className="text-2xl font-bold">{value}</div>
          <div className="text-sm text-gray-400">{label}</div>
        </div>
      </div>
    </div>
  );
}

const DEMO_REPORT: ShiftReport = {
  title: "Shift report",
  start_time: new Date(Date.now() - 8 * 3600_000).toISOString(),
  end_time: new Date().toISOString(),
  bay_id: null,
  total_events: 34,
  tier_counts: { Critical: 4, High: 9, Medium: 13, Low: 8 },
  top_behaviour: "product thrown",
  report_text: [
    "Shift report — All bays, " + new Date().toLocaleDateString(),
    "34 events: 4 critical, 9 high, 13 medium, 8 low.",
    "Bay A - Unloading Dock: 18 events, 8 high or critical.",
    "Bay B - Staging and Stacking Area: 12 events, 4 high or critical.",
    "Top behaviour: product thrown (12x); also product dropped (8x), stepping on packages (6x).",
    "Recommended action: Immediate corrective briefing; escalating discipline for repeat offences.",
    "Priority bay: Bay A - Unloading Dock (8 high/critical events).",
    "Supervisor review: 32 events reviewed, false-positive rate 25%.",
  ].join("\n"),
  filename: "argus-shift-report-all-bays-demo.txt",
  feedback: { confirms: 24, dismissals: 8, total: 32, false_positive_rate: 0.25 },
};

const DEMO_BEHAVIOURS: TopBehaviour[] = [
  { behaviour_type: "3_product_thrown", count: 12, avg_risk: 82.3 },
  { behaviour_type: "1_product_dropped", count: 8, avg_risk: 65.1 },
  { behaviour_type: "6_stepping_on_packages", count: 6, avg_risk: 71.4 },
  { behaviour_type: "2_product_dragged", count: 5, avg_risk: 48.2 },
  { behaviour_type: "5_unstable_stacking", count: 3, avg_risk: 38.7 },
];

const DEMO_FEEDBACK: FeedbackStats = {
  confirms: 24,
  dismissals: 8,
  total: 32,
  false_positive_rate: 0.25,
};

const DEMO_RECALIBRATION = {
  false_positive_rate: 0.25,
  total_feedback: 32,
  adjustment_direction: "stable",
  confidence_threshold_adjustment: 0.0,
  per_behaviour: {
    "3_product_thrown": { fpr: 0.08, samples: 12, adjustment: "loosen" },
    "1_product_dropped": { fpr: 0.38, samples: 8, adjustment: "tighten" },
    "6_stepping_on_packages": { fpr: 0.33, samples: 6, adjustment: "tighten" },
  },
};
