import { useState } from "react";
import { Users, Shield, Eye, Lock, ChevronDown } from "lucide-react";

interface OperatorScore {
  operator_id: string;
  name: string;
  role: string;
  events_total: number;
  confirms: number;
  dismissals: number;
  false_positive_rate: number;
  avg_risk: number;
  shift: string;
}

interface TeamScore {
  team_id: string;
  name: string;
  operator_count: number;
  events_total: number;
  avg_risk: number;
  top_behaviour: string;
}

type AccessRole = "operator" | "supervisor" | "auditor";

const ROLE_LABELS: Record<AccessRole, { label: string; color: string; icon: typeof Eye }> = {
  operator: { label: "Operator (Own Shift)", color: "text-blue-400", icon: Eye },
  supervisor: { label: "Supervisor (Bay-Wide)", color: "text-sky-400", icon: Shield },
  auditor: { label: "Auditor (Full History)", color: "text-green-400", icon: Lock },
};

const DEMO_OPERATORS: OperatorScore[] = [
  { operator_id: "OP-001", name: "Rajesh K.", role: "Operator", events_total: 8, confirms: 5, dismissals: 3, false_positive_rate: 0.375, avg_risk: 52.3, shift: "Morning" },
  { operator_id: "OP-002", name: "Priya S.", role: "Operator", events_total: 3, confirms: 2, dismissals: 1, false_positive_rate: 0.333, avg_risk: 38.1, shift: "Morning" },
  { operator_id: "OP-003", name: "Amit T.", role: "Operator", events_total: 12, confirms: 9, dismissals: 3, false_positive_rate: 0.25, avg_risk: 67.8, shift: "Evening" },
  { operator_id: "OP-004", name: "Deepa M.", role: "Operator", events_total: 5, confirms: 3, dismissals: 2, false_positive_rate: 0.4, avg_risk: 44.6, shift: "Evening" },
  { operator_id: "OP-005", name: "Vikram R.", role: "Operator", events_total: 1, confirms: 1, dismissals: 0, false_positive_rate: 0, avg_risk: 22.0, shift: "Morning" },
];

const DEMO_TEAMS: TeamScore[] = [
  { team_id: "T1", name: "Unloading Team", operator_count: 3, events_total: 20, avg_risk: 55.2, top_behaviour: "product_thrown" },
  { team_id: "T2", name: "Loading Team", operator_count: 2, events_total: 9, avg_risk: 41.5, top_behaviour: "product_dragged" },
  { team_id: "T3", name: "Storage Team", operator_count: 2, events_total: 4, avg_risk: 32.1, top_behaviour: "unstable_stacking" },
];

export default function Scorecards() {
  const [accessRole, setAccessRole] = useState<AccessRole>("supervisor");
  const [viewMode, setViewMode] = useState<"operators" | "teams">("operators");
  const [operators] = useState<OperatorScore[]>(DEMO_OPERATORS);
  const [teams] = useState<TeamScore[]>(DEMO_TEAMS);

  const visibleOperators = accessRole === "operator"
    ? operators.slice(0, 1) // Operator sees only own data
    : operators;

  const visibleTeams = accessRole === "operator" ? [] : teams;

  const roleInfo = ROLE_LABELS[accessRole];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Operator & Team Scorecards</h2>

        {/* Role-Based Access Toggle */}
        <div className="flex items-center gap-3">
          <span className="text-sm text-gray-400">View as:</span>
          <div className="relative">
            <select
              value={accessRole}
              onChange={(e) => setAccessRole(e.target.value as AccessRole)}
              className="bg-argus-dark border border-argus-border rounded-lg px-4 py-2 text-sm appearance-none pr-10 cursor-pointer"
            >
              <option value="operator">Operator</option>
              <option value="supervisor">Supervisor</option>
              <option value="auditor">Auditor</option>
            </select>
            <ChevronDown className="w-4 h-4 absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none" />
          </div>
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg bg-argus-card border border-argus-border ${roleInfo.color}`}>
            <roleInfo.icon className="w-4 h-4" />
            <span className="text-xs font-medium">{roleInfo.label}</span>
          </div>
        </div>
      </div>

      {/* Responsible AI Notice */}
      <div className="bg-argus-card rounded-xl border border-argus-border p-4">
        <div className="flex items-start gap-3">
          <Shield className="w-5 h-5 text-argus-accent mt-0.5 flex-shrink-0" />
          <div>
            <h3 className="font-semibold text-sm">Responsible AI — Identity Gating</h3>
            <p className="text-xs text-gray-400 mt-1">
              Operator identities are shown behind an explicit review action. The system reports
              <strong className="text-white"> events and locations</strong>, not named individuals —
              operator identity is only visible to supervisors and auditors with logged review access.
              No automated punitive action is ever taken.
            </p>
          </div>
        </div>
      </div>

      {/* View Mode Toggle */}
      <div className="flex gap-2">
        <button
          onClick={() => setViewMode("operators")}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            viewMode === "operators"
              ? "bg-argus-accent text-white"
              : "bg-argus-card text-gray-400 border border-argus-border hover:text-white"
          }`}
        >
          <Users className="w-4 h-4 inline mr-2" />
          Operators
        </button>
        <button
          onClick={() => setViewMode("teams")}
          disabled={accessRole === "operator"}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            viewMode === "teams"
              ? "bg-argus-accent text-white"
              : "bg-argus-card text-gray-400 border border-argus-border hover:text-white"
          } ${accessRole === "operator" ? "opacity-50 cursor-not-allowed" : ""}`}
        >
          <Users className="w-4 h-4 inline mr-2" />
          Teams
        </button>
      </div>

      {/* Operators Table */}
      {viewMode === "operators" && (
        <div className="bg-argus-card rounded-xl border border-argus-border overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="border-b border-argus-border text-left text-xs text-gray-400 uppercase tracking-wider">
                <th className="px-4 py-3">Operator</th>
                <th className="px-4 py-3">Shift</th>
                <th className="px-4 py-3 text-right">Events</th>
                <th className="px-4 py-3 text-right">Confirmed</th>
                <th className="px-4 py-3 text-right">Dismissed</th>
                <th className="px-4 py-3 text-right">FPR</th>
                <th className="px-4 py-3 text-right">Avg Risk</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-argus-border">
              {visibleOperators.map((op) => (
                <tr key={op.operator_id} className="hover:bg-argus-dark/50 transition-colors">
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-argus-border flex items-center justify-center text-xs font-bold text-white">
                        {op.name.charAt(0)}
                      </div>
                      <div>
                        <div className="font-medium text-sm">{op.name}</div>
                        <div className="text-xs text-gray-500">{op.operator_id}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-400">{op.shift}</td>
                  <td className="px-4 py-3 text-right font-mono text-sm">{op.events_total}</td>
                  <td className="px-4 py-3 text-right">
                    <span className="text-green-400 font-mono text-sm">{op.confirms}</span>
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span className="text-red-400 font-mono text-sm">{op.dismissals}</span>
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span className={`font-mono text-sm ${op.false_positive_rate > 0.3 ? "text-red-400" : op.false_positive_rate > 0.15 ? "text-yellow-400" : "text-green-400"}`}>
                      {(op.false_positive_rate * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span className={`font-mono text-sm ${op.avg_risk >= 75 ? "text-red-400" : op.avg_risk >= 50 ? "text-orange-400" : op.avg_risk >= 25 ? "text-yellow-400" : "text-green-400"}`}>
                      {op.avg_risk.toFixed(1)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {accessRole === "operator" && (
            <div className="px-4 py-3 bg-argus-dark/50 text-xs text-gray-500 border-t border-argus-border">
              <Lock className="w-3 h-3 inline mr-1" />
              Showing only your own shift data. Supervisor or Auditor access required for full view.
            </div>
          )}
        </div>
      )}

      {/* Teams Grid */}
      {viewMode === "teams" && (
        <div className="grid grid-cols-3 gap-4">
          {visibleTeams.map((team) => (
            <div key={team.team_id} className="bg-argus-card rounded-xl border border-argus-border p-4">
              <h3 className="font-semibold mb-3">{team.name}</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Operators</span>
                  <span className="font-mono">{team.operator_count}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Total Events</span>
                  <span className="font-mono">{team.events_total}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Avg Risk</span>
                  <span className={`font-mono ${team.avg_risk >= 50 ? "text-orange-400" : "text-yellow-400"}`}>
                    {team.avg_risk.toFixed(1)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Top Behaviour</span>
                  <span className="font-mono text-xs">{team.top_behaviour.replace(/_/g, " ")}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
