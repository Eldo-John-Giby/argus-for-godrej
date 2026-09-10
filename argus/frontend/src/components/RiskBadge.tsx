import type { ReactNode } from "react";

/**
 * Tier color — the ONLY source of tier color in the UI.
 * Amber (#f59e0b) belongs to High risk and nothing else; UI chrome
 * (nav, buttons, focus) uses argus-accent blue, never a risk color.
 */
export const TIER_COLOR: Record<string, string> = {
  Critical: "#ef4444",
  High: "#f59e0b",
  Medium: "#eab308",
  Low: "#22c55e",
};

const TIER_SUBTLE: Record<string, string> = {
  Critical: "bg-red-900/60 text-red-300 border-red-700/70",
  High: "bg-amber-900/60 text-amber-300 border-amber-700/70",
  Medium: "bg-yellow-900/60 text-yellow-200 border-yellow-700/70",
  Low: "bg-green-900/60 text-green-300 border-green-700/70",
};

/** Tier chip: subtle colored fill + colored text. */
export function RiskBadge({ tier }: { tier: string }) {
  return (
    <span
      className={`px-2 py-0.5 rounded text-xs font-medium border whitespace-nowrap ${
        TIER_SUBTLE[tier] ?? "bg-gray-800 text-gray-300 border-gray-700"
      }`}
    >
      {tier}
    </span>
  );
}

/**
 * Confidence-ladder chip (design spec: severity and confidence are two
 * separate legible dimensions, encoded by fill so it survives colorblind
 * viewing):
 *   confirmed  → solid tier fill
 *   potential  → 40% tier fill + solid tier border
 *   observed   → outline only, no fill
 *   dismissed  → muted grey
 */
export function StatusBadge({ status, tier }: { status: string; tier: string }) {
  const color = TIER_COLOR[tier] ?? "#94a3b8";

  if (status === "dismissed") {
    return (
      <span className="px-2 py-0.5 rounded text-xs font-medium border border-gray-700 text-gray-500 bg-transparent whitespace-nowrap">
        dismissed
      </span>
    );
  }
  if (status === "confirmed") {
    return (
      <span
        className="px-2 py-0.5 rounded text-xs font-medium whitespace-nowrap"
        style={{ backgroundColor: color, color: "#0f172a" }}
      >
        confirmed
      </span>
    );
  }
  if (status === "potential") {
    return (
      <span
        className="px-2 py-0.5 rounded text-xs font-medium border whitespace-nowrap"
        style={{ backgroundColor: `${color}59`, borderColor: color, color }}
      >
        potential
      </span>
    );
  }
  // observed — outline only
  return (
    <span
      className="px-2 py-0.5 rounded text-xs font-medium border bg-transparent whitespace-nowrap"
      style={{ borderColor: color, color }}
    >
      observed
    </span>
  );
}

/** Small dot used where a full chip is too heavy (map cells, legends). */
export function StatusDot({ status, tier }: { status: string; tier: string }): ReactNode {
  const color = TIER_COLOR[tier] ?? "#94a3b8";
  const style =
    status === "confirmed"
      ? { backgroundColor: color }
      : status === "potential"
        ? { backgroundColor: `${color}59`, border: `1px solid ${color}` }
        : { border: `1.5px solid ${color}` };
  return <span className="inline-block w-2.5 h-2.5 rounded-full" style={style} />;
}
